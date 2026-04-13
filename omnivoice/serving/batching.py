#!/usr/bin/env python3

"""Online micro-batching utilities for OmniVoice serving."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Lock, Thread
from typing import Callable, Optional

import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import GenerationTask, VoiceClonePrompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationBatchKey:
    """Compatibility key for online micro-batching."""

    num_step: int
    guidance_scale: float
    t_shift: float
    layer_penalty_factor: float
    position_temperature: float
    class_temperature: float
    denoise: bool
    audio_chunk_duration: float
    audio_chunk_threshold: float
    lane: str


@dataclass(frozen=True)
class GenerationBatcherConfig:
    """Tuning knobs for online API micro-batching."""

    collect_ms: float = 10.0
    max_batch_requests: int = 32
    max_batch_prompts: int = 32
    max_total_target_tokens: int = 4096
    max_total_conditioning_tokens: int = 8192
    max_padding_ratio: float = 2.5
    gpu_memory_utilization: float = 0.85
    gpu_memory_reserve_mb: int = 1024
    max_estimated_batch_memory_mb: Optional[int] = None


@dataclass
class PendingGenerationResult:
    """Result returned to one queued API request."""

    audios: list[torch.Tensor]
    batch_requests: int
    batch_prompts: int
    batch_target_tokens: int
    batch_max_sequence_length: int
    estimated_batch_memory_mb: Optional[float]
    queue_wait_ms: float
    batch_exec_ms: float


@dataclass
class PendingGeneration:
    """A queued generation job that may contain one or more prompt units."""

    request_id: str
    mode: str
    created_at: float
    batch_key: GenerationBatchKey
    task: GenerationTask
    generation_config: OmniVoiceGenerationConfig
    postprocess_flags: list[bool]
    estimated_sequence_lengths: list[int]
    future: Future = field(default_factory=Future)

    @property
    def request_count(self) -> int:
        return 1

    @property
    def prompt_count(self) -> int:
        return self.task.batch_size

    @property
    def target_token_count(self) -> int:
        return sum(self.task.target_lens)

    @property
    def conditioning_token_count(self) -> int:
        return sum(self.estimated_sequence_lengths)

    @property
    def max_sequence_length(self) -> int:
        return max(self.estimated_sequence_lengths) if self.estimated_sequence_lengths else 0


class ClonePromptCache:
    """Simple in-memory LRU cache for prepared voice-clone prompts."""

    def __init__(self, max_entries: int = 256):
        self.max_entries = max(1, int(max_entries))
        self._lock = Lock()
        self._cache: OrderedDict[str, VoiceClonePrompt] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_or_create(
        self, cache_key: str, factory: Callable[[], VoiceClonePrompt]
    ) -> VoiceClonePrompt:
        with self._lock:
            prompt = self._cache.get(cache_key)
            if prompt is not None:
                self._hits += 1
                self._cache.move_to_end(cache_key)
                return prompt

        prompt = factory()
        prompt = VoiceClonePrompt(
            ref_audio_tokens=prompt.ref_audio_tokens.detach().cpu(),
            ref_text=prompt.ref_text,
            ref_rms=prompt.ref_rms,
        )

        with self._lock:
            existing = self._cache.get(cache_key)
            if existing is not None:
                self._hits += 1
                self._cache.move_to_end(cache_key)
                return existing

            self._misses += 1
            self._cache[cache_key] = prompt
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1
            return prompt

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }


def build_clone_prompt_cache_key(
    audio_bytes: bytes,
    ref_text: Optional[str],
    preprocess_prompt: bool,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(audio_bytes)
    hasher.update(b"\0")
    hasher.update((ref_text or "").encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(b"1" if preprocess_prompt else b"0")
    return hasher.hexdigest()


def merge_generation_tasks(tasks: list[GenerationTask]) -> GenerationTask:
    """Merge many prepared tasks into one batch task."""

    if not tasks:
        raise ValueError("Cannot merge an empty task list")

    texts = []
    target_lens = []
    langs = []
    instructs = []
    ref_texts = []
    ref_audio_tokens = []
    ref_rms = []

    any_speed = any(task.speed is not None for task in tasks)
    speed_list = [] if any_speed else None

    for task in tasks:
        texts.extend(task.texts)
        target_lens.extend(task.target_lens)
        langs.extend(task.langs)
        instructs.extend(task.instructs)
        ref_texts.extend(task.ref_texts)
        ref_audio_tokens.extend(task.ref_audio_tokens)
        ref_rms.extend(task.ref_rms)
        if speed_list is not None:
            if task.speed is None:
                speed_list.extend([1.0] * task.batch_size)
            else:
                speed_list.extend(task.speed)

    return GenerationTask(
        batch_size=len(texts),
        texts=texts,
        target_lens=target_lens,
        langs=langs,
        instructs=instructs,
        ref_texts=ref_texts,
        ref_audio_tokens=ref_audio_tokens,
        ref_rms=ref_rms,
        speed=speed_list,
    )


class GenerationBatcher:
    """Background micro-batcher for API requests.

    This implements the Chatterbox-style anchor-based queue collection pattern:
    one in-flight batch at a time, with a short collection window to gather
    nearby compatible requests.
    """

    def __init__(
        self,
        model: OmniVoice,
        config: GenerationBatcherConfig,
        name: str = "omnivoice-batcher",
    ):
        self._model = model
        self._config = config
        self._name = name
        self._condition = Condition()
        self._pending: list[PendingGeneration] = []
        self._stopping = False
        self._stats_lock = Lock()
        self._batches_started = 0
        self._batches_completed = 0
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._last_batch_summary: Optional[dict] = None
        self._cuda_device = self._resolve_cuda_device()
        self._worker = Thread(target=self._run_loop, name=name, daemon=True)
        self._worker.start()

    def submit(self, job: PendingGeneration) -> PendingGenerationResult:
        with self._condition:
            self._pending.append(job)
            self._condition.notify()
        return job.future.result()

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def stats(self) -> dict:
        with self._stats_lock:
            last_summary = dict(self._last_batch_summary) if self._last_batch_summary else None
            batches_started = self._batches_started
            batches_completed = self._batches_completed
            jobs_completed = self._jobs_completed
            jobs_failed = self._jobs_failed
        with self._condition:
            pending = len(self._pending)
        current_budget = self._current_memory_budget_bytes()
        current_free = self._current_free_memory_bytes()
        return {
            "collect_ms": self._config.collect_ms,
            "max_batch_requests": self._config.max_batch_requests,
            "max_batch_prompts": self._config.max_batch_prompts,
            "max_total_target_tokens": self._config.max_total_target_tokens,
            "max_total_conditioning_tokens": self._config.max_total_conditioning_tokens,
            "max_padding_ratio": self._config.max_padding_ratio,
            "gpu_memory_utilization": self._config.gpu_memory_utilization,
            "gpu_memory_reserve_mb": self._config.gpu_memory_reserve_mb,
            "max_estimated_batch_memory_mb": self._config.max_estimated_batch_memory_mb,
            "current_free_memory_mb": (
                round(current_free / (1024 * 1024), 2)
                if current_free is not None
                else None
            ),
            "current_batch_memory_budget_mb": (
                round(current_budget / (1024 * 1024), 2)
                if current_budget is not None
                else None
            ),
            "pending_jobs": pending,
            "batches_started": batches_started,
            "batches_completed": batches_completed,
            "jobs_completed": jobs_completed,
            "jobs_failed": jobs_failed,
            "last_batch_summary": last_summary,
        }

    def _resolve_cuda_device(self) -> Optional[torch.device]:
        try:
            device = torch.device(self._model.device)
        except Exception:
            return None
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        return device

    def _current_free_memory_bytes(self) -> Optional[int]:
        if self._cuda_device is None:
            return None
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self._cuda_device)
        except Exception:
            logger.exception("Failed to query CUDA free memory")
            return None
        return int(free_bytes)

    def _current_memory_budget_bytes(self) -> Optional[int]:
        if self._config.max_estimated_batch_memory_mb is not None:
            return max(0, int(self._config.max_estimated_batch_memory_mb)) * 1024 * 1024

        free_bytes = self._current_free_memory_bytes()
        if free_bytes is None:
            return None

        reserve_bytes = max(0, int(self._config.gpu_memory_reserve_mb)) * 1024 * 1024
        usable_free = max(0, free_bytes - reserve_bytes)
        utilization = min(max(float(self._config.gpu_memory_utilization), 0.05), 1.0)
        return int(usable_free * utilization)

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return

                collect_deadline = time.monotonic() + max(0.0, self._config.collect_ms) / 1000.0
                while not self._stopping:
                    remaining = collect_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
                if self._stopping:
                    return

                jobs = self._select_batch_locked()

            self._process_batch(jobs)

    def _select_batch_locked(self) -> list[PendingGeneration]:
        anchor = self._pending[0]
        memory_budget_bytes = self._current_memory_budget_bytes()
        selected_indices = [0]
        batch_requests = 1
        batch_prompts = anchor.prompt_count
        target_tokens = anchor.target_token_count
        conditioning_tokens = anchor.conditioning_token_count
        max_sequence_length = anchor.max_sequence_length
        sequence_lengths = list(anchor.estimated_sequence_lengths)
        target_lengths = list(anchor.task.target_lens)
        estimated_batch_memory_bytes = self._model.estimate_generation_batch_memory_bytes(
            sequence_lengths,
            target_lengths,
            anchor.generation_config.guidance_scale,
        )

        for index in range(1, len(self._pending)):
            candidate = self._pending[index]
            if candidate.batch_key != anchor.batch_key:
                continue

            next_requests = batch_requests + candidate.request_count
            next_prompts = batch_prompts + candidate.prompt_count
            next_target_tokens = target_tokens + candidate.target_token_count
            next_conditioning_tokens = (
                conditioning_tokens + candidate.conditioning_token_count
            )
            next_max_sequence = max(max_sequence_length, candidate.max_sequence_length)
            padding_ratio = (
                (next_max_sequence * next_prompts) / next_conditioning_tokens
                if next_conditioning_tokens > 0
                else 1.0
            )
            next_sequence_lengths = sequence_lengths + candidate.estimated_sequence_lengths
            next_target_lengths = target_lengths + candidate.task.target_lens
            next_estimated_batch_memory_bytes = (
                self._model.estimate_generation_batch_memory_bytes(
                    next_sequence_lengths,
                    next_target_lengths,
                    anchor.generation_config.guidance_scale,
                )
            )

            if next_requests > self._config.max_batch_requests:
                continue
            if next_prompts > self._config.max_batch_prompts:
                continue
            if next_target_tokens > self._config.max_total_target_tokens:
                continue
            if next_conditioning_tokens > self._config.max_total_conditioning_tokens:
                continue
            if padding_ratio > self._config.max_padding_ratio:
                continue
            if (
                memory_budget_bytes is not None
                and next_estimated_batch_memory_bytes > memory_budget_bytes
            ):
                continue

            selected_indices.append(index)
            batch_requests = next_requests
            batch_prompts = next_prompts
            target_tokens = next_target_tokens
            conditioning_tokens = next_conditioning_tokens
            max_sequence_length = next_max_sequence
            sequence_lengths = next_sequence_lengths
            target_lengths = next_target_lengths
            estimated_batch_memory_bytes = next_estimated_batch_memory_bytes

            if batch_requests >= self._config.max_batch_requests:
                break

        jobs = [self._pending[index] for index in selected_indices]
        for index in reversed(selected_indices):
            self._pending.pop(index)
        return jobs

    def _process_batch(self, jobs: list[PendingGeneration]) -> None:
        batch_started = time.perf_counter()
        merged_task = merge_generation_tasks([job.task for job in jobs])
        merged_postprocess_flags = [
            flag for job in jobs for flag in job.postprocess_flags
        ]

        with self._stats_lock:
            self._batches_started += 1

        try:
            token_outputs = self._model.generate_tokens(
                merged_task,
                generation_config=jobs[0].generation_config,
            )
            audios = self._model.decode_tokens(
                token_outputs,
                ref_rms=merged_task.ref_rms,
                generation_config=jobs[0].generation_config,
                postprocess_output=merged_postprocess_flags,
            )

            batch_exec_ms = (time.perf_counter() - batch_started) * 1000.0
            batch_requests = len(jobs)
            batch_prompts = merged_task.batch_size
            batch_target_tokens = sum(merged_task.target_lens)
            batch_max_sequence_length = max(
                (
                    seq_len
                    for job in jobs
                    for seq_len in job.estimated_sequence_lengths
                ),
                default=0,
            )
            estimated_batch_memory_mb = (
                self._model.estimate_generation_batch_memory_bytes(
                    [
                        seq_len
                        for job in jobs
                        for seq_len in job.estimated_sequence_lengths
                    ],
                    merged_task.target_lens,
                    jobs[0].generation_config.guidance_scale,
                )
                / (1024 * 1024)
            )

            offset = 0
            for job in jobs:
                prompt_count = job.prompt_count
                job_audios = audios[offset : offset + prompt_count]
                offset += prompt_count
                queue_wait_ms = (batch_started - job.created_at) * 1000.0
                job.future.set_result(
                    PendingGenerationResult(
                        audios=job_audios,
                        batch_requests=batch_requests,
                        batch_prompts=batch_prompts,
                        batch_target_tokens=batch_target_tokens,
                        batch_max_sequence_length=batch_max_sequence_length,
                        estimated_batch_memory_mb=estimated_batch_memory_mb,
                        queue_wait_ms=queue_wait_ms,
                        batch_exec_ms=batch_exec_ms,
                    )
                )

            with self._stats_lock:
                self._batches_completed += 1
                self._jobs_completed += len(jobs)
                self._last_batch_summary = {
                    "batch_requests": batch_requests,
                    "batch_prompts": batch_prompts,
                    "batch_target_tokens": batch_target_tokens,
                    "batch_max_sequence_length": batch_max_sequence_length,
                    "estimated_batch_memory_mb": round(estimated_batch_memory_mb, 2),
                    "batch_exec_ms": round(batch_exec_ms, 2),
                    "lane": jobs[0].batch_key.lane,
                }

            logger.info(
                "batch status=success requests=%d prompts=%d lane=%s exec_ms=%.2f target_tokens=%d max_sequence_length=%d est_mem_mb=%.2f",
                batch_requests,
                batch_prompts,
                jobs[0].batch_key.lane,
                batch_exec_ms,
                batch_target_tokens,
                batch_max_sequence_length,
                estimated_batch_memory_mb,
            )
        except Exception as exc:
            with self._stats_lock:
                self._jobs_failed += len(jobs)
            for job in jobs:
                job.future.set_exception(exc)
            logger.exception(
                "batch status=failed requests=%d lane=%s",
                len(jobs),
                jobs[0].batch_key.lane if jobs else "unknown",
            )
