#!/usr/bin/env python3

"""Online micro-batching utilities for OmniVoice serving."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Lock, Thread
from typing import Callable, Optional

import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import GenerationTask, VoiceClonePrompt
from omnivoice.utils.profiling import timed_stage

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
    shape_bucketing_enabled: bool = True
    shape_bucket_target_tokens: int = 16
    shape_bucket_conditioning_tokens: int = 32
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
    shape_bucket_id: Optional[str]
    exact_shape_homogeneous: bool
    batch_requests: int
    batch_prompts: int
    batch_target_tokens: int
    batch_max_sequence_length: int
    estimated_batch_memory_mb: Optional[float]
    gpu_metrics: dict[str, object]
    queue_wait_ms: float
    batch_exec_ms: float
    stage_timings_ms: dict[str, float]


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
    shape_bucket_id: Optional[str]
    exact_shape_signature: str
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


def _round_up_to_multiple(value: int, step: int) -> int:
    step = max(1, int(step))
    value = max(1, int(value))
    return int(((value + step - 1) // step) * step)


def _bucketize_prompt_count(prompt_count: int) -> int:
    prompt_count = max(1, int(prompt_count))
    bucket = 1
    while bucket < prompt_count:
        bucket <<= 1
    return bucket


def build_generation_shape_bucket_id(
    *,
    lane: str,
    guidance_scale: float,
    prompt_count: int,
    estimated_sequence_lengths: list[int],
    target_lens: list[int],
    conditioning_bucket_tokens: int,
    target_bucket_tokens: int,
) -> str:
    max_conditioning = max(estimated_sequence_lengths) if estimated_sequence_lengths else 1
    max_target = max(target_lens) if target_lens else 1
    prompt_bucket = _bucketize_prompt_count(prompt_count)
    conditioning_bucket = _round_up_to_multiple(
        max_conditioning,
        conditioning_bucket_tokens,
    )
    target_bucket = _round_up_to_multiple(
        max_target,
        target_bucket_tokens,
    )
    guidance_bucket = "cfg" if float(guidance_scale) != 0.0 else "nog"
    return (
        f"{lane}|{guidance_bucket}|p{prompt_bucket}|"
        f"t{target_bucket}|c{conditioning_bucket}"
    )


def build_generation_shape_signature(
    *,
    prompt_count: int,
    estimated_sequence_lengths: list[int],
    target_lens: list[int],
) -> str:
    conditioning_part = ",".join(str(int(length)) for length in estimated_sequence_lengths)
    target_part = ",".join(str(int(length)) for length in target_lens)
    return f"p{int(prompt_count)}|c[{conditioning_part}]|t[{target_part}]"


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
        prompt, _created = self.get_or_create_with_meta(cache_key, factory)
        return prompt

    def get_or_create_with_meta(
        self, cache_key: str, factory: Callable[[], VoiceClonePrompt]
    ) -> tuple[VoiceClonePrompt, bool]:
        with self._lock:
            prompt = self._cache.get(cache_key)
            if prompt is not None:
                self._hits += 1
                self._cache.move_to_end(cache_key)
                return prompt, False

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
                return existing, False

            self._misses += 1
            self._cache[cache_key] = prompt
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1
            return prompt, True

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }


class GpuTelemetryMonitor:
    """Lightweight CUDA telemetry sampler for live and per-batch stats."""

    def __init__(self, cuda_device: Optional[torch.device], poll_ms: float = 250.0):
        self._cuda_device = cuda_device
        self._poll_interval_s = max(0.05, float(poll_ms) / 1000.0)
        self._gpu_index = None
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._lock = Lock()
        self._latest_snapshot: Optional[dict[str, object]] = None
        self._batch_active = False
        self._batch_free_before_mb: Optional[float] = None
        self._batch_peak_utilization_pct: Optional[float] = None
        self._batch_peak_memory_used_mb: Optional[float] = None
        self._stopping = False
        self._worker: Optional[Thread] = None

        if self._cuda_device is not None:
            self._gpu_index = (
                self._cuda_device.index
                if self._cuda_device.index is not None
                else torch.cuda.current_device()
            )
            self._latest_snapshot = self._query_nvidia_smi()
            if self._nvidia_smi is not None:
                self._worker = Thread(
                    target=self._run_loop,
                    name="omnivoice-gpu-telemetry",
                    daemon=True,
                )
                self._worker.start()

    def close(self) -> None:
        self._stopping = True
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def latest_snapshot(self) -> Optional[dict[str, object]]:
        if self._cuda_device is None:
            return None

        snapshot = None
        with self._lock:
            if self._latest_snapshot is not None:
                snapshot = dict(self._latest_snapshot)

        if snapshot is None:
            snapshot = self._query_nvidia_smi()

        torch_snapshot = self._torch_memory_snapshot()
        if snapshot is None:
            snapshot = {}
        snapshot.update(torch_snapshot)
        snapshot["gpu_index"] = self._gpu_index
        return snapshot

    def begin_batch(self) -> None:
        if self._cuda_device is None:
            return
        try:
            torch.cuda.synchronize(self._cuda_device)
            torch.cuda.reset_peak_memory_stats(self._cuda_device)
        except Exception:
            logger.exception("Failed to reset CUDA peak memory stats")

        snapshot = self.latest_snapshot()
        with self._lock:
            self._batch_active = True
            self._batch_free_before_mb = (
                float(snapshot["gpu_memory_free_mb"])
                if snapshot and snapshot.get("gpu_memory_free_mb") is not None
                else None
            )
            self._batch_peak_utilization_pct = (
                float(snapshot["gpu_utilization_pct"])
                if snapshot and snapshot.get("gpu_utilization_pct") is not None
                else None
            )
            self._batch_peak_memory_used_mb = (
                float(snapshot["gpu_memory_used_mb"])
                if snapshot and snapshot.get("gpu_memory_used_mb") is not None
                else None
            )

    def end_batch(self) -> dict[str, object]:
        if self._cuda_device is None:
            return {}

        try:
            torch.cuda.synchronize(self._cuda_device)
        except Exception:
            logger.exception("Failed to synchronize CUDA device for telemetry")

        snapshot = self.latest_snapshot() or {}
        with self._lock:
            result = {
                "gpu_index": self._gpu_index,
                "gpu_name": snapshot.get("gpu_name"),
                "gpu_utilization_pct": snapshot.get("gpu_utilization_pct"),
                "gpu_utilization_peak_pct": self._batch_peak_utilization_pct,
                "gpu_memory_total_mb": snapshot.get("gpu_memory_total_mb"),
                "gpu_memory_used_mb": snapshot.get("gpu_memory_used_mb"),
                "gpu_memory_used_peak_mb": self._batch_peak_memory_used_mb,
                "gpu_memory_free_before_mb": self._batch_free_before_mb,
                "gpu_memory_free_after_mb": snapshot.get("gpu_memory_free_mb"),
                "torch_allocated_mb": snapshot.get("torch_allocated_mb"),
                "torch_reserved_mb": snapshot.get("torch_reserved_mb"),
                "torch_peak_allocated_mb": snapshot.get("torch_peak_allocated_mb"),
                "torch_peak_reserved_mb": snapshot.get("torch_peak_reserved_mb"),
            }
            self._batch_active = False
            self._batch_free_before_mb = None
            self._batch_peak_utilization_pct = None
            self._batch_peak_memory_used_mb = None
        return result

    def _run_loop(self) -> None:
        while not self._stopping:
            snapshot = self._query_nvidia_smi()
            if snapshot is not None:
                with self._lock:
                    self._latest_snapshot = snapshot
                    if self._batch_active:
                        util_pct = snapshot.get("gpu_utilization_pct")
                        used_mb = snapshot.get("gpu_memory_used_mb")
                        if util_pct is not None:
                            util_pct = float(util_pct)
                            if (
                                self._batch_peak_utilization_pct is None
                                or util_pct > self._batch_peak_utilization_pct
                            ):
                                self._batch_peak_utilization_pct = util_pct
                        if used_mb is not None:
                            used_mb = float(used_mb)
                            if (
                                self._batch_peak_memory_used_mb is None
                                or used_mb > self._batch_peak_memory_used_mb
                            ):
                                self._batch_peak_memory_used_mb = used_mb
            time.sleep(self._poll_interval_s)

    def _query_nvidia_smi(self) -> Optional[dict[str, object]]:
        if self._gpu_index is None or self._nvidia_smi is None:
            return None
        try:
            completed = subprocess.run(
                [
                    self._nvidia_smi,
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self._gpu_index),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=1.0,
            )
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        line = completed.stdout.strip().splitlines()
        if not line:
            return None
        parts = [part.strip() for part in line[0].split(",")]
        if len(parts) != 5:
            return None
        try:
            return {
                "gpu_name": parts[0],
                "gpu_memory_total_mb": float(parts[1]),
                "gpu_memory_used_mb": float(parts[2]),
                "gpu_memory_free_mb": float(parts[3]),
                "gpu_utilization_pct": float(parts[4]),
            }
        except ValueError:
            return None

    def _torch_memory_snapshot(self) -> dict[str, object]:
        if self._cuda_device is None:
            return {}
        try:
            return {
                "torch_allocated_mb": round(
                    torch.cuda.memory_allocated(self._cuda_device) / (1024 * 1024), 2
                ),
                "torch_reserved_mb": round(
                    torch.cuda.memory_reserved(self._cuda_device) / (1024 * 1024), 2
                ),
                "torch_peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated(self._cuda_device) / (1024 * 1024),
                    2,
                ),
                "torch_peak_reserved_mb": round(
                    torch.cuda.max_memory_reserved(self._cuda_device) / (1024 * 1024),
                    2,
                ),
            }
        except Exception:
            logger.exception("Failed to query torch CUDA memory stats")
            return {}


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
        self._gpu_monitor = GpuTelemetryMonitor(self._cuda_device)
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
        self._gpu_monitor.close()

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
            "shape_bucketing_enabled": self._config.shape_bucketing_enabled,
            "shape_bucket_target_tokens": self._config.shape_bucket_target_tokens,
            "shape_bucket_conditioning_tokens": self._config.shape_bucket_conditioning_tokens,
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
            "gpu": self._gpu_monitor.latest_snapshot(),
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
        selected_index_set = {0}
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

        def try_include(index: int) -> bool:
            nonlocal batch_requests
            nonlocal batch_prompts
            nonlocal target_tokens
            nonlocal conditioning_tokens
            nonlocal max_sequence_length
            nonlocal sequence_lengths
            nonlocal target_lengths
            nonlocal estimated_batch_memory_bytes

            candidate = self._pending[index]
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
                return False
            if next_prompts > self._config.max_batch_prompts:
                return False
            if next_target_tokens > self._config.max_total_target_tokens:
                return False
            if next_conditioning_tokens > self._config.max_total_conditioning_tokens:
                return False
            if padding_ratio > self._config.max_padding_ratio:
                return False
            if (
                memory_budget_bytes is not None
                and next_estimated_batch_memory_bytes > memory_budget_bytes
            ):
                return False

            selected_indices.append(index)
            selected_index_set.add(index)
            batch_requests = next_requests
            batch_prompts = next_prompts
            target_tokens = next_target_tokens
            conditioning_tokens = next_conditioning_tokens
            max_sequence_length = next_max_sequence
            sequence_lengths = next_sequence_lengths
            target_lengths = next_target_lengths
            estimated_batch_memory_bytes = next_estimated_batch_memory_bytes
            return True

        def is_bucket_compatible(candidate: PendingGeneration) -> bool:
            if candidate.batch_key != anchor.batch_key:
                return False
            if not self._config.shape_bucketing_enabled:
                return True
            return candidate.shape_bucket_id == anchor.shape_bucket_id

        for exact_only in (True, False):
            for index in range(1, len(self._pending)):
                if index in selected_index_set:
                    continue
                candidate = self._pending[index]
                if not is_bucket_compatible(candidate):
                    continue
                if exact_only and candidate.exact_shape_signature != anchor.exact_shape_signature:
                    continue
                if try_include(index) and batch_requests >= self._config.max_batch_requests:
                    break
            if batch_requests >= self._config.max_batch_requests:
                break

        jobs = [self._pending[index] for index in selected_indices]
        for index in sorted(selected_indices, reverse=True):
            self._pending.pop(index)
        return jobs

    def _process_batch(self, jobs: list[PendingGeneration]) -> None:
        batch_started = time.perf_counter()
        stage_timings_ms: dict[str, float] = {}
        with timed_stage(
            stage_timings_ms,
            "batch_merge_task_ms",
            "omnivoice.batch.merge_task",
        ):
            merged_task = merge_generation_tasks([job.task for job in jobs])
            merged_postprocess_flags = [
                flag for job in jobs for flag in job.postprocess_flags
            ]
        self._gpu_monitor.begin_batch()

        with self._stats_lock:
            self._batches_started += 1

        try:
            model_generate_timings_ms: dict[str, float] = {}
            with timed_stage(
                stage_timings_ms,
                "batch_generate_tokens_ms",
                "omnivoice.batch.generate_tokens",
            ):
                token_outputs = self._model.generate_tokens(
                    merged_task,
                    generation_config=jobs[0].generation_config,
                    _profile=model_generate_timings_ms,
                )
            stage_timings_ms.update(model_generate_timings_ms)

            model_decode_timings_ms: dict[str, float] = {}
            with timed_stage(
                stage_timings_ms,
                "batch_decode_tokens_ms",
                "omnivoice.batch.decode_tokens",
            ):
                audios = self._model.decode_tokens(
                    token_outputs,
                    ref_rms=merged_task.ref_rms,
                    generation_config=jobs[0].generation_config,
                    postprocess_output=merged_postprocess_flags,
                    _profile=model_decode_timings_ms,
                )
            stage_timings_ms.update(model_decode_timings_ms)

            batch_exec_ms = (time.perf_counter() - batch_started) * 1000.0
            batch_requests = len(jobs)
            batch_prompts = merged_task.batch_size
            shape_bucket_id = jobs[0].shape_bucket_id
            exact_shape_homogeneous = (
                len({job.exact_shape_signature for job in jobs}) == 1
            )
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
            gpu_metrics = self._gpu_monitor.end_batch()

            offset = 0
            for job in jobs:
                prompt_count = job.prompt_count
                job_audios = audios[offset : offset + prompt_count]
                offset += prompt_count
                queue_wait_ms = (batch_started - job.created_at) * 1000.0
                job.future.set_result(
                    PendingGenerationResult(
                        audios=job_audios,
                        shape_bucket_id=shape_bucket_id,
                        exact_shape_homogeneous=exact_shape_homogeneous,
                        batch_requests=batch_requests,
                        batch_prompts=batch_prompts,
                        batch_target_tokens=batch_target_tokens,
                        batch_max_sequence_length=batch_max_sequence_length,
                        estimated_batch_memory_mb=estimated_batch_memory_mb,
                        gpu_metrics=dict(gpu_metrics),
                        queue_wait_ms=queue_wait_ms,
                        batch_exec_ms=batch_exec_ms,
                        stage_timings_ms=dict(stage_timings_ms),
                    )
                )

            with self._stats_lock:
                self._batches_completed += 1
                self._jobs_completed += len(jobs)
                self._last_batch_summary = {
                    "batch_requests": batch_requests,
                    "batch_prompts": batch_prompts,
                    "shape_bucket_id": shape_bucket_id,
                    "exact_shape_homogeneous": exact_shape_homogeneous,
                    "batch_target_tokens": batch_target_tokens,
                    "batch_max_sequence_length": batch_max_sequence_length,
                    "estimated_batch_memory_mb": round(estimated_batch_memory_mb, 2),
                    "gpu_utilization_peak_pct": gpu_metrics.get(
                        "gpu_utilization_peak_pct"
                    ),
                    "gpu_memory_used_peak_mb": gpu_metrics.get(
                        "gpu_memory_used_peak_mb"
                    ),
                    "torch_peak_allocated_mb": gpu_metrics.get(
                        "torch_peak_allocated_mb"
                    ),
                    "torch_peak_reserved_mb": gpu_metrics.get(
                        "torch_peak_reserved_mb"
                    ),
                    "batch_exec_ms": round(batch_exec_ms, 2),
                    "lane": jobs[0].batch_key.lane,
                    "stage_timings_ms": {
                        key: round(value, 2)
                        for key, value in sorted(stage_timings_ms.items())
                    },
                }

            logger.info(
                "[%s] batch status=success requests=%d prompts=%d lane=%s shape_bucket=%s exact_shape_homogeneous=%s exec_ms=%.2f target_tokens=%d max_sequence_length=%d est_mem_mb=%.2f gpu_peak_util_pct=%s gpu_peak_used_mb=%s gpu_free_before_mb=%s gpu_free_after_mb=%s torch_alloc_mb=%s torch_reserved_mb=%s torch_peak_alloc_mb=%s torch_peak_reserved_mb=%s stage_timings_ms=%s",
                self._name,
                batch_requests,
                batch_prompts,
                jobs[0].batch_key.lane,
                shape_bucket_id,
                exact_shape_homogeneous,
                batch_exec_ms,
                batch_target_tokens,
                batch_max_sequence_length,
                estimated_batch_memory_mb,
                gpu_metrics.get("gpu_utilization_peak_pct"),
                gpu_metrics.get("gpu_memory_used_peak_mb"),
                gpu_metrics.get("gpu_memory_free_before_mb"),
                gpu_metrics.get("gpu_memory_free_after_mb"),
                gpu_metrics.get("torch_allocated_mb"),
                gpu_metrics.get("torch_reserved_mb"),
                gpu_metrics.get("torch_peak_allocated_mb"),
                gpu_metrics.get("torch_peak_reserved_mb"),
                {
                    key: round(value, 2)
                    for key, value in sorted(stage_timings_ms.items())
                },
            )
        except Exception as exc:
            self._gpu_monitor.end_batch()
            with self._stats_lock:
                self._jobs_failed += len(jobs)
            for job in jobs:
                job.future.set_exception(exc)
            logger.exception(
                "[%s] batch status=failed requests=%d lane=%s",
                self._name,
                len(jobs),
                jobs[0].batch_key.lane if jobs else "unknown",
            )
