#!/usr/bin/env python3

"""Reusable local generation service for OmniVoice serving backends."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.serving.batching import (
    ClonePromptCache,
    GenerationBatchKey,
    GenerationBatcher,
    GenerationBatcherConfig,
    PendingGeneration,
    build_clone_prompt_cache_key,
)

logger = logging.getLogger(__name__)


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_inference_dtype(device: str) -> torch.dtype:
    return torch.float32 if device == "cpu" else torch.float16


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_language(value: Optional[str]) -> Optional[str]:
    stripped = _normalize_optional_text(value)
    if stripped is None or stripped.lower() == "auto":
        return None
    return stripped


def _save_bytes_to_tempfile(data: bytes, filename: Optional[str] = None) -> str:
    suffix = Path(filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(data)
        return tmp_file.name


def _audio_to_wav_bytes(audio: torch.Tensor, sampling_rate: int) -> bytes:
    waveform = audio.squeeze(0).detach().cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sampling_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _slugify_filename_text(text: str, max_len: int = 40) -> str:
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_"}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug[:max_len].rstrip("-")) or "audio"


def _persist_wav_bytes(
    wav_bytes: bytes,
    save_dir: Path,
    mode: str,
    text: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    text_slug = _slugify_filename_text(text)
    filename = f"{timestamp}_{mode}_{text_slug}_{os.getpid()}.wav"
    output_path = save_dir / filename
    output_path.write_bytes(wav_bytes)
    return output_path


def _iso_utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _call_optional_model_method(
    model: OmniVoice,
    method_name: str,
    flag_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    method = getattr(model, method_name, None)
    if method is None or not callable(method):
        raise RuntimeError(
            f"{flag_name} was provided, but this OmniVoice build does not "
            f"implement `{method_name}()`."
        )
    return method(*args, **kwargs)


def _describe_optional_runtime(obj: Any) -> Any:
    if obj is None:
        return None
    describe = getattr(obj, "describe_sessions", None)
    if callable(describe):
        try:
            return describe()
        except Exception:
            logger.exception("Failed to describe optional runtime sessions")
    return None


def _native_coreml_runtime_enabled(model: OmniVoice) -> bool:
    fn = getattr(model, "_native_coreml_runtime_enabled", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            logger.exception("Failed to query native Core ML runtime state")
    return False


@dataclass(frozen=True)
class GenerationServiceConfig:
    model_checkpoint: str = "k2-fsa/OmniVoice"
    device: Optional[str] = None
    load_asr: bool = True
    coreml_backbone: Optional[str] = None
    coreml_compute_units: str = "cpu_and_ne"
    coreml_backbone_allow_fixed_padding: bool = False
    coreml_decoder: Optional[str] = None
    coreml_decoder_compute_units: str = "all"
    onnx_backbone: Optional[str] = None
    onnx_provider: str = "auto"
    onnx_backbone_allow_fixed_padding: bool = False
    onnx_decoder: Optional[str] = None
    onnx_decoder_provider: str = "auto"
    save_dir: Optional[str] = None
    batch_collect_ms: float = 10.0
    max_batch_requests: int = 32
    max_batch_prompts: int = 32
    max_batch_target_tokens: int = 4096
    max_batch_conditioning_tokens: int = 8192
    max_batch_padding_ratio: float = 2.5
    clone_prompt_cache_size: int = 256
    gpu_memory_utilization: float = 0.85
    gpu_memory_reserve_mb: int = 1024
    max_estimated_batch_memory_mb: Optional[int] = None


@dataclass(frozen=True)
class GenerationRequestPayload:
    request_id: str
    mode: str
    text: str
    language: Optional[str] = None
    instruct: Optional[str] = None
    ref_text: Optional[str] = None
    ref_audio_bytes: Optional[bytes] = None
    ref_audio_filename: Optional[str] = None
    num_step: int = 32
    guidance_scale: float = 2.0
    speed: Optional[float] = None
    duration: Optional[float] = None
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    created_at_iso: Optional[str] = None


@dataclass(frozen=True)
class GenerationResponsePayload:
    request_id: str
    ok: bool
    status_code: int
    error: Optional[str]
    wav_bytes: Optional[bytes]
    headers: dict[str, str]
    worker_id: str
    worker_pid: int


class GenerationService:
    """Local model owner used by direct mode and worker subprocesses."""

    def __init__(
        self,
        config: GenerationServiceConfig,
        service_label: str = "direct",
    ):
        self.config = config
        self.service_label = service_label
        self.device = config.device or get_best_device()
        self.dtype = get_inference_dtype(self.device)
        self.save_dir = Path(config.save_dir).expanduser() if config.save_dir else None
        self.worker_id = service_label
        self.worker_pid = os.getpid()

        logger.info(
            "[%s] Loading model from %s on %s ...",
            self.service_label,
            config.model_checkpoint,
            self.device,
        )
        self.model = OmniVoice.from_pretrained(
            config.model_checkpoint,
            device_map=self.device,
            dtype=self.dtype,
            load_asr=config.load_asr,
        )

        if config.coreml_backbone:
            logger.info(
                "[%s] Loading native Core ML backbone from %s ...",
                self.service_label,
                config.coreml_backbone,
            )
            _call_optional_model_method(
                self.model,
                "load_coreml_backbone",
                "--coreml-backbone",
                config.coreml_backbone,
                compute_units=config.coreml_compute_units,
                allow_fixed_shape_padding=config.coreml_backbone_allow_fixed_padding,
            )
        if config.coreml_decoder:
            logger.info(
                "[%s] Loading native Core ML decoder from %s ...",
                self.service_label,
                config.coreml_decoder,
            )
            _call_optional_model_method(
                self.model,
                "load_coreml_decoder",
                "--coreml-decoder",
                config.coreml_decoder,
                compute_units=config.coreml_decoder_compute_units,
            )
        if config.onnx_backbone:
            logger.info(
                "[%s] Loading ONNX backbone from %s ...",
                self.service_label,
                config.onnx_backbone,
            )
            _call_optional_model_method(
                self.model,
                "load_onnx_backbone",
                "--onnx-backbone",
                config.onnx_backbone,
                provider=config.onnx_provider,
                allow_fixed_shape_padding=config.onnx_backbone_allow_fixed_padding,
            )
        if config.onnx_decoder:
            logger.info(
                "[%s] Loading ONNX decoder from %s ...",
                self.service_label,
                config.onnx_decoder,
            )
            _call_optional_model_method(
                self.model,
                "load_onnx_decoder",
                "--onnx-decoder",
                config.onnx_decoder,
                provider=config.onnx_decoder_provider,
            )

        self.batch_config = GenerationBatcherConfig(
            collect_ms=config.batch_collect_ms,
            max_batch_requests=max(1, int(config.max_batch_requests)),
            max_batch_prompts=max(1, int(config.max_batch_prompts)),
            max_total_target_tokens=max(1, int(config.max_batch_target_tokens)),
            max_total_conditioning_tokens=max(
                1, int(config.max_batch_conditioning_tokens)
            ),
            max_padding_ratio=max(1.0, float(config.max_batch_padding_ratio)),
            gpu_memory_utilization=min(
                max(float(config.gpu_memory_utilization), 0.05), 1.0
            ),
            gpu_memory_reserve_mb=max(0, int(config.gpu_memory_reserve_mb)),
            max_estimated_batch_memory_mb=(
                max(1, int(config.max_estimated_batch_memory_mb))
                if config.max_estimated_batch_memory_mb is not None
                else None
            ),
        )
        self.clone_prompt_cache = ClonePromptCache(
            max_entries=max(1, int(config.clone_prompt_cache_size))
        )
        self.generation_batcher = GenerationBatcher(
            model=self.model,
            config=self.batch_config,
            name=f"omnivoice-batcher-{self.service_label}",
        )
        logger.info(
            "[%s] Model loaded. Sampling rate=%s",
            self.service_label,
            self.model.sampling_rate,
        )

    def close(self) -> None:
        self.generation_batcher.close()

    def health(self) -> dict[str, Any]:
        coreml_backbone_runtime = getattr(self.model, "_coreml_backbone", None)
        coreml_decoder_runtime = getattr(self.model, "_coreml_decoder", None)
        onnx_backbone_runtime = getattr(self.model, "_onnx_backbone", None)
        onnx_decoder_runtime = getattr(self.model, "_onnx_decoder", None)
        return {
            "service_label": self.service_label,
            "worker_id": self.worker_id,
            "worker_pid": self.worker_pid,
            "model": self.config.model_checkpoint,
            "device": self.device,
            "sampling_rate": self.model.sampling_rate,
            "asr_loaded": self.model._asr_pipe is not None,
            "native_coreml_runtime_active": _native_coreml_runtime_enabled(
                self.model
            ),
            "native_clone_requires_ref_text": _native_coreml_runtime_enabled(
                self.model
            ),
            "coreml_backbone": self.config.coreml_backbone,
            "coreml_compute_units": (
                self.config.coreml_compute_units if self.config.coreml_backbone else None
            ),
            "coreml_backbone_allow_fixed_padding": (
                self.config.coreml_backbone_allow_fixed_padding
                if self.config.coreml_backbone
                else None
            ),
            "coreml_backbone_sessions": _describe_optional_runtime(
                coreml_backbone_runtime
            ),
            "coreml_decoder": self.config.coreml_decoder,
            "coreml_decoder_compute_units": (
                self.config.coreml_decoder_compute_units
                if self.config.coreml_decoder
                else None
            ),
            "coreml_decoder_sessions": _describe_optional_runtime(
                coreml_decoder_runtime
            ),
            "onnx_backbone": self.config.onnx_backbone,
            "onnx_provider": (
                self.config.onnx_provider if self.config.onnx_backbone else None
            ),
            "onnx_backbone_allow_fixed_padding": (
                self.config.onnx_backbone_allow_fixed_padding
                if self.config.onnx_backbone
                else None
            ),
            "onnx_backbone_sessions": _describe_optional_runtime(
                onnx_backbone_runtime
            ),
            "onnx_decoder": self.config.onnx_decoder,
            "onnx_decoder_provider": (
                self.config.onnx_decoder_provider if self.config.onnx_decoder else None
            ),
            "onnx_runtime_providers": getattr(
                onnx_backbone_runtime, "providers", None
            ),
            "onnx_decoder_runtime_providers": getattr(
                onnx_decoder_runtime, "providers", None
            ),
            "save_dir": str(self.save_dir) if self.save_dir else None,
            "batching": self.generation_batcher.stats(),
            "clone_prompt_cache": self.clone_prompt_cache.stats(),
        }

    def generate(
        self,
        request: GenerationRequestPayload,
    ) -> GenerationResponsePayload:
        started_at = request.created_at_iso or _iso_utc_now()
        request_started = time.perf_counter()
        cleaned_text = _normalize_optional_text(request.text)
        cleaned_instruct = _normalize_optional_text(request.instruct)
        cleaned_ref_text = _normalize_optional_text(request.ref_text)
        cleaned_language = _normalize_language(request.language)
        saved_path = None
        audio_duration = None
        latency_ms = None
        rtf = None
        queue_wait_ms = None
        batch_exec_ms = None
        batch_requests = None
        batch_prompts = None
        batch_target_tokens = None
        batch_max_sequence_length = None
        estimated_batch_memory_mb = None
        gpu_metrics: dict[str, object] = {}
        lane = None

        try:
            if cleaned_text is None:
                raise ValueError("text is required.")

            if request.mode == "auto":
                if request.ref_audio_bytes is not None:
                    raise ValueError("ref_audio is only supported for clone mode.")
                if cleaned_instruct is not None:
                    raise ValueError(
                        "instruct is only supported for design or clone mode."
                    )
                if cleaned_ref_text is not None:
                    raise ValueError("ref_text is only supported for clone mode.")

            if request.mode == "design":
                if cleaned_instruct is None:
                    raise ValueError("instruct is required for design mode.")
                if request.ref_audio_bytes is not None:
                    raise ValueError("ref_audio is only supported for clone mode.")
                if cleaned_ref_text is not None:
                    raise ValueError("ref_text is only supported for clone mode.")

            if request.mode == "clone" and request.ref_audio_bytes is None:
                raise ValueError("ref_audio is required for clone mode.")

            if (
                request.mode == "clone"
                and _native_coreml_runtime_enabled(self.model)
                and cleaned_ref_text is None
            ):
                raise ValueError(
                    "Clone mode with native Core ML runtime requires ref_text. "
                    "Whisper auto-transcription is not part of the Apple-native path."
                )

            generation_config = OmniVoiceGenerationConfig(
                num_step=int(request.num_step),
                guidance_scale=float(request.guidance_scale),
                denoise=bool(request.denoise),
                preprocess_prompt=bool(request.preprocess_prompt),
                postprocess_output=bool(request.postprocess_output),
            )

            task_kwargs: dict[str, Any] = {
                "text": cleaned_text,
                "language": cleaned_language,
            }
            if cleaned_instruct is not None:
                task_kwargs["instruct"] = cleaned_instruct
            if request.speed is not None and float(request.speed) != 1.0:
                task_kwargs["speed"] = float(request.speed)
            if request.duration is not None:
                if float(request.duration) <= 0:
                    raise ValueError("duration must be greater than 0 when provided.")
                task_kwargs["duration"] = float(request.duration)

            if request.ref_audio_bytes is not None:
                cache_key = build_clone_prompt_cache_key(
                    request.ref_audio_bytes,
                    cleaned_ref_text,
                    bool(request.preprocess_prompt),
                )

                def _create_prompt():
                    temp_path = _save_bytes_to_tempfile(
                        request.ref_audio_bytes or b"",
                        request.ref_audio_filename,
                    )
                    try:
                        return self.model.create_voice_clone_prompt(
                            ref_audio=temp_path,
                            ref_text=cleaned_ref_text,
                            preprocess_prompt=bool(request.preprocess_prompt),
                        )
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

                task_kwargs["voice_clone_prompt"] = self.clone_prompt_cache.get_or_create(
                    cache_key,
                    _create_prompt,
                )

            prepared_task = self.model.prepare_generation_task(
                preprocess_prompt=bool(request.preprocess_prompt),
                **task_kwargs,
            )

            estimated_sequence_lengths = [
                self.model.estimate_inference_sequence_length(
                    text=prepared_task.texts[i],
                    num_target_tokens=prepared_task.target_lens[i],
                    ref_text=prepared_task.ref_texts[i],
                    ref_audio_tokens=prepared_task.ref_audio_tokens[i],
                    lang=prepared_task.langs[i],
                    instruct=prepared_task.instructs[i],
                    denoise=generation_config.denoise,
                )
                for i in range(prepared_task.batch_size)
            ]
            short_idx, long_idx = prepared_task.get_indices(
                generation_config,
                self.model.audio_tokenizer.config.frame_rate,
            )
            if short_idx:
                lane = "short_mixed"
            elif prepared_task.ref_audio_tokens[0] is not None:
                lane = "long_ref"
            else:
                lane = "long_no_ref"
            assert not (
                short_idx and long_idx
            ), "Single request should map to one batching lane"

            batch_key = GenerationBatchKey(
                num_step=generation_config.num_step,
                guidance_scale=generation_config.guidance_scale,
                t_shift=generation_config.t_shift,
                layer_penalty_factor=generation_config.layer_penalty_factor,
                position_temperature=generation_config.position_temperature,
                class_temperature=generation_config.class_temperature,
                denoise=generation_config.denoise,
                audio_chunk_duration=generation_config.audio_chunk_duration,
                audio_chunk_threshold=generation_config.audio_chunk_threshold,
                lane=lane,
            )

            batch_result = self.generation_batcher.submit(
                PendingGeneration(
                    request_id=request.request_id,
                    mode=request.mode,
                    created_at=time.perf_counter(),
                    batch_key=batch_key,
                    task=prepared_task,
                    generation_config=generation_config,
                    postprocess_flags=[
                        bool(request.postprocess_output)
                    ]
                    * prepared_task.batch_size,
                    estimated_sequence_lengths=estimated_sequence_lengths,
                )
            )

            queue_wait_ms = batch_result.queue_wait_ms
            batch_exec_ms = batch_result.batch_exec_ms
            batch_requests = batch_result.batch_requests
            batch_prompts = batch_result.batch_prompts
            batch_target_tokens = batch_result.batch_target_tokens
            batch_max_sequence_length = batch_result.batch_max_sequence_length
            estimated_batch_memory_mb = batch_result.estimated_batch_memory_mb
            gpu_metrics = batch_result.gpu_metrics

            audio_duration = (
                batch_result.audios[0].shape[-1] / self.model.sampling_rate
            )
            wav_bytes = _audio_to_wav_bytes(
                batch_result.audios[0], self.model.sampling_rate
            )
            latency_ms = (time.perf_counter() - request_started) * 1000.0
            rtf = (
                (latency_ms / 1000.0) / audio_duration
                if audio_duration and audio_duration > 0
                else None
            )

            headers = {
                "Content-Disposition": 'inline; filename="omnivoice.wav"',
                "X-OmniVoice-Request-Id": request.request_id,
                "X-OmniVoice-Started-At": started_at,
                "X-OmniVoice-Finished-At": _iso_utc_now(),
                "X-OmniVoice-Latency-Ms": f"{latency_ms:.2f}",
                "X-OmniVoice-Audio-Duration-S": f"{audio_duration:.3f}",
                "X-OmniVoice-Queue-Wait-Ms": f"{queue_wait_ms:.2f}",
                "X-OmniVoice-Batch-Exec-Ms": f"{batch_exec_ms:.2f}",
                "X-OmniVoice-Batch-Requests": str(batch_requests),
                "X-OmniVoice-Batch-Prompts": str(batch_prompts),
                "X-OmniVoice-Batch-Target-Tokens": str(batch_target_tokens),
                "X-OmniVoice-Batch-Max-Sequence-Length": str(
                    batch_max_sequence_length
                ),
                "X-OmniVoice-Worker-Id": self.worker_id,
                "X-OmniVoice-Worker-Pid": str(self.worker_pid),
            }
            if estimated_batch_memory_mb is not None:
                headers["X-OmniVoice-Batch-Estimated-Memory-Mb"] = (
                    f"{estimated_batch_memory_mb:.2f}"
                )
            if gpu_metrics.get("gpu_utilization_peak_pct") is not None:
                headers["X-OmniVoice-GPU-Utilization-Peak-Pct"] = (
                    f"{float(gpu_metrics['gpu_utilization_peak_pct']):.2f}"
                )
            if gpu_metrics.get("gpu_memory_total_mb") is not None:
                headers["X-OmniVoice-GPU-Memory-Total-Mb"] = (
                    f"{float(gpu_metrics['gpu_memory_total_mb']):.2f}"
                )
            if gpu_metrics.get("gpu_memory_used_peak_mb") is not None:
                headers["X-OmniVoice-GPU-Memory-Used-Peak-Mb"] = (
                    f"{float(gpu_metrics['gpu_memory_used_peak_mb']):.2f}"
                )
            if gpu_metrics.get("gpu_memory_free_before_mb") is not None:
                headers["X-OmniVoice-GPU-Memory-Free-Before-Mb"] = (
                    f"{float(gpu_metrics['gpu_memory_free_before_mb']):.2f}"
                )
            if gpu_metrics.get("gpu_memory_free_after_mb") is not None:
                headers["X-OmniVoice-GPU-Memory-Free-After-Mb"] = (
                    f"{float(gpu_metrics['gpu_memory_free_after_mb']):.2f}"
                )
            if gpu_metrics.get("torch_allocated_mb") is not None:
                headers["X-OmniVoice-GPU-Allocator-Allocated-Mb"] = (
                    f"{float(gpu_metrics['torch_allocated_mb']):.2f}"
                )
            if gpu_metrics.get("torch_reserved_mb") is not None:
                headers["X-OmniVoice-GPU-Allocator-Reserved-Mb"] = (
                    f"{float(gpu_metrics['torch_reserved_mb']):.2f}"
                )
            if gpu_metrics.get("torch_peak_allocated_mb") is not None:
                headers["X-OmniVoice-GPU-Allocator-Peak-Allocated-Mb"] = (
                    f"{float(gpu_metrics['torch_peak_allocated_mb']):.2f}"
                )
            if gpu_metrics.get("torch_peak_reserved_mb") is not None:
                headers["X-OmniVoice-GPU-Allocator-Peak-Reserved-Mb"] = (
                    f"{float(gpu_metrics['torch_peak_reserved_mb']):.2f}"
                )
            if rtf is not None:
                headers["X-OmniVoice-RTF"] = f"{rtf:.4f}"
            if self.save_dir is not None:
                saved_path = _persist_wav_bytes(
                    wav_bytes=wav_bytes,
                    save_dir=self.save_dir,
                    mode=request.mode,
                    text=cleaned_text,
                )
                headers["X-OmniVoice-Saved-Path"] = str(saved_path)

            logger.info(
                "[%s] request_id=%s status=success mode=%s started_at=%s finished_at=%s latency_ms=%.2f queue_wait_ms=%.2f batch_exec_ms=%.2f batch_requests=%s batch_prompts=%s batch_est_mem_mb=%s gpu_peak_util_pct=%s gpu_peak_used_mb=%s torch_peak_alloc_mb=%s torch_peak_reserved_mb=%s lane=%s audio_s=%.3f rtf=%s text_chars=%d has_ref_audio=%s language=%s device=%s saved_path=%s",
                self.service_label,
                request.request_id,
                request.mode,
                started_at,
                headers["X-OmniVoice-Finished-At"],
                latency_ms,
                queue_wait_ms,
                batch_exec_ms,
                batch_requests,
                batch_prompts,
                f"{estimated_batch_memory_mb:.2f}"
                if estimated_batch_memory_mb is not None
                else "n/a",
                (
                    f"{float(gpu_metrics['gpu_utilization_peak_pct']):.2f}"
                    if gpu_metrics.get("gpu_utilization_peak_pct") is not None
                    else "n/a"
                ),
                (
                    f"{float(gpu_metrics['gpu_memory_used_peak_mb']):.2f}"
                    if gpu_metrics.get("gpu_memory_used_peak_mb") is not None
                    else "n/a"
                ),
                (
                    f"{float(gpu_metrics['torch_peak_allocated_mb']):.2f}"
                    if gpu_metrics.get("torch_peak_allocated_mb") is not None
                    else "n/a"
                ),
                (
                    f"{float(gpu_metrics['torch_peak_reserved_mb']):.2f}"
                    if gpu_metrics.get("torch_peak_reserved_mb") is not None
                    else "n/a"
                ),
                lane,
                audio_duration,
                f"{rtf:.4f}" if rtf is not None else "n/a",
                len(cleaned_text),
                request.ref_audio_bytes is not None,
                cleaned_language or "auto",
                self.device,
                str(saved_path) if saved_path else "-",
            )

            return GenerationResponsePayload(
                request_id=request.request_id,
                ok=True,
                status_code=200,
                error=None,
                wav_bytes=wav_bytes,
                headers=headers,
                worker_id=self.worker_id,
                worker_pid=self.worker_pid,
            )
        except ValueError as exc:
            logger.warning(
                "[%s] request_id=%s status=invalid_request started_at=%s error=%s",
                self.service_label,
                request.request_id,
                started_at,
                exc,
            )
            return GenerationResponsePayload(
                request_id=request.request_id,
                ok=False,
                status_code=400,
                error=str(exc),
                wav_bytes=None,
                headers={
                    "X-OmniVoice-Request-Id": request.request_id,
                    "X-OmniVoice-Worker-Id": self.worker_id,
                    "X-OmniVoice-Worker-Pid": str(self.worker_pid),
                },
                worker_id=self.worker_id,
                worker_pid=self.worker_pid,
            )
        except RuntimeError as exc:
            logger.exception(
                "[%s] request_id=%s status=runtime_error",
                self.service_label,
                request.request_id,
            )
            return GenerationResponsePayload(
                request_id=request.request_id,
                ok=False,
                status_code=500,
                error=str(exc),
                wav_bytes=None,
                headers={
                    "X-OmniVoice-Request-Id": request.request_id,
                    "X-OmniVoice-Worker-Id": self.worker_id,
                    "X-OmniVoice-Worker-Pid": str(self.worker_pid),
                },
                worker_id=self.worker_id,
                worker_pid=self.worker_pid,
            )
        except Exception as exc:
            logger.exception(
                "[%s] request_id=%s status=failed",
                self.service_label,
                request.request_id,
            )
            return GenerationResponsePayload(
                request_id=request.request_id,
                ok=False,
                status_code=500,
                error=f"{type(exc).__name__}: {exc}",
                wav_bytes=None,
                headers={
                    "X-OmniVoice-Request-Id": request.request_id,
                    "X-OmniVoice-Worker-Id": self.worker_id,
                    "X-OmniVoice-Worker-Pid": str(self.worker_pid),
                },
                worker_id=self.worker_id,
                worker_pid=self.worker_pid,
            )
