#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI server for OmniVoice."""

from __future__ import annotations

import argparse
import logging
import os
from enum import Enum
from typing import Optional
from uuid import uuid4

import anyio.to_thread
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from omnivoice import __version__
from omnivoice.serving import (
    GenerationRequestPayload,
    GenerationService,
    GenerationServiceConfig,
    MultiWorkerGenerationBackend,
)
from omnivoice.utils.lang_map import LANG_NAME_TO_ID, lang_display_name

logger = logging.getLogger(__name__)


class GenerateMode(str, Enum):
    auto = "auto"
    design = "design"
    clone = "clone"


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-api",
        description="Launch a FastAPI server for OmniVoice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use. Auto-detected if not specified.",
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP to bind to.")
    parser.add_argument("--port", type=int, default=8002, help="Server port.")
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR at startup. Clone mode without ref_text "
        "will load ASR on demand instead.",
    )
    parser.add_argument(
        "--same-gpu-workers",
        "--gpu-workers",
        dest="same_gpu_workers",
        type=int,
        default=1,
        help="Number of same-GPU OmniVoice worker processes to run. Use 1 for direct mode.",
    )
    parser.add_argument(
        "--api-thread-limit",
        type=int,
        default=256,
        help="Thread limit for FastAPI sync request handlers. Raise this for large request bursts.",
    )
    parser.add_argument(
        "--onnx-backbone",
        default=None,
        help="Optional ONNX backbone file, directory, or comma-separated path list for ORT/CoreML acceleration.",
    )
    parser.add_argument(
        "--coreml-backbone",
        default=None,
        help="Optional native Core ML backbone .mlpackage path, directory, or comma-separated path list.",
    )
    parser.add_argument(
        "--coreml-compute-units",
        default="cpu_and_ne",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        help="Compute-unit preference when --coreml-backbone is set.",
    )
    parser.add_argument(
        "--coreml-backbone-allow-fixed-padding",
        action="store_true",
        default=False,
        help="Allow fixed-shape native Core ML backbone exports to pad shorter requests up to the exported seq-len.",
    )
    parser.add_argument(
        "--coreml-decoder",
        default=None,
        help="Optional native Core ML decoder .mlpackage path, directory, or comma-separated path list.",
    )
    parser.add_argument(
        "--coreml-decoder-compute-units",
        default="all",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        help="Compute-unit preference when --coreml-decoder is set.",
    )
    parser.add_argument(
        "--onnx-provider",
        default="auto",
        choices=["auto", "cpu", "coreml"],
        help="Provider preference when --onnx-backbone is set.",
    )
    parser.add_argument(
        "--onnx-backbone-allow-fixed-padding",
        action="store_true",
        default=False,
        help="Allow fixed-shape ONNX backbone exports to pad shorter requests up to the exported seq-len.",
    )
    parser.add_argument(
        "--onnx-decoder",
        default=None,
        help="Optional ONNX decoder path for decoder-side ORT acceleration.",
    )
    parser.add_argument(
        "--onnx-decoder-provider",
        default="auto",
        choices=["auto", "cpu", "coreml"],
        help="Provider preference when --onnx-decoder is set.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to persist a copy of each generated WAV.",
    )
    parser.add_argument(
        "--batch-collect-ms",
        type=float,
        default=10.0,
        help="Micro-batch collection window in milliseconds.",
    )
    parser.add_argument(
        "--max-batch-requests",
        type=int,
        default=32,
        help="Maximum queued API requests to merge into one inference batch.",
    )
    parser.add_argument(
        "--max-batch-prompts",
        type=int,
        default=32,
        help="Maximum prompt units allowed in one batch.",
    )
    parser.add_argument(
        "--max-batch-target-tokens",
        type=int,
        default=4096,
        help="Maximum summed target tokens in one batch.",
    )
    parser.add_argument(
        "--max-batch-conditioning-tokens",
        type=int,
        default=8192,
        help="Maximum summed padded sequence tokens in one batch.",
    )
    parser.add_argument(
        "--max-batch-padding-ratio",
        type=float,
        default=2.5,
        help="Maximum allowed padding ratio for a selected batch.",
    )
    parser.add_argument(
        "--clone-prompt-cache-size",
        type=int,
        default=256,
        help="Maximum number of prepared clone prompts to cache in memory.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of currently free CUDA memory the batcher may target for one merged batch.",
    )
    parser.add_argument(
        "--gpu-memory-reserve-mb",
        type=int,
        default=1024,
        help="CUDA memory to keep in reserve for allocator overhead and runtime jitter.",
    )
    parser.add_argument(
        "--max-estimated-batch-memory-mb",
        type=int,
        default=None,
        help="Optional hard cap for estimated incremental batch memory. Overrides automatic CUDA-based budgeting when set.",
    )
    return parser


def _build_service_config(
    model_checkpoint: str,
    device: Optional[str],
    load_asr: bool,
    coreml_backbone: Optional[str],
    coreml_compute_units: str,
    coreml_backbone_allow_fixed_padding: bool,
    coreml_decoder: Optional[str],
    coreml_decoder_compute_units: str,
    onnx_backbone: Optional[str],
    onnx_provider: str,
    onnx_backbone_allow_fixed_padding: bool,
    onnx_decoder: Optional[str],
    onnx_decoder_provider: str,
    save_dir: Optional[str],
    batch_collect_ms: float,
    max_batch_requests: int,
    max_batch_prompts: int,
    max_batch_target_tokens: int,
    max_batch_conditioning_tokens: int,
    max_batch_padding_ratio: float,
    clone_prompt_cache_size: int,
    gpu_memory_utilization: float,
    gpu_memory_reserve_mb: int,
    max_estimated_batch_memory_mb: Optional[int],
) -> GenerationServiceConfig:
    return GenerationServiceConfig(
        model_checkpoint=model_checkpoint,
        device=device,
        load_asr=load_asr,
        coreml_backbone=coreml_backbone,
        coreml_compute_units=coreml_compute_units,
        coreml_backbone_allow_fixed_padding=coreml_backbone_allow_fixed_padding,
        coreml_decoder=coreml_decoder,
        coreml_decoder_compute_units=coreml_decoder_compute_units,
        onnx_backbone=onnx_backbone,
        onnx_provider=onnx_provider,
        onnx_backbone_allow_fixed_padding=onnx_backbone_allow_fixed_padding,
        onnx_decoder=onnx_decoder,
        onnx_decoder_provider=onnx_decoder_provider,
        save_dir=save_dir,
        batch_collect_ms=batch_collect_ms,
        max_batch_requests=max_batch_requests,
        max_batch_prompts=max_batch_prompts,
        max_batch_target_tokens=max_batch_target_tokens,
        max_batch_conditioning_tokens=max_batch_conditioning_tokens,
        max_batch_padding_ratio=max_batch_padding_ratio,
        clone_prompt_cache_size=clone_prompt_cache_size,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_memory_reserve_mb=gpu_memory_reserve_mb,
        max_estimated_batch_memory_mb=max_estimated_batch_memory_mb,
    )


def create_app(
    model_checkpoint: str = "k2-fsa/OmniVoice",
    device: Optional[str] = None,
    load_asr: bool = True,
    same_gpu_workers: int = 1,
    api_thread_limit: int = 256,
    coreml_backbone: Optional[str] = None,
    coreml_compute_units: str = "cpu_and_ne",
    coreml_backbone_allow_fixed_padding: bool = False,
    coreml_decoder: Optional[str] = None,
    coreml_decoder_compute_units: str = "all",
    onnx_backbone: Optional[str] = None,
    onnx_provider: str = "auto",
    onnx_backbone_allow_fixed_padding: bool = False,
    onnx_decoder: Optional[str] = None,
    onnx_decoder_provider: str = "auto",
    save_dir: Optional[str] = None,
    batch_collect_ms: float = 10.0,
    max_batch_requests: int = 32,
    max_batch_prompts: int = 32,
    max_batch_target_tokens: int = 4096,
    max_batch_conditioning_tokens: int = 8192,
    max_batch_padding_ratio: float = 2.5,
    clone_prompt_cache_size: int = 256,
    gpu_memory_utilization: float = 0.85,
    gpu_memory_reserve_mb: int = 1024,
    max_estimated_batch_memory_mb: Optional[int] = None,
) -> FastAPI:
    resolved_device = device or get_best_device()
    if same_gpu_workers < 1:
        raise ValueError("same_gpu_workers must be >= 1")
    if same_gpu_workers > 1 and not resolved_device.startswith("cuda"):
        raise ValueError(
            "same_gpu_workers > 1 is only supported on CUDA devices."
        )

    service_config = _build_service_config(
        model_checkpoint=model_checkpoint,
        device=resolved_device,
        load_asr=load_asr,
        coreml_backbone=coreml_backbone,
        coreml_compute_units=coreml_compute_units,
        coreml_backbone_allow_fixed_padding=coreml_backbone_allow_fixed_padding,
        coreml_decoder=coreml_decoder,
        coreml_decoder_compute_units=coreml_decoder_compute_units,
        onnx_backbone=onnx_backbone,
        onnx_provider=onnx_provider,
        onnx_backbone_allow_fixed_padding=onnx_backbone_allow_fixed_padding,
        onnx_decoder=onnx_decoder,
        onnx_decoder_provider=onnx_decoder_provider,
        save_dir=save_dir,
        batch_collect_ms=batch_collect_ms,
        max_batch_requests=max_batch_requests,
        max_batch_prompts=max_batch_prompts,
        max_batch_target_tokens=max_batch_target_tokens,
        max_batch_conditioning_tokens=max_batch_conditioning_tokens,
        max_batch_padding_ratio=max_batch_padding_ratio,
        clone_prompt_cache_size=clone_prompt_cache_size,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_memory_reserve_mb=gpu_memory_reserve_mb,
        max_estimated_batch_memory_mb=max_estimated_batch_memory_mb,
    )

    if same_gpu_workers > 1:
        backend = MultiWorkerGenerationBackend(
            config=service_config,
            worker_count=same_gpu_workers,
        )
        logger.info(
            "Serving backend initialized in same_gpu_workers mode with %d workers on %s",
            same_gpu_workers,
            resolved_device,
        )
    else:
        backend = GenerationService(service_config, service_label="direct")
        logger.info(
            "Serving backend initialized in direct mode on %s",
            resolved_device,
        )

    app = FastAPI(
        title="OmniVoice API",
        version=__version__,
        description="HTTP API for OmniVoice text-to-speech inference.",
    )
    app.state.backend = backend
    app.state.api_thread_limit = max(32, int(api_thread_limit))

    @app.on_event("startup")
    async def startup_event() -> None:
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max(limiter.total_tokens, app.state.api_thread_limit)
        logger.info(
            "Configured AnyIO thread limiter to %d tokens",
            limiter.total_tokens,
        )

    @app.get("/health")
    def health() -> dict:
        backend_health = app.state.backend.health()
        if isinstance(app.state.backend, GenerationService):
            return {
                "status": "ok",
                "backend_mode": "direct",
                "gpu_workers": 1,
                "api_thread_limit": app.state.api_thread_limit,
                **backend_health,
            }
        return {
            "status": "ok",
            "api_thread_limit": app.state.api_thread_limit,
            **backend_health,
        }

    @app.get("/languages")
    def languages() -> dict:
        items = [
            {
                "id": LANG_NAME_TO_ID[name],
                "name": name,
                "display_name": lang_display_name(name),
            }
            for name in sorted(LANG_NAME_TO_ID, key=lang_display_name)
        ]
        return {"count": len(items), "languages": items}

    @app.post(
        "/generate",
        response_class=Response,
        responses={200: {"content": {"audio/wav": {}}}},
    )
    def generate(
        mode: GenerateMode = Form(...),
        text: str = Form(...),
        language: Optional[str] = Form(None),
        instruct: Optional[str] = Form(None),
        ref_text: Optional[str] = Form(None),
        num_step: int = Form(32),
        guidance_scale: float = Form(2.0),
        speed: Optional[float] = Form(None),
        duration: Optional[float] = Form(None),
        denoise: bool = Form(True),
        preprocess_prompt: bool = Form(True),
        postprocess_output: bool = Form(True),
        ref_audio: UploadFile | None = File(None),
    ) -> Response:
        request_id = uuid4().hex[:12]
        ref_audio_bytes = None
        ref_audio_filename = None
        try:
            if ref_audio is not None:
                ref_audio_bytes = ref_audio.file.read()
                ref_audio_filename = ref_audio.filename

            response = app.state.backend.generate(
                GenerationRequestPayload(
                    request_id=request_id,
                    mode=mode.value,
                    text=text,
                    language=language,
                    instruct=instruct,
                    ref_text=ref_text,
                    ref_audio_bytes=ref_audio_bytes,
                    ref_audio_filename=ref_audio_filename,
                    num_step=num_step,
                    guidance_scale=guidance_scale,
                    speed=speed,
                    duration=duration,
                    denoise=denoise,
                    preprocess_prompt=preprocess_prompt,
                    postprocess_output=postprocess_output,
                )
            )
            if not response.ok:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.error or "Generation failed.",
                    headers=response.headers,
                )
            if response.wav_bytes is None:
                raise HTTPException(
                    status_code=500,
                    detail="Worker returned no audio bytes.",
                    headers=response.headers,
                )
            return Response(
                content=response.wav_bytes,
                media_type="audio/wav",
                headers=response.headers,
            )
        finally:
            if ref_audio is not None:
                ref_audio.file.close()

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        app.state.backend.close()

    return app


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = os.environ.get("OMNIVOICE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [pid=%(process)d] %(name)s %(levelname)s: %(message)s",
    )

    app = create_app(
        model_checkpoint=args.model,
        device=args.device,
        load_asr=not args.no_asr,
        same_gpu_workers=args.same_gpu_workers,
        api_thread_limit=args.api_thread_limit,
        coreml_backbone=args.coreml_backbone,
        coreml_compute_units=args.coreml_compute_units,
        coreml_backbone_allow_fixed_padding=args.coreml_backbone_allow_fixed_padding,
        coreml_decoder=args.coreml_decoder,
        coreml_decoder_compute_units=args.coreml_decoder_compute_units,
        onnx_backbone=args.onnx_backbone,
        onnx_provider=args.onnx_provider,
        onnx_backbone_allow_fixed_padding=args.onnx_backbone_allow_fixed_padding,
        onnx_decoder=args.onnx_decoder,
        onnx_decoder_provider=args.onnx_decoder_provider,
        save_dir=args.save_dir,
        batch_collect_ms=args.batch_collect_ms,
        max_batch_requests=args.max_batch_requests,
        max_batch_prompts=args.max_batch_prompts,
        max_batch_target_tokens=args.max_batch_target_tokens,
        max_batch_conditioning_tokens=args.max_batch_conditioning_tokens,
        max_batch_padding_ratio=args.max_batch_padding_ratio,
        clone_prompt_cache_size=args.clone_prompt_cache_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_reserve_mb=args.gpu_memory_reserve_mb,
        max_estimated_batch_memory_mb=args.max_estimated_batch_memory_mb,
    )

    uvicorn.run(
        app,
        host=args.ip,
        port=args.port,
        root_path=args.root_path or "",
    )


if __name__ == "__main__":
    main()
