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

"""FastAPI server for OmniVoice.

Exposes a minimal HTTP API around the existing OmniVoice Python inference API.

Usage:
    omnivoice-api --model k2-fsa/OmniVoice --ip 0.0.0.0 --port 8002
"""

import argparse
import io
import logging
import os
import tempfile
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from uuid import uuid4

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from omnivoice import OmniVoice, OmniVoiceGenerationConfig, __version__
from omnivoice.utils.lang_map import LANG_NAME_TO_ID, lang_display_name

logger = logging.getLogger(__name__)


class GenerateMode(str, Enum):
    auto = "auto"
    design = "design"
    clone = "clone"


def get_best_device() -> str:
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_inference_dtype(device: str) -> torch.dtype:
    """Use float32 on CPU to avoid slow or unsupported half-precision paths."""
    return torch.float32 if device == "cpu" else torch.float16


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
    return parser


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


def _save_upload_to_tempfile(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(upload.file.read())
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
    mode: GenerateMode,
    text: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    text_slug = _slugify_filename_text(text)
    filename = f"{timestamp}_{mode.value}_{text_slug}_{uuid4().hex[:8]}.wav"
    output_path = save_dir / filename
    output_path.write_bytes(wav_bytes)
    return output_path


def _iso_utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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


def create_app(
    model_checkpoint: str = "k2-fsa/OmniVoice",
    device: Optional[str] = None,
    load_asr: bool = True,
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
) -> FastAPI:
    device = device or get_best_device()
    dtype = get_inference_dtype(device)

    logger.info("Loading model from %s on %s ...", model_checkpoint, device)
    model = OmniVoice.from_pretrained(
        model_checkpoint,
        device_map=device,
        dtype=dtype,
        load_asr=load_asr,
    )

    if coreml_backbone:
        logger.info(
            "Loading native Core ML backbone from %s with compute_units=%s (allow_fixed_padding=%s) ...",
            coreml_backbone,
            coreml_compute_units,
            coreml_backbone_allow_fixed_padding,
        )
        _call_optional_model_method(
            model,
            "load_coreml_backbone",
            "--coreml-backbone",
            coreml_backbone,
            compute_units=coreml_compute_units,
            allow_fixed_shape_padding=coreml_backbone_allow_fixed_padding,
        )
    if coreml_decoder:
        logger.info(
            "Loading native Core ML decoder from %s with compute_units=%s ...",
            coreml_decoder,
            coreml_decoder_compute_units,
        )
        _call_optional_model_method(
            model,
            "load_coreml_decoder",
            "--coreml-decoder",
            coreml_decoder,
            compute_units=coreml_decoder_compute_units,
        )
    if onnx_backbone:
        logger.info(
            "Loading ONNX backbone from %s with provider=%s (allow_fixed_padding=%s) ...",
            onnx_backbone,
            onnx_provider,
            onnx_backbone_allow_fixed_padding,
        )
        _call_optional_model_method(
            model,
            "load_onnx_backbone",
            "--onnx-backbone",
            onnx_backbone,
            provider=onnx_provider,
            allow_fixed_shape_padding=onnx_backbone_allow_fixed_padding,
        )
    if onnx_decoder:
        logger.info(
            "Loading ONNX decoder from %s with provider=%s ...",
            onnx_decoder,
            onnx_decoder_provider,
        )
        _call_optional_model_method(
            model,
            "load_onnx_decoder",
            "--onnx-decoder",
            onnx_decoder,
            provider=onnx_decoder_provider,
        )
    logger.info("Model loaded. Sampling rate=%s", model.sampling_rate)

    app = FastAPI(
        title="OmniVoice API",
        version=__version__,
        description="HTTP API for OmniVoice text-to-speech inference.",
    )
    app.state.model = model
    app.state.model_checkpoint = model_checkpoint
    app.state.device = device
    app.state.generate_lock = Lock()
    app.state.coreml_backbone = coreml_backbone
    app.state.coreml_compute_units = coreml_compute_units if coreml_backbone else None
    app.state.coreml_backbone_allow_fixed_padding = (
        coreml_backbone_allow_fixed_padding if coreml_backbone else None
    )
    app.state.coreml_decoder = coreml_decoder
    app.state.coreml_decoder_compute_units = (
        coreml_decoder_compute_units if coreml_decoder else None
    )
    app.state.onnx_backbone = onnx_backbone
    app.state.onnx_provider = onnx_provider if onnx_backbone else None
    app.state.onnx_backbone_allow_fixed_padding = (
        onnx_backbone_allow_fixed_padding if onnx_backbone else None
    )
    app.state.onnx_decoder = onnx_decoder
    app.state.onnx_decoder_provider = (
        onnx_decoder_provider if onnx_decoder else None
    )
    app.state.save_dir = Path(save_dir).expanduser() if save_dir else None

    @app.get("/health")
    def health() -> dict:
        coreml_backbone_runtime = getattr(app.state.model, "_coreml_backbone", None)
        coreml_decoder_runtime = getattr(app.state.model, "_coreml_decoder", None)
        onnx_backbone_runtime = getattr(app.state.model, "_onnx_backbone", None)
        onnx_decoder_runtime = getattr(app.state.model, "_onnx_decoder", None)
        return {
            "status": "ok",
            "model": app.state.model_checkpoint,
            "device": app.state.device,
            "sampling_rate": app.state.model.sampling_rate,
            "asr_loaded": app.state.model._asr_pipe is not None,
            "native_coreml_runtime_active": _native_coreml_runtime_enabled(
                app.state.model
            ),
            "native_clone_requires_ref_text": _native_coreml_runtime_enabled(
                app.state.model
            ),
            "coreml_backbone": app.state.coreml_backbone,
            "coreml_compute_units": app.state.coreml_compute_units,
            "coreml_backbone_allow_fixed_padding": app.state.coreml_backbone_allow_fixed_padding,
            "coreml_backbone_sessions": _describe_optional_runtime(
                coreml_backbone_runtime
            ),
            "coreml_decoder": app.state.coreml_decoder,
            "coreml_decoder_compute_units": app.state.coreml_decoder_compute_units,
            "coreml_decoder_sessions": _describe_optional_runtime(
                coreml_decoder_runtime
            ),
            "onnx_backbone": app.state.onnx_backbone,
            "onnx_provider": app.state.onnx_provider,
            "onnx_backbone_allow_fixed_padding": app.state.onnx_backbone_allow_fixed_padding,
            "onnx_backbone_sessions": _describe_optional_runtime(
                onnx_backbone_runtime
            ),
            "onnx_decoder": app.state.onnx_decoder,
            "onnx_decoder_provider": app.state.onnx_decoder_provider,
            "onnx_runtime_providers": getattr(onnx_backbone_runtime, "providers", None),
            "onnx_decoder_runtime_providers": getattr(
                onnx_decoder_runtime, "providers", None
            ),
            "save_dir": str(app.state.save_dir) if app.state.save_dir else None,
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
        started_at = _iso_utc_now()
        request_started = time.perf_counter()
        cleaned_text = _normalize_optional_text(text)
        cleaned_instruct = _normalize_optional_text(instruct)
        cleaned_ref_text = _normalize_optional_text(ref_text)
        cleaned_language = _normalize_language(language)
        saved_path = None
        audio_duration = None
        latency_ms = None
        rtf = None

        if cleaned_text is None:
            raise HTTPException(status_code=400, detail="text is required.")

        if mode == GenerateMode.auto:
            if ref_audio is not None:
                raise HTTPException(
                    status_code=400,
                    detail="ref_audio is only supported for clone mode.",
                )
            if cleaned_instruct is not None:
                raise HTTPException(
                    status_code=400,
                    detail="instruct is only supported for design or clone mode.",
                )
            if cleaned_ref_text is not None:
                raise HTTPException(
                    status_code=400,
                    detail="ref_text is only supported for clone mode.",
                )

        if mode == GenerateMode.design:
            if cleaned_instruct is None:
                raise HTTPException(
                    status_code=400,
                    detail="instruct is required for design mode.",
                )
            if ref_audio is not None:
                raise HTTPException(
                    status_code=400,
                    detail="ref_audio is only supported for clone mode.",
                )
            if cleaned_ref_text is not None:
                raise HTTPException(
                    status_code=400,
                    detail="ref_text is only supported for clone mode.",
                )

        if mode == GenerateMode.clone and ref_audio is None:
            raise HTTPException(
                status_code=400,
                detail="ref_audio is required for clone mode.",
            )
        if (
            mode == GenerateMode.clone
            and _native_coreml_runtime_enabled(app.state.model)
            and cleaned_ref_text is None
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Clone mode with native Core ML runtime requires ref_text. "
                    "Whisper auto-transcription is not part of the Apple-native path."
                ),
            )

        generation_config = OmniVoiceGenerationConfig(
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        kwargs = {
            "text": cleaned_text,
            "language": cleaned_language,
            "generation_config": generation_config,
        }
        if cleaned_instruct is not None:
            kwargs["instruct"] = cleaned_instruct
        if speed is not None and float(speed) != 1.0:
            kwargs["speed"] = float(speed)
        if duration is not None:
            if float(duration) <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="duration must be greater than 0 when provided.",
                )
            kwargs["duration"] = float(duration)

        temp_audio_path = None
        try:
            if ref_audio is not None:
                temp_audio_path = _save_upload_to_tempfile(ref_audio)
                kwargs["ref_audio"] = temp_audio_path
                if cleaned_ref_text is not None:
                    kwargs["ref_text"] = cleaned_ref_text

            with app.state.generate_lock:
                audios = app.state.model.generate(**kwargs)

            audio_duration = audios[0].shape[-1] / app.state.model.sampling_rate
            wav_bytes = _audio_to_wav_bytes(audios[0], app.state.model.sampling_rate)
            latency_ms = (time.perf_counter() - request_started) * 1000.0
            rtf = (
                (latency_ms / 1000.0) / audio_duration
                if audio_duration and audio_duration > 0
                else None
            )

            headers = {
                "Content-Disposition": 'inline; filename="omnivoice.wav"',
                "X-OmniVoice-Request-Id": request_id,
                "X-OmniVoice-Started-At": started_at,
                "X-OmniVoice-Finished-At": _iso_utc_now(),
                "X-OmniVoice-Latency-Ms": f"{latency_ms:.2f}",
                "X-OmniVoice-Audio-Duration-S": f"{audio_duration:.3f}",
            }
            if rtf is not None:
                headers["X-OmniVoice-RTF"] = f"{rtf:.4f}"
            if app.state.save_dir is not None:
                saved_path = _persist_wav_bytes(
                    wav_bytes=wav_bytes,
                    save_dir=app.state.save_dir,
                    mode=mode,
                    text=cleaned_text,
                )
                headers["X-OmniVoice-Saved-Path"] = str(saved_path)

            logger.info(
                "request_id=%s status=success mode=%s started_at=%s finished_at=%s latency_ms=%.2f audio_s=%.3f rtf=%s text_chars=%d has_ref_audio=%s language=%s device=%s saved_path=%s",
                request_id,
                mode.value,
                started_at,
                headers["X-OmniVoice-Finished-At"],
                latency_ms,
                audio_duration,
                f"{rtf:.4f}" if rtf is not None else "n/a",
                len(cleaned_text),
                ref_audio is not None,
                cleaned_language or "auto",
                app.state.device,
                str(saved_path) if saved_path else "-",
            )

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers=headers,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            logger.warning(
                "request_id=%s status=invalid_request started_at=%s error=%s",
                request_id,
                started_at,
                exc,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("request_id=%s status=runtime_error", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("request_id=%s status=failed", request_id)
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if ref_audio is not None:
                ref_audio.file.close()
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            if latency_ms is None:
                final_latency_ms = (time.perf_counter() - request_started) * 1000.0
                logger.info(
                    "request_id=%s status=completed_without_audio mode=%s started_at=%s finished_at=%s latency_ms=%.2f text_chars=%d has_ref_audio=%s language=%s device=%s",
                    request_id,
                    mode.value,
                    started_at,
                    _iso_utc_now(),
                    final_latency_ms,
                    len(cleaned_text),
                    ref_audio is not None,
                    cleaned_language or "auto",
                    app.state.device,
                )

    return app


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = os.environ.get("OMNIVOICE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    app = create_app(
        model_checkpoint=args.model,
        device=args.device,
        load_asr=not args.no_asr,
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
    )

    uvicorn.run(
        app,
        host=args.ip,
        port=args.port,
        root_path=args.root_path or "",
    )


if __name__ == "__main__":
    main()
