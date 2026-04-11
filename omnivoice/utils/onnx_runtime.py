#!/usr/bin/env python3
"""Helpers for running OmniVoice backbone inference with ONNX Runtime."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

ProviderPreference = Literal["auto", "cpu", "coreml"]


def _maybe_int_dim(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _build_provider_list(ort_module, preference: ProviderPreference) -> list[Any]:
    available = set(ort_module.get_available_providers())
    providers: list[Any] = []

    if preference in ("auto", "coreml") and "CoreMLExecutionProvider" in available:
        providers.append(
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                },
            )
        )

    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    if not providers:
        raise RuntimeError(
            "No supported ONNX Runtime providers are available. "
            f"Available providers: {sorted(available)}"
        )

    return providers


def _build_coreml_provider_options(
    onnx_path: str,
    model_kind: Literal["backbone", "decoder"],
) -> dict[str, str]:
    onnx_file = Path(onnx_path).resolve()
    cache_dir = onnx_file.parent / f"{onnx_file.stem}.coreml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "ModelFormat": "NeuralNetwork" if model_kind == "backbone" else "MLProgram",
        "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1",
        "EnableOnSubgraphs": "0",
        "ModelCacheDirectory": str(cache_dir),
    }

    if os.environ.get("OMNIVOICE_COREML_PROFILE_PLAN") == "1":
        options["ProfileComputePlan"] = "1"

    return options


@dataclass
class OnnxBackboneSession:
    path: str
    session: Any
    providers: list[str]
    fixed_batch_size: int | None
    fixed_seq_len: int | None
    input_names: dict[str, str]
    output_name: str

    @classmethod
    def create(
        cls,
        onnx_path: str,
        provider: ProviderPreference = "auto",
    ) -> "OnnxBackboneSession":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with "
                "`uv sync --extra onnx` before enabling the ONNX backend."
            ) from exc

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = _build_provider_list(ort, provider)
        if providers and isinstance(providers[0], tuple):
            name, options = providers[0]
            if name == "CoreMLExecutionProvider":
                providers[0] = (
                    name,
                    {**options, **_build_coreml_provider_options(onnx_path, "backbone")},
                )
        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers,
        )

        inputs = {i.name: i.shape for i in session.get_inputs()}
        output_name = session.get_outputs()[0].name

        batch_shape = inputs["input_ids"][0]
        seq_shape = inputs["input_ids"][2]

        return cls(
            path=onnx_path,
            session=session,
            providers=session.get_providers(),
            fixed_batch_size=_maybe_int_dim(batch_shape),
            fixed_seq_len=_maybe_int_dim(seq_shape),
            input_names={
                "input_ids": "input_ids",
                "audio_mask": "audio_mask",
                "attention_mask": "attention_mask",
            },
            output_name=output_name,
        )

    def supports(self, batch_size: int, seq_len: int) -> bool:
        if self.fixed_batch_size is not None and batch_size != self.fixed_batch_size:
            return False
        if self.fixed_seq_len is not None and seq_len != self.fixed_seq_len:
            return False
        return True

    def run(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.device.type != "cpu":
            raise RuntimeError("ONNX Runtime backbone expects CPU tensors as inputs.")

        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[-1])
        if not self.supports(batch_size=batch_size, seq_len=seq_len):
            raise RuntimeError(
                f"ONNX backbone export at {self.path} does not support batch_size={batch_size}, "
                f"seq_len={seq_len}. Fixed batch={self.fixed_batch_size}, "
                f"fixed seq_len={self.fixed_seq_len}."
            )

        ort_inputs = {
            self.input_names["input_ids"]: np.ascontiguousarray(
                input_ids.numpy().astype(np.int64, copy=False)
            ),
            self.input_names["audio_mask"]: np.ascontiguousarray(
                audio_mask.numpy().astype(np.bool_, copy=False)
            ),
            self.input_names["attention_mask"]: np.ascontiguousarray(
                attention_mask.numpy().astype(np.bool_, copy=False)
            ),
        }

        outputs = self.session.run([self.output_name], ort_inputs)[0]
        return torch.from_numpy(outputs.astype(np.float32, copy=False))


@dataclass
class OnnxAudioDecoderSession:
    path: str
    session: Any
    providers: list[str]
    fixed_batch_size: int | None
    fixed_num_codebooks: int | None
    fixed_seq_len: int | None
    input_name: str
    output_name: str

    @classmethod
    def create(
        cls,
        onnx_path: str,
        provider: ProviderPreference = "auto",
    ) -> "OnnxAudioDecoderSession":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with "
                "`uv sync --extra onnx` before enabling the ONNX backend."
            ) from exc

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        providers = _build_provider_list(ort, provider)
        if providers and isinstance(providers[0], tuple):
            name, options = providers[0]
            if name == "CoreMLExecutionProvider":
                providers[0] = (
                    name,
                    {**options, **_build_coreml_provider_options(onnx_path, "decoder")},
                )

        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers,
        )

        inputs = {i.name: i.shape for i in session.get_inputs()}
        input_shape = inputs["audio_codes"]
        output_name = session.get_outputs()[0].name

        return cls(
            path=onnx_path,
            session=session,
            providers=session.get_providers(),
            fixed_batch_size=_maybe_int_dim(input_shape[0]),
            fixed_num_codebooks=_maybe_int_dim(input_shape[1]),
            fixed_seq_len=_maybe_int_dim(input_shape[2]),
            input_name="audio_codes",
            output_name=output_name,
        )

    def supports(
        self,
        batch_size: int,
        num_codebooks: int,
        seq_len: int,
    ) -> bool:
        if self.fixed_batch_size is not None and batch_size != self.fixed_batch_size:
            return False
        if (
            self.fixed_num_codebooks is not None
            and num_codebooks != self.fixed_num_codebooks
        ):
            return False
        if self.fixed_seq_len is not None and seq_len != self.fixed_seq_len:
            return False
        return True

    def run(self, audio_codes: torch.Tensor) -> torch.Tensor:
        if audio_codes.device.type != "cpu":
            raise RuntimeError("ONNX Runtime decoder expects CPU tensors as inputs.")

        if audio_codes.dim() != 3:
            raise RuntimeError(
                f"ONNX Runtime decoder expects audio_codes with shape [B, C, T], got {tuple(audio_codes.shape)}."
            )

        batch_size, num_codebooks, seq_len = map(int, audio_codes.shape)
        if not self.supports(
            batch_size=batch_size,
            num_codebooks=num_codebooks,
            seq_len=seq_len,
        ):
            raise RuntimeError(
                f"ONNX decoder export at {self.path} does not support batch_size={batch_size}, "
                f"num_codebooks={num_codebooks}, seq_len={seq_len}. Fixed batch={self.fixed_batch_size}, "
                f"fixed num_codebooks={self.fixed_num_codebooks}, fixed seq_len={self.fixed_seq_len}."
            )

        ort_inputs = {
            self.input_name: np.ascontiguousarray(
                audio_codes.numpy().astype(np.int64, copy=False)
            )
        }
        outputs = self.session.run([self.output_name], ort_inputs)[0]
        return torch.from_numpy(outputs.astype(np.float32, copy=False))
