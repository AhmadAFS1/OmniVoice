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


def _provider_loaded(
    requested: ProviderPreference,
    actual_providers: list[str],
) -> bool:
    if requested == "coreml":
        return "CoreMLExecutionProvider" in actual_providers
    if requested == "cpu":
        return "CPUExecutionProvider" in actual_providers
    return True


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


def _normalize_onnx_paths(onnx_paths: str | list[str]) -> list[str]:
    raw_items = [onnx_paths] if isinstance(onnx_paths, str) else list(onnx_paths)
    resolved_paths: list[str] = []

    for raw_item in raw_items:
        for item in str(raw_item).split(","):
            item = item.strip()
            if not item:
                continue

            path = Path(item).expanduser()
            if path.is_dir():
                onnx_files = sorted(str(p.resolve()) for p in path.rglob("*.onnx"))
                if not onnx_files:
                    raise FileNotFoundError(
                        f"No .onnx files were found under directory: {path}"
                    )
                resolved_paths.extend(onnx_files)
            else:
                resolved_paths.append(str(path.resolve()))

    deduped: list[str] = []
    seen: set[str] = set()
    for path in resolved_paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)

    if not deduped:
        raise ValueError("No ONNX backbone paths were provided.")

    return deduped


@dataclass
class OnnxBackboneSession:
    path: str
    session: Any
    providers: list[str]
    fixed_batch_size: int | None
    fixed_seq_len: int | None
    allow_fixed_shape_padding: bool
    input_names: dict[str, str]
    output_name: str

    @classmethod
    def create(
        cls,
        onnx_path: str,
        provider: ProviderPreference = "auto",
        allow_fixed_shape_padding: bool = False,
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
        actual_providers = session.get_providers()
        if not _provider_loaded(provider, actual_providers):
            raise RuntimeError(
                f"Requested provider={provider} for ONNX backbone {onnx_path}, "
                f"but ONNX Runtime loaded providers={actual_providers}."
            )

        inputs = {i.name: i.shape for i in session.get_inputs()}
        output_name = session.get_outputs()[0].name

        batch_shape = inputs["input_ids"][0]
        seq_shape = inputs["input_ids"][2]

        return cls(
            path=onnx_path,
            session=session,
            providers=actual_providers,
            fixed_batch_size=_maybe_int_dim(batch_shape),
            fixed_seq_len=_maybe_int_dim(seq_shape),
            allow_fixed_shape_padding=allow_fixed_shape_padding,
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
        if self.fixed_seq_len is not None:
            if seq_len == self.fixed_seq_len:
                return True
            if self.allow_fixed_shape_padding and seq_len < self.fixed_seq_len:
                return True
            return False
        return True

    def _pad_inputs_to_fixed_seq_len(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_seq_len = self.fixed_seq_len
        current_seq_len = int(input_ids.shape[-1])
        if target_seq_len is None or current_seq_len >= target_seq_len:
            return input_ids, audio_mask, attention_mask

        padded_input_ids = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], target_seq_len),
            dtype=input_ids.dtype,
        )
        padded_input_ids[:, :, :current_seq_len] = input_ids

        padded_audio_mask = torch.zeros(
            (audio_mask.shape[0], target_seq_len),
            dtype=audio_mask.dtype,
        )
        padded_audio_mask[:, :current_seq_len] = audio_mask

        padded_attention_mask = torch.zeros(
            (attention_mask.shape[0], 1, target_seq_len, target_seq_len),
            dtype=attention_mask.dtype,
        )
        padded_attention_mask[:, :, :current_seq_len, :current_seq_len] = attention_mask
        pad_diag = torch.arange(current_seq_len, target_seq_len)
        padded_attention_mask[:, :, pad_diag, pad_diag] = True

        return padded_input_ids, padded_audio_mask, padded_attention_mask

    def run(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.device.type != "cpu":
            raise RuntimeError("ONNX Runtime backbone expects CPU tensors as inputs.")

        batch_size = int(input_ids.shape[0])
        original_seq_len = int(input_ids.shape[-1])
        seq_len = original_seq_len
        if not self.supports(batch_size=batch_size, seq_len=seq_len):
            raise RuntimeError(
                f"ONNX backbone export at {self.path} does not support batch_size={batch_size}, "
                f"seq_len={seq_len}. Fixed batch={self.fixed_batch_size}, "
                f"fixed seq_len={self.fixed_seq_len}."
            )

        if (
            self.fixed_seq_len is not None
            and self.allow_fixed_shape_padding
            and original_seq_len < self.fixed_seq_len
        ):
            input_ids, audio_mask, attention_mask = self._pad_inputs_to_fixed_seq_len(
                input_ids=input_ids,
                audio_mask=audio_mask,
                attention_mask=attention_mask,
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
        if outputs.shape[2] != original_seq_len:
            outputs = outputs[:, :, :original_seq_len, :]
        return torch.from_numpy(outputs.astype(np.float32, copy=False))


@dataclass
class OnnxBackboneRouter:
    paths: list[str]
    sessions: list[OnnxBackboneSession]
    providers: list[str]
    fixed_batch_size: int | None
    fixed_seq_len: int | None
    allow_fixed_shape_padding: bool
    last_selected_path: str | None = None
    last_selected_providers: list[str] | None = None
    last_selected_fixed_seq_len: int | None = None

    @classmethod
    def create(
        cls,
        onnx_paths: str | list[str],
        provider: ProviderPreference = "auto",
        allow_fixed_shape_padding: bool = False,
    ) -> "OnnxBackboneRouter":
        paths = _normalize_onnx_paths(onnx_paths)
        sessions: list[OnnxBackboneSession] = []
        load_errors: list[str] = []
        for path in paths:
            try:
                sessions.append(
                    OnnxBackboneSession.create(
                        path,
                        provider=provider,
                        allow_fixed_shape_padding=allow_fixed_shape_padding,
                    )
                )
            except Exception as exc:
                load_errors.append(f"{path}: {exc}")
                logger.warning(
                    "Skipping ONNX backbone session %s because it did not initialize with provider=%s: %s",
                    path,
                    provider,
                    exc,
                )

        if not sessions:
            details = "; ".join(load_errors) if load_errors else "no sessions loaded"
            raise RuntimeError(
                f"Failed to load any ONNX backbone sessions for provider={provider}. {details}"
            )
        sessions.sort(
            key=lambda session: (
                session.fixed_seq_len is None,
                session.fixed_seq_len if session.fixed_seq_len is not None else float("inf"),
                session.path,
            )
        )

        providers: list[str] = []
        for session in sessions:
            for provider_name in session.providers:
                if provider_name not in providers:
                    providers.append(provider_name)

        fixed_batch_sizes = {s.fixed_batch_size for s in sessions}
        fixed_batch_size = (
            next(iter(fixed_batch_sizes)) if len(fixed_batch_sizes) == 1 else None
        )

        return cls(
            paths=paths,
            sessions=sessions,
            providers=providers,
            fixed_batch_size=fixed_batch_size,
            fixed_seq_len=None,
            allow_fixed_shape_padding=allow_fixed_shape_padding,
        )

    def describe_sessions(self) -> list[dict[str, Any]]:
        descriptions = []
        for session in self.sessions:
            descriptions.append(
                {
                    "path": session.path,
                    "fixed_batch_size": session.fixed_batch_size,
                    "fixed_seq_len": session.fixed_seq_len,
                    "providers": session.providers,
                    "allow_fixed_shape_padding": session.allow_fixed_shape_padding,
                }
            )
        return descriptions

    def select_session(
        self,
        batch_size: int,
        seq_len: int,
    ) -> OnnxBackboneSession | None:
        best_fixed: OnnxBackboneSession | None = None
        dynamic_fallback: OnnxBackboneSession | None = None

        for session in self.sessions:
            if not session.supports(batch_size=batch_size, seq_len=seq_len):
                continue
            if session.fixed_seq_len is None:
                dynamic_fallback = session
                continue
            if best_fixed is None or session.fixed_seq_len < best_fixed.fixed_seq_len:
                best_fixed = session

        return best_fixed or dynamic_fallback

    def supports(self, batch_size: int, seq_len: int) -> bool:
        return self.select_session(batch_size=batch_size, seq_len=seq_len) is not None

    def run(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[-1])
        session = self.select_session(batch_size=batch_size, seq_len=seq_len)
        if session is None:
            available = ", ".join(
                "dynamic"
                if s.fixed_seq_len is None
                else f"seq{s.fixed_seq_len}"
                for s in self.sessions
            )
            raise RuntimeError(
                f"No loaded ONNX backbone session supports batch_size={batch_size}, seq_len={seq_len}. "
                f"Loaded sessions: {available or 'none'}."
            )
        self.last_selected_path = session.path
        self.last_selected_providers = list(session.providers)
        self.last_selected_fixed_seq_len = session.fixed_seq_len
        return session.run(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
        )


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
        actual_providers = session.get_providers()
        if not _provider_loaded(provider, actual_providers):
            raise RuntimeError(
                f"Requested provider={provider} for ONNX decoder {onnx_path}, "
                f"but ONNX Runtime loaded providers={actual_providers}."
            )

        inputs = {i.name: i.shape for i in session.get_inputs()}
        input_shape = inputs["audio_codes"]
        output_name = session.get_outputs()[0].name

        return cls(
            path=onnx_path,
            session=session,
            providers=actual_providers,
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
