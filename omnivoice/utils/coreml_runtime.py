#!/usr/bin/env python3
"""Helpers for running OmniVoice backbone inference with native Core ML."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

ComputeUnitsPreference = Literal["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"]


def _maybe_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _metadata_path(model_path: str) -> Path:
    return Path(model_path).with_suffix(".metadata.json")


def _normalize_coreml_paths(model_paths: str | list[str]) -> list[str]:
    raw_items = [model_paths] if isinstance(model_paths, str) else list(model_paths)
    resolved_paths: list[str] = []

    for raw_item in raw_items:
        for item in str(raw_item).split(","):
            item = item.strip()
            if not item:
                continue

            path = Path(item).expanduser()
            if path.is_dir() and path.suffix != ".mlpackage":
                coreml_files = sorted(str(p.resolve()) for p in path.rglob("*.mlpackage"))
                if not coreml_files:
                    raise FileNotFoundError(
                        f"No .mlpackage files were found under directory: {path}"
                    )
                resolved_paths.extend(coreml_files)
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
        raise ValueError("No Core ML backbone paths were provided.")

    return deduped


def _resolve_compute_units(ct_module, preference: ComputeUnitsPreference):
    mapping = {
        "all": ct_module.ComputeUnit.ALL,
        "cpu_only": ct_module.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct_module.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct_module.ComputeUnit.CPU_AND_NE,
    }
    return mapping[preference]


@dataclass
class CoreMLBackboneSession:
    path: str
    model: Any
    compute_units: str
    fixed_batch_size: int | None
    fixed_seq_len: int | None
    max_seq_len: int | None
    dynamic_seq: bool
    output_name: str

    @classmethod
    def create(
        cls,
        model_path: str,
        compute_units: ComputeUnitsPreference = "all",
    ) -> "CoreMLBackboneSession":
        if sys.platform != "darwin":
            raise RuntimeError(
                "Native Core ML inference is only available on macOS."
            )

        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError(
                "coremltools is not installed. Install it with "
                "`uv sync --extra coreml` before enabling native Core ML."
            ) from exc

        metadata = {}
        metadata_file = _metadata_path(model_path)
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text())

        model = ct.models.MLModel(
            model_path,
            compute_units=_resolve_compute_units(ct, compute_units),
        )
        output_name = metadata.get("output_name") or model.get_spec().description.output[0].name

        return cls(
            path=str(Path(model_path).resolve()),
            model=model,
            compute_units=compute_units,
            fixed_batch_size=_maybe_int(metadata.get("fixed_batch_size")),
            fixed_seq_len=_maybe_int(metadata.get("fixed_seq_len")),
            max_seq_len=_maybe_int(metadata.get("max_seq_len")),
            dynamic_seq=bool(metadata.get("dynamic_seq", False)),
            output_name=output_name,
        )

    def supports(self, batch_size: int, seq_len: int) -> bool:
        if self.fixed_batch_size is not None and batch_size != self.fixed_batch_size:
            return False
        if self.fixed_seq_len is not None:
            return seq_len == self.fixed_seq_len
        if self.dynamic_seq and self.max_seq_len is not None:
            return seq_len <= self.max_seq_len
        return True

    def run(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.device.type != "cpu":
            raise RuntimeError("Native Core ML backbone expects CPU tensors as inputs.")

        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[-1])
        if not self.supports(batch_size=batch_size, seq_len=seq_len):
            raise RuntimeError(
                f"Core ML backbone at {self.path} does not support batch_size={batch_size}, "
                f"seq_len={seq_len}. Fixed batch={self.fixed_batch_size}, "
                f"fixed seq_len={self.fixed_seq_len}, max_seq_len={self.max_seq_len}, "
                f"dynamic_seq={self.dynamic_seq}."
            )

        outputs = self.model.predict(
            {
                "input_ids": np.ascontiguousarray(
                    input_ids.numpy().astype(np.int32, copy=False)
                ),
                "audio_mask": np.ascontiguousarray(
                    audio_mask.numpy().astype(np.int32, copy=False)
                ),
                "attention_mask": np.ascontiguousarray(
                    attention_mask.numpy().astype(np.int32, copy=False)
                ),
            }
        )
        audio_logits = np.asarray(outputs[self.output_name])
        return torch.from_numpy(audio_logits.astype(np.float32, copy=False))


@dataclass
class CoreMLBackboneRouter:
    paths: list[str]
    sessions: list[CoreMLBackboneSession]
    compute_units: str
    last_selected_path: str | None = None
    last_selected_fixed_seq_len: int | None = None
    last_selected_compute_units: str | None = None

    @classmethod
    def create(
        cls,
        model_paths: str | list[str],
        compute_units: ComputeUnitsPreference = "all",
    ) -> "CoreMLBackboneRouter":
        paths = _normalize_coreml_paths(model_paths)
        sessions: list[CoreMLBackboneSession] = []
        load_errors: list[str] = []
        for path in paths:
            try:
                sessions.append(
                    CoreMLBackboneSession.create(path, compute_units=compute_units)
                )
            except Exception as exc:
                load_errors.append(f"{path}: {exc}")
                logger.warning(
                    "Skipping native Core ML backbone session %s because it failed to initialize: %s",
                    path,
                    exc,
                )

        if not sessions:
            details = "; ".join(load_errors) if load_errors else "no sessions loaded"
            raise RuntimeError(
                f"Failed to load any native Core ML backbone sessions. {details}"
            )
        sessions.sort(
            key=lambda session: (
                session.fixed_seq_len is None,
                session.fixed_seq_len if session.fixed_seq_len is not None else float("inf"),
                session.path,
            )
        )
        return cls(paths=paths, sessions=sessions, compute_units=compute_units)

    def describe_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "path": session.path,
                "compute_units": session.compute_units,
                "fixed_batch_size": session.fixed_batch_size,
                "fixed_seq_len": session.fixed_seq_len,
                "max_seq_len": session.max_seq_len,
                "dynamic_seq": session.dynamic_seq,
            }
            for session in self.sessions
        ]

    def select_session(self, batch_size: int, seq_len: int) -> CoreMLBackboneSession | None:
        best_fixed: CoreMLBackboneSession | None = None
        dynamic_fallback: CoreMLBackboneSession | None = None

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
                f"No loaded Core ML backbone session supports batch_size={batch_size}, "
                f"seq_len={seq_len}. Loaded sessions: {available or 'none'}."
            )
        self.last_selected_path = session.path
        self.last_selected_fixed_seq_len = session.fixed_seq_len
        self.last_selected_compute_units = session.compute_units
        return session.run(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
        )


@dataclass
class CoreMLAudioDecoderSession:
    path: str
    model: Any
    compute_units: str
    fixed_batch_size: int | None
    fixed_num_codebooks: int | None
    fixed_seq_len: int | None
    max_seq_len: int | None
    dynamic_seq: bool
    output_name: str

    @classmethod
    def create(
        cls,
        model_path: str,
        compute_units: ComputeUnitsPreference = "all",
    ) -> "CoreMLAudioDecoderSession":
        if sys.platform != "darwin":
            raise RuntimeError("Native Core ML inference is only available on macOS.")

        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError(
                "coremltools is not installed. Install it with "
                "`uv sync --extra coreml` before enabling native Core ML."
            ) from exc

        metadata = {}
        metadata_file = _metadata_path(model_path)
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text())

        model = ct.models.MLModel(
            model_path,
            compute_units=_resolve_compute_units(ct, compute_units),
        )
        output_name = metadata.get("output_name") or model.get_spec().description.output[0].name

        return cls(
            path=str(Path(model_path).resolve()),
            model=model,
            compute_units=compute_units,
            fixed_batch_size=_maybe_int(metadata.get("fixed_batch_size")),
            fixed_num_codebooks=_maybe_int(metadata.get("fixed_num_codebooks")),
            fixed_seq_len=_maybe_int(metadata.get("fixed_seq_len")),
            max_seq_len=_maybe_int(metadata.get("max_seq_len")),
            dynamic_seq=bool(metadata.get("dynamic_seq", False)),
            output_name=output_name,
        )

    def supports(self, batch_size: int, num_codebooks: int, seq_len: int) -> bool:
        if self.fixed_batch_size is not None and batch_size != self.fixed_batch_size:
            return False
        if (
            self.fixed_num_codebooks is not None
            and num_codebooks != self.fixed_num_codebooks
        ):
            return False
        if self.fixed_seq_len is not None:
            return seq_len == self.fixed_seq_len
        if self.dynamic_seq and self.max_seq_len is not None:
            return seq_len <= self.max_seq_len
        return True

    def run(self, audio_codes: torch.Tensor) -> torch.Tensor:
        if audio_codes.device.type != "cpu":
            raise RuntimeError("Native Core ML decoder expects CPU tensors as inputs.")

        batch_size = int(audio_codes.shape[0])
        num_codebooks = int(audio_codes.shape[1])
        seq_len = int(audio_codes.shape[-1])
        if not self.supports(
            batch_size=batch_size,
            num_codebooks=num_codebooks,
            seq_len=seq_len,
        ):
            raise RuntimeError(
                f"Core ML decoder at {self.path} does not support "
                f"batch_size={batch_size}, num_codebooks={num_codebooks}, seq_len={seq_len}. "
                f"Fixed batch={self.fixed_batch_size}, fixed_num_codebooks={self.fixed_num_codebooks}, "
                f"fixed seq_len={self.fixed_seq_len}, max_seq_len={self.max_seq_len}, "
                f"dynamic_seq={self.dynamic_seq}."
            )

        outputs = self.model.predict(
            {
                "audio_codes": np.ascontiguousarray(
                    audio_codes.numpy().astype(np.int32, copy=False)
                ),
            }
        )
        audio_values = np.asarray(outputs[self.output_name])
        return torch.from_numpy(audio_values.astype(np.float32, copy=False))


@dataclass
class CoreMLAudioDecoderRouter:
    paths: list[str]
    sessions: list[CoreMLAudioDecoderSession]
    compute_units: str
    last_selected_path: str | None = None
    last_selected_fixed_seq_len: int | None = None
    last_selected_compute_units: str | None = None

    @classmethod
    def create(
        cls,
        model_paths: str | list[str],
        compute_units: ComputeUnitsPreference = "all",
    ) -> "CoreMLAudioDecoderRouter":
        paths = _normalize_coreml_paths(model_paths)
        sessions: list[CoreMLAudioDecoderSession] = []
        load_errors: list[str] = []
        for path in paths:
            try:
                sessions.append(
                    CoreMLAudioDecoderSession.create(path, compute_units=compute_units)
                )
            except Exception as exc:
                load_errors.append(f"{path}: {exc}")
                logger.warning(
                    "Skipping native Core ML decoder session %s because it failed to initialize: %s",
                    path,
                    exc,
                )

        if not sessions:
            details = "; ".join(load_errors) if load_errors else "no sessions loaded"
            raise RuntimeError(
                f"Failed to load any native Core ML decoder sessions. {details}"
            )
        sessions.sort(
            key=lambda session: (
                session.fixed_seq_len is None,
                session.fixed_seq_len if session.fixed_seq_len is not None else float("inf"),
                session.path,
            )
        )
        return cls(paths=paths, sessions=sessions, compute_units=compute_units)

    def describe_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "path": session.path,
                "compute_units": session.compute_units,
                "fixed_batch_size": session.fixed_batch_size,
                "fixed_num_codebooks": session.fixed_num_codebooks,
                "fixed_seq_len": session.fixed_seq_len,
                "max_seq_len": session.max_seq_len,
                "dynamic_seq": session.dynamic_seq,
            }
            for session in self.sessions
        ]

    def select_session(
        self,
        batch_size: int,
        num_codebooks: int,
        seq_len: int,
    ) -> CoreMLAudioDecoderSession | None:
        best_fixed: CoreMLAudioDecoderSession | None = None
        dynamic_fallback: CoreMLAudioDecoderSession | None = None

        for session in self.sessions:
            if not session.supports(
                batch_size=batch_size,
                num_codebooks=num_codebooks,
                seq_len=seq_len,
            ):
                continue
            if session.fixed_seq_len is None:
                dynamic_fallback = session
                continue
            if best_fixed is None or session.fixed_seq_len < best_fixed.fixed_seq_len:
                best_fixed = session

        return best_fixed or dynamic_fallback

    def supports(self, batch_size: int, num_codebooks: int, seq_len: int) -> bool:
        return (
            self.select_session(
                batch_size=batch_size,
                num_codebooks=num_codebooks,
                seq_len=seq_len,
            )
            is not None
        )

    def run(self, audio_codes: torch.Tensor) -> torch.Tensor:
        batch_size = int(audio_codes.shape[0])
        num_codebooks = int(audio_codes.shape[1])
        seq_len = int(audio_codes.shape[-1])
        session = self.select_session(
            batch_size=batch_size,
            num_codebooks=num_codebooks,
            seq_len=seq_len,
        )
        if session is None:
            available = ", ".join(
                "dynamic"
                if s.fixed_seq_len is None
                else f"seq{s.fixed_seq_len}"
                for s in self.sessions
            )
            raise RuntimeError(
                f"No loaded Core ML decoder session supports batch_size={batch_size}, "
                f"num_codebooks={num_codebooks}, seq_len={seq_len}. "
                f"Loaded sessions: {available or 'none'}."
            )
        self.last_selected_path = session.path
        self.last_selected_fixed_seq_len = session.fixed_seq_len
        self.last_selected_compute_units = session.compute_units
        return session.run(audio_codes=audio_codes)
