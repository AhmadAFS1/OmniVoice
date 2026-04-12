#!/usr/bin/env python3
"""Export the OmniVoice backbone to a native Core ML model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from omnivoice import OmniVoice
from omnivoice.models.coreml_backbone import OmniVoiceBackboneForCoreML


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the OmniVoice backbone to a native Core ML model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .mlpackage path.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Representative or fixed sequence length used during export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Fixed batch size used during export. Use 2 for single-item CFG generation.",
    )
    parser.add_argument(
        "--dynamic-seq",
        action="store_true",
        help="Export with a flexible sequence dimension up to --seq-len.",
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Core ML compute precision.",
    )
    parser.add_argument(
        "--target",
        choices=["macos13", "ios17"],
        default="macos13",
        help="Minimum deployment target for the exported Core ML model.",
    )
    return parser


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".metadata.json")


def _resolve_precision(ct_module, precision_name: str):
    if precision_name == "float32":
        return ct_module.precision.FLOAT32
    return ct_module.precision.FLOAT16


def _resolve_target(ct_module, target_name: str):
    if target_name == "ios17":
        return ct_module.target.iOS17
    return ct_module.target.macOS13


def _build_input_types(
    ct_module,
    batch_size: int,
    num_codebooks: int,
    seq_len: int,
    dynamic_seq: bool,
):
    seq_dim = ct_module.RangeDim(1, seq_len) if dynamic_seq else seq_len
    return [
        ct_module.TensorType(
            name="input_ids",
            shape=(batch_size, num_codebooks, seq_dim),
            dtype=np.int32,
        ),
        ct_module.TensorType(
            name="audio_mask",
            shape=(batch_size, seq_dim),
            dtype=np.int32,
        ),
        ct_module.TensorType(
            name="attention_mask",
            shape=(batch_size, 1, seq_dim, seq_dim),
            dtype=np.int32,
        ),
    ]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(
            "coremltools is not installed. Install it with `uv sync --extra coreml` "
            "before exporting a native Core ML backbone."
        ) from exc

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = _metadata_path(output_path)

    logging.info("Loading OmniVoice from %s on CPU for Core ML export ...", args.model)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
        attn_implementation="eager",
    )
    backbone = OmniVoiceBackboneForCoreML(model).eval()

    batch_size = args.batch_size
    seq_len = args.seq_len
    num_codebooks = model.config.num_audio_codebook

    example_inputs = (
        torch.zeros((batch_size, num_codebooks, seq_len), dtype=torch.int32),
        torch.zeros((batch_size, seq_len), dtype=torch.int32),
        torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.int32),
    )
    dynamic_shapes = None
    if args.dynamic_seq:
        seq_dim = torch.export.Dim("seq_len", min=1, max=seq_len)
        dynamic_shapes = (
            {2: seq_dim},
            {1: seq_dim},
            {2: seq_dim, 3: seq_dim},
        )
    exported_program = torch.export.export(
        backbone,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    ).run_decompositions({})

    logging.info(
        "Converting backbone to Core ML at %s (batch_size=%s, seq_len=%s, dynamic_seq=%s, precision=%s, target=%s) ...",
        output_path,
        batch_size,
        seq_len,
        args.dynamic_seq,
        args.precision,
        args.target,
    )

    mlmodel = ct.convert(
        exported_program,
        convert_to="mlprogram",
        inputs=_build_input_types(
            ct_module=ct,
            batch_size=batch_size,
            num_codebooks=num_codebooks,
            seq_len=seq_len,
            dynamic_seq=args.dynamic_seq,
        ),
        compute_precision=_resolve_precision(ct, args.precision),
        minimum_deployment_target=_resolve_target(ct, args.target),
    )
    output_name = mlmodel.get_spec().description.output[0].name
    mlmodel.save(str(output_path))

    metadata = {
        "path": str(output_path),
        "kind": "backbone",
        "backend": "coreml",
        "fixed_batch_size": batch_size,
        "fixed_seq_len": None if args.dynamic_seq else seq_len,
        "max_seq_len": seq_len,
        "dynamic_seq": bool(args.dynamic_seq),
        "num_audio_codebook": num_codebooks,
        "input_dtypes": {
            "input_ids": "int32",
            "audio_mask": "int32",
            "attention_mask": "int32",
        },
        "output_name": output_name,
        "precision": args.precision,
        "target": args.target,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logging.info("Core ML backbone export complete: %s", output_path)
    logging.info("Core ML backbone metadata written to: %s", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
