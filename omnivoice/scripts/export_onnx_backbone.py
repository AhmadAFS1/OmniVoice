#!/usr/bin/env python3
"""Export the OmniVoice backbone to ONNX for phase-1 ORT benchmarking."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from omnivoice import OmniVoice
from omnivoice.models.onnx_backbone import OmniVoiceBackboneForOnnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the OmniVoice backbone to ONNX.",
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
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Fixed sequence length to export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Fixed batch size to export. Use 2 for single-item CFG generation.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dynamic-seq",
        action="store_true",
        help="Export with dynamic sequence axes instead of a fixed seq-len.",
    )
    return parser


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading OmniVoice from %s on CPU for export ...", args.model)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
        attn_implementation="eager",
    )
    backbone = OmniVoiceBackboneForOnnx(model).eval()

    batch_size = args.batch_size
    seq_len = args.seq_len
    input_ids = torch.zeros(
        (batch_size, model.config.num_audio_codebook, seq_len),
        dtype=torch.long,
    )
    audio_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)

    logging.info(
        "Exporting backbone to %s (batch_size=%s, seq_len=%s, dynamic_seq=%s) ...",
        output_path,
        batch_size,
        seq_len,
        args.dynamic_seq,
    )

    dynamic_axes = None
    if args.dynamic_seq:
        dynamic_axes = {
            "input_ids": {2: "seq_len"},
            "audio_mask": {1: "seq_len"},
            "attention_mask": {2: "seq_len", 3: "seq_len"},
            "logits": {2: "seq_len"},
        }

    torch.onnx.export(
        backbone,
        (input_ids, audio_mask, attention_mask),
        str(output_path),
        input_names=["input_ids", "audio_mask", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=True,
    )

    logging.info("Backbone export complete: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
