#!/usr/bin/env python3
"""Export multiple native Core ML backbone bucket artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omnivoice.scripts.export_coreml_backbone import export_coreml_backbone


def _parse_seq_lens(raw: str) -> list[int]:
    seq_lens = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        seq_lens.append(int(item))
    if not seq_lens:
        raise argparse.ArgumentTypeError("At least one sequence length is required.")
    return sorted(set(seq_lens))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export multiple native Core ML backbone bucket artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where bucket artifacts will be written.",
    )
    parser.add_argument(
        "--seq-lens",
        type=_parse_seq_lens,
        required=True,
        help="Comma-separated list of static sequence lengths, for example 64,128,256.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Fixed batch size used during export.",
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
        help="Minimum deployment target for the exported Core ML models.",
    )
    parser.add_argument(
        "--dynamic-fallback-max-seq-len",
        type=int,
        default=None,
        help="Optional dynamic fallback max sequence length to export alongside the static buckets.",
    )
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_len in args.seq_lens:
        bucket_dir = output_dir / f"seq{seq_len}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        output_path = bucket_dir / f"omnivoice_backbone_bs{args.batch_size}_seq{seq_len}.mlpackage"
        export_coreml_backbone(
            args.model,
            output_path,
            seq_len=seq_len,
            batch_size=args.batch_size,
            dynamic_seq=False,
            precision=args.precision,
            target=args.target,
        )

    if args.dynamic_fallback_max_seq_len is not None:
        dynamic_dir = output_dir / "dynamic"
        dynamic_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            dynamic_dir
            / f"omnivoice_backbone_bs{args.batch_size}_maxseq{args.dynamic_fallback_max_seq_len}_dynamic.mlpackage"
        )
        export_coreml_backbone(
            args.model,
            output_path,
            seq_len=args.dynamic_fallback_max_seq_len,
            batch_size=args.batch_size,
            dynamic_seq=True,
            precision=args.precision,
            target=args.target,
        )

    logging.info("Backbone bucket export complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
