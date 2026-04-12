#!/usr/bin/env python3
"""Benchmark OmniVoice runtime backends with consistent prompts."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch

from omnivoice import OmniVoice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OmniVoice runtime backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device for non-CoreML pieces.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Prompt text to synthesize.",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Voice design instruction.",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Reference audio path for clone mode.",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Reference transcript for clone mode.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=16,
        help="Generation steps.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before measurement.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured runs.",
    )
    parser.add_argument(
        "--coreml-backbone",
        default=None,
        help="Optional native Core ML backbone .mlpackage.",
    )
    parser.add_argument(
        "--coreml-compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="cpu_and_ne",
        help="Compute units for native Core ML backbone.",
    )
    parser.add_argument(
        "--coreml-backbone-allow-fixed-padding",
        action="store_true",
        default=False,
        help="Allow fixed-shape native Core ML backbone buckets to pad shorter requests up to the exported seq-len.",
    )
    parser.add_argument(
        "--coreml-decoder",
        default=None,
        help="Optional native Core ML decoder .mlpackage.",
    )
    parser.add_argument(
        "--coreml-decoder-compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="all",
        help="Compute units for native Core ML decoder.",
    )
    parser.add_argument(
        "--onnx-backbone",
        default=None,
        help="Optional ONNX backbone path.",
    )
    parser.add_argument(
        "--onnx-provider",
        choices=["auto", "cpu", "coreml"],
        default="auto",
        help="Provider for ONNX backbone.",
    )
    parser.add_argument(
        "--onnx-decoder",
        default=None,
        help="Optional ONNX decoder path.",
    )
    parser.add_argument(
        "--onnx-decoder-provider",
        choices=["auto", "cpu", "coreml"],
        default="auto",
        help="Provider for ONNX decoder.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON results.",
    )
    return parser


def _load_model(args) -> OmniVoice:
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float16 if args.device != "cpu" else torch.float32,
        load_asr=False,
    )
    if args.coreml_backbone:
        model.load_coreml_backbone(
            args.coreml_backbone,
            compute_units=args.coreml_compute_units,
            allow_fixed_shape_padding=args.coreml_backbone_allow_fixed_padding,
        )
    if args.coreml_decoder:
        model.load_coreml_decoder(
            args.coreml_decoder,
            compute_units=args.coreml_decoder_compute_units,
        )
    if args.onnx_backbone:
        model.load_onnx_backbone(
            args.onnx_backbone,
            provider=args.onnx_provider,
        )
    if args.onnx_decoder:
        model.load_onnx_decoder(
            args.onnx_decoder,
            provider=args.onnx_decoder_provider,
        )
    return model


def _run_once(model: OmniVoice, args) -> dict:
    generate_kwargs = {
        "text": args.text,
        "instruct": args.instruct,
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
        "num_step": args.num_step,
    }
    start = time.perf_counter()
    audios = model.generate(**generate_kwargs)
    wall_sec = time.perf_counter() - start
    audio = audios[0]
    audio_sec = float(audio.shape[-1]) / float(model.sampling_rate)
    profile = dict(model._last_generation_profile or {})
    return {
        "wall_sec": wall_sec,
        "audio_sec": audio_sec,
        "rtf": wall_sec / audio_sec if audio_sec > 0 else None,
        "profile": profile,
    }


def _summarize(runs: list[dict]) -> dict:
    wall_secs = [run["wall_sec"] for run in runs]
    audio_secs = [run["audio_sec"] for run in runs]
    rtfs = [run["rtf"] for run in runs if run["rtf"] is not None]
    last_profile = runs[-1]["profile"] if runs else {}
    return {
        "repeats": len(runs),
        "wall_sec": {
            "mean": statistics.mean(wall_secs),
            "min": min(wall_secs),
            "max": max(wall_secs),
        },
        "audio_sec": {
            "mean": statistics.mean(audio_secs),
            "min": min(audio_secs),
            "max": max(audio_secs),
        },
        "rtf": {
            "mean": statistics.mean(rtfs),
            "min": min(rtfs),
            "max": max(rtfs),
        }
        if rtfs
        else None,
        "last_profile": last_profile,
    }


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    if (args.coreml_backbone or args.coreml_decoder) and args.ref_audio and not args.ref_text:
        raise SystemExit(
            "Clone mode with native Core ML runtime requires --ref-text."
        )

    model = _load_model(args)

    for _ in range(args.warmup):
        _run_once(model, args)

    runs = [_run_once(model, args) for _ in range(args.repeats)]
    summary = {
        "model": args.model,
        "device": args.device,
        "text": args.text,
        "instruct": args.instruct,
        "num_step": args.num_step,
        "coreml_backbone": args.coreml_backbone,
        "coreml_compute_units": args.coreml_compute_units if args.coreml_backbone else None,
        "coreml_backbone_allow_fixed_padding": (
            args.coreml_backbone_allow_fixed_padding if args.coreml_backbone else None
        ),
        "coreml_decoder": args.coreml_decoder,
        "coreml_decoder_compute_units": (
            args.coreml_decoder_compute_units if args.coreml_decoder else None
        ),
        "onnx_backbone": args.onnx_backbone,
        "onnx_provider": args.onnx_provider if args.onnx_backbone else None,
        "onnx_decoder": args.onnx_decoder,
        "onnx_decoder_provider": (
            args.onnx_decoder_provider if args.onnx_decoder else None
        ),
        "summary": _summarize(runs),
        "runs": runs,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
