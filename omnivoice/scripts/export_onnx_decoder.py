#!/usr/bin/env python3
"""Export the OmniVoice audio decoder to ONNX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from omnivoice import OmniVoice
from omnivoice.models.onnx_decoder import OmniVoiceAudioDecoderForOnnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the OmniVoice audio decoder to ONNX.",
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
        default=128,
        help="Representative audio-token sequence length to export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Fixed batch size to export.",
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
        help="Export with a dynamic sequence axis instead of a fixed seq-len.",
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

    logging.info("Loading OmniVoice from %s on CPU for decoder export ...", args.model)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
        attn_implementation="eager",
    )
    decoder = OmniVoiceAudioDecoderForOnnx(model.audio_tokenizer).eval()

    batch_size = args.batch_size
    num_codebooks = model.config.num_audio_codebook
    seq_len = args.seq_len
    audio_codes = torch.zeros(
        (batch_size, num_codebooks, seq_len),
        dtype=torch.long,
    )

    logging.info(
        "Exporting decoder to %s (batch_size=%s, seq_len=%s, dynamic_seq=%s) ...",
        output_path,
        batch_size,
        seq_len,
        args.dynamic_seq,
    )

    dynamic_axes = None
    if args.dynamic_seq:
        dynamic_axes = {
            "audio_codes": {2: "seq_len"},
            "audio_values": {2: "num_samples"},
        }

    torch.onnx.export(
        decoder,
        (audio_codes,),
        str(output_path),
        input_names=["audio_codes"],
        output_names=["audio_values"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=True,
    )

    logging.info("Decoder export complete: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
