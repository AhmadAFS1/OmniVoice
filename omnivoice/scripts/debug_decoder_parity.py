#!/usr/bin/env python3
"""Compare PyTorch and ONNX Runtime decoder outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.onnx_runtime import OnnxAudioDecoderSession

logger = logging.getLogger(__name__)


def _summarize(pt_audio: torch.Tensor, ort_audio: torch.Tensor) -> dict[str, float | int]:
    pt_cpu = pt_audio.detach().cpu().to(torch.float32)
    ort_cpu = ort_audio.detach().cpu().to(torch.float32)

    min_len = min(pt_cpu.shape[-1], ort_cpu.shape[-1])
    pt_trim = pt_cpu[..., :min_len]
    ort_trim = ort_cpu[..., :min_len]
    diff = (pt_trim - ort_trim).abs()

    return {
        "pt_num_samples": int(pt_cpu.shape[-1]),
        "ort_num_samples": int(ort_cpu.shape[-1]),
        "shared_num_samples": int(min_len),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _save_wav(audio: torch.Tensor, output_path: Path, sampling_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        audio.squeeze(0).detach().cpu().numpy(),
        sampling_rate,
        format="WAV",
        subtype="PCM_16",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare decoder parity between PyTorch and ONNX Runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--onnx-decoder", required=True)
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "coreml"],
        default="auto",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--text",
        default="This is a short test of local text to speech.",
    )
    parser.add_argument(
        "--instruct",
        default="female, american accent",
    )
    parser.add_argument("--num-step", type=int, default=4)
    parser.add_argument(
        "--save-prefix",
        default=None,
        help="Optional prefix for saving generated-token decode WAVs.",
    )
    return parser


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)

    logger.info("Loading OmniVoice from %s on CPU ...", args.model)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
        attn_implementation="eager",
    )
    ort_decoder = OnnxAudioDecoderSession.create(
        args.onnx_decoder,
        provider=args.provider,
    )
    codebook_size = int(model.audio_tokenizer.quantizer.quantizers[0].codebook.codebook_size)

    random_codes = torch.randint(
        0,
        codebook_size,
        (1, model.config.num_audio_codebook, args.seq_len),
        dtype=torch.long,
    )

    pt_random = model.audio_tokenizer.decode(random_codes).audio_values
    ort_random = ort_decoder.run(random_codes)

    logger.info("Preparing deterministic generated-token sample ...")
    task = model._preprocess_all(
        text=args.text,
        instruct=args.instruct,
    )
    gen_config = OmniVoiceGenerationConfig(
        num_step=args.num_step,
        denoise=False,
        postprocess_output=False,
        position_temperature=0.0,
        class_temperature=0.0,
    )
    generated_tokens = model._generate_iterative(task, gen_config)[0].unsqueeze(0).cpu()
    pt_generated = model.audio_tokenizer.decode(generated_tokens).audio_values
    ort_generated = ort_decoder.run(generated_tokens)

    if args.save_prefix:
        save_prefix = Path(args.save_prefix)
        _save_wav(
            pt_generated[0],
            save_prefix.with_name(f"{save_prefix.name}_pt.wav"),
            model.sampling_rate,
        )
        _save_wav(
            ort_generated[0],
            save_prefix.with_name(f"{save_prefix.name}_ort.wav"),
            model.sampling_rate,
        )

    payload = {
        "provider": args.provider,
        "loaded_runtime_providers": ort_decoder.providers,
        "random_codes": _summarize(pt_random, ort_random),
        "generated_codes": _summarize(pt_generated, ort_generated),
        "generated_token_seq_len": int(generated_tokens.shape[-1]),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
