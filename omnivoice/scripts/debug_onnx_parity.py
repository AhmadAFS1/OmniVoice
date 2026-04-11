#!/usr/bin/env python3
"""Compare OmniVoice backbone outputs between PyTorch and ONNX Runtime."""

from __future__ import annotations

import argparse

import torch

from omnivoice import OmniVoice
from omnivoice.models.onnx_backbone import OmniVoiceBackboneForOnnx
from omnivoice.utils.onnx_runtime import OnnxBackboneSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX backbone parity for a single prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--onnx-backbone", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--provider", choices=["auto", "cpu", "coreml"], default="cpu")
    parser.add_argument(
        "--allow-fixed-padding",
        action="store_true",
        default=False,
        help="Allow fixed-shape ONNX exports to pad shorter requests up to the exported seq-len.",
    )
    return parser


def _build_batch(model: OmniVoice, text: str, instruct: str | None):
    target_len = model._estimate_target_tokens(text, None, 0)
    prepared = model._prepare_inference_inputs(
        text=text,
        num_target_tokens=target_len,
        ref_text=None,
        ref_audio_tokens=None,
        lang=None,
        instruct=instruct,
        denoise=False,
        device=torch.device("cpu"),
    )

    c_len = prepared["input_ids"].size(2)
    u_len = target_len
    pad_id = model.config.audio_mask_id

    batch_input_ids = torch.full(
        (2, model.config.num_audio_codebook, c_len),
        pad_id,
        dtype=torch.long,
    )
    batch_audio_mask = torch.zeros((2, c_len), dtype=torch.bool)
    batch_attention_mask = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool)

    input_ids = prepared["input_ids"]
    audio_mask = prepared["audio_mask"]

    batch_input_ids[0, :, :c_len] = input_ids
    batch_audio_mask[0, :c_len] = audio_mask
    batch_attention_mask[0, :, :c_len, :c_len] = True

    batch_input_ids[1, :, :u_len] = input_ids[..., -u_len:]
    batch_audio_mask[1, :u_len] = audio_mask[..., -u_len:]
    batch_attention_mask[1, :, :u_len, :u_len] = True
    if c_len > u_len:
        pad_diag = torch.arange(u_len, c_len)
        batch_attention_mask[1, :, pad_diag, pad_diag] = True

    return batch_input_ids, batch_audio_mask, batch_attention_mask


def _pad_batch(
    batch_input_ids: torch.Tensor,
    batch_audio_mask: torch.Tensor,
    batch_attention_mask: torch.Tensor,
    seq_len: int,
):
    current_seq_len = batch_input_ids.shape[-1]
    padded_input_ids = torch.full(
        (batch_input_ids.shape[0], batch_input_ids.shape[1], seq_len),
        0,
        dtype=batch_input_ids.dtype,
    )
    padded_input_ids[:, :, :current_seq_len] = batch_input_ids

    padded_audio_mask = torch.zeros(
        (batch_audio_mask.shape[0], seq_len),
        dtype=batch_audio_mask.dtype,
    )
    padded_audio_mask[:, :current_seq_len] = batch_audio_mask

    padded_attention_mask = torch.zeros(
        (batch_attention_mask.shape[0], 1, seq_len, seq_len),
        dtype=batch_attention_mask.dtype,
    )
    padded_attention_mask[:, :, :current_seq_len, :current_seq_len] = batch_attention_mask
    pad_diag = torch.arange(current_seq_len, seq_len)
    padded_attention_mask[:, :, pad_diag, pad_diag] = True

    return padded_input_ids, padded_audio_mask, padded_attention_mask


def _report(name: str, ref: torch.Tensor, other: torch.Tensor) -> None:
    diff = (ref - other).abs()
    arg_ref = ref.argmax(dim=-1)
    arg_other = other.argmax(dim=-1)
    mismatch_rate = (arg_ref != arg_other).float().mean().item()
    print(name)
    print("  max_abs:", float(diff.max()))
    print("  mean_abs:", float(diff.mean()))
    print("  argmax_mismatch_rate:", mismatch_rate)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    model = OmniVoice.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
        attn_implementation="eager",
    )
    wrapper = OmniVoiceBackboneForOnnx(model).eval()
    ort = OnnxBackboneSession.create(
        args.onnx_backbone,
        provider=args.provider,
        allow_fixed_shape_padding=args.allow_fixed_padding,
    )

    batch_input_ids, batch_audio_mask, batch_attention_mask = _build_batch(
        model,
        text=args.text,
        instruct=args.instruct,
    )
    current_seq_len = batch_input_ids.shape[-1]

    with torch.inference_mode():
        logits_unpadded = wrapper(
            batch_input_ids,
            batch_audio_mask,
            batch_attention_mask,
        ).to(torch.float32)

    if ort.fixed_seq_len is not None and ort.fixed_seq_len > current_seq_len:
        padded_inputs = _pad_batch(
            batch_input_ids,
            batch_audio_mask,
            batch_attention_mask,
            seq_len=ort.fixed_seq_len,
        )
        with torch.inference_mode():
            logits_padded = wrapper(*padded_inputs).to(torch.float32)[
                :, :, :current_seq_len, :
            ]
    else:
        logits_padded = logits_unpadded

    logits_ort = ort.run(
        batch_input_ids,
        batch_audio_mask,
        batch_attention_mask,
    ).to(torch.float32)

    _report("pt_unpadded_vs_pt_padded", logits_unpadded, logits_padded)
    _report("pt_padded_vs_ort", logits_padded, logits_ort)
    _report("pt_unpadded_vs_ort", logits_unpadded, logits_ort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
