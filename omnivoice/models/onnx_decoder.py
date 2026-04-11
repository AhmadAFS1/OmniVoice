#!/usr/bin/env python3
"""ONNX export wrapper for the OmniVoice audio decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import HiggsAudioV2TokenizerModel


class OmniVoiceAudioDecoderForOnnx(nn.Module):
    """Standalone decoder wrapper around the Higgs audio tokenizer model.

    Input:
      audio_codes: [B, C, T]

    Output:
      audio_values: [B, 1, S]
    """

    def __init__(self, audio_tokenizer: HiggsAudioV2TokenizerModel):
        super().__init__()
        self.audio_tokenizer = audio_tokenizer

    def forward(self, audio_codes: torch.LongTensor) -> torch.Tensor:
        return self.audio_tokenizer.decode(audio_codes).audio_values
