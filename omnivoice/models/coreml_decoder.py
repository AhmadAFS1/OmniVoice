#!/usr/bin/env python3
"""Core ML export wrapper for the OmniVoice audio decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import HiggsAudioV2TokenizerModel


class OmniVoiceAudioDecoderForCoreML(nn.Module):
    """Standalone decoder wrapper with Core ML-friendly input dtypes.

    Input:
      audio_codes: [B, C, T] int32

    Output:
      audio_values: [B, 1, S] float32
    """

    def __init__(self, audio_tokenizer: HiggsAudioV2TokenizerModel):
        super().__init__()
        self.audio_tokenizer = audio_tokenizer

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        return self.audio_tokenizer.decode(
            audio_codes.to(dtype=torch.long)
        ).audio_values
