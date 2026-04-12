#!/usr/bin/env python3
"""Core ML export wrapper for the OmniVoice diffusion backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from omnivoice.models.omnivoice import OmniVoice


class OmniVoiceBackboneForCoreML(nn.Module):
    """Standalone backbone wrapper with Core ML-friendly input dtypes.

    Inputs:
      input_ids: [B, C, S] int32
      audio_mask: [B, S] int32
      attention_mask: [B, 1, S, S] int32

    Output:
      audio_logits: [B, C, S, V] float32
    """

    def __init__(self, omnivoice_model: OmniVoice):
        super().__init__()

        cfg = omnivoice_model.config
        self.text_embeddings = omnivoice_model.get_input_embeddings()
        self.audio_embeddings = omnivoice_model.audio_embeddings
        self.llm = omnivoice_model.llm
        self.audio_heads = omnivoice_model.audio_heads

        self.num_audio_codebook = cfg.num_audio_codebook
        self.audio_vocab_size = cfg.audio_vocab_size

        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(cfg.num_audio_codebook).view(1, -1, 1)
            * cfg.audio_vocab_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(dtype=torch.long)
        audio_mask = audio_mask.to(dtype=torch.bool)
        attention_mask = attention_mask.to(dtype=torch.bool)

        text_embeds = self.text_embeddings(input_ids[:, 0, :])
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1).to(dtype=input_ids.dtype)
        ) + self.codebook_layer_offsets
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        inputs_embeds = torch.where(
            audio_mask.unsqueeze(-1),
            audio_embeds,
            text_embeds,
        )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = llm_outputs[0]

        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)
        audio_logits = logits_flat.view(
            batch_size,
            seq_len,
            self.num_audio_codebook,
            self.audio_vocab_size,
        ).permute(0, 2, 1, 3)

        return audio_logits
