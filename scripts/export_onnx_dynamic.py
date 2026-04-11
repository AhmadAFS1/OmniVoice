"""Export OmniVoice backbone to ONNX with dynamic sequence length."""

import os
import torch
from omnivoice.models.omnivoice import OmniVoice


def export_dynamic(
    model_name: str = "k2-fsa/OmniVoice",
    output_path: str = "artifacts/onnx/omnivoice_backbone_dynamic.onnx",
    opset_version: int = 17,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = OmniVoice.from_pretrained(model_name, device_map="cpu", no_asr=True)
    model.eval()

    # Use a representative shape for tracing, but mark seq as dynamic
    batch_size = 2
    num_codebooks = model.config.num_audio_codebook
    seq_len = 128  # representative, not fixed

    dummy_input_ids = torch.randint(
        0, model.config.audio_vocab_size, (batch_size, num_codebooks, seq_len)
    )
    dummy_audio_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    dummy_audio_mask[:, -32:] = True
    dummy_attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)

    dynamic_axes = {
        "input_ids": {0: "batch", 2: "seq_len"},
        "audio_mask": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 2: "seq_len", 3: "seq_len"},
        "logits": {0: "batch", 2: "seq_len"},
    }

    class BackboneWrapper(torch.nn.Module):
        def __init__(self, omnivoice_model):
            super().__init__()
            self.model = omnivoice_model

        def forward(self, input_ids, audio_mask, attention_mask):
            out = self.model(
                input_ids=input_ids,
                audio_mask=audio_mask,
                attention_mask=attention_mask,
            )
            return out.logits.to(torch.float32)

    wrapper = BackboneWrapper(model)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_audio_mask, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "audio_mask", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Exported dynamic ONNX backbone to {output_path}")


if __name__ == "__main__":
    export_dynamic()