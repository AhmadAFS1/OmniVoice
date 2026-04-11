"""Export the audio decoder to ONNX, then convert to CoreML."""

import torch
import coremltools as ct
from transformers import HiggsAudioV2TokenizerModel


def export_decoder():
    tokenizer_model = HiggsAudioV2TokenizerModel.from_pretrained(
        "eustlb/higgs-audio-v2-tokenizer",
        device_map="cpu",
    )

    # The decoder takes audio_codes of shape (1, num_codebooks, seq_len)
    # and returns audio_values of shape (1, 1, num_samples)
    num_codebooks = 8
    seq_len = 100  # representative

    dummy_codes = torch.randint(0, 1024, (1, num_codebooks, seq_len))

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio_codes):
            return self.model.decode(audio_codes).audio_values

    wrapper = DecoderWrapper(tokenizer_model)

    torch.onnx.export(
        wrapper,
        (dummy_codes,),
        "artifacts/onnx/omnivoice_decoder.onnx",
        input_names=["audio_codes"],
        output_names=["audio"],
        dynamic_axes={
            "audio_codes": {2: "seq_len"},
            "audio": {2: "num_samples"},
        },
        opset_version=17,
    )

    # Then convert to CoreML
    mlmodel = ct.converters.convert(
        "artifacts/onnx/omnivoice_decoder.onnx",
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save("artifacts/coreml/OmniVoiceDecoder.mlpackage")


if __name__ == "__main__":
    export_decoder()