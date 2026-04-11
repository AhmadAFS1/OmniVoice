"""Convert ONNX backbone to CoreML for iOS deployment."""

import coremltools as ct
import numpy as np


def convert_to_coreml(
    onnx_path: str = "artifacts/onnx/omnivoice_backbone_dynamic.onnx",
    output_path: str = "artifacts/coreml/OmniVoiceBackbone.mlpackage",
    compute_precision: ct.precision = ct.precision.FLOAT16,
):
    # Convert from ONNX to CoreML
    mlmodel = ct.converters.convert(
        onnx_path,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.save(output_path)
    print(f"Saved CoreML model to {output_path}")


if __name__ == "__main__":
    convert_to_coreml()