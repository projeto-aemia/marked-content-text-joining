"""
ONNX and Client-Side Web Export
Exports trained models (GBDT and Transformer Cross-Encoder) to ONNX format
and applies INT8 dynamic quantization for ultra-fast, low-memory browser execution.
"""

import os
import sys
import json
import argparse
from typing import Optional
import torch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.cross_encoder import CrossEncoderClassifier
from src.features.feature_builder import FEATURE_NAMES
from src.export.export_json import export_gbdt_to_clean_json


def export_gbdt_to_web(
    gbdt_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    name: str = "gbdt_universal"
) -> str:
    """
    Exports LightGBM model to a clean client-side JSON evaluator format.
    Can be run in any browser with ZERO external dependencies in <0.01ms.
    """
    if gbdt_path is None:
        gbdt_path = os.path.join(PROJECT_ROOT, "saved_models", "multilingual", "gbdt_universal.pkl")
        if not os.path.exists(gbdt_path):
            gbdt_path = os.path.join(PROJECT_ROOT, "saved_models", "gbdt_model.pkl")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "src", "export", "web_demo")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "gbdt_model.json")

    export_gbdt_to_clean_json(
        model_input=gbdt_path,
        output_path=json_path,
        name=name,
        minify=True
    )

    print(f"GBDT tree dump saved to {json_path} ({os.path.getsize(json_path) / 1024:.1f} KB)")
    return json_path


def export_cross_encoder_to_onnx(
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    quantize_int8: bool = True
) -> str:
    """
    Exports a Transformer Cross-Encoder to ONNX format and quantizes to INT8.
    """
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if model_name is None:
        # Default to local fine-tuned universal encoder if available, else HuggingFace checkpoint
        local_univ = os.path.join(PROJECT_ROOT, "saved_models", "multilingual", "cross_encoder_universal")
        if os.path.exists(local_univ):
            model_name = local_univ
        else:
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "src", "export", "web_demo")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading Cross-Encoder from: {model_name}...")
    clf = CrossEncoderClassifier(model_name=model_name, device="cpu")
    model = clf.model
    tokenizer = clf.tokenizer

    # Save tokenizer files for client-side loading
    tokenizer.save_pretrained(output_dir)

    model.eval()

    dummy_input = tokenizer(
        "Perspectivas de gênero e sexualidade: uma análise",
        "qualitativa de evento científico",
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True
    )

    input_ids = dummy_input['input_ids']
    attention_mask = dummy_input['attention_mask']

    onnx_fp32_path = os.path.join(output_dir, "model_fp32.onnx")
    onnx_int8_path = os.path.join(output_dir, "model_quantized.onnx")

    print(f"Exporting FP32 ONNX model to {onnx_fp32_path}...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False
    )

    fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    print(f"FP32 ONNX export complete! Size: {fp32_size:.2f} MB")

    if quantize_int8:
        print(f"Applying dynamic INT8 quantization to {onnx_int8_path}...")
        quantize_dynamic(
            model_input=onnx_fp32_path,
            model_output=onnx_int8_path,
            weight_type=QuantType.QInt8,
            extra_options={'DisableShapeInference': True}
        )
        int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
        print(f"INT8 ONNX model created! Size: {int8_size:.2f} MB (Compression: {(1 - int8_size/fp32_size)*100:.1f}%)")

    # Verify ONNX model with onnxruntime
    target_onnx = onnx_int8_path if quantize_int8 else onnx_fp32_path
    session = ort.InferenceSession(target_onnx)
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy()
    }
    ort_outs = session.run(None, ort_inputs)
    probs = torch.softmax(torch.tensor(ort_outs[0]), dim=-1).numpy()
    print(f"ONNX Verification successful! Sample output probabilities: {probs[0]}")

    return target_onnx


def main():
    parser = argparse.ArgumentParser(
        description="Export GBDT and Transformer Cross-Encoder models to ONNX and web demo formats."
    )
    parser.add_argument("--gbdt-only", action="store_true", help="Only export GBDT model to JSON")
    parser.add_argument("--onnx-only", action="store_true", help="Only export Transformer to ONNX")
    parser.add_argument("--gbdt-path", type=str, default=None, help="Path to input GBDT .pkl file")
    parser.add_argument("--encoder-path", type=str, default=None, help="Path/name of input Cross-Encoder")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save exported web models")
    parser.add_argument("--no-quantize", action="store_true", help="Disable dynamic INT8 ONNX quantization")

    args = parser.parse_args()

    if args.gbdt_only:
        export_gbdt_to_web(gbdt_path=args.gbdt_path, output_dir=args.output_dir)
    elif args.onnx_only:
        export_cross_encoder_to_onnx(
            model_name=args.encoder_path,
            output_dir=args.output_dir,
            quantize_int8=not args.no_quantize
        )
    else:
        export_gbdt_to_web(gbdt_path=args.gbdt_path, output_dir=args.output_dir)
        export_cross_encoder_to_onnx(
            model_name=args.encoder_path,
            output_dir=args.output_dir,
            quantize_int8=not args.no_quantize
        )


if __name__ == '__main__':
    main()
