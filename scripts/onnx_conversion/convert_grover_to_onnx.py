#!/usr/bin/env python3
"""
Convert GROVER (PoetschLab/GROVER) model to ONNX format for fastembed-bio.

Usage:
    python scripts/convert_grover_to_onnx.py \
        --model PoetschLab/GROVER \
        --output ./converted_models/grover-onnx

Requirements:
    pip install torch transformers onnx onnxruntime
"""
import argparse
import json

from pathlib import Path

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel


def convert_to_onnx(
    model_name: str,
    output_dir: str,
    opset_version: int = 18,
    max_length: int = 512,
) -> Path:
    """
    Convert GROVER model to ONNX format.

    Args:
        model_name: HuggingFace model identifier (e.g., PoetschLab/GROVER)
        output_dir: Directory to save ONNX model and tokenizer files
        opset_version: ONNX opset version (default 14)
        max_length: Maximum sequence length for dummy input

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # create dummy input
    dummy_sequence = "ATCG" * (max_length // 4)
    print(f"Creating dummy input with sequence length: {len(dummy_sequence)}")

    batch = tokenizer(
        [dummy_sequence],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    print(f"Input shapes - input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}")

    # export to ONNX
    print("Exporting to ONNX...")
    onnx_path = output_path / "model.onnx"

    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model,
            (batch["input_ids"], batch["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    print(f"ONNX model saved to: {onnx_path}")

    # save tokenizer files
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_path)

    # save model config info
    model_config = {
        "hidden_size": model.config.hidden_size,
        "model_type": "grover",
        "source_model": model_name,
        "output_type": "embeddings_only",
    }
    model_config_path = output_path / "model_config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Model config saved to: {model_config_path}")

    return output_path


def validate_onnx_model(
    model_name: str,
    onnx_path: str,
):
    """
    Validate the ONNX model by checking its structure and running a test inference.

    Args:
        model_name: HuggingFace model identifier
        onnx_path: Path to the ONNX model file
    """
    import onnxruntime as ort

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    test_seq = 'ATCGATCGATCGATCG' * 8
    batch = tokenizer([test_seq], return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        pt_out = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_dict=True,
        )

    onnx_out = session.run(None, {
        'input_ids': batch['input_ids'].numpy().astype(np.int64),
        'attention_mask': batch['attention_mask'].numpy().astype(np.int64),
    })[0]

    pt_emb = pt_out.last_hidden_state.numpy()
    print(f'PyTorch: {pt_emb.shape}, ONNX: {onnx_out.shape}')
    print(f'Max diff: {np.abs(pt_emb - onnx_out).max():.2e}')
    print('✓ Match!' if np.allclose(pt_emb, onnx_out, atol=1e-3) else '✗ Mismatch')


def main():
    parser = argparse.ArgumentParser(
        description="Convert GROVER model to ONNX format for fastembed-bio"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PoetschLab/GROVER",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./converted_models/grover-onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for export",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the exported ONNX model",
    )

    args = parser.parse_args()
    output_path = convert_to_onnx(args.model, args.output, args.opset, args.max_length)

    if args.validate:
        validate_onnx_model(args.model, str(output_path / "model.onnx"))


if __name__ == "__main__":
    main()