#!/usr/bin/env python3
"""
Convert NTv3 (Nucleotide Transformer v3) models to ONNX format for fastembed-bio.

Usage:
    python scripts/convert_ntv3_to_onnx.py \
        --model InstaDeepAI/NTv3_650M_post \
        --output ./converted_models/ntv3-650m-onnx

Requirements:
    pip install torch transformers onnx onnxruntime
"""
import argparse
import json

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoTokenizer, AutoModel


class NTv3EmbeddingWrapper(nn.Module):
    """
    A simplified wrapper around NTv3 that only outputs embeddings.

    This avoids the data-dependent control flow in the full model's forward pass
    (asserts on species_ids values, conditional head routing) which can't be traced
    for ONNX export.
    """

    def __init__(self, model):
        super().__init__()
        self.core = model.core
        self.config = model.config
        # get the parent class that has the actual transformer forward method
        self._parent_class = self.core.__class__.__bases__[0]

    def forward(
        self,
        input_ids: torch.LongTensor,
        species_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward pass that returns only the embedding.

        Args:
            input_ids: Input token IDs of shape (B, L)
            species_ids: Species token IDs of shape (B,)

        Returns:
            Embedding tensor of shape (B, L', H) where L' may be cropped from L
        """
        # call the parent class forward directly, bypassing asserts -- since these break tracing (symbolic shape)
        # discreteConditionedNTv3PreTrainedCore.forward expects condition_ids as a list
        outs = self._parent_class.forward(
            self.core,
            input_ids=input_ids,
            condition_ids=[species_ids],
            conditions_masks=None,
            inputs_embeds=None,
            output_hidden_states=False,
            output_attentions=False,
        )

        # return the full embedding (uncropped), matching the original model's behavior
        # the _crop_to_center is only used internally for the bigwig/bed heads
        return outs["embedding"]

def convert_to_onnx(
    model_name: str,
    output_dir: str,
    opset_version: int = 18,
    max_length: int = 512,
) -> Path:
    """
    Convert NTv3 model to ONNX format.

    Args:
        model_name: HuggingFace model identifier (e.g., InstaDeepAI/NTv3_650M_post)
        output_dir: Directory to save ONNX model and tokenizer files
        opset_version: ONNX opset version (default 18)
        max_length: Maximum sequence length for dummy input (must be multiple of 128)

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ensure max_length is multiple of 128 (NTv3 requirement)
    max_length = ((max_length + 127) // 128) * 128

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # get supported species
    print("Extracting species information...")
    species_list = model.supported_species
    species_to_id = {species: idx for idx, species in enumerate(species_list)}
    print(f"Supported species ({len(species_list)}): {species_list}")

    # create dummy input (length must be multiple of 128)
    dummy_sequence = "A" * max_length
    print(f"Creating dummy input with sequence length: {max_length}")

    batch = tokenizer(
        [dummy_sequence],
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=128,
        add_special_tokens=False,
    )

    # get species IDs for dummy input
    species_ids = model.encode_species(["human"])

    print(f"Input shapes - input_ids: {batch['input_ids'].shape}, species_ids: {species_ids.shape}")

    # export to ONNX
    print("Exporting to ONNX...")
    onnx_path = output_path / "model.onnx"

    # Create a simplified wrapper that only outputs embeddings
    # This avoids the data-dependent control flow that can't be traced
    wrapper = NTv3EmbeddingWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (batch["input_ids"], species_ids),
            onnx_path,
            input_names=["input_ids", "species_ids"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "species_ids": {0: "batch_size"},
                "embeddings": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            report=True,
        )

    print(f"ONNX model saved to: {onnx_path}")

    # save tokenizer files
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_path)

    # save species mapping
    species_config = {
        "species_to_id": species_to_id,
        "id_to_species": {v: k for k, v in species_to_id.items()},
        "supported_species": species_list,
    }
    species_config_path = output_path / "species_config.json"
    with open(species_config_path, "w") as f:
        json.dump(species_config, f, indent=2)
    print(f"Species config saved to: {species_config_path}")

    # save model config info
    model_config = {
        "hidden_size": model.config.hidden_size if hasattr(model.config, "hidden_size") else 1536,
        "model_type": "ntv3",
        "source_model": model_name,
        "output_type": "embeddings_only",
    }
    model_config_path = output_path / "model_config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Model config saved to: {model_config_path}")

def validate_onnx_model(
    model_name: str,
    onnx_path: str,
):
    """
    Validate the ONNX model by checking its structure and running a test inference.

    Args:
        onnx_path: Path to the ONNX model file
        model_name: HuggingFace model identifier
    """
    import onnxruntime as ort

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    test_seq = 'ATCGATCGATCGATCG' * 8
    batch = tokenizer([test_seq], return_tensors='pt', padding=True, pad_to_multiple_of=128, add_special_tokens=False)
    species_ids = model.encode_species(['human'])

    with torch.no_grad():
        pt_out = model(input_ids=batch['input_ids'], species_ids=species_ids)

    onnx_out = session.run(None, {
        'input_ids': batch['input_ids'].numpy().astype(np.int64),
        'species_ids': species_ids.numpy().astype(np.int64),
    })[0]

    pt_emb = pt_out.embedding.numpy()
    print(f'PyTorch: {pt_emb.shape}, ONNX: {onnx_out.shape}')
    print(f'Max diff: {np.abs(pt_emb - onnx_out).max():.2e}')
    print('✓ Match!' if np.allclose(pt_emb, onnx_out, atol=1e-3) else '✗ Mismatch')

def main():
    parser = argparse.ArgumentParser(
        description="Convert NTv3 models to ONNX format for fastembed-bio"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="InstaDeepAI/NTv3_650M_post",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./converted_models/ntv3-650m-onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for export (will be rounded to multiple of 128)",
    )

    args = parser.parse_args()
    convert_to_onnx(args.model, args.output, args.opset, args.max_length)

if __name__ == "__main__":
    main()


