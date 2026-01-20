# fastembed-bio

Fast, lightweight biological sequence embeddings using ONNX. Built on [FastEmbed](https://github.com/qdrant/fastembed).

## Why fastembed-bio?

1. **Light**: No GPU required. No PyTorch. Just ONNX Runtime. Perfect for serverless and resource-constrained environments.

2. **Fast**: ONNX Runtime is faster than PyTorch inference. Batch processing and parallelism built-in.

3. **Simple**: Same interface patterns as FastEmbed. If you've used FastEmbed for text, you already know how to use this.

## Installation

```bash
pip install fastembed-bio
```

## Quickstart

```python
from fastembed.bio import ProteinEmbedding

sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAK",
]

model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D")
embeddings = list(model.embed(sequences))

# [
#   array([-0.0055, -0.0144,  0.0355, -0.0049, ...], dtype=float32),
#   array([ 0.0114,  0.0020, -0.0247,  0.0060, ...], dtype=float32)
# ]
```

## Supported Models

### Protein Embeddings

| Model | Parameters | Dimensions | Description |
|-------|------------|------------|-------------|
| `facebook/esm2_t12_35M_UR50D` | 35M | 480 | ESM-2 protein language model |

```python
from fastembed.bio import ProteinEmbedding

model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D")
embeddings = list(model.embed(["MKTVRQERLKS", "GKGDPKKPRGK"]))
```

### DNA Embeddings (Coming Soon)

DNABert and similar models for DNA sequence embeddings.

### RNA Embeddings (Coming Soon)

RNA foundation models for RNA sequence embeddings.

## GPU Support

```python
from fastembed.bio import ProteinEmbedding

model = ProteinEmbedding(
    "facebook/esm2_t12_35M_UR50D",
    providers=["CUDAExecutionProvider"]
)
```

Requires `onnxruntime-gpu` instead of `onnxruntime`.

## Relationship to FastEmbed

This project is a community-driven fork of [FastEmbed](https://github.com/qdrant/fastembed) focused on biological sequence embeddings. It uses the same core infrastructure (ONNX models, model management, etc.) but is specialized for proteins, DNA, and RNA.

The goal is to make biological embeddings as accessible and efficient as text embeddings.

## Contributing

Contributions welcome! Areas of interest:

- Additional ESM-2 model sizes
- DNABert and other DNA models
- RNA foundation models
- Performance optimizations

## License

Apache 2.0