---
icon: lucide/dna
---

# fastembed-bio

Fast, lightweight biological sequence embeddings using ONNX. No PyTorch or GPU required.

## Installation

```bash
pip install fastembed-bio
```

## Quick Example

```python
from fastembed_bio import DNAEmbedding, ProteinEmbedding

# DNA embeddings
dna_model = DNAEmbedding("InstaDeepAI/NTv3_650M_post")
dna_embeddings = list(dna_model.embed(["ATCGATCGATCG", "GCTAGCTAGCTA"]))

# Protein embeddings
protein_model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D")
protein_embeddings = list(protein_model.embed(["MKTVRQERLKS", "GKGDPKKPRGKM"]))
```

## Why fastembed-bio?

- **Fast**: ONNX runtime for CPU inference, no GPU needed
- **Lightweight**: Minimal dependencies, small model files
- **Simple**: Clean API inspired by [fastembed](https://github.com/qdrant/fastembed)
- **Biological focus**: DNA, protein, and (coming soon) single-cell embeddings

## Supported Models

### DNA Embeddings

| Model | Dimensions | Description |
|-------|------------|-------------|
| `InstaDeepAI/NTv3_650M_post` | 1536 | Nucleotide Transformer v3, species-conditioned |

### Protein Embeddings

| Model | Dimensions | Description |
|-------|------------|-------------|
| `facebook/esm2_t12_35M_UR50D` | 480 | ESM-2 35M parameters |

## Next Steps

- [DNA Embedding Quickstart](quickstart/dna.md) - Get started with DNA sequence embeddings
- [Protein Embedding Quickstart](quickstart/protein.md) - Get started with protein sequence embeddings