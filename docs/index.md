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
dna_model = DNAEmbedding("PoetschLab/GROVER")
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
| `PoetschLab/GROVER` | 768 | GROVER DNA foundation model |

### Protein Embeddings

| Model | Dimensions | Description |
|-------|------------|-------------|
| `facebook/esm2_t12_35M_UR50D` | 480 | ESM-2 35M parameters |

## Next Steps

- [DNA Embedding Quickstart](quickstart/dna.md) - Get started with DNA sequence embeddings
- [Protein Embedding Quickstart](quickstart/protein.md) - Get started with protein sequence embeddings

## Roadmap

I'm actively expanding fastembed-bio. Here's where my mind is at for future model support:

### DNA/RNA Models
- [ ] Nucleotide Transformer v3 (species-conditioned embeddings)
- [ ] Hyena DNA
- [ ] Additional GROVER variants

### Protein Models
- [ ] ESM-2 larger variants (150M, 650M)
- [ ] ESMFold embeddings

### Single-Cell Models
- [ ] Geneformer (scRNA-seq)
- [ ] scGPT
- [ ] Tahoe-x1

### Other Modalities
- [ ] ATAC-seq embeddings (Atacformer)
- [ ] Multi-modal embeddings

Want to contribute? Check out our [GitHub repository](https://github.com/nleroy917/fastembed-bio)!