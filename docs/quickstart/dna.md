---
icon: lucide/dna
---

# DNA Embedding Quickstart

Generate embeddings for DNA sequences using foundation models like GROVER and Nucleotide Transformer v3.

## Available Models

| Model | Dimensions | Species Conditioning |
|-------|------------|---------------------|
| `PoetschLab/GROVER` | 768 | No |
| `InstaDeepAI/NTv3_650M_post` | 1536 | Yes |

## Basic Usage

```python
from fastembed_bio import DNAEmbedding

# GROVER - simple, no species needed
model = DNAEmbedding("PoetschLab/GROVER")
embeddings = list(model.embed(["ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA"]))
print(f"Shape: {embeddings[0].shape}")  # (768,)

# NTv3 - species-conditioned
model = DNAEmbedding("InstaDeepAI/NTv3_650M_post")
embeddings = list(model.embed(["ATCGATCGATCGATCG"], species="human"))
print(f"Shape: {embeddings[0].shape}")  # (1536,)
```

## Species-Conditioned Embeddings (NTv3)

The NTv3 model supports species conditioning, which improves embedding quality for sequences from specific organisms. GROVER does not use species conditioning.

### Default Species

For NTv3, all sequences use `species="human"` by default:

```python
# These are equivalent
embeddings = list(model.embed(["ATCGATCG"]))
embeddings = list(model.embed(["ATCGATCG"], species="human"))
```

### Per-Sequence Species

Use `DNAInput` to specify species for each sequence individually:

```python
from fastembed_bio import DNAEmbedding, DNAInput

model = DNAEmbedding("InstaDeepAI/NTv3_650M_post")

# Different species for each sequence
inputs = [
    DNAInput("ATCGATCGATCG", species="human"),
    DNAInput("GCTAGCTAGCTA", species="mouse"),
    DNAInput("NNNNNNNNNNNN", species="zebrafish"),
]

embeddings = list(model.embed(inputs))
```

### Mixed Inputs

You can mix plain strings and `DNAInput` objects:

```python
inputs = [
    "ATCGATCG",  # Uses default species (human)
    DNAInput("GCTAGCTA", species="mouse"),
]
embeddings = list(model.embed(inputs))
```

### List Supported Species

```python
species = model.list_supported_species()
print(species)  # ['human', 'mouse', 'zebrafish', ...]
```

## Batch Processing

Control batch size for memory efficiency:

```python
# Large dataset
sequences = ["ATCG" * 100] * 1000

# Process in batches of 16
embeddings = list(model.embed(sequences, batch_size=16))
```

## Lazy Loading

Defer model loading until first use:

```python
# Model files are downloaded but not loaded into memory
model = DNAEmbedding("InstaDeepAI/NTv3_650M_post", lazy_load=True)

# Model loads here on first embed call
embeddings = list(model.embed(["ATCGATCG"]))
```

## List Available Models

```python
models = DNAEmbedding.list_supported_models()
for m in models:
    print(f"{m['model']}: {m['dim']} dimensions")
```

## Get Embedding Dimensions

```python
# Without loading the model
dim = DNAEmbedding.get_embedding_size("InstaDeepAI/NTv3_650M_post")
print(dim)  # 1536

# From an instance
model = DNAEmbedding("InstaDeepAI/NTv3_650M_post")
print(model.embedding_size)  # 1536
```