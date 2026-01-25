---
icon: lucide/dna
---

# DNA Embedding Quickstart

Generate embeddings for DNA sequences using the Nucleotide Transformer v3 model.

## Basic Usage

```python
from fastembed_bio import DNAEmbedding

# Initialize the model (downloads on first use)
model = DNAEmbedding("InstaDeepAI/NTv3_650M_post")

# Embed sequences
sequences = ["ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA"]
embeddings = list(model.embed(sequences))

print(f"Shape: {embeddings[0].shape}")  # (1536,)
```

## Species-Conditioned Embeddings

The NTv3 model supports species conditioning, which improves embedding quality for sequences from specific organisms.

### Default Species

By default, all sequences use `species="human"`:

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