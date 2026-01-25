---
icon: lucide/dna
---

# DNA Embedding Quickstart

Generate embeddings for DNA sequences using the GROVER foundation model.

## Basic Usage

```python
from fastembed_bio import DNAEmbedding

# Initialize the model (downloads on first use)
model = DNAEmbedding("PoetschLab/GROVER")

# Embed sequences
sequences = ["ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA"]
embeddings = list(model.embed(sequences))

print(f"Shape: {embeddings[0].shape}")  # (768,)
```

## Single Sequence

You can pass a single sequence as a string:

```python
embedding = list(model.embed("ATCGATCGATCGATCG"))
print(len(embedding))  # 1
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
model = DNAEmbedding("PoetschLab/GROVER", lazy_load=True)

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
dim = DNAEmbedding.get_embedding_size("PoetschLab/GROVER")
print(dim)  # 768

# From an instance
model = DNAEmbedding("PoetschLab/GROVER")
print(model.embedding_size)  # 768
```

## Embeddings are Normalized

All embeddings are L2-normalized (unit length):

```python
import numpy as np

embeddings = list(model.embed(["ATCGATCG"]))
norm = np.linalg.norm(embeddings[0])
print(f"L2 norm: {norm:.4f}")  # ~1.0000
```

This makes them ready for cosine similarity comparisons or use with vector databases.