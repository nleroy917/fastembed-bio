---
icon: lucide/flask-conical
---

# Protein Embedding Quickstart

Generate embeddings for protein sequences using ESM-2 models.

## Basic Usage

```python
from fastembed_bio import ProteinEmbedding

# Initialize the model (downloads on first use)
model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D")

# Embed sequences (amino acid strings)
sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTY",
]
embeddings = list(model.embed(sequences))

print(f"Shape: {embeddings[0].shape}")  # (480,)
```

## Single Sequence

You can pass a single sequence as a string:

```python
embedding = list(model.embed("MKTVRQERLKS"))
print(len(embedding))  # 1
```

## Batch Processing

Control batch size for memory efficiency:

```python
# Many sequences
sequences = ["MKTVRQERLKS"] * 100

# Process in batches
embeddings = list(model.embed(sequences, batch_size=32))
```

## Lazy Loading

Defer model loading until first use:

```python
# Model files are downloaded but not loaded into memory
model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D", lazy_load=True)

# Model loads here on first embed call
embeddings = list(model.embed(["MKTVRQERLKS"]))
```

## List Available Models

```python
models = ProteinEmbedding.list_supported_models()
for m in models:
    print(f"{m['model']}: {m['dim']} dimensions")
```

## Get Embedding Dimensions

```python
# Without loading the model
dim = ProteinEmbedding.get_embedding_size("facebook/esm2_t12_35M_UR50D")
print(dim)  # 480

# From an instance
model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D")
print(model.embedding_size)  # 480
```

## Embeddings are Normalized

All embeddings are L2-normalized (unit length):

```python
import numpy as np

embeddings = list(model.embed(["MKTVRQERLKS"]))
norm = np.linalg.norm(embeddings[0])
print(f"L2 norm: {norm:.4f}")  # ~1.0000
```

This makes them ready for cosine similarity comparisons or use with vector databases.