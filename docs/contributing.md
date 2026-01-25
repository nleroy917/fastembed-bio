---
icon: lucide/git-pull-request
---

# Contributing

Thanks for your interest in contributing to fastembed-bio! I'm focused on building a solid foundation of biological embedding models and welcome help expanding the library.

## Current Focus

I'm currently prioritizing:

- Getting core DNA and protein models working reliably
- Clean, simple APIs that mirror the original [fastembed](https://github.com/qdrant/fastembed)
- Expanding to more advanced models (single-cell, multi-modal, etc.)

## Adding a New Model

The general workflow for adding a model:

### 1. Identify the Model

Pick a model you want to add. Good candidates are:

- Models with permissive licenses (Apache 2.0, MIT, CC-BY)
- Models available on HuggingFace
- Models that produce fixed-size embeddings

### 2. Get the Forward Pass Running Locally

Before exporting to ONNX, make sure you can run inference with pure PyTorch/Transformers:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("org/model-name")
tokenizer = AutoTokenizer.from_pretrained("org/model-name")

# Run a forward pass
inputs = tokenizer(["ATCGATCG"], return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # or however the model outputs embeddings
```

### 3. Export to ONNX

Create a conversion script in `scripts/onnx_conversion/`. See existing scripts for examples:

```python
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "embeddings": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=14,
)
```

Validate the export matches PyTorch outputs:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
onnx_output = session.run(None, {"input_ids": ..., "attention_mask": ...})

# Compare with PyTorch output
assert np.allclose(pytorch_output, onnx_output, atol=1e-4)
```

### 4. Upload to HuggingFace

Upload the ONNX files to a HuggingFace repo following the naming convention:

```
nleroy917/<model-name>-onnx
```

Include:

- `model.onnx` (and `model.onnx.data` if weights are external)
- `tokenizer.json` and/or `vocab.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `model_config.json` (optional, for model metadata)

### 5. Add to fastembed-bio

1. Add the model to `supported_dna_models` or `supported_protein_models` in the appropriate embedding file
2. Update tokenizer loading if the model uses a different format
3. Add tests in `tests/`
4. Update docs in `docs/`

### 6. Submit a PR

Open a pull request with:

- The conversion script
- Code changes to support the model
- Tests
- Doc updates

## Development Setup

```bash
# Clone the repo
git clone https://github.com/nleroy917/fastembed-bio
cd fastembed-bio

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Questions?

Open an issue on [GitHub](https://github.com/nleroy917/fastembed-bio/issues) if you have questions or need help with a model conversion.