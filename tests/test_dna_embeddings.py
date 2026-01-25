import os

import numpy as np
import pytest

from fastembed_bio import DNAEmbedding, DNAInput
from tests.utils import delete_model_cache


# sample DNA sequences for testing
SAMPLE_SEQUENCES = [
    "ATCGATCGATCGATCG" * 8,  # 128 nucleotides
    "GCTAGCTAGCTAGCTA" * 8,
]


@pytest.fixture(scope="module")
def model_fixture():
    """
    Fixture that provides the DNA embedding model and handles cleanup.
    """
    is_ci = os.getenv("CI")
    model = DNAEmbedding("PoetschLab/GROVER")
    yield model
    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_dna_embedding(model_fixture) -> None:
    """
    Test basic DNA embedding functionality.
    """
    model = model_fixture
    dim = 768  # GROVER has 768 dimensions

    embeddings = list(model.embed(SAMPLE_SEQUENCES))
    embeddings_arr = np.stack(embeddings, axis=0)

    assert embeddings_arr.shape == (
        2,
        dim,
    ), f"Expected shape (2, {dim}), got {embeddings_arr.shape}"

    # check that embeddings are normalized (L2 norm close to 1)
    norms = np.linalg.norm(embeddings_arr, axis=1)
    assert np.allclose(
        norms, 1.0, atol=1e-5
    ), f"Embeddings should be normalized, got norms: {norms}"


def test_dna_embedding_single_sequence(model_fixture) -> None:
    """
    Test embedding a single sequence passed as a string.
    """
    model = model_fixture
    dim = 768

    # single sequence as string
    embedding = list(model.embed("ATCGATCGATCGATCG" * 8))
    assert len(embedding) == 1
    assert embedding[0].shape == (dim,)


def test_dna_embedding_size() -> None:
    """
    Test get_embedding_size class method.
    """
    assert DNAEmbedding.get_embedding_size("PoetschLab/GROVER") == 768


def test_dna_embedding_lazy_load() -> None:
    """
    Test lazy loading functionality.
    """
    is_ci = os.getenv("CI")

    model = DNAEmbedding("PoetschLab/GROVER", lazy_load=True)
    # model should not be loaded yet
    assert not hasattr(model.model, "model") or model.model.model is None

    # after embedding, model should be loaded
    list(model.embed(SAMPLE_SEQUENCES[:1]))
    assert model.model.model is not None

    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_list_supported_models() -> None:
    """
    Test listing supported DNA models.
    """
    models = DNAEmbedding.list_supported_models()
    assert len(models) > 0
    assert any(m["model"] == "PoetschLab/GROVER" for m in models)

    # check required fields
    for model_info in models:
        assert "model" in model_info
        assert "dim" in model_info
        assert "description" in model_info


def test_unsupported_model() -> None:
    """
    Test that unsupported model raises ValueError.
    """
    with pytest.raises(ValueError, match="not supported"):
        DNAEmbedding("nonexistent/model")


def test_dna_input_basic() -> None:
    """
    Test DNAInput dataclass basic functionality.
    """
    inp = DNAInput("ATCGATCG", species="human")
    assert inp.sequence == "ATCGATCG"
    assert inp.species == "human"

    # default species
    inp_default = DNAInput("GCTAGCTA")
    assert inp_default.species == "human"


def test_dna_input_empty_sequence() -> None:
    """
    Test that DNAInput rejects empty sequences.
    """
    with pytest.raises(ValueError, match="cannot be empty"):
        DNAInput("")


def test_dna_embedding_batch(model_fixture) -> None:
    """
    Test batch embedding with different batch sizes.
    """
    model = model_fixture
    dim = 768

    sequences = SAMPLE_SEQUENCES * 5  # 10 sequences

    # test with small batch size
    embeddings_small_batch = list(model.embed(sequences, batch_size=2))
    embeddings_small_batch = np.stack(embeddings_small_batch, axis=0)
    assert embeddings_small_batch.shape == (len(sequences), dim)

    # test with larger batch size
    embeddings_large_batch = list(model.embed(sequences, batch_size=8))
    embeddings_large_batch = np.stack(embeddings_large_batch, axis=0)
    assert embeddings_large_batch.shape == (len(sequences), dim)

    # results should be the same regardless of batch size
    assert np.allclose(embeddings_small_batch, embeddings_large_batch, atol=1e-5)