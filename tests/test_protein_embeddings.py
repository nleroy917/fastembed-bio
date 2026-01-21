import os

import numpy as np
import pytest

from fastembed_bio.bio import ProteinEmbedding
from tests.utils import delete_model_cache


# Sample protein sequences for testing
SAMPLE_SEQUENCES = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "GKGDPKKPRGKMSSYAFFVQTSREEHKKKHPDASVNFSEFSKKCSERWKTMSAKEKGKFEDMAKADKARYEREMKTY",
]


CANONICAL_VECTOR_VALUES = {
    "facebook/esm2_t12_35M_UR50D": np.array(
        [-0.0055, -0.0144, 0.0355, -0.0049, 0.0071]
    ),
}


@pytest.fixture(scope="module")
def model_fixture():
    """
    Fixture that provides the protein embedding model and handles cleanup.
    """
    is_ci = os.getenv("CI")
    model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D")
    yield model
    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_protein_embedding(model_fixture) -> None:
    """Test basic protein embedding functionality."""
    model = model_fixture
    dim = 480  # ESM2 t12 35M has 480 dimensions

    embeddings = list(model.embed(SAMPLE_SEQUENCES))
    embeddings = np.stack(embeddings, axis=0)

    assert embeddings.shape == (2, dim), f"Expected shape (2, {dim}), got {embeddings.shape}"

    # Check that embeddings are normalized (L2 norm close to 1)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Embeddings should be normalized, got norms: {norms}"


def test_protein_embedding_canonical_values(model_fixture) -> None:
    """
    Test that embeddings match expected canonical values.
    """
    model = model_fixture
    canonical_vector = CANONICAL_VECTOR_VALUES["facebook/esm2_t12_35M_UR50D"]

    embeddings = list(model.embed(SAMPLE_SEQUENCES[:1]))
    embedding = embeddings[0]

    assert np.allclose(
        embedding[: canonical_vector.shape[0]], canonical_vector, atol=1e-3
    ), f"First 5 values {embedding[:5]} don't match canonical {canonical_vector}"


def test_protein_embedding_single_sequence(model_fixture) -> None:
    """
    Test embedding a single sequence passed as a string.
    """
    model = model_fixture
    dim = 480

    # Single sequence as string
    embedding = list(model.embed("MKTVRQERLKS"))
    assert len(embedding) == 1
    assert embedding[0].shape == (dim,)


def test_protein_embedding_batch(model_fixture) -> None:
    """
    Test batch embedding with different batch sizes.
    """
    model = model_fixture
    dim = 480

    sequences = SAMPLE_SEQUENCES * 10  # 20 sequences

    # test with small batch size
    embeddings_small_batch = list(model.embed(sequences, batch_size=4))
    embeddings_small_batch = np.stack(embeddings_small_batch, axis=0)
    assert embeddings_small_batch.shape == (len(sequences), dim)

    # test with larger batch size
    embeddings_large_batch = list(model.embed(sequences, batch_size=16))
    embeddings_large_batch = np.stack(embeddings_large_batch, axis=0)
    assert embeddings_large_batch.shape == (len(sequences), dim)

    # results should be the same regardless of batch size
    assert np.allclose(embeddings_small_batch, embeddings_large_batch, atol=1e-5)


def test_protein_embedding_size() -> None:
    """
    Test get_embedding_size class method.
    """
    assert ProteinEmbedding.get_embedding_size("facebook/esm2_t12_35M_UR50D") == 480


def test_protein_embedding_lazy_load() -> None:
    """
    Test lazy loading functionality.
    """
    is_ci = os.getenv("CI")

    model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D", lazy_load=True)
    # model should not be loaded yet
    assert not hasattr(model.model, "model") or model.model.model is None

    # after embedding, model should be loaded
    list(model.embed(SAMPLE_SEQUENCES[:1]))
    assert model.model.model is not None

    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_list_supported_models() -> None:
    """
    Test listing supported protein models.
    """
    models = ProteinEmbedding.list_supported_models()
    assert len(models) > 0
    assert any(m["model"] == "facebook/esm2_t12_35M_UR50D" for m in models)

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
        ProteinEmbedding("nonexistent/model")