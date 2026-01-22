import os

import numpy as np
import pytest

from fastembed_bio.bio import NucleotideEmbedding
from tests.utils import delete_model_cache


# sample DNA sequences for testing
SAMPLE_SEQUENCES = [
    "ATCGATCGATCGATCG" * 8,  # 128 nucleotides
    "GCTAGCTAGCTAGCTA" * 8,
]


@pytest.fixture(scope="module")
def model_fixture():
    """
    Fixture that provides the nucleotide embedding model and handles cleanup.
    """
    is_ci = os.getenv("CI")
    model = NucleotideEmbedding("InstaDeepAI/NTv3_650M_post")
    yield model
    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_nucleotide_embedding(model_fixture) -> None:
    """
    Test basic nucleotide embedding functionality.
    """
    model = model_fixture
    dim = 1536  # NTv3 650M has 1536 dimensions

    # Note: batch_size=1 required due to ONNX export limitation
    embeddings = list(model.embed(SAMPLE_SEQUENCES, batch_size=1))
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


def test_nucleotide_embedding_single_sequence(model_fixture) -> None:
    """
    Test embedding a single sequence passed as a string.
    """
    model = model_fixture
    dim = 1536

    # single sequence as string
    embedding = list(model.embed("ATCGATCGATCGATCG" * 8))
    assert len(embedding) == 1
    assert embedding[0].shape == (dim,)


def test_nucleotide_embedding_species(model_fixture) -> None:
    """
    Test embedding with different species conditioning.
    """
    model = model_fixture

    # test with human (default)
    emb_human = list(model.embed(SAMPLE_SEQUENCES[:1], species="human"))[0]

    # test with mouse
    emb_mouse = list(model.embed(SAMPLE_SEQUENCES[:1], species="mouse"))[0]

    # embeddings should be different for different species
    assert not np.allclose(
        emb_human, emb_mouse, atol=1e-3
    ), "Embeddings should differ between species"


def test_nucleotide_embedding_size() -> None:
    """
    Test get_embedding_size class method.
    """
    assert NucleotideEmbedding.get_embedding_size("InstaDeepAI/NTv3_650M_post") == 1536


def test_nucleotide_embedding_lazy_load() -> None:
    """
    Test lazy loading functionality.
    """
    is_ci = os.getenv("CI")

    model = NucleotideEmbedding("InstaDeepAI/NTv3_650M_post", lazy_load=True)
    # model should not be loaded yet
    assert not hasattr(model.model, "model") or model.model.model is None

    # after embedding, model should be loaded
    list(model.embed(SAMPLE_SEQUENCES[:1]))
    assert model.model.model is not None

    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_list_supported_models() -> None:
    """
    Test listing supported nucleotide models.
    """
    models = NucleotideEmbedding.list_supported_models()
    assert len(models) > 0
    assert any(m["model"] == "InstaDeepAI/NTv3_650M_post" for m in models)

    # check required fields
    for model_info in models:
        assert "model" in model_info
        assert "dim" in model_info
        assert "description" in model_info


def test_list_supported_species(model_fixture) -> None:
    """
    Test listing supported species.
    """
    model = model_fixture
    species = model.list_supported_species()

    assert len(species) > 0
    assert "human" in species
    assert "mouse" in species


def test_unsupported_model() -> None:
    """
    Test that unsupported model raises ValueError.
    """
    with pytest.raises(ValueError, match="not supported"):
        NucleotideEmbedding("nonexistent/model")


def test_unsupported_species(model_fixture) -> None:
    """
    Test that unsupported species raises ValueError.
    """
    model = model_fixture

    with pytest.raises(ValueError, match="not supported"):
        list(model.embed(SAMPLE_SEQUENCES[:1], species="invalid_species"))