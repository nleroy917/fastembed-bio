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
    model = DNAEmbedding("InstaDeepAI/NTv3_650M_post")
    yield model
    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_dna_embedding(model_fixture) -> None:
    """
    Test basic DNA embedding functionality.
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


def test_dna_embedding_single_sequence(model_fixture) -> None:
    """
    Test embedding a single sequence passed as a string.
    """
    model = model_fixture
    dim = 1536

    # single sequence as string
    embedding = list(model.embed("ATCGATCGATCGATCG" * 8))
    assert len(embedding) == 1
    assert embedding[0].shape == (dim,)


def test_dna_embedding_species(model_fixture) -> None:
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


def test_dna_embedding_size() -> None:
    """
    Test get_embedding_size class method.
    """
    assert DNAEmbedding.get_embedding_size("InstaDeepAI/NTv3_650M_post") == 1536


def test_dna_embedding_lazy_load() -> None:
    """
    Test lazy loading functionality.
    """
    is_ci = os.getenv("CI")

    model = DNAEmbedding("InstaDeepAI/NTv3_650M_post", lazy_load=True)
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
        DNAEmbedding("nonexistent/model")


def test_unsupported_species(model_fixture) -> None:
    """
    Test that unsupported species raises ValueError.
    """
    model = model_fixture

    with pytest.raises(ValueError, match="not supported"):
        list(model.embed(SAMPLE_SEQUENCES[:1], species="invalid_species"))


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


def test_typed_input_single(model_fixture) -> None:
    """
    Test embedding with a single DNAInput object.
    """
    model = model_fixture
    dim = 1536

    inp = DNAInput(SAMPLE_SEQUENCES[0], species="human")
    embeddings = list(model.embed(inp))

    assert len(embeddings) == 1
    assert embeddings[0].shape == (dim,)


def test_typed_input_list(model_fixture) -> None:
    """
    Test embedding with a list of DNAInput objects.
    """
    model = model_fixture
    dim = 1536

    inputs = [
        DNAInput(SAMPLE_SEQUENCES[0], species="human"),
        DNAInput(SAMPLE_SEQUENCES[1], species="human"),
    ]
    embeddings = list(model.embed(inputs))

    assert len(embeddings) == 2
    assert all(emb.shape == (dim,) for emb in embeddings)


def test_typed_input_per_sequence_species(model_fixture) -> None:
    """
    Test embedding with different species per sequence using DNAInput.
    """
    model = model_fixture

    # same sequence, different species should give different embeddings
    inputs = [
        DNAInput(SAMPLE_SEQUENCES[0], species="human"),
        DNAInput(SAMPLE_SEQUENCES[0], species="mouse"),
    ]
    embeddings = list(model.embed(inputs))

    assert len(embeddings) == 2
    # embeddings should differ between species (same sequence)
    assert not np.allclose(
        embeddings[0], embeddings[1], atol=1e-3
    ), "Same sequence with different species should produce different embeddings"


def test_typed_input_mixed(model_fixture) -> None:
    """
    Test embedding with mixed inputs (strings and DNAInput objects).
    """
    model = model_fixture
    dim = 1536

    # mix of string (uses default species) and DNAInput
    inputs = [
        SAMPLE_SEQUENCES[0],  # string - uses default species
        DNAInput(SAMPLE_SEQUENCES[1], species="mouse"),
    ]
    embeddings = list(model.embed(inputs))

    assert len(embeddings) == 2
    assert all(emb.shape == (dim,) for emb in embeddings)


def test_typed_input_backwards_compatibility(model_fixture) -> None:
    """
    Test that the old API (plain strings with species kwarg) still works.
    """
    model = model_fixture

    # old API - should still work exactly as before
    emb_old = list(model.embed(SAMPLE_SEQUENCES[:1], species="human"))[0]

    # new API with DNAInput - should produce identical results
    emb_new = list(model.embed([DNAInput(SAMPLE_SEQUENCES[0], species="human")]))[0]

    assert np.allclose(
        emb_old, emb_new, atol=1e-6
    ), "Old and new API should produce identical embeddings"