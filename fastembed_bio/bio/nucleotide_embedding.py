import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence, Type

import numpy as np
from tokenizers import Tokenizer

from fastembed_bio.common.model_description import DenseModelDescription, ModelSource
from fastembed_bio.common.model_management import ModelManagement
from fastembed_bio.common.onnx_model import EmbeddingWorker, OnnxModel, OnnxOutputContext
from fastembed_bio.common.types import Device, NumpyArray, OnnxProvider
from fastembed_bio.common.utils import define_cache_dir, iter_batch, normalize


supported_nucleotide_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="InstaDeepAI/NTv3_650M_post",
        dim=1536,
        description="Nucleotide Transformer v3, 650M parameters, 1536 dimensions, species-conditioned DNA embeddings",
        license="cc-by-nc-sa-4.0",
        size_in_GB=2.6,
        sources=ModelSource(hf="nleroy917/ntv3-650m-post-onnx"),
        model_file="model.onnx",
        additional_files=[
            "vocab.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "species_config.json",
            "model_config.json",
            "model.onnx.data",
        ],
    ),
]


def load_nucleotide_tokenizer(model_dir: Path, max_length: int = 6144) -> Tokenizer:
    """
    Load a nucleotide tokenizer from model directory.

    Attempts to load in order:
    1. tokenizer.json (standard HuggingFace fast tokenizer format)
    2. Build from vocab.json (fallback for NTv3-style tokenizers)

    Args:
        model_dir: Path to model directory containing tokenizer files
        max_length: Maximum sequence length (default 6144, must be multiple of 128 for NTv3)

    Returns:
        Configured Tokenizer instance
    """
    from tokenizers import pre_tokenizers
    from tokenizers.models import WordLevel

    tokenizer_json_path = model_dir / "tokenizer.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    vocab_json_path = model_dir / "vocab.json"

    # try to load tokenizer.json directly (preferred)
    if tokenizer_json_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                config = json.load(f)
                config_max_length = config.get("model_max_length", max_length)
                if config_max_length <= max_length:
                    max_length = config_max_length
        tokenizer.enable_truncation(max_length=max_length)
        tokenizer.enable_padding(pad_id=1, pad_token="<pad>", pad_to_multiple_of=128)
        return tokenizer

    # fall back to building from vocab.json (NTv3 style)
    if not vocab_json_path.exists():
        raise ValueError(
            f"Could not find tokenizer.json or vocab.json in {model_dir}"
        )

    with open(vocab_json_path) as f:
        vocab: dict[str, int] = json.load(f)

    # build tokenizer from vocab
    unk_token = "<unk>"
    pad_token = "<pad>"

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))

    # character-level pre-tokenizer (split each character)
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern="", behavior="isolated", invert=False
    )

    # no special tokens added for NTv3 (add_special_tokens=False in original)
    pad_token_id = vocab.get(pad_token, 1)
    tokenizer.enable_padding(
        pad_id=pad_token_id, pad_token=pad_token, pad_to_multiple_of=128
    )
    tokenizer.enable_truncation(max_length=max_length)

    return tokenizer


def load_species_config(model_dir: Path) -> dict[str, Any]:
    """
    Load species configuration from model directory.

    Args:
        model_dir: Path to model directory

    Returns:
        Dictionary with species_to_id, id_to_species, and supported_species
    """
    species_config_path = model_dir / "species_config.json"
    if not species_config_path.exists():
        raise ValueError(f"Could not find species_config.json in {model_dir}")

    with open(species_config_path) as f:
        return json.load(f)


class NucleotideEmbeddingBase(ModelManagement[DenseModelDescription]):
    """
    Base class for nucleotide sequence embeddings.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)
        self._embedding_size: int | None = None

    def embed(
        self,
        sequences: str | Iterable[str],
        batch_size: int = 32,
        parallel: int | None = None,
        species: str = "human",
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Embed nucleotide (DNA) sequences.

        Args:
            sequences: Single DNA sequence or iterable of sequences
            batch_size: Batch size for encoding
            parallel: Number of parallel workers (None for single-threaded)
            species: Species name for conditioning (default: "human")

        Yields:
            Embeddings as numpy arrays
        """
        raise NotImplementedError()

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """
        Returns embedding size of the passed model.

        Args:
            model_name: Name of the model
        """
        descriptions = cls._list_supported_models()
        for description in descriptions:
            if description.model.lower() == model_name.lower():
                if description.dim is not None:
                    return description.dim
        raise ValueError(f"Model {model_name} not found")

    @property
    def embedding_size(self) -> int:
        """
        Returns embedding size for the current model.
        """
        if self._embedding_size is None:
            self._embedding_size = self.get_embedding_size(self.model_name)
        return self._embedding_size

    def list_supported_species(self) -> list[str]:
        """
        Returns list of supported species for conditioning.

        Returns:
            List of species names
        """
        raise NotImplementedError()


class OnnxNucleotideModel(OnnxModel[NumpyArray]):
    """
    ONNX model handler for nucleotide embeddings.
    """

    ONNX_OUTPUT_NAMES: list[str] | None = None
    # NTv3 model uses special tokens offset for species IDs
    # The actual species_id = base_index + num_species_special_tokens
    NUM_SPECIES_SPECIAL_TOKENS: int = 13

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer: Tokenizer | None = None
        self.species_config: dict[str, Any] | None = None

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        threads: int | None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_id: int | None = None,
        extra_session_options: dict[str, Any] | None = None,
    ) -> None:
        super()._load_onnx_model(
            model_dir=model_dir,
            model_file=model_file,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_id=device_id,
            extra_session_options=extra_session_options,
        )
        self.tokenizer = load_nucleotide_tokenizer(model_dir)
        self.species_config = load_species_config(model_dir)

    def _get_species_id(self, species: str) -> int:
        """Convert species name to model species ID.

        Args:
            species: Species name (e.g., "human", "mouse")

        Returns:
            Species ID for the model (includes special token offset)
        """
        if self.species_config is None:
            raise ValueError("Species config not loaded")

        species_to_id = self.species_config.get("species_to_id", {})
        if species not in species_to_id:
            supported = list(species_to_id.keys())
            raise ValueError(
                f"Species '{species}' not supported. Supported species: {supported}"
            )

        base_id = species_to_id[species]
        # add special token offset to get actual model species ID
        return base_id + self.NUM_SPECIES_SPECIAL_TOKENS

    def onnx_embed(
        self, sequences: list[str], species: str = "human", **kwargs: Any
    ) -> OnnxOutputContext:
        """
        Run ONNX inference on nucleotide sequences.

        Args:
            sequences: List of DNA sequences (A, T, C, G, N characters)
            species: Species name for conditioning

        Returns:
            OnnxOutputContext containing model output and inputs
        """
        assert self.tokenizer is not None

        # normalize sequences to uppercase
        sequences = [seq.upper() for seq in sequences]

        # tokenize
        encoded = self.tokenizer.encode_batch(sequences)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        # Get species IDs for the batch
        species_id = self._get_species_id(species)
        batch_size = len(sequences)
        species_ids = np.full((batch_size,), species_id, dtype=np.int64)

        onnx_input: dict[str, NumpyArray] = {
            "input_ids": input_ids,
            "species_ids": species_ids,
        }

        model_output = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)  # type: ignore[union-attr]

        return OnnxOutputContext(
            model_output=model_output[0],
            attention_mask=attention_mask,
            input_ids=input_ids,
        )

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        """Convert ONNX output to embeddings with mean pooling."""
        embeddings = output.model_output
        attention_mask = output.attention_mask

        if attention_mask is None:
            raise ValueError("attention_mask is required for mean pooling")

        # Mean pooling over sequence length
        mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
        sum_mask = np.sum(mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        mean_embeddings = sum_embeddings / sum_mask

        return normalize(mean_embeddings)


class OnnxNucleotideEmbedding(NucleotideEmbeddingBase, OnnxNucleotideModel):
    """
    ONNX-based nucleotide embedding implementation.
    """

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_nucleotide_models

    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_650M_post",
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        device_id: int | None = None,
        specific_model_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load
        self._extra_session_options = self._select_exposed_session_options(kwargs)
        self.device_ids = device_ids
        self.cuda = cuda

        self.device_id: int | None = None
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))
        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

        if not self.lazy_load:
            self.load_onnx_model()

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
            extra_session_options=self._extra_session_options,
        )

    def embed(
        self,
        sequences: str | Iterable[str],
        batch_size: int = 32,
        parallel: int | None = None,
        species: str = "human",
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Embed nucleotide (DNA) sequences.

        Args:
            sequences: Single DNA sequence or iterable of sequences (A, T, C, G, N)
            batch_size: Batch size for encoding
            parallel: Number of parallel workers (not yet supported)
            species: Species name for conditioning (default: "human")

        Yields:
            Embeddings as numpy arrays, one per sequence
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

        for batch in iter_batch(sequences, batch_size):
            yield from self._post_process_onnx_output(
                self.onnx_embed(batch, species=species, **kwargs), **kwargs
            )

    def list_supported_species(self) -> list[str]:
        """Returns list of supported species for conditioning."""
        if self.species_config is None:
            # Load species config if not yet loaded
            self.load_onnx_model()
        if self.species_config is None:
            raise ValueError("Could not load species config")
        return self.species_config.get("supported_species", [])

    @classmethod
    def _get_worker_class(cls) -> Type["NucleotideEmbeddingWorker"]:
        return NucleotideEmbeddingWorker


class NucleotideEmbeddingWorker(EmbeddingWorker[NumpyArray]):
    """Worker class for parallel nucleotide embedding processing."""

    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxNucleotideEmbedding:
        return OnnxNucleotideEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )

    def process(
        self, items: Iterable[tuple[int, Any]]
    ) -> Iterable[tuple[int, OnnxOutputContext]]:
        for idx, batch in items:
            onnx_output = self.model.onnx_embed(batch)
            yield idx, onnx_output


class NucleotideEmbedding(NucleotideEmbeddingBase):
    """
    Nucleotide (DNA) sequence embedding using Nucleotide Transformer v3 and similar models.

    Example:
        >>> from fastembed_bio import NucleotideEmbedding
        >>> model = NucleotideEmbedding("InstaDeepAI/NTv3_650M_post")
        >>> embeddings = list(model.embed(["ATCGATCGATCG", "GCTAGCTAGCTA"], species="human"))
        >>> print(embeddings[0].shape)
        (1536,)

    The model supports species-conditioned embeddings. Use `list_supported_species()`
    to see available species options.
    """

    EMBEDDINGS_REGISTRY: list[Type[NucleotideEmbeddingBase]] = [OnnxNucleotideEmbedding]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        result: list[DenseModelDescription] = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding._list_supported_models())
        return result

    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_650M_post",
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize NucleotideEmbedding.

        Args:
            model_name: Name of the model to use
            cache_dir: Path to cache directory
            threads: Number of threads for ONNX runtime
            providers: ONNX execution providers
            cuda: Whether to use CUDA
            device_ids: List of device IDs for multi-GPU
            lazy_load: Whether to load model lazily
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)

        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE._list_supported_models()
            if any(
                model_name.lower() == model.model.lower() for model in supported_models
            ):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    threads=threads,
                    providers=providers,
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in NucleotideEmbedding. "
            "Please check the supported models using `NucleotideEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        sequences: str | Iterable[str],
        batch_size: int = 32,
        parallel: int | None = None,
        species: str = "human",
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Embed nucleotide (DNA) sequences.

        Args:
            sequences: Single DNA sequence or iterable of sequences (A, T, C, G, N)
            batch_size: Batch size for encoding
            parallel: Number of parallel workers
            species: Species name for conditioning (default: "human")

        Yields:
            Embeddings as numpy arrays, one per sequence
        """
        yield from self.model.embed(
            sequences, batch_size, parallel, species=species, **kwargs
        )

    def list_supported_species(self) -> list[str]:
        """
        Returns list of supported species for conditioning.

        Returns:
            List of species names (e.g., ["human", "mouse", ...])
        """
        return self.model.list_supported_species()