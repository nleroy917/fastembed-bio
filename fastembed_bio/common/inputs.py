from dataclasses import dataclass


@dataclass(frozen=True)
class DNAInput:
    """
    Input for DNA embedding models that support additional metadata.

    Attributes:
        sequence: DNA sequence string (A, T, C, G, N characters)
        species: Species name for conditioning (e.g., "human", "mouse").
                 Only used by models that support species conditioning.

    Example:
        >>> inp = DNAInput("ATCGATCG", species="human")
        >>> inp.sequence
        'ATCGATCG'
    """

    sequence: str
    species: str = "human"

    def __post_init__(self) -> None:
        if not self.sequence:
            raise ValueError("sequence cannot be empty")
    
    @classmethod
    def from_dict(cls, data: dict) -> "DNAInput":
        """
        Create a DNAInput instance from a dictionary.

        Args:
            data: Dictionary with keys 'sequence' and optional 'species'.
        Returns:
            DNAInput instance.
        """
        return cls(
            sequence=data["sequence"],
            species=data.get("species", "human")
        )


@dataclass(frozen=True)
class ProteinInput:
    """
    Input for protein embedding models.

    Currently protein models only need sequences, but this class
    future-proofs the API for models that may need additional metadata
    (e.g., organism, structure hints).

    Attributes:
        sequence: Protein sequence string (amino acid characters)

    Example:
        >>> inp = ProteinInput("MKTVRQERLKS")
        >>> inp.sequence
        'MKTVRQERLKS'
    """

    sequence: str

    def __post_init__(self) -> None:
        if not self.sequence:
            raise ValueError("sequence cannot be empty")