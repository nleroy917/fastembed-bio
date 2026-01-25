import importlib.metadata

from fastembed_bio.common import DNAInput, ProteinInput
from fastembed_bio.dna import DNAEmbedding
from fastembed_bio.protein import ProteinEmbedding

try:
    version = importlib.metadata.version("fastembed-bio")
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"

__version__ = version
__all__ = [
    "DNAEmbedding",
    "DNAInput",
    "ProteinEmbedding",
    "ProteinInput",
]