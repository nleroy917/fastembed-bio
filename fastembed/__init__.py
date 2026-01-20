import importlib.metadata

from fastembed.bio import ProteinEmbedding

try:
    version = importlib.metadata.version("fastembed-bio")
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"

__version__ = version
__all__ = [
    "ProteinEmbedding",
]