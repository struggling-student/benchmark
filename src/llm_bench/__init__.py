"""Small utilities for reproducible LLM inference benchmarks."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sapienza-llm-bench")
except PackageNotFoundError:  # pragma: no cover - useful when imported from a checkout
    __version__ = "0+unknown"

__all__ = ["__version__"]
