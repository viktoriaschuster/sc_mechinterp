"""
scFeatureLens: A tool for mechanistic interpretability of single-cell data embeddings.

This package provides functionality for:
- Training sparse autoencoders on embeddings
- Extracting interpretable features
- Performing differential expression analysis
- Gene set enrichment analysis

Author: Viktoria Schuster
License: MIT
"""

from .pipeline import SCFeatureLensPipeline
from .utils import AnalysisConfig
from .sae import SparseAutoencoder

__version__ = "0.1.0"
__author__ = "Viktoria Schuster"

__all__ = [
    "SCFeatureLensPipeline",
    "AnalysisConfig", 
    "SparseAutoencoder",
]
