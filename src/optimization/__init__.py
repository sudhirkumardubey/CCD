"""Optimization entrypoints for CCD."""

from .compressor_optimization import (
    compute_optimal_compressor,
    CompressorOptimizationProblem,
)

__all__ = ["compute_optimal_compressor", "CompressorOptimizationProblem"]
