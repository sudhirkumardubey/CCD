"""
CCD - Centrifugal Compressor Design Framework
"""

__version__ = "0.1.0"
__author__ = "Sudhir Kumar Dubey"

# Import main classes for easy access
from .core.compressor import CentrifugalCompressor
from .geometry.geometry import Geometry
from .conditions.operating import OperatingCondition, ThermoProp, thermo_prop
from .analysis.performance import PerformanceMap, run_radcomp_performance_plot
from .optimization.compressor_optimization import (
    compute_optimal_compressor,
    CompressorOptimizationProblem,
)

__all__ = [
    "CentrifugalCompressor",
    "Geometry",
    "OperatingCondition",
    "ThermoProp",
    "thermo_prop",
    "PerformanceMap",
    "run_radcomp_performance_plot",
    "compute_optimal_compressor",
    "CompressorOptimizationProblem",
]