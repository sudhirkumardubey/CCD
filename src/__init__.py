"""
CCD - Centrifugal Compressor Design Framework
"""

__version__ = "0.1.0"
__author__ = "Sudhir Kumar Dubey"

# Import main classes for easy access
from .core.compressor import CentrifugalCompressor
from .geometry.geometry import Geometry
from .conditions.operating import OperatingCondition, ThermoProp, thermo_prop
from .analysis.performance import PerformanceMap

__all__ = [
    "CentrifugalCompressor",
    "Geometry",
    "OperatingCondition",
    "ThermoProp",
    "thermo_prop",
    "PerformanceMap",
]