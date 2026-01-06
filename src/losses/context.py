"""Context object for impeller loss functions."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ImpellerContext:
    """Lightweight container passed into impeller loss functions."""

    geom: Any
    st2: Any
    st4: Any

    U2: float = 0.0
    U4: float = 0.0
    V2: float = 0.0
    V4: float = 0.0
    V2u: float = 0.0
    V4u: float = 0.0
    W2: float = 0.0
    W4: float = 0.0


__all__ = ["ImpellerContext"]
