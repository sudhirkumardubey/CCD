"""Impeller loss models"""

# Import all loss models to register them
from .  import skin_friction
from . import blade_loading
from . import clearance
from . import disc_friction
from . import recirculation
from . import incidence
from . import wake_mixing

__all__ = [
    "skin_friction",
    "blade_loading",
    "clearance",
    "disc_friction",
    "recirculation",
    "incidence",
    "wake_mixing",
]