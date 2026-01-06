"""
Geometry implementation aligned with RadComp, with compatibility aliases
for existing CCD code. Areas, slip, blockage, and hydraulic diameter match
radcomp-main/radcompressor/geometry.py behavior.
"""

from dataclasses import dataclass, field, fields
from typing import List, Union
import math


@dataclass
class Geometry:
    """Radial compressor geometry (RadComp-compatible).

    Primary fields mirror radcomp-main/radcompressor/geometry.py. Compatibility
    properties are provided for prior CCD usage (r1h/r1s, BF2/BF4, etc.).
    """

    # RadComp-native fields
    r1: float  # Inducer inlet radius
    r2s: float  # Shroud tip radius
    r2h: float  # Impeller hub radius
    beta2: float  # Mid-blade impeller inlet angle (deg)
    beta2s: float  # Impeller shroud angle (deg)
    alpha2: float  # Inlet flow angle (deg)
    r4: float  # Impeller tip radius at exit
    b4: float  # Impeller blade height at exit
    r5: float  # Diffuser outlet radius
    b5: float  # Diffuser passage width
    beta4: float  # Impeller outlet angle (deg)
    n_blades: int  # Number of full blades
    n_splits: int  # Number of splitter blades
    blade_e: float  # Blade thickness
    rug_imp: float  # Impeller surface roughness
    clearance: float  # Tip clearance
    backface: float  # Backface clearance
    rug_ind: float  # Inducer surface roughness
    l_ind: float  # Inducer length
    l_comp: float  # Impeller length (no impact on calc in RadComp)

    # Optional vaned diffuser inputs (TurboFlow-style)
    vd_leading_edge_angle: float = 0.0
    vd_trailing_edge_angle: float = 0.0
    vd_number_of_vanes: int = 0
    vd_throat_location_factor: float = 0.5
    vd_area_throat_ratio: float = 1.0

    blockage: List[float] = field(default_factory=lambda: [1, 1, 1, 1, 1])

    @property
    def r2rms(self) -> float:
        """RMS radius at impeller inlet."""
        return math.sqrt((self.r2s ** 2 + self.r2h ** 2) / 2.0)

    @property
    def A1_eff(self) -> float:
        """Effective area at station 1 including blockage."""
        return self.r1 ** 2 * math.pi * self.blockage[0]

    @property
    def A2(self) -> float:
        """Geometric area at impeller inlet."""
        return (self.r2s ** 2 - self.r2h ** 2) * math.pi

    @property
    def A2_eff(self) -> float:
        """Effective area at station 2 including blockage and flow angle."""
        return self.A2 * self.blockage[1] * math.cos(self.alpha2 / 180.0 * math.pi)

    @property
    def A_x(self) -> float:
        """Axial projection of inlet flow area (station 2)."""
        return self.A2 * self.blockage[1] * math.cos(self.beta2 / 180.0 * math.pi)

    @property
    def A_y(self) -> float:
        """Tangential projection of inlet flow area accounting for blades."""
        return (
            (self.r2s ** 2 - self.r2h ** 2) * math.pi * math.cos(self.beta2 / 180 * math.pi)
            - (self.r2s - self.r2h) * self.blade_e * self.n_blades
        ) * self.blockage[2]

    @property
    def beta2_opt(self) -> float:
        """Optimal beta2 for given A_x/A_y (RadComp helper)."""
        return math.atan(self.A_x / self.A_y * math.tan(self.beta2 / 180 * math.pi)) * 180 / math.pi

    @property
    def slip(self) -> float:
        """Slip factor according to Wiesner-Busemann (RadComp)."""
        return 1 - (math.cos(self.beta4 / 180 * math.pi)) ** 0.5 / (self.n_blades + self.n_splits) ** 0.7

    @property
    def eps_limit(self):
        """Placeholder kept for parity with RadComp API."""
        return None

    @property
    def hydraulic_diameter(self):
        """Return hydraulic diameter and length per RadComp formulation."""
        la = self.r2h / self.r2s
        Dh = 2 * self.r4 * (
            1.0 /
            (self.n_blades / math.pi / math.cos(self.beta4 / 180 * math.pi) + 2.0 * self.r4 / self.b4)
            + self.r2s / self.r4 /
            (2.0 / (1.0 - la)
             + 2.0 * self.n_blades / math.pi / (1 + la)
             * math.sqrt(1 + (1 + la ** 2 / 2) * math.tan(self.beta2s / 180 * math.pi) ** 2))
        )
        Lh = self.r4 * (1 - self.r2rms * 2 / 0.3048) / math.cos(self.beta4 / 180 * math.pi)
        return Dh, Lh

    @classmethod
    def from_dict(cls, data: dict, blockage: Union[List[float], None] = None):
        """Create Geometry from dict (RadComp-style)."""
        safe_names = [f.name for f in fields(cls)]
        parsed = {}

        if blockage is None and "blockage1" in data:
            blockage = [data[f"blockage{i+1}"] for i in range(5)]

        if blockage is None:
            raise ValueError("Blockage needs to be provided as an argument or in data.")

        for key, val in data.items():
            key_l = key.lower()
            if key_l in safe_names:
                parsed[key_l] = val

        parsed["blockage"] = blockage
        return cls(**parsed)

    # Compatibility aliases for existing CCD code ---------------------------------
    @property
    def r1s(self) -> float:
        return self.r1

    @property
    def r1h(self) -> float:
        return 0.0

    @property
    def b2(self) -> float:
        return self.r2s - self.r2h

    @property
    def t_cl(self) -> float:
        return self.clearance

    @property
    def n_splitter(self) -> int:
        return self.n_splits

    @property
    def BF2(self) -> float:
        return self.blockage[1]

    @property
    def BF4(self) -> float:
        return self.blockage[3]

    @property
    def A1(self) -> float:
        return math.pi * (self.r1s ** 2 - self.r1h ** 2)

    @property
    def A4(self) -> float:
        return 2 * math.pi * self.r4 * self.b4

    @property
    def A4_eff(self) -> float:
        return self.A4 * self.blockage[3]

    @property
    def A5(self) -> float:
        return 2 * math.pi * self.r5 * self.b5 * self.blockage[4]