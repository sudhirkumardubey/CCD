"""
Geometry module based on RadComp structure
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from math import pi


@dataclass
class Geometry: 
    """
    Centrifugal compressor geometry following RadComp structure
    """
    # Primary radii
    r1h: float  # Hub radius at inducer inlet
    r1s: float  # Shroud radius at inducer inlet
    r2h: float  # Hub radius at impeller inlet
    r2s:  float  # Shroud radius at impeller inlet (inducer exit)
    r4:  float   # Impeller outlet radius
    r5: float   # Diffuser outlet radius
    
    # Widths
    b2:  float   # Impeller inlet width
    b4: float   # Impeller outlet width
    b5: float   # Diffuser outlet width
    
    # Blade angles (in degrees)
    alpha2: float = 0.0     # Absolute flow angle at impeller inlet
    beta2: float = 0.0      # Relative blade angle at impeller inlet
    beta4: float = -50.0    # Blade angle at impeller outlet (typically negative)
    
    # Blade parameters
    n_blades: int = 12      # Number of impeller blades
    n_splitter: int = 0     # Number of splitter blades
    t_b: float = 0.002      # Blade thickness (m)
    t_cl: float = 0.0002    # Tip clearance (m)
    
    # Slip factor
    slip: float = 0.9       # Slip factor (Wiesner, Stanitz, etc.)
    
    # Blockage factors
    BF2: float = 0.95       # Blockage factor at impeller inlet
    BF4: float = 0.95       # Blockage factor at impeller outlet
    
    # Inducer parameters
    L_ind: float = 0.05     # Inducer axial length (m)
    
    # Derived parameters (calculated in __post_init__)
    r2rms: float = field(init=False)  # RMS radius at impeller inlet
    A1: float = field(init=False)     # Inducer inlet area
    A2: float = field(init=False)     # Impeller inlet area
    A2_eff: float = field(init=False) # Effective impeller inlet area
    A4: float = field(init=False)     # Impeller outlet area
    A4_eff: float = field(init=False) # Effective impeller outlet area
    A5: float = field(init=False)     # Diffuser outlet area
    A_x: float = field(init=False)    # Inlet flow area (axial projection)
    A_y: float = field(init=False)    # Inlet flow area (tangential projection)
    
    def __post_init__(self):
        """Calculate derived geometric parameters"""
        # RMS radius at impeller inlet
        self.r2rms = np.sqrt(0.5 * (self.r2s**2 + self.r2h**2))
        
        # Areas
        self.A1 = pi * (self.r1s**2 - self.r1h**2)
        self.A2 = pi * (self.r2s**2 - self.r2h**2)
        self.A2_eff = self.A2 * self.BF2 * np.cos(np.radians(self.alpha2))
        self.A4 = 2 * pi * self.r4 * self.b4
        self.A4_eff = self.A4 * self.BF4
        self.A5 = 2 * pi * self.r5 * self.b5
        
        # Inlet flow areas for velocity triangle calculations
        self.A_x = self.A2 * np.cos(np.radians(self.alpha2))
        self.A_y = self.A2 * np.sin(np.radians(self.alpha2))
    
    @classmethod
    def from_nondimensional(cls, r4: float, beta4: float, 
                           b4r4: float, n_blades: int,
                           r2sor4: float, r2sor2h: float,
                           Cmin: float, CR: float, **kwargs):
        """
        Create geometry from non-dimensional parameters (Mounier et al.  approach)
        
        Parameters
        ----------
        r4 : float
            Impeller outlet radius
        beta4 : float
            Blade angle at outlet (degrees)
        b4r4 : float
            Width-to-radius ratio at outlet
        n_blades : int
            Number of blades
        r2sor4 : float
            Shroud radius ratio at inlet
        r2sor2h : float
            Shroud-to-hub radius ratio at inlet
        Cmin : float
            Minimum clearance ratio
        CR : float
            Contraction ratio
        """
        b4 = b4r4 * r4
        r2s = r2sor4 * r4
        r2h = r2s / r2sor2h
        
        # Inducer sizing
        r1s = r2s + Cmin * r4
        r1h = r2h - Cmin * r4
        
        # Diffuser sizing
        r5 = CR * r4
        b5 = b4  # Assume constant width for simplicity
        
        # Impeller inlet width
        b2 = r2s - r2h
        
        return cls(
            r1h=r1h, r1s=r1s,
            r2h=r2h, r2s=r2s,
            r4=r4, r5=r5,
            b2=b2, b4=b4, b5=b5,
            beta4=beta4,
            n_blades=n_blades,
            **kwargs
        )
    
    def validate(self):
        """Validate geometry parameters"""
        errors = []
        
        if self.r1h >= self.r1s:
            errors.append("Inducer hub radius must be less than shroud radius")
        
        if self.r2h >= self.r2s:
            errors.append("Impeller inlet hub radius must be less than shroud radius")
        
        if self. r2s >= self.r4:
            errors.append("Impeller inlet shroud radius must be less than outlet radius")
        
        if self.r4 >= self.r5:
            errors.append("Impeller outlet radius must be less than diffuser outlet radius")
        
        if self.n_blades < 4:
            errors.append("Number of blades must be at least 4")
        
        if not -90 < self.beta4 < 0:
            errors.append("Outlet blade angle should be between -90 and 0 degrees")
        
        if errors:
            raise ValueError("Geometry validation failed:\n" + "\n".join(errors))
        
        return True