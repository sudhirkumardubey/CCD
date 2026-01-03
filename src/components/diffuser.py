"""
Diffuser components - Vaneless and vaned diffusers
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from math import pi, sin, cos, tan, atan, radians, degrees

from geometry.geometry import Geometry
from conditions.operating import (
    OperatingCondition, ThermoProp,
    static_from_total, total_from_static, thermo_prop
)
from losses.registry import LossModelRegistry, LossContext


@dataclass
class DiffuserState:
    """State at a station in the diffuser"""
    total: Optional[ThermoProp] = None
    static: Optional[ThermoProp] = None
    isentropic: Optional[ThermoProp] = None
    
    # Velocity
    c: float = 0.0          # Absolute velocity
    c_m: float = 0.0        # Meridional component
    c_t: float = 0.0        # Tangential component
    alpha: float = 0.0      # Flow angle
    m_abs: float = 0.0      # Absolute Mach number


@dataclass
class VanelessDiffuser:
    """
    Vaneless diffuser component
    Flow from impeller outlet (r4) to diffuser outlet (r5)
    """
    geom: Geometry
    op: OperatingCondition
    imp: Any  # Impeller object
    loss_config: dict = field(default_factory=dict)
    
    # States
    inlet: DiffuserState = field(default_factory=DiffuserState)
    out: DiffuserState = field(default_factory=DiffuserState)
    
    # Performance metrics
    dh0s: float = 0.0
    eff: float = 0.0
    choke_flag: bool = False
    Cp: float = 0.0  # Pressure recovery coefficient
    
    def __post_init__(self):
        """Initialize and calculate diffuser"""
        # Inlet is impeller outlet
        self.inlet. total = self.imp.out. total
        self.inlet.static = self.imp.out.static
        self.inlet.c = self.imp.out.c
        self.inlet.c_m = self.imp.out.c_m
        self. inlet.c_t = self. imp.out.c_t
        self.inlet.alpha = self.imp.out.alpha
        
        self.calculate()
    
    def calculate(self):
        """Main calculation for vaneless diffuser"""
        geom = self.geom
        op = self.op
        
        # Conservation of angular momentum in vaneless diffuser
        # r * c_theta = constant
        r4 = geom.r4
        r5 = geom.r5
        c_t4 = self.inlet.c_t
        c_t5 = c_t4 * r4 / r5
        
        # Calculate meridional velocity from continuity
        # m = rho * c_m * A
        # For vaneless diffuser: assume constant static density (approximation)
        # or iterate to find consistent solution
        
        # Simplified approach: assume constant flow angle
        alpha5 = self.inlet.alpha
        c5 = c_t5 / sin(radians(alpha5))
        c_m5 = c5 * cos(radians(alpha5))
        
        # More accurate:  iterate to satisfy continuity
        def residual(c5):
            c_t5_calc = c_t4 * r4 / r5
            c_m5_calc = np.sqrt(c5**2 - c_t5_calc**2)
            
            # Static state
            h5 = self.inlet.total. H - 0.5 * c5**2
            
            # Estimate density (assuming isentropic for first approximation)
            try:
                tp5_static = thermo_prop(op.fld, "HS", h5, self.inlet.total.S)
                
                # Continuity
                A5 = geom.A5
                mass_calc = tp5_static. D * c_m5_calc * A5
                
                return (mass_calc - op.m) / op.m
            except: 
                return 100.0
        
        from scipy.optimize import brentq
        
        try: 
            # Find velocity that satisfies continuity
            c5_min = 10.0
            c5_max = 200.0
            c5_solution = brentq(residual, c5_min, c5_max)
            
            # Calculate final state
            c_t5 = c_t4 * r4 / r5
            c_m5 = np.sqrt(c5_solution**2 - c_t5**2)
            alpha5 = degrees(atan(c_t5 / c_m5))
            
            # Store outlet state
            self.out.c = c5_solution
            self.out.c_m = c_m5
            self.out.c_t = c_t5
            self.out.alpha = alpha5
            
            # Thermodynamic state
            h5_static = self.inlet.total.H - 0.5 * c5_solution**2
            
            # Account for losses
            loss_factor = 0.05  # Typical diffuser loss
            h5_static_actual = h5_static - loss_factor * 0.5 * self.inlet.c**2
            
            self.out.static = thermo_prop(
                op. fld, "HS", h5_static_actual, self. inlet.static.S
            )
            self.out.total = total_from_static(self.out.static, c5_solution)
            self.out.isentropic = thermo_prop(
                op.fld, "PS", self.out.total. P, self.inlet.total.S
            )
            
            # Mach number
            self.out.m_abs = c5_solution / self.out.static.A
            
            # Performance metrics
            self.dh0s = self.out.isentropic.H - self.inlet.total.H
            delta_h = self.out.total. H - self.inlet.total. H
            
            if abs(delta_h) > 1e-6:
                self.eff = self. dh0s / delta_h
            else:
                self. eff = 1.0
            
            # Pressure recovery coefficient
            q_in = 0.5 * self.inlet.static.D * self.inlet.c**2
            self.Cp = (self.out.static.P - self.inlet.static.P) / q_in
            
            # Check for choking
            if self.out.m_abs >= 0.99:
                self.choke_flag = True
        
        except Exception as e:
            print(f"Vaneless diffuser calculation failed: {e}")
            self.choke_flag = True


@dataclass
class VanedDiffuser:
    """
    Vaned diffuser component (optional)
    """
    geom: Geometry
    op: OperatingCondition
    imp: Any  # Impeller object
    loss_config: dict = field(default_factory=dict)
    
    # States
    inlet: DiffuserState = field(default_factory=DiffuserState)
    out: DiffuserState = field(default_factory=DiffuserState)
    
    # Performance
    dh0s: float = 0.0
    eff: float = 0.0
    choke_flag: bool = False
    
    def __post_init__(self):
        """Initialize and calculate vaned diffuser"""
        self.inlet.total = self.imp.out.total
        self. inlet.static = self.imp. out.static
        self.inlet.c = self.imp.out.c
        self.calculate()
    
    def calculate(self):
        """Calculate vaned diffuser performance"""
        # Simplified vaned diffuser model
        # In practice, this would be more complex
        
        # Assume guided flow to outlet angle
        # Typical vaned diffuser provides better pressure recovery
        # but adds complexity and potential for separation
        
        # For now, use vaneless diffuser model with better recovery
        pass


def surge_critical_angle(r5:  float, r4: float, b4: float, Ma4: float) -> float:
    """
    Calculate critical angle for surge onset
    Based on empirical correlations
    
    Parameters
    ----------
    r5 :  float
        Diffuser outlet radius
    r4 : float
        Impeller outlet radius
    b4 : float
        Impeller outlet width
    Ma4 : float
        Mach number at impeller outlet
    
    Returns
    -------
    float
        Critical angle (degrees) above which surge occurs
    """
    # Empirical correlation (needs calibration)
    radius_ratio = r5 / r4
    b_r_ratio = b4 / r4
    
    # Critical angle increases with radius ratio
    alpha_crit = 70.0 + 10.0 * (radius_ratio - 1.5) - 5.0 * b_r_ratio
    
    # Mach number effect
    alpha_crit -= 5.0 * (Ma4 - 0.8)
    
    return np.clip(alpha_crit, 60.0, 85.0)