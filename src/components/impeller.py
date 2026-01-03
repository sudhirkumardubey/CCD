"""
Impeller component - Hybrid RadComp + TurboFlow approach
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
from math import pi, sin, cos, tan, atan, radians, degrees
from scipy import optimize

from geometry. geometry import Geometry
from conditions.operating import (
    OperatingCondition, ThermoProp, 
    static_from_total, total_from_static
)
from losses.registry import LossModelRegistry, LossContext


@dataclass
class ImpellerState:
    """State at a station in the impeller"""
    total:  Optional[ThermoProp] = None
    static: Optional[ThermoProp] = None
    relative:  Optional[ThermoProp] = None
    isentropic: Optional[ThermoProp] = None
    
    # Velocities
    c:  float = 0.0      # Absolute velocity
    w: float = 0.0      # Relative velocity
    c_m: float = 0.0    # Meridional component (absolute)
    c_t: float = 0.0    # Tangential component (absolute)
    w_m: float = 0.0    # Meridional component (relative)
    w_t: float = 0.0    # Tangential component (relative)
    
    # Angles
    alpha: float = 0.0  # Absolute flow angle
    beta: float = 0.0   # Relative flow angle
    
    # Mach numbers
    m_abs: float = 0.0  # Absolute Mach number
    m_rel: float = 0.0  # Relative Mach number


@dataclass
class ImpellerLosses:
    """Impeller losses"""
    incidence: float = 0.0
    skin_friction: float = 0.0
    blade_loading: float = 0.0
    clearance: float = 0.0
    disc_friction: float = 0.0
    recirculation: float = 0.0
    total: float = 0.0
    
    def compute_total(self):
        """Compute total losses"""
        self.total = (
            self.incidence + 
            self.skin_friction + 
            self.blade_loading + 
            self.clearance + 
            self.disc_friction + 
            self.recirculation
        )
        return self.total


@dataclass
class Impeller:
    """
    Impeller component with hybrid calculation approach
    """
    geom: Geometry
    op:  OperatingCondition
    ind: Any  # Inducer object
    loss_config: Dict  # Loss model configuration
    
    # States
    in2: ImpellerState = field(default_factory=ImpellerState)
    out:  ImpellerState = field(default_factory=ImpellerState)
    losses: ImpellerLosses = field(default_factory=ImpellerLosses)
    
    # Performance metrics
    dh0s: float = 0.0
    eff: float = 0.0
    choke_flag: bool = False
    wet:  bool = False
    
    def __post_init__(self):
        """Initialize and calculate impeller"""
        self.in2.total = self.ind.out.total
        self.in2.c = self.ind.out.c
        self.calculate()
    
    def calculate(self):
        """
        Main calculation routine (RadComp style with TurboFlow losses)
        """
        # Step 1: Calculate inlet velocity triangle
        self._calculate_inlet_triangle()
        
        if self.choke_flag or self.wet:
            return
        
        # Step 2:  Solve discharge triangle iteratively
        self._solve_discharge_triangle()
        
        if self.choke_flag:
            return
        
        # Step 3: Calculate losses using TurboFlow registry
        self._calculate_losses()
        
        # Step 4: Calculate performance metrics
        self._calculate_performance()
    
    def _calculate_inlet_triangle(self):
        """Calculate velocity triangle at impeller inlet (Station 2)"""
        geom = self.geom
        op = self.op
        
        # Absolute velocity components
        c2 = self.in2.c
        c2_theta = c2 * sin(radians(geom.alpha2))
        c2_m = c2 * cos(radians(geom.alpha2))
        
        # Relative velocity at shroud
        w2t_s = geom.r2s * op.n_rot - c2_theta
        beta2_fs = -degrees(atan(w2t_s / c2_m))
        w2_s = c2_m / cos(radians(beta2_fs))
        
        # Relative velocity at RMS radius
        w2t = geom.r2rms * op.n_rot - c2_theta
        beta2_f = -degrees(atan(w2t / c2_m))
        w2 = c2_m / cos(radians(beta2_f))
        
        # Store values
        self.in2.w = w2
        self.in2.beta = beta2_f
        self.in2.c_m = c2_m
        self.in2.c_t = c2_theta
        self.in2.w_m = c2_m
        self.in2.w_t = w2t
        
        # Calculate thermodynamic state
        try:
            self.in2.static = static_from_total(self.in2.total, c2)
            self.in2.relative = total_from_static(self.in2.static, w2)
        except Exception: 
            self.wet = True
            return
        
        # Mach numbers
        self.in2.m_abs = c2 / self.in2.static.A
        self.in2.m_rel = w2 / self.in2.static.A
        
        # Check for choking
        if self.in2.m_rel >= 0.99:
            self.choke_flag = True
    
    def _solve_discharge_triangle(self):
        """Solve discharge velocity triangle iteratively"""
        geom = self.geom
        op = self.op
        
        # Initial guesses
        beta4_f_guess = geom.beta4  # Start with blade angle
        P4_guess = self.in2.total.P * 1.5  # Assume some pressure rise
        
        def residuals(x):
            beta4_f, P4 = x
            err = []
            
            # Calculate effective area
            A4_total = geom.A4_eff
            
            # Solve for w4 (relative velocity)
            w4 = op.m / (A4_total * self.in2.static.D) / cos(radians(beta4_f))
            
            # Calculate absolute velocity components
            c4m = op.m / A4_total / self.in2.static.D
            c4t = c4m * tan(radians(geom.beta4)) + geom.slip * (geom.r4 * op.n_rot)
            
            # Relative tangential component
            w4t = (geom.r4 * op.n_rot) - c4t
            
            # New relative velocity
            w4_new = np.sqrt(w4t**2 + c4m**2)
            beta4_f_new = -degrees(np.arcsin(w4t / w4_new))
            
            # Error in beta
            err.append((beta4_f_new - beta4_f) / 60.0)
            
            # Calculate absolute velocity
            c4 = np.sqrt(c4t**2 + c4m**2)
            alpha4 = degrees(atan(c4t / c4m))
            
            # Thermodynamic state
            try: 
                tp4_tot = total_from_static(self.in2.static, c4)  # Simplified
                out_H = tp4_tot. H - self.in2.total.H
                
                # Calculate losses (simplified for iteration)
                # Full loss calculation happens after convergence
                dh_losses_est = 0.1 * out_H  # Rough estimate
                
                # Energy balance
                work_input = (geom.r4 * op. n_rot) * c4t
                H4_expected = self.in2.total.H + work_input - dh_losses_est
                
                # Error in enthalpy
                err.append((tp4_tot.H - H4_expected) / H4_expected)
                
            except Exception:
                err = [100.0, 100.0]  # Large error if calculation fails
            
            return err
        
        # Solve system
        try:
            sol = optimize.root(residuals, x0=[beta4_f_guess, P4_guess], tol=1e-4)
            
            if not sol.success or (np.abs(sol.fun) > 0.01).any():
                self.choke_flag = True
                return
            
            beta4_f, P4 = sol.x
            
            # Store solution
            self.out.beta = beta4_f
            
            # Calculate final discharge state
            A4_total = geom.A4_eff
            w4 = op.m / (A4_total * self.in2.static.D) / cos(radians(beta4_f))
            c4m = op.m / A4_total / self.in2.static.D
            c4t = c4m * tan(radians(geom.beta4)) + geom.slip * (geom.r4 * op.n_rot)
            c4 = np.sqrt(c4t**2 + c4m**2)
            alpha4 = degrees(atan(c4t / c4m))
            
            # Store velocities
            self.out.c = c4
            self.out.w = w4
            self.out.c_m = c4m
            self.out. c_t = c4t
            self.out.alpha = alpha4
            self.out.w_m = c4m
            self.out. w_t = (geom.r4 * op.n_rot) - c4t
            
            # Thermodynamic state
            self.out.static = static_from_total(self.in2.relative, w4)
            self.out. total = total_from_static(self.out.static, c4)
            self.out.relative = total_from_static(self.out.static, w4)
            self.out.isentropic = ThermoProp. from_coolprop(
                op.fld, "PS", self.out.total.P, self.in2.static.S
            )
            
            # Mach numbers
            self.out.m_abs = c4 / self.out.static.A
            self.out.m_rel = w4 / self. out.static.A
            
            # Check for choking
            if self.out.m_rel >= 0.99 or self.out.m_abs >= 0.99:
                self.choke_flag = True
                return
            
            if self.out.total.P < self.in2.total.P:
                self.choke_flag = True
                
        except Exception as e:
            print(f"Discharge triangle solution failed: {e}")
            self.choke_flag = True
    
    def _calculate_losses(self):
        """Calculate all losses using TurboFlow registry"""
        # Diffusion factor (for several loss correlations)
        out_H = self.out.total.H - self.in2.total.H
        Df = 1 - self.out.w / self.in2.w + (out_H / (self.geom.r4 * self. op.n_rot)**2)
        
        # Create loss context
        velocity_triangle = {
            "w2": self.in2.w,
            "w4": self.out.w,
            "v_t4": self.out.c_t,
            "v_m4": self.out.c_m,
            "alpha4": self.out.alpha,
            "diffusion_factor": Df,
            "flow_coefficient": self. out.c_m / (self.geom.r4 * self.op.n_rot),
        }
        
        context = LossContext(
            component="impeller",
            geometry=self.geom,
            operating_condition=self.op,
            inlet_state=self.in2.static,
            outlet_state=self.out.static,
            velocity_triangle=velocity_triangle
        )
        
        # Calculate each loss using registered models
        self.losses.skin_friction = LossModelRegistry. get_model(
            "impeller", self.loss_config. get("skin_friction", "jansen_skin_friction")
        )(context)
        
        self.losses.blade_loading = LossModelRegistry.get_model(
            "impeller", self. loss_config.get("blade_loading", "rodgers_blade_loading")
        )(context)
        
        self.losses.clearance = LossModelRegistry.get_model(
            "impeller", self. loss_config.get("clearance", "jansen_clearance")
        )(context)
        
        self.losses.disc_friction = LossModelRegistry.get_model(
            "impeller", self.loss_config.get("disc_friction", "daily_nece_disc_friction")
        )(context)
        
        self.losses.recirculation = LossModelRegistry.get_model(
            "impeller", self. loss_config.get("recirculation", "rodgers_recirculation")
        )(context)
        
        # Compute total losses
        self.losses. compute_total()
    
    def _calculate_performance(self):
        """Calculate performance metrics"""
        self.dh0s = self. out.isentropic.H - self. in2.total.H
        out_H = self.out.total.H - self.in2.total.H
        self. eff = self. dh0s / out_H if out_H > 0 else 0.0