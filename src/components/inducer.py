"""
Inducer component - Based on RadComp with enhancements
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import optimize

from geometry. geometry import Geometry
from conditions.operating import (
    OperatingCondition, ThermoProp,
    static_from_total, total_from_static, thermo_prop
)


@dataclass
class InducerState:
    """State at a station in the inducer"""
    total:  Optional[ThermoProp] = None
    static: Optional[ThermoProp] = None
    isentropic: Optional[ThermoProp] = None
    
    # Velocity
    c: float = 0.0          # Absolute velocity
    m_abs: float = 0.0      # Absolute Mach number
    A_eff: float = 1.0      # Effective flow area factor


@dataclass
class Inducer:
    """
    Inducer component
    Handles inlet flow from station 1 to station 2
    """
    geom:  Geometry
    op:  OperatingCondition
    
    # States
    in1: InducerState = field(default_factory=InducerState)
    out:  InducerState = field(default_factory=InducerState)
    
    # Performance metrics
    dh0s: float = 0.0
    eff: float = 0.0
    choke_flag: bool = False
    heat:  float = 0.0  # Heat addition (if any)
    
    def __post_init__(self):
        """Initialize and calculate inducer"""
        self.in1.total = self.op.in0
        self.calculate()
    
    def calculate(self):
        """Main calculation routine for inducer"""
        geom = self.geom
        op = self.op
        
        # Inlet conditions (Station 1)
        in_total = self.in1.total
        
        # Initial guesses for outlet
        c2_guess = 50.0  # m/s - typical inlet velocity
        Pout_guess = in_total.P * 0.98  # Slight pressure drop
        
        def resolve_out(x):
            """
            Residual function for inducer outlet conditions
            Solves continuity and energy equations
            """
            c2, Pout = x
            
            # Calculate static state from velocity
            h_static = in_total.H + self.heat / op.m - 0.5 * c2**2
            
            try:
                # Get static properties
                tp_static = thermo_prop(op.fld, "PH", Pout, h_static)
                
                # Continuity equation:  m = rho * V * A_eff
                mass_calc = tp_static.D * c2 * geom.A2_eff
                err_mass = (mass_calc - op.m) / op.m
                
                # Pressure constraint (should be slightly lower than inlet)
                # Allow small losses
                err_pressure = (Pout - in_total.P * 0.95) / in_total.P
                
                return [err_mass, err_pressure]
            
            except Exception:
                # Return large error if thermodynamic calculation fails
                return [100.0, 100.0]
        
        # Solve for outlet conditions
        try:
            sol = optimize.root(resolve_out, x0=[c2_guess, Pout_guess], tol=1e-4)
            
            if not sol.success or (np.abs(sol.fun) > 0.01).any():
                self.choke_flag = True
                return
            
            c2, Pout = sol.x
            
            # Calculate outlet state
            self.out.total = thermo_prop(
                op.fld, "PH", Pout, in_total.H + self.heat / op.m
            )
            self.out.isentropic = thermo_prop(
                op.fld, "PS", Pout, in_total.S
            )
            self.out.c = c2
            self.out.static = static_from_total(self.out.total, c2)
            self.out.m_abs = c2 / self. out.static.A
            self.out.A_eff = geom.A2_eff
            
            # Performance metrics
            self.dh0s = self.out.isentropic.H - self.in1.total.H
            delta_h = self.out.total.H - self.in1.total.H
            
            if abs(delta_h) <= 1e-6:
                self.eff = np.inf if self. dh0s > 0 else -np.inf
            else:
                self.eff = self. dh0s / delta_h
            
            # Check for choking
            if self.out.m_abs >= 0.99:
                self.choke_flag = True
        
        except Exception as e: 
            print(f"Inducer calculation failed: {e}")
            self.choke_flag = True