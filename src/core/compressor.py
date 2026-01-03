"""
Complete centrifugal compressor assembly
Combines all components following RadComp structure
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

from geometry.geometry import Geometry
from conditions.operating import OperatingCondition, thermo_prop
from components.inducer import Inducer
from components.impeller import Impeller
from components.diffuser import VanelessDiffuser, surge_critical_angle


@dataclass
class CompressorResults:
    """Results from compressor calculation"""
    # Overall performance
    PR: float = 0.0              # Pressure ratio
    eff: float = 0.0             # Isentropic efficiency
    power: float = 0.0           # Power consumption (W)
    dh0s: float = 0.0            # Isentropic enthalpy rise (J/kg)
    head: float = 0.0            # Specific head coefficient
    
    # Non-dimensional parameters
    Ns: float = 0.0              # Specific speed
    Ds: float = 0.0              # Specific diameter
    flow_coeff: float = 0.0      # Flow coefficient
    work_coeff: float = 0.0      # Work coefficient
    
    # Inlet conditions
    m_in: float = 0.0            # Inlet Mach number
    tip_speed: float = 0.0       # Tip speed (m/s)
    n_rot_corr: float = 0.0      # Corrected rotational speed
    
    # Stability
    d_head_d_flow: float = 0.0   # Slope of head-flow curve
    surge_margin: float = 0.0     # Surge margin
    
    # Validity flags
    valid: bool = False
    choke:  bool = False
    surge: bool = False
    wet:  bool = False


@dataclass
class CentrifugalCompressor:
    """
    Complete centrifugal compressor
    Integrates inducer, impeller, and diffuser
    """
    geom:  Geometry
    op: OperatingCondition
    loss_config: Dict = field(default_factory=dict)
    
    # Components
    ind: Optional[Inducer] = None
    imp: Optional[Impeller] = None
    dif:  Optional[VanelessDiffuser] = None
    
    # Overall states
    inlet: Optional[any] = None
    outlet: Optional[any] = None
    
    # Results
    results: CompressorResults = field(default_factory=CompressorResults)
    
    # Flags
    invalid_flag: bool = False
    
    def calculate(self, delta_check:  bool = True) -> bool:
        """
        Calculate complete compressor performance
        
        Parameters
        ----------
        delta_check : bool
            If True, check surge stability by perturbing flow
        
        Returns
        -------
        bool
            True if calculation successful, False otherwise
        """
        # Calculate non-dimensional parameters
        self.results.tip_speed = self.geom.r4 * self.op.n_rot
        self.results.n_rot_corr = self.results.tip_speed / self.op.in0.A
        V_in = self.op.m / self.op.in0.D
        self.results.flow_coeff = V_in / (self.results.tip_speed * self.geom.r4**2)
        
        # Step 1: Calculate inducer
        self.ind = Inducer(self.geom, self.op)
        if self.ind. choke_flag:
            self. invalid_flag = True
            self.results.choke = True
            return False
        
        self.inlet = self.ind.in1
        self.results.m_in = self.ind.out.c / self.inlet.total.A
        
        # Step 2: Calculate impeller
        self.imp = Impeller(
            geom=self.geom,
            op=self.op,
            ind=self.ind,
            loss_config=self.loss_config. get("impeller", {})
        )
        
        if self.imp.choke_flag:
            self.invalid_flag = True
            self. results.choke = True
            return False
        
        if self.imp.wet:
            self.invalid_flag = True
            self. results.wet = True
            return False
        
        # Step 3: Check for surge at impeller exit
        alpha_crit = surge_critical_angle(
            self.geom.r5, self.geom.r4, self.geom.b4, self.imp.out.m_abs
        )
        
        if self.imp.out.alpha > alpha_crit:
            self.invalid_flag = True
            self.results.surge = True
            return False
        
        # Step 4: Calculate diffuser
        self.dif = VanelessDiffuser(
            geom=self.geom,
            op=self.op,
            imp=self.imp,
            loss_config=self.loss_config.get("diffuser", {})
        )
        
        if self.dif.choke_flag:
            self. invalid_flag = True
            self.results.choke = True
            return False
        
        # Set outlet
        self.outlet = self. dif.out
        
        # Step 5: Calculate overall performance
        dh = self.outlet.total.H - self.inlet.total.H
        PR = self.outlet.total.P / self.inlet.total.P
        
        if dh < 0 or PR < 1.0:
            self.invalid_flag = True
            return False
        
        # Isentropic state
        tp_is = thermo_prop(
            self.op.fld, "PS", self.outlet.total.P, self.inlet.total.S
        )
        self.results. dh0s = tp_is. H - self.inlet.total.H
        self.results.head = self.results.dh0s / (self.results.tip_speed**2)
        
        # Step 6: Surge stability check
        if delta_check: 
            # Slightly increase mass flow
            from copy import deepcopy
            d_op = deepcopy(self.op)
            d_op.m *= 1.005
            d_comp = CentrifugalCompressor(self.geom, d_op, self.loss_config)
            
            if d_comp.calculate(delta_check=False):
                # Calculate slope dHead/dFlow
                self.results.d_head_d_flow = (
                    (d_comp.results.head - self.results.head) / 
                    (d_comp.results.flow_coeff - self. results.flow_coeff)
                )
                
                # Should be negative for stable operation
                if self.results. d_head_d_flow > -1e-4:
                    self.invalid_flag = True
                    self. results.surge = True
                    return False
        
        # Step 7: Calculate final metrics
        self.results. eff = self.results.dh0s / dh
        self.results.PR = PR
        self.results. power = self.op.m * dh
        
        # Non-dimensional parameters
        sqrt_v_in = V_in**0.5
        self.results.Ns = self.op.n_rot * sqrt_v_in / (self.results. dh0s**0.75)
        self.results. Ds = 2 * self.geom.r4 * self.results. dh0s**0.25 / sqrt_v_in
        
        # Work coefficient
        self.results.work_coeff = dh / self.results.tip_speed**2
        
        self.results.valid = True
        return True
    
    def get_summary(self) -> Dict:
        """Get summary of compressor performance"""
        if not self.results.valid:
            return {
                "valid": False,
                "error": "Calculation failed",
                "choke": self.results.choke,
                "surge": self. results.surge,
                "wet": self.results.wet,
            }
        
        return {
            "valid": True,
            "PR": self.results.PR,
            "efficiency": self.results.eff,
            "power_kW": self.results.power / 1000,
            "head":  self.results.head,
            "tip_speed": self.results.tip_speed,
            "flow_coefficient": self.results.flow_coeff,
            "work_coefficient": self.results.work_coeff,
            "Ns":  self.results.Ns,
            "Ds": self.results.Ds,
            "surge_stable": self.results. d_head_d_flow < 0,
        }