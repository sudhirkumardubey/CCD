"""
Operating conditions module based on RadComp approach
Direct replication of RadComp's thermo. py implementation
"""

from dataclasses import dataclass
import CoolProp.CoolProp as CP
from typing import Optional
import numpy as np


@dataclass
class ThermoProp:
    """
    Thermodynamic properties at a point
    Matches RadComp's ThermoProp class structure
    """
    P:  float  # Pressure (Pa)
    T: float  # Temperature (K)
    D: float  # Density (kg/m³)
    H: float  # Enthalpy (J/kg)
    S: float  # Entropy (J/kg-K)
    A: float  # Speed of sound (m/s)
    V: Optional[float] = None  # Kinematic viscosity (m²/s)
    mu: Optional[float] = None  # Dynamic viscosity (Pa·s)
    cp: Optional[float] = None  # Specific heat at constant pressure (J/kg-K)
    cv: Optional[float] = None  # Specific heat at constant volume (J/kg-K)
    gamma: Optional[float] = None  # Specific heat ratio
    fld: Optional[str] = None  # Fluid name


class CoolPropFluid:
    """
    CoolProp fluid wrapper - matches RadComp's implementation
    """
    
    def __init__(self, fluid_name: str):
        """
        Initialize CoolProp fluid
        
        Parameters
        ----------
        fluid_name : str
            Fluid name (e.g., 'Air', 'N2', 'CO2')
        """
        self.name = fluid_name
        
    def thermo_prop(self, input_pair: str, prop1: float, prop2: float) -> ThermoProp:
        """
        Get thermodynamic properties
        Replicates RadComp's thermo_prop method exactly
        
        Parameters
        ----------
        input_pair : str
            Input pair:  'PT', 'PS', 'PH', 'HS', 'DH', etc.
        prop1 : float
            First property value
        prop2 :  float
            Second property value
            
        Returns
        -------
        ThermoProp
            Thermodynamic properties
        """
        try:
            # Get the two input properties based on input_pair
            if input_pair == "PT": 
                P, T = prop1, prop2
                D = CP.PropsSI("D", "P", P, "T", T, self.name)
                H = CP.PropsSI("H", "P", P, "T", T, self.name)
                S = CP.PropsSI("S", "P", P, "T", T, self.name)
                
            elif input_pair == "PS":
                P, S = prop1, prop2
                D = CP.PropsSI("D", "P", P, "S", S, self.name)
                T = CP.PropsSI("T", "P", P, "S", S, self.name)
                H = CP.PropsSI("H", "P", P, "S", S, self.name)
                
            elif input_pair == "PH":
                P, H = prop1, prop2
                D = CP.PropsSI("D", "P", P, "H", H, self.name)
                T = CP.PropsSI("T", "P", P, "H", H, self.name)
                S = CP.PropsSI("S", "P", P, "H", H, self.name)
                
            elif input_pair == "HS":
                H, S = prop1, prop2
                P = CP.PropsSI("P", "H", H, "S", S, self.name)
                T = CP.PropsSI("T", "H", H, "S", S, self.name)
                D = CP.PropsSI("D", "H", H, "S", S, self.name)
                
            elif input_pair == "DH": 
                D, H = prop1, prop2
                P = CP.PropsSI("P", "D", D, "H", H, self.name)
                T = CP.PropsSI("T", "D", D, "H", H, self.name)
                S = CP.PropsSI("S", "D", D, "H", H, self.name)
                
            elif input_pair == "DS":
                D, S = prop1, prop2
                P = CP.PropsSI("P", "D", D, "S", S, self.name)
                T = CP.PropsSI("T", "D", D, "S", S, self.name)
                H = CP.PropsSI("H", "D", D, "S", S, self.name)
                
            else: 
                raise ValueError(f"Input pair '{input_pair}' not supported")
            
            # Get remaining properties using P and T
            A = CP.PropsSI("A", "P", P, "T", T, self.name)
            mu = CP.PropsSI("V", "P", P, "T", T, self.name)  # Dynamic viscosity
            V = mu / D  # Kinematic viscosity
            cp_val = CP.PropsSI("C", "P", P, "T", T, self.name)
            cv_val = CP.PropsSI("O", "P", P, "T", T, self.name)
            gamma = cp_val / cv_val
            
            return ThermoProp(
                P=P, T=T, D=D, H=H, S=S, A=A,
                mu=mu, V=V, cp=cp_val, cv=cv_val, 
                gamma=gamma, fld=self.name
            )
            
        except Exception as e: 
            raise RuntimeError(
                f"CoolProp error for {self.name}: {e}\n"
                f"Input:  {input_pair}, values: {prop1}, {prop2}"
            )


@dataclass
class OperatingCondition:
    """
    Operating condition definition
    Matches RadComp's OperatingCondition structure
    """
    in0: ThermoProp  # Inlet thermodynamic state
    fld: str         # Fluid name (kept for compatibility, but use in0. fld)
    m: float         # Mass flow rate (kg/s)
    n_rot: float     # Rotational speed (rad/s)
    
    def __post_init__(self):
        """Validate operating condition"""
        if self.m <= 0:
            raise ValueError("Mass flow rate must be positive")
        if self.n_rot < 0:
            raise ValueError("Rotational speed cannot be negative")


# Convenience functions to match RadComp's API
def thermo_prop(fld: str, input_pair:  str, prop1: float, prop2: float) -> ThermoProp:
    """
    Convenience function matching RadComp's usage pattern
    
    Parameters
    ----------
    fld : str
        Fluid name
    input_pair : str
        Input pair:  'PT', 'PS', 'PH', 'HS', 'DH'
    prop1 : float
        First property
    prop2 :  float
        Second property
        
    Returns
    -------
    ThermoProp
        Thermodynamic properties
    """
    fluid = CoolPropFluid(fld)
    return fluid.thermo_prop(input_pair, prop1, prop2)


def static_from_total(total: ThermoProp, velocity:  float, fluid: str = None) -> ThermoProp:
    """
    Calculate static properties from total properties
    
    Parameters
    ----------
    total : ThermoProp
        Total properties
    velocity : float
        Flow velocity (m/s)
    fluid : str, optional
        Fluid name (uses total.fld if not provided)
        
    Returns
    -------
    ThermoProp
        Static properties
    """
    if fluid is None:
        fluid = total.fld if total.fld else "Air"
    
    h_static = total.H - 0.5 * velocity**2
    s_static = total.S  # Isentropic assumption
    
    return thermo_prop(fluid, "HS", h_static, s_static)


def total_from_static(static: ThermoProp, velocity: float, fluid: str = None) -> ThermoProp:
    """
    Calculate total properties from static properties
    
    Parameters
    ----------
    static : ThermoProp
        Static properties
    velocity : float
        Flow velocity (m/s)
    fluid : str, optional
        Fluid name (uses static. fld if not provided)
        
    Returns
    -------
    ThermoProp
        Total properties
    """
    if fluid is None:
        fluid = static.fld if static.fld else "Air"
    
    h_total = static.H + 0.5 * velocity**2
    s_total = static.S  # Isentropic assumption
    
    return thermo_prop(fluid, "HS", h_total, s_total)


# Alternative:  Direct instantiation matching RadComp's pattern
def create_fluid(name: str) -> CoolPropFluid:
    """
    Create a CoolProp fluid object
    Matches RadComp's CoolPropFluid instantiation
    
    Parameters
    ----------
    name : str
        Fluid name
        
    Returns
    -------
    CoolPropFluid
        Fluid object with thermo_prop method
    """
    return CoolPropFluid(name)