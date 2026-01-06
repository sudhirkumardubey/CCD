"""
Operating conditions module based on RadComp approach
Direct replication of RadComp's thermo. py implementation
"""

from dataclasses import dataclass
import CoolProp as CP
from typing import Optional, Union
import numpy as np

# Optional: use AbstractState for robust HS flashes when PropsSI fails
try:
    from CoolProp import AbstractState
except Exception:  # pragma: no cover
    AbstractState = None

# Mapping constants to mirror RadComp's thermolibs/coolprop.py
cp_inputs = {
    "PT": (CP.iP, CP.iT),
    "HS": (CP.iHmass, CP.iSmass),
    "PH": (CP.iP, CP.iHmass),
    "PS": (CP.iP, CP.iSmass),
    "TQ": (CP.iT, CP.iQ),
    "PQ": (CP.iP, CP.iQ),
}

cp_outputs = {"P": CP.iP, "T": CP.iT, "D": CP.iDmass, "H": CP.iHmass, "S": CP.iSmass}

cp_phases = {
    CP.iphase_gas: "gas",
    CP.iphase_twophase: "twophase",
    CP.iphase_supercritical: "supercritical",
    CP.iphase_supercritical_gas: "supercritical_gas",
}


@dataclass
class ThermoProp:
    """
    Thermodynamic properties at a point (defaults allow empty construction).
    """

    P: float = float("nan")  # Pressure (Pa)
    T: float = float("nan")  # Temperature (K)
    D: float = float("nan")  # Density (kg/m³)
    H: float = float("nan")  # Enthalpy (J/kg)
    S: float = float("nan")  # Entropy (J/kg-K)
    A: float = float("nan")  # Speed of sound (m/s)
    V: Optional[float] = None  # Dynamic viscosity (Pa·s) to match RadComp's V usage
    mu: Optional[float] = None  # Dynamic viscosity (Pa·s) duplicate for compatibility
    cp: Optional[float] = None  # Specific heat at constant pressure (J/kg-K)
    cv: Optional[float] = None  # Specific heat at constant volume (J/kg-K)
    gamma: Optional[float] = None  # Specific heat ratio
    phase: Optional[str] = None  # Phase descriptor
    fld: Optional[Union["CoolPropFluid", str]] = None  # Fluid reference (object or name)


class CoolPropFluid:
    """CoolProp fluid wrapper replicating RadComp thermolibs.coolprop behavior."""
    
    def __init__(self, fluid_name: str):
        """Initialize CoolProp fluid with HEOS backend."""
        self.name = fluid_name
        if AbstractState is None:
            raise ImportError("CoolProp AbstractState not available")
        self.state = AbstractState("HEOS", self.name.upper())

    def __getstate__(self):
        return {"name": self.name}

    def __setstate__(self, state):
        self.name = state.get("name")
        if AbstractState is None:
            raise ImportError("CoolProp AbstractState not available")
        self.state = AbstractState("HEOS", self.name.upper())
        
    def thermo_prop(self, input_pair: Union[str, int], prop1: float, prop2: float) -> ThermoProp:
        """Get thermodynamic properties using CoolProp AbstractState (RadComp parity)."""
        prop1 = float(np.asarray(prop1))
        prop2 = float(np.asarray(prop2))

        try:
            # Scalars only; guard against numpy arrays
            prop1_s = float(np.asarray(prop1).reshape(()))
            prop2_s = float(np.asarray(prop2).reshape(()))

            if isinstance(input_pair, str):
                i1, i2 = cp_inputs[input_pair]
                update_pair = CP.CoolProp.generate_update_pair(i1, prop1_s, i2, prop2_s)
            else:
                update_pair = input_pair

            self.state.update(*update_pair)

            phase_code = self.state.phase()
            if phase_code not in cp_phases:
                raise RuntimeError("Not gas or two-phase")
            phase = cp_phases[phase_code]

            d = {k: self.state.keyed_output(v) for k, v in cp_outputs.items()}
            if phase_code == CP.iphase_twophase:
                A = self.state.saturated_vapor_keyed_output(CP.ispeed_sound)
                V = self.state.saturated_vapor_keyed_output(CP.iviscosity)
            else:
                A = self.state.keyed_output(CP.ispeed_sound)
                V = self.state.keyed_output(CP.iviscosity)

            return ThermoProp(
                P=d["P"],
                T=d["T"],
                D=d["D"],
                H=d["H"],
                S=d["S"],
                A=A,
                V=V,
                mu=V,
                phase=phase,
                fld=self,
            )

        except Exception as e:
            raise RuntimeError(
                f"CoolProp error for {self.name}: {e}\n"
                f"Input:  {input_pair}, values: {prop1}, {prop2}"
            )


@dataclass
class OperatingCondition:
    """Operating condition definition matching RadComp's structure."""
    in0: ThermoProp                  # Inlet thermodynamic state
    fld: Union[CoolPropFluid, str]   # Fluid (object or name string)
    m: float                         # Mass flow rate (kg/s)
    n_rot: float                     # Rotational speed (rad/s)
    
    def __post_init__(self):
        """Validate operating condition"""
        if self.m <= 0:
            raise ValueError("Mass flow rate must be positive")
        if self.n_rot < 0:
            raise ValueError("Rotational speed cannot be negative")


# Convenience functions to match RadComp's API
def thermo_prop(fld: Union[CoolPropFluid, str], input_pair: str, prop1: float, prop2: float) -> ThermoProp:
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
    if hasattr(fld, "thermo_prop"):
        fluid = fld
    else:
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
    
    vel = float(np.asarray(velocity).reshape(()))
    h_static = total.H - 0.5 * vel**2
    s_static = total.S  # Isentropic assumption

    try:
        return thermo_prop(fluid, "HS", h_static, s_static)
    except Exception:
        # Fallback: use AbstractState HS flash (handles two-phase bracketing better)
        if AbstractState is None:
            raise
        fluid_name = getattr(fluid, "name", fluid) or getattr(total.fld, "name", total.fld) or "Air"
        try:
            backend = fluid_name if isinstance(fluid_name, str) and "::" in fluid_name else f"HEOS::{fluid_name}"
            AS = AbstractState(backend.split("::")[0], backend.split("::")[-1])
            AS.update(CP.HmassSmass_INPUTS, float(h_static), float(s_static))
            P = AS.p()
            T = AS.T()
            D = AS.rhomass()
            A = AS.speed_sound()
            mu = AS.viscosity()
            V = mu if mu is not None else None
            cp_val = AS.cpmass()
            cv_val = AS.cvmass()
            gamma = cp_val / cv_val if cv_val else None
            return ThermoProp(P=P, T=T, D=D, H=h_static, S=s_static, A=A, V=V, mu=mu, cp=cp_val, cv=cv_val, gamma=gamma, fld=fluid or total.fld)
        except Exception:
            raise


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

    vel = float(np.asarray(velocity).reshape(()))
    h_total = static.H + 0.5 * vel**2
    s_total = static.S

    try:
        return thermo_prop(fluid, "HS", h_total, s_total)
    except Exception:
        # Fallback: use AbstractState HS flash (handles two-phase bracketing better)
        if AbstractState is None:
            raise
        fluid_name = getattr(fluid, "name", fluid) or getattr(static.fld, "name", static.fld) or "Air"
        try:
            backend = fluid_name if isinstance(fluid_name, str) and "::" in fluid_name else f"HEOS::{fluid_name}"
            AS = AbstractState(backend.split("::")[0], backend.split("::")[-1])
            AS.update(CP.HmassSmass_INPUTS, float(h_total), float(s_total))
            P = AS.p()
            T = AS.T()
            D = AS.rhomass()
            A = AS.speed_sound()
            mu = AS.viscosity()
            V = mu if mu is not None else None
            cp_val = AS.cpmass()
            cv_val = AS.cvmass()
            gamma = cp_val / cv_val if cv_val else None
            return ThermoProp(P=P, T=T, D=D, H=h_total, S=s_total, A=A, V=V, mu=mu, cp=cp_val, cv=cv_val, gamma=gamma, fld=fluid or static.fld)
        except Exception:
            raise
