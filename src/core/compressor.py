"""
Complete centrifugal compressor assembly (RadComp-aligned) with pluggable loss models.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from geometry.geometry import Geometry
from conditions.operating import OperatingCondition, thermo_prop
from components.inducer import Inducer
from components.impeller import Impeller
from components.diffuser import VanelessDiffuser, VanedDiffuser, surge_critical_angle


@dataclass
class CompressorResults:
    PR: float = math.nan
    eff: float = math.nan
    power: float = math.nan
    dh0s: float = math.nan
    head: float = math.nan
    Ns: float = math.nan
    Ds: float = math.nan
    flow_coeff: float = math.nan
    work_coeff: float = math.nan
    m_in: float = math.nan
    tip_speed: float = math.nan
    n_rot_corr: float = math.nan
    d_head_d_flow: float = math.nan
    valid: bool = False
    choke: bool = False
    surge: bool = False
    wet: bool = False


@dataclass
class CentrifugalCompressor:
    geom: Geometry
    op: OperatingCondition
    loss_config: Dict[str, Any] = field(default_factory=dict)

    ind: Optional[Inducer] = None
    imp: Optional[Impeller] = None
    dif: Optional[Any] = None

    inlet: Optional[Any] = None
    outlet: Optional[Any] = None

    results: CompressorResults = field(default_factory=CompressorResults)
    invalid_flag: bool = False

    def calculate(self, delta_check: bool = True) -> bool:
        self.results.tip_speed = self.geom.r4 * self.op.n_rot
        self.results.n_rot_corr = self.results.tip_speed / self.op.in0.A
        V_in = self.op.m / self.op.in0.D
        self.results.flow_coeff = V_in / (self.results.tip_speed * self.geom.r4**2)

        # Inducer
        self.ind = Inducer(self.geom, self.op)
        if self.ind.choke_flag:
            self.invalid_flag = True
            self.results.choke = True
            return False

        self.inlet = self.ind.in1
        self.results.m_in = self.ind.out.c / self.inlet.total.A

        # Impeller
        self.imp = Impeller(
            geom=self.geom,
            op=self.op,
            ind=self.ind,
            loss_config=self.loss_config.get("impeller"),
        )
        if self.imp.choke_flag:
            self.invalid_flag = True
            self.results.choke = True
            return False
        if self.imp.wet:
            self.invalid_flag = True
            self.results.wet = True
            return False

        # Surge check at impeller exit
        alpha_crit = surge_critical_angle(
            self.geom.r5, self.geom.r4, self.geom.b4, self.imp.out.m_abs
        )
        if self.imp.out.alpha > alpha_crit:
            self.invalid_flag = True
            self.results.surge = True
            return False

        # Diffuser selection (vaneless by default)
        dif_type = self.loss_config.get("diffuser_type", "vaneless")
        dif_cfg = self.loss_config.get("diffuser", {})
        if dif_type == "vaned":
            self.dif = VanedDiffuser(self.geom, self.op, self.imp, loss_config=dif_cfg)
        else:
            self.dif = VanelessDiffuser(self.geom, self.op, self.imp, loss_config=dif_cfg)

        if self.dif.choke_flag:
            self.invalid_flag = True
            self.results.choke = True
            return False

        self.outlet = self.dif.out

        dh = self.outlet.total.H - self.inlet.total.H
        PR = self.outlet.total.P / self.inlet.total.P
        if dh < 0 or PR < 1.0:
            self.invalid_flag = True
            return False

        tp_is = thermo_prop(self.op.fld, "PS", self.outlet.total.P, self.inlet.total.S)
        self.results.dh0s = tp_is.H - self.inlet.total.H
        self.results.head = self.results.dh0s / (self.results.tip_speed**2)

        if delta_check:
            from copy import deepcopy

            d_op = deepcopy(self.op)
            d_op.m *= 1.005
            d_comp = CentrifugalCompressor(self.geom, d_op, self.loss_config)
            if d_comp.calculate(delta_check=False):
                self.results.d_head_d_flow = (
                    (d_comp.results.head - self.results.head)
                    / (d_comp.results.flow_coeff - self.results.flow_coeff)
                )
                if self.results.d_head_d_flow > -1e-4:
                    self.invalid_flag = True
                    self.results.surge = True
                    return False

        self.results.eff = self.results.dh0s / dh
        self.results.PR = PR
        self.results.power = self.op.m * dh

        sqrt_v_in = V_in**0.5
        self.results.Ns = self.op.n_rot * sqrt_v_in / (self.results.dh0s**0.75)
        self.results.Ds = 2 * self.geom.r4 * self.results.dh0s**0.25 / sqrt_v_in
        self.results.work_coeff = dh / self.results.tip_speed**2

        self.results.valid = True
        return True

    def get_summary(self) -> Dict:
        if not self.results.valid:
            return {
                "valid": False,
                "error": "Calculation failed",
                "choke": self.results.choke,
                "surge": self.results.surge,
                "wet": self.results.wet,
            }

        return {
            "valid": True,
            "PR": self.results.PR,
            "efficiency": self.results.eff,
            "power_kW": self.results.power / 1000,
            "head": self.results.head,
            "tip_speed": self.results.tip_speed,
            "flow_coefficient": self.results.flow_coeff,
            "work_coefficient": self.results.work_coeff,
            "Ns": self.results.Ns,
            "Ds": self.results.Ds,
            "surge_stable": self.results.d_head_d_flow < 0,
            # Additional kinematic/performance metrics for constraints
            "M2s_rel": getattr(self.imp.in2, "m_rels", math.nan) if self.imp else math.nan,
            "M2_rel": getattr(self.imp.in2, "m_rel", math.nan) if self.imp else math.nan,
            "W4_over_W2s": (
                (self.imp.out.w / self.imp.in2.ws)
                if self.imp
                and hasattr(self.imp.in2, "ws")
                and self.imp.in2.ws is not None
                and math.isfinite(self.imp.in2.ws)
                and self.imp.in2.ws != 0.0
                else math.nan
            ),
            "alpha4_exit": getattr(self.imp.out, "alpha", math.nan) if self.imp else math.nan,
            "U4": self.results.tip_speed,
            "r5_over_r4": (self.geom.r5 / self.geom.r4) if self.geom and self.geom.r4 else math.nan,
        }