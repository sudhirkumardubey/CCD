"""Inducer component (kept identical to RadComp reference)."""

import math
from dataclasses import InitVar, dataclass, field, fields
from typing import Type, TypeVar

from scipy import optimize

from geometry.geometry import Geometry
from conditions.operating import OperatingCondition, ThermoProp, static_from_total
from utils import moody


class ThermoException(Exception):
    """Raised when thermo calculations fail (parity with RadComp)."""


State = TypeVar("State", bound="InducerState")


@dataclass
class InducerState:
    total: ThermoProp = field(default_factory=ThermoProp)
    m_abs: float = math.nan
    static: ThermoProp = field(default_factory=ThermoProp)
    isentropic: ThermoProp = field(default_factory=ThermoProp)
    A_eff: float = math.nan
    c: float = math.nan
    alpha: float = math.nan

    @property
    def is_not_set(self):
        return math.isnan(self.total.P)

    @classmethod
    def from_state(cls: Type[State], state: "InducerState") -> State:
        cls_fields = [f.name for f in fields(cls)]
        content = {k: v for k, v in state.__dict__.items() if k in cls_fields}
        return cls(**content)


@dataclass
class Inducer:
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    in1: InducerState = field(init=False)
    out: InducerState = field(default_factory=InducerState)
    dh0s: float = math.nan
    eff: float = math.nan
    choke_flag: bool = False
    heat: float = 0

    def __post_init__(self, geom: Geometry, op: OperatingCondition) -> None:
        self.in1 = InducerState(total=op.in0)
        if self.out.is_not_set:
            try:
                self.calculate(geom, op)
            except ThermoException:
                self.choke_flag = True

    def calculate(self, geom: Geometry, op: OperatingCondition):
        """Calculates the output and the inlet of the inducer (Station 1→2)."""
        in_total = self.in1.total

        def resolve_c1(x):
            # optimizer passes ndarray; convert to scalar for thermo
            c1 = float(x[0]) if hasattr(x, "__len__") else float(x)
            try:
                stat1 = static_from_total(in_total, c1)
                err1 = (op.m - geom.A1_eff * c1 * stat1.D) / op.m
            except ThermoException:
                return 1e3
            else:
                return err1

        def resolve_out(x):
            c2, Pout = x
            try:
                tot2 = op.fld.thermo_prop("PH", Pout, in_total.H + self.heat / op.m)
                stat2 = static_from_total(tot2, c2)

                err2 = (op.m - geom.A2_eff * c2 * stat2.D) / op.m

                Re = c2 * 2 * geom.r2s * stat2.D / stat2.V
                Cf = moody(Re, geom.rug_ind / (2 * geom.r2s))
                dP = 4 * Cf * geom.l_ind * c2**2 / (4 * geom.r2s) * stat2.D
                Pout_calc = in_total.P - dP
                err3 = (Pout_calc - tot2.P) / in_total.P
            except ThermoException:
                return [1e3, 1e3]
            else:
                return [err2, err3]

        c1_guess = op.m / geom.A1_eff / in_total.D
        if c1_guess / in_total.A > 1.5:
            self.choke_flag = True
            return

        sol = optimize.root(resolve_c1, x0=c1_guess)
        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return

        c1 = float(sol.x[0])
        self.in1.c = c1
        self.in1.A_eff = geom.A1_eff
        self.in1.static = static_from_total(in_total, c1)
        self.in1.m_abs = c1 / self.in1.static.A

        # Test for choke
        if self.in1.m_abs * geom.A1_eff / geom.A2_eff >= 0.99:
            self.choke_flag = True
            return

        c2_guess = op.m / geom.A2_eff / self.in1.static.D
        Re_g = c2_guess * 2 * geom.r2s * self.in1.static.D / self.in1.static.V
        Cf_g = moody(Re_g, geom.rug_ind / (2 * geom.r2s))
        dP = 4 * Cf_g * geom.l_ind * c2_guess**2 / (4 * geom.r2s) * self.in1.static.D

        Pout_guess = self.in1.total.P - dP

        sol = optimize.root(resolve_out, x0=[c2_guess, Pout_guess], tol=1e-4)
        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return sol

        c2, Pout = sol.x

        # Assign output state
        self.out = InducerState(
            total=op.fld.thermo_prop("PH", Pout, in_total.H + self.heat / op.m),
            isentropic=op.fld.thermo_prop("PS", Pout, in_total.S),
        )
        self.out.c = c2
        self.out.static = static_from_total(self.out.total, c2)
        self.out.m_abs = c2 / self.out.static.A
        self.out.A_eff = geom.A2_eff

        self.dh0s = self.out.isentropic.H - self.in1.total.H
        delta_h = self.out.total.H - self.in1.total.H
        if abs(delta_h) <= 1e-6:
            self.eff = math.copysign(math.inf, self.dh0s)
        else:
            self.eff = self.dh0s / delta_h