"""
Diffuser components (RadComp-aligned vaneless diffuser) with optional registry losses.
"""

import math
from dataclasses import InitVar, dataclass, field
from math import cos, pi, sin, tan
from typing import Optional, Dict, Tuple

import numpy as np
from numpy.polynomial import polynomial
from scipy import optimize

from geometry.geometry import Geometry
from conditions.operating import OperatingCondition, static_from_total
from components.impeller import Impeller
from components.inducer import InducerState
from losses.registry import LossModelRegistry, LossContext


class ThermoException(Exception):
    """Fallback exception to mirror RadComp API when thermo calc fails."""


class VanelessState(InducerState):
    pass

# Backward-compatibility alias used by legacy imports
DiffuserState = VanelessState


@dataclass
class VanelessDiffuser:
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    imp: InitVar[Impeller]
    in4: VanelessState = field(init=False)
    out: VanelessState = field(default_factory=VanelessState)
    loss: float = math.nan
    dh0s: float = math.nan
    eff: float = math.nan
    choke_flag = False
    n_steps: int = 15
    loss_config: Optional[Dict[str, str]] = None
    registry_losses: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self, geom: Geometry, op: OperatingCondition, imp: Impeller) -> None:
        self.in4 = VanelessState.from_state(imp.out)
        self.calculate(geom, op)
        self._apply_registry_losses(geom, op)

    def calculate(self, geom: Geometry, op: OperatingCondition):
        r = np.linspace(geom.r4, geom.r5, 1 + self.n_steps, endpoint=True)
        dr = np.diff(r)
        b = np.linspace(geom.b4, geom.b5, 1 + self.n_steps, endpoint=True)

        Dh = np.sqrt(8 * r[:-1] * b[1:] * geom.blockage[4])
        A_eff = 2 * r[1:] * b[1:] * pi * geom.blockage[4]

        k = 0.02

        def resolve_speed(x, return_values=False):
            in_ = VanelessState.from_state(self.in4)
            err = []
            for i in range(self.n_steps):
                Re = in_.c * in_.static.D / in_.static.V * b[i + 1]
                Cf = k * (1.8e5 / Re) ** 0.2  # Japikse

                ds = ((dr[i] / tan((90 - in_.alpha) / 180 * pi)) ** 2 + dr[i] ** 2) ** 0.5
                dp0 = 4.0 * Cf * ds * in_.c**2 * in_.static.D / 2 / Dh[i]

                c4t = in_.c * sin(in_.alpha / 180 * pi)
                c4m = in_.c * cos(in_.alpha / 180 * pi)
                dCtdr = (
                    -(c4t / r[i] + Cf * in_.c**2 * sin(in_.alpha / 180 * pi) / c4m / b[i + 1])
                    * dr[i]
                )
                c5t = c4t + dCtdr

                P0 = in_.total.P - dp0
                if P0 <= 0 and P0 < op.in0.P:
                    err.extend((self.n_steps - i) * [1e4])
                    return err
                tot = op.fld.thermo_prop("PH", P0, in_.total.H)

                c5m = float(x[i])
                c5 = (c5m**2 + c5t**2) ** 0.5
                if c5 > 1.25 * in_.total.A:
                    err.extend((self.n_steps - i) * [1e4])
                    return err

                try:
                    stat = static_from_total(tot, c5)
                except ThermoException:
                    err.extend((self.n_steps - i) * [1e4])
                    return err

                err.append((op.m - A_eff[i] * c5m * stat.D) / op.m)

                in_.c = c5
                in_.alpha = math.asin(c5t / c5) * 180 / pi
                in_.total = tot
                in_.static = stat
                in_.m_abs = in_.c * cos(in_.alpha / 180 * pi) / in_.static.A
                if in_.m_abs >= 0.99:
                    err[-1] += in_.m_abs - 0.99
            if return_values:
                return err, in_
            return err

        c4m = self.in4.c * cos(self.in4.alpha / 180 * pi)

        if c4m / self.in4.static.A >= 0.99:
            self.choke_flag = True
            return

        speed_guess = c4m * r[:-1] / r[1:]
        sol = optimize.root(resolve_speed, x0=speed_guess)

        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return

        _, out = resolve_speed(sol.x, return_values=True)
        out.m_abs = out.c * cos(out.alpha / 180 * pi) / out.static.A
        if out.m_abs >= 0.99:
            self.choke_flag = True
        self.out = out

        out_is = op.fld.thermo_prop("PS", out.total.P, self.in4.total.S)
        self.out.isentropic = out_is
        self.loss = out.total.H - out_is.H
        self.dh0s = out_is.H - self.in4.total.H

        delta_h = out.total.H - self.in4.total.H
        if abs(delta_h) <= 1e-6:
            self.eff = math.copysign(math.inf, self.dh0s)
        else:
            self.eff = self.dh0s / delta_h

    def _apply_registry_losses(self, geom: Geometry, op: OperatingCondition) -> None:
        if self.loss_config is None:
            return
        try:
            ctx = LossContext(
                component="diffuser",
                geometry=geom,
                operating_condition=op,
                inlet_state=self.in4,
                outlet_state=self.out,
                velocity_triangle={
                    "c4": self.in4.c,
                    "c5": self.out.c,
                    "alpha4": self.in4.alpha,
                    "alpha5": self.out.alpha,
                },
            )
            losses = LossModelRegistry.calculate_losses(ctx, self.loss_config)
            losses["total"] = sum(losses.values()) if losses else 0.0
            self.registry_losses = losses
        except Exception:
            self.registry_losses = {}


@dataclass
class VanedDiffuser:
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    imp: InitVar[Impeller]
    in4: VanelessState = field(init=False)
    out: VanelessState = field(default_factory=VanelessState)
    loss: float = math.nan
    dh0s: float = math.nan
    eff: float = math.nan
    choke_flag = False
    n_steps: int = 10
    loss_config: Optional[Dict[str, str]] = None
    registry_losses: Dict[str, float] = field(default_factory=dict)
    geometry_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self, geom: Geometry, op: OperatingCondition, imp: Impeller) -> None:
        self.in4 = VanelessState.from_state(imp.out)
        self._compute_geometry(geom)
        self.calculate(geom, op)
        self._apply_registry_losses(geom, op)

    def _compute_geometry(self, geom: Geometry) -> None:
        r_in = geom.r4
        r_out = geom.r5
        b_in = geom.b4
        b_out = geom.b5
        theta_in = geom.vd_leading_edge_angle
        theta_out = geom.vd_trailing_edge_angle
        z = geom.vd_number_of_vanes
        throat_location_factor = geom.vd_throat_location_factor
        area_throat_ratio = geom.vd_area_throat_ratio

        A_in = 2 * pi * r_in * b_in
        A_out = 2 * pi * r_out * b_out
        theta_avg = (theta_in + theta_out) / 2.0
        camber_angle = abs(theta_in - theta_out)
        solidity = 0.0
        if z > 0 and cos(theta_avg * pi / 180) != 0:
            solidity = z * (r_out - r_in) / (2 * pi * r_in * cos(theta_avg * pi / 180))
        loc_camber_max = 0.5
        if camber_angle != 0:
            loc_camber_max = (2 - abs(theta_in - theta_avg) / camber_angle) / 3
        pitch = math.inf if z == 0 else 2 * pi * r_out / z
        b_throat = b_in * (1 - throat_location_factor) + b_out * throat_location_factor
        area_throat = A_in * area_throat_ratio

        self.geometry_metrics = {
            "area_in": A_in,
            "area_out": A_out,
            "theta_mean": theta_avg,
            "camber_angle": camber_angle,
            "solidity": solidity,
            "loc_camber_max": loc_camber_max,
            "pitch": pitch,
            "width_throat": b_throat,
            "area_throat": area_throat,
        }

    def calculate(self, geom: Geometry, op: OperatingCondition):
        r_in = geom.r4
        r_out = geom.r5
        b_in = geom.b4
        b_out = geom.b5
        theta_in = geom.vd_leading_edge_angle
        theta_out = geom.vd_trailing_edge_angle

        r = np.linspace(r_in, r_out, 1 + self.n_steps, endpoint=True)
        dr = np.diff(r)
        b = np.linspace(b_in, b_out, 1 + self.n_steps, endpoint=True)

        Dh = np.sqrt(8 * r[:-1] * b[1:] * geom.blockage[4])
        A_eff = 2 * r[1:] * b[1:] * pi * geom.blockage[4]

        k = 0.02
        solidity = self.geometry_metrics.get("solidity", 0.0)
        cf_scale = 1.0 + 0.1 * solidity

        def target_alpha(step: int) -> float:
            frac = (step + 1) / self.n_steps
            return theta_in + (theta_out - theta_in) * frac

        def resolve_speed(x, return_values=False):
            in_ = VanelessState.from_state(self.in4)
            err = []
            for i in range(self.n_steps):
                alpha_target = target_alpha(i)

                Re = in_.c * in_.static.D / in_.static.V * b[i + 1]
                Cf = cf_scale * k * (1.8e5 / Re) ** 0.2

                ds = ((dr[i] / tan((90 - in_.alpha) / 180 * pi)) ** 2 + dr[i] ** 2) ** 0.5
                dp0 = 4.0 * Cf * ds * in_.c**2 * in_.static.D / 2 / Dh[i]

                c5m = float(x[i])
                c5t = c5m * tan(alpha_target * pi / 180)
                c5 = (c5m**2 + c5t**2) ** 0.5
                if c5 > 1.25 * in_.total.A:
                    err.extend((self.n_steps - i) * [1e4])
                    return err

                P0 = in_.total.P - dp0
                if P0 <= 0 and P0 < op.in0.P:
                    err.extend((self.n_steps - i) * [1e4])
                    return err
                try:
                    tot = op.fld.thermo_prop("PH", P0, in_.total.H)
                    stat = static_from_total(tot, c5)
                except ThermoException:
                    err.extend((self.n_steps - i) * [1e4])
                    return err

                err.append((op.m - A_eff[i] * c5m * stat.D) / op.m)

                in_.c = c5
                in_.alpha = alpha_target
                in_.total = tot
                in_.static = stat
                in_.m_abs = in_.c * cos(in_.alpha / 180 * pi) / in_.static.A
                if in_.m_abs >= 0.99:
                    err[-1] += in_.m_abs - 0.99
            if return_values:
                return err, in_
            return err

        c4m = self.in4.c * cos(theta_in * pi / 180)

        if c4m / self.in4.static.A >= 0.99:
            self.choke_flag = True
            return

        speed_guess = c4m * r[:-1] / r[1:]
        sol = optimize.root(resolve_speed, x0=speed_guess)

        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return

        _, out = resolve_speed(sol.x, return_values=True)
        out.m_abs = out.c * cos(out.alpha / 180 * pi) / out.static.A
        if out.m_abs >= 0.99:
            self.choke_flag = True
        self.out = out

        out_is = op.fld.thermo_prop("PS", out.total.P, self.in4.total.S)
        self.out.isentropic = out_is
        self.loss = out.total.H - out_is.H
        self.dh0s = out_is.H - self.in4.total.H

        delta_h = out.total.H - self.in4.total.H
        if abs(delta_h) <= 1e-6:
            self.eff = math.copysign(math.inf, self.dh0s)
        else:
            self.eff = self.dh0s / delta_h

    def _apply_registry_losses(self, geom: Geometry, op: OperatingCondition) -> None:
        if self.loss_config is None:
            return
        try:
            ctx = LossContext(
                component="diffuser",
                geometry=geom,
                operating_condition=op,
                inlet_state=self.in4,
                outlet_state=self.out,
                velocity_triangle={
                    "c4": self.in4.c,
                    "c5": self.out.c,
                    "alpha4": self.in4.alpha,
                    "alpha5": self.out.alpha,
                },
            )
            losses = LossModelRegistry.calculate_losses(ctx, self.loss_config)
            losses["total"] = sum(losses.values()) if losses else 0.0
            self.registry_losses = losses
        except Exception:
            self.registry_losses = {}


# Surge (RadComp fits)
def _generate_fits():
    mach_values = np.array([0, 0.4, 0.8, 1.2, 1.6])
    b_ratio = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
    deg = np.array([3, 3])

    a_12 = np.array(
        [
            [80.78, 80, 78.59, 76.41, 73.9],
            [76.71, 75.47, 73.28, 70.47, 67.19],
            [73.91, 72.97, 70.63, 66.25, 60],
            [72.81, 71.87, 69.53, 64.53, 55.63],
            [72.19, 71.25, 68.75, 63.59, 54.22],
        ]
    )

    a_20 = np.array(
        [
            [80.78, 80.16, 78.59, 76.41, 73.91],
            [76.56, 77.19, 73.44, 70.63, 67.19],
            [74.06, 71.56, 68.75, 64.84, 60.31],
            [70.47, 69.38, 66.25, 61.25, 55.16],
            [69.22, 68.13, 64.84, 59.38, 52.97],
        ]
    )

    def polyfit2d(x, y, z, deg):
        xx, yy = np.meshgrid(x, y)
        lhs = polynomial.polyvander2d(xx.ravel(), yy.ravel(), deg).T
        rhs = z.ravel().T

        scl = np.sqrt(np.square(lhs).sum(1))
        scl[scl == 0] = 1

        rcond = xx.size * np.finfo(xx.dtype).eps

        c1, _, _, _ = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
        c1 = (c1.T / scl).T

        return c1

    c12 = polyfit2d(mach_values, b_ratio, a_12, deg)
    c20 = polyfit2d(mach_values, b_ratio, a_20, deg)

    shape = deg + 1

    return c12.reshape(shape), c20.reshape(shape)


c_12, c_20 = _generate_fits()


def surge_critical_angle(r5: float, r4: float, b4: float, m2: float) -> float:
    ratio = b4 / r4
    length = r5 / r4

    angle_12 = polynomial.polyval2d(m2, ratio, c_12)
    angle_20 = polynomial.polyval2d(m2, ratio, c_20)

    alpha_r = angle_12 + (angle_20 - angle_12) * (length - 1.2) / (2.0 - 1.2)
    return 90.0 - 0.35 * (90.0 - alpha_r)