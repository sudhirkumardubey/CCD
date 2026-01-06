"""
Disc Friction Loss Models

Models for disc friction losses on impeller shrouds and hub.  
Require outlet conditions for proper Reynolds number calculation (Pass B timing).
"""

import logging
from ..registry import register_impeller
from ..context import ImpellerContext

logger = logging.getLogger(__name__)

@register_impeller("disc_friction", "daily_nece")
def disc_friction_daily_nece(ctx: ImpellerContext, **kwargs) -> float:
    """
    Daily & Nece (1960) as used in Oh (1997) optimum set:
    ΔH_df = f_df * ( r * r2 * U2^3 ) / ( 4 * m_dot )
    with f_df piecewise in Re_df = U2 * r2 / nu2
    """
    r4 = ctx.geom.r4
    rho2 = ctx.st2.static.D
    rho4 = ctx.st4.static.D
    nu2  = ctx.st4.static.V / max(1e-9, rho2)

    # Reynolds and piecewise f_df
    Re_df = max(1.0, ctx.U4 * r4 / max(1e-12, nu2))
    if Re_df < 3.0e5:
        f_df = 2.67 / (Re_df**0.5)
    else:
        f_df = 0.0622 / (Re_df**0.2)

    mdot = kwargs.get('mdot', getattr(ctx, "operating_condition", None))
    if hasattr(mdot, "m"):
        mdot = mdot.m
    mdot = mdot if isinstance(mdot, (int, float)) and mdot > 0 else 1.0
    return max(0.0, (f_df * ((rho2 + rho4)) * r4**2 * (ctx.U4**3) / (8.0 * mdot)))

@register_impeller("disc_friction", "shepherd")
def disc_friction_shepherd(ctx: ImpellerContext, **kwargs) -> float:
    """
    Shepherd (1963) form:
    ΔH_df = 0.01356 * ρ4 * U4^3 * D4^2 / ( m_dot * Re^0.2 ),
    Re = U4 * D4 / nu04
    """
    D4  = ctx.geom.r4 * 2.0
    rho2 = ctx.st2.static.D
    rho4 = ctx.st4.static.D
    mu04 = ctx.st4.total.V
    nu04  = mu04 / rho4
    Re   = max(1.0, ctx.U4 * D4 / max(1e-12, nu04))

    mdot = kwargs.get('mdot', getattr(ctx, "operating_condition", None))
    if hasattr(mdot, "m"):
        mdot = mdot.m
    mdot = mdot if isinstance(mdot, (int, float)) and mdot > 0 else 1.0
    return max(0.0, (0.01356 * rho4 * (ctx.U4**3) * (D4**2) / (mdot * (Re**0.2))))

@register_impeller("disc_friction", "boyce")
def disc_friction_boyce(ctx: ImpellerContext, **kwargs) -> float:
    """
    Boyce (2002) form:
    ΔH_df = 0.005 * ρ4 * U4^3 * D4^2 / ( m_dot * Re^0.2 ),
    Re = U4 * D4 / nu04
    """
    r2s = ctx.geom.r2s
    r2h = ctx.geom.r2h
    r4 = ctx.geom.r4
    rho2 = ctx.st2.static.D
    rho4 = ctx.st4.static.D
    nu2  = ctx.st4.static.V / max(1e-9, rho2)

    # Reynolds and piecewise f_df
    Re_df = max(1.0, ctx.U4 * r4 / max(1e-12, nu2))
    if Re_df < 3.0e5:
        f_df = 2.67 / (Re_df**0.5)
    else:
        f_df = 0.0622 / (Re_df**0.2)

    # Euler head
    dH_Euler = ctx.U4 * ctx.V4u - ctx.U2 * ctx.V2u
    dH_Euler = max(dH_Euler, 1e-6)

    mdot = kwargs.get('mdot', getattr(ctx, "operating_condition", None))
    if hasattr(mdot, "m"):
        mdot = mdot.m
    mdot = mdot if isinstance(mdot, (int, float)) and mdot > 0 else 1.0

    return max(0.0, (f_df * ((rho2 + rho4)) * r4**2 * (ctx.U4**3) / (8.0 * mdot)))