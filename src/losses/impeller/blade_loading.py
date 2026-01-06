"""
Blade Loading Loss Models

Models for losses due to blade loading and flow diffusion.
Require outlet conditions for diffusion factor calculation (Pass B timing).
"""

import numpy as np
import logging
from ..registry import register_impeller
from ..context import ImpellerContext

logger = logging.getLogger(__name__)

@register_impeller("blade_loading", "coppage")
def blade_loading_coppage(ctx: ImpellerContext, **kwargs) -> float:
    """
    Coppage (1956): ΔH_bl = 0.05 * Df^2 * U2^2
    Df per Coppage with Euler head and geometry bracket.
    """
    try:
        Z = ctx.geom.n_blades + ctx.geom.n_splits
        r2s = ctx.geom.r2s       # inlet tip radius
        r4  = ctx.geom.r4

        # Prefer inlet shroud (tip) relative speed if you store it; fall back to station 1:
        W2s = ctx.st2.Ws    # relative at inlet shroud
        W4  = ctx.st4.W    # relative at exit

        # Euler head
        dH_Euler = ctx.U4 * ctx.V4u - ctx.U2 * ctx.V2u
        dH_Euler = max(dH_Euler, 1e-6)

        # Coppage diffusion factor (note the geometry bracket)
        geom_bracket = (Z/np.pi) * (1.0 - r2s/r4) + 2.0 * (r2s/r4)
        Df = 1.0 - (W4 / W2s) + 0.75 * (dH_Euler / ctx.U4**2) * (W4 / W2s) * geom_bracket

        # Guardrails (same spirit as literature usage)
        Df = max(Df, 0.0)

        return 0.05 * (Df**2) * (ctx.U4**2)
    except Exception as e:
        logger.warning(f"Coppage blade loading calculation failed: {e}")
        return 0.0

@register_impeller("blade_loading", "aungier")
def blade_loading_aungier(ctx: ImpellerContext, **kwargs) -> float:
    """
    Aungier (1995): ΔH_bl = (ΔW)^2 / 48,  ΔW = 2π d2 V2u / (Z Lb)
    """
    try:
        Z  = ctx.geom.n_blades + ctx.geom.n_splits
        d4 = ctx.geom.r4 * 2.0
        Lb = ctx.geom.l_comp if hasattr(ctx.geom, "l_comp") else (ctx.geom.r4 - ctx.geom.r2s + ctx.geom.b4)  # meridional length from 2 to 3
        dW = (2.0 * np.pi * d4 * ctx.V4u) / (Z * Lb)
        return (dW**2) / 48.0
    except Exception as e:
        logger.warning(f"Aungier blade loading calculation failed: {e}")
        return 0.0

@register_impeller("blade_loading", "zhang_select")
def blade_loading_zhang_select(ctx: ImpellerContext, **kwargs) -> float:
    """
    Zhang (2019) selection logic:
    if Mw_inlet_tip < 0.8: use Aungier
    elif Mw_inlet_tip >= 0.8 and ns < 0.7: use Coppage
    else: use Aungier
    """
    # You supply Mw_tip and ns if you already compute them; else approximate here.
    Mw_tip = ctx.st2.M_rels if hasattr(ctx.st2, "M_rels") else None
    ns     = kwargs.get("ns")

    # Fallback approximations if not provided (optional):
    if Mw_tip is None:
        a   = ctx.st2.static.A
        Mw_tip = ctx.st2.Ws / a

    if ns is None:
        # ns requires Q and isentropic head; accept user-provided value or compute from your solver.
        ns = kwargs.get("ns_fallback", 0.6)

    if Mw_tip < 0.8:
        return blade_loading_aungier(ctx, **kwargs)
    elif ns < 0.7:
        return blade_loading_coppage(ctx, **kwargs)
    else:
        return blade_loading_aungier(ctx, **kwargs)