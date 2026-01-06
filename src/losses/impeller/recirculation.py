"""
Recirculation Loss Models

Models for recirculation losses in impellers due to flow separation.
Require outlet conditions for proper diffusion analysis (Pass B timing).
"""

import numpy as np
import logging
from ..registry import register_impeller
from ..context import ImpellerContext

logger = logging.getLogger(__name__)

@register_impeller("recirculation", "coppage")
def recirculation_coppage(ctx: ImpellerContext, **kwargs) -> float:
    """
    Coppage (1956) recirculation loss (literature-consistent).
    Δh_rc = 0.02 * sqrt(tan(alpha_exit)) * Df * U_exit^2
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

        return max(0.0, 0.02 * np.sqrt(max(0.0, np.tan(ctx.st4.alpha))) * Df**2 * (ctx.U4**2))  # alpha in radians

    except Exception as e:
        logger.warning(f"Coppage recirculation calculation failed: {e}")
        return 0.0

@register_impeller("recirculation", "oh_hyperbolic")
def recirculation_oh_hyperbolic(ctx: ImpellerContext, **kwargs) -> float:
    """
    Oh et al. (1997) hyperbolic recirculation loss (angles in radians).
    Δh_rc = 8e-5 * sinh(3.5 * alpha_exit^3) * (Df^2) * U_exit^2
    """
    try:
        Z = ctx.geom.n_blades + ctx.geom.n_splits
        r2s = ctx.geom.r2s       # inlet tip radius
        r4  = ctx.geom.r4

        # Prefer inlet shroud (tip) relative speed if you store it; fall back to station 1:
        W2s = ctx.st2.Ws    # relative at inlet shroud
        W4  = ctx.st4.W     # relative at exit

        # Euler head
        dH_Euler = ctx.U4 * ctx.V4u - ctx.U2 * ctx.V2u
        dH_Euler = max(dH_Euler, 1e-6)

        # Coppage diffusion factor (note the geometry bracket)
        geom_bracket = (Z/np.pi) * (1.0 - r2s/r4) + 2.0 * (r2s/r4)
        Df = 1.0 - (W4 / W2s) + 0.75 * (dH_Euler / ctx.U4**2) * (W4 / W2s) * geom_bracket

        return max(0.0, 8e-5 * np.sinh(3.5 * (ctx.st4.alpha**3)) * (Df**2) * (ctx.U4**2))  # alpha in radians

    except Exception as e:
        logger.warning(f"Oh hyperbolic recirculation calculation failed: {e}")
        return 0.0