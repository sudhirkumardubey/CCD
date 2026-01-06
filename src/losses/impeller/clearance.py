"""
Clearance Loss Models

Models for tip clearance losses in impellers.
Require outlet conditions for proper analysis (Pass B timing).
"""

import numpy as np
import logging
from ..registry import register_impeller  
from ..context import ImpellerContext

logger = logging.getLogger(__name__)

@register_impeller("clearance", "jansen")
def clearance_jansen(ctx: ImpellerContext, **kwargs) -> float:
    """
    Jansen tip-clearance loss (used by Oh's optimum set).
    ΔH_cl = 0.6 * (c/b4) * V4u * sqrt( G * (V4u * V2) / (1 + rho2/rho1) )
    where G = (4π/(b4*Z)) * (r2s^2 - r2h^2) / (r4 - r2s) 
    """
    try:
        c   = getattr(ctx.geom, "tip_cl", getattr(ctx.geom, "clearance", 0.0))
        b4  = ctx.geom.b4
        Z   = ctx.geom.n_blades + ctx.geom.n_splits
        r2s = ctx.geom.r2s
        r2h = ctx.geom.r2h
        r2m = ctx.geom.r2rms
        r4  = ctx.geom.r4

        # geometry factor (dimensionless)
        G = (4.0 * np.pi / (b4*Z)) * ((r2s**2 - r2h**2) / max(1e-9, (r4 - r2s)))
        G = max(G, 0.0)

        # Zhang's density variant (benign if densities missing)
        rho1 = max(1e-9, ctx.st2.static.D) if ctx.st2.static.D else 1.0
        rho2 = max(1e-9, ctx.st4.static.D) if ctx.st4.static.D else rho1
        dens_term = 1.0 + (rho2 / rho1)

        Dh = 0.6 * (c / max(1e-9, b4)) * abs(ctx.V4u) * np.sqrt(max(0.0, G * (abs(ctx.V4u) * abs(ctx.V2)) / dens_term))
        return Dh
    except Exception as e:
        logger.warning(f"Jansen clearance calculation failed: {e}")
        return 0.0

@register_impeller("clearance", "krylov_spunde")
def clearance_krylov_spunde(ctx: ImpellerContext, **kwargs) -> float:
    """
    Krylov & Spunde:
    ΔH_cl = 2 * (c/b4) * [ (r2h + r2s)/(2*r4) - 0.275 ] * U4^2
    """
    try:
        c   = getattr(ctx.geom, "tip_cl", getattr(ctx.geom, "clearance", 0.0))
        b4  = ctx.geom.b4
        r4  = ctx.geom.r4
        r2h = ctx.geom.r2h
        r2s = ctx.geom.r2s
        bracket = (((r2h + r2s)/(2.0*max(1e-9, r4))) - 0.275)
        return 2.0 * (c / max(1e-9, b4)) * bracket * (ctx.U4**2)
    except Exception as e:
        logger.warning(f"Krylov–Spunde clearance calculation failed: {e}")
        return 0.0