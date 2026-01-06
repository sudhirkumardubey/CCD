"""
Leakage Loss Models

Models for leakage losses due to secondary flows through clearances.
Require outlet conditions for proper pressure difference analysis (Pass B timing).
"""

import numpy as np
import logging
from ..registry import register_impeller
from ..context import ImpellerContext

logger = logging.getLogger(__name__)

@register_impeller("leakage", "aungier")
def leakage_aungier(ctx: ImpellerContext, **kwargs) -> float:
    """
    Aungier/Oh leakage loss:
    Δh_lk = (m_dot_lk * U_lk * U2) / (2 * m_dot)
    U_lk = 0.816 * sqrt( 2*ΔP_lk / ρ2 )
    m_dot_lk = ρ2 * N_eff * ε * L_v * U_lk
    ΔP_lk = m_dot * ( r2*V2u - r1*V1u ) / ( N_eff * r_bar * b_bar * L_v )
    """
    try:
        # Geometry & counts
        Z      = ctx.geom.n_blades + ctx.geom.n_splits               # blades (N_eff)
        r2     = ctx.geom.r2rms  # inlet tip radius; fallback
        r4     = ctx.geom.r4     # exit radius in your indexing
        b2    = ctx.geom.r2s - ctx.geom.r2h   # inlet width in your indexing
        b4     = ctx.geom.b4    # exit width in your indexing
        c      = ctx.geom.back_cl  # seal gap
        rho4   = ctx.st4.static.D
        
        b_mean = 0.5 * (b2 + b4)
        r_mean = 0.5 * (ctx.geom.r2rms + ctx.geom.r4)   
        # Leakage jet speed and leakage mass flow
        
        term1   = rho4 * c * ctx.U4 * 1.332 
        term2  = (r4 * ctx.V4u - r2 * ctx.V2u) / (2.0 * r_mean * b_mean)

        # Specific enthalpy rise due to leakage mixing
        return max(0, term1 * term2)

    except Exception as e:
        logger.warning(f"Aungier leakage calculation failed: {e}")
        return 0.0

@register_impeller("leakage", "jansen")
def leakage_jansen(ctx: ImpellerContext, **kwargs) -> float:
    """
    Jansen-style leakage (Zhang 2019, Eq. (48)):
    Δh_lk = 0.6 * (c/b3) * V3 * sqrt( (4π)/(b3 Z) * ((r2t - r2h)/(r3 - r2t)) * (1 + ρ3/ρ2) ) * V3u
    Notes:
    • Uses exit quantities and seal gap c.
    • Matches the structure given in Zhang's review; parameters map to your geometry.
    """
    try:
        # Geometry (tip/hub at inlet radius are needed)
        Z   = ctx.geom.n_blades + ctx.geom.n_splits
        r2s = ctx.geom.r2s  # inlet tip radius; fallback
        r2h = ctx.geom.r2h  # inlet hub radius; fallback
        r4  = ctx.geom.r4   # exit radius in your indexing
        b4  = ctx.geom.b4   # exit width in your indexing
        c   = ctx.geom.back_cl  # seal gap

        rho2= ctx.st2.static.D
        rho4= ctx.st4.static.D

        geom = ((4.0 * np.pi)/(b4 * Z)) * max(0.0, (r2s - r2h)) / max(1e-9, (r4 - r2s))
        scale= np.sqrt(geom  * ctx.V4u * ctx.V2 / (1.0 + rho4/rho2))

        return 0.6 * (c / max(1e-9, b4)) * ctx.V4 * scale

    except Exception as e:
        logger.warning(f"Jansen leakage calculation failed: {e}")
        return 0.0