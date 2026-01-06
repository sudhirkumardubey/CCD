"""
Skin Friction Loss Models

Models for viscous losses along blade and shroud surfaces.
Require full flow field (Pass B timing).
"""

from ..registry import register_impeller
from ..context import ImpellerContext
from ..util import (compute_W_avg_for_sf, compute_Lb_and_Dhyd_Jansen, 
                   compute_Cf_Jansen, ensure_positive, compute_reynolds_disc_friction)

@register_impeller("skin_friction", "jansen")
def skin_friction_jansen(ctx: ImpellerContext, **kwargs) -> float:
    """
    Jansen skin friction model per Zhang (2019) Eqs. (4)-(9).
    
    Most accurate model using proper hydraulic diameter and flow length.
    ΔH_sf = 2 * Cf * (Lb/D_hyd) * W̅²
    """
    try:
        # Weighted average relative velocity
        Wavg = compute_W_avg_for_sf(ctx.st2, ctx.st4)
        
        # Kinematic viscosity at inlet
        mu2 = ctx.st2.static.V  # Dynamic viscosity
        rho2 = max(1e-9, ctx.st2.static.D)
        nu02 = mu2 / rho2
        
        # Geometry factors from Zhang (2019)
        Lb, Dhyd = compute_Lb_and_Dhyd_Jansen(ctx.geom)
        
        # Friction coefficient
        Cf = compute_Cf_Jansen(Dhyd, ctx.U4, nu02)
        
        # Loss calculation
        loss = 2.0 * Cf * (Lb / Dhyd) * (Wavg**2)
        return ensure_positive(loss)
        
    except Exception:
        return 0.0

