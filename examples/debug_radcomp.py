"""
Debug script to identify where choking occurs
"""

import sys
import os
import numpy as np

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from geometry.geometry import Geometry
from conditions.operating import thermo_prop, OperatingCondition
from components.inducer import Inducer
from components.impeller import Impeller

# Import loss models
from losses.impeller import skin_friction, blade_loading, clearance
from losses.impeller import disc_friction, recirculation


def debug_radcomp_geometry():
    """Debug the RadComp case step by step"""
    
    # Geometry
    geom = Geometry(
        r1h=0.01 * 0.5,
        r1s=0.01,
        r2h=0.002,
        r2s=0.0056,
        r4=0.01,
        b4=0.0012,
        r5=0.0165,
        b5=0.0012,
        b2=0.0056 - 0.002,
        alpha2=0.0,
        beta2=-45.0,
        beta4=-45.0,
        n_blades=9,
        n_splitter=9,
        t_cl=15e-6,
        slip=0.9,
        BF2=1.0,
        BF4=1.0,
        L_ind=0.02,
    )
    
    print("Geometry Check:")
    print(f"  r1h = {geom.r1h*1000:.3f} mm")
    print(f"  r1s = {geom.r1s*1000:.3f} mm")
    print(f"  r2h = {geom.r2h*1000:.3f} mm")
    print(f"  r2s = {geom.r2s*1000:.3f} mm")
    print(f"  r4 = {geom.r4*1000:.3f} mm")
    print(f"  b4 = {geom.b4*1000:.3f} mm")
    
    print(f"\nFlow Areas:")
    print(f"  A1 = {geom.A1*1e6:.3f} mm²")
    print(f"  A2 = {geom.A2*1e6:.3f} mm²")
    print(f"  A2_eff = {geom.A2_eff*1e6:.3f} mm²")
    print(f"  A4 = {geom.A4*1e6:.3f} mm²")
    
    # Operating conditions
    P0_in = 1.65e5
    T0_in = 265.0
    m_flow = 0.120
    n_rpm = 130000
    n_rot = n_rpm * 2 * np.pi / 60
    
    in0 = thermo_prop("Air", "PT", P0_in, T0_in)
    
    print(f"\nInlet Conditions:")
    print(f"  P0 = {in0.P/1e5:.3f} bar")
    print(f"  T0 = {in0.T:.2f} K")
    print(f"  ρ0 = {in0.D:.3f} kg/m³")
    print(f"  a0 = {in0.A:.2f} m/s")
    
    print(f"\nOperating Point:")
    print(f"  m = {m_flow:.3f} kg/s")
    print(f"  N = {n_rpm:.0f} RPM")
    print(f"  ω = {n_rot:.2f} rad/s")
    print(f"  U4 = {geom.r4 * n_rot:.2f} m/s")
    
    # Check characteristic velocities
    U4 = geom.r4 * n_rot
    print(f"\nVelocity Check:")
    print(f"  U4/a0 = {U4/in0.A:.3f} (tip Mach number)")
    
    # Check mass flow feasibility
    # For rough estimate:  m = ρ * V * A
    # At inlet: V1 ≈ m/(ρ*A2_eff)
    V1_estimate = m_flow / (in0.D * geom.A2_eff)
    M1_estimate = V1_estimate / in0.A
    
    print(f"  Estimated V1 = {V1_estimate:.2f} m/s")
    print(f"  Estimated M1 = {M1_estimate:.3f}")
    
    if M1_estimate > 0.9:
        print(f"  ⚠️  WARNING: Inlet Mach number very high!")
    
    # Try to run inducer
    print("\n" + "="*70)
    print("INDUCER CALCULATION")
    print("="*70)
    
    op = OperatingCondition(in0=in0, fld="Air", m=m_flow, n_rot=n_rot)
    
    try:
        ind = Inducer(geom, op)
        
        if ind.choke_flag:
            print("✗ Inducer:  CHOKED")
            print(f"  Outlet Mach:  {ind.out.m_abs:.3f}")
        else:
            print("✓ Inducer: OK")
            print(f"  Outlet P: {ind.out.total.P/1e5:.3f} bar")
            print(f"  Outlet T: {ind.out.total.T:.2f} K")
            print(f"  Outlet M: {ind.out.m_abs:.3f}")
            print(f"  Outlet V: {ind.out.c:.2f} m/s")
            
            # Try impeller
            print("\n" + "="*70)
            print("IMPELLER CALCULATION")
            print("="*70)
            
            loss_config = {
                "skin_friction": "jansen_skin_friction",
                "blade_loading": "rodgers_blade_loading",
                "clearance": "jansen_clearance",
                "disc_friction": "daily_nece_disc_friction",
                "recirculation": "rodgers_recirculation",
            }
            
            imp = Impeller(geom=geom, op=op, ind=ind, loss_config=loss_config)
            
            if imp.choke_flag:
                print("✗ Impeller: CHOKED")
                print(f"  Outlet Mach: {imp.out.m_abs:.3f}")
            else:
                print("✓ Impeller: OK")
                print(f"  Outlet P: {imp.out.total.P/1e5:.3f} bar")
                print(f"  Outlet T:  {imp.out.total.T:.2f} K")
                print(f"  Outlet M: {imp.out.m_abs:.3f}")
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_radcomp_geometry()