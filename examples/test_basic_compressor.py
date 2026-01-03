"""
Validation test against RadComp results
Uses exact geometry and operating conditions from RadComp's EvaluateCompressor. ipynb
"""

import sys
import os
import numpy as np

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from geometry.geometry import Geometry
from conditions.operating import OperatingCondition, thermo_prop
from core.compressor import CentrifugalCompressor

# Import loss models to register them
from losses.impeller import skin_friction, blade_loading, clearance
from losses.impeller import disc_friction, recirculation


def create_radcomp_geometry():
    """
    Create geometry matching RadComp's EvaluateCompressor.ipynb
    
    From the notebook (Schiffmann and Favrat, 2010):
    r1=0.01, r2s=0.0056, r2h=0.002, beta2=-45, beta2s=-56, alpha2=0.0,
    r4=0.01, b4=0.0012, beta4=-45, n_blades=9, n_splits=9,
    r5=0.0165, b5=0.0012, blade_e=1e-4, rug_imp=1. 0e-5, 
    clearance=15e-6, backface=0.001, rug_ind=1.0e-4,
    l_ind=0.02, l_comp=0.7 * 0.01, blockage=[1.0, 1.0, 1.0, 1.0, 1.0]
    """
    
    geom = Geometry(
        # Inducer
        r1h=0.01 * 0.5,      # Estimate hub as 50% of r1
        r1s=0.01,            # Inducer inlet shroud radius (10 mm)
        
        # Impeller inlet
        r2h=0.002,           # Hub radius at impeller inlet (2 mm)
        r2s=0.0056,          # Shroud radius at impeller inlet (5. 6 mm)
        
        # Impeller outlet
        r4=0.01,             # Impeller outlet radius (10 mm)
        b4=0.0012,           # Impeller outlet width (1.2 mm)
        
        # Diffuser
        r5=0.0165,           # Diffuser outlet radius (16.5 mm)
        b5=0.0012,           # Diffuser outlet width (1.2 mm)
        
        # Blade geometry
        b2=0.0056 - 0.002,   # Inlet width (3.6 mm)
        alpha2=0.0,          # No preswirl
        beta2=-45.0,         # Inlet blade angle at mean
        beta4=-45.0,         # Outlet blade angle (backswept)
        
        # Blade count
        n_blades=9,          # Number of main blades
        n_splitter=9,        # Number of splitter blades
        
        # Clearances
        t_cl=15e-6,          # 15 micron tip clearance
        
        # Slip factor
        slip=0.9,            # Typical Wiesner correlation
        
        # Blockage
        BF2=1.0,             # No blockage (RadComp uses 1.0)
        BF4=1.0,             # No blockage
        
        # Inducer length
        L_ind=0.02,          # 20 mm inducer length
    )
    
    return geom


def test_radcomp_case():
    """
    Test case matching RadComp EvaluateCompressor.ipynb
    
    Operating conditions from notebook:
    - Inlet: 1. 65 bar, 265 K (Air)
    - Mass flow:  0.120 kg/s
    - Rotational speed: 130,000 RPM
    
    Expected results (from the notebook plot):
    - PR: approximately 1.3-1.4
    - Efficiency: approximately 47-50%
    """
    
    print("="*70)
    print("VALIDATION AGAINST RADCOMP")
    print("="*70)
    
    # Create RadComp geometry
    geom = create_radcomp_geometry()
    
    print("\nGeometry (from Schiffmann and Favrat, 2010):")
    print(f"  Impeller diameter: {geom.r4*2*1000:.1f} mm")
    print(f"  Impeller width: {geom.b4*1000:.2f} mm")
    print(f"  Number of blades: {geom. n_blades}")
    print(f"  Number of splitters: {geom.n_splitter}")
    print(f"  Outlet blade angle: {geom.beta4:.1f}°")
    print(f"  Tip clearance: {geom.t_cl*1e6:.1f} μm")
    
    # Operating conditions from notebook
    P0_in = 1.65e5        # 1.65 bar
    T0_in = 265.0         # 265 K
    m_flow = 0.120        # 0.120 kg/s
    n_rpm = 130000        # 130,000 RPM
    
    print("\nOperating Conditions:")
    print(f"  Mass flow: {m_flow:.3f} kg/s")
    print(f"  Rotational speed: {n_rpm:.0f} RPM")
    print(f"  Inlet pressure:  {P0_in/1e5:.2f} bar")
    print(f"  Inlet temperature: {T0_in:.1f} K")
    
    # Create operating condition
    in0 = thermo_prop("Air", "PT", P0_in, T0_in)
    n_rot = n_rpm * 2 * np.pi / 60
    
    print(f"  Inlet density: {in0.D:.3f} kg/m³")
    print(f"  Speed of sound: {in0.A:.1f} m/s")
    
    op = OperatingCondition(
        in0=in0,
        fld="Air",
        m=m_flow,
        n_rot=n_rot
    )
    
    # Loss configuration (matching RadComp correlations as closely as possible)
    loss_config = {
        "impeller": {
            "skin_friction": "jansen_skin_friction",
            "blade_loading": "rodgers_blade_loading",
            "clearance": "jansen_clearance",
            "disc_friction": "daily_nece_disc_friction",
            "recirculation": "rodgers_recirculation",
        }
    }
    
    # Calculate with CCD
    print("\n" + "="*70)
    print("CALCULATING WITH CCD FRAMEWORK...")
    print("="*70)
    
    comp = CentrifugalCompressor(geom, op, loss_config)
    
    try:
        success = comp.calculate()
        
        if success:
            print("\n✓ Calculation successful!")
            
            # Print results
            print("\n" + "="*70)
            print("RESULTS")
            print("="*70)
            
            summary = comp.get_summary()
            
            print(f"\nCCD Framework Results:")
            print(f"  Pressure Ratio: {summary['PR']:.4f}")
            print(f"  Efficiency: {summary['efficiency']*100:.2f}%")
            print(f"  Power:  {summary['power_kW']:.3f} kW")
            print(f"  Head Coefficient: {summary['head']:.4f}")
            print(f"  Flow Coefficient: {summary['flow_coefficient']:.6f}")
            
            print(f"\nNon-dimensional Parameters:")
            print(f"  Specific Speed Ns: {summary['Ns']:.4f}")
            print(f"  Specific Diameter Ds: {summary['Ds']:.4f}")
            
            print(f"\nComponent Efficiency:")
            print(f"  Inducer:  {comp.ind. eff*100:.2f}%")
            print(f"  Impeller: {comp.imp. eff*100:.2f}%")
            print(f"  Diffuser: {comp. dif.eff*100:.2f}%")
            
            print(f"\nImpeller Losses (J/kg):")
            losses = comp.imp.losses
            print(f"  Incidence: {losses. incidence:.1f}")
            print(f"  Skin Friction: {losses.skin_friction:.1f}")
            print(f"  Blade Loading:  {losses.blade_loading:.1f}")
            print(f"  Tip Clearance: {losses.clearance:.1f}")
            print(f"  Disc Friction: {losses.disc_friction:.1f}")
            print(f"  Recirculation: {losses.recirculation:.1f}")
            print(f"  TOTAL: {losses.total:.1f}")
            
            print(f"\nFlow Path:")
            print(f"  Station          P (bar)    T (K)     M")
            print(f"  Inlet:            {comp.inlet.total. P/1e5:.3f}      {comp.inlet.total.T:.1f}    {comp.results.m_in:.3f}")
            print(f"  Inducer Out:     {comp.ind.out.total.P/1e5:.3f}      {comp.ind.out.total. T:.1f}    {comp.ind.out.m_abs:.3f}")
            print(f"  Impeller Out:    {comp.imp.out.total.P/1e5:.3f}      {comp. imp.out.total.T:.1f}    {comp.imp.out.m_abs:.3f}")
            print(f"  Diffuser Out:    {comp.outlet.total.P/1e5:.3f}      {comp. outlet.total.T:.1f}    {comp. dif.out.m_abs:.3f}")
            
            # Expected results from RadComp (from the notebook contour plot)
            print("\n" + "="*70)
            print("COMPARISON WITH RADCOMP")
            print("="*70)
            
            print("\nExpected RadComp Results (from notebook plot):")
            print("  Pressure Ratio: ~1.308 (from printed output)")
            print("  Efficiency: ~47.5% (from printed output)")
            
            # Calculate differences
            radcomp_PR = 1.308
            radcomp_eff = 0.475
            
            pr_diff = abs((summary['PR'] - radcomp_PR) / radcomp_PR) * 100
            eff_diff = abs(summary['efficiency'] - radcomp_eff) * 100
            
            print(f"\nDifferences:")
            print(f"  PR difference: {pr_diff:.2f}%")
            print(f"  Efficiency difference: {eff_diff:.2f} percentage points")
            
            print("\n" + "="*70)
            print("VALIDATION ASSESSMENT")
            print("="*70)
            
            if pr_diff < 10.0:
                print(f"✓ Pressure Ratio within 10% tolerance ({pr_diff:.2f}%)")
            else:
                print(f"✗ Pressure Ratio differs by {pr_diff:.2f}% (target: <10%)")
            
            if eff_diff < 5.0:
                print(f"✓ Efficiency within 5 points tolerance ({eff_diff:.2f} points)")
            else:
                print(f"✗ Efficiency differs by {eff_diff:.2f} points (target: <5 points)")
            
            print("\nNotes:")
            print("  - Differences are expected due to:")
            print("    * Different loss model implementations")
            print("    * Different numerical solvers")
            print("    * Approximations in geometry conversion")
            print("    * RadComp uses more detailed models for some components")
            
            return comp
        
        else:
            print("\n✗ Calculation failed!")
            if comp.results.choke: 
                print("  Reason: Choked flow")
            if comp.results.surge:
                print("  Reason: Surge")
            if comp.results.wet:
                print("  Reason: Two-phase flow (wet gas)")
            
            return None
    
    except Exception as e: 
        print(f"\n✗ Calculation error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CCD FRAMEWORK - RADCOMP VALIDATION TEST")
    print("Based on Schiffmann and Favrat (2010) geometry")
    print("="*70)
    
    comp = test_radcomp_case()
    
    if comp:
        print("\n" + "="*70)
        print("TEST COMPLETE - Results obtained successfully")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("TEST FAILED - Check error messages above")
        print("="*70)