"""
Performance analysis tools
"""

import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from geometry.geometry import Geometry
from conditions.operating import (
    CoolPropFluid,
    OperatingCondition,
    ThermoProp,
    thermo_prop,
)
from core.compressor import CentrifugalCompressor


def upper_bounds(
    geom: Geometry, 
    in0: ThermoProp, 
    max_mach_rot: float = 2.5, 
    max_mach_flow: float = 0.7
) -> Tuple[float, float]:
    """
    Calculate upper bounds for operating conditions
    
    Parameters
    ----------
    geom : Geometry
        Compressor geometry
    in0 :  ThermoProp
        Inlet thermodynamic state
    max_mach_rot : float
        Maximum tip speed Mach number
    max_mach_flow : float
        Maximum inlet flow Mach number
    
    Returns
    -------
    n_rot_max : float
        Maximum rotational speed (rad/s)
    m_flow_max : float
        Maximum mass flow rate (kg/s)
    """
    n_rot_max = max_mach_rot * in0.A / geom.r4
    m_flow_max = max_mach_flow * in0.A * in0.D * geom.A2_eff
    
    return n_rot_max, m_flow_max


def plot_compressor_map(
    n_rot_grid: np.ndarray,
    m_flow_grid: np.ndarray,
    pr_results: np.ndarray,
    eta_results: np.ndarray,
    valid_results: np.ndarray,
    design_point: Optional[Dict] = None,
    title: str = "Compressor Performance Map",
    save_path: Optional[str] = None,
):
    """
    Plot compressor performance map (matching notebook cell style)
    
    Parameters
    ----------
    n_rot_grid : np.ndarray
        Rotational speed grid (rad/s)
    m_flow_grid : np.ndarray
        Mass flow rate grid (kg/s)
    pr_results : np.ndarray
        Pressure ratio results
    eta_results : np.ndarray
        Efficiency results
    valid_results : np.ndarray
        Boolean array of valid points
    design_point : dict, optional
        Design point {'m': mass_flow, 'PR': pressure_ratio, 'eta': efficiency}
    title : str
        Plot title prefix
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    valid_mask = valid_results & (~np.isnan(pr_results)) & (~np.isnan(eta_results))
    m_valid = m_flow_grid[valid_mask]
    pr_valid = pr_results[valid_mask]
    eta_valid = eta_results[valid_mask]
    n_valid = n_rot_grid[valid_mask]
    
    # Plot 1: Pressure Ratio vs Mass Flow (colored by efficiency)
    if len(m_valid) > 0:
        sc1 = ax1.scatter(m_valid, pr_valid, c=eta_valid, cmap="viridis", s=50, alpha=0.8)
        ax1.set_xlabel("Mass Flow Rate (kg/s)")
        ax1.set_ylabel("Pressure Ratio")
        ax1.set_title(f"{title} - Pressure Ratio")
        ax1.grid(True, alpha=0.3)
        if len(pr_valid) > 0:
            ax1.set_ylim(0.9, max(pr_valid) * 1.1)
        cb1 = plt.colorbar(sc1, ax=ax1)
        cb1.set_label("Isentropic Efficiency")
        
        # Add design point if provided
        if design_point:
            ax1.scatter(
                [design_point['m']], [design_point['PR']], 
                c="red", s=100, marker="*", edgecolor="black", 
                linewidth=2, label="Design Point"
            )
            ax1.legend()
    
    # Plot 2: Efficiency vs Mass Flow (colored by speed)
    if len(m_valid) > 0:
        sc2 = ax2.scatter(m_valid, eta_valid, c=n_valid / 1000.0, cmap="plasma", s=50, alpha=0.8)
        ax2.set_xlabel("Mass Flow Rate (kg/s)")
        ax2.set_ylabel("Isentropic Efficiency")
        ax2.set_title(f"{title} - Efficiency")
        ax2.grid(True, alpha=0.3)
        if len(eta_valid) > 0:
            ax2.set_ylim(0, 1)
        cb2 = plt.colorbar(sc2, ax=ax2)
        cb2.set_label("Speed (krad/s)")
        
        # Add design point if provided
        if design_point:
            ax2.scatter(
                [design_point['m']], [design_point['eta']], 
                c="red", s=100, marker="*", edgecolor="black", 
                linewidth=2, label="Design Point"
            )
            ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)


def calculate_on_op_grid(
    geom: Geometry,
    in0: ThermoProp,
    fld: str,
    lb: np.ndarray,
    ub: np.ndarray,
    resolution: float = 0.02,
    loss_config: Dict = None,
    map_func=map,
    solver: str = "ccd",
) -> Tuple[np.ndarray, Iterator]: 
    """
    Calculate compressor performance on a grid of operating points
    
    Parameters
    ----------
    geom : Geometry
        Compressor geometry
    in0 : ThermoProp
        Inlet thermodynamic state
    fld : str
        Fluid name
    lb : np.ndarray
        Lower bounds [n_rot_min, m_flow_min]
    ub : np.ndarray
        Upper bounds [n_rot_max, m_flow_max]
    resolution : float
        Grid resolution (fraction of range)
    loss_config : dict
        Loss model configuration
    map_func : callable
        Mapping function (map, multiprocessing. Pool. map, etc.)
    
    Returns
    -------
    X : np.ndarray
        Grid of operating points [n_rot, m_flow]
    results : Iterator
        Iterator of (compressor, calculation_time) tuples
    """
    if loss_config is None:
        loss_config = {}
    
    # Generate grid
    n_rot_range = np.arange(lb[0], ub[0], resolution * (ub[0] - lb[0]))
    m_flow_range = np.arange(lb[1], ub[1], resolution * (ub[1] - lb[1]))
    
    X = np.array(list(itertools.product(n_rot_range, m_flow_range)))
    
    def calculate_point(point):
        """Calculate single operating point"""
        n_rot, m_flow = point

        op = OperatingCondition(in0=in0, fld=fld, m=m_flow, n_rot=n_rot)

        if solver == "radcomp":
            try:
                from radcompressor.compressor import Compressor as RCCompressor
                from radcompressor.geometry import Geometry as RCGeometry
                from radcompressor.condition import OperatingCondition as RCOperatingCondition
                from radcompressor.thermo import CoolPropFluid as RCCoolPropFluid
            except Exception:
                # Fallback to CCD if radcompressor is unavailable
                rc_comp = None
            else:
                g_dict = geom.to_dict() if hasattr(geom, "to_dict") else geom.__dict__
                blockage = None
                if hasattr(geom, "blockage") and geom.blockage is not None:
                    blockage = list(getattr(geom, "blockage"))
                elif isinstance(g_dict, dict):
                    blockage = [
                        g_dict.get(f"blockage{i+1}") for i in range(5)
                    ]
                    if any(b is None for b in blockage):
                        blockage = None
                if blockage is None:
                    blockage = [1.0] * 5
                rc_geom = RCGeometry.from_dict(g_dict, blockage=blockage)
                rc_fld = RCCoolPropFluid(getattr(fld, "name", fld))
                rc_op = RCOperatingCondition(in0=in0, fld=rc_fld, m=m_flow, n_rot=n_rot)
                rc_comp = RCCompressor(rc_geom, rc_op)
                t0 = time.perf_counter()
                try:
                    rc_comp.calculate()
                    dt = time.perf_counter() - t0
                except Exception:
                    dt = time.perf_counter() - t0
                    rc_comp.invalid_flag = True
                return rc_comp, dt

        comp = CentrifugalCompressor(geom, op, loss_config)

        t0 = time.perf_counter()
        try:
            comp.calculate()
            dt = time.perf_counter() - t0
        except Exception:
            dt = time.perf_counter() - t0
            comp.invalid_flag = True

        return comp, dt
    
    # Calculate all points
    results = map_func(calculate_point, X)
    
    return X, results


class PerformanceMap:
    """
    Performance map generator and analyzer
    """
    
    def __init__(self, geom: Geometry, fld: str, in0: ThermoProp, loss_config: Dict = None, solver: str = "ccd"):
        """
        Initialize performance map
        
        Parameters
        ----------
        geom : Geometry
            Compressor geometry
        fld : str
            Fluid name
        in0 : ThermoProp
            Inlet thermodynamic state
        loss_config : dict
            Loss model configuration
        """
        self.geom = geom
        self.fld = fld
        self.in0 = in0
        self.loss_config = loss_config or {}
        self.solver = solver
        
        # Storage
        self.X = None
        self.compressors = []
        self.results_df = None
    
    def generate(
        self, 
        lb: np.ndarray = None, 
        ub: np. ndarray = None,
        resolution: List[float] = None,
        n_rot_range: np.ndarray = None,
        m_flow_range: np. ndarray = None,
        delta_check: bool = True,
    ):
        """
        Generate performance map
        
        Parameters
        ----------
        lb : np.ndarray
            Lower bounds [n_rot_min, m_flow_min]
        ub : np. ndarray
            Upper bounds [n_rot_max, m_flow_max]
        resolution : list
            Grid resolution [n_rot_resolution, m_flow_resolution]
        n_rot_range :  np.ndarray
            Explicit rotational speed range (overrides lb, ub)
        m_flow_range : np.ndarray
            Explicit mass flow range (overrides lb, ub)
        """
        # Set up ranges
        if n_rot_range is None or m_flow_range is None: 
            if lb is None or ub is None: 
                n_rot_max, m_flow_max = upper_bounds(self.geom, self.in0)
                lb = np.array([0.5 * n_rot_max, 0.1 * m_flow_max])
                ub = np.array([n_rot_max, m_flow_max])
            
            if resolution is None:
                resolution = [0.02, 0.02]
            
            n_rot_range = np.arange(lb[0], ub[0], resolution[0] * (ub[0] - lb[0]))
            m_flow_range = np.arange(lb[1], ub[1], resolution[1] * (ub[1] - lb[1]))
        
        # Generate grid
        self.X = np.array(list(itertools.product(n_rot_range, m_flow_range)))
        
        # Calculate all points
        print(f"Calculating {len(self.X)} operating points...")
        
        self.compressors = []
        comp_valid = []
        eta = []
        pr = []
        power = []
        flow_coeff = []
        work_coeff = []
        head = []
        m_in = []
        dtime = []
        reasons = []
        choke_flags = []
        surge_flags = []
        wet_flags = []
        invalid_flags = []
        
        for n_rot, m_flow in tqdm(self.X):
            op = OperatingCondition(
                in0=self.in0,
                fld=self. fld,
                m=m_flow,
                n_rot=n_rot
            )
            
            if self.solver == "radcomp":
                try:
                    from radcompressor.compressor import Compressor as RCCompressor
                    from radcompressor.geometry import Geometry as RCGeometry
                    from radcompressor.condition import OperatingCondition as RCOperatingCondition
                    from radcompressor.thermo import CoolPropFluid as RCCoolPropFluid
                except Exception:
                    comp = CentrifugalCompressor(self.geom, op, self.loss_config)
                else:
                    g_dict = self.geom.to_dict() if hasattr(self.geom, "to_dict") else self.geom.__dict__
                    blockage = None
                    if hasattr(self.geom, "blockage") and self.geom.blockage is not None:
                        blockage = list(getattr(self.geom, "blockage"))
                    elif isinstance(g_dict, dict):
                        blockage = [
                            g_dict.get(f"blockage{i+1}") for i in range(5)
                        ]
                        if any(b is None for b in blockage):
                            blockage = None
                    if blockage is None:
                        blockage = [1.0] * 5
                    rc_geom = RCGeometry.from_dict(g_dict, blockage=blockage)
                    rc_fld = RCCoolPropFluid(getattr(self.fld, "name", self.fld))
                    rc_in0 = self.in0  # already a ThermoProp; RC accepts fld on op
                    rc_op = RCOperatingCondition(in0=rc_in0, fld=rc_fld, m=m_flow, n_rot=n_rot)
                    comp = RCCompressor(rc_geom, rc_op)
            else:
                comp = CentrifugalCompressor(self.geom, op, self.loss_config)
            
            t0 = time.perf_counter()
            try:
                if hasattr(comp, "calculate"):
                    comp.calculate(delta_check=delta_check) if self.solver == "ccd" else comp.calculate()
                dt = time.perf_counter() - t0
            except Exception:
                dt = time.perf_counter() - t0
                comp.invalid_flag = True
            
            self.compressors.append(comp)
            comp_valid.append(not comp.invalid_flag)
            eta.append(comp.results.eff)
            pr.append(comp.results.PR)
            power.append(comp.results.power)
            flow_coeff.append(comp.results.flow_coeff)
            work_coeff.append(comp.results.work_coeff)
            head.append(comp.results.head)
            m_in.append(comp.results.m_in)
            dtime.append(dt)
            choke_flags.append(comp.results.choke)
            surge_flags.append(comp.results.surge)
            wet_flags.append(comp.results.wet)
            invalid_flags.append(comp.invalid_flag)

            reason = "valid"
            if comp.invalid_flag or not comp.results.valid:
                if comp.results.choke:
                    reason = "choke"
                elif comp.results.surge:
                    reason = "surge"
                elif comp.results.wet:
                    reason = "wet"
                elif not np.isfinite(comp.results.PR) or comp.results.PR < 1.0:
                    reason = "low_or_nan_PR"
                elif not np.isfinite(comp.results.eff):
                    reason = "nan_eff"
                else:
                    reason = "invalid"
            reasons.append(reason)
        
        # Create results dataframe
        self.results_df = pd.DataFrame({
            'n_rot': self.X[:, 0],
            'm_flow': self.X[:, 1],
            'valid': comp_valid,
            'efficiency': eta,
            'PR':  pr,
            'power': power,
            'flow_coefficient':  flow_coeff,
            'work_coefficient': work_coeff,
            'head': head,
            'M_inlet': m_in,
            'calc_time': dtime,
            'reason': reasons,
            'choke': choke_flags,
            'surge': surge_flags,
            'wet': wet_flags,
            'invalid_flag': invalid_flags,
        })
        
        print(f"Valid points: {sum(comp_valid)} / {len(self.X)}")
        
        return self.results_df
    
    def filter_valid(self):
        """Return only valid operating points"""
        if self.results_df is None:
            raise ValueError("Must generate map first")
        
        return self.results_df[self.results_df['valid'] == True]
    
    def get_speed_lines(self, n_speeds: int = 5):
        """
        Extract constant speed lines from performance map
        
        Parameters
        ----------
        n_speeds : int
            Number of speed lines to extract
        
        Returns
        -------
        dict
            Dictionary of speed lines
        """
        if self.results_df is None:
            raise ValueError("Must generate map first")
        
        valid_df = self.filter_valid()
        
        # Get unique rotational speeds
        unique_speeds = np.sort(valid_df['n_rot'].unique())
        
        # Select evenly spaced speeds
        if len(unique_speeds) > n_speeds:
            indices = np.linspace(0, len(unique_speeds) - 1, n_speeds, dtype=int)
            selected_speeds = unique_speeds[indices]
        else:
            selected_speeds = unique_speeds
        
        speed_lines = {}
        for speed in selected_speeds:
            speed_data = valid_df[valid_df['n_rot'] == speed]. sort_values('m_flow')
            speed_lines[speed] = speed_data
        
        return speed_lines
    
    def export_to_csv(self, filename: str):
        """Export results to CSV file"""
        if self.results_df is None:
            raise ValueError("Must generate map first")
        
        self.results_df.to_csv(filename, index=False)
        print(f"Exported to {filename}")


# Test case to check if performance map is correct
def run_radcomp_performance_plot(
    case_index: int = -1,
    mass: float = 0.12,
    speed_rpm: float = 130000.0,
    diffuser: str = "vaneless",
    resolution: Tuple[float, float] = (0.03, 0.03),
    n_speed_lines: int = 6,
    output_dir: Optional[Union[str, Path]] = None,
    n_rot_span: Tuple[float, float] = (0.7, 1.1),
    m_flow_span: Tuple[float, float] = (0.5, 1.2),
    delta_check: bool = True,
    print_reason_counts: bool = True,
    solver: str = "ccd",
):
    """
    Generate a RadComp database performance map and save CSV + PNG under output/.

    Uses a known compressor entry (radcomp-main/data/known_compressors.yml), builds a
    PerformanceMap grid, and plots with the existing RadComp-style plotting utility.
    """

    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "radcomp-main" / "data" / "known_compressors.yml"
    if not data_path.exists():
        raise FileNotFoundError(f"RadComp database not found at {data_path}")

    entries = yaml.safe_load(data_path.read_text())
    if not entries:
        raise ValueError("No compressor entries in known_compressors.yml")

    idx = case_index if case_index >= 0 else len(entries) + case_index
    if idx < 0 or idx >= len(entries):
        raise IndexError(f"case_index {case_index} is out of range for {len(entries)} entries")

    entry = entries[idx]
    geom = Geometry.from_dict(entry["geom"], blockage=entry["geom"].get("blockage") if isinstance(entry["geom"], dict) else None)

    cond = entry["conditions"]
    fluid = cond["fluid"]
    fld = CoolPropFluid(fluid)
    in0 = thermo_prop(fld, "PT", cond["in_P"], cond["in_T"])

    perf_map = PerformanceMap(geom, fld, in0, loss_config={"diffuser_type": diffuser}, solver=solver)

    base_n_rot = speed_rpm * np.pi / 30.0
    n_lo = max(1e-6, n_rot_span[0] * base_n_rot)
    n_hi = n_rot_span[1] * base_n_rot
    m_lo = max(1e-6, m_flow_span[0] * mass)
    m_hi = m_flow_span[1] * mass

    n_step = max(resolution[0] * (n_hi - n_lo), 1e-6)
    m_step = max(resolution[1] * (m_hi - m_lo), 1e-6)
    n_rot_range = np.arange(n_lo, n_hi + 0.5 * n_step, n_step)
    m_flow_range = np.arange(m_lo, m_hi + 0.5 * m_step, m_step)

    results_df = perf_map.generate(
        n_rot_range=n_rot_range,
        m_flow_range=m_flow_range,
        delta_check=delta_check,
    )
    speed_lines = perf_map.get_speed_lines(n_speed_lines)

    valid_points = int(results_df["valid"].sum()) if "valid" in results_df else 0

    if print_reason_counts and "reason" in results_df:
        print("Reason counts (including valid):")
        print(results_df["reason"].value_counts(dropna=False).to_string())

    out_dir = Path(output_dir) if output_dir else repo_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"performance_map_case{idx}_{stamp}.csv"
    png_path = out_dir / f"performance_map_case{idx}_{stamp}.png"

    perf_map.export_to_csv(str(csv_path))
    
    # Print summary statistics
    print(f"\n=== {entry.get('name', f'case_{idx}')} Performance Map Summary ===")
    print(f"Valid operating points: {valid_points}/{len(results_df)} ({100*valid_points/len(results_df):.1f}%)")
    
    figure_path: Optional[Path] = None
    if valid_points > 0:
        valid_df = results_df[results_df['valid'] == True]
        pr_valid = valid_df['PR'].values
        eta_valid = valid_df['efficiency'].values
        m_valid = valid_df['m_flow'].values
        n_valid = valid_df['n_rot'].values
        
        if len(pr_valid) > 0:
            print(f"Pressure ratio range: {np.nanmin(pr_valid):.2f} - {np.nanmax(pr_valid):.2f}")
        if len(eta_valid) > 0:
            print(f"Efficiency range: {np.nanmin(eta_valid):.3f} - {np.nanmax(eta_valid):.3f}")
        if len(m_valid) > 0:
            print(f"Operating mass flow range: {np.nanmin(m_valid):.1f} - {np.nanmax(m_valid):.1f} kg/s")
            print(f"Operating speed range: {np.nanmin(n_valid)/1000:.1f} - {np.nanmax(n_valid)/1000:.1f} krad/s")
        
        # Get design point if available
        design_pt = None
        if "design" in entry:
            design_pt = {
                'm': mass,
                'PR': entry["design"].get("PR", np.nanmean(pr_valid)),
                'eta': entry["design"].get("efficiency", np.nanmean(eta_valid)),
            }
        
        # Plot using the simple plotting function
        plot_compressor_map(
            n_rot_grid=results_df['n_rot'].values,
            m_flow_grid=results_df['m_flow'].values,
            pr_results=results_df['PR'].values,
            eta_results=results_df['efficiency'].values,
            valid_results=results_df['valid'].values,
            design_point=design_pt,
            title=entry.get('name', f'case_{idx}'),
            save_path=str(png_path)
        )
        figure_path = png_path
        
        # Print speed lines tested
        unique_speeds = np.sort(np.unique(results_df['n_rot'].values))
        n_display = min(len(unique_speeds), 10)
        print(f"\nSpeed lines tested ({n_display} shown):" )
        for n_rot in unique_speeds[::max(1, len(unique_speeds)//n_display)][:n_display]:
            rpm = n_rot * 30.0 / np.pi
            print(f"  {n_rot/1000:.1f} krad/s ({rpm:.0f} rpm)")
    else:
        print("No valid points to plot; PNG not saved. Adjust n_rot_span/m_flow_span/resolution if needed.")

    return {
        "case_name": entry.get("name", f"case_{idx}"),
        "csv": csv_path,
        "figure": figure_path,
        "n_points": len(results_df),
        "valid_points": valid_points,
    }