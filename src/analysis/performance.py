"""
Performance analysis tools
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Iterator
import itertools
import time
from tqdm import tqdm

from geometry.geometry import Geometry
from conditions.operating import OperatingCondition, ThermoProp, thermo_prop
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


def calculate_on_op_grid(
    geom: Geometry,
    in0: ThermoProp,
    fld: str,
    lb: np.ndarray,
    ub: np.ndarray,
    resolution: float = 0.02,
    loss_config: Dict = None,
    map_func=map,
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
        
        # Create operating condition
        op = OperatingCondition(
            in0=in0,
            fld=fld,
            m=m_flow,
            n_rot=n_rot
        )
        
        # Create and calculate compressor
        comp = CentrifugalCompressor(geom, op, loss_config)
        
        t0 = time.perf_counter()
        try:
            comp.calculate()
            dt = time.perf_counter() - t0
        except Exception as e:
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
    
    def __init__(self, geom: Geometry, fld: str, in0: ThermoProp, loss_config: Dict = None):
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
        
        # Storage
        self.X = None
        self.compressors = []
        self. results_df = None
    
    def generate(
        self, 
        lb: np.ndarray = None, 
        ub: np. ndarray = None,
        resolution: List[float] = None,
        n_rot_range: np.ndarray = None,
        m_flow_range: np. ndarray = None,
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
        
        for n_rot, m_flow in tqdm(self.X):
            op = OperatingCondition(
                in0=self.in0,
                fld=self. fld,
                m=m_flow,
                n_rot=n_rot
            )
            
            comp = CentrifugalCompressor(self.geom, op, self.loss_config)
            
            t0 = time.perf_counter()
            try:
                comp.calculate()
                dt = time.perf_counter() - t0
            except Exception: 
                dt = time.perf_counter() - t0
                comp.invalid_flag = True
            
            self.compressors.append(comp)
            comp_valid.append(not comp.invalid_flag)
            eta.append(comp.results. eff)
            pr.append(comp.results.PR)
            power.append(comp.results. power)
            flow_coeff.append(comp.results.flow_coeff)
            work_coeff.append(comp.results.work_coeff)
            head.append(comp.results. head)
            m_in.append(comp.results.m_in)
            dtime.append(dt)
        
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
        })
        
        print(f"Valid points: {sum(comp_valid)} / {len(self.X)}")
        
        return self.results_df
    
    def filter_valid(self):
        """Return only valid operating points"""
        if self.results_df is None:
            raise ValueError("Must generate map first")
        
        return self.results_df[self. results_df['valid'] == True]
    
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