"""
Plotting utilities for performance maps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Tuple

def plot_performance_map(
    results_df,
    plot_type: str = "efficiency",
    speed_lines: Dict = None,
    figsize:  Tuple[int, int] = (12, 8),
    save_path: str = None,
):
    """
    Plot compressor performance map
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from PerformanceMap
    plot_type : str
        Type of plot:  'efficiency', 'PR', 'power', 'head'
    speed_lines : dict
        Dictionary of speed lines from get_speed_lines()
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    """
    # Filter valid points
    valid_df = results_df[results_df['valid'] == True]
    
    if len(valid_df) == 0:
        print("No valid points to plot")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot 1: Efficiency vs mass flow
    ax = axes[0]
    if speed_lines is not None:
        for speed, data in speed_lines.items():
            rpm = speed * 60 / (2 * np.pi)
            ax.plot(
                data['m_flow'], 
                data['efficiency'] * 100,
                marker='o',
                label=f'{rpm:.0f} RPM'
            )
    else:
        scatter = ax.scatter(
            valid_df['m_flow'],
            valid_df['efficiency'] * 100,
            c=valid_df['n_rot'],
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Rotational Speed (rad/s)')
    
    ax.set_xlabel('Mass Flow Rate (kg/s)')
    ax.set_ylabel('Isentropic Efficiency (%)')
    ax.set_title('Efficiency Map')
    ax.grid(True, alpha=0.3)
    if speed_lines is not None:
        ax.legend()
    
    # Plot 2: Pressure ratio vs mass flow
    ax = axes[1]
    if speed_lines is not None: 
        for speed, data in speed_lines.items():
            rpm = speed * 60 / (2 * np.pi)
            ax.plot(
                data['m_flow'],
                data['PR'],
                marker='o',
                label=f'{rpm:.0f} RPM'
            )
    else:
        scatter = ax.scatter(
            valid_df['m_flow'],
            valid_df['PR'],
            c=valid_df['n_rot'],
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Rotational Speed (rad/s)')
    
    ax.set_xlabel('Mass Flow Rate (kg/s)')
    ax.set_ylabel('Pressure Ratio')
    ax.set_title('Pressure Ratio Map')
    ax.grid(True, alpha=0.3)
    if speed_lines is not None: 
        ax.legend()
    
    # Plot 3: Non-dimensional plot (head vs flow coefficient)
    ax = axes[2]
    if speed_lines is not None:
        for speed, data in speed_lines.items():
            rpm = speed * 60 / (2 * np.pi)
            ax.plot(
                data['flow_coefficient'],
                data['head'],
                marker='o',
                label=f'{rpm:.0f} RPM'
            )
    else:
        scatter = ax.scatter(
            valid_df['flow_coefficient'],
            valid_df['head'],
            c=valid_df['efficiency'],
            cmap='RdYlGn',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Efficiency')
    
    ax.set_xlabel('Flow Coefficient')
    ax.set_ylabel('Head Coefficient')
    ax.set_title('Non-dimensional Performance')
    ax.grid(True, alpha=0.3)
    if speed_lines is not None: 
        ax.legend()
    
    # Plot 4: Efficiency contours
    ax = axes[3]
    if len(valid_df) > 10:
        from scipy.interpolate import griddata
        
        # Create grid
        n_rot_grid = np.linspace(
            valid_df['n_rot'].min(),
            valid_df['n_rot'].max(),
            50
        )
        m_flow_grid = np.linspace(
            valid_df['m_flow'].min(),
            valid_df['m_flow'].max(),
            50
        )
        N_GRID, M_GRID = np. meshgrid(n_rot_grid, m_flow_grid)
        
        # Interpolate efficiency
        eff_grid = griddata(
            (valid_df['n_rot'], valid_df['m_flow']),
            valid_df['efficiency'] * 100,
            (N_GRID, M_GRID),
            method='linear'
        )
        
        # Plot contours
        contour = ax.contourf(
            M_GRID, N_GRID * 60 / (2 * np.pi),
            eff_grid,
            levels=15,
            cmap='RdYlGn'
        )
        plt.colorbar(contour, ax=ax, label='Efficiency (%)')
        ax.contour(M_GRID, N_GRID * 60 / (2 * np.pi), eff_grid, levels=10, colors='k', alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel('Mass Flow Rate (kg/s)')
    ax.set_ylabel('Rotational Speed (RPM)')
    ax.set_title('Efficiency Contours')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_component_analysis(compressor, figsize: Tuple[int, int] = (14, 10)):
    """
    Plot detailed component analysis for a single operating point
    
    Parameters
    ----------
    compressor : CentrifugalCompressor
        Calculated compressor object
    figsize :  tuple
        Figure size
    """
    if compressor.invalid_flag:
        print("Invalid compressor - cannot plot")
        return
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Meridional view
    ax1 = fig. add_subplot(gs[0, : ])
    plot_meridional_view(ax1, compressor)
    
    # Plot 2: Pressure distribution
    ax2 = fig. add_subplot(gs[1, 0])
    plot_pressure_distribution(ax2, compressor)
    
    # Plot 3: Temperature distribution
    ax3 = fig.add_subplot(gs[1, 1])
    plot_temperature_distribution(ax3, compressor)
    
    # Plot 4: Velocity triangles
    ax4 = fig.add_subplot(gs[1, 2])
    plot_velocity_triangles(ax4, compressor)
    
    # Plot 5: Loss breakdown
    ax5 = fig. add_subplot(gs[2, 0])
    plot_loss_breakdown(ax5, compressor)
    
    # Plot 6: Efficiency through components
    ax6 = fig. add_subplot(gs[2, 1])
    plot_component_efficiency(ax6, compressor)
    
    # Plot 7: Performance summary
    ax7 = fig. add_subplot(gs[2, 2])
    plot_performance_summary(ax7, compressor)
    
    plt.suptitle('Centrifugal Compressor Component Analysis', fontsize=16, fontweight='bold')
    plt.show()


def plot_meridional_view(ax, compressor):
    """Plot meridional view of compressor"""
    geom = compressor.geom
    
    # Draw hub line
    ax.plot([0, geom.L_ind], [geom.r1h, geom.r2h], 'k-', linewidth=2, label='Hub')
    ax.plot([geom.L_ind, geom.L_ind + 0.05], [geom.r2h, geom.r2h], 'k-', linewidth=2)
    
    # Draw shroud line
    ax.plot([0, geom.L_ind], [geom.r1s, geom.r2s], 'k-', linewidth=2, label='Shroud')
    ax.plot([geom.L_ind, geom.L_ind + 0.05], [geom.r2s, geom.r4], 'k-', linewidth=2)
    
    # Draw impeller outlet
    ax.plot([geom.L_ind + 0.05, geom.L_ind + 0.05], 
            [geom.r4 - geom.b4, geom.r4], 'k-', linewidth=2)
    
    # Draw diffuser
    ax.plot([geom.L_ind + 0.05, geom.L_ind + 0.1], [geom.r4, geom.r5], 'b--', linewidth=1.5, label='Diffuser')
    
    ax.set_xlabel('Axial Position (m)')
    ax.set_ylabel('Radius (m)')
    ax.set_title('Meridional View')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_pressure_distribution(ax, compressor):
    """Plot pressure distribution through compressor"""
    stations = ['Inlet', 'Inducer Out', 'Impeller Out', 'Diffuser Out']
    pressures = [
        compressor.inlet.total. P / 1e5,
        compressor.ind. out.total.P / 1e5,
        compressor.imp.out.total.P / 1e5,
        compressor.outlet.total.P / 1e5,
    ]
    
    ax.plot(stations, pressures, 'bo-', linewidth=2, markersize=8)
    ax.set_ylabel('Total Pressure (bar)')
    ax.set_title('Pressure Distribution')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_temperature_distribution(ax, compressor):
    """Plot temperature distribution through compressor"""
    stations = ['Inlet', 'Inducer Out', 'Impeller Out', 'Diffuser Out']
    temperatures = [
        compressor.inlet.total.T - 273.15,
        compressor.ind.out.total.T - 273.15,
        compressor.imp.out.total.T - 273.15,
        compressor.outlet.total.T - 273.15,
    ]
    
    ax.plot(stations, temperatures, 'ro-', linewidth=2, markersize=8)
    ax.set_ylabel('Total Temperature (°C)')
    ax.set_title('Temperature Distribution')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_velocity_triangles(ax, compressor):
    """Plot velocity triangles at key stations"""
    # Simplified velocity triangle visualization
    ax.text(0.5, 0.5, 'Velocity\nTriangles', 
            ha='center', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Velocity Triangles')


def plot_loss_breakdown(ax, compressor):
    """Plot loss breakdown for impeller"""
    losses = compressor.imp.losses
    
    loss_types = ['Incidence', 'Skin Friction', 'Blade Loading', 
                  'Clearance', 'Disc Friction', 'Recirculation']
    loss_values = [
        losses.incidence / 1000,
        losses.skin_friction / 1000,
        losses.blade_loading / 1000,
        losses.clearance / 1000,
        losses. disc_friction / 1000,
        losses.recirculation / 1000,
    ]
    
    ax.barh(loss_types, loss_values, color='coral')
    ax.set_xlabel('Loss (kJ/kg)')
    ax.set_title('Impeller Loss Breakdown')
    ax.grid(True, alpha=0.3, axis='x')


def plot_component_efficiency(ax, compressor):
    """Plot efficiency of each component"""
    components = ['Inducer', 'Impeller', 'Diffuser', 'Overall']
    efficiencies = [
        compressor.ind. eff * 100,
        compressor. imp.eff * 100,
        compressor. dif.eff * 100,
        compressor.results.eff * 100,
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    ax.bar(components, efficiencies, color=colors, edgecolor='black')
    ax.set_ylabel('Isentropic Efficiency (%)')
    ax.set_title('Component Efficiency')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])


def plot_performance_summary(ax, compressor):
    """Plot performance summary"""
    summary = compressor.get_summary()
    
    text_str = (
        f"Pressure Ratio: {summary['PR']:.3f}\n"
        f"Efficiency: {summary['efficiency']*100:.2f}%\n"
        f"Power: {summary['power_kW']:.2f} kW\n"
        f"Tip Speed:  {summary['tip_speed']:.1f} m/s\n"
        f"Flow Coeff: {summary['flow_coefficient']:.4f}\n"
        f"Work Coeff: {summary['work_coefficient']:.4f}\n"
        f"Ns:  {summary['Ns']:.4f}\n"
        f"Ds: {summary['Ds']:. 4f}"
    )
    
    ax.text(0.1, 0.5, text_str, fontsize=11, family='monospace',
            verticalalignment='center')
    ax.axis('off')
    ax.set_title('Performance Summary')