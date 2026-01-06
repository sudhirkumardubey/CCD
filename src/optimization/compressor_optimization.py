"""
Centrifugal compressor design optimization (CCD).

This is a lightweight adaptation of TurboFlow's axial-turbine optimizer
pattern, wired to the CCD compressor solver. It expects a configuration dict
with a `design_optimization` block specifying objective, design variables,
constraints, and solver options. Geometry and operating baselines come from
`config["geometry"]` and `config["operating"]`.

The fitness method is intentionally written with clear steps so you can adapt
variable mapping and constraints to your project needs.
"""

from __future__ import annotations

import copy
import datetime
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import pysolver_view as psv
from core.compressor import CentrifugalCompressor
from conditions.operating import CoolPropFluid, OperatingCondition, thermo_prop
from geometry.geometry import Geometry

# Re-export solver families for convenience
GRADIENT_ALGORITHMS = psv.GRADIENT_SOLVERS
GENETIC_ALGORITHMS = psv.GENETIC_SOLVERS


class CompressorOptimizationProblem(psv.OptimizationProblem):
    """
    Define the compressor design optimization problem.

    Expected config shape (minimal):
    {
      "geometry": {... baseline Geometry kwargs ...},
      "operating": {"fluid": "Air", "in_P": 101325, "in_T": 300, "mass": 0.12, "speed_rpm": 130000},
      "loss_config": {... optional ...},
      "design_optimization": {
        "objective_function": {"variable": "summary.efficiency", "scale": 1.0, "type": "maximize"},
        "variables": {"geom.r4": {"value": 0.02, "lower_bound": 0.018, "upper_bound": 0.023}, ...},
        "constraints": [
            {"variable": "summary.PR", "value": 2.5, "type": ">", "normalize": True},
        ],
        "solver_options": {"library": "scipy", "method": "slsqp", "options": {"maxiter": 100}},
      }
    }
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        design_cfg = config["design_optimization"]

        self.objective = self._parse_objective(design_cfg["objective_function"])
        self.eq_constraints, self.ineq_constraints = self._parse_constraints(
            design_cfg.get("constraints", [])
        )

        variables = design_cfg["variables"]
        fixed_params, design_variables = self._split_fixed_and_design(variables)
        self.fixed_params = self._index_variables(fixed_params)
        indexed_design = self._index_variables(design_variables)

        self.design_variable_keys = list(indexed_design.keys())
        lb, ub = self._collect_bounds(design_variables)
        self.bounds = (lb, ub)

        # Single start; extend with LHS/multistart if needed.
        self.initial_guesses = [np.array(list(indexed_design.values()), dtype=float)]

        algorithm = design_cfg["solver_options"]["method"]
        if algorithm in GRADIENT_ALGORITHMS:
            self._fitness_impl = self._fitness_gradient_based
        elif algorithm in GENETIC_ALGORITHMS:
            self._fitness_impl = self._fitness_black_box
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    # --- OptimizationProblem API ---
    def fitness(self, x: np.ndarray):
        return self._fitness_impl(x)

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self.bounds

    def get_nec(self) -> int:
        return psv.count_constraints(self.eq_constraints)

    def get_nic(self) -> int:
        return psv.count_constraints(self.ineq_constraints)

    # --- Fitness implementations ---
    def _fitness_gradient_based(self, x: np.ndarray):
        # Map design vector to geometry/operating inputs
        design_vars = dict(zip(self.design_variable_keys, x.tolist()))
        geom_kwargs, op_kwargs = self._build_case_inputs(design_vars)

        comp, summary = self._evaluate_compressor(geom_kwargs, op_kwargs)

        # Objective
        f_raw = self._get_nested_value(summary, self.objective["variable"], default=1e6)
        f = f_raw / self.objective["scale"]

        # Constraints
        c_eq = []
        for c in self.eq_constraints:
            val = self._get_nested_value(summary, c["variable"], default=np.nan)
            c_eq.append((val - c["value"]) / c["scale"])

        c_ineq = []
        for c in self.ineq_constraints:
            val = self._get_nested_value(summary, c["variable"], default=np.nan)
            c_ineq.append((val - c["value"]) / c["scale"])

        # Penalize invalid solutions softly if comp failed
        if not summary.get("valid", False):
            f = float(np.abs(f) + 1e6)
            c_eq = [1e6 if not np.isfinite(v) else v for v in c_eq]
            c_ineq = [1e6 if not np.isfinite(v) else v for v in c_ineq]

        return psv.combine_objective_and_constraints(f, c_eq, c_ineq)

    def _fitness_black_box(self, x: np.ndarray):
        # For now reuse the gradient-based logic; penalty handling is already inside
        return self._fitness_gradient_based(x)

    # --- Helpers ---
    def _build_case_inputs(self, design_vars: Dict[str, float]):
        # Start from baselines
        geom_kwargs = copy.deepcopy(self.config["geometry"])
        op_kwargs = copy.deepcopy(self.config["operating"])

        ratio_vars: Dict[str, float] = {}

        # Apply fixed params
        for key, val in self.fixed_params.items():
            if key.startswith("ratio."):
                ratio_vars[key] = val
            else:
                self._apply_var(key, val, geom_kwargs, op_kwargs)

        # Apply design vars
        for key, val in design_vars.items():
            if key.startswith("ratio."):
                ratio_vars[key] = val
            else:
                self._apply_var(key, val, geom_kwargs, op_kwargs)

        # Apply ratio-derived geometry updates after base values are set
        if ratio_vars:
            self._apply_ratio_vars(ratio_vars, geom_kwargs, op_kwargs)

        return geom_kwargs, op_kwargs

    def _apply_var(self, key: str, val: float, geom_kwargs: Dict, op_kwargs: Dict):
        """Set a variable onto geometry or operating dict using prefixes or key presence."""
        if key.startswith("ratio."):
            return  # handled separately
        if key.startswith("geom."):
            geom_kwargs[key.split("geom.", 1)[1]] = val
        elif key.startswith("op."):
            op_kwargs[key.split("op.", 1)[1]] = val
        elif key in geom_kwargs:
            geom_kwargs[key] = val
        elif key in op_kwargs:
            op_kwargs[key] = val
        else:
            # Fallback: nested set using dots
            parts = key.split(".")
            if parts[0] in ("geom", "geometry"):
                self._set_nested(geom_kwargs, parts[1:], val)
            elif parts[0] in ("op", "operating"):
                self._set_nested(op_kwargs, parts[1:], val)
            else:
                geom_kwargs[key] = val  # best-effort

    def _apply_ratio_vars(self, ratio_vars: Dict[str, float], geom_kwargs: Dict, op_kwargs: Dict):
        """Map ratio.* variables to geometry dimensions using current geom_kwargs."""
        r4 = geom_kwargs.get("r4")
        if r4 is None:
            return

        r2s_ratio = ratio_vars.get("ratio.r2s_over_r4")
        if r2s_ratio is not None:
            geom_kwargs["r2s"] = r2s_ratio * r4

        r2h_ratio = ratio_vars.get("ratio.r2h_over_r2s")
        if r2h_ratio is not None:
            base_r2s = geom_kwargs.get("r2s")
            if base_r2s is None and r2s_ratio is not None:
                base_r2s = r2s_ratio * r4
            if base_r2s is not None:
                geom_kwargs["r2h"] = r2h_ratio * base_r2s

        b4_ratio = ratio_vars.get("ratio.b4_over_r4")
        if b4_ratio is not None:
            geom_kwargs["b4"] = b4_ratio * r4

    def _set_nested(self, target: Dict[str, Any], path: List[str], val: Any):
        cur = target
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = val

    def _evaluate_compressor(self, geom_kwargs: Dict, op_kwargs: Dict):
        # Build Geometry
        geom = Geometry(**geom_kwargs)

        # Build inlet thermodynamic state
        fluid = CoolPropFluid(op_kwargs["fluid"])
        in0 = thermo_prop(fluid, "PT", op_kwargs["in_P"], op_kwargs["in_T"])

        # Build OperatingCondition
        op = OperatingCondition(
            in0=in0,
            fld=fluid,
            m=op_kwargs["mass"],
            n_rot=op_kwargs["speed_rpm"] * np.pi / 30.0,
        )

        comp = CentrifugalCompressor(geom, op, loss_config=self.config.get("loss_config", {}))
        ok = comp.calculate()
        summary = comp.get_summary()
        if not ok:
            summary["valid"] = False
        return comp, summary

    def _parse_objective(self, objective: Dict[str, Any]) -> Dict[str, Any]:
        obj = copy.deepcopy(objective)
        if obj.get("type", "minimize") == "maximize":
            obj["scale"] = -abs(obj["scale"])
        else:
            obj["scale"] = abs(obj["scale"])
        return obj

    def _parse_constraints(self, constraints: List[Dict[str, Any]]):
        eq_constraints, ineq_constraints = [], []
        for c in constraints:
            scale = abs(c["value"]) if c.get("normalize") else 1.0
            entry = {
                "variable": c["variable"],
                "value": c["value"],
                "scale": scale,
            }
            if c["type"] == "=":
                eq_constraints.append(entry)
            elif c["type"] == "<":
                ineq_constraints.append(entry)
            elif c["type"] == ">":
                entry["scale"] *= -1.0
                ineq_constraints.append(entry)
            else:
                raise ValueError(f"Unknown constraint type: {c['type']}")
        return eq_constraints, ineq_constraints

    def _split_fixed_and_design(self, variables: Dict[str, Any]):
        fixed_params, design_vars = {}, {}
        for key, spec in variables.items():
            if spec.get("lower_bound") is None:
                fixed_params[key] = spec
            else:
                design_vars[key] = spec
        return fixed_params, design_vars

    def _index_variables(self, variables: Dict[str, Any]):
        indexed = {}
        for key, spec in variables.items():
            val = spec["value"]
            if isinstance(val, (list, tuple, np.ndarray)):
                for i, v in enumerate(val):
                    indexed[f"{key}_{i+1}"] = v
            else:
                indexed[key] = val
        return indexed

    def _collect_bounds(self, design_vars: Dict[str, Any]):
        lb, ub = [], []
        for spec in design_vars.values():
            val = spec["value"]
            if isinstance(val, (list, tuple, np.ndarray)):
                lb += list(spec["lower_bound"])
                ub += list(spec["upper_bound"])
            else:
                lb.append(spec["lower_bound"])
                ub.append(spec["upper_bound"])
        return lb, ub

    def _get_nested_value(self, data: Dict[str, Any], path: str, default=np.nan):
        cur = data
        if path.startswith("summary."):
            path = path.split("summary.", 1)[1]
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def compute_optimal_compressor(
    config: Dict[str, Any],
    out_filename: str | None = None,
    out_dir: str = "output",
    export_results: bool = True,
    logger=None,
):
    """
    Entry point mirroring turboflow.compute_optimal_turbine but for CCD.

    Returns the OptimizationSolver instance (contains solution and history).
    """
    problem = CompressorOptimizationProblem(config)

    # Seed constraints/objective once to initialize shapes
    problem.fitness(problem.initial_guesses[0])

    solver_cfg = config["design_optimization"]["solver_options"]
    solver = psv.OptimizationSolver(problem, **solver_cfg, logger=logger)
    solver.solve(problem.initial_guesses[0])

    if export_results:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if out_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_filename = f"compressor_design_optimization_{timestamp}"

        cfg_copy = copy.deepcopy(config)
        cfg_path = os.path.join(out_dir, f"{out_filename}.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(yaml_dump(cfg_copy))

    return solver


def yaml_dump(data: Dict[str, Any]) -> str:
    """Minimal YAML dumper (avoids bringing ruamel dependency here)."""
    try:
        import yaml as _yaml
    except ImportError:  # pragma: no cover
        raise ImportError("PyYAML is required to export results")
    return _yaml.safe_dump(data, sort_keys=False)
