"""Single-phase reel-out optimization and simulation.

This module provides the ReeloutSimple class for optimizing and simulating
reel-out maneuvers for airborne wind energy systems.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import copy
import warnings
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from phase_parametrized import PhaseParameterized


@dataclass
class SimulationResult:
    """Container for simulation results and optimization outputs."""

    solution: Any  # CasADi solution object
    optimized_config: Dict[str, Any]
    final_distance: float
    phase_variables: Dict[str, Any]
    energy_objective: float
    total_time: float


class Reelout:
    """Handles single-phase reel-out optimization and simulation.

    This class manages the optimization and simulation of a reel-out maneuver
    with configurable pattern type, path parameters, and radial parameters.

    Example:
        >>> config = {
        ...     "pattern_type": "figure8",  # or "circle", "helix", etc.
        ...     "path_parameters": {
        ...         "distance_radial_start": 100,  # Required
        ...         "distance_radial_end": 360,    # Required
        ...     },
        ...     "radial_parameters": {
        ...         "vr": 0.5  # Example parameter
        ...     }
        ... }
        >>> reelout = ReeloutSimple(
        ...     system_model=my_model,
        ...     pattern_config=config,
        ...     depower=1.0
        ... )
        >>> result = reelout.run_simulation_opti()
    """

    def __init__(
        self,
        *,
        system_model: Any,  # Should be SystemModel but avoiding circular import
        pattern_config: Optional[Dict[str, Any]] = None,
        depower: float = 0.0,
    ) -> None:
        """Initialize ReeloutSimple instance.

        Args:
            system_model: The system model to use for simulation/optimization
            pattern_config: Configuration dictionary with pattern_type, path_parameters,
                          and radial_parameters
            depower: Depower setting for the kite (0 to 1)
        """
        self.pattern_config = pattern_config or {}
        self._required_config = {
            "pattern_type": None,  # Must be provided
            "path_parameters": {},
            "radial_parameters": {},  # Optional with defaults
        }
        # self._validate_config()

        self.depower = depower
        self.system_model = system_model

        # Derived configuration/state containers
        self.variables_to_plot = [
            "speed_tangential",
            "tension_tether_ground",
            "angle_of_attack",
            "speed_radial",
        ]
        self._opti_params = {}

    # def _validate_config(self) -> None:
    #     """Validate the pattern configuration and warn about missing required parameters."""
    #     missing_required = []
    #     using_defaults = []

    #     def check_section(required: Dict, actual: Dict, path: str = "") -> None:
    #         for key, default in required.items():
    #             current_path = f"{path}.{key}" if path else key
    #             if isinstance(default, dict):
    #                 # Recursively check nested dictionaries
    #                 if key not in actual:
    #                     if all(v is None for v in default.values()):
    #                         missing_required.append(current_path)
    #                     actual[key] = {}
    #                 check_section(default, actual[key], current_path)
    #             else:
    #                 if key not in actual:
    #                     if default is None:
    #                         missing_required.append(current_path)
    #                     else:
    #                         actual[key] = default
    #                         using_defaults.append(f"{current_path} = {default}")

    #     check_section(self._required_config, self.pattern_config)

    #     if missing_required:
    #         missing_str = "\n  - ".join(missing_required)
    #         raise ValueError(
    #             f"Missing required configuration parameters:\n  - {missing_str}"
    #         )

    #     if using_defaults:
    #         defaults_str = "\n  - ".join(using_defaults)
    #         warnings.warn(
    #             f"Using default values for configuration parameters:\n  - {defaults_str}",
    #             RuntimeWarning,
    #             stacklevel=2,
    #         )

    def initialize_phase(self) -> PhaseParameterized:
        """Initialize and prepare the optimization phase."""
        self.system_model.input_depower = self.depower

        pattern_config_opti = copy.deepcopy(self.pattern_config)
        start_state = {
            "t": 0,
            "s": 0,
            "s_dot": 2,
            "input_steering": 0,
            "tension_tether_ground": 1e10,
            "distance_radial": self.pattern_config["path_parameters"][
                "distance_radial_start"
            ],
            "speed_radial": 0,  # Positive for reel-out
        }

        pattern_config_opti = copy.deepcopy(self.pattern_config)
        start_state_opti = copy.deepcopy(start_state)
        for var_name, mx in self._opti_params.items():
            for entry in ["path_parameters", "radial_parameters", "sim_parameters"]:
                if var_name in pattern_config_opti.get(entry, {}):
                    pattern_config_opti[entry][var_name] = mx
        self._phase = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
            pattern_config=self.pattern_config,
            pattern_config_opti=pattern_config_opti,
        )

        return self._phase

    def get_opti_components(
        self,
        optimization_params: List[str] = None,
        optimization_dict: Dict[str, Any] = None,
        opti: Any = None,
    ) -> tuple:
        """Get optimization components (optimizer, variables, objective).

        Args:
            optimization_params: List of parameter names to optimize
            opti: Optional existing CasADi Opti instance

        Returns:
            Tuple of (optimizer, variables dict, objective dict, param dict)
        """
        if opti is None:
            opti = ca.Opti()
        self._opti = opti
        self._opti_params = {}

        if optimization_params:
            for var in optimization_params:
                self._opti_params[var] = opti.variable()
            if "coeffs" in var:
                num_coeffs = len(self.pattern_config["path_parameters"].get(var, []))
                self._opti_params[var] = opti.variable(num_coeffs)
        elif optimization_dict:
            self._opti_params = optimization_dict

        self.initialize_phase()

        return self._opti, self._opti_vars, self._objective, self._opti_params

    def run_simulation_opti(
        self, optimization_params: List[str] = None, target: str = "power"
    ) -> Optional[SimulationResult]:
        """Run optimization and return results.

        Args:
            optimization_params: List of parameters to optimize

        Returns:
            SimulationResult object or None if optimization failed
        """
        opti, opti_vars, objective_dict, self._opti_params = self.get_opti_components(
            optimization_params=optimization_params
        )

        # Maximize average power
        if target == "power":
            total_objective = -(
                objective_dict["energy"]
                / objective_dict["total_time"]
                / objective_dict["power_scale"]
            )
        elif target == "energy":
            total_objective = -objective_dict["energy"]
        elif target == "zero":
            total_objective = 0.0

        solution = self.run_opti(opti, total_objective)
        if solution is None:
            return None

        return SimulationResult(
            solution=solution,
            optimized_config=self.pattern_config,
            final_distance=objective_dict.get("distance_radial_final", 0.0),
            phase_variables=opti_vars,
            energy_objective=objective_dict.get("energy", 0.0),
            total_time=objective_dict.get("total_time", 0.0),
        )

    def run_opti(self, opti: Any, objective: Any) -> Optional[Any]:
        """Run the optimization problem.

        Args:
            opti: CasADi Opti instance
            objective: Objective function to minimize

        Returns:
            Solution object or None if optimization failed
        """
        opti.minimize(objective)
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "bound_relax_factor": 1e-8,
                    "tol": 1e-4,
                    "acceptable_iter": 3,
                    "acceptable_tol": 1e-4,
                    "constr_viol_tol": 1e-4,
                    "dual_inf_tol": 1e-4,
                    "hessian_approximation": "limited-memory",
                    "mu_strategy": "adaptive",
                }
            },
        )

        try:
            solution = opti.solve()

            print("\nOptimized Pattern Variables:")
            optimized_config = self.pattern_config.copy()
            for var_name, mx in self._opti_params.items():
                val = solution.value(mx)
                print(f"  {var_name}: {val}")
                if var_name in optimized_config.get("path_parameters", {}):
                    optimized_config["path_parameters"][var_name] = val
                elif var_name in optimized_config.get("radial_parameters", {}):
                    optimized_config["radial_parameters"][var_name] = val
                elif var_name in optimized_config.get("sim_parameters", {}):
                    optimized_config["sim_parameters"][var_name] = val
            self.pattern_config = optimized_config
            return solution

        except Exception as exc:
            print("Debug optimization information:")
            for var_name, mx in self._opti_params.items():
                try:
                    print(f"  {var_name}: {opti.debug.value(mx)}")
                except Exception:
                    pass
            print("Optimization failed:", exc)
            return None

    def run_simulation(
        self, *, run_plots: bool = False, axes: Any = None, variables_to_plot=None
    ) -> None:
        """Execute the reel-out simulation.

        Args:
            solution: Optional CasADi solution from optimization
            run_plots: When True, produce overview plots
        """
        self.initialize_phase()
        self.system_model.input_depower = self.depower

        phase = self._run_parametrized_phase(
            label_prefix="a",
            pattern_config=self.pattern_config,
            phase_sym=True,
        )

        if run_plots:
            variables_to_plot = variables_to_plot or self.variables_to_plot
            if axes is not None:
                fig, axes, slider = phase.plot_overview_3d(
                    x_param="t",
                    variables=variables_to_plot,
                    axes=axes,
                )
            else:
                fig, axes, slider = phase.plot_overview_3d(
                    x_param="t",
                    variables=variables_to_plot,
                )
            metrics = phase.energy_metrics()
            power = metrics["avg_power"]
            fig.suptitle(
                f"Reel-out Simulation Overview (Avg Power: {power/1e3:.2f} kW)"
            )

        return phase, axes, slider

    def _run_parametrized_phase(
        self,
        label_prefix: str,
        pattern_config: Dict[str, Any],
        phase_sym: bool = False,
    ) -> PhaseParameterized:
        """Run a parametrized phase simulation.

        Args:
            label_prefix: Prefix for labeling outputs
            pattern_config: Configuration for this phase
            phase_sym: Whether to run in symbolic mode

        Returns:
            PhaseParameterized object with simulation results
        """
        sim_type = "quasi steady"
        print(f"Running simulation for {sim_type} with label: {label_prefix}")

        start_state = {
            "t": 0,
            "s": 0,
            "s_dot": 2,
            "input_steering": 0,
            "tension_tether_ground": 1e10,
            "distance_radial": pattern_config["path_parameters"][
                "distance_radial_start"
            ],
            "speed_radial": 0,  # Positive for reel-out
        }

        phase = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
            pattern_config=pattern_config,
        )
        if phase_sym:
            phase.run_simulation_phase(start_state=start_state)
        else:
            phase.run_simulation(start_state=start_state)
        return phase
