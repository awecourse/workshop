from matplotlib import pyplot as plt
from timeseries import TimeSeries
from system_model import SystemModel
from parametrized_patterns import create_pattern_from_dict
from kinematics import ParametrizedKinematics
import casadi as ca
import numpy as np
import copy
from system_model import State
from kite import Kite
from winch import Winch
import logging


logger = logging.getLogger(__name__)


class PhaseParameterized(TimeSeries):
    def __init__(
        self,
        kite_model: SystemModel,
        quasi_steady: bool = False,
        pattern_config: dict = None,
        pattern_config_opti: dict = None,
        sharpness_beta: float = 1e-4,
        tension_min: float = 0.0,
        tension_max: float = 1e5,
    ):
        """
        Args:

        """

        super().__init__(
            kite_model=kite_model,
        )
        self.pattern_config = pattern_config
        if not pattern_config_opti:
            self.pattern_config_opti = copy.deepcopy(pattern_config)
        else:
            self.pattern_config_opti = pattern_config_opti
        self.quasi_steady = quasi_steady

        self.kite_model = kite_model
        self.target_drag_coefficient = None
        self.target_lift_coefficient = None
        self.s = ca.MX.sym("s")
        self.t = ca.MX.sym("t")
        self.s_dot = ca.MX.sym("s_dot")
        self.s_ddot = ca.MX.sym("s_ddot")
        self.sharpness_beta = sharpness_beta
        self.tension_min = tension_min
        self.tension_max = tension_max
        self.winch_model = Winch(
            pattern_config=self.pattern_config["radial_parameters"]
        )

        pattern = create_pattern_from_dict(self.pattern_config["path_parameters"])
        km_copy = self.substitute_parametrized_kinematics(pattern)
        self.km_param = km_copy
        # self.find_optimal_angle_pitch_tether()

    def run_simulation(self, start_state, allow_failure=True, return_states=False):

        # print("Starting state:", start_state)
        pattern = create_pattern_from_dict(
            self.pattern_config["pattern_type"], self.pattern_config["path_parameters"]
        )
        km_copy = self.substitute_parametrized_kinematics(pattern=pattern)
        self.states = []
        km_copy.reset_solver()
        self.km_param = km_copy

        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot", "speed_radial"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot", "speed_radial"]

        if km_copy.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"
        # Initialize state
        if isinstance(start_state, dict):
            state_obj = State(**start_state)
        else:
            state_obj = start_state

        N = self.pattern_config["n_points"]
        time_step = self.pattern_config["end_time"] / self.pattern_config["n_points"]
        intg = self.integrator(time_step=time_step, kite_model=km_copy)
        qs_solver = self.residual_solver(km_copy)

        # print("New state:", qs_solver)
        if self.quasi_steady:
            x0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            p = ca.vertcat(state_obj.s, state_obj.distance_radial)
            lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
            sol = qs_solver(x0=x0, p=p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            x0 = p
            z0 = sol["x"]
        else:
            x0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            p = ca.vertcat(state_obj.s, state_obj.s_dot, state_obj.distance_radial)
            lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
            sol = qs_solver(x0=x0, p=p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            x0 = p
            z0 = sol["x"]
        # self.states.append(new_state.to_dict())
        t = self.pattern_config["start_time"]
        for i in range(N):
            # print(f"Time: {t}, State: {x0}, Inputs: {z0}")
            try:
                sol = intg(
                    x0=x0,
                    p=t,
                    z0=z0,
                )
            except Exception as e:
                print(f"Error occurred: {e}")
                if not allow_failure:
                    raise
                break
            x0 = sol["xf"]
            z0 = sol["zf"]
            if self.quasi_steady:
                new_state = State(
                    t=t,
                    s=x0[0],
                    input_steering=float(z0[1]),
                    tension_tether_ground=float(z0[0]),
                    s_dot=float(z0[2]),
                    distance_radial=float(x0[1]),
                    speed_radial=float(z0[3]),
                )
            else:
                new_state = State(
                    t=t,
                    s=x0[0],
                    s_dot=float(x0[1]),
                    input_steering=float(z0[1]),
                    tension_tether_ground=float(z0[0]),
                    s_ddot=float(z0[2]),
                    distance_radial=float(x0[2]),
                    speed_radial=float(z0[3]),
                )
            t += time_step
            self.states.append(new_state.to_dict())

    def run_simulation_phase(
        self, start_state, allow_failure=True, return_states=False
    ):
        """
        March along an s-grid. At each grid point:
        - solve residuals for unknowns (z)
        - record state at current (t, s_i)
        - if not last grid point, compute dt from ds, v, a and advance x, t.

        Conventions:
        QS   : a_s = 0  -> ds = v_s * dt
        Dyn  : ds = v_s * dt + 0.5 * a_s * dt^2  (stable quadratic root used)
        """

        # --- setup / housekeeping
        self.kite_model.reset_solver()
        pattern = create_pattern_from_dict(self.pattern_config["path_parameters"])
        km_copy = self.substitute_parametrized_kinematics(pattern)
        self.km_param = km_copy
        self.states = []

        # unknowns to solve at each s-node
        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot", "speed_radial"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot", "speed_radial"]

        if km_copy.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"

        # initial state object
        state_obj = (
            State(**start_state) if isinstance(start_state, dict) else start_state
        )

        # grid and solver
        N = int(self.pattern_config["sim_parameters"]["n_points"])
        s_grid = np.linspace(
            self.pattern_config["sim_parameters"]["start_angle"],
            self.pattern_config["sim_parameters"]["end_angle"],
            N + 1,
        )
        qs_solver = self.residual_solver(km_copy)

        # pack initial guesses / states
        if self.quasi_steady:
            # z = [tension_tether_ground, input_steering, s_dot, speed_radial]
            z = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            # x = [s, distance_radial]
            x = ca.vertcat(s_grid[0], state_obj.distance_radial)
        else:
            # z = [tension_tether_ground, input_steering, s_ddot, speed_radial]
            z = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                0.01,  # initial guess for s_ddot
                state_obj.speed_radial,
            )
            # x = [s, s_dot, distance_radial]
            x = ca.vertcat(s_grid[0], state_obj.s_dot, state_obj.distance_radial)

        lbx, ubx, lbg, ubg = self.get_boundaries(state_obj, unknown_vars, km_copy)
        t = float(state_obj.t)

        # --- helper: stable Δt from ds, v, a  (ds = v*dt + 0.5*a*dt^2)
        def _dt_from_ds_v_a(ds_scalar, v_s, a_s):
            """
            Numerically stable positive root:
                dt = 2*ds / ( v + sqrt(v*v + 2*a*ds) )
            - uses CasADi ops so it works with DM/MX/SX.
            - if discriminant < 0: clip to 0 if allow_failure else raise.
            """
            disc = v_s * v_s + 2.0 * a_s * ds_scalar
            if allow_failure:
                disc = ca.fmax(
                    disc, 0.0
                )  # clip; produces the limiting solution if negative
            else:
                # optional hard check
                if isinstance(disc, (float, int)) and disc < 0:
                    raise ValueError(f"Negative discriminant: v^2+2*a*ds={disc}")
            denom = v_s + ca.sqrt(disc)
            # add tiny epsilon to avoid divide-by-zero when v≈0 and a→0
            return 2.0 * ds_scalar / (denom + 1e-12)

        # --- main loop
        for i in range(N):
            # 1) solve residuals at current s-grid node
            sol = qs_solver(x0=z, p=x, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            z = sol["x"]  # CasADi DM

            # 2) record current state (BEFORE stepping to next s)
            if self.quasi_steady:
                curr_state = State(
                    t=t,
                    s=float(x[0]),
                    input_steering=float(z[1]),
                    tension_tether_ground=float(z[0]),
                    s_dot=float(z[2]),
                    distance_radial=float(x[1]),
                    speed_radial=float(z[3]),
                )
            else:
                curr_state = State(
                    t=t,
                    s=float(x[0]),
                    s_dot=float(x[1]),
                    input_steering=float(z[1]),
                    tension_tether_ground=float(z[0]),
                    s_ddot=float(z[2]),
                    distance_radial=float(x[2]),
                    speed_radial=float(z[3]),
                )
            self.states.append(curr_state.to_dict())

            # 4) step to next s using appropriate time increment
            ds = float(s_grid[i + 1] - s_grid[i])  # scalar number

            if self.quasi_steady:
                # a_s = 0 => dt = ds / v_s
                v_s = z[2]  # s_dot from QS solve
                dt = ds / (v_s + 1e-12)  # small epsilon to avoid division by zero
                next_r = x[1] + z[3] * dt
                x = ca.vertcat(s_grid[i + 1], next_r)
            else:
                # dynamic: ds = v*dt + 0.5*a*dt^2
                v_s = x[1]  # current s_dot (state)
                a_s = z[2]  # current s_ddot (solve result)
                dt = _dt_from_ds_v_a(ds, v_s, a_s)

                next_s_dot = v_s + a_s * dt
                next_r = x[2] + z[3] * dt
                x = ca.vertcat(s_grid[i + 1], next_s_dot, next_r)

            # 5) advance time (dt is a CasADi scalar DM; cast to float)
            t += float(dt)

        print("Total time:", t)
        return self.states if return_states else None

    def run_simulation_euler(
        self, start_state, allow_failure=True, return_states=False
    ):

        # print("Starting state:", start_state)
        self.substitute_parametrized_kinematics()
        self.states = []
        self.kite_model.reset_solver()

        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot", "speed_radial"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot", "speed_radial"]

        if self.kite_model.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"
        # Initialize state
        if isinstance(start_state, dict):
            state_obj = State(**start_state)
        else:
            state_obj = start_state

        N = self.pattern_config["n_points"]
        time_step = self.pattern_config["end_time"] / self.pattern_config["n_points"]
        qs_solver = self.residual_solver()

        if self.quasi_steady:
            z0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            x0 = ca.vertcat(state_obj.s, state_obj.distance_radial)
        else:
            z0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                0,
                state_obj.speed_radial,
            )
            x0 = ca.vertcat(
                state_obj.s,
                state_obj.s_dot,
                state_obj.distance_radial,
            )
        lbx, ubx, lbg, ubg = self.get_boundaries(state_obj, unknown_vars)

        t = self.pattern_config["start_time"]
        for i in range(N):
            # print(f"Time: {t}, State: {x0}, Inputs: {z0}")

            if self.quasi_steady:
                sol = qs_solver(x0=z0, p=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
                z0 = sol["x"]
                new_s = x0[0] + z0[2] * time_step
                new_r = x0[1] + z0[3] * time_step
                x0 = ca.vertcat(new_s, new_r)
            else:
                sol = qs_solver(x0=z0, p=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
                z0 = sol["x"]
                new_s = x0[0] + x0[1] * time_step
                new_s_dot = x0[1] + z0[2] * time_step
                new_r = x0[2] + z0[3] * time_step
                x0 = ca.vertcat(new_s, new_s_dot, new_r)
            if self.quasi_steady:
                new_state = State(
                    t=t,
                    s=x0[0],
                    input_steering=float(z0[1]),
                    tension_tether_ground=float(z0[0]),
                    s_dot=float(z0[2]),
                    distance_radial=float(x0[1]),
                    speed_radial=float(z0[3]),
                )
            else:
                new_state = State(
                    t=t,
                    s=x0[0],
                    s_dot=float(x0[1]),
                    input_steering=float(z0[1]),
                    tension_tether_ground=float(z0[0]),
                    s_ddot=float(z0[2]),
                    distance_radial=float(x0[2]),
                    speed_radial=float(z0[3]),
                )
            t += time_step
            self.states.append(new_state.to_dict())

    def opti_phase(
        self,
        start_state,
        opti=None,
        start_state_opti=None,
        opti_params=None,
        relax_tol=0.0,
    ):

        if not opti:
            opti = ca.Opti()
        self.run_simulation_phase(start_state, return_states=True)
        self.kite_model.reset_solver()

        if start_state_opti:
            start_state = start_state_opti
        # initial state object
        state_obj = (
            State(**start_state) if isinstance(start_state, dict) else start_state
        )
        # Replace optimized parameters with symbolic variables
        path_params = copy.deepcopy(self.pattern_config_opti.get("path_parameters", {}))
        radial_params = copy.deepcopy(
            self.pattern_config_opti.get("radial_parameters", {})
        )
        sim_params = copy.deepcopy(self.pattern_config_opti.get("sim_parameters", {}))

        pattern = create_pattern_from_dict(
            self.pattern_config_opti["pattern_type"], path_params
        )

        N = int(sim_params["n_points"])

        tau = ca.DM(np.linspace(0, 1, N + 1))  # numeric grid (DM column vector)

        s0 = sim_params["start_angle"]  # can be float or MX
        s1 = sim_params["end_angle"]  # MX (Opti variable) in your case

        # Symbolic affine map: s_grid is MX because s1 is MX
        s_grid = s0 + (s1 - s0) * tau
        winch_model = Winch(pattern_config=radial_params)
        km_copy = self.substitute_parametrized_kinematics(pattern)
        self.km_param = km_copy

        # --- Decision variables per node (N nodes for intervals 0..N-1)
        opti_vars = {
            "s": s_grid,
            "s_dot": opti.variable(N),  # tangential speed
            "input_steering": opti.variable(N),
            "speed_radial": opti.variable(N),  # reel speed v_r
            "distance_radial": opti.variable(N),  # radius r
            "tension_tether_ground": opti.variable(N),  # tether tension T
        }
        # # expose design params too
        # for var in self.optimization_vars:
        #     opti_vars[var] = self.optimization_vars[var]

        # --- Helper to check warm start against bounds
        def check_warm_start(var_name, values, bounds):
            if not bounds or len(bounds) != 2:
                return
            lb, ub = bounds
            values = np.asarray(values).ravel()
            violations_lb = values < lb
            violations_ub = values > ub
            if np.any(violations_lb) or np.any(violations_ub):
                n_violations = np.sum(violations_lb) + np.sum(violations_ub)
                print(
                    f"Warning: Warm start for {var_name} violates bounds in {n_violations} points"
                )
                if np.any(violations_lb):
                    min_val = np.min(values[violations_lb])
                    print(f"  - Below lower bound ({lb}): min value = {min_val}")
                if np.any(violations_ub):
                    max_val = np.max(values[violations_ub])
                    print(f"  - Above upper bound ({ub}): max value = {max_val}")

        # --- Warm starts from simulation (with bound checking)
        warm_starts = {
            "s_dot": self.return_variable("s_dot"),
            "input_steering": self.return_variable("input_steering"),
            "speed_radial": self.return_variable("speed_radial"),
            "distance_radial": self.return_variable("distance_radial"),
            "tension_tether_ground": self.return_variable("tension_tether_ground"),
        }

        print("\nChecking warm start values against bounds:")
        for var_name, values in warm_starts.items():
            # Check against optimization bounds if defined
            if var_name in DEFAULT_OPTI_LIMITS:
                check_warm_start(var_name, values, DEFAULT_OPTI_LIMITS[var_name])
            # Set the initial value regardless of violations
            opti.set_initial(opti_vars[var_name], values)

        # # Fix initial radius
        opti.subject_to(opti_vars["distance_radial"][0] == state_obj.distance_radial)

        # --- Build model functions
        km_copy.establish_residual()
        flat_syms = [ca.vertcat(*opti_params.values())] if opti_params else []

        residual = ca.Function(
            "residual",
            [
                self.s,
                self.s_dot,
                km_copy.input_steering,
                km_copy.tension_tether_ground,
                km_copy.speed_radial,
                km_copy.distance_radial,
            ]
            + flat_syms,
            [km_copy.residual],
        )
        tether_tension_eq = ca.Function(
            "tether_tension_eq",
            [
                self.s,
                self.s_dot,
                km_copy.input_steering,
                km_copy.speed_radial,
                km_copy.distance_radial,
                km_copy.tension_tether_ground,
            ]
            + flat_syms,
            [km_copy.tension_tether_equation],
        )

        # --- Safety / geometry constraint
        height = pattern.z(opti_vars["distance_radial"], s_grid[:-1])  # N entries
        opti.subject_to(height >= 50)

        # --- Power scale based on the simulated trajectory (LEFT RULE, consistent)
        t_hist = self.return_variable("t")  # length N (QS) or N+1
        P_hist = self.return_variable("mechanical_power")  # same length
        dt_hist = np.diff(t_hist)  # length N-1
        E0 = float(np.sum(P_hist[:-1] * dt_hist))  # left Riemann sum
        T0 = float(np.sum(dt_hist))
        P0 = E0 / (T0 + 1e-12)
        P_scale = max(abs(P0), 1.0)

        # --- Auto scales from warm start (robust to outliers)
        def _scale(x, floor=1.0):
            x = np.asarray(x).ravel()
            if x.size == 0:
                return float(floor)
            s = np.percentile(np.abs(x), 90)  # “typical large” value
            return float(max(s, floor))

        r_hist = self.return_variable("distance_radial")
        vr_hist = self.return_variable("speed_radial")
        sd_hist = self.return_variable("s_dot")
        T_hist = self.return_variable("tension_tether_ground")
        u_hist = self.return_variable("input_steering")

        S = {
            "r": _scale(r_hist, floor=1.0),
            "vr": _scale(vr_hist, floor=1.0),
            "sd": _scale(sd_hist, floor=1.0),
            "T": _scale(T_hist, floor=1.0),
            "u": _scale(u_hist, floor=1.0),
        }
        # Residual equation scales (fallback: tie to tension scale)
        S_res = [max(S["T"], 1.0)] * 3

        # --- Helpful bounds to keep NLP well-posed
        sdot_min = 1e-2  # ensures dt>0
        opti.subject_to(opti_vars["s_dot"] >= sdot_min)
        if "speed_radial" in DEFAULT_OPTI_LIMITS:
            lb, ub = DEFAULT_OPTI_LIMITS["speed_radial"]
            opti.subject_to(opti_vars["speed_radial"] >= lb)
            opti.subject_to(opti_vars["speed_radial"] <= ub)
        if "distance_radial" in DEFAULT_OPTI_LIMITS:
            lb, ub = DEFAULT_OPTI_LIMITS["distance_radial"]
            opti.subject_to(opti_vars["distance_radial"] >= lb)
            opti.subject_to(opti_vars["distance_radial"] <= ub)

        # --- Objective assembly with SAME quadrature as simulation (left rule)
        energy = 0
        t_eff = 0

        for i in range(N):

            # Model tension at node i
            T_i = tether_tension_eq(
                s_grid[i],
                opti_vars["s_dot"][i],
                opti_vars["input_steering"][i],
                opti_vars["speed_radial"][i],
                opti_vars["distance_radial"][i],
                opti_vars["tension_tether_ground"][i],
                *flat_syms,
            )
            T_model = winch_model.tension_curve(opti_vars["speed_radial"][i])

            # Scale the tether law residual
            opti.subject_to((T_i - T_model) / S["T"] == 0)

            # Residual equations (scaled)
            res_i = residual(
                s_grid[i],
                opti_vars["s_dot"][i],
                opti_vars["input_steering"][i],
                T_i,
                opti_vars["speed_radial"][i],
                opti_vars["distance_radial"][i],
                *flat_syms,
            )
            opti.subject_to(res_i[0] / S_res[0] == 0)
            opti.subject_to(res_i[1] / S_res[1] == 0)
            opti.subject_to(res_i[2] / S_res[2] == 0)

            # Left-rule dt_i = Δs_i / s_dot[i], guarded to avoid blow-up
            if i < N - 1:
                ds_i = s_grid[i + 1] - s_grid[i]
                sd_safe = ca.fmax(opti_vars["s_dot"][i], max(sdot_min, S["sd"] * 1e-3))
                dt_i = ds_i / sd_safe

                # r_{i+1} propagation (scaled residual)
                opti.subject_to(
                    (
                        opti_vars["distance_radial"][i + 1]
                        - opti_vars["distance_radial"][i]
                        - opti_vars["speed_radial"][i] * dt_i
                    )
                    / S["r"]
                    == 0
                )

                # Accumulate energy and time: power_i = T_i * v_r_i
                energy += T_i * opti_vars["speed_radial"][i] * dt_i
                t_eff += dt_i

        power = energy / (t_eff + 1e-12)

        # --- Tiny Tikhonov regularization in scaled variables (stabilizes curvature)
        eps = 1e-6
        reg = eps * (
            ca.sumsqr(opti_vars["input_steering"] / S["u"])
            + ca.sumsqr(opti_vars["s_dot"] / S["sd"])
            + ca.sumsqr(opti_vars["speed_radial"] / S["vr"])
        )

        # --- Initials for optimization parameters
        for var, mx in opti_params.items():

            if var in self.pattern_config["path_parameters"]:
                init_val = self.pattern_config["path_parameters"][var]
                opti.set_initial(mx, init_val)
                # print(f"Applying constraints for {var}")
                lb, ub = DEFAULT_OPTI_LIMITS[var]
                opti.subject_to(mx >= lb)
                opti.subject_to(mx <= ub)
            elif var in self.pattern_config["radial_parameters"]:
                init_val = self.pattern_config["radial_parameters"][var]
                opti.set_initial(mx, init_val)
                # print(f"Setting initial for {var} to {init_val}")

                lb, ub = DEFAULT_OPTI_LIMITS[var]
                # print(f"Applying constraints for {var}: lb={lb}, ub={ub}")
                opti.subject_to(mx >= lb)
                opti.subject_to(mx <= ub)
            elif var in self.pattern_config["sim_parameters"]:
                init_val = self.pattern_config["sim_parameters"][var]
                opti.set_initial(mx, init_val)
                # print(f"Applying constraints for {var}")
                lb, ub = DEFAULT_OPTI_LIMITS[var]
                opti.subject_to(mx >= lb)
                opti.subject_to(mx <= ub)
            # else:
            #     raise ValueError(
            #         f"Optimization parameter '{var}' not found in 'path_parameters' or 'radial_parameters'."
            #     )

        # --- Default limits for vector vars (if provided)
        for var_name, mx in opti_vars.items():
            if isinstance(mx, ca.MX) and var_name in DEFAULT_OPTI_LIMITS:
                # print(f"Applying constraints for {var_name}")
                lb, ub = DEFAULT_OPTI_LIMITS[var_name]
                if relax_tol > 0:
                    # expand bounds outward even if bounds are negative
                    lb = lb - relax_tol * np.abs(lb)
                    ub = ub + relax_tol * np.abs(ub)
                if mx.shape[0] == N:
                    opti.subject_to(lb <= mx[:])
                    opti.subject_to(mx[:] <= ub)
                else:
                    opti.subject_to(lb <= mx)
                    opti.subject_to(mx <= ub)

        angle_elevation = pattern.elevation(opti_vars["distance_radial"], s_grid[:-1])
        objective_dict = {
            "energy": energy,
            "total_time": t_eff,
            "power_scale": P_scale,
            "reg": reg,
            "angle_elevation_start": angle_elevation[0],
            "angle_elevation_end": angle_elevation[-1],
        }
        return (
            opti,
            opti_vars,
            objective_dict,
        )

    def run_simulation_opti(self, opti, objective):
        # Keep your solver choice; just add reg to the objective
        opti.minimize(objective)

        # --- Solver (UNCHANGED as requested)
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

            # stiffness_report(opti, solution, name="My OCP")

            print("\nOptimized Pattern Variables:")
            for var_name, mx in self.optimization_vars.items():
                val = solution.value(mx)
                print(f"  {var_name}: {val}")

                # write back optimized parameters
                optimized_config = self.pattern_config.copy()
                if var_name in optimized_config["path_parameters"]:
                    optimized_config["path_parameters"][var_name] = solution.value(mx)
                elif var_name in optimized_config["radial_parameters"]:
                    optimized_config["radial_parameters"][var_name] = solution.value(mx)
                elif var_name in optimized_config["sim_parameters"]:
                    optimized_config["sim_parameters"][var_name] = solution.value(mx)
                self.pattern_config = optimized_config
            return solution

        except Exception as e:
            print("Debug optimization information:")
            for var_name, mx in self.optimization_vars.items():
                try:
                    print(f"  {var_name}: {opti.debug.value(mx)}")
                except Exception:
                    pass
            print("Optimization failed:", e)

    def substitute_parametrized_kinematics(self, pattern):

        kinematics = ParametrizedKinematics(pattern, self)

        km_copy = copy.deepcopy(self.kite_model)

        km_copy.angle_course = kinematics.chi
        # Optimal analytical solution for speed_radial should be part of the pattern class
        # km_copy.speed_radial = km_copy.speed_radial
        # print(km_copy.speed_radial)
        # km_copy.speed_radial = kinematics.vr
        km_copy.speed_tangential = kinematics.vtau
        km_copy.timeder_angle_course = kinematics.dot_chi
        if not self.quasi_steady:
            km_copy.timeder_speed_radial = kinematics.dot_vr
            km_copy.timeder_speed_tangential = kinematics.dot_vtau
        else:
            km_copy.timeder_speed_radial = 0
            km_copy.timeder_speed_tangential = 0

        km_copy.angle_azimuth = kinematics.phi
        km_copy.angle_elevation = kinematics.beta

        return km_copy

    def integrator(self, time_step, kite_model=None):
        if kite_model is None:
            kite_model = self.kite_model
        kite_model.establish_residual()
        if self.quasi_steady:
            x = ca.vertcat(self.s, kite_model.distance_radial)
            if kite_model.is_tether_rigid:
                z = ca.vertcat(
                    kite_model.tension_tether_ground,
                    kite_model.input_steering,
                    self.s_dot,
                )
            else:
                z = ca.vertcat(
                    kite_model.length_tether,
                    kite_model.input_steering,
                    self.s_dot,
                )
            ode = ca.vertcat(
                self.s_dot,
            )

        else:
            x = ca.vertcat(
                self.s,
                self.s_dot,
                kite_model.distance_radial,
            )
            if kite_model.is_tether_rigid:
                z = ca.vertcat(
                    kite_model.tension_tether_ground,
                    kite_model.input_steering,
                    self.s_ddot,
                )
            else:
                z = ca.vertcat(
                    kite_model.length_tether,
                    kite_model.input_steering,
                    self.s_ddot,
                )

            ode = ca.vertcat(
                self.s_dot,
                self.s_ddot,
            )

        alg = kite_model.residual
        alg = ca.vertcat(
            alg,
            self.winch_model.radial_equation(
                tension_tether_ground=kite_model.tension_tether_ground,
                speed_radial=kite_model.speed_radial,
            ),
        )
        z = ca.vertcat(z, kite_model.speed_radial)
        ode = ca.vertcat(ode, kite_model.speed_radial)

        dae = {"x": x, "z": z, "ode": ode, "alg": alg}
        # Create the integrator
        opts = {
            "abstol": 1e-6,
            "reltol": 1e-6,
            # "max_num_steps": 20000,
            # "max_step_size": 0.01,  # Or even 1e-3 if very stiff
        }

        # intg = ca.integrator("intg", "idas", dae, opts)
        intg = ca.integrator("intg", "idas", dae, 0, time_step, opts)
        return intg

    def residual_solver(self, km_copy=None):
        if km_copy is None:
            km_copy = self.kite_model

        km_copy.establish_residual()
        if self.quasi_steady:
            if km_copy.is_tether_rigid:
                z = ca.vertcat(
                    km_copy.tension_tether_ground,
                    km_copy.input_steering,
                    self.s_dot,
                )
            else:
                z = ca.vertcat(
                    km_copy.length_tether,
                    km_copy.input_steering,
                    km_copy.s_dot,
                )
            p = ca.vertcat(
                self.s,
                km_copy.distance_radial,
            )

        else:
            if km_copy.is_tether_rigid:
                z = ca.vertcat(
                    km_copy.tension_tether_ground,
                    km_copy.input_steering,
                    self.s_ddot,
                )
            else:
                z = ca.vertcat(
                    km_copy.length_tether,
                    km_copy.input_steering,
                    self.s_ddot,
                )
            p = ca.vertcat(
                self.s,
                self.s_dot,
                km_copy.distance_radial,
            )

        alg = km_copy.residual
        alg = ca.vertcat(
            alg,
            self.winch_model.radial_equation(
                tension_tether_ground=km_copy.tension_tether_ground,
                speed_radial=km_copy.speed_radial,
            ),
        )
        z = ca.vertcat(z, km_copy.speed_radial)
        nlp = {
            "x": z,
            "f": 0,
            "g": alg,
            "p": p,
        }
        solver_options = {
            "ipopt": {
                "print_level": 0,  # Suppresses IPOPT output
                "max_iter": 200,  # Maximum number of iterations
                "sb": "yes",  # Suppresses more detailed solver information
            },
            "print_time": False,  # Disables CasADi's internal timing output
        }
        return ca.nlpsol("solver", "ipopt", nlp, solver_options)

    def get_boundaries(self, state_obj, unknown_vars, km_copy):
        lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
        return lbx, ubx, lbg, ubg


import casadi as ca
import numpy as np


def register_opti_vars(obj, store=None, *, name_prefix=None):
    """
    Recursively scan `obj` (dict/list/tuple/numpy/MX) for CasADi MX symbols
    and add their leaf variables (via ca.symvar) to `store` exactly once.

    Parameters
    ----------
    obj : any
        Container (dict/list/tuple/ndarray) or MX expression/symbol.
    store : dict | None
        Mapping name -> MX to update (created if None).
    name_prefix : str | None
        If set, only add variables whose .name() starts with this prefix (e.g. "opti").

    Returns
    -------
    dict : updated store
    """
    if store is None:
        store = {}

    def _scan(x):
        # Base cases
        if isinstance(x, ca.MX):
            # collect leaf symbols from the expression/symbol
            for v in ca.symvar(x):
                nm = v.name()
                if (
                    name_prefix is None or nm.startswith(name_prefix)
                ) and nm not in store:
                    store[nm] = v
            return

        # Recurse into common containers
        if isinstance(x, dict):
            for v in x.values():
                _scan(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                _scan(v)
        elif isinstance(x, np.ndarray):
            for v in x.flat:
                _scan(v)
        # else: ignore scalars/others

    _scan(obj)
    return store
