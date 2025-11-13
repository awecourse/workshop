import casadi as ca
from abc import ABC, abstractmethod
import numpy as np
from math import comb
import matplotlib.pyplot as plt


class ParametrizedPatterns(ABC):

    def __init__(self, **kwargs):
        self.optimization_vars = (
            {}
        )  # Dictionary to store symbolic optimization variables
        for key, value in kwargs.items():
            setattr(self, key, value)
            if isinstance(value, ca.MX):  # If value is symbolic, store it separately
                self.optimization_vars[key] = value

    def x(self, t, s):
        return self.xd(t, s) * ca.cos(self.beta(t)) - self.zd(t, s) * ca.sin(
            self.beta(t)
        )

    def z(self, t, s):
        return self.xd(t, s) * ca.sin(self.beta(t)) + self.zd(t, s) * ca.cos(
            self.beta(t)
        )

    def y(self, t, s):
        return self.yd(t, s)

    def azimuth(self, t, s):
        return ca.atan2(self.y(t, s), self.x(t, s))

    def elevation(self, t, s):
        return ca.atan2(self.z(t, s), ca.sqrt(self.x(t, s) ** 2 + self.y(t, s) ** 2))

    def curvature(self, t_array, s_array):

        # --- Get scalar fields as expressions of s (t is fixed here) ---
        # If your methods are r(s,t), phi(s), beta(s), call accordingly.
        # The user code showed self.r(t) and gradient(..., s), so we mimic that.
        t = ca.MX.sym("t")
        s = ca.MX.sym("s")
        r = self.r(t)  # expression that depends on s
        phi = self.azimuth(t, s)  # expression that depends on s
        beta = self.elevation(t, s)  # expression that depends on s

        # --- Cartesian curve r_vec(s) ---
        x = r * ca.cos(beta) * ca.cos(phi)
        y = r * ca.cos(beta) * ca.sin(phi)
        z = r * ca.sin(beta)
        r_vec = ca.vertcat(x, y, z)  # 3x1

        # --- First and second derivatives wrt s ---
        print(r_vec)
        r_s = ca.jacobian(r_vec, s)  # 3x1
        r_ss = ca.jacobian(r_s, s)  # 3x1

        # --- Curvature and radius ---
        # (use a tiny epsilon to avoid division by zero in degenerate cases)
        eps = 1e-12
        cross_rs_rss = ca.cross(r_s, r_ss)  # 3x1
        num = ca.norm_2(cross_rs_rss)  # ||r_s x r_ss||
        den = ca.power(ca.norm_2(r_s), 3) + eps  # ||r_s||^3
        kappa = num / den
        rho = 1.0 / (kappa + eps)

        kappa_fun = ca.Function("kappa_fun", [t, s], [kappa])
        kappa = kappa_fun(t_array, s_array)

        return kappa

    def radius_curvature(self, t, s):
        return 1.0 / (self.curvature(t, s) + 1e-12)


class ParametrizedPatternsAngles(ParametrizedPatterns):
    def __init__(self, **kwargs):
        self.optimization_vars = {}  # Dictionary to store symbolic MX variables
        for key, value in kwargs.items():
            setattr(self, key, value)
            if isinstance(value, ca.MX):  # If value is symbolic, store it separately
                self.optimization_vars[key] = value

    def x(self, r, s):
        return r * ca.cos(self.azimuth(r, s)) * ca.cos(self.elevation(r, s))

    def y(self, r, s):
        return r * ca.sin(self.azimuth(r, s)) * ca.cos(self.elevation(r, s))

    def z(self, r, s):
        return r * ca.sin(self.elevation(r, s))

    def curvature(self, r_array, s_array):

        # --- Get scalar fields as expressions of s (t is fixed here) ---
        # If your methods are r(s,t), phi(s), beta(s), call accordingly.
        # The user code showed self.r(t) and gradient(..., s), so we mimic that.

        s = ca.MX.sym("s")
        r = ca.MX.sym("r")  # expression that depends on s
        phi = self.azimuth(r, s)  # expression that depends on s
        beta = self.elevation(r, s)  # expression that depends on s

        # --- Cartesian curve r_vec(s) ---
        x = r * ca.cos(beta) * ca.cos(phi)
        y = r * ca.cos(beta) * ca.sin(phi)
        z = r * ca.sin(beta)
        r_vec = ca.vertcat(x, y, z)  # 3x1

        # --- First and second derivatives wrt s ---
        print(r_vec)
        r_s = ca.jacobian(r_vec, s)  # 3x1
        r_ss = ca.jacobian(r_s, s)  # 3x1

        # --- Curvature and radius ---
        # (use a tiny epsilon to avoid division by zero in degenerate cases)
        eps = 1e-12
        cross_rs_rss = ca.cross(r_s, r_ss)  # 3x1
        num = ca.norm_2(cross_rs_rss)  # ||r_s x r_ss||
        den = ca.power(ca.norm_2(r_s), 3) + eps  # ||r_s||^3
        kappa = num / den
        rho = 1.0 / (kappa + eps)

        kappa_fun = ca.Function("kappa_fun", [r, s], [kappa], {"allow_free": True})
        kappa = kappa_fun(r_array, s_array)

        return kappa

    def radius_curvature(self, r, s):
        return 1.0 / (self.curvature(r, s) + 1e-12)


def create_pattern_from_dict(
    parameters,
) -> ParametrizedPatterns:

    required_params = {
        "helix": ["omega", "r0", "d0", "vr", "beta0", "kappa"],
        "lissajous": ["omega", "r0", "a0", "h0", "vr", "beta0", "kappa"],
        "lissajous_angles": [
            "omega",
            "r0",
            "az_amp0",
            "beta_amp0",
            "vr",
            "beta0",
            "kappa",
        ],
        "figure_eight": ["omega", "r0", "ry", "rz", "vr", "beta0", "ky", "kz", "kappa"],
        "figure_eight_angles": [
            "omega",
            "r0",
            "az_amp0",
            "beta_amp0",
            "vr",
            "beta0",
            "ky",
            "kz",
            "kappa",
        ],
        "cst_lissajous": [
            "r0",
            "az_amp0",
            "beta_amp0",
            "beta0",
        ],
        "spline": ["r0", "r1", "C_az", "C_el", "s_norm_az", "s_norm_el"],
        "cst_helix": [
            "r0",
            "az_amp0",
            "beta_amp0",
            "beta0",
        ],
        "reel_in_simple": ["elevation_start_ri", "elevation_start_riro"],
        "transition_simple": ["elevation_start_riro", "elevation_start_ro"],
    }
    pattern_type = parameters.get("pattern_type", None)
    if pattern_type not in required_params:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    missing_params = [
        param for param in required_params[pattern_type] if param not in parameters
    ]
    if missing_params:
        raise ValueError(
            f"Missing required parameters in 'parameters' for '{pattern_type}': {', '.join(missing_params)}"
        )

    # Instantiate the appropriate pattern class
    pattern_classes = {
        "cst_lissajous": CST_Lissajous,
        "cst_helix": CST_Helix,
        "reel_in_simple": Reelin_Simple,
        "transition_simple": Transition_Simple,
    }

    return pattern_classes[pattern_type](**parameters)


class CST_Lissajous(ParametrizedPatternsAngles):
    def __init__(
        self,
        az_amp0,
        beta_amp0,
        beta0,
        beta_coeffs=[0, 0],
        az_coeffs=[0, 0],
        kappa=0.0,
        kbeta=0.0,
        width_phi=0.5,
        width_beta=0.5,
        left_first=True,
        normalize_bumps=False,
        repeat_phi=True,
        repeat_beta=True,
        downloops=True,
        **kwargs,
    ):  # <- only flags
        super().__init__(
            az_amp0=az_amp0,
            beta_amp0=beta_amp0,
            beta0=beta0,
            kappa=kappa,
            kbeta=kbeta,
            beta_coeffs=beta_coeffs,
            az_coeffs=az_coeffs,
            width_phi=width_phi,
            width_beta=width_beta,
            left_first=left_first,
            normalize_bumps=normalize_bumps,
            **kwargs,
        )

        self.omega = 1.0 if downloops else -1.0
        # Base weight vectors
        self.az_coeffs = ca.vertcat(az_coeffs)
        self.beta_coeffs = ca.vertcat(beta_coeffs)
        P_phi = int(self.az_coeffs.numel())
        P_beta = int(self.beta_coeffs.numel())

        # Total number of bumps = len(weights) or 2× if repeating
        self.K_phi = 2 * P_phi if repeat_phi else P_phi
        self.K_beta = 2 * P_beta if repeat_beta else P_beta

        self.width_phi, self.width_beta = float(width_phi), float(width_beta)
        self.normalize_bumps = bool(normalize_bumps)
        self.sgn = -1.0 if left_first else +1.0

    def beta_center(self, r):
        return self.beta0 * (self.r0 / (self.r0 + (r - self.r0) * self.kbeta))

    def az_amp(self, r):
        return self.az_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    def beta_amp(self, r):
        return self.beta_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    @staticmethod
    def _mod1(x):
        return x - ca.floor(x)

    def _bump(self, u, a, width, normalize=False):
        delta = self._mod1(u - a)
        s = delta / width
        val = 6.0 * (s**2) * ((1.0 - s) ** 2)
        inside = ca.if_else(delta <= width, 1.0, 0.0)
        bump = inside * val
        return bump / width if normalize else bump

    def _build_shape_repeat(self, u, K, width, base_vec):
        """N(u) = 1 + Σ_{k=0..K-1} w_{k mod P} * bump(u; a=k/K, width)."""
        P = int(base_vec.numel())
        N = 1.0
        for k in range(K):
            wk = base_vec[k % P]
            a = k / K
            N = N + wk * self._bump(u, a=a, width=width, normalize=self.normalize_bumps)
        return N

    def _u(self, s):  # unit-phase for shaping
        return self._mod1(self.omega * s / (2.0 * ca.pi))

    def azimuth(self, r, s):
        a_phi = self.az_amp(r)
        phi_class = self.sgn * a_phi * ca.sin(self.omega * s)
        u = self._u(s)
        N_phi = self._build_shape_repeat(u, self.K_phi, self.width_phi, self.az_coeffs)
        return phi_class * N_phi  # c_phi = 0

    def elevation(self, r, s):
        c_beta = self.beta_center(r)
        b_beta = self.beta_amp(r)
        beta_class = c_beta + b_beta * ca.sin(2.0 * self.omega * s)
        u = self._u(s)
        N_beta = self._build_shape_repeat(
            u, self.K_beta, self.width_beta, self.beta_coeffs
        )
        return (beta_class) * N_beta


class CST_Helix(ParametrizedPatternsAngles):
    def __init__(
        self,
        r0,
        az_amp0,
        beta_amp0,
        beta0,
        beta_coeffs=[0, 0],
        az_coeffs=[0, 0],
        kappa=0.0,
        kbeta=0.0,
        width_phi=0.5,
        width_beta=0.5,
        left_first=True,
        normalize_bumps=False,
        repeat_phi=False,
        repeat_beta=False,
        **kwargs,
    ):  # <- only flags
        super().__init__(
            r0=r0,
            az_amp0=az_amp0,
            beta_amp0=beta_amp0,
            beta0=beta0,
            kappa=kappa,
            kbeta=kbeta,
            beta_coeffs=beta_coeffs,
            az_coeffs=az_coeffs,
            width_phi=width_phi,
            width_beta=width_beta,
            left_first=left_first,
            normalize_bumps=normalize_bumps,
            repeat_phi=repeat_phi,
            repeat_beta=repeat_beta,
            **kwargs,
        )

        self.omega = 1.0
        # Base weight vectors
        self.az_coeffs = ca.vertcat(az_coeffs)
        self.beta_coeffs = ca.vertcat(beta_coeffs)
        P_phi = int(self.az_coeffs.numel())
        P_beta = int(self.beta_coeffs.numel())

        # Total number of bumps = len(weights) or 2× if repeating
        self.K_phi = 2 * P_phi if repeat_phi else P_phi
        self.K_beta = 2 * P_beta if repeat_beta else P_beta

        self.width_phi, self.width_beta = float(width_phi), float(width_beta)
        self.normalize_bumps = bool(normalize_bumps)
        self.sgn = -1.0 if left_first else +1.0

    def beta_center(self, r):
        return self.beta0 * (self.r0 / (self.r0 + (r - self.r0) * self.kbeta))

    def az_amp(self, r):
        return self.az_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    def beta_amp(self, r):
        return self.beta_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    @staticmethod
    def _mod1(x):
        return x - ca.floor(x)

    def _bump(self, u, a, width, normalize=False):
        delta = self._mod1(u - a)
        s = delta / width
        val = 6.0 * (s**2) * ((1.0 - s) ** 2)
        inside = ca.if_else(delta <= width, 1.0, 0.0)
        bump = inside * val
        return bump / width if normalize else bump

    def _build_shape_repeat(self, u, K, width, base_vec):
        """N(u) = 1 + Σ_{k=0..K-1} w_{k mod P} * bump(u; a=k/K, width)."""
        P = int(base_vec.numel())
        N = 1.0
        for k in range(K):
            wk = base_vec[k % P]
            a = k / K
            N = N + wk * self._bump(u, a=a, width=width, normalize=self.normalize_bumps)
        return N

    def _u(self, s):  # unit-phase for shaping
        return self._mod1(self.omega * s / (2.0 * ca.pi))

    def azimuth(self, r, s):
        a_phi = self.az_amp(r)
        phi_class = self.sgn * a_phi * ca.sin(self.omega * s)
        u = self._u(s)
        N_phi = self._build_shape_repeat(u, self.K_phi, self.width_phi, self.az_coeffs)
        return phi_class * N_phi  # c_phi = 0

    def elevation(self, r, s):
        c_beta = self.beta_center(r)
        b_beta = self.beta_amp(r)
        beta_class = c_beta + b_beta * ca.cos(self.omega * s)
        u = self._u(s)
        N_beta = self._build_shape_repeat(
            u, self.K_beta, self.width_beta, self.beta_coeffs
        )
        return (beta_class) * N_beta


class Reelin_Simple(ParametrizedPatternsAngles):
    def __init__(
        self,
        elevation_start_ri,
        elevation_start_riro,
    ):  # <- only flags
        super().__init__(
            elevation_start_ri=elevation_start_ri,
            elevation_start_riro=elevation_start_riro,
        )

    def elevation(self, r, s):
        return self.elevation_start_ri + s * (
            self.elevation_start_riro - self.elevation_start_ri
        )

    def azimuth(self, r, s):
        return 0


class Transition_Simple(ParametrizedPatternsAngles):
    def __init__(
        self,
        elevation_start_riro,
        elevation_start_ro,
    ):  # <- only flags
        super().__init__(
            elevation_start_riro=elevation_start_riro,
            elevation_start_ro=elevation_start_ro,
        )

    def elevation(self, r, s):
        return self.elevation_start_riro + s * (
            self.elevation_start_ro - self.elevation_start_riro
        )

    def azimuth(self, r, s):
        return 0
