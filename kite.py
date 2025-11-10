import casadi as ca
from utils import (
    transformation_C_from_W,
    transformation_C_from_A,
    transformation_C_from_K,
)


class Wing:

    def __init__(self, mass_wing, area_wing, aero_input):
        """
        Initialize the kite system with its parameters.
        """
        self.mass_wing = mass_wing
        self.area_wing = area_wing
        self.input_steering = ca.MX.sym("input_steering")
        self.input_depower = ca.MX.sym("input_depower")
        # Aerodynamic inputs
        self.angle_pitch_tether = aero_input["params"].get(
            "angle_pitch_depower_0", ca.MX.sym("angle_pitch_tether")
        )
        self.delta_pitch_depower = aero_input["params"].get(
            "delta_pitch_depower", ca.MX.sym("delta_pitch_depower")
        )
        # self.aerodynamic_coeffs_function(aero_input)
        self.aero_input = aero_input
        self._velocity_apparent_wind_wing = None
        self._angle_of_attack = None
        self._lift_coefficient = None
        self._drag_coefficient = None

    @property
    def aerodynamic_force_coefficients(self):

        aero_input = self.aero_input

        # Define symbolic variables
        variables = {
            "alpha": self.angle_of_attack,
            "u_s": self.input_steering,
            "u_p": self.input_depower,
        }
        # Also support derived variables
        variables["alpha_squared"] = variables["alpha"] ** 2

        # Inviscid model
        if aero_input["model"] == "inviscid":
            e = aero_input["params"]["oswald_efficiency"]
            AR = aero_input["params"]["aspect_ratio"]
            CD0 = aero_input["params"]["CD0"]
            C_L = 2 * ca.pi * variables["alpha"] / (1 + 2 / (AR * e))
            C_D = C_L**2 / (ca.pi * e * AR) + CD0
            C_L = C_L * ca.cos(self.input_steering * self.k_steering)
            C_S = C_L * ca.sin(self.input_steering * self.k_steering)
            return C_L, C_D, C_S

        # Coeff-based model
        elif aero_input["model"] == "coeffs":
            C_L = aero_input["params"].get("CL0", 0)
            C_D = aero_input["params"].get("CD0", 0)
            C_S = aero_input["params"].get("CS0", 0)

            # Loop over defined terms per coefficient
            for coeff_key, terms in aero_input.get("coefficients", {}).items():
                for term in terms:
                    var = term["var"]
                    power = term.get("power", 1)
                    coef = term["coef"]
                    if var in variables:
                        value = variables[var] ** power
                        if coeff_key == "CL":
                            C_L += coef * value
                        elif coeff_key == "CD":
                            C_D += coef * ca.fabs(value)
                        elif coeff_key == "CS":
                            C_S += coef * value
            return C_L, C_D

        else:
            raise ValueError(
                "Invalid aerodynamic model type. Choose 'inviscid' or 'coeffs'."
            )

    @property
    def lift_coefficient(self):
        if self._lift_coefficient is None:
            self._lift_coefficient = self.aerodynamic_force_coefficients[0]
        return self._lift_coefficient

    @property
    def drag_coefficient(self):
        if self._drag_coefficient is None:
            self._drag_coefficient = self.aerodynamic_force_coefficients[1]
        return self._drag_coefficient

    @property
    def angle_pitch_depower(self):
        """
        Compute the tether angle based on the powered angle and the tether angle at t=0.
        """
        return self.angle_pitch_tether + self.input_depower * self.delta_pitch_depower

    @property
    def pitch_bridle(self):
        force_bridle = (
            transformation_C_from_A(
                self.angle_pitch_aerodynamic, self.angle_yaw_aerodynamic, 0
            ).T
            @ self.force_tether_at_kite
        )
        tow_line = project_onto_plane(force_bridle, ca.vertcat(0, 1, 0))
        angle_bridle = ca.atan2(tow_line[0], -tow_line[2] + 1e-6)
        return angle_bridle

    @property
    def angle_of_attack(self):
        """
        Compute the angle of attack based on the air velocity vector and tether angle.
        """

        if self._angle_of_attack is None:

            self._angle_of_attack = (
                self.angle_pitch_aerodynamic + self.angle_pitch_depower
            )

        self._angle_of_attack = self.pitch_bridle + self.angle_pitch_depower

        return self._angle_of_attack

    @property
    def velocity_apparent_wind(self):

        return self.wind.velocity_wind(self) - self.velocity_kite

    @property
    def speed_apparent_wind(self):
        va = self.velocity_apparent_wind
        return ca.sqrt(ca.mtimes(va.T, va))

    @property
    def angle_pitch_aerodynamic(self):
        velocity_apparent_wind_K = self.velocity_apparent_wind

        return ca.atan2(
            velocity_apparent_wind_K[2],
            ca.sqrt(
                velocity_apparent_wind_K[0] ** 2
                + velocity_apparent_wind_K[1] ** 2
                + 1e-6
            ),
        )

    @property
    def angle_yaw_aerodynamic(self):
        velocity_apparent_wind_K = self.velocity_apparent_wind
        return -ca.atan(
            velocity_apparent_wind_K[1] / (velocity_apparent_wind_K[0] + 1e-6)
        )

    @property
    def force_aerodynamic(self):
        """
        Compute the aerodynamic forces based on the aerodynamic coefficients.
        """
        vec_va = self.velocity_apparent_wind
        va_sq = ca.mtimes(vec_va.T, vec_va)
        va = ca.sqrt(va_sq)

        CL, CD = self.aerodynamic_force_coefficients

        va_tau = ca.sqrt(vec_va[0] ** 2 + vec_va[1] ** 2)
        lift_direction = ca.vertcat(
            va * vec_va[1] * ca.sin(self.angle_roll_aerodynamic)
            - vec_va[2] * vec_va[0] * ca.cos(self.angle_roll_aerodynamic),
            -va * vec_va[0] * ca.sin(self.angle_roll_aerodynamic)
            - vec_va[2] * vec_va[1] * ca.cos(self.angle_roll_aerodynamic),
            va_tau**2 * ca.cos(self.angle_roll_aerodynamic),
        ) / (va * va_tau + 1e-10)
        drag_direction = vec_va / (va + 1e-10)
        # Aerodynamic forces
        D = 0.5 * self.rho * va_sq * self.area_wing * CD
        L = 0.5 * self.rho * va_sq * self.area_wing * CL

        aero_forces = D * drag_direction + L * lift_direction
        return aero_forces

    @property
    def force_gravity_wing(self):

        return (
            -self.mass_wing
            * self.g
            * ca.vertcat(
                ca.cos(self.angle_elevation) * ca.cos(self.angle_course),
                ca.cos(self.angle_elevation) * ca.sin(self.angle_course),
                ca.sin(self.angle_elevation),
            )
        )


class Kite(Wing):

    def __init__(
        self,
        mass_wing,
        area_wing,
        aero_input,
        mass_kcu=0,
        g=9.81,
        rho=1.225,
        center_aerodynamic_wing=[0, 0, 10],
        center_gravity_wing=[0, 0, 10],
        steering_control="roll",
    ):
        """
        Initialize the kite system with its parameters.
        """

        super().__init__(mass_wing, area_wing, aero_input)
        self.mass_kcu = mass_kcu  # Mass of the kite control unit
        self.steering_control = steering_control
        self.g = g  # Gravitational acceleration
        self.rho = rho  # Air density
        self.center_aerodynamic_wing = (
            center_aerodynamic_wing  # Center of aerodynamic pressure
        )
        self.center_gravity_wing = center_gravity_wing  # Center of gravity

        # Add these missing symbolic variables
        self.pitch_kcu = ca.MX.sym("pitch_kcu")
        self.roll_kcu = ca.MX.sym("roll_kcu")

        self._override_gravity = False
        self._override_centripetal = False
        self._override_coriolis = False
        # print(aero_input)
        if self.steering_control == "asymmetric":
            cs_terms = aero_input["coefficients"].get("CS", [])
            k_steering = -next(
                (term["coef"] for term in cs_terms if term["var"] == "u_s"), 0.0
            )
            self.k_steering = k_steering
        else:
            self.k_steering = 1.0

        self._acceleration_total = None  # Cache for total acceleration

    @property
    def angle_roll(self):
        return self.roll_kcu

    @property
    def angle_roll_aerodynamic(self):
        return self.input_steering * self.k_steering

    @property
    def angle_pitch(self):
        return self.pitch_kcu

    @property
    def force_gravity_kcu(self):

        T = transformation_C_from_W(
            self.angle_azimuth, self.angle_elevation, self.angle_course
        )
        return T @ ca.vertcat(0, 0, -self.mass_kcu * self.g)

    @property
    def force_gravity(self):
        if self._override_gravity == True:
            return ca.vertcat(0, 0, 0)
        return self.force_gravity_wing + self.force_gravity_kcu

    @property
    def override_gravity(self):
        return self._override_gravity

    @override_gravity.setter
    def override_gravity(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_gravity = value

    @property
    def override_centripetal(self):
        return self._override_centripetal

    @override_centripetal.setter
    def override_centripetal(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_centripetal = value

    @property
    def override_coriolis(self):
        return self._override_coriolis

    @override_coriolis.setter
    def override_coriolis(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_coriolis = value

    @property
    def acceleration_rotation_course(self):
        if self._override_centripetal == True:
            return ca.vertcat(
                self.speed_tangential * self.speed_radial / self.distance_radial, 0, 0
            )
        if self._override_coriolis == True:
            return ca.cross(
                self.velocity_rotation_course_frame, self.velocity_kite
            ) - ca.vertcat(
                2 * self.speed_tangential * self.speed_radial / self.distance_radial,
                0,
                0,
            )
        return ca.cross(self.velocity_rotation_course_frame, self.velocity_kite)

    @property
    def acceleration_local(self):
        return ca.vertcat(self.timeder_speed_tangential, 0, self.timeder_speed_radial)

    @property
    def acceleration(self):
        return self.acceleration_local + self.acceleration_rotation_course

    @property
    def force_external(self):
        # print("force_external:", self.force_aerodynamic, self.force_gravity)

        return self.force_aerodynamic + self.force_gravity + self.force_tether_at_kite

    @property
    def tension_tether_equation(self):
        # TODO: Write explicit equation for tether force
        lhs = (self.mass_wing + self.mass_kcu) * self.acceleration
        return (
            -lhs[2]
            + self.force_aerodynamic[2]
            + self.force_gravity[2]
            + self.drag_tether_at_kite[2]
            + self.force_gravity_tether_at_kite[2]
        )

    @property
    def acceleration_external(self):
        acc = self.force_external / (self.mass_wing + self.mass_kcu)
        vtau = self.speed_tangential

        acc[1] = ca.if_else(
            vtau > 1e-3,
            -acc[1] / vtau,
            -ca.sign(acc[1] + 1e-6) * 1,
        )
        return acc

    @property
    def acceleration_inertial(self):
        return ca.vertcat(
            -self.speed_tangential * self.speed_radial / self.distance_radial,
            self.speed_tangential
            * ca.sin(self.angle_course)
            * ca.tan(self.angle_elevation)
            / self.distance_radial,
            self.speed_tangential**2 / self.distance_radial,
        )

    @property
    def acceleration_total(self):
        if self._acceleration_total is None:
            self._acceleration_total = (
                self.acceleration_inertial + self.acceleration_external
            )
        return self._acceleration_total

    @property
    def force_residual(self):
        """
        Compute the residual for the kite system dynamics.
        """
        # LHS and RHS
        lhs = (self.mass_wing) * self.acceleration
        # Residual
        # print(self.force_external)
        # print(lhs)
        return -lhs + self.force_external

    @property
    def angle_yaw(self):
        return self.angle_yaw_aerodynamic

    @property
    def velocity_wind(self):
        """Wind velocity at the kite position in the kite frame."""
        # Wind velocity in wind frame
        return self.wind.velocity_wind(self)


def project_onto_plane(v, n):
    n_norm2 = ca.dot(n, n) + 1e-12  # avoid div by zero
    return v - (ca.dot(n, v) / n_norm2) * n
