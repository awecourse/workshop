import casadi as ca
from awetrim.utils.reference_frames import (
    transformation_AZR_from_W,
    transformation_C_from_W,
)


class Position:
    """Position of the point particle representing the kite in the inertial reference frame (spherical)

    Attributes:
        distance_radial (float or None): Radius of point particle w.r.t. origin [m].
        angle_azimuth (float or None): Azimuth angle of point particle w.r.t. GRF's x-axis [rad].
        angle_elevation (float or None): Angle of point particle w.r.t. GRF's x,y-plane [rad] (= pi/2 - polar angle).
    """

    def __init__(self):
        """
        Args:
            distance_radial (float, optional): Value for `straight_tether_length` attribute.
            angle_azimuth (float, optional): Value for `azimuth_angle` attribute.
            angle_elevation (float, optional): Value for `elevation_angle` attribute.

        """

        # Spherical coordinates of point particle in ground reference frame.
        self._angle_azimuth = ca.MX.sym("angle_azimuth")
        self._angle_elevation = ca.MX.sym("angle_elevation")
        self._distance_radial = ca.MX.sym("distance_radial")

    @property
    def angle_azimuth(
        self,
    ):  # We cache these as properties because they are overridden in ParameterizedKinematics
        return self._angle_azimuth

    @angle_azimuth.setter
    def angle_azimuth(self, val):
        self._angle_azimuth = val

    @property
    def angle_elevation(self):
        return self._angle_elevation

    @angle_elevation.setter
    def angle_elevation(self, val):
        self._angle_elevation = val

    @property
    def distance_radial(self):
        return self._distance_radial

    @distance_radial.setter
    def distance_radial(self, val):
        self._distance_radial = val

    @property
    def polar_angle(self):
        return ca.pi / 2 - self.angle_elevation

    @property
    def x(self):
        return self.position_W[0]

    @property
    def y(self):
        return self.position_W[1]

    @property
    def z(self):
        return self.position_W[2]

    @property
    def position(self):
        return ca.vertcat(0, 0, self.distance_radial)

    @property
    def position_W(self):
        return (
            ca.transpose(
                transformation_C_from_W(
                    self.angle_azimuth, self.angle_elevation, self.angle_course
                )
            )
            @ self.position
        )


class KiteKinematics(Position):

    def __init__(self):
        super().__init__()  # Initialize the base class
        self._timeder_speed_tangential = ca.MX.sym("timeder_speed_tangential")
        self._timeder_speed_radial = ca.MX.sym("timeder_speed_radial")
        self._define_symbolic_variables_kin()

    def _define_symbolic_variables_kin(self):
        """
        Define symbolic variables used in the model.
        """
        base_symbolic_variables = {
            "speed_tangential": "speed_tangential",
            "speed_radial": "speed_radial",
            "timeder_angle_course": "timeder_angle_course",
            "angle_course": "angle_course",
        }
        for var_name in base_symbolic_variables.keys():
            setattr(self, var_name, ca.MX.sym(var_name))

    @property
    def timeder_angle_elevation(self):
        return self.speed_tangential * ca.cos(self.angle_course) / self.distance_radial

    @property
    def timeder_angle_azimuth(self):
        return (
            self.speed_tangential
            * ca.sin(self.angle_course)
            / (self.distance_radial * ca.cos(self.angle_elevation))
        )

    @property
    def velocity_kite(self):
        return ca.vertcat(self.speed_tangential, 0, self.speed_radial)

    @property
    def velocity_kite_W(self):
        return (
            ca.transpose(
                transformation_C_from_W(
                    self.angle_azimuth, self.angle_elevation, self.angle_course
                )
            )
            @ self.velocity_kite
        )

    @property
    def timeder_speed_tangential(self):
        return self._timeder_speed_tangential

    @timeder_speed_tangential.setter
    def timeder_speed_tangential(self, value):
        self._timeder_speed_tangential = value

    @property
    def timeder_speed_radial(self):
        return self._timeder_speed_radial

    @timeder_speed_radial.setter
    def timeder_speed_radial(self, value):
        self._timeder_speed_radial = value

    @property
    def velocity_rotation_course_frame(self):
        return ca.vertcat(
            0,
            self.speed_tangential / self.distance_radial,
            self.speed_tangential
            / self.distance_radial
            * ca.tan(self.angle_elevation + 1e-6)  # Avoid division by zero
            * ca.sin(self.angle_course)
            - self.timeder_angle_course,
        )


class ParametrizedKinematics:

    def __init__(self, pattern, phase):
        self.pattern = pattern

        self.s = phase.s
        self.r = phase.kite_model.distance_radial
        self.vr = phase.kite_model.speed_radial
        self.s_dot = phase.s_dot
        self.s_ddot = phase.s_ddot

    @property
    def beta(self):
        return self.pattern.elevation(self.r, self.s)

    @property
    def phi(self):
        return self.pattern.azimuth(self.r, self.s)

    @property
    def dtheta_ds(self):
        return (
            ca.gradient(self.phi, self.s)
            + ca.gradient(self.phi, self.r) * self.vr / self.s_dot
        )

    @property
    def dbeta_ds(self):
        return (
            ca.gradient(self.beta, self.s)
            + ca.gradient(self.beta, self.r) * self.vr / self.s_dot
        )

    @property
    def dr_ds(self):
        return self.vr / self.s_dot

    # TODO: This is not correct, important for the dynamic case with vr_dot not equal to zero
    @property
    def dr_ds2(self):
        return ca.gradient(self.dr_ds, self.s)

    @property
    def dR_ds(self):
        return ca.vertcat(
            self.r * self.dtheta_ds * ca.cos(self.beta),
            self.r * self.dbeta_ds,
            self.dr_ds,
        )

    @property
    def vk(self):
        return ca.norm_2(self.dR_ds) * self.s_dot

    @property
    def vtau(self):
        return ca.sqrt(self.vk**2 - self.vr**2)

    # TODO: This is not correct, important for the dynamic case with vr_dot not equal to zero
    @property
    def dot_vr(self):
        return self.dr_ds2 * self.s_dot**2 + self.s_ddot * self.dr_ds

    @property
    def dot_vtau(self):
        return self.sqrt_A * (
            self.s_dot**2 * self.dr_ds + self.s_ddot * self.r
        ) + self.s_dot * self.r * self.dot_A / (2 * self.sqrt_A)

    @property
    def dbeta_ds2(self):
        return (
            ca.gradient(self.dbeta_ds, self.s)
            + ca.gradient(self.dbeta_ds, self.r) * self.vr / self.s_dot
        )

    @property
    def dtheta_ds2(self):
        return (
            ca.gradient(self.dtheta_ds, self.s)
            + ca.gradient(self.dtheta_ds, self.r) * self.vr / self.s_dot
        )

    @property
    def chi(self):
        return ca.atan2(
            self.dtheta_ds * ca.cos(self.beta),
            self.dbeta_ds,
        )

    @property
    def dot_chi(self):
        return (
            ca.gradient(self.chi, self.s) * self.s_dot
            + ca.gradient(self.chi, self.r) * self.vr
        )

    @property
    def sqrt_A(self):
        return self.vtau / (self.s_dot * self.r)

    @property
    def dot_A(self):
        return (
            2
            * self.s_dot
            * (
                self.dbeta_ds * self.dbeta_ds2
                + self.dtheta_ds * self.dtheta_ds2 * ca.cos(self.beta) ** 2
                - self.dtheta_ds**2
                * self.dbeta_ds
                * ca.sin(self.beta)
                * ca.cos(self.beta)
            )
        )

    def extract_function(self, attr_name):
        """
        Returns a CasADi function for a given symbolic attribute name.

        :param attr_name: Name of the attribute (string)
        :return: CasADi function
        """
        if not hasattr(self, attr_name):
            raise AttributeError(
                f"'ParametrizedKinematics' has no attribute '{attr_name}'"
            )

        expression = getattr(self, attr_name)  # Get the symbolic expression
        if not isinstance(expression, ca.MX) and not isinstance(expression, ca.SX):
            raise TypeError(f"'{attr_name}' is not a CasADi symbolic expression")

        # Extract all symbolic variables used in the expression
        variables = list(ca.symvar(expression))
        variables.sort(key=lambda x: x.name())  # Ensure consistent ordering

        return ca.Function(
            attr_name,
            variables,
            [expression],
            [var.name() for var in variables],
            [attr_name],
        )
