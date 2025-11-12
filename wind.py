import casadi as ca
from utils import transformation_C_from_W


class Wind:
    def __init__(
        self,
        wind_model="logarithmic",
        speed_wind_ref=10.0,
        height_ref=6,
        z0=0.01,
        kappa=0.41,
        tabulated_heights=None,
        tabulated_speeds=None,
    ):
        self.kappa = kappa
        self._speed_wind_ref = speed_wind_ref
        self._speed_friction = speed_wind_ref * self.kappa / ca.log(height_ref / z0)
        self._height_ref = height_ref
        self.wind_model = wind_model

        self.z0 = z0

        # Store tabulated data if applicable
        self.tabulated_heights = tabulated_heights
        self.tabulated_speeds = tabulated_speeds

        if self.wind_model == "tabulated":
            if tabulated_heights is None or tabulated_speeds is None:
                raise ValueError("Tabulated wind model requires heights and speeds.")

            # Create linear interpolant (1D)
            self.wind_interp = ca.interpolant(
                "wind_interp",
                "linear",
                [tabulated_heights],
                tabulated_speeds,
            )

    @property
    def speed_wind_ref(self):
        return self._speed_wind_ref

    @speed_wind_ref.setter
    def speed_wind_ref(self, value):
        self._speed_friction = value * self.kappa / ca.log(self.height_ref / self.z0)
        self._speed_wind_ref = value

    @property
    def height_ref(self):
        return self._height_ref

    @height_ref.setter
    def height_ref(self, value):
        self._height_ref = value

    @property
    def speed_friction(self):
        return self._speed_friction

    @speed_friction.setter
    def speed_friction(self, value):
        self._speed_friction = value
        self._speed_wind_ref = value / self.kappa * ca.log(self.height_ref / self.z0)

    # Should be renamed to speed_wind_kite
    def speed_wind(self, state):
        if self.wind_model == "uniform":
            return self.speed_wind_ref
        elif self.wind_model == "logarithmic":
            return self._speed_friction / self.kappa * ca.log(state.z / self.z0)
        elif self.wind_model == "tabulated":
            return self.wind_interp(state.z)

    def velocity_wind_W(self, state):
        return ca.vertcat(self.speed_wind(state), 0, 0)

    def velocity_wind(self, state):
        """
        Compute the wind velocity in the body frame.
        """
        T_C_from_W = transformation_C_from_W(
            state.angle_azimuth, state.angle_elevation, state.angle_course
        )
        return T_C_from_W @ self.velocity_wind_W(state)

    def speed_wind_at_height(self, height):
        if self.wind_model == "uniform":
            return self.speed_wind_ref
        elif self.wind_model == "logarithmic":
            return self._speed_friction / self.kappa * ca.log(height / self.z0)
        elif self.wind_model == "tabulated":
            return self.wind_interp(height)

    def velocity_wind_at_height_W(self, height):
        return ca.vertcat(self.speed_wind_at_height(height), 0, 0)

    def velocity_wind_at_height(self, state, height):
        """
        Compute the wind velocity in the body frame.
        """
        T_C_from_W = transformation_C_from_W(
            state.angle_azimuth, state.angle_elevation, state.angle_course
        )
        return T_C_from_W @ self.velocity_wind_at_height_W(height)
