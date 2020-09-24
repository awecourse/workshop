#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of the model as presented in `Quasi-Steady Model of a Pumping Kite Power System`_ by R. Van der Vlugt
et al. The model is implemented in such a way that it can be used for numerical optimization.

.. _Quasi-Steady Model of a Pumping Kite Power System:
    https://arxiv.org/abs/1705.04133

"""
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
from awe_quasi_steady_model.utils import zip_el, plot_traces, default_colors

np.seterr(all='raise')


class OperationalLimitViolation(Exception):
    """Violation of an operational limit as imposed by the wind profile specifications and system properties.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int, optional): Exception error code. 0 is reserved as default error code.

    """
    def __init__(self, msg, code=0):
        self.msg = msg
        self.code = code


class SteadyStateError(Exception):
    """Errors related to finding a realistic steady state.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int, optional): Exception error code. 0 is reserved as default error code.

    """
    def __init__(self, msg, code=0):
        self.msg = msg
        self.code = code


class PhaseError(Exception):
    """Errors regarding solving the pumping cycle.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int, optional): Exception error code. 0 is reserved as default error code.

    """
    def __init__(self, msg, code=0):
        self.msg = msg
        self.code = code


class Environment:
    """Base environment state class - defining the wind profile and air density, which (for this class) is independent
    of the height.

    Attributes:
        GRAVITATIONAL_ACCELERATION (float): Conventional standard value of gravitational acceleration [m/s^2].
        wind_speed (float): Wind speed [m/s] at requested height.
        downwind_direction (float): Direction that the wind is blowing to w.r.t. x-axis of ground reference frame (GRF)
            [rad] at requested height. Positive in CCW direction. Convention for GRF is East, North, Up or Parallel,
            Perpendicular, Up.
        air_density (float): Air density [kg/m^3] at requested height.

    """
    GRAVITATIONAL_ACCELERATION = 9.81

    def __init__(self, wind_speed, air_density):
        """
        Args:
            wind_speed (float): Value for `wind_speed` attribute.
            air_density (float): Value for `air_density` attribute.

        """
        self.wind_speed = wind_speed
        self.downwind_direction = 0.
        self.air_density = air_density

    def set_reference_wind_speed(self, v):
        """When using a wind profile shape function, the absolute wind profile is controlled using the wind speed -
        commonly at a specific reference height.

        Args:
            v (float): Control wind speed [m/s].

        """
        self.wind_speed = v

    def calculate(self, height, altitude_ground=0.):
        """Calculate the values of the attributes for the given height/altitude. Placeholder method for child classes.

        Args:
            height (float): Height above ground [m].
            altitude_ground (float, optional): Altitude of ground level [m].

        """
        pass


class EnvAtmosphericPressure(Environment):
    """Environment state class introducing height dependent air density. Inherits from `Environment`.

    Attributes:
        rho_0 (float): Standard atmospheric density at sea level at the standard temperature [kg/m^3].
        h_p (float): Scale height for density [m].

    """
    def __init__(self, wind_speed):
        """
        Args:
            wind_speed (float): Value for `wind_speed` attribute.

        """
        super().__init__(wind_speed, None)
        self.rho_0 = 1.225
        self.h_p = 8.55e3

    def calculate(self, height, altitude_ground=0.):
        altitude = height + altitude_ground
        self.calculate_density(altitude)

    def calculate_density(self, altitude):
        """Calculate the air density for the given altitude.

        Args:
            altitude (float): Altitude [m].

        """
        self.air_density = self.rho_0*np.exp(-altitude/self.h_p)


class LogProfile(EnvAtmosphericPressure):
    """Environment state class introducing a logarithmic wind profile. Inherits from `EnvAtmosphericPressure`.

    Attributes:
        wind_speed_ref (float): Wind speed at reference height [m/s].
        h_ref (float): Reference height [m].
        h_0 (float): Roughness length [m].

    """
    def __init__(self):
        super().__init__(None)
        self.wind_speed_ref = 8.
        self.h_ref = 100.
        self.h_0 = 0.005

    def set_reference_height(self, h_ref):
        self.h_ref = h_ref

    def set_reference_wind_speed(self, v):
        self.wind_speed_ref = v

    def calculate(self, height, altitude_ground=0.):
        altitude = height + altitude_ground
        self.calculate_density(altitude)
        self.calculate_wind(height)

    def calculate_wind(self, height):
        """Calculate the wind speed for the given height above ground.

        Args:
            height (float): Height above ground [m].

        """
        if height < 0.:
            raise OperationalLimitViolation("Invalid height is given: {:.1f}.".format(height))
        elif height == 0.:
            self.wind_speed = 0.
        else:
            self.wind_speed = self.wind_speed_ref * np.log(height / self.h_0) / np.log(self.h_ref / self.h_0)
        return self.wind_speed

    def plot_wind_profile(self):
        """Plot the wind speed versus the height above ground."""
        heights = [50., 75., 100., 150., 200., 300., 400., 500.]
        wind_speeds = [self.calculate_wind(h) for h in heights]
        plt.plot(wind_speeds, heights)
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Height [m]')
        plt.grid(True)


class NormalisedWindTable1D(EnvAtmosphericPressure):
    """Environment state class introducing a wind profile specified by a normalised wind speed look-up table. Inherits
    from `EnvAtmosphericPressure`.

    Attributes:
        h_ref (float): Reference height [m].
        heights (list of floats): Heights [m] for which the wind speeds are specified.
        normalised_wind_speeds (list of floats): Wind speeds [m/s] corresponding to `height_table`.

    """
    def __init__(self):
        super().__init__(None)
        self.wind_speed_ref = 8.
        self.h_ref = 100.
        self.heights = [10.0, 31.0, 53.9, 79.0, 106.5, 136.6, 169.5, 205.4, 244.7, 287.5, 334.2, 385.1, 440.6, 500.9]
        self.normalised_wind_speeds = [0.88, 0.94, 0.97, 0.99, 1.0, 1.01, 1.02, 1.03, 1.03, 1.04, 1.04, 1.05, 1.05, 1.06]

    def set_reference_height(self, h_ref):
        self.h_ref = h_ref
        v_norm_ref = np.interp(h_ref, self.heights, self.normalised_wind_speeds, left=np.nan, right=np.nan)
        self.normalised_wind_speeds = np.array(self.normalised_wind_speeds)/v_norm_ref

    def set_reference_wind_speed(self, v):
        self.wind_speed_ref = v

    def calculate(self, height, altitude_ground=0.):
        altitude = height + altitude_ground
        self.calculate_wind(height)
        self.calculate_density(altitude)

    def calculate_wind(self, height):
        # Linear interpolation is used to determine the wind speed between points along the height.
        v = np.interp(height, self.heights, self.normalised_wind_speeds,
                      left=np.nan, right=np.nan) * self.wind_speed_ref
        if height <= 0. or np.isnan(v):
            raise OperationalLimitViolation("Invalid height is given: {:.1f}.".format(height))
        self.wind_speed = v
        return v

    def plot_wind_profile(self, label=None):
        """Plot the wind speed versus the height above ground."""
        wind_speeds = np.array(self.normalised_wind_speeds) * self.wind_speed_ref
        plt.plot(wind_speeds, self.heights, label=label)
        plt.xlabel('Wind speed [m s$^{-1}$]')
        plt.ylabel('Height [m]')
        plt.grid(True)


class WindTable2D(EnvAtmosphericPressure):
    """Environment state class introducing a 2 component wind profile specified by 2 wind speed look-up tables. Inherits
    from `EnvAtmosphericPressure`.

    Attributes:
        h_ref (float): Reference height [m].
        height_table (list of floats): Heights [m] for which the wind speeds are specified.
        wind_speed_x_table (list of floats): Wind speeds in GRF's x-direction [m/s] corresponding to `height_table`.
        wind_speed_y_table (list of floats): Wind speeds in GRF's y-direction [m/s] corresponding to `height_table`.

    """
    def __init__(self):
        super().__init__(None)
        self.h_ref = 100.
        self.height_table = [50., 75., 100., 150., 200., 300., 400., 500.]
        self.wind_speed_x_table = [6.51, 7.33, 7.99, 9.06, 9.91, 11.06, 11.60, 11.70]
        self.wind_speed_y_table = [0.28, 0.16, 0., -0.39, -0.85, -1.94, -3.06, -4.04]

    def calculate(self, height, altitude_ground=0.):
        altitude = height + altitude_ground
        self.calculate_wind(height)
        self.calculate_density(altitude)

    def calculate_wind(self, height):
        # Linear interpolation is used to determine the wind speed components between points along the height.
        v_x = np.interp(height, self.height_table, self.wind_speed_x_table)
        v_y = np.interp(height, self.height_table, self.wind_speed_y_table)
        if height <= 0. or height > self.height_table[-1]:
            raise OperationalLimitViolation("Invalid height is given: {:.1f}.".format(height))

        # The absolute wind speed and direction follow from the 2 wind speed components.
        self.wind_speed = np.sqrt(v_x**2 + v_y**2)
        self.downwind_direction = np.arctan2(v_y, v_x)
        return self.wind_speed

    def plot_wind_profile(self):
        """Plot the wind speed versus the height above ground."""
        plt.plot(self.wind_speed_x_table, self.height_table, label="x-component")
        plt.plot(self.wind_speed_y_table, self.height_table, label="y-component")
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Height [m]')
        plt.ylim([0., 500.])
        plt.legend()
        plt.grid(True)


class SysPropsFixedAeroCoeffs:
    """Base system properties class - setting the properties of the airborne components. For this class, the aerodynamic
    characteristics are assumed constant. Only the tether mass is varying with the tether length.

    Attributes:
        kite_projected_area (float): Projected area of the kite [m^2] - not to be confused with flat area.
        kite_mass (float): Point mass of the kite [kg] - including KCU.
        tether_density (float): Material density of the tether [kg/m^3].
        tether_diameter (float): Diameter of the tether [m].
        tether_length (float): Airborne tether length [m].
        tether_mass (float): Airborne tether mass [kg].
        aerodynamic_force_coefficient (float): Resultant aerodynamic force coefficient of kite and tether
            combination [-].
        lift_to_drag (float): Lift-to-drag ratio of kite and tether combination [-].

    """
    def __init__(self, kite_projected_area, kite_mass, tether_density, tether_diameter, aerodynamic_force_coefficient,
                 lift_to_drag):
        """
        Args:
            kite_projected_area (float): Value for `kite_projected_area` attribute.
            kite_mass (float): Value for `kite_mass` attribute.
            tether_density (float): Value for `tether_density` attribute.
            tether_diameter (float): Value for `tether_diameter` attribute.
            aerodynamic_force_coefficient (float): Value for `aerodynamic_force_coefficient` attribute.
            lift_to_drag (float): Value for `lift_to_drag` attribute.

        """
        # Kite properties.
        self.kite_projected_area = kite_projected_area
        self.kite_mass = kite_mass

        # Tether properties.
        self.tether_density = tether_density
        self.tether_diameter = tether_diameter

        # Calculated properties of (airborne) tether.
        self.tether_length = None
        self.tether_mass = None

        # Aerodynamic characteristics of kite and tether combination.
        self.aerodynamic_force_coefficient = aerodynamic_force_coefficient
        self.lift_to_drag = lift_to_drag

    def update(self, tether_length):  #, kite_powered):
        """Update the system properties for the given tether length.

        Args:
            tether_length (float): Updated airborne tether length [m].
            kite_powered (bool): Switch between powered and de-powered state of the kite.

        """
        self.tether_length = tether_length
        self.calculate_tether_mass()

    def calculate_tether_mass(self):
        """Calculate the tether mass."""
        self.tether_mass = self.tether_density * 0.25 * np.pi * self.tether_diameter ** 2 * self.tether_length


class SystemProperties(SysPropsFixedAeroCoeffs):
    """System properties class introducing binary aerodynamic characteristics: a powered and de-powered state of the
    kite. Inherits from `SysPropsFixedAeroCoeffs`.

    Attributes:
        kite_projected_area (float): Projected area of the kite [m^2] - not to be confused with flat area.
        kite_mass (float): Point mass of the kite [kg] - including KCU.
        tether_density (float): Material density of the tether [kg/m^3].
        tether_diameter (float): Diameter of the tether [m].
        kite_lift_coefficient_powered (float): Lift coefficient of kite in powered state [-].
        kite_drag_coefficient_powered (float): Drag coefficient of kite in powered state [-].
        kite_lift_coefficient_depowered (float): Lift coefficient of kite in de-powered state [-].
        kite_drag_coefficient_depowered (float): Drag coefficient of kite in de-powered state [-].
        tether_drag_coefficient (float): Drag coefficient of tether [-].
        reeling_speed_min_limit (float): Minimum reeling speed [m/s]
        reeling_speed_max_limit (float): Maximum reeling speed [m/s]
        tether_force_min_limit (float): Minimum tether force [N]
        tether_force_max_limit (float): Maximum tether force [N]
        tether_length (float): Airborne tether length [m].
        tether_mass (float): Airborne tether mass [kg].
        aerodynamic_force_coefficient (float): Resultant aerodynamic force coefficient of kite and tether
            combination [-].
        lift_to_drag (float): Lift-to-drag ratio of kite and tether combination [-].

    """
    def __init__(self, props):
        """
        Args:
            props (dict): System properties collected in a dictionary.

        """
        # Kite properties.
        self.kite_projected_area = 16.7  # [m^2]
        self.kite_mass = 20.  # [kg]

        # Tether properties.
        self.tether_density = 724.  # [kg/m^3]
        self.tether_diameter = 0.004  # [m]

        # Aerodynamic coefficients of kite and tether.
        self.kite_lift_coefficient_powered = .8  # [-]
        self.kite_drag_coefficient_powered = .2  # [-]
        self.kite_lift_coefficient_depowered = .34  # [-]
        self.kite_drag_coefficient_depowered = .15  # [-]
        self.tether_drag_coefficient = 1.1  # [-]

        # Relevant operational limits.
        self.reeling_speed_min_limit = 0.
        self.reeling_speed_max_limit = 8.
        self.tether_force_min_limit = 1200.
        self.tether_force_max_limit = 3200.

        # Procedure to set attributes using input dictionary.
        for key, val in props.items():
            if hasattr(self, key):
                self.__setattr__(key, val)
            else:
                print("Unexpected property provided: {}.".format(key))

        # Calculated properties of (airborne) tether.
        self.tether_length = None  # [m]
        self.tether_mass = None  # [kg]

        # Calculated aerodynamic characteristics of kite and tether combination.
        self.aerodynamic_force_coefficient = None  # [-]
        self.lift_to_drag = None  # [-]

    def calculate_tether_mass(self):
        self.tether_mass = self.tether_density * 0.25 * np.pi * self.tether_diameter ** 2 * self.tether_length

    def calculate_aerodynamic_properties(self, kite_powered):
        """Calculate the the aerodynamic characteristics of kite and tether combination.

        Args:
            kite_powered (bool): Use powered aerodynamic characteristics of the kite if True.

        """
        d = self.tether_diameter
        s = self.kite_projected_area
        le = self.tether_length

        if kite_powered:
            kite_lift_coefficient = self.kite_lift_coefficient_powered
            kite_drag_coefficient = self.kite_drag_coefficient_powered
        else:
            kite_lift_coefficient = self.kite_lift_coefficient_depowered
            kite_drag_coefficient = self.kite_drag_coefficient_depowered
        c_l = kite_lift_coefficient
        c_d = kite_drag_coefficient + .25*d*le/s*self.tether_drag_coefficient

        self.aerodynamic_force_coefficient = np.sqrt(c_l**2 + c_d**2)
        self.lift_to_drag = c_l/c_d

    def update(self, tether_length, kite_powered=True):
        self.tether_length = tether_length
        self.calculate_tether_mass()
        self.calculate_aerodynamic_properties(kite_powered)


class SysPropsAeroCurves(SysPropsFixedAeroCoeffs):
    def __init__(self, props):
        """
        Args:
            props (dict): System properties collected in a dictionary.

        """
        # Kite properties.
        self.kite_projected_area = 16.7  # [m^2]
        self.kite_mass = 20.  # [kg]

        # Tether properties.
        self.tether_density = 724.  # [kg/m^3]
        self.tether_diameter = 0.004  # [m]

        # Difference between powered and depowered state.
        self.pitch_powered = 5 * np.pi/180.  # [rad]
        self.pitch_depowered = -5 * np.pi/180.  # [rad]

        # Aerodynamic coefficients of kite and tether.
        self.angles_of_attack_curve = np.linspace(0, 25, 26) * np.pi/180.
        self.kite_lift_coefficients_curve_parameters = np.array([0.1, 2.5, 10*np.pi/180., 8*np.pi/180.])*1.15
        self.kite_drag_coefficients_curve_parameters = np.array([0.1108, 1.3822, -1.384])/2
        self.tether_drag_coefficient = 1.1  # [-]

        # Relevant operational limits.
        self.reeling_speed_min_limit = 0.
        self.reeling_speed_max_limit = 8.
        self.tether_force_min_limit = 1200.
        self.tether_force_max_limit = 3200.

        # Procedure to set attributes using input dictionary.
        for key, val in props.items():
            if hasattr(self, key):
                self.__setattr__(key, val)
            else:
                print("Unexpected property provided: {}.".format(key))

        # Calculated properties of (airborne) tether.
        self.tether_length = None  # [m]
        self.tether_mass = None  # [kg]

        # Calculated aerodynamic characteristics of kite and tether combination.
        self.aerodynamic_force_coefficient = None  # [-]
        self.lift_to_drag = None  # [-]
        self.pitch = None  # [rad]

    def calculate_tether_mass(self):
        self.tether_mass = self.tether_density * 0.25 * np.pi * self.tether_diameter ** 2 * self.tether_length

    def calculate_aerodynamic_properties(self, alpha):
        """Calculate the the aerodynamic characteristics of kite and tether combination.

        Args:
            alpha (float): Angle of attack [rad].

        """
        d = self.tether_diameter
        s = self.kite_projected_area
        le = self.tether_length

        def lift_curve(a):
            coeffs_part1 = self.kite_lift_coefficients_curve_parameters[:2]
            alpha_switch = self.kite_lift_coefficients_curve_parameters[2]
            d_alpha_peak = self.kite_lift_coefficients_curve_parameters[3]

            c22 = -coeffs_part1[1]/(2*d_alpha_peak)
            c20 = coeffs_part1[0]+coeffs_part1[1]*alpha_switch-c22*d_alpha_peak**2
            coeffs_part2 = [c20, 0, c22]

            if a < self.kite_lift_coefficients_curve_parameters[2]:
                return np.array([1, alpha]).dot(coeffs_part1)
            else:
                x = a - alpha_switch - d_alpha_peak
                return np.array([1, x, x**2]).dot(coeffs_part2)

        kite_lift_coefficient = lift_curve(alpha)
        kite_drag_coefficient = np.array([1, alpha, alpha**2]).dot(self.kite_drag_coefficients_curve_parameters)
        c_l = kite_lift_coefficient
        c_d_tether = .25*d*le/s*self.tether_drag_coefficient
        c_d = kite_drag_coefficient + c_d_tether

        self.aerodynamic_force_coefficient = np.sqrt(c_l**2 + c_d**2)
        self.lift_to_drag = c_l/c_d

    def update(self, tether_length, kite_powered=True):
        self.tether_length = tether_length
        self.calculate_tether_mass()
        if kite_powered:
            self.pitch = self.pitch_powered
        else:
            self.pitch = self.pitch_depowered


class KitePosition:
    """Position of the point particle representing the kite. In case this class is used to specify the end criteria of a
    phase, the attributes are allowed to be None's.

    Attributes:
        straight_tether_length (float or None): Radius of point particle w.r.t. origin [m].
        azimuth_angle (float or None): Azimuth angle of point particle w.r.t. GRF's x-axis [rad].
        elevation_angle (float or None): Angle of point particle w.r.t. GRF's x,y-plane [rad] (= pi/2 - polar angle).

    """
    def __init__(self, straight_tether_length=None, azimuth_angle=None, elevation_angle=None):
        """
        Args:
            straight_tether_length (float, optional): Value for `straight_tether_length` attribute.
            azimuth_angle (float, optional): Value for `azimuth_angle` attribute.
            elevation_angle (float, optional): Value for `elevation_angle` attribute.

        """
        # Spherical coordinates of point particle in ground reference frame.
        self.straight_tether_length = straight_tether_length
        self.azimuth_angle = azimuth_angle  # Note that the azimuth angle is not given w.r.t. the wind velocity per se -
        # as is typically done in literature. The wind direction can vary over the height.
        self.elevation_angle = elevation_angle


class KiteKinematics(KitePosition):
    """Basic kinematic properties. Inherits from `KitePosition`. Expanding the latter with the kite course angle and
    the position in cartesian coordinates.

    Attributes:
        course_angle (float): Direction of kite's velocity in tangential plane [rad].
        x (float): Longitudinal position in ground reference frame [m].
        y (float): Lateral position in ground reference frame [m].
        z (float): Up-position in ground reference frame [m].

    """
    def __init__(self, straight_tether_length, azimuth_angle, elevation_angle, course_angle):
        """
        Args:
            straight_tether_length (float): Value for `straight_tether_length` attribute.
            azimuth_angle (float): Value for `azimuth_angle` attribute.
            elevation_angle (float): Value for `elevation_angle` attribute.
            course_angle (float): Value for `course_angle` attribute.

        """
        super().__init__(straight_tether_length, azimuth_angle, elevation_angle)

        # Velocity properties of point particle.
        self.course_angle = course_angle

        # Cartesian coordinates of point particle in ground reference frame.
        self.x = None
        self.y = None
        self.z = None

        # Calculate position in cartesian coordinates.
        self.update()

    def update(self):
        """Update the cartesian coordinates, which follow from the spherical coordinates."""
        self.x = np.cos(self.elevation_angle)*np.cos(self.azimuth_angle)*self.straight_tether_length
        self.y = np.cos(self.elevation_angle)*np.sin(self.azimuth_angle)*self.straight_tether_length
        self.z = np.sin(self.elevation_angle)*self.straight_tether_length


class SteadyState:
    """Given the system properties, control settings, and wind velocity and kinematics of the kite; a kinematic ratio
    might exist for which the kite is in a steady state. A procedure is provided for finding this steady state. Is
    separated from KiteKinematics such that the steady state can be easily evaluated for different kinematics.

    Attributes:
        control_settings (tuple): Tuple containing the controlled parameter and the setpoint. The controlled parameter
            should be either: 'tether_force_ground', 'tether_force_kite', 'reeling_factor', or 'reeling_speed'.
        reeling_factor (float): Ratio of the reeling speed and the wind speed [-].
        kinematic_ratio (float): Ratio of the apparent wind velocity components [-].
        tangential_speed_factor (float): Ratio of the tangential kite speed and the wind speed [-].
        wind_speed
        apparent_wind_speed (float): Apparent wind speed experienced by the kite [m/s].
        heading
        inflow_angle
        angle_of_attack (float): Angle of attack seen by the kite [rad].
        lift_to_drag
        aerodynamic_force (float): Resultant aerodynamic force acting on the point particle [N].
        tether_force_kite (float): Tether force acting on the point particle [N].
        tether_force_ground (float): Tether force at the ground [N].
        power_ground (float): Power at the ground [W].
        kite_speed
        kite_tangential_speed
        reeling_speed (float): Reeling speed [m/s].
        elevation_rate (float): Elevation rate of the point particle [rad/s].
        azimuth_rate (float): Azimuth rate of the point particle [rad/s].
        lift_to_drag_error (float): Error between calculated and actual lift-to-drag ratio [-].
        n_iterations (int): Number of iterations used in iterative procedure.
        n_iterations_aoa
        converged (bool): Flag indicating if the convergence criteria is met.
        error_message (str): Error message in case an error has occurred.
        error_code (int): Error code in case an error has occurred.
        force_n_iterations (int): Force the iterative procedure to use a desired number of iterations.
        max_iterations (int): Maximum number of iterations before stopping the iterative procedure.
        enable_steady_state_errors (bool): Raising the exception if True.
        convergence_tolerance (float): Metric declaring convergence: normalized lift-to-drag error [-].
        tether_force_max_limit_violated (bool): Flag indicating if the maximum tether force limit is violated at
            the kite.
        tether_force_min_limit_violated (bool): Flag indicating if the minimum tether force limit is violated at
            the kite.
        (to be removed) tether_force_limit_violation (float): Tether force minimum/maximum limit violation [N] at the kite.

    """
    def __init__(self, iterative_procedure_config={}):
        """
        Args:
            iterative_procedure_config (dict): Iterative procedure settings collected in a dictionary.

        """
        # Control settings: control parameter and setpoint_value.
        self.control_settings = ('tether_force_ground', None)

        # Calculated operational parameters.
        self.reeling_factor = None
        self.kinematic_ratio = None
        self.tangential_speed_factor = None

        # Flow conditions at the kite.
        self.wind_speed = None
        self.apparent_wind_speed = None
        self.heading = None
        self.inflow_angle = None
        self.angle_of_attack = None
        self.lift_to_drag = None

        # Calculated forces and power.
        self.aerodynamic_force = None
        self.tether_force_kite = None
        self.tether_force_ground = None
        self.power_ground = None

        # Calculated velocity of the kite.
        self.kite_speed = None
        self.kite_tangential_speed = None
        self.reeling_speed = None
        self.elevation_rate = None
        self.azimuth_rate = None

        # Iterative procedure state.
        self.lift_to_drag_error = np.inf
        self.n_iterations = None
        self.n_iterations_aoa = None
        self.converged = False
        self.error_message = None
        self.error_code = -1

        # Iterative procedure settings.
        self.force_n_iterations = iterative_procedure_config.get('force_n_iterations', None)
        self.max_iterations = iterative_procedure_config.get('max_iterations', 250)
        self.enable_steady_state_errors = iterative_procedure_config.get('enable_steady_state_errors', True)
        self.convergence_tolerance = iterative_procedure_config.get('convergence_tolerance', 1e-6)

        # Monitoring parameters for tether force limit violation.
        self.tether_force_max_limit_violated = False
        self.tether_force_min_limit_violated = False
        # self.tether_force_limit_violation = 0

    def process_error(self, error_message, error_code=0, print_message=True):
        """Processing of steady state errors.

        Args:
            error_message (str): Description of error.
            error_code (int, optional): Identifier for error.
            print_message (bool, optional): Prints error message to screen if True.

        Raises:
            SteadyStateError: If steady state errors are enabled.

        """
        if print_message:
            print(error_message)
        if self.error_message is None:  # If no error has occurred earlier.
            self.error_message = error_message
            self.error_code = error_code
            if self.enable_steady_state_errors:
                raise SteadyStateError(error_message, error_code)

    def find_state(self, system_properties, environment_state, basic_kinematics, print_details=False):
        """Iterative procedure for finding the kinematic ratio yielding the steady state of the kite.

        Args:
            system_properties (`SysPropsFixedAeroCoeffs` or child): Collection of system properties.
            environment_state (`Environment` or child): Specification of environment.
            basic_kinematics (`KiteKinematics`): Basic kinematic properties required for finding the steady state.
            print_details (bool, optional): Prints procedure details to screen if True.

        """
        self.lift_to_drag_error, self.n_iterations, self.converged = np.inf, None, False
        self.error_message, self.error_code = None, None

        # System description.
        s = system_properties.kite_projected_area
        m = system_properties.kite_mass
        m_tether = system_properties.tether_mass

        # Position of point particle in wind reference frame.
        # Vectors are expressed in the kite reference frame with axes e_r, e_theta, and e_phi.
        phi = basic_kinematics.azimuth_angle - environment_state.downwind_direction  # Convert from ground reference frame to
        # wind reference frame.
        theta = np.pi / 2 - basic_kinematics.elevation_angle
        chi = basic_kinematics.course_angle
        r = basic_kinematics.straight_tether_length

        # Environmental state.
        v_wind = environment_state.wind_speed
        rho = environment_state.air_density
        g = environment_state.GRAVITATIONAL_ACCELERATION

        q = .5*rho*v_wind**2
        g_vector = np.array([-np.cos(theta)*g, np.sin(theta)*g, 0])

        a = np.cos(theta) * np.cos(phi) * np.cos(chi) - np.sin(phi) * np.sin(chi)
        b = np.sin(theta) * np.cos(phi)

        # Pre-iteration calculations: implications of operational setpoints on invariant forces.
        # If a tether force is used as setpoint - reeling factor is updated each iteration. The evaluated forces are
        # expressed as experienced by the kite.
        # If the reeling factor/speed is used as setpoint - f_aero is updated each iteration.
        if self.control_settings[0] == 'tether_force_kite':
            f_tether_kite = self.control_settings[1]  # Force controlled at kite.

            f_tether_theta = .5 * np.sin(theta) * m_tether * g
            try:
                f_tether_r = -np.sqrt(f_tether_kite**2 - f_tether_theta**2)
            except ValueError:
                self.process_error("Tether force setpoint is too small.", 1, print_details)
                f_tether_r = 0

            f_tether_vector = np.array([f_tether_r, f_tether_theta, 0])

            f_aero_vector = -f_tether_vector - m * g_vector
            f_aero = np.linalg.norm(f_aero_vector)
        elif self.control_settings[0] == 'tether_force_ground':
            f_tether_ground = self.control_settings[1]  # Force controlled at ground.
            f_tether_theta = .5 * np.sin(theta) * m_tether * g
            try:
                f_tether_r_ground = np.sqrt(f_tether_ground**2 - f_tether_theta**2)
            except FloatingPointError:
                f_tether_r_ground = 0.
            f_tether_r = -(f_tether_r_ground + np.cos(theta) * m_tether * g)
            f_tether_vector = np.array([f_tether_r, f_tether_theta, 0])

            f_aero_vector = -f_tether_vector - m * g_vector
            f_aero = np.linalg.norm(f_aero_vector)
        else:
            if self.control_settings[0] == 'reeling_factor':  # Reeling factor controlled.
                rf = self.control_settings[1]
            elif self.control_settings[0] == 'reeling_speed':  # Reeling speed controlled.
                rf = self.control_settings[1]/v_wind
            else:
                raise ValueError("Invalid control setting.")

            if np.sin(theta) * np.cos(phi) < rf:
                error_message = "Reeling factor of {} is not feasible.".format(rf)
                if np.sin(theta) < 0.:
                    error_message += " Elevation angle is larger than 90 degrees."
                    self.process_error(error_message, 7, print_details)
                else:
                    self.process_error(error_message, 2, print_details)

            f_aero_theta = -(.5*m_tether + m)*g*np.sin(theta)  # tangential aerodynamic force

        # Iterative procedure to determine the angle of attack.
        if system_properties.__class__.__name__ == "SysPropsAerodynamicCurves":
            update_aero_coefficients = True
            alpha = 15*np.pi/180.  # Initial assumption for angle of attack.
            system_properties.calculate_aerodynamic_properties(alpha)
        else:
            update_aero_coefficients = False

        self.n_iterations_aoa = 0
        # fraction_d_alpha = 1.  #
        while True:
            # Aerodynamics of kite.
            c_r = system_properties.aerodynamic_force_coefficient
            lift_to_drag = system_properties.lift_to_drag

            # Parameters used for loop condition.
            kappa = lift_to_drag  # Initial assumption for kinematic ratio (massless solution).
            self.n_iterations = 0  # Counter for number of iterations.

            # Iterative procedure to determine true kinematic ratio.
            while True:
                if 'tether_force' in self.control_settings[0]:  # Updating reeling factor.
                    rf = np.sin(theta) * np.cos(phi) - np.sqrt(f_aero / (q*s*c_r*(1+kappa**2)))
                else:  # Updating aerodynamic force.
                    f_aero = c_r*(1+kappa**2)*(np.sin(theta)*np.cos(phi)-rf)**2*q*s  # Magnitude of aerodynamic force.

                    try:
                        f_aero_r = np.sqrt(f_aero**2 - f_aero_theta**2)  # Radial aerodynamic force.
                    except (ValueError, FloatingPointError):
                        error_message = "No feasible solution found for radial aerodynamic force " \
                                        "after {} iterations - aerodynamic force is too small to keep " \
                                        "the kite in the air.".format(self.n_iterations)
                        self.process_error(error_message, 3, print_details)
                        f_aero_r = 0.

                    f_aero_vector = np.array([f_aero_r, f_aero_theta, 0.])

                # Updating tangential velocity factor.
                try:
                    lambda_ = a + np.sqrt(a**2+b**2-1+kappa**2*(b-rf)**2)
                except (ValueError, FloatingPointError):
                    error_message = "No feasible solution found for tangential velocity factor " \
                                    "after {} iterations.".format(self.n_iterations)
                    self.process_error(error_message, 4, print_details)
                    lambda_ = a

                # Updating the apparent wind speed.
                v_app = (np.sin(theta) * np.cos(phi) - rf) * np.sqrt(1 + kappa ** 2) * v_wind
                v_app_vector = np.array([
                    (np.sin(theta) * np.cos(phi) - rf) * v_wind,
                    (np.cos(theta) * np.cos(phi) - lambda_ * np.cos(chi)) * v_wind,
                    (-np.sin(phi) - lambda_ * np.sin(chi)) * v_wind,
                ])

                if v_app < 1e-6:
                    self.process_error("Unrealistic apparent wind speed.", 7, print_details)
                else:
                    # Evaluate the convergence of the calculated to the actual lift-to-drag ratio.
                    drag = np.dot(f_aero_vector, v_app_vector)/v_app
                    try:
                        lift_to_drag_calc = np.sqrt((f_aero/drag)**2-1)
                        kappa *= np.sqrt(lift_to_drag/lift_to_drag_calc)
                    except (ValueError, FloatingPointError):
                        error_message = "No feasible solution for found for calculated lift-to-drag " \
                                        "after {} iterations.".format(self.n_iterations)
                        self.process_error(error_message, 5, print_details)

                if kappa < 1e-6:
                    self.process_error("Unrealistic kappa.", 6, print_details)

                if self.error_message is not None:
                    kappa = None
                    break

                self.lift_to_drag_error = lift_to_drag-lift_to_drag_calc
                eps = abs(self.lift_to_drag_error)/lift_to_drag

                # Check loop conditions.
                self.n_iterations += 1

                if self.force_n_iterations is not None and self.n_iterations == self.force_n_iterations:
                    break
                elif eps < self.convergence_tolerance:
                    self.converged = True
                    break

                if self.max_iterations is not None and self.n_iterations == self.max_iterations:
                    error_message = "Maximum of {} iterations reached before convergence.".format(self.n_iterations)
                    self.process_error(error_message, 6, print_details)
                    break

            # Determine inflow angle with respect to tangential plane.
            try:
                inflow_angle = np.arcsin(v_app_vector[0]/v_app)  # Assuming wing is parallel to the unit sphere.
            except FloatingPointError:
                inflow_angle = 0.

            self.n_iterations_aoa += 1

            if update_aero_coefficients:
                alpha_new = inflow_angle + system_properties.pitch
                d_alpha = alpha_new - alpha
                if abs(d_alpha) < .01 * np.pi/180.:
                    self.angle_of_attack = alpha
                    self.lift_to_drag = system_properties.lift_to_drag
                    break
                elif self.n_iterations_aoa == 50:
                    self.process_error("Angle of attack did not converge.", 9, print_details)
                else:
                    # fraction_d_alpha -= .01
                    alpha = alpha + d_alpha*.95  #*fraction_d_alpha
                    system_properties.calculate_aerodynamic_properties(alpha)
            else:
                break

        if lambda_ < 0.:
            self.process_error("Solution converged to an unrealistic lambda.", 8, print_details)

        if print_details and self.error_message is None:
            print("Calculated lift-to-drag matches its expected value after {} iterations".format(self.n_iterations))

        # Forces from the free body diagram of tether, note that the tether forces as experienced by the kite switch
        # sign.
        f_tether_vector = f_aero_vector + m*g_vector
        f_tether = np.linalg.norm(f_tether_vector)

        f_tether_r_ground = -(f_tether_vector[0] - np.cos(theta)*m_tether*g)
        f_tether_ground = np.sqrt(f_tether_r_ground**2 + f_tether_vector[1]**2)

        # Calculating mechanical power of system.
        reeling_speed = v_wind*rf
        # # Imposing p = 0 results in failure of the unit test
        # if self.error_code is None:
        p = f_tether_ground*reeling_speed
        # else:
        #     p = 0.

        self.reeling_factor = rf
        self.kinematic_ratio = kappa
        self.tangential_speed_factor = lambda_
        self.kite_tangential_speed = lambda_ * v_wind
        self.wind_speed = v_wind
        self.apparent_wind_speed = v_app
        self.heading = np.arctan2(v_app_vector[2], v_app_vector[1])
        self.inflow_angle = inflow_angle
        self.aerodynamic_force = f_aero
        self.tether_force_kite = f_tether
        self.tether_force_ground = f_tether_ground
        self.power_ground = p

        # Kite velocity in spherical coordinates, see eq. 2.58-2.60 AWE book.
        self.kite_speed = np.sqrt(reeling_speed**2 + (lambda_*v_wind)**2)
        self.reeling_speed = reeling_speed
        try:
            self.elevation_rate = - v_wind * lambda_ / r * np.cos(chi)
        except (ZeroDivisionError, FloatingPointError):
            self.elevation_rate = 0.
        try:
            self.azimuth_rate = v_wind * lambda_ / r * np.sin(chi) / np.sin(theta)
        except (ZeroDivisionError, FloatingPointError):
            self.azimuth_rate = 0.

        # Update monitoring parameters for tether force violations.
        if 'tether_force' not in self.control_settings[0]:
            min_force = getattr(system_properties, 'tether_force_min_limit', None)
            max_force = getattr(system_properties, 'tether_force_max_limit', None)
            if max_force is not None and self.tether_force_ground > max_force:
                self.tether_force_max_limit_violated = True
                self.tether_force_min_limit_violated = False
                # self.tether_force_limit_violation = self.tether_force_ground - max_force
            elif min_force is not None and self.tether_force_ground < min_force:
                self.tether_force_max_limit_violated = False
                self.tether_force_min_limit_violated = True
                # self.tether_force_limit_violation = min_force - self.tether_force_ground


class TimeSeries:
    """A solution to the quasi-steady motion simulation of the kite. The distance covered by the point particle is
    solved as a transition through steady states using the finite difference method.

    Attributes:
        time (list): Points in time for which the states are solved.
        kinematics (list): Time series of `KiteKinematics` objects.
        steady_states (list): Time series of `SteadyState` objects.
        system_properties (`SystemProperties`): Collection of system properties.
        environment_state (`Environment` or child): Specification of environment.
        steady_state_config (dict): Iterative procedure settings for finding the steady state.
        energy (float): Energy produced in the evaluated time interval [J].
        average_power (float): Time average of the produced power [W].
        duration (float): Length of evaluated time interval [s].
        
    """
    def __init__(self):
        # Result lists with time and states.
        self.time = []
        self.kinematics = []
        self.steady_states = []
        self.n_time_points = None

        # Side conditions.
        self.system_properties = None
        self.environment_state = None
        self.steady_state_config = None

        # Performance properties of the time series.
        self.energy = None
        self.average_power = None
        self.duration = None

    def time_plot(self, plot_parameters, y_labels=None, y_scaling=None, plot_markers=None, fig_num=None):
        """Generic plotting method for making a time plot of `KiteKinematics` and `SteadyState` attributes.

        Args:
            plot_parameters (tuple): Sequence of `KiteKinematics` or `SteadyState` attributes.
            y_labels (tuple, optional): Y-axis labels corresponding to `plot_parameters`.
            y_scaling (tuple, optional): Scaling factors corresponding to `plot_parameters`.

        """
        data_sources = (self.kinematics, self.steady_states)
        source_labels = ('kin', 'ss')

        plot_traces(self.time, data_sources, source_labels, plot_parameters, y_labels, y_scaling, plot_markers=plot_markers, fig_num=fig_num)

    def trajectory_plot(self, fig_num=None, plot_kwargs={'linestyle': ':'}, steady_state_markers=True):
        """Plot of the downwind versus vertical position of the kite.

        Args:
            fig_num (int, optional): Number of figure used for the plot, if None a new figure is created.
            plot_kwargs (dict, optional): Line plot keyword arguments.
            steady_state_markers (bool, optional): Use the steady state results to mark non-converged adn erroneous
                points if True.

        """
        if fig_num != -1:
            plt.figure(fig_num)
        ax = plt.gca()
        # Plot x vs. z of trajectory.
        x_traj = [kp.x for kp in self.kinematics]
        z_traj = [kp.z for kp in self.kinematics]
        plt.plot(x_traj, z_traj, **plot_kwargs)

        markers_plotted = False
        if steady_state_markers:
            # Plot all points for which the steady state did not converge.
            x_not_converged = [x for x, ss in zip_el(x_traj, self.steady_states) if not ss.converged]
            if x_not_converged:
                z_not_converged = [z for z, ss in zip_el(z_traj, self.steady_states) if not ss.converged]
                plt.plot(x_not_converged, z_not_converged, 'kx', label='not converged')
                markers_plotted = True

            # Plot all points for which the steady state error occurred.
            x_ss_error = [x for x, ss in zip_el(x_traj, self.steady_states) if ss.error_message is not None]
            if x_ss_error:
                z_ss_error = [z for z, ss in zip_el(z_traj, self.steady_states) if ss.error_message is not None]
                plt.plot(x_ss_error, z_ss_error, 'rs', label='ss error', markerfacecolor='None')
                ss_error_code = [ss.error_code for ss in self.steady_states if ss.error_code is not None]
                for x, z, ec in zip(x_ss_error, z_ss_error, ss_error_code):
                    plt.plot(x+5, z, marker='${}$'.format(ec), mec='k')  #, alpha=1, ms=7)
                markers_plotted = True

            # Plot all points for which the force limits were violated.
            x_max_limit = [x for x, ss in zip_el(x_traj, self.steady_states) if ss.tether_force_max_limit_violated]
            z_max_limit = [z for z, ss in zip_el(z_traj, self.steady_states) if ss.tether_force_max_limit_violated]
            plt.plot(x_max_limit, z_max_limit, 'ro', label='max force violated', markerfacecolor='None', markersize=10)
            x_min_limit = [x for x, ss in zip_el(x_traj, self.steady_states) if ss.tether_force_min_limit_violated]
            z_min_limit = [z for z, ss in zip_el(z_traj, self.steady_states) if ss.tether_force_min_limit_violated]
            plt.plot(x_min_limit, z_min_limit, 'go', label='min force violated', markerfacecolor='None')
            if x_max_limit or x_min_limit:
                markers_plotted = True

        plt.xlabel('x [m]')
        plt.ylabel('z [m]')

        # if ax.get_xlim()[0] > 0.:
        #     plt.xlim([0., None])
        # plt.ylim([0., None])
        plt.grid(True)
        plt.gca().set_aspect('equal')
        if markers_plotted:
            plt.legend()

    def trajectory_plot3d(self, fig_num=None, animation=True, plot_kwargs={'linestyle': '-'}, gradient_color=None,
                          plot_point_type=None):
        """Animation of the 3D trajectory of the kite.

        Args:
            fig_num (int, optional): Number of figure used for the plot, if None a new figure is created.
            animation (bool, optional): Make animation of the plot by changing the view angle.
            plot_kwargs (dict, optional): Line plot keyword arguments.
            plot_point_type (int, optional): If not None, only plot points for which the phase identifier corresponds to
                the given integer.

        """
        from mpl_toolkits.mplot3d import axes3d
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.colors import ListedColormap
        from itertools import cycle
        fig = plt.figure(fig_num)
        ax = fig.gca(projection='3d')

        if plot_point_type is not None:
            point_types = getattr(self, 'phase_id', np.zeros(self.n_time_points))
            x_traj, y_traj, z_traj = zip(*[(kp.x, kp.y, kp.z) for kp, pt in
                                           zip(self.kinematics, point_types) if pt == plot_point_type])
        else:
            x_traj, y_traj, z_traj = zip(*[(kp.x, kp.y, kp.z) for kp in self.kinematics])

        if gradient_color is not None:
            vals = gradient_color[1]
            vals_range = (vals.min(), vals.max())
            points = np.array([x_traj, y_traj, z_traj]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            if gradient_color[0] in ['Phase id [-]', 'Pattern id [-]']:
                cmap = plt.get_cmap('Dark2', vals_range[1]+1-vals_range[0])
                norm = plt.Normalize(vals_range[0]-.5, vals_range[1]+.5)
                ticks = np.arange(vals_range[0], vals_range[1]+1.)
            # elif gradient_color[0] == 'Power [kW]':
            else:
                cmap_basis = plt.get_cmap('coolwarm')

                if vals_range[0] < 0. < vals_range[1]:
                    positive_fraction = vals_range[1]/(vals_range[1]-vals_range[0])
                    clrs_negative_values = cmap_basis(np.linspace(0., .5, int(256*(1-positive_fraction))))
                    clrs_positive_values = cmap_basis(np.linspace(.5, 1., int(256*positive_fraction)))
                    cmap = ListedColormap(np.vstack((clrs_negative_values, clrs_positive_values)))
                elif vals_range[0] >= 0.:
                    clrs = cmap_basis(np.linspace(.5+vals_range[0]/vals_range[1], 1., 256))
                    cmap = ListedColormap(clrs)
                else:
                    clrs = cmap_basis(np.linspace(0., .5-vals_range[1]/vals_range[0], 256))
                    cmap = ListedColormap(clrs)
                norm = None
                ticks = None  #[-15., 0., 45.]
            lc = Line3DCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(vals)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            cbar = fig.colorbar(lc, ax=ax, shrink=.4, aspect=10., pad=0., ticks=ticks)
            cbar.set_label(gradient_color[0])
        else:
            plt.plot(x_traj, y_traj, z_traj, **plot_kwargs)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.grid(True)

        for coords, fun in zip((x_traj, y_traj, z_traj), (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            coords_range = (1.1*min(coords), 1.1*max(coords))
            if coords_range[0] > 0.:
                fun([0., coords_range[1]])
            elif coords_range[1] < 0.:
                fun([coords_range[0], 0.])
            else:
                fun(coords_range)

        # ax.set_xlim([0, 500])
        # ax.set_ylim([-250, 250])
        # ax.set_zlim([0, 500])
        # ax.set_aspect('equal')  # Looks a bit silly.

        if animation:
            # Rotate the axes and update plot.
            for angle in cycle(range(0, 360)):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(.05)
                if ax != plt.gca():  # Stop loop after closing figure.
                    break
        else:
            ax.view_init(41, 50)


class Phase(TimeSeries):
    """The solution to the quasi-steady motion simulation conform a particular control strategy. Inherits from
    `TimeSeries`.

    Attributes:
        time_step (float): Step between consecutive time points for which the kite's motion is solved, last time step
            may deviate [s].
        control_settings (tuple): Tuple containing the controlled parameter and the setpoint. The controlled parameter
            should be either: 'tether_force_ground', 'tether_force_kite', 'reeling_factor', or 'reeling_speed'.
        impose_operational_limits (bool): Specifies whether to automatically switch the control settings when they lead
            to violating the operational limits.
        kite_powered (bool): Use powered aerodynamic characteristics of the kite if True.
        follow_wind (bool): Specifies whether kite is 'aligned' with the wind. Controlled azimuth angle is expressed
            w.r.t. wind reference frame if True, or ground reference frame if False.
        kinematics_start (`KiteKinematics`): Kinematics of kite at start of phase.
        position_end (`KitePosition`): Kite position criteria for ending the phase.
        timer (float): Time spent in phase [s] - updated after calculating a new time point.
        n_time_points (int): Counter for the number of time points.
        min_reeling_speed (float): Minimum speed that has occurred in the phase [m/s].
        max_reeling_speed (float): Maximum speed that has occurred in the phase [m/s].
        min_tether_force (float): Minimum tether force at ground that has occurred in the phase [N].
        max_tether_force (float): Maximum tether force at ground that has occurred in the phase [N].
        enable_limit_violation_error (bool): Flag specifying whether to raise an error when the reeling
            speed or tether force limit is violated.
        max_time_points (int): Number of time points for which the simulation is terminated.
        average_reeling_factor (float): Time average of the reeling factor [-].
        average_reeling_speed (float): Time average of the reeling speed [m/s].
        average_tether_force_ground (float): Time average of the tether force at the ground station [N].
        path_length (float): Distance covered in the evaluated time interval [m].
        path_length_effective (float): Distance between the first and last position [m].
        reeling_tether_length (float): Tether length that is reeled out [m].

    """
    def __init__(self, phase_settings, impose_operational_limits=True):  # control_settings default value use to be ('tether_force_ground', 1200)
        """
        Args:
            phase_settings (tuple): Value for `control_settings` attribute.
            impose_operational_limits (bool, optional): Value for `impose_operational_limits` attribute.

        """
        super().__init__()
        # Simulation setting.
        self.time_step = phase_settings.get('time_step', 1.)

        # Control settings.
        self.control_settings = phase_settings['control']
        self.impose_operational_limits = impose_operational_limits
        self.kite_powered = True
        self.follow_wind = False

        # Start and stop conditions of phase.
        self.kinematics_start = None
        self.position_end = None

        # Monitoring parameters.
        self.timer = None
        self.n_time_points = 0
        self.min_reeling_speed = np.inf
        self.max_reeling_speed = -np.inf
        self.min_tether_force = np.inf  # Forces at ground station.
        self.max_tether_force = 0

        # Monitoring settings.
        self.enable_limit_violation_error = True
        self.max_time_points = 5000

        # Operational properties of the phase.
        self.average_reeling_factor = None
        self.average_reeling_speed = None
        self.average_tether_force_ground = None
        self.path_length = None
        self.path_length_effective = None
        self.reeling_tether_length = None

    def run_simulation(self, system_properties, environment_state, steady_state_config={}, timer_start=0.):
        """Solve quasi-steady motion using finite-difference method.

        Args:
            system_properties (`SystemProperties`): Collection of system properties.
            environment_state (`Environment` or child): Specification of environment.
            steady_state_config (dict, optional): Iterative procedure settings for finding the steady state.
            timer_start (float, optional): Start point for time trace [s].

        """
        # Empty the result lists.
        self.time, self.kinematics, self.steady_states, self.n_time_points = [], [], [], 0
        self.min_reeling_speed, self.max_reeling_speed = np.inf, -np.inf

        self.system_properties = system_properties
        self.environment_state = environment_state
        self.steady_state_config = steady_state_config

        # Add first time point, kite kinematics, and steady state to corresponding result lists.
        self.timer = timer_start
        environment_state.calculate(self.kinematics_start.z)
        if self.follow_wind:
            self.kinematics_start.azimuth_angle += environment_state.downwind_direction
            self.kinematics_start.update()
        steady_state_start = self.determine_new_steady_state(self.kinematics_start)
        self.time.append(self.timer)
        self.kinematics.append(self.kinematics_start)
        self.steady_states.append(steady_state_start)

        # Monitor stopping criteria in case of infinite loop.
        end_phase = False
        while not end_phase:
            end_phase, new_kinematics = self.determine_new_kinematics(self.kinematics[-1], self.steady_states[-1])
            environment_state.calculate(new_kinematics.z)
            if self.follow_wind:
                new_kinematics.azimuth_angle += environment_state.downwind_direction
                new_kinematics.update()

            # Add new time, kinematics, and steady state to corresponding result lists.
            new_steady_state = self.determine_new_steady_state(new_kinematics)
            self.time.append(self.timer)
            self.kinematics.append(new_kinematics)
            self.steady_states.append(new_steady_state)

            self.n_time_points += 1

            if self.max_time_points is not None and self.n_time_points == self.max_time_points:
                error_message = "Maximum of {} iterations reached in phase for {} setpoint: {} and " \
                                "end criteria: {}.".format(self.max_time_points, self.control_settings[0],
                                                           self.control_settings[1], self.position_end.__dict__)
                raise PhaseError(error_message, 1)

        # Processing resulting steady states to determine the phase performance.
        self.energy = np.trapz([ss.power_ground for ss in self.steady_states], self.time)
        self.duration = self.timer - timer_start
        if self.duration > 0:
            self.average_power = self.energy / self.duration
            self.calc_operational_properties()
        else:
            self.average_power = 0

        # print("Energy for comparison: ", simple_integration([s.power_ground for s in self.steady_states][:-1], self.time[:-1]))
        # print("{:.1f} seconds passed to reach, {:.0f}J energy produced.".format(self.timer-timer_start, self.energy))

    def calc_operational_properties(self):
        """Calculate the operational properties of the phase."""
        # Calculate time averages.
        self.average_reeling_factor = np.trapz([ss.reeling_factor for ss in self.steady_states],
                                               self.time) / self.duration
        self.average_reeling_speed = np.trapz([ss.reeling_speed for ss in self.steady_states],
                                              self.time) / self.duration
        self.average_tether_force_ground = np.trapz([ss.tether_force_ground for ss in self.steady_states],
                                                    self.time) / self.duration

        # Calculate the length properties of the path covered by the kite.
        self.path_length = 0
        for kp0, kp1 in zip(self.kinematics[:-1], self.kinematics[1:]):
            self.path_length += np.sqrt((kp1.x - kp0.x)**2 + (kp1.y - kp0.y)**2 + (kp1.z - kp0.z)**2)

        kp0, kp1 = self.kinematics[0], self.kinematics[-1]
        self.path_length_effective = np.sqrt((kp1.x - kp0.x)**2 + (kp1.y - kp0.y)**2 + (kp1.z - kp0.z)**2)
        self.reeling_tether_length = kp1.straight_tether_length - kp0.straight_tether_length

    def determine_new_kinematics(self, last_kinematics, last_steady_state):
        """Determine new kinematics based on kinematics and steady state of previous time point. Moreover, evaluate if
        phase end criteria are met. Placeholder method for child classes.

        Args:
            last_kinematics (`KiteKinematics`): Kinematics object of previous time point.
            last_steady_state (`SteadyState`): Steady state of previous time point.

        Returns:
            bool: Flag indicating meeting phase ending criteria.
            `KiteKinematics`: Kinematic state of the kite for the new time point.

        """
        pass

    def determine_new_steady_state(self, kinematics):
        """Determine new steady state based on new kinematics and updated system properties and environment state.

        Args:
            kinematics (`KiteKinematics`): Kinematics object of current time point.

        Returns:
            `SteadyState`: Steady state of current time point.

        """
        # Update system properties, environment state is already up-to-date.
        sys_props = self.system_properties
        env_state = self.environment_state
        sys_props.update(kinematics.straight_tether_length, self.kite_powered)

        # Instantiate new steady state using the primary control setting.
        new_state = SteadyState(self.steady_state_config)
        new_state.control_settings = self.control_settings[:2]

        # Find steady state.
        temporary_suppress_steady_state_errors = False
        if self.impose_operational_limits:
            if new_state.enable_steady_state_errors:
                temporary_suppress_steady_state_errors = True
                new_state.enable_steady_state_errors = False
        new_state.find_state(sys_props, env_state, kinematics)
        if temporary_suppress_steady_state_errors:
            new_state.enable_steady_state_errors = True

        # Operational limits.
        if 'tether_force' not in self.control_settings[0] and len(self.control_settings) == 4:
            min_force = self.control_settings[2]
            max_force = self.control_settings[3]
        else:
            min_force = sys_props.tether_force_min_limit
            max_force = sys_props.tether_force_max_limit
        max_speed = sys_props.reeling_speed_max_limit
        min_speed = sys_props.reeling_speed_min_limit
        assert max_speed > 0 and min_speed >= 0, "Reeling speed limits should be positive."

        # When operational limits are imposed, evaluate if the primary control setting yield limit violations.
        if self.impose_operational_limits:
            if 'tether_force' not in self.control_settings[0]:  # If speed controlled.
                # Check if the tether force limits are violated. If so, use the force limit as controlled parameter.
                if max_force is not None and new_state.tether_force_ground > max_force:
                    new_state.control_settings = ('tether_force_ground', max_force)
                    new_state.find_state(sys_props, env_state, kinematics)
                elif (min_force is not None and new_state.tether_force_ground < min_force) or \
                        (self.__class__.__name__ == "RetractionPhase" and new_state.error_code == 7):
                    new_state.control_settings = ('tether_force_ground', min_force)
                    new_state.find_state(sys_props, env_state, kinematics)
                elif new_state.error_message is not None and temporary_suppress_steady_state_errors:
                    raise SteadyStateError(new_state.error_message, new_state.error_code)
            #TODO: find way to improve lower check
            elif self.__class__.__name__ != "TransitionPhase":  # ["TractionPhase", "TractionPhasePattern", "TractionPhaseHybrid", "EvaluatePattern", "RetractionPhase"]:  # If force controlled.
                setpoint_speed = None

                if max_speed is not None and abs(new_state.reeling_speed) > max_speed:
                    if self.__class__.__name__ == "RetractionPhase":
                        setpoint_speed = -max_speed
                    else:
                        setpoint_speed = max_speed
                elif min_speed is not None and abs(new_state.reeling_speed) < min_speed:
                    if self.__class__.__name__ == "RetractionPhase":
                        setpoint_speed = -min_speed
                    else:
                        setpoint_speed = min_speed

                if setpoint_speed is not None:
                    new_state = SteadyState(self.steady_state_config)
                    new_state.control_settings = ('reeling_speed', setpoint_speed)
                    new_state.find_state(sys_props, env_state, kinematics)
                elif new_state.error_message is not None and temporary_suppress_steady_state_errors:
                    raise SteadyStateError(new_state.error_message, new_state.error_code)

        # Update the monitoring parameters.
        if new_state.converged:
            if new_state.reeling_speed > self.max_reeling_speed:
                self.max_reeling_speed = new_state.reeling_speed
            if new_state.reeling_speed < self.min_reeling_speed:
                self.min_reeling_speed = new_state.reeling_speed
            if new_state.tether_force_ground > self.max_tether_force:
                self.max_tether_force = new_state.tether_force_ground
            if new_state.tether_force_ground < self.min_tether_force:
                self.min_tether_force = new_state.tether_force_ground

        # Check for limit violations.
        if self.enable_limit_violation_error:
            error_message = None
            if max_speed is not None and abs(new_state.reeling_speed) > max_speed + 1e-3:
                error_message = "The reeling speed: {:.1f} m/s is exceeding the {:.1f} m/s " \
                                "maximum limit.".format(new_state.reeling_speed, max_speed)
                error_code = 1
            elif min_speed is not None and abs(new_state.reeling_speed) < min_speed - 1e-3:
                error_message = "The reeling speed: {:.1f} m/s is smaller than the {:.1f} m/s " \
                                "minimum limit.".format(new_state.reeling_speed, min_speed)
                error_code = 2
            elif max_force is not None and new_state.tether_force_ground > max_force + 1e-3:
                error_message = "The tether force: {:.1f} N is exceeding the {:.1f} N " \
                                "maximum limit.".format(new_state.tether_force_ground, max_force)
                error_code = 3
            elif min_force is not None and new_state.tether_force_ground < min_force - 1e-3:
                error_message = "The tether force: {:.1f} N is smaller than the {:.1f} N " \
                                "minimum limit.".format(new_state.tether_force_ground, min_force)
                error_code = 4
            if error_message:
                raise OperationalLimitViolation(error_message, error_code)

        return new_state


class RetractionPhase(Phase):
    """The solution to the quasi-steady motion simulation conform the idealized trajectory assumption for the retraction
    phase. Inherits from `Phase`.

    Attributes:
        AZIMUTH_ANGLE (float): Constant azimuth angle [rad] of representative flight state.
        COURSE_ANGLE (float): Constant course angle [rad] of representative flight state.
        tether_length_start (float): Tether length [m] used for initial state.
        tether_length_end (float): Tether length [m] used for phase ending criteria.
        elevation_angle_start (float): Elevation angle [rad] used for initial state.
        fix_tether_length (bool): Tether length stays constant despite the reel-in speed, if True. Should only be set to
            True when abusing this class for the unrealistic simulation to determine the steady reel-in elevation angle.

    """
    # General kinematics assumptions of representative flight state for the retraction phase.
    AZIMUTH_ANGLE = 0.
    COURSE_ANGLE = np.pi

    def __init__(self, phase_settings={'control': ('tether_force_ground', 1200.)}, impose_operational_limits=True):
        """
        Args:
            phase_settings (tuple, optional): Setting parent's `control_settings` attribute.
            impose_operational_limits (bool, optional): Setting parent's `impose_operational_limits` attribute.

        """
        super().__init__(phase_settings, impose_operational_limits)

        # Binary kite aerodynamic state.
        self.kite_powered = False

        # Properties of initial state and final position.
        self.tether_length_start = 385.
        self.tether_length_end = 240.
        self.elevation_angle_start = 30.*np.pi/180.

        self.fix_tether_length = False  # Should be False for realistic simulation.

    def finalize_start_and_end_kite_obj(self):
        """Finalize the initial state and ending criteria before running the simulation, respectively `kinematics_start`
        and `position_end`."""
        self.kinematics_start = KiteKinematics(self.tether_length_start, self.AZIMUTH_ANGLE,
                                               self.elevation_angle_start, self.COURSE_ANGLE)
        self.position_end = KitePosition(straight_tether_length=self.tether_length_end)

    def determine_new_kinematics(self, last_kinematics, last_steady_state):
        """Determine kinematic state of the kite for the new time point based on the previous kinematic and steady state
        properties. For the retraction phase, the tether length and elevation angle are updated.

        Args:
            last_kinematics (`KiteKinematics`): Kinematics object of previous time point.
            last_steady_state (`SteadyState`): Steady state of previous time point.

        Returns:
            bool: Flag indicating meeting phase ending criteria.
            `KiteKinematics`: Kinematic state of the kite for the new time point.

        """
        kin = copy(last_kinematics)

        # Determine the difference in tether length and elevation angle for regular time step.
        if not self.fix_tether_length:
            d_tether_length = last_steady_state.reeling_speed*self.time_step
        else:
            d_tether_length = 0.
        d_elevation = last_steady_state.elevation_rate*self.time_step
        if d_elevation < 1e-4 and d_tether_length > 0.:
            raise PhaseError("Reeling out at constant elevation angle during reel-in phase.", 3)

        # Check if target tether length is not exceeded next iteration.
        if kin.straight_tether_length + d_tether_length > self.position_end.straight_tether_length:
            # Set timer and kite kinematics for next iteration.
            self.timer += self.time_step
            kin.straight_tether_length += d_tether_length
            kin.elevation_angle += d_elevation
            kin.update()
            end_phase = False
        else:
            # Determine the time needed for reaching the target tether length.
            d_tether_length_remaining = self.position_end.straight_tether_length - kin.straight_tether_length
            reduced_time_step = d_tether_length_remaining/last_steady_state.reeling_speed

            # Set final timer and kite kinematics.
            self.timer += reduced_time_step
            kin.straight_tether_length += d_tether_length_remaining
            kin.elevation_angle += last_steady_state.elevation_rate*reduced_time_step
            kin.update()
            end_phase = True
        return end_phase, kin


class RetractionPhaseElevationStop(RetractionPhase):
    def __init__(self, phase_settings={'control': ('tether_force_ground', 1200.)}, impose_operational_limits=True):
        """
        Args:
            phase_settings (tuple, optional): Setting parent's `control_settings` attribute.
            impose_operational_limits (bool, optional): Setting parent's `impose_operational_limits` attribute.

        """
        super().__init__(phase_settings, impose_operational_limits)

        # Binary kite aerodynamic state.
        self.kite_powered = False

        # Properties of initial state and final position.
        self.tether_length_start = 385.
        self.elevation_angle_start = 30.*np.pi/180.
        self.elevation_angle_end = 60.*np.pi/180.

        self.fix_tether_length = False  # Should be False for realistic simulation.

    def finalize_start_and_end_kite_obj(self):
        """Finalize the initial state and ending criteria before running the simulation, respectively `kinematics_start`
        and `position_end`."""
        self.kinematics_start = KiteKinematics(self.tether_length_start, self.AZIMUTH_ANGLE,
                                               self.elevation_angle_start, self.COURSE_ANGLE)
        self.position_end = KitePosition(elevation_angle=self.elevation_angle_end)

    def determine_new_kinematics(self, last_kinematics, last_steady_state):
        """Determine kinematic state of the kite for the new time point based on the previous kinematic and steady state
        properties. For the retraction phase, the tether length and elevation angle are updated.

        Args:
            last_kinematics (`KiteKinematics`): Kinematics object of previous time point.
            last_steady_state (`SteadyState`): Steady state of previous time point.

        Returns:
            bool: Flag indicating meeting phase ending criteria.
            `KiteKinematics`: Kinematic state of the kite for the new time point.

        """
        kin = copy(last_kinematics)

        # Determine the difference in tether length and elevation angle for regular time step.
        if not self.fix_tether_length:
            d_tether_length = last_steady_state.reeling_speed*self.time_step
        else:
            d_tether_length = 0.
        d_elevation = last_steady_state.elevation_rate*self.time_step
        if d_elevation < 1e-4 and d_tether_length > 0.:
            raise PhaseError("Reeling out at constant elevation angle during reel-in phase.", 3)

        # Check if target tether length is not exceeded next iteration.
        if kin.elevation_angle + d_elevation < self.position_end.elevation_angle:
            # Set timer and kite kinematics for next iteration.
            self.timer += self.time_step
            kin.straight_tether_length += d_tether_length
            kin.elevation_angle += d_elevation
            kin.update()
            end_phase = False
        else:
            d_elevation_remaining = self.position_end.elevation_angle - kin.elevation_angle
            reduced_time_step = d_elevation_remaining/last_steady_state.elevation_rate

            # Set final timer and kite kinematics.
            self.timer += reduced_time_step
            kin.elevation_angle += d_elevation_remaining
            kin.straight_tether_length += last_steady_state.reeling_speed*reduced_time_step
            kin.update()
            end_phase = True
        return end_phase, kin


class TransitionPhase(Phase):
    """The solution to the quasi-steady motion simulation conform the idealized trajectory assumption for the transition
    phase. Inherits from `Phase`.

    Attributes:
        AZIMUTH_ANGLE (float): Constant azimuth angle [rad] of representative flight state.
        COURSE_ANGLE (float): Constant course angle [rad] of representative flight state.
        tether_length_start (float): Tether length [m] used for initial state.
        elevation_angle_start (float): Elevation angle [rad] used for initial state.
        elevation_angle_end (float): Elevation angle [rad] used for phase ending criteria.

    """
    # General kinematics assumptions of representative flight state for the transition phase.
    AZIMUTH_ANGLE = 0.
    COURSE_ANGLE = 0.

    def __init__(self, phase_settings={'control': ('reeling_factor', 0.)}, impose_operational_limits=True):
        """
        Args:
            phase_settings (tuple, optional): Setting parent's `control_settings` attribute.
            impose_operational_limits (bool, optional): Setting parent's `impose_operational_limits` attribute.

        """
        super().__init__(phase_settings, impose_operational_limits)

        # Binary kite aerodynamic state.
        #TODO: kite powered or not?
        self.kite_powered = True

        # Properties of initial state and final position.
        self.tether_length_start = 240.
        self.elevation_angle_start = 80.*np.pi/180.
        self.elevation_angle_end = 25.*np.pi/180.

    def finalize_start_and_end_kite_obj(self):
        """Finalize the initial state and ending criteria before running the simulation, respectively `kinematics_start`
        and `position_end`."""
        self.kinematics_start = KiteKinematics(self.tether_length_start, self.AZIMUTH_ANGLE,
                                               self.elevation_angle_start, self.COURSE_ANGLE)
        self.position_end = KitePosition(elevation_angle=self.elevation_angle_end)

    def determine_new_kinematics(self, last_kinematics, last_steady_state):
        """Determine kinematic state of the kite for the new time point based on the previous kinematic and steady state
        properties. For the transition phase, the tether length and elevation angle are updated.

        Args:
            last_kinematics (`KiteKinematics`): Kinematics object of previous time point.
            last_steady_state (`SteadyState`): Steady state of previous time point.

        Returns:
            bool: Flag indicating meeting phase ending criteria.
            `KiteKinematics`: Kinematic state of the kite for the new time point.

        """
        kin = copy(last_kinematics)

        # Determine the difference in tether length and elevation angle for regular time step.
        d_tether_length = last_steady_state.reeling_speed*self.time_step
        d_elevation = last_steady_state.elevation_rate*self.time_step

        # Check if target elevation angle is not exceeded next iteration.
        if kin.elevation_angle + d_elevation > self.position_end.elevation_angle:
            # Set timer and kite kinematics for next iteration.
            self.timer += self.time_step
            kin.straight_tether_length += d_tether_length
            kin.elevation_angle += d_elevation
            kin.update()
            end_phase = False
        else:
            # Determine the time needed for reaching the target elevation angle.
            d_elevation_remaining = self.position_end.elevation_angle - kin.elevation_angle
            reduced_time_step = d_elevation_remaining/last_steady_state.elevation_rate

            # Set final timer and kite kinematics.
            self.timer += reduced_time_step
            # if len(self.time) == 1 and reduced_time_step < 1e-6:
            #     raise PhaseError("Negative reduced time step detected.")
            kin.straight_tether_length += last_steady_state.reeling_speed*reduced_time_step
            kin.elevation_angle += d_elevation_remaining
            kin.update()
            end_phase = True
        return end_phase, kin


class TractionConstantElevation:
    def __init__(self, elevation_angle):
        self.elevation_angle = elevation_angle

    def calculate(self, tether_length):
        """Calculate the elevation angle as function of the tether length.

        Returns:
            float: Elevation angle [rad].

        """
        return self.elevation_angle


class TractionVariableElevation(TractionConstantElevation):
    def __init__(self, tether_length0, tether_length1, elevation_angle0, elevation_angle1):
        # Calculating the difference between elevation angle start and the path inclination angle.
        # Applying the cosine rule.
        self.elevation_angle0 = elevation_angle0
        self.a = tether_length0
        self.b = tether_length1
        gamma = elevation_angle0 - elevation_angle1
        c = np.sqrt(self.a**2 + self.b**2 - 2*self.a*self.b*np.cos(gamma))
        # Applying the sine rule.
        if gamma > 0.:
            beta = np.pi - np.arcsin(self.b*np.sin(gamma)/c)
            # Difference between elevation angle start and the path inclination angle.
            self.delta_path_angle = np.pi - beta
        else:
            beta = np.pi + np.arcsin(self.b*np.sin(gamma)/c)
            # Difference between elevation angle start and the path inclination angle.
            self.delta_path_angle = -np.pi + beta

    def calculate(self, tether_length):
        """Calculate the elevation angle as function of the tether length.

        Returns:
            float: Elevation angle [rad].

        """
        # Applying the sine rule.
        b = tether_length
        beta = np.pi - abs(self.delta_path_angle)
        alpha = np.arcsin(self.a*np.sin(beta)/b)

        gamma = np.pi - beta - alpha

        if self.delta_path_angle > 0.:
            elevation_angle = self.elevation_angle0 - gamma
        else:
            elevation_angle = self.elevation_angle0 + gamma
        return elevation_angle


class TractionPhase(Phase):
    """The solution to the quasi-steady motion simulation conform the idealized trajectory assumption for the traction
    phase. Inherits from `Phase`.

    Attributes:
        azimuth_angle (float): Constant azimuth angle [rad] of representative flight state.
        course_angle (float): Constant course angle [rad] of representative flight state.
        tether_length_start (float): Tether length [m] used for initial state.
        tether_length_end (float): Tether length [m] used for phase ending criteria.
        elevation_angle_start_aim (float): Elevation angle [rad] used for initial state.
        elevation_angle_end (float): Elevation angle [rad] at end of the phase.

    """
    def __init__(self, phase_settings={'control': ('reeling_factor', .37)}, impose_operational_limits=True):
        """
        Args:
            phase_settings (tuple, optional): Setting parent's `control_settings` attribute.
            impose_operational_limits (bool, optional): Setting parent's `impose_operational_limits` attribute.

        """
        super().__init__(phase_settings, impose_operational_limits)

        # Binary kite aerodynamic state.
        self.kite_powered = True

        # Kinematics assumptions of representative flight state for the traction phase.
        self.azimuth_angle = phase_settings.get('azimuth_angle', 10. * np.pi / 180.)
        #TODO: Kitepower uses 90 degree course angle, which makes a significant difference. Validation with experiments
        # should show what value is sensible.
        self.course_angle = phase_settings.get('course_angle', 110. * np.pi / 180.)

        # Properties of initial state and final position.
        self.tether_length_start = 240.
        self.tether_length_end = 385.
        self.elevation_angle = TractionConstantElevation(25. * np.pi / 180.)

    def finalize_start_and_end_kite_obj(self):
        """Finalize the initial state and ending criteria before running the simulation, respectively `kinematics_start`
        and `position_end`. Furthermore, calculating `delta_path_angle`."""
        elevation_angle_start = self.elevation_angle.calculate(self.tether_length_start)

        self.kinematics_start = KiteKinematics(self.tether_length_start, self.azimuth_angle,
                                               elevation_angle_start, self.course_angle)
        self.position_end = KitePosition(straight_tether_length=self.tether_length_end)

    def determine_new_kinematics(self, last_kinematics, last_steady_state):
        """Determine kinematic state of the kite for the new time point based on the previous kinematic and steady state
        properties. For the traction phase, the tether length is updated. If the elevation angle at the start and end of
        the phase are different, then also the elevation angle is updated.

        Args:
            last_kinematics (`KiteKinematics`): Kinematics object of previous time point.
            last_steady_state (`SteadyState`): Steady state of previous time point.

        Returns:
            bool: Flag indicating meeting phase ending criteria.
            `KiteKinematics`: Kinematic state of the kite for the new time point.

        """
        kin = copy(last_kinematics)

        # Determine the difference in tether length for regular time step.
        d_tether_length = last_steady_state.reeling_speed * self.time_step

        if d_tether_length < 0.:
            raise PhaseError("Reeling in during reel-out phase.", 2)
        elif last_steady_state.reeling_speed < 1e-6:
            raise PhaseError("Reeling speed too low.")

        # Check if target tether length is not exceeded next iteration.
        if kin.straight_tether_length + d_tether_length < self.position_end.straight_tether_length:
            # Set timer and kite kinematics for next iteration.
            self.timer += self.time_step

            kin.straight_tether_length += d_tether_length
            kin.elevation_angle = self.elevation_angle.calculate(kin.straight_tether_length)
            kin.update()
            end_phase = False
        else:
            # Determine the time needed for reaching the target tether length.
            d_tether_length_remaining = self.position_end.straight_tether_length - kin.straight_tether_length
            reduced_time_step = d_tether_length_remaining / last_steady_state.reeling_speed

            # Set final timer and kite kinematics.
            self.timer += reduced_time_step
            kin.straight_tether_length += d_tether_length_remaining
            kin.elevation_angle = self.elevation_angle.calculate(kin.straight_tether_length)
            kin.update()
            end_phase = True
        return end_phase, kin


class TractionPhaseHybrid(TractionPhase):
    # Estimates how many crosswind patterns can be flown within the traction phase.
    def __init__(self, phase_settings={'control': ('reeling_factor', .37)}, impose_operational_limits=True):
        super().__init__(phase_settings, impose_operational_limits)
        self.tether_length_start_aim = self.tether_length_start

        # State of kite along the cross-wind pattern.
        self.n_crosswind_patterns = 0.

    def run_simulation(self, system_properties, environment_state, steady_state_config={}, timer_start=0., n_patterns=6):
        # TODO: check what n_patterns needs to be
        super().run_simulation(system_properties, environment_state, steady_state_config, timer_start)

        tether_lengths = np.linspace(self.tether_length_start_aim, self.tether_length_end, n_patterns)

        pattern_durations = []
        reeling_speeds = []
        for le in tether_lengths:
            elev = self.elevation_angle.calculate(le)
            kin = KiteKinematics(le, self.azimuth_angle, elev, self.course_angle)
            environment_state.calculate(kin.z)
            rs = self.determine_new_steady_state(kin).reeling_speed
            pattern_settings = {
                'tether_length': le,
                'elevation_angle_ref': elev,
                'control': self.control_settings,
                'time_step': .5,
            }
            pattern = EvaluatePattern(pattern_settings)
            pattern.enable_limit_violation_error = False
            pattern_duration = pattern.calc_performance_along_pattern(system_properties, environment_state,
                                                                              steady_state_config=steady_state_config)
            pattern_durations.append(pattern_duration)
            reeling_speeds.append(rs)

        avg_pattern_duration = np.mean(pattern_durations)
        phase_duration_aim = (self.tether_length_end - self.tether_length_start_aim)/np.mean(reeling_speeds)
        self.n_crosswind_patterns = phase_duration_aim/avg_pattern_duration


class LissajousPattern:
    def __init__(self):
        # Lissajous curve properties for figure 8.
        self.elevation_max = 4 * np.pi / 180  # [rad] sets max (relative) elevation angle: positive value -> flying up
        # at edges
        self.azimuth_max = 20 * np.pi / 180  # [rad] sets max azimuth angle

        # Calculated property.
        self.curve_length_unit_sphere = self.calc_curve_length_unit_sphere()

    def get_properties_along_curve(self, s):
        # Elevation and azimuth as function of normalized arc length.
        theta = self.elevation_max * np.sin(4 * np.pi * s)  # [rad]
        phi = self.azimuth_max * np.sin(2 * np.pi * s)  # [rad]

        # Derivatives wrt normalized arc length.
        dtheta_ds = 4 * np.pi * self.elevation_max * np.cos(4 * np.pi * s)  # [-]
        dphi_ds = 2 * np.pi * self.azimuth_max * np.cos(2 * np.pi * s)  # [-]

        chi = np.arctan2(dphi_ds, -dtheta_ds)

        return theta, phi, chi, dtheta_ds, dphi_ds

    def calc_curve_length_unit_sphere(self):
        # Curve length of pattern on unit sphere.
        s_range = np.linspace(0, 1, 101)
        ds = s_range[1]
        curve_length = 0.

        for s in s_range:
            dtheta_ds, dphi_ds = self.get_properties_along_curve(s)[3:]
            curve_length += np.sqrt(dtheta_ds**2 + dphi_ds**2) * ds

        return curve_length


class LookupPattern:
    #TODO: check if different order filtering smoothens the results
    def __init__(self):
        self.lookup_table = pd.read_csv('flight_data/realistic_pattern.csv', sep=";")
        scale_factor = 1.
        self.lookup_table['azimuth'] = scale_factor * self.lookup_table['azimuth']
        self.lookup_table['elevation'] = scale_factor * self.lookup_table['elevation']
        self.lookup_table['length_unit_sphere'] = scale_factor * self.lookup_table['length_unit_sphere']

        # Calculated property.
        self.curve_length_unit_sphere = self.lookup_table['length_unit_sphere'].iloc[-1]

    def get_properties_along_curve(self, s):
        phi = np.interp(s, self.lookup_table['s'], self.lookup_table['azimuth'])
        theta = np.interp(s, self.lookup_table['s'], self.lookup_table['elevation'])

        i = (self.lookup_table['s'] > s).idxmax()
        dphi = self.lookup_table['azimuth'].iloc[i] - self.lookup_table['azimuth'].iloc[i - 1]
        dtheta = self.lookup_table['elevation'].iloc[i] - self.lookup_table['elevation'].iloc[i - 1]
        chi = np.arctan2(dphi, -dtheta)

        return theta, phi, chi


class TractionPhasePattern(Phase):
    def __init__(self, phase_settings={'control': ('reeling_factor', .37)}, impose_operational_limits=True):
        """
        Args:
            phase_settings (tuple, optional): Setting parent's `control_settings` attribute.
            impose_operational_limits (bool, optional): Setting parent's `impose_operational_limits` attribute.

        """
        super().__init__(phase_settings, impose_operational_limits)

        # Binary kite aerodynamic state.
        self.kite_powered = True

        # Properties of initial state and final position.
        self.tether_length_start = 240.
        self.tether_length_end = 385.
        self.elevation_angle = TractionConstantElevation(25. * np.pi / 180.)

        # State of kite along the cross-wind pattern.
        self.n_crosswind_patterns = 0.
        self.pattern = LissajousPattern()

    def finalize_start_and_end_kite_obj(self):
        """Finalize the initial state and ending criteria before running the simulation, respectively `kinematics_start`
        and `position_end`. Furthermore, calculating `delta_path_angle`."""
        elevation_angle_ref = self.elevation_angle.calculate(self.tether_length_start)
        theta, phi, chi = self.pattern.get_properties_along_curve(self.n_crosswind_patterns % 1)[:3]
        self.kinematics_start = KiteKinematics(self.tether_length_start, phi, elevation_angle_ref+theta, chi)
        self.position_end = KitePosition(straight_tether_length=self.tether_length_end)

    def determine_new_kinematics(self, last_kinematics, last_steady_state):
        """Determine kinematic state of the kite for the new time point based on the previous kinematic and steady state
        properties. For the traction phase, the tether length is updated. If the elevation angle at the start and end of
        the phase are different, then also the elevation angle is updated.

        Args:
            last_kinematics (`KiteKinematics`): Kinematics object of previous time point.
            last_steady_state (`SteadyState`): Steady state of previous time point.

        Returns:
            bool: Flag indicating meeting phase ending criteria.
            `KiteKinematics`: Kinematic state of the kite for the new time point.

        """
        kin = copy(last_kinematics)

        # Determine the difference in tether length for regular time step.
        d_tether_length = last_steady_state.reeling_speed * self.time_step

        if d_tether_length < 0.:
            raise PhaseError("Reeling in during reel-out phase.", 2)
        elif last_steady_state.reeling_speed < 1e-6:
            raise PhaseError("Reeling speed too low.")

        # Check if target tether length is not exceeded next iteration.
        if kin.straight_tether_length + d_tether_length < self.position_end.straight_tether_length:
            dt = self.time_step
            end_phase = False
        else:
            # Determine the time needed for reaching the target tether length.
            d_tether_length = self.position_end.straight_tether_length - kin.straight_tether_length
            dt = d_tether_length / last_steady_state.reeling_speed
            end_phase = True

        # Set next timer and kite kinematics.
        self.timer += dt
        kin.straight_tether_length += d_tether_length
        elevation_angle_ref = self.elevation_angle.calculate(kin.straight_tether_length)

        pattern_length = self.pattern.curve_length_unit_sphere * last_kinematics.straight_tether_length
        d_cross_wind_distance = last_steady_state.kite_tangential_speed * self.time_step

        self.n_crosswind_patterns += d_cross_wind_distance / pattern_length
        theta, phi, chi = self.pattern.get_properties_along_curve(self.n_crosswind_patterns % 1)[:3]
        kin.elevation_angle = elevation_angle_ref + theta
        kin.azimuth_angle = phi
        kin.course_angle = chi
        kin.update()

        return end_phase, kin


class EvaluatePattern(Phase):  # Determine performance along cross wind pattern at representative traction point. Has
    # Phase as parent as it uses its determine_new_steady_state method.
    def __init__(self, settings, impose_operational_limits=True):
        # Simulation setting.
        self.time_step = settings.get('time_step', .5)

        # State of kite along the cross-wind pattern.
        self.n_crosswind_patterns = 0.

        # Representative traction state of kite along the cross-wind pattern.
        self.tether_length = settings['tether_length']
        self.elevation_angle_ref = settings['elevation_angle_ref']

        # Result lists with time and states.
        self.kinematics = None
        self.steady_states = None
        self.time = None
        self.s = None

        # Side conditions.
        self.system_properties = None
        self.environment_state = None
        self.steady_state_config = None

        # Control settings.
        self.control_settings = settings['control']
        self.impose_operational_limits = impose_operational_limits
        self.kite_powered = True
        self.follow_wind = False

        # Monitoring parameters.
        self.timer = None
        self.min_reeling_speed = np.inf
        self.max_reeling_speed = -np.inf
        self.min_tether_force = np.inf  # Forces at ground station.
        self.max_tether_force = -np.inf

        # Monitoring settings.
        self.enable_limit_violation_error = True
        self.pattern = LissajousPattern()

    def calc_performance_along_pattern(self, system_properties, environment_state, n_points=100, steady_state_config={}, print_details=False):
        self.time, self.kinematics, self.steady_states = [0.], [], []
        self.min_reeling_speed, self.max_reeling_speed = np.inf, -np.inf
        self.s = np.linspace(0, 1, n_points)
        ds = self.s[1]

        self.system_properties = system_properties
        self.environment_state = environment_state
        self.steady_state_config = steady_state_config

        pattern_length = self.pattern.curve_length_unit_sphere * self.tether_length
        cos_phi, cos_theta, cos_chi = [], [], []
        valid_pattern = True
        for s in self.s:
            theta, phi, chi = self.pattern.get_properties_along_curve(s)[:3]

            kin = KiteKinematics(self.tether_length, phi, self.elevation_angle_ref + theta, chi)
            self.kinematics.append(kin)

            # Add first time point, kite kinematics, and steady state to corresponding result lists.
            environment_state.calculate(kin.z)
            if self.follow_wind:
                kin.azimuth_angle += environment_state.downwind_direction
                kin.update()
            ss = self.determine_new_steady_state(kin)
            self.steady_states.append(ss)

            cos_phi.append(np.cos(kin.azimuth_angle))
            cos_theta.append(np.cos(kin.elevation_angle))
            cos_chi.append(np.cos(kin.course_angle))

            if s != self.s[-1]:
                if ss.kite_tangential_speed > 0:
                    dt = ds * pattern_length / ss.kite_tangential_speed
                else:
                    dt = 1e1  # Some optimizations where not converging when setting this value to 1e2.
                    valid_pattern = False
                next_time = self.time[-1] + dt

                self.time.append(next_time)

        # if valid_pattern:
        pattern_duration = self.time[-1]
        # else:
        #     pattern_duration = 1e3

        if print_details:
            print("Pattern duration [s]:", pattern_duration)
            i_end = 0
            flying_down = 0.
            for i, chi in enumerate([90, -90, -90, 90]):
                i_start = i_end
                if i == 3:
                    i_end = n_points
                else:
                    i_end += n_points//4
                course_angles = [kin.course_angle for kin in self.kinematics[i_start:i_end]]
                time_window = self.time[i_start:i_end]
                if i <= 1:
                    course_angles = course_angles[::-1]
                    time_window = time_window[::-1]
                t = np.interp(chi*np.pi/180., course_angles, time_window)
                if i in [1, 3]:
                    flying_down += t - t_last
                t_last = t
            print("Flying down [%]:", flying_down/pattern_duration*100.)
            print("Representative azimuth angle [deg]:", np.arccos(np.trapz(cos_phi, self.time)/pattern_duration)*180./np.pi)
            print("Representative elevation angle [deg]:", np.arccos(np.trapz(cos_theta, self.time)/pattern_duration)*180./np.pi)
            print("Representative course angle [deg]:", np.arccos(np.trapz(cos_chi, self.time)/pattern_duration)*180./np.pi)

        return pattern_duration

    def plot_traces(self, x, plot_parameters, y_labels=None, y_scaling=None):
        """Generic plotting method for making a plot of `KiteKinematics` and `SteadyState` attributes.

        Args:
            plot_parameters (tuple): Sequence of `KiteKinematics` or `SteadyState` attributes.
            y_labels (tuple, optional): Y-axis labels corresponding to `plot_parameters`.
            y_scaling (tuple, optional): Scaling factors corresponding to `plot_parameters`.

        """
        data_sources = (self.kinematics, self.steady_states)
        source_labels = ('kin', 'ss')

        plot_traces(x[0], data_sources, source_labels, plot_parameters, y_labels, y_scaling, x_label=x[1])

    def plot_pattern(self):
        plt.figure()
        plt.plot([kin.azimuth_angle*180./np.pi for kin in self.kinematics],
                 [kin.elevation_angle*180./np.pi for kin in self.kinematics])
        kin = self.kinematics[5]
        plt.plot([kin.azimuth_angle*180./np.pi], [kin.elevation_angle*180./np.pi], 's')


class Cycle(TimeSeries):
    """Combination of phases: `RetractionPhase`, `TransitionPhase`, and `TractionPhase`, which together make a pumping
    cycle. Inherits from `TimeSeries`. The retraction phase is simulated first and the traction phase last. The results
    are however manipulated such that the cycle starts with the traction phase.

    Attributes:
        tether_length_start_retraction (float): Tether length [m] at the start of the retraction phase.
        tether_length_end_retraction (float): Tether length [m] at the end of the retraction phase.
        elevation_angle_traction (float): Elevation angle [rad] of the traction phase.
        retraction_phase (`RetractionPhase`): Retraction phase simulation object.
        transition_phase (`TransitionPhase`): Transition phase simulation object.
        traction_phase (`TractionPhase`): Traction phase simulation object.
        follow_wind (bool): Specifies whether kite is 'aligned' with the wind. Controlled azimuth angle is expressed
            w.r.t. wind reference frame if True, or ground reference frame if False.

    """
    def __init__(self, settings=None, impose_operational_limits=True):
        """
        Args:
            control_settings (dict, optional): Collection of the `control_settings` attributes for the 3 phases.
            impose_operational_limits (bool, optional): Setting `impose_operational_limits` attribute of retraction and
                traction phase.

        """
        super().__init__()
        if settings is None:
            settings = {
                'cycle': {
                    'traction_phase': TractionPhase,
                },
                'retraction': {
                    'control': ('tether_force_ground', 1200),
                },
                'transition': {
                    'control': ('reeling_speed', 0.),
                },
                'traction': {
                    'control': ('reeling_factor', .37),
                },
            }
        cycle_settings = settings.get('cycle', {})

        # Properties of idealized pumping cycle trajectory.
        self.tether_length_start_retraction = cycle_settings.get('tether_length_start_retraction', 385.)
        self.tether_length_end_retraction = cycle_settings.get('tether_length_end_retraction', 240.)
        # Setting the lower attribute imposes the traction phase to start at the given length.
        self.tether_length_start_traction = cycle_settings.get('tether_length_start_traction', None)
        self.elevation_angle_traction = cycle_settings.get('elevation_angle_traction', 25.*np.pi/180.)

        # Initiating phases, allows manipulating its attribute before running the simulation.
        self.retraction_phase = RetractionPhase(settings['retraction'], impose_operational_limits)
        self.transition_phase = TransitionPhase(settings['transition'], True)
        self.traction_phase = cycle_settings.get('traction_phase',
                                                 TractionPhase)(settings['traction'], impose_operational_limits)

        self.follow_wind = cycle_settings.get('follow_wind', False)
        self.include_transition_energy = cycle_settings.get('include_transition_energy', True)

    def run_simulation(self, system_properties, environment_state, steady_state_config={},
                       enable_limit_violation_error=False, print_summary=False):
        """Consecutively run the simulations of the 3 phases.

        Args:
            system_properties (`SystemProperties`): Collection of system properties.
            environment_state (`Environment` or child): Specification of environment.
            steady_state_config (dict, optional): Iterative procedure settings for finding the steady state.
            enable_limit_violation_error (bool, optional): Flag specifying whether to raise an error when the reeling
                speed or tether force limit is violated in retraction and traction phase.
            print_summary (bool, optional): Print cycle performance summary to screen if True.

        Returns:
            str: Phase for which the simulation does not seem to reach end criteria, should be either: 'retraction' or
                'traction'.
            float: Time average of the produced power [W].

        """
        if isinstance(environment_state, list):
            env_retr, env_trans, env_trac = environment_state
        else:
            env_retr, env_trans, env_trac = environment_state, environment_state, environment_state
        error_in_phase = None
        reorder = True

        # Start with running the retraction phase, since its start and stop conditions are predefined.
        retr = self.retraction_phase
        retr.follow_wind = self.follow_wind
        retr.enable_limit_violation_error = enable_limit_violation_error

        # Set start and stop conditions of retraction phase.
        retr.tether_length_start = self.tether_length_start_retraction
        retr.tether_length_end = self.tether_length_end_retraction
        retr.elevation_angle_start = self.elevation_angle_traction
        retr.finalize_start_and_end_kite_obj()

        try:
            retr.run_simulation(system_properties, env_retr, steady_state_config, 0.)
            last_straight_tether_length = retr.kinematics[-1].straight_tether_length
        except PhaseError as e:
            if e.code not in [1, 3]:  # Simulation does not seem to reach end criteria.
                raise
            retr.energy = -1e8
            last_straight_tether_length = self.tether_length_end_retraction
            retr.duration = 100.
            error_in_phase = "retraction"
        last_kinematics = retr.kinematics[-1]
        last_time = retr.time[-1]

        # Second, run the transition phase.
        trans = self.transition_phase
        trans.follow_wind = self.follow_wind
        trans.enable_limit_violation_error = False

        # Set start and stop conditions of transition phase.
        trans.tether_length_start = last_straight_tether_length
        trans.elevation_angle_start = last_kinematics.elevation_angle
        trans.elevation_angle_end = self.elevation_angle_traction
        trans.finalize_start_and_end_kite_obj()

        if reorder:
            timer_start = 0
        else:
            timer_start = last_time
        trans.run_simulation(system_properties, env_trans, steady_state_config, timer_start)
        last_kinematics = trans.kinematics[-1]
        last_time = trans.time[-1]
        # trans.average_power = 0
        # trans.energy = 0

        # Third, run the traction phase.
        trac = self.traction_phase
        trac.follow_wind = self.follow_wind
        trac.enable_limit_violation_error = enable_limit_violation_error

        # Start and stop conditions of traction phase. Note that the traction phase uses an azimuth angle in contrast to
        # the other phases, which results in jumps of the kite position.
        if trac.__class__.__name__ == "TractionPhaseHybrid":
            trac.tether_length_start_aim = self.tether_length_end_retraction
        trac.elevation_angle = TractionConstantElevation(self.elevation_angle_traction)
        if self.tether_length_start_traction is None:
            trac.tether_length_start = last_kinematics.straight_tether_length
        else:
            trac.tether_length_start = self.tether_length_start_traction
        trac.tether_length_end = self.tether_length_start_retraction
        trac.finalize_start_and_end_kite_obj()

        try:
            trac.run_simulation(system_properties, env_trac, steady_state_config, last_time)
        except PhaseError as e:
            if e.code not in [1, 2]:  # Simulation does not seem to reach end criteria.
                raise
            trac.energy = -1e2
            trac.duration = 1.
            error_in_phase = "traction"
        last_time = trac.time[-1]

        # Resulting time series
        if reorder:
            self.time = trans.time + trac.time + [t + last_time for t in retr.time]
            self.kinematics = trans.kinematics + trac.kinematics + retr.kinematics
            self.steady_states = trans.steady_states + trac.steady_states + retr.steady_states
        else:
            self.time = retr.time + trans.time + trac.time
            self.kinematics = retr.kinematics + trans.kinematics + trac.kinematics
            self.steady_states = retr.steady_states + trans.steady_states + trac.steady_states
        self.energy = trac.energy + retr.energy
        if self.include_transition_energy:
            self.energy += trans.energy
        self.duration = self.time[-1]
        self.average_power = self.energy / self.time[-1]

        if print_summary:
            print("Total cycle: {:.1f} seconds in which {:.0f}J energy produced.".format(self.time[-1], self.energy))
            print("Mean cycle power: {:.1f}W".format(self.average_power))
            print("Retraction power: {:.1f}W".format(retr.average_power))
            print("Transition power: {:.1f}W".format(trans.average_power))
            print("Traction power: {:.1f}W".format(trac.average_power))

        return error_in_phase, self.average_power

    def trajectory_plot3d(self, fig_num=None):
        """Plot the 3D pumping cycle trajectory of the kite using the identical named methods of the 3 phase objects.

        Args:
            fig_num (int, optional): Number of figure used for the plot, if None a new figure is created.

        """
        if fig_num is None:
            plt.figure()
        fig_num = plt.gcf().number
        plot_kwargs = {'color': 'k'}  #default_colors[2]}
        self.retraction_phase.trajectory_plot3d(fig_num=fig_num, animation=False, plot_kwargs=plot_kwargs)
        self.transition_phase.trajectory_plot3d(fig_num=fig_num, animation=False, plot_kwargs=plot_kwargs)
        self.traction_phase.trajectory_plot3d(fig_num=fig_num, animation=False, plot_kwargs=plot_kwargs)


if __name__ == "__main__":
    # Expected performance summary:
    #   Total cycle: 87.4 seconds in which 51903J energy produced.
    #   Mean cycle power: 593.7W
    #   Retraction power: -5326.4W
    #   Transition power: 3832.6W
    #   Traction power: 4285.0W

    # Create environment object.
    env_state = LogProfile()

    # Create system properties object.
    sys_props = {
        'kite_projected_area': 18,  # [m^2]
        'kite_mass': 20,  # [kg]
        'tether_density': 724,  # [kg/m^3]
        'tether_diameter': 0.004,  # [m]
    }
    sys_props = SystemProperties(sys_props)

    # Create pumping cycle simulation object, run simulation, and plot results.
    settings = {
        'cycle': {
            'traction_phase': TractionPhase,
        },
        'retraction': {
            'control': ('tether_force_ground', 1200),
        },
        'transition': {
            'control': ('reeling_speed', 0.),
        },
        'traction': {
            'control': ('reeling_factor', .37),
            'time_step': .05,
        },
    }
    # pattern_settings = settings['traction']
    # pattern_settings['tether_length'] = 100.
    # pattern_settings['elevation_angle_ref'] = 25.*np.pi/180.
    # cwp = EvaluatePattern(pattern_settings)
    # cwp.calc_performance_along_pattern(sys_props, env_state, 100, print_details=True)
    # # cwp.plot_traces((cwp.s, 'Normalised path distance [-]'), ('reeling_speed', 'power_ground', 'apparent_wind_speed', 'tether_force_ground', 'kite_tangential_speed'),
    # #                 ('Reeling speed [m/s]', 'Power [W]', 'Apparent wind speed [m/s]', 'Tether force [N]', 'Tangential speed [m/s]'))
    # cwp.plot_traces((cwp.s, 'Normalised path distance [-]'), ('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
    #                 ('Radius [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
    #                 (None, 180./np.pi, 180./np.pi, 180./np.pi))
    # cwp.plot_traces((cwp.time, 'Time [s]'), ('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
    #                 ('Radius [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
    #                 (None, 180./np.pi, 180./np.pi, 180./np.pi))
    # cwp.plot_pattern()

    cycle = Cycle(settings)
    cycle.follow_wind = True
    cycle.run_simulation(sys_props, env_state, print_summary=True)
    cycle.time_plot(('reeling_speed', 'power_ground', 'apparent_wind_speed', 'tether_force_ground'),
                    ('Reeling speed [m/s]', 'Power [W]', 'Apparent wind speed [m/s]', 'Tether force [N]'))
    cycle.time_plot(('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
                    ('Radius [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
                    (None, 180./np.pi, 180./np.pi, 180./np.pi))
    cycle.time_plot(('straight_tether_length', 'reeling_speed', 'x', 'y', 'z'),
                    ('r [m]', r'$\dot{\mathrm{r}}$ [m/s]', 'x [m]', 'y [m]', 'z [m]'))
    cycle.trajectory_plot()
    cycle.trajectory_plot3d()
    plt.show()
