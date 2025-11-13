import json
from pathlib import Path

import numpy as np

from system_model import SystemModel
from wind import Wind
from kite import Kite
from tether import RigidLumpedTether
from reelout_phase import Reelout

# ---------------------------------------------------------------------------
# Configuration knobs – tweak these values to experiment with the setup.
# ---------------------------------------------------------------------------
PHYSICAL_CONFIG = {
    "mass_wing": 61,
    "mass_kcu": 30,
    "area_wing": 46.85,
    "tether_diameter": 0.014,
}

PATH_PARAMETERS = {
    "pattern_type": "cst_helix",
    "r0": 200,
    "az_amp0": np.radians(20),
    "beta_amp0": np.radians(20),
    "beta0": np.radians(25),
    # "downloops": True,
}

RADIAL_PARAMETERS = {
    "reeling_strategy": "constant",  # "force" or "constant"
    "reeling_speed": 0.0,  # m/s, only for constant reeling
}

N = 2  # Number of half eight loops
SIM_PARAMETERS = {
    "start_time": 0,
    "end_time": 35,
    "start_angle": 0,
    "end_angle": N * np.pi,
    "n_points": 400,
}

REELOUT_CONFIG = {
    "path_parameters": PATH_PARAMETERS,
    "radial_parameters": RADIAL_PARAMETERS,
    "sim_parameters": SIM_PARAMETERS,
}

AERO_INPUT_FILE = Path("v3_aero_input.json")

WIND_CONFIG = {
    "speed_wind_at_100": 10,
    "z0": 0.002,
    "model_type": "uniform",
}


def load_aero_input(path: Path = AERO_INPUT_FILE):
    """Load aerodynamic input data from disk."""
    with path.open("r") as file:
        return json.load(file)


def build_wind_model(speed_wind_at_100=8, z0=0.01, model_type="uniform"):
    """Create a wind model using the supplied parameters."""
    wind_model = Wind(
        wind_model=model_type,
        z0=z0,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    if model_type == "logarithmic":
        wind_model.speed_friction = speed_friction
    elif model_type == "uniform":
        wind_model.speed_wind_ref = speed_wind_at_100
    return wind_model


def define_system(
    tether_diameter,
    mass_wing,
    mass_kcu,
    area_wing,
    aero_input,
    wind_model,
):
    """Instantiate a SystemModel with the supplied components."""

    tether = RigidLumpedTether(diameter=tether_diameter)
    kite = Kite(
        mass_wing=mass_wing,
        mass_kcu=mass_kcu,
        area_wing=area_wing,
        aero_input=aero_input,
        steering_control="asymmetric",
    )

    model = SystemModel(
        dof=3,
        kite=kite,
        tether=tether,
        wind_model=wind_model,
    )
    return model


def create_system_model(wind_config=WIND_CONFIG, physical_config=PHYSICAL_CONFIG):
    """Assemble the system model using the configuration dictionaries above."""
    aero_input = load_aero_input()
    wind_model = build_wind_model(
        speed_wind_at_100=wind_config["speed_wind_at_100"],
        z0=wind_config["z0"],
        model_type=wind_config["model_type"],
    )
    return define_system(
        tether_diameter=physical_config["tether_diameter"],
        mass_wing=physical_config["mass_wing"],
        mass_kcu=physical_config["mass_kcu"],
        area_wing=physical_config["area_wing"],
        aero_input=aero_input,
        wind_model=wind_model,
    )


def simulate_pattern(
    run_plots=False,
    path_parameters=PATH_PARAMETERS,
    radial_parameters=RADIAL_PARAMETERS,
    sim_parameters=SIM_PARAMETERS,
    wind_config=WIND_CONFIG,
    physical_config=PHYSICAL_CONFIG,
    variables_to_plot=None,
):
    system_model = create_system_model(
        wind_config=wind_config, physical_config=physical_config
    )
    reelout_config = {
        "path_parameters": path_parameters,
        "radial_parameters": radial_parameters,
        "sim_parameters": sim_parameters,
    }
    reelout = Reelout(
        system_model=system_model,
        pattern_config=reelout_config,
        depower=0,
    )
    if variables_to_plot is None:
        variables_to_plot = [
            "speed_tangential",
            "tension_tether_ground",
            "angle_of_attack",
            "distance_radial",
        ]

    phase, axes, slider = reelout.run_simulation(
        run_plots=run_plots, variables_to_plot=variables_to_plot
    )

    return slider, axes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    simulate_pattern(run_plots=True)
    plt.show()
