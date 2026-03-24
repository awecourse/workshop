"""
UI helpers for the quasi-steady forces demo.

This module encapsulates the ipywidgets UI so notebooks can just call one
function without showing the widget wiring.
"""

from typing import Optional

import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

from qs_state_forces import plot_qs_forces, build_qs_model


def launch_qs_forces_ui(
    sys_model,
    aero_input,
    kite,
    tether,
    *,
    course_deg: float = 90.0,
    mass_wing: float = 0.0,
    elevation_deg: float = 0.0,
    azimuth_deg: float = 0.0,
    wind_speed: float = 9.0,
    speed_radial: float = 0.0,
    depower: float = 0.0,
    continuous_update: bool = True,
):
    """Display the interactive UI for the QS forces plot.

    Parameters are initial slider values; all are optional. The function will
    render the controls and the figure side-by-side and update the plot when
    sliders change.
    """

    # Controls
    style = {"description_width": "50px"}
    slider_layout = widgets.Layout(width="300px")

    course_slider = widgets.FloatSlider(
        min=0,
        max=360,
        step=5,
        value=float(course_deg),
        description="Course °:",
        style=style,
        layout=slider_layout,
        readout=True,
        readout_format=".1f",
        continuous_update=continuous_update,
    )
    mass_slider = widgets.FloatSlider(
        min=0,
        max=120,
        step=5,
        value=float(mass_wing),
        description="Mass kg:",
        style=style,
        layout=slider_layout,
        readout=True,
        readout_format=".1f",
        continuous_update=continuous_update,
    )
    elevation_slider = widgets.FloatSlider(
        min=0,
        max=90,
        step=0.5,
        value=float(elevation_deg),
        description="Elev. °:",
        style=style,
        layout=slider_layout,
        readout=True,
        readout_format=".1f",
        continuous_update=continuous_update,
    )
    azimuth_slider = widgets.FloatSlider(
        min=-90,
        max=90,
        step=2,
        value=float(azimuth_deg),
        description="Azim. °:",
        style=style,
        layout=slider_layout,
        readout=True,
        readout_format=".1f",
        continuous_update=continuous_update,
    )
    wind_slider = widgets.FloatSlider(
        min=4,
        max=25,
        step=1,
        value=float(wind_speed),
        description="Wind m/s:",
        style=style,
        layout=slider_layout,
        readout=True,
        readout_format=".1f",
        continuous_update=continuous_update,
    )
    radial_slider = widgets.FloatSlider(
        min=-10,
        max=6,
        step=0.1,
        value=float(speed_radial),
        description="Radial m/s:",
        style=style,
        layout=slider_layout,
        readout=True,
        readout_format=".1f",
        continuous_update=continuous_update,
    )
    depower_slider = widgets.FloatSlider(
        min=0,
        max=1,
        step=1,
        value=float(depower),
        description="Depower:",
        style=style,
        layout=slider_layout,
        readout=True,
        readout_format=".0f",
        continuous_update=continuous_update,
    )

    controls_layout = widgets.Layout(width="300px", min_width="300px")
    controls = widgets.VBox(
        [
            course_slider,
            mass_slider,
            elevation_slider,
            azimuth_slider,
            wind_slider,
            radial_slider,
            depower_slider,
        ],
        layout=controls_layout,
    )
    out = widgets.Output()

    def render_forces(change=None):  # noqa: ARG001 - signature per widgets.observe
        with out:
            out.clear_output(wait=True)

            # Rebuild a fresh SystemModel for updated parameters to ensure
            # CasADi symbolics are consistent with the current settings.
            try:
                area_val = kite.area_wing
                tether_d = tether.diameter
                sys_model_local = build_qs_model(
                    aero_input=aero_input,
                    wind_speed=float(wind_slider.value),
                    tether_diam=float(tether_d),
                    mass_wing=float(mass_slider.value),
                    area_wing=float(area_val),
                )
            except Exception:
                sys_model_local = sys_model

            fig, axes, state, forces = plot_qs_forces(
                course_deg=float(course_slider.value),
                mass_wing=float(mass_slider.value),
                area_wing = float(kite.area_wing),
                elevation_deg=float(elevation_slider.value),
                azimuth_deg=float(azimuth_slider.value),
                wind_speed=float(wind_slider.value),
                speed_radial=float(radial_slider.value),
                depower=float(depower_slider.value),
                sys_model=sys_model_local,
                show=False,
                close=False,
            )
            display(fig)
            plt.close(fig)

    for slider in [
        course_slider,
        mass_slider,
        elevation_slider,
        azimuth_slider,
        wind_slider,
        radial_slider,
        depower_slider,
    ]:
        slider.observe(render_forces, names="value")

    # Initial draw
    render_forces()
    display(widgets.HBox([controls, out]))
