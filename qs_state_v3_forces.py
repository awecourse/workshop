import contextlib
import io
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from system_model import SystemModel, State
from wind import Wind
from kite import Kite
from tether import RigidLumpedTether

DATA_ROOT = Path(__file__).resolve().parents[0]


def load_v3_aero(path: Optional[Path] = None) -> dict:
    if path is None:
        path = DATA_ROOT / "v3_aero_input.json"
    with open(path, "r") as f:
        return json.load(f)


def build_qs_model(
    aero_input: dict,
    wind_speed: float = 12.0,
    tether_diam: float = 1e-3,
    mass_wing: float = 60.0,
    area_wing: float = 47.0,
) -> SystemModel:
    wind = Wind(wind_model="uniform", z0=0.1)
    wind.speed_wind_ref = float(wind_speed)
    wind.speed_friction = 0.41 * wind_speed / np.log(100 / wind.z0)

    kite = Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=0.0,
        steering_control="asymmetric",
    )
    tether = RigidLumpedTether(diameter=float(tether_diam))
    model = SystemModel(
        dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind
    )
    model.input_depower = 0.0
    return model


def solve_qs_state(
    model: SystemModel,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    course_angle: float = np.pi / 2,
    radius: float = 200.0,
    speed_radial: float = 0.0,
    vt_guess: float = 60.0,
    tension_guess: float = 1e8,
) -> State:
    st0 = State(
        distance_radial=float(radius),
        angle_azimuth=float(azimuth),
        angle_elevation=float(elevation),
        angle_course=float(course_angle),
        speed_radial=float(speed_radial),
        speed_tangential=float(vt_guess),
        input_steering=0.0,
        input_depower=0.0,
        timeder_angle_course=0.0,
        tension_tether_ground=float(tension_guess),
    )
    unknown_vars = [
        "speed_tangential",
        "input_steering",
        "tension_tether_ground",
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        solved = model.solve_quasi_steady(st0, unknown_vars=unknown_vars)
    if solved is None:
        raise RuntimeError(
            "Quasi-steady solver did not converge for the requested configuration."
        )
    return solved


def evaluate_forces(model: SystemModel, state_dict: dict) -> dict:
    out = {}
    for name in ["force_aerodynamic", "force_gravity", "force_tether_at_kite"]:
        try:
            fn = model.extract_function(name)
        except Exception:
            out[name] = None
            continue

        kwargs = {}
        for arg in fn.name_in():
            if arg not in state_dict:
                kwargs = None
                break
            kwargs[arg] = state_dict[arg]
        if kwargs is None:
            out[name] = None
            continue

        try:
            val = fn(**kwargs)[name]
            out[name] = np.asarray(val, dtype=float).ravel()
        except Exception:
            out[name] = None
    return out


def compute_lift_drag(model: SystemModel, state_dict: dict, forces: dict) -> dict:
    aero = forces.get("force_aerodynamic")
    if aero is None:
        return {}
    try:
        fn = model.extract_function("velocity_apparent_wind")
    except Exception:
        return {}

    kwargs = {}
    for arg in fn.name_in():
        if arg not in state_dict:
            return {}
        kwargs[arg] = state_dict[arg]

    try:
        va = np.asarray(fn(**kwargs)["velocity_apparent_wind"], dtype=float).ravel()
    except Exception:
        return {}

    va_norm = np.linalg.norm(va)
    if va_norm == 0:
        return {}

    drag_dir = va / va_norm
    aero_vec = np.asarray(aero, dtype=float)
    drag_mag = np.dot(aero_vec, drag_dir)
    drag_vec = drag_mag * drag_dir
    lift_vec = aero_vec - drag_vec
    return {"force_drag": drag_vec, "force_lift": lift_vec}


def extract_velocities(model: SystemModel, state_dict: dict) -> dict:
    """Extract velocity components for the velocity triangle."""
    velocities = {}

    # Apparent wind velocity
    try:
        fn = model.extract_function("velocity_apparent_wind")
        kwargs = {arg: state_dict[arg] for arg in fn.name_in() if arg in state_dict}
        va = np.asarray(fn(**kwargs)["velocity_apparent_wind"], dtype=float).ravel()
        velocities["velocity_apparent"] = va
    except Exception:
        pass

    # Kite velocity (tangential)
    try:
        fn = model.extract_function("speed_tangential")
        kwargs = {arg: state_dict[arg] for arg in fn.name_in() if arg in state_dict}
        vk = float(fn(**kwargs)["speed_tangential"])
        velocities["velocity_kite"] = vk * np.array([1.0, 0.0, 0.0])
    except Exception:
        pass

    # Kite velocity (radial) --- ADDED CODE ---
    try:
        fn = model.extract_function("speed_radial")
        kwargs = {arg: state_dict[arg] for arg in fn.name_in() if arg in state_dict}
        vr = float(fn(**kwargs)["speed_radial"])
        velocities["velocity_radial"] = vr * np.array([0.0, 0.0, -1.0])
    except Exception:
        pass

    # Wind velocity
    try:
        fn = model.extract_function("velocity_wind")
        kwargs = {arg: state_dict[arg] for arg in fn.name_in() if arg in state_dict}
        vw = np.asarray(fn(**kwargs)["velocity_wind"], dtype=float).ravel()
        velocities["velocity_wind"] = vw
    except Exception:
        pass

    return velocities


def extract_3d_position_velocity(model: SystemModel, state_dict: dict) -> dict:
    """Extract 3D position and velocity in world frame."""
    data_3d = {}

    # Position in world frame
    try:
        fn = model.extract_function("position_W")
        kwargs = {arg: state_dict[arg] for arg in fn.name_in() if arg in state_dict}
        pos_W = np.asarray(fn(**kwargs)["position_W"], dtype=float).ravel()
        data_3d["position_W"] = pos_W
    except Exception:
        pass

    # Velocity in world frame
    try:
        fn = model.extract_function("velocity_kite_W")
        kwargs = {arg: state_dict[arg] for arg in fn.name_in() if arg in state_dict}
        vel_W = np.asarray(fn(**kwargs)["velocity_kite_W"], dtype=float).ravel()
        data_3d["velocity_kite_W"] = vel_W
    except Exception:
        pass

    return data_3d


def naca0012_outline(chord: float = 4.0, n_points: int = 200) -> np.ndarray:
    t = 0.12
    x = np.linspace(0.0, chord, n_points)
    x_c = x / chord
    y_t = (
        5
        * t
        * chord
        * (
            0.2969 * np.sqrt(x_c)
            - 0.1260 * x_c
            - 0.3516 * x_c**2
            + 0.2843 * x_c**3
            - 0.1015 * x_c**4
        )
    )
    upper = np.column_stack((x, y_t))
    lower = np.column_stack((x[::-1], -y_t[::-1]))
    return np.vstack((upper, lower))


def plot_qs_forces(
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    course_deg: float = 90.0,
    wind_speed: float = 9.0,
    mass_wing: float = 40,
    area_wing: float = 19.75,
    speed_radial: float = 2.0,
    radius: float = 200.0,
    chord_length: float = 4.0,
    save_path: Optional[Path] = None,
    show: bool = True,
    close: bool = True,
) -> Tuple[plt.Figure, tuple, State, dict]:
    """
    Plot quasi-steady forces and 3D kite position/velocity.

    Returns:
        fig: Matplotlib figure
        axes: Tuple of (ax_2d, ax_3d) - 2D force plot and 3D position plot
        state: Solved quasi-steady state
        forces: Dictionary of force vectors
    """
    aero_input = load_v3_aero()
    model = build_qs_model(
        aero_input,
        wind_speed=wind_speed,
        tether_diam=1e-3,
        mass_wing=mass_wing,
        area_wing=area_wing,
    )
    state = solve_qs_state(
        model,
        azimuth=np.radians(azimuth_deg),
        elevation=np.radians(elevation_deg),
        course_angle=np.radians(course_deg),
        radius=radius,
        speed_radial=speed_radial,
    )

    state_dict = state.to_dict()
    forces = evaluate_forces(model, state_dict)

    forces.update(compute_lift_drag(model, state_dict, forces))

    # Extract velocities for velocity triangle
    velocities = extract_velocities(model, state_dict)
    # print("Velocities for velocity triangle:", velocities)

    # Extract 3D position and velocity
    data_3d = extract_3d_position_velocity(model, state_dict)

    # Extract angle of attack
    try:
        aoa_fn = model.extract_function("angle_of_attack")
        kwargs = {arg: state_dict[arg] for arg in aoa_fn.name_in() if arg in state_dict}
        angle_attack = float(aoa_fn(**kwargs)["angle_of_attack"])
    except Exception:
        angle_attack = None

    # Create figure with two subplots: 2D force plot (left) and 3D position plot (right)
    fig = plt.figure(figsize=(15, 5.5), facecolor="white")
    ax = fig.add_subplot(121)  # 2D force plot
    ax.set_facecolor("white")
    ax3d = fig.add_subplot(122, projection="3d")  # 3D position plot

    legend_handles = {
        "Kite": Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="k",
            markersize=6,
            label="Kite",
        ),
        "Bridle point B": Line2D(
            [0],
            [0],
            marker="x",
            color="k",
            linestyle="none",
            markersize=8,
            label="Bridle point B",
        ),
    }

    vector_specs = {
        "force_lift": {"color": "C0", "label": "Lift"},
        "force_drag": {"color": "C3", "label": "Drag"},
        "force_aerodynamic": {"color": "C4", "label": "Aerodynamic"},
        "force_gravity": {"color": "C1", "label": "Gravity"},
        "force_tether_at_kite": {"color": "C2", "label": "Tether"},
    }

    pos = np.array([0, 0, 0])
    kite_point = pos[[0, 2]]
    bridle_point = np.array([0.0, -5.0])
    airfoil = naca0012_outline(chord=chord_length, n_points=200)
    chord_dir = np.array([-1.0, 0.0])

    thickness_dir = np.array([-chord_dir[1], chord_dir[0]])
    translation = kite_point - chord_dir * (chord_length / 3.0)
    airfoil_pts = np.array(
        [translation + s * chord_dir + t * thickness_dir for s, t in airfoil]
    )
    airfoil_handle = ax.fill(
        airfoil_pts[:, 0], airfoil_pts[:, 1], color="gray", alpha=0.3, label="NACA0012"
    )[0]
    if "NACA0012" not in legend_handles:
        legend_handles["NACA0012"] = airfoil_handle

    leading_edge = translation
    trailing_edge = translation + chord_dir * chord_length
    ax.plot(
        [bridle_point[0], leading_edge[0]],
        [bridle_point[1], leading_edge[1]],
        "k-",
        linewidth=1.0,
    )
    ax.plot(
        [bridle_point[0], trailing_edge[0]],
        [bridle_point[1], trailing_edge[1]],
        "k-",
        linewidth=1.0,
    )

    ax.scatter(*kite_point, color="k", s=40)
    ax.scatter(*bridle_point, color="k", marker="x", s=50)

    ax.plot(
        [bridle_point[0], kite_point[0]],
        [bridle_point[1], kite_point[1]],
        "k--",
        linewidth=1.0,
    )

    vectors_xz = {}
    for name, spec in vector_specs.items():
        vec = forces.get(name)
        if vec is not None:
            vectors_xz[name] = np.array([vec[0], vec[2]])

    # Calculate separate scaling factors
    # Gravity gets its own scale to be visible, others share a scale
    vectors_no_grav = {k: v for k, v in vectors_xz.items() if k != "force_gravity"}
    max_mag_no_grav = max(
        (np.linalg.norm(v) for v in vectors_no_grav.values()), default=1.0
    )
    target_length = max(2.0, 0.35 * np.linalg.norm(kite_point))
    scale_factor = target_length / max_mag_no_grav if max_mag_no_grav > 0 else 1.0

    # Scale gravity to be 20-30% of the main force vectors for visibility
    gravity_vec = vectors_xz.get("force_gravity")
    gravity_scale_factor = 2.2 * scale_factor

    for name, spec in vector_specs.items():
        vec_xz = vectors_xz.get(name)
        if vec_xz is None:
            continue

        # Use different scale for gravity
        if name == "force_gravity":
            scaled_vec = vec_xz * gravity_scale_factor
            # Add asterisk to label to indicate different scale
            label_text = spec["label"] + "*"
        else:
            scaled_vec = vec_xz * scale_factor
            label_text = spec["label"]

        origin = bridle_point if name == "force_tether_at_kite" else kite_point
        ax.quiver(
            origin[0],
            origin[1],
            scaled_vec[0],
            scaled_vec[1],
            color=spec["color"],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.007,
        )
        if label_text not in legend_handles:
            legend_handles[label_text] = Line2D(
                [0, 1], [0, 0], color=spec["color"], linewidth=2, label=label_text
            )

    # --- Velocity Triangle at Leading Edge ---
    # Draw velocity triangle: v_a = v_wind - v_kite
    # where v_kite = v_tangential + v_radial
    velocity_specs = {
        "velocity_apparent": {
            "color": "darkred",
            "label": "V_a (apparent)",
            "linewidth": 2.5,
        },
        "velocity_kite": {"color": "darkblue", "label": "V_kite", "linewidth": 2.0},
        "velocity_wind": {"color": "darkgreen", "label": "V_wind", "linewidth": 2.0},
        "velocity_radial": {"color": "purple", "label": "V_radial", "linewidth": 1.5},
    }

    # Position velocity triangle at leading edge
    vel_origin = leading_edge.copy()

    # Scale velocities to reasonable arrow size (different from force scaling)
    vel_scale = 0.12  # adjust to make arrows visible but not overwhelming

    velocities_xz = {}
    for name in velocity_specs.keys():
        if name in velocities:
            vec = velocities[name]
            velocities_xz[name] = np.array([vec[0], vec[2]])

    # Draw velocity triangle in sequence:
    # 1. Start at leading edge
    # 2. Draw -v_kite (pointing backwards from kite motion)
    # 3. From tip of -v_kite, draw v_radial
    # 4. From tip of v_radial, draw v_wind
    # 5. Draw v_a from origin to final point (should close the triangle)

    if "velocity_kite" in velocities_xz and "velocity_wind" in velocities_xz:
        vk_xz = velocities_xz["velocity_kite"]
        vw_xz = velocities_xz["velocity_wind"]
        vr_xz = velocities_xz.get("velocity_radial", np.array([0.0, 0.0]))

        # Draw kite velocity (from origin)
        vk_scaled = vk_xz * vel_scale
        ax.quiver(
            vel_origin[0] + vk_scaled[0],
            vel_origin[1] + vk_scaled[1],
            -vk_scaled[0],
            -vk_scaled[1],
            color=velocity_specs["velocity_kite"]["color"],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.006,
            linewidth=velocity_specs["velocity_kite"]["linewidth"],
            alpha=0.8,
        )

        # Draw apparent velocity (closes the triangle, points TO the leading edge)
        if "velocity_apparent" in velocities_xz:
            va_xz = velocities_xz["velocity_apparent"]
            va_scaled = -va_xz * vel_scale  # Negative to point towards leading edge
            ax.quiver(
                vel_origin[0] + va_scaled[0],
                vel_origin[1] + va_scaled[1],
                -va_scaled[0],
                -va_scaled[1],
                color=velocity_specs["velocity_apparent"]["color"],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.007,
                linewidth=velocity_specs["velocity_apparent"]["linewidth"],
                alpha=0.9,
            )

        # Draw wind velocity (from tip of kite + radial)
        vw_origin = vel_origin + va_scaled
        vw_scaled = vw_xz * vel_scale
        ax.quiver(
            vw_origin[0],
            vw_origin[1],
            vw_scaled[0],
            vw_scaled[1],
            color=velocity_specs["velocity_wind"]["color"],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.006,
            linewidth=velocity_specs["velocity_wind"]["linewidth"],
            alpha=0.8,
        )
        # Draw radial velocity (from tip of kite velocity)
        vr_origin = vel_origin + va_scaled + vw_scaled
        vr_scaled = vr_xz * vel_scale
        if np.linalg.norm(vr_scaled) > 0.01:  # Only draw if non-negligible
            ax.quiver(
                vr_origin[0] + 0.2,
                vr_origin[1],
                vr_scaled[0],
                vr_scaled[1],
                color=velocity_specs["velocity_radial"]["color"],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
                linewidth=velocity_specs["velocity_radial"]["linewidth"],
                alpha=0.8,
            )
        else:
            vr_scaled = np.array([0.0, 0.0])

        # Add velocity legend handles
        for name, spec in velocity_specs.items():
            if name in velocities_xz and spec["label"] not in legend_handles:
                legend_handles[spec["label"]] = Line2D(
                    [0, 1],
                    [0, 0],
                    color=spec["color"],
                    linewidth=spec["linewidth"],
                    label=spec["label"],
                )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    pad = chord_length * 0.25 + 1.5
    x_vals = [bridle_point[0], kite_point[0], leading_edge[0], trailing_edge[0]]
    z_vals = [bridle_point[1], kite_point[1], leading_edge[1], trailing_edge[1]]
    ax.set_xlim(min(x_vals) - pad, max(x_vals) + pad + 2)
    ax.set_ylim(min(z_vals) - pad, max(z_vals) + pad)
    ax.set_title("Force Vectors (X–Z Plane Projection)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # Build summary text with angle of attack if available
    summary_lines = [
        f"V_t = {state.speed_tangential:.2f} m/s",
        f"Tether tension = {state.tension_tether_ground:.1f} N",
    ]
    if angle_attack is not None:
        summary_lines.append(f"α (AoA) = {np.degrees(angle_attack):.2f}°")
    summary_lines.append(
        f"* Gravity scaled {gravity_scale_factor/scale_factor:.1f}× for visibility"
    )
    summary = "\n".join(summary_lines)

    ax.text(
        0.02,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    ax.legend(
        handles=list(legend_handles.values()),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    # --- 3D Position and Velocity Plot ---
    if "position_W" in data_3d:
        pos_W = data_3d["position_W"]

        # Plot ground origin
        ax3d.scatter(
            [0], [0], [0], color="black", s=100, marker="o", label="Ground (origin)"
        )

        # Plot kite position
        ax3d.scatter(
            [pos_W[0]],
            [pos_W[1]],
            [pos_W[2]],
            color="red",
            s=100,
            marker="o",
            label="Kite position",
        )

        # Plot tether line from ground to kite
        ax3d.plot(
            [0, pos_W[0]],
            [0, pos_W[1]],
            [0, pos_W[2]],
            "k--",
            linewidth=1.5,
            alpha=0.6,
            label="Tether",
        )

        # Plot velocity vector if available
        if "velocity_kite_W" in data_3d:
            vel_W = data_3d["velocity_kite_W"]
            vel_scale = 3.0  # Scale factor for visibility
            ax3d.quiver(
                pos_W[0],
                pos_W[1],
                pos_W[2],
                vel_W[0],
                vel_W[1],
                vel_W[2],
                color="blue",
                arrow_length_ratio=0.2,
                linewidth=2.5,
                label=f"Velocity (×{vel_scale:.1f})",
                length=vel_scale,
                normalize=False,
            )

        # Set labels and title
        ax3d.set_xlabel("X [m] (East)")
        ax3d.set_ylabel("Y [m] (North)")
        ax3d.set_zlabel("Z [m] (Up)")
        ax3d.set_title("Kite Position & Velocity (World Frame)")

        # Set equal aspect ratio
        max_range = max(abs(pos_W[0]), abs(pos_W[1]), abs(pos_W[2])) * 1.2
        ax3d.set_xlim([-max_range, max_range])
        ax3d.set_ylim([-max_range, max_range])
        ax3d.set_zlim([0, max_range * 1.5])

        # Add legend
        ax3d.legend(loc="upper left", frameon=True, framealpha=0.9)

        # Set viewing angle for better perspective
        ax3d.view_init(elev=20, azim=45)

        # Grid
        ax3d.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, (ax, ax3d), state, forces


def main():
    plot_qs_forces(
        show=True,
        close=False,
    )


if __name__ == "__main__":
    main()
