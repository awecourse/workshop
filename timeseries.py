from system_model import SystemModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
from typing import Collection

PLOT_LABELS = {
    "acceleration_normal": "$a_n$ ($m/s^2$)",
    "acceleration_radial": "$a_r$ ($m/s^2$)",
    "acceleration_tangential": "$a_{\\tau}$ ($m/s^2$)",
    "angle_course": "$\\chi$ ($^\\circ$)",
    "angle_flight_path_aerodynamic": "$\\gamma_a$ ($^\\circ$)",
    "angle_heading_aerodynamic": "$\\chi_a$ ($^\\circ$)",
    "steering_input": "$u_s$",
    "angle_roll": "$\\phi_a$ ($^\\circ$)",
    "dchi_ds": "$d\\chi / ds$",
    "distance_radial": "$r$ (m)",
    "force_tether_talmar": "$F_{t, \\mathrm{Talmar}}}$",
    "input_steering": "$u_s$",
    "phi_unwrapped": "$\\phi_a$ ($^\\circ$)",
    "ratio_kinematic": "$\\kappa$",
    "ratio_tether": "$\\frac{F_t}{F_{t, \\mathrm{Talmar}}}}$",
    "s": "s (-)",
    "s_dot": "$\\dot{s}$ (-)",
    "s_ddot": "$\\ddot{s}$ (-)",
    "speed": "$v_k$ (m/s)",
    "speed_radial": "$v_r$ (m/s)",
    "speed_tangential": "$v_\\tau$ (m/s)",
    "speed_wind_true": "$v_{w,true}$ (m/s)",
    "speed_wind_apparent": "$v_{w,app}$ (m/s)",
    "tension_tether_ground": "$F_{t,g}$ (N)",
    "tension_kite": "$F_{t,k}$ (N)",
    "power_ground": "$P_g$ (W)",
    "kite_angle_of_attack": "$\\alpha$ ($^\\circ$)",
    "time": "$t$ (s)",
    "timeder_angle_course": "$\\dot{\\chi}$ ($^\\circ$/s)",
    "timeder_speed_radial": "$\\dot{v}_r$ ($m/s^2$)",
    "timeder_speed_tangential": "$\\dot{v}_\\tau$ ($m/s^2$)",
    "x": "x (m)",
    "y": "y (m)",
    "z": "z (m)",
    "input_steering": "$u_s$",
    "phase": "$\\Phi (^\circ)$",
    "angle_azimuth": "$\\phi (^\\circ)$",
    "angle_elevation": "$\\beta (^\\circ)$",
    "angle_of_attack": "$\\alpha (^\\circ)$",
}

PLOT_PARAMETERS = [
    "speed_tangential",
    "speed_radial",
    "input_steering",
    "tension_tether_ground",
]


class TimeSeries:
    """

    Attributes:
        states (list): Time series of `SteadyState` objects.
        kite_model (SystemModel): System model used to generate the time series.

    """

    def __init__(
        self,
        kite_model: SystemModel,
    ):

        # System configuration.
        self.kite_model = kite_model

        # Time series states.
        self.states = []

    # @property
    # def converged(self):
    #     return np.all(np.array([s.converged for s in self.states])) if len(self.states) > 0 else False

    # @property
    # def duration(self):
    #     if self.states:
    #         return self.states[-1].time - self.states[0].time
    #     else:
    #         return np.nan

    @property
    def energy(self):
        tension = self.return_variable("tension_tether_ground")
        speed_radial = self.return_variable("speed_radial")
        time = self.return_variable("t")
        dt = np.diff(time, prepend=time[0])
        energy = np.sum(tension * speed_radial * dt)
        return energy

    @property
    def total_time(self):
        time = self.return_variable("t")
        return time[-1] - time[0]

    # @property
    # def average_power(self):
    #     return self.energy / self.duration

    # @property
    # def average_factor_reeling(self):
    #     return np.trapz(np.array([(s.factor_reeling, s.time) for s in self.states])) / self.duration

    # @property
    # def average_tension_ground(self):
    #     tensions = np.array([s.tension_ground for s in self.states])
    #     times = np.array([s.time for s in self.states])
    #     integrated_tension = np.trapz(tensions, times)
    #     return integrated_tension / self.duration

    def return_variable(self, variable: str):

        def _lookup_optional(state_obj, key):
            if isinstance(state_obj, dict):
                return state_obj.get(key)
            return getattr(state_obj, key, None)

        def _lookup_required(state_obj, key):
            if isinstance(state_obj, dict):
                return state_obj[key]
            return getattr(state_obj, key)

        var = []
        for state in self.states:
            value = _lookup_optional(state, variable)

            if value is not None:
                try:
                    var.append(float(value))
                except (TypeError, ValueError):
                    var.append(float(np.array(value).item()))
                continue

            if variable in {"x", "y", "z"}:
                r_val = _lookup_optional(state, "distance_radial")
                beta_val = _lookup_optional(state, "angle_elevation")
                phi_val = _lookup_optional(state, "angle_azimuth")

                if r_val is not None and beta_val is not None and phi_val is not None:
                    if variable == "x":
                        computed = r_val * np.cos(beta_val) * np.cos(phi_val)
                    elif variable == "y":
                        computed = r_val * np.cos(beta_val) * np.sin(phi_val)
                    else:
                        computed = r_val * np.sin(beta_val)
                    var.append(float(computed))
                    continue

            try:
                var_func = self.kite_model.extract_function(variable)
                input_dict = {
                    name: _lookup_required(state, name) for name in var_func.name_in()
                }
                output = var_func(**input_dict)[variable]
                var.append(float(output))
            except Exception:
                var_func = self.km_param.extract_function(variable)
                input_dict = {
                    name: _lookup_required(state, name) for name in var_func.name_in()
                }
                output = var_func(**input_dict)[variable]
                var.append(float(output))

        return np.array(var)

    def plot_trace_on_plane(
        self,
        plot_markers=True,
        plot_kwargs=None,
        ax=None,
        gradient_color: tuple = None,
        plane=("x", "z"),
    ):
        """Plot of the downwind versus vertical position of the kite.

        Parameters:
            :param plot_kwargs: Line plot keyword arguments.
            :param plot_markers: Use the steady state results to mark non-converged points and points where control or
                path limits are violated.
            :param gradient_color: tuple of (attribute, colormap) to shade the trajectory, e.g. ('speed', 'coolwarm')
            :param plane: tuple of state attributes that define the plane, e.g. ('x', 'z')

        """
        if plot_kwargs is None:
            plot_kwargs = {}

        norm = plot_kwargs.get("norm", None)
        linewidth = plot_kwargs.get("linewidth", 2)
        figsize = plot_kwargs.get("figsize", (None, None))
        cbar = plot_kwargs.get("cbar", True)
        legend = plot_kwargs.get("legend", False)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        x = self.return_variable(plane[0])
        y = self.return_variable(plane[1])
        speed = self.return_variable("speed_tangential")

        if gradient_color is None:
            # ax.scatter(x_traj, z_traj, **plot_kwargs)  # Scatter plots the dots at each time step.
            ax.plot(x, y)
        else:

            vals = np.array(
                [s.__getattribute__(gradient_color[0]) for s in self.states]
            )

            if norm is None:
                norm = Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
            cmap = plt.get_cmap(gradient_color[1])

            # Matplotlib has no feature to plot colormap over line..
            # So we plot each line segment individually and assign a color
            points = np.array([x, y]).T
            fc = cmap(norm((vals[:-1] + vals[1:]) / 2))
            for i, segment in enumerate(zip(points[:-1], points[1:])):
                ax.plot(*np.array(segment).T, c=fc[i], linewidth=linewidth)

            # Create colorbar
            m = cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array(np.linspace(norm.vmin, norm.vmax, 30))

            if cbar:
                cb = fig.colorbar(
                    m,
                    aspect=15,
                    label=PLOT_LABELS.get(gradient_color[0], gradient_color[0]),
                    ax=ax,
                )
            # cb.set_ticks(np.linspace(norm.vmin, norm.vmax, 8))

        # if plot_markers:
        #     # Plot all points for which the steady state did not converge.
        #     for state in self.states:
        #         if not state.converged:
        #             ax.plot(
        #                 getattr(state, plane[0]),
        #                 getattr(state, plane[1]),
        #                 'kx', label='not converged'
        #             )

        #         # Plot all points for which control limits are violated
        #         for violation in state.assess_limit_violations(self.control_limits):
        #             ax.plot(
        #                 getattr(violation, plane[0]),
        #                 getattr(violation, plane[1]),
        #                 'ro',
        #                 label=violation.summary
        #             )

        #         # Plot all points for which path limits are violated
        #         for violation in state.assess_limit_violations(self.path_limits):
        #             ax.plot(
        #                 getattr(violation.state, plane[0]),
        #                 getattr(violation.state, plane[1]),
        #                 'go',
        #                 label=violation.summary
        #             )

        ax.set_xlabel(PLOT_LABELS.get(plane[0], plane[0]))
        ax.set_ylabel(PLOT_LABELS.get(plane[1], plane[1]))

        # if ax.get_xlim()[0] > 0.:
        #     plt.xlim([0., None])
        # plt.ylim([0., None])
        ax.grid(True)
        ax.set_aspect("equal")

        if legend:
            ax.legend(loc=legend if type(legend) is str else None)

        return fig, ax

    def trajectory_plot3d(
        self,
        fig=None,
        ax=None,
        animate=False,
        animate_kwargs=None,
        plot_markers=None,
        plot_kwargs=None,
        gradient_color=None,
    ):
        """Animation of the 3D trajectory of the kite.

        Args:
            fig_num (int, optional): Number of figure used for the plot, if None a new figure is created.
            animate (bool, optional): Make animation of the plot by changing the view angle.
            plot_kwargs (dict, optional): Line plot keyword arguments.
            plot_point_type (int, optional): If not None, only plot points for which the phase identifier corresponds to
                the given integer.
            gradient_color: tuple of attribute + colormap name

        """
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.colors import ListedColormap

        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
        else:
            fig = ax.figure

        if plot_markers is None:
            plot_markers = []
        if plot_kwargs is None:
            plot_kwargs = {}
        if animate_kwargs is None:
            animate_kwargs = {}

        label = plot_kwargs.get("label", None)
        marker_label = plot_kwargs.get("marker_label", None)
        marker_color = plot_kwargs.get("marker_color", None)
        norm = plot_kwargs.get("norm", None)
        plot_ground_station = plot_kwargs.get("ground_station", True)
        color = plot_kwargs.get("color", None)
        legend = plot_kwargs.get("legend", False)

        t = self.return_variable("t")
        x = self.return_variable("x")
        y = self.return_variable("y")
        z = self.return_variable("z")
        speed = self.return_variable("speed_tangential")

        t = t[~np.isnan(t)]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        z = z[~np.isnan(z)]
        speed = speed[~np.isnan(speed)]

        t = t.round(6)  # required to get rid of numeric error

        if gradient_color is not None:
            vals = np.array(
                [s.__getattribute__(gradient_color[0]) for s in self.states]
            )

            if "angle" in gradient_color[0]:
                vals = np.degrees(vals)

            norm = (
                Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
                if norm is None
                else norm
            )
            cmap = plt.get_cmap(gradient_color[1])

            # Matplotlib has no feature to plot colormap over line..
            # So we plot each line segment individually and assign a color
            points = np.array([x, y, z]).T
            fc = cmap(norm((vals[:-1] + vals[1:]) / 2))
            for i, segment in enumerate(zip(points[:-1], points[1:])):
                start, end = segment
                if (
                    np.linalg.norm(np.array(end) - np.array(start)) < 10
                ):  # Don't plot big discontinuities. TODO remove?
                    ax.plot(*np.array(segment).T, c=fc[i])

            # Create colorbar
            m = cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array(vals)

            if plot_kwargs.get("colorbar", True):
                fig.colorbar(
                    m,
                    shrink=0.5,
                    aspect=10,
                    label=PLOT_LABELS.get(gradient_color[0], gradient_color[0]),
                    ax=ax,
                )

        else:
            ax.plot(x, y, z, label=label, color=color)

        # Plot the markers if given
        if plot_markers:
            ax.plot(
                x[np.isin(t, plot_markers)],
                y[np.isin(t, plot_markers)],
                z[np.isin(t, plot_markers)],
                "s",
                markerfacecolor="None",
                label=marker_label,
                color=marker_color,
            )

        if legend:
            ax.legend()

        if plot_ground_station:
            ax.plot(0, 0, 0, marker="o", color="tab:brown")  # plot ground station

        if plot_kwargs.get("labels", True):
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")

        if not plot_kwargs.get("ticks", True):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        plt.grid(True)

        x_min, x_max = ax.get_xlim()
        ax.set_ylim(
            -0.5 * (x_max - x_min), 0.5 * (x_max - x_min)
        )  # set y lim as the same of x, but centered
        ax.set_zlim(
            ax.get_zlim()[0], max(ax.get_zlim()[1], 100)
        )  # z lim minimum height 100 m seems reasonable
        ax.set_aspect(
            "equal"
        )  # Set equal aspect ratio of ax, else the path looks distorted

        if animate:
            # Rotate the axes and update plot.
            def init():
                ax.view_init(animate_kwargs.get("elevation_angle", 30), 0)
                return [fig]

            def animate(i):
                ax.view_init(animate_kwargs.get("elevation_angle", 30), i)
                return [fig]

            anim = animation.FuncAnimation(
                fig, animate, init_func=init, frames=720, interval=2, blit=True
            )
            writervideo = animation.FFMpegWriter(fps=30)
            anim.save("trajectory_plot.mp4", writer=writervideo)

        ax.view_init(30, 50)

        return fig, ax

    def plot_traces(
        self,
        y_params: Collection,
        x_param: str = "t",
        y_labels: dict = None,
        y_scaling=None,
        plot_markers=None,
        fig_num=None,
        axes=None,
        plot_kwargs: dict = None,
    ):
        """Plot the time trace of a parameter from multiple sources.

        Args:
            y_params: list of strings with y_labels to plot
            plot_markers: list of x-values at which to plot a marker
            y_labels (tuple, optional): Y-axis y_labels corresponding to `plot_parameters`.
            y_scaling (tuple, optional): Scaling factors corresponding to `plot_parameters`.
            fig_num (int, optional): Number of figure used for the plot, if None a new figure is created..

            plot_kwargs:
                legend: bool or str. If str, this will be the location at which legend is plotted.

        :return fig, axes. Axes is always a list of Axis objects, even if only one 1 y parameter is plotted.
        """

        if plot_kwargs is None:
            plot_kwargs = {}

        unwrap = plot_kwargs.get("unwrap", True)
        remove_x_labels = plot_kwargs.get("remove_x_labels", False)
        label = plot_kwargs.get("label", None)
        linestyle = plot_kwargs.get("linestyle", None)
        marker_label = plot_kwargs.get("marker_label", None)
        marker_color = plot_kwargs.get("marker_color", None)
        legend = plot_kwargs.get("legend", False)
        color = plot_kwargs.get("color", None)
        x_label = plot_kwargs.get("x_label", PLOT_LABELS.get(x_param, x_param))

        ncols = plot_kwargs.get("ncols", 1)
        nrows = int(np.ceil(len(y_params) / ncols))

        if y_labels is None:
            y_labels = {}
        if y_scaling is None:
            y_scaling = [None for _ in range(len(y_params))]
        if fig_num:
            axes = plt.figure(fig_num).get_axes()
        if axes is None:
            fig, axes = plt.subplots(nrows, ncols, sharex="all", num=fig_num)
            if len(y_params) == 1:
                axes = [axes]
        else:
            fig = axes[0].figure

        if plot_markers is None:
            plot_markers = []

        x = self.return_variable(x_param)

        for i, (p, f, ax) in enumerate(zip(y_params, y_scaling, axes)):
            try:
                y = self.return_variable(p)
            except AttributeError:
                print(p)
                print(
                    f"Cannot plot the trace of attribute {p} as it does not exist. Valid attributes are: "
                    f"{self.states[0].list_traceable_attributes()}"
                )
                continue

            # Plot angles in degrees and if required, unwrap to avoid large discontinuities
            if "angle" in p or "rate" in p or p == "s":
                if unwrap:
                    y = np.unwrap(y)
                y = np.degrees(y)

            ax.plot(x, y, label=label, c=color, linestyle=linestyle)

            # Plot the markers if given
            if plot_markers:
                marker_vals = y[np.isin(x, plot_markers)]
                ax.plot(
                    plot_markers,
                    marker_vals,
                    "s",
                    markerfacecolor="None",
                    label=marker_label,
                    color=marker_color,
                )

            # Label axes and set ticks
            try:
                y_lbl = y_labels[p] if p in y_labels.keys() else PLOT_LABELS[p]
            except KeyError:
                print(f"Label not specified for {p} and not in defaults.")
                y_lbl = p

            ax.set_ylabel(y_lbl)
            ax.ticklabel_format(useOffset=False)  # disable scientific notation offset
            # ax.set_xticks(np.arange(np.round(x[0]), x[-1], 5))  # x-tick every 5 seconds
            ax.grid(True, which="both")

        if legend:
            # If legend is given as a string, that'll be the location
            handles, labels = axes[0].get_legend_handles_labels()
            loc = legend if isinstance(legend, str) else None
            fig.legend(handles, labels, loc=loc, ncol=len(y_params))

        if remove_x_labels:
            for ax in axes[:-1]:
                ax.set_xticklabels([])

        axes[-1].set_xlabel(x_label)
        # axes[-1].set_xlim([0, None])

        return fig, axes

    def plot_variables_grid(
        self,
        variables=None,
        x_param: str = "s",
        axes=None,
        label: str = None,
        color: str = None,
        linestyle: str = None,
        y_labels: dict = None,
        y_scaling: dict = None,
    ):
        """Plot a set of variables against a common x on stacked axes.

        Args:
            variables: list of variable names to plot (default matches CST_curve).
            x_param: name of x-axis variable, default 's' (phase degrees in CST script).
            axes: optional pre-created list of axes to plot on; if None, create new.
            label, color, linestyle: styling applied to each trace.
            y_labels: optional mapping var->label; falls back to PLOT_LABELS.
            y_scaling: optional mapping var->scale factor applied to y data.

        Returns:
            (fig, axes): the figure and list of axes used.
        """
        if variables is None:
            variables = [
                "speed_tangential",
                "tension_tether_ground",
                "input_steering",
                "speed_radial",
            ]

        if y_scaling is None:
            y_scaling = {}
        if y_labels is None:
            y_labels = {}

        import matplotlib.pyplot as plt
        import numpy as np

        # Create axes if not provided
        if axes is None:
            fig, axes = plt.subplots(len(variables), 1, sharex=True)
            if not isinstance(axes, (list, tuple, np.ndarray)):
                axes = [axes]
        else:
            # infer figure from first axis
            fig = axes[0].figure

        # Prepare common x data (convert to degrees for phase variable)
        x = self.return_variable(x_param)
        # if x_param == "s":
        #     x = np.degrees(x)
        #     # Remove offset so x starts at zero
        #     if len(x) > 0:
        #         x = x - x[0]

        # Plot each variable
        for ax, var in zip(axes, variables):
            y = self.return_variable(var)
            scale = y_scaling.get(var, 1.0) if isinstance(y_scaling, dict) else 1.0
            if "angle" in var:
                y = np.degrees(y)
            ax.plot(x, y * scale, label=label, color=color, linestyle=linestyle)
            ax.set_ylabel(y_labels.get(var, PLOT_LABELS.get(var, var)))

        return fig, axes

    def plot_overview_3d(
        self,
        label: str = None,
        color: str = None,
        linestyle: str = None,
        variables=None,
        x_param: str = "s",
        axes: dict | None = None,
        coord: str = "cartesian",
        interactive: bool = True,
        pointer_color: str = "tab:red",
    ):
        """Plot this series on an overview with a single 3D left panel.

        Left: 3D trajectory in (azimuth [deg], elevation [deg], radial distance [m]).
        Right: stacked traces of variables vs x_param (default 's' in degrees, offset removed).

        Reuse axes by passing the dict {"left_3d": ax, "right_axes": [ax3,...]}.

        Parameters
        ----------
        interactive : bool, default False
            When True, adds a slider that moves a pointer along the 3D trajectory
            and vertical lines across the right-hand traces at the matching x_param.
        pointer_color : str, default 'tab:red'
            Color used for the moving pointer and vertical lines.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if variables is None:
            variables = [
                "speed_tangential",
                "tension_tether_ground",
                "input_steering",
                "speed_radial",
                "mechanical_power",
                "lift_coefficient",
                "drag_coefficient",
            ]

        created = False
        if axes is None:
            created = True
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(
                8, 3, width_ratios=[1, 0.25, 2], height_ratios=[1] * 8
            )
            ax_left3d = fig.add_subplot(gs[:, 0], projection="3d")
            ax3 = fig.add_subplot(gs[:2, 2])
            ax4 = fig.add_subplot(gs[2:4, 2])
            ax5 = fig.add_subplot(gs[4:6, 2])
            ax6 = fig.add_subplot(gs[6:, 2])
            right_axes = [ax3, ax4, ax5, ax6][: len(variables)]
        else:
            ax_left3d = axes.get("left_3d")
            right_axes = axes.get("right_axes", [])
            fig = (ax_left3d or (right_axes[0] if right_axes else None)).figure

        # Left 3D: plot trajectory

        phi_vals = self.return_variable("angle_azimuth")
        beta_vals = self.return_variable("angle_elevation")
        r_vals = self.return_variable("distance_radial")
        x_vals = r_vals * np.cos(beta_vals) * np.cos(phi_vals)
        y_vals = r_vals * np.cos(beta_vals) * np.sin(phi_vals)
        z_vals = r_vals * np.sin(beta_vals)
        x_lab = "x [m]"
        y_lab = "y [m]"
        z_lab = "z [m]"

        ax_left3d.plot(
            x_vals, y_vals, z_vals, label=label, color=color, linestyle=linestyle
        )
        if created:
            # Ground station at origin
            ax_left3d.scatter([0], [0], [0], marker="o", color="tab:brown")
        ax_left3d.set_xlabel(x_lab)
        ax_left3d.set_ylabel(y_lab)
        ax_left3d.set_zlabel(z_lab)

        # Right: traces (with tension scaled to kN)
        y_scaling = {"tension_tether_ground": 1 / 1000.0}
        self.plot_variables_grid(
            variables=variables,
            x_param=x_param,
            axes=right_axes,
            label=label,
            color=color,
            linestyle=linestyle,
            y_scaling=y_scaling,
        )

        # if created and right_axes:
        #     # Set x limits from 0 to last value
        #     x = self.return_variable(x_param)
        #     if x_param == "s":
        #         x = np.degrees(x)
        #         if len(x) > 0:
        #             x = x - x[0]
        #     if len(x) > 0:
        #         for ax in right_axes:
        #             ax.set_xlim(0, x[-1])
        slider = None
        if interactive:
            # Prepare trajectory arrays and x trace for vertical line placement
            x_trace = self.return_variable(x_param)
            if x_param == "s":
                x_trace = np.degrees(x_trace)
                if len(x_trace) > 0:
                    x_trace = x_trace - x_trace[0]

            # Safety: ensure lengths match for pointer movement
            n_pts = min(len(x_vals), len(y_vals), len(z_vals))
            if n_pts > 0:
                pointer3d = ax_left3d.plot(
                    [x_vals[0]],
                    [y_vals[0]],
                    [z_vals[0]],
                    marker="o",
                    color=pointer_color,
                    markersize=6,
                    zorder=5,
                )[0]
                vlines = []
                for ax in right_axes:
                    vlines.append(
                        ax.axvline(
                            x_trace[0] if len(x_trace) else 0.0,
                            color=pointer_color,
                            lw=1,
                            linestyle="-",
                        )
                    )

                # Allocate slider axis (avoid overlap) only once per figure
                slider_ax = fig.add_axes([0.12, 0.02, 0.7, 0.03])
                slider = Slider(
                    ax=slider_ax,
                    label="Index",
                    valmin=0,
                    valmax=n_pts - 1,
                    valinit=0,
                    valstep=1,
                )

                def _on_slide(idx):
                    i = int(idx)
                    # Update pointer position
                    pointer3d.set_data_3d([x_vals[i]], [y_vals[i]], [z_vals[i]])
                    # Update vertical lines
                    if len(x_trace) > i:
                        xval = x_trace[i]
                        for vl in vlines:
                            vl.set_xdata([xval, xval])
                    fig.canvas.draw_idle()

                slider.on_changed(_on_slide)

        return fig, {"left_3d": ax_left3d, "right_axes": right_axes}, slider

    def energy_metrics(self, phase_window_degrees: float = 360.0) -> dict:
        """Compute energy / power metrics for this time series over a phase window.

        This function no longer compares two TimeSeries objects. It extracts a
        single-cycle window in phase (defaults to 360 degrees) starting at the
        series' initial phase value and returns energy/power/timing statistics
        for that window.
        """
        import numpy as np

        # Helper to get masked arrays for one phase window
        def _mask_self(ts: "TimeSeries"):
            s_deg = np.degrees(ts.return_variable("s"))
            t = ts.return_variable("t")
            vtau = ts.return_variable("speed_tangential")
            tension = ts.return_variable("tension_tether_ground")
            vr = ts.return_variable("speed_radial")
            if len(s_deg) == 0:
                return slice(None), s_deg, t, vtau, tension, vr
            start = s_deg[0]
            mask = (s_deg > start) & (s_deg < start + phase_window_degrees)
            # Fallback if mask empty
            if not np.any(mask):
                mask = slice(None)
            return mask, s_deg, t, vtau, tension, vr

        mask, s_deg, t, vtau, tension, vr = _mask_self(self)

        s_m = s_deg[mask]
        t_m = t[mask]
        vtau_m = vtau[mask]
        tension_m = tension[mask]
        vr_m = vr[mask]

        # Energy and average power over window
        if len(t_m) > 1:
            dt = np.diff(t_m, prepend=t_m[0])
            energy = np.sum(tension_m * vr_m * dt)
            avg_power = energy / (t_m[-1] - t_m[0] + 1e-12)
        else:
            energy = 0.0
            avg_power = 0.0

        # Mean mechanical power (if available)
        try:
            pow_hist = self.return_variable("mechanical_power")[mask]
            mean_power = float(np.mean(pow_hist)) if len(pow_hist) else 0.0
        except Exception:
            mean_power = avg_power

        # Basic statistics (tension and vtau)
        tension_mean = float(np.mean(tension_m)) if len(tension_m) else 0.0
        tension_max = float(np.max(tension_m)) if len(tension_m) else 0.0
        tension_min = float(np.min(tension_m)) if len(tension_m) else 0.0

        vtau_mean = float(np.mean(vtau_m)) if len(vtau_m) else 0.0
        vtau_max = float(np.max(vtau_m)) if len(vtau_m) else 0.0
        vtau_min = float(np.min(vtau_m)) if len(vtau_m) else 0.0

        return {
            "energy": float(energy),
            "avg_power": float(avg_power),
            "mean_power": float(mean_power),
            "total_time": float(t_m[-1] - t_m[0]) if len(t_m) > 1 else 0.0,
            "tension_mean": float(tension_mean),
            "tension_max": float(tension_max),
            "tension_min": float(tension_min),
            "vtau_mean": float(vtau_mean),
            "vtau_max": float(vtau_max),
            "vtau_min": float(vtau_min),
            "phase_start_deg": float(s_m[0]) if len(s_m) else 0.0,
            "phase_end_deg": float(s_m[-1]) if len(s_m) else 0.0,
        }

    def plot_angles_scatter(
        self,
        ax=None,
        color_var: str = "speed_tangential",
        scatter_kwargs: dict = None,
    ):
        """Scatter of azimuth vs elevation colored by a variable.

        Args:
            ax: optional axis to plot on; creates a new one if None.
            color_var: variable name for color mapping.
            scatter_kwargs: dict with kwargs like s, cmap, vmin, vmax.

        Returns:
            PathCollection: matplotlib scatter artist.
        """

        if scatter_kwargs is None:
            scatter_kwargs = {}

        if ax is None:
            fig, ax = plt.subplots()

        az = np.degrees(self.return_variable("angle_azimuth"))
        el = np.degrees(self.return_variable("angle_elevation"))
        c = self.return_variable(color_var)

        sc = ax.scatter(az, el, c=c, **scatter_kwargs)
        ax.set_xlabel(PLOT_LABELS.get("angle_azimuth", "angle_azimuth"))
        ax.set_ylabel(PLOT_LABELS.get("angle_elevation", "angle_elevation"))
        return sc

    def plot_overview(
        self,
        label: str = None,
        color: str = None,
        linestyle: str = None,
        variables=None,
        x_param: str = "s",
        scatter_kwargs: dict = None,
        add_colorbar: bool = True,
        axes: dict | None = None,
    ):
        """Plot this series on the standard overview figure.

        Layout:
        - Left column: two panels (dynamic top, quasi-steady bottom) with azimuth vs elevation,
          colored by speed_tangential.
        - Right column: stacked traces vs phase `s` (or chosen x).

        Call this once to create the figure and again with axes to overlay another phase.

        Args:
            label, color, linestyle: styling for this series.
            variables: list of variables to plot on the right column (defaults to CST layout).
            x_param: x-axis variable for traces (default 's').
            scatter_kwargs: dict of kwargs for the scatter plot.
            add_colorbar: whether to add a colorbar (only applies when creating axes).
            axes: dict of axes to reuse: {"left_dynamic": ax, "left_qs": ax, "right_axes": [ax3,...]}.

        Returns:
            (fig, axes_dict, scatter): figure, axes mapping, and the scatter artist for this series.
        """

        if variables is None:
            variables = [
                "speed_tangential",
                "tension_tether_ground",
                "input_steering",
                "speed_radial",
            ]

        if scatter_kwargs is None:
            scatter_kwargs = {}

        created = False
        if axes is None:
            created = True
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(
                8, 3, width_ratios=[1, 0.25, 2], height_ratios=[1] * 8
            )
            ax_dyn = fig.add_subplot(gs[:4, 0])
            ax_qs = fig.add_subplot(gs[4:, 0])
            ax3 = fig.add_subplot(gs[:2, 2])
            ax4 = fig.add_subplot(gs[2:4, 2])
            ax5 = fig.add_subplot(gs[4:6, 2])
            ax6 = fig.add_subplot(gs[6:, 2])
            right_axes = [ax3, ax4, ax5, ax6][: len(variables)]
        else:
            ax_dyn = axes.get("left_dynamic")
            ax_qs = axes.get("left_qs")
            right_axes = axes.get("right_axes", [])
            # infer figure from any provided axis
            fig = (ax_dyn or ax_qs or (right_axes[0] if right_axes else None)).figure

        # Choose top/bottom axis based on quasi_steady flag
        is_qs = getattr(self, "quasi_steady", False)
        ax_left = ax_qs if is_qs else ax_dyn
        last_scatter = self.plot_angles_scatter(
            ax=ax_left,
            color_var="speed_tangential",
            scatter_kwargs=scatter_kwargs,
        )
        # Tension scaling to kN for readability
        y_scaling = {"tension_tether_ground": 1 / 1000.0}
        self.plot_variables_grid(
            variables=variables,
            x_param=x_param,
            axes=right_axes,
            label=label,
            color=color,
            linestyle=linestyle,
            y_scaling=y_scaling,
        )

        # Formatting similar to CST script
        if created:
            for ax in [ax_dyn, ax_qs]:
                ax.set_ylim(0, 50)
                ax.set_xlim(-50, 50)
                ax.legend(loc="lower right", fontsize=9)

        for ax in right_axes:
            if x_param == "s":
                ax.set_xlim(0, 360)
        # Left labels and annotations
        if created:
            ax_dyn.text(
                0.95,
                0.95,
                "Dynamic",
                transform=ax_dyn.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                weight="bold",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            )
            ax_qs.text(
                0.95,
                0.95,
                "Quasi-Steady",
                transform=ax_qs.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                weight="bold",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            )

        # Y labels for right axes
        for ax, var in zip(right_axes, variables):
            ax.set_ylabel(PLOT_LABELS.get(var, var))
        if right_axes:
            right_axes[-1].set_xlabel(PLOT_LABELS.get(x_param, x_param))

        # Optional colorbar next to the left panels (only when creating new axes)
        if created and add_colorbar and last_scatter is not None:
            cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])
            cbar = fig.colorbar(last_scatter, cax=cbar_ax)
            cbar.set_label(PLOT_LABELS.get("speed_tangential", "speed_tangential"))

        return (
            fig,
            {"left_dynamic": ax_dyn, "left_qs": ax_qs, "right_axes": right_axes},
            last_scatter,
        )

    def interactive_plot(
        self,
        parameters: list = None,
        plot_vectors: dict = None,
        vector_directions: list = None,
        vector_scaling: dict = None,
        animate=False,
        y_labels=None,
        gradient_color: tuple = None,
    ):
        """Interactive plot. To make the slider work, you must keep a reference to the figure and slider in the
        __main__ thread. I.e. you must call this method like: fig, slider = timeseries.interactive_plot()
        """

        if parameters is None:
            parameters = PLOT_PARAMETERS
        if plot_vectors is None:
            plot_vectors = {}
        if y_labels is None:
            y_labels = {}

        if vector_directions is None:
            vector_directions = np.ones((3, 1))

        if vector_scaling is None:
            vector_scaling = {v: 0.05 for v in plot_vectors}

        if len(parameters) < 4:
            raise ValueError(f"Interactive plot needs at least 4 parameters to plot")

        try:
            t = self.return_variable("t")
            dt = t[1] - t[0]
            fps = 1 / dt
            print(
                f"Frame rate determined to be {fps} Hz."
            )  # 24 fps is standard for movies
        except Exception as e:
            print("Could not determine frame rate to animate, using default value.")
            fps = 24

        # Grid is such that the 3d plot spans half of the timeplots, 2d plot the other half and the slider as well.
        halfway_point = int(np.ceil(len(parameters) / 2))
        grid = (
            [["3d", p] for p in parameters[:halfway_point]]
            + [["2d", p] for p in parameters[halfway_point:-1]]
            + [["slider", parameters[-1]]]
        )

        # Create figs + axes
        fig, axs = plt.subplot_mosaic(
            grid,
            # figsize=(screen_width/100, screen_height/100),  # 100 dpi
            figsize=(15, 7),
            per_subplot_kw={"3d": {"projection": "3d"}},
            gridspec_kw={"width_ratios": [1, 2], "wspace": 0.3, "hspace": 0.2},
        )

        # Fig size + ax size
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        # axs['3d'].set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))

        # Plot 3d plot, 2d plot, and time plots in the right axes.
        param_axs = [ax for k, ax in axs.items() if k not in ["3d", "2d", "slider"]]
        self.plot_trace_on_plane(
            ax=axs["2d"], gradient_color=gradient_color, plot_kwargs={"legend": True}
        )
        self.trajectory_plot3d(ax=axs["3d"], gradient_color=gradient_color)
        self.plot_traces(
            y_params=parameters,
            axes=param_axs,
            y_labels=y_labels,
            plot_kwargs={"unwrap": True, "remove_x_labels": True},
        )

        x = self.return_variable("x")
        y = self.return_variable("y")
        z = self.return_variable("z")
        t = self.return_variable("t")
        extract_state = {}
        for p in parameters:
            extract_state[p] = self.return_variable(p)
        # Plot time markers
        state = self.states[0]
        markers = {}
        for p in parameters:
            value = (
                extract_state[p][0]
                if "angle" not in p
                else np.degrees(extract_state[p][0])
            )
            markers[p] = axs[p].plot(
                state["t"], value, color="tab:red", linewidth=2, marker="o"
            )[0]

        markers["3d"] = axs["3d"].plot(
            [x[0]], [y[0]], [z[0]], color="tab:red", marker="o", linewidth=1
        )[0]
        markers["2d"] = axs["2d"].plot(
            x[0], z[0], color="tab:red", marker="o", linewidth=1
        )[0]

        # Plot vectors
        vectors = {}

        # Create slider, and cache time vector
        self.__cached_time = [round(s["t"], 3) for s in self.states]

        time_slider = Slider(
            ax=axs["slider"],
            label="Time [s]",
            # valmin = self.states[0]["t"],
            # valmax=self.states[-1]["t"],
            valstep=self.__cached_time,
            valmin=self.__cached_time[0],
            valmax=self.__cached_time[-1],
            valinit=self.__cached_time[0],
        )

        # The function to be called anytime a slider's value changes
        def update(time):
            index = self.__cached_time.index(round(time, 3))

            # Update markers
            for p in parameters:
                val = (
                    extract_state[p][index]
                    if "angle" not in p
                    else np.degrees(extract_state[p][index])
                )
                markers[p].set_data([time], [val])

            markers["3d"].set_data_3d([x[index]], [y[index]], [z[index]])
            markers["2d"].set_data([x[index]], [z[index]])

            # For each vector, plot components required directions
            for v, c in plot_vectors.items():
                for i, d in enumerate(vector_directions):
                    # Vector component in Cartesian coordinates:
                    vec = np.matmul(
                        state.transformation_C_from_W.T,
                        getattr(state, v) * d * vector_scaling[v],
                    )
                    vec_length = np.linalg.norm(vec)
                    try:
                        vectors[v + str(i)].remove()
                    except KeyError:
                        pass
                    vectors[v + str(i)] = axs["3d"].quiver(
                        *state.position_W,
                        *vec,
                        length=vec_length,
                        color=c,
                        arrow_length_ratio=0.2,
                    )

            fig.canvas.draw_idle()

        time_slider.on_changed(update)

        if animate:
            # Rotate the axes and update plot.
            def init():
                axs["3d"].view_init(30, -40)
                return [fig]

            def animate(i):
                axs["3d"].view_init(30, -40 + i / 5)
                time_slider.set_val(self.__cached_time[i])
                return [fig]

            anim = animation.FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=len(self.__cached_time),
                interval=1,
                blit=True,
            )

            writergif = animation.PillowWriter(fps=fps)
            anim.save("interactive_plot.gif", writer=writergif)

        # mng = plt.get_current_fig_manager()
        # mng.window.state('zoomed')
        # mng.resize(screen_width, screen_height)
        return fig, time_slider
