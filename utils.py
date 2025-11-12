# -*- coding: utf-8 -*-
"""Utility functions."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from numpy import all, diff, array

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def flatten_dict(input_dict, parent_key="", sep="."):
    """Recursive function to convert multi-level dictionary to flat dictionary.

    Args:
        input_dict (dict): Dictionary to be flattened.
        parent_key (str): Key under which `input_dict` is stored in the higher-level dictionary.
        sep (str): Separator used for joining together the keys pointing to the lower-level object.

    """
    items = []  # list for gathering resulting key, value pairs
    for k, v in input_dict.items():
        new_key = (
            parent_key + sep + k.replace(" ", "") if parent_key else k.replace(" ", "")
        )
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def zip_el(*args):
    """ "Zip iterables, only if input lists have same length.

    Args:
        *args: Variable number of lists.

    Returns:
        list: Iterator that aggregates elements from each of the input lists.

    Raises:
        AssertError: If input lists do not have the same length.

    """
    lengths = [len(l) for l in [*args]]
    assert all(diff(lengths) == 0), "All the input lists should have the same length."
    return zip(*args)


def plot_traces(
    x,
    data_sources,
    source_labels,
    plot_parameters,
    y_labels=None,
    y_scaling=None,
    fig_num=None,
    plot_kwargs={},
    plot_markers=None,
    x_label="Time [s]",
):
    """Plot the time trace of a parameter from multiple sources.

    Args:
        x (tuple): Sequence of points along x.
        data_sources (tuple): Sequence of time traces of the different data sources.
        source_labels (tuple): Labels corresponding to the data sources.
        plot_parameters (tuple): Sequence of attributes/keys of the objects/dictionaries of the time traces.
        y_labels (tuple, optional): Y-axis labels corresponding to `plot_parameters`.
        y_scaling (tuple, optional): Scaling factors corresponding to `plot_parameters`.
        fig_num (int, optional): Number of figure used for the plot, if None a new figure is created.
        plot_kwargs (dict, optional): Line plot keyword arguments.

    """
    if y_labels is None:
        y_labels = plot_parameters
    if y_scaling is None:
        y_scaling = [None for _ in range(len(plot_parameters))]
    if fig_num:
        axes = plt.figure(fig_num).get_axes()
    else:
        axes = []
    if not axes:
        _, axes = plt.subplots(len(plot_parameters), 1, sharex=True, num=fig_num)
    if len(axes) == 1:
        axes = (axes,)

    for p, y_lbl, f, ax in zip_el(plot_parameters, y_labels, y_scaling, axes):
        for trace, s_lbl in zip_el(data_sources, source_labels):
            y = None
            # TODO: see if it is a better option to make this function a method and check if p is an attribute as condition
            if p == s_lbl:
                y = trace
            elif isinstance(trace[0], dict):
                if p in trace[0]:
                    y = [item[p] for item in trace]
            elif hasattr(trace[0], p):
                y = [getattr(item, p) for item in trace]
            if y:
                if f:
                    y = array(y) * f
                ax.plot(x, y, label=s_lbl, **plot_kwargs)
                if plot_markers:
                    marker_vals = [y[x.index(t)] for t in plot_markers]
                    ax.plot(plot_markers, marker_vals, "s", markerfacecolor="None")
        ax.set_ylabel(y_lbl)
        ax.grid(True)
        # ax.legend()
    axes[-1].set_xlabel(x_label)
    axes[-1].set_xlim([0, None])


# def subplots_from_dicts(x, time_traces, plot_keys=None, mark_points=None):
#     if plot_keys is None:
#         plot_keys = list(time_traces[0])
#
#     fig, ax = plt.subplots(len(plot_keys), 1, sharex=True)
#     if len(plot_keys) == 1:
#         ax = (ax,)
#     for i, param in enumerate(plot_keys):
#         source_labels = [src[0] for src in time_traces[0][param]]
#         for j, lbl in enumerate(source_labels):
#             # if 'angle' in param:
#             #     y = [d[param]*180/pi for d in plot_dicts]
#             # else:
#             y = [point[param][j][1] for point in time_traces]
#             ax[i].plot(x, y, label=lbl)
#             if mark_points:
#                 x_mark, y_mark = zip(*[(a, b) for a, b in zip_el(x, y) if a in mark_points])
#                 ax[i].plot(x_mark, y_mark, 's', markerfacecolor='None')
#         ax[i].set_ylabel(param)
#         ax[i].grid()
#         ax[i].legend()
#     ax[-1].set_xlabel('Time [s]')
#
#
# def simple_integration(y, x):
#     integral = 0
#     for x0, x1, y1 in zip_el(x[:-1], x[1:], y[1:]):
#         integral += y1*(x1-x0)
#     return integral

import casadi as ca


def transformation_AZR_from_W(azimuth, elevation):
    phi = azimuth
    beta = elevation
    # Create the transformation matrix
    transformation = ca.vertcat(
        ca.horzcat(-ca.sin(phi), ca.cos(phi), 0),
        ca.horzcat(
            -ca.sin(beta) * ca.cos(phi), -ca.sin(beta) * ca.sin(phi), ca.cos(beta)
        ),
        ca.horzcat(
            ca.cos(beta) * ca.cos(phi), ca.cos(beta) * ca.sin(phi), ca.sin(beta)
        ),
    )
    return transformation


def transformation_C_from_AZR(chi):
    # Directly create the transformation matrix using CasADi
    transformation = ca.vertcat(
        ca.horzcat(ca.sin(chi), ca.cos(chi), 0),
        ca.horzcat(-ca.cos(chi), ca.sin(chi), 0),
        ca.horzcat(0, 0, 1),
    )
    return transformation


def transformation_C_from_A(theta_a, chi_a, roll):
    # Define the Pitch matrix
    Pitch = ca.vertcat(
        ca.horzcat(ca.cos(theta_a), 0, ca.sin(theta_a)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(theta_a), 0, ca.cos(theta_a)),
    )

    # Define the Yaw matrix
    Yaw = ca.vertcat(
        ca.horzcat(ca.cos(chi_a), -ca.sin(chi_a), 0),
        ca.horzcat(ca.sin(chi_a), ca.cos(chi_a), 0),
        ca.horzcat(0, 0, 1),
    )

    # Define the Roll matrix
    Roll = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
        ca.horzcat(0, ca.sin(roll), ca.cos(roll)),
    )

    # Compute the transformation matrix T using the @ operator
    T = Yaw @ Pitch @ Roll

    return T


def transformation_C_from_K(pitch, roll, yaw=0):

    # Define the Pitch matrix
    Pitch = ca.vertcat(
        ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch)),
    )

    # Define the Yaw matrix
    Yaw = ca.vertcat(
        ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
        ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
        ca.horzcat(0, 0, 1),
    )

    # Define the Roll matrix
    Roll = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
        ca.horzcat(0, ca.sin(roll), ca.cos(roll)),
    )

    T = Yaw @ Pitch @ Roll
    return T


def transformation_C_from_W(azimuth, elevation, course):
    # Create the transformation matrix
    return transformation_C_from_AZR(course) @ transformation_AZR_from_W(
        azimuth, elevation
    )


DATA_ROOT = Path(__file__).resolve().parents[0]


def load_aero_input(path: Optional[Path] = None) -> dict:
    if path is None:
        path = DATA_ROOT / "v3_aero_input.json"
    with open(path, "r") as f:
        return json.load(f)
