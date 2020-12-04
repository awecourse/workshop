"""Small earth plots to get more insight into kinematics of kite."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


def calc_proj(elevation_angle, azimuth_angle):
    elevation_angle = elevation_angle
    y = np.cos(elevation_angle)*np.sin(azimuth_angle)
    z = np.sin(elevation_angle)
    z = z - .3*np.cos(azimuth_angle)*np.cos(elevation_angle)  # offset for 3d effect
    return y, z


def plot_small_earth(ax):
    azimuth_lines = np.linspace(-90, 90, 13)*np.pi/180.
    elevation_lines = np.linspace(0, 90, 7)*np.pi/180.

    ax.axis('equal')
    ax.set_xticks([], minor=[])
    ax.set_yticks([], minor=[])
    ax.set_ylim([-.55, 1.1])

    for beta in elevation_lines:
        phis = np.linspace(azimuth_lines[0], azimuth_lines[-1], 181)
        y, z = calc_proj(beta, phis)
        if beta != np.pi/2.:
            ax.text(y[0]-.1, z[0], "{:.0f}".format(beta*180./np.pi), horizontalalignment='center')
        if beta == 30.*np.pi/180.:
            ax.text(y[0]-.2, z[0]+.2, r"$\beta$ [$^\circ$]", horizontalalignment='center')
        ax.plot(y, z, 'grey', linewidth=.5)

    for phi in azimuth_lines:
        betas = np.linspace(elevation_lines[0], elevation_lines[-1], 91)
        y, z = calc_proj(betas, phi)
        if -np.pi/2. < phi < np.pi/2.:
            ax.text(y[0], z[0]-.1, "{:.0f}".format(phi*180./np.pi), horizontalalignment='center')
        if phi == 0.:
            ax.text(0., z[0]-.2, r"$\phi$ [$^\circ$]", horizontalalignment='center')
        ax.plot(y, z, 'grey', linewidth=.5)


def corrected_orientation_angle(beta, phi, yaw):
    y0, z0 = calc_proj(beta, phi)
    step_size = 1e-6
    dbeta = -np.cos(yaw) * step_size
    dphi = np.sin(yaw) * step_size
    y1, z1 = calc_proj(beta+dbeta, phi+dphi)
    dy = (y1 - y0)/step_size
    dz = (z1 - z0)/step_size
    yaw_proj = np.arctan2(dy, -dz)
    return dy, dz, yaw_proj


def plot_kite(ax, beta, phi, psi=np.pi):
    y, z = calc_proj(beta, phi)
    psi_proj = corrected_orientation_angle(beta, phi, psi)[2]
    t = MarkerStyle(marker=7)
    t._transform = t.get_transform().rotate_deg(psi_proj*180./np.pi)
    marker_obj = ax.plot(y, z, 's', marker=t, ms=20, mfc='None', mew=2, mec='k')[0]


def plot_vector(ax, beta, phi, vector_orientation, vector_magnitude=.2, clr='C1', lbl='Course', fmt='-'):
    # Determines plotting orientation such that it aligns with the small earth grid, i.e.: a vector that points towards
    # zenith is not plotted vertically, but tangent to azimuth grid lines.
    y0, z0 = calc_proj(beta, phi)
    dy, dz = corrected_orientation_angle(beta, phi, vector_orientation)[:2]
    le = vector_magnitude  # vector length
    ax.plot([y0, y0+dy*le], [z0, z0+dz*le], fmt, color=clr, label=lbl)


if __name__ == '__main__':
    plt.figure(figsize=[6.4, 4])
    plt.subplots_adjust(top=0.95, bottom=0.06, left=0.11, right=0.9, hspace=0.1, wspace=0.2)
    plot_small_earth(plt.gca())
    beta, phi, psi = 0.*np.pi/180., 15.*np.pi/180., 30.*np.pi/180.
    chi = 30.*np.pi/180.
    plot_kite(plt.gca(), beta, phi, psi)
    plot_vector(plt.gca(), beta, phi, chi)

    plt.show()