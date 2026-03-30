from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scienceplots
import os
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

def apply_science_style():
    """Apply the preferred matplotlib style."""
    plt.style.use(['science', 'grid'])

def apply_science_style_nogrid():
    plt.style.use(['science'])

CONTOURPLOT_LEVELS_SLOW = 40
CONTOURPLOT_LEVELS_FAST = 20
FIGSIZE_AREA = 64
FIGSIZE_DIMENSIONS_SQUARE = (np.sqrt(FIGSIZE_AREA), np.sqrt(FIGSIZE_AREA))
FIGSIZE_DIMENSIONS_RECTANGLE = ((4/3)*np.sqrt((3/4)*FIGSIZE_AREA), np.sqrt((3/4)*FIGSIZE_AREA))
STATIC_PLOT_ROW_MAX_1D = 2
STATIC_PLOT_ROW_MAX_2D = 4
STATIC_PLOT_NUM = 16

def plot_ellipse(mean_vector, covariance_matrix, ax=None, n_std=2, **kwargs):
    """
    Plots an ellipse representing the covariance matrix around a mean vector.

    Parameters:
    mean_vector : (2,) array-like
        The mean position of the 2D point.
    covariance_matrix : (2,2) array-like
        Covariance matrix defining the ellipse shape.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (creates a new one if None).
    n_std : float, optional
        Number of standard deviations to scale the ellipse (default: 2).
    kwargs : dict
        Additional keyword arguments for the `Ellipse` object.
    """
    assert len(mean_vector) == 2, "Mean vector must be 2-dimensional"

    if ax is None:
        fig, ax = plt.subplots()

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse = Ellipse(xy=mean_vector, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

def plot_animate_field_1D(field, 
                          x_coordinates, 
                          time_range,
                          save_path=None,
                          file_name="animation",
                          file_format="mp4", 
                          fig_dimensions=FIGSIZE_DIMENSIONS_RECTANGLE, 
                          ylim=None,
                          interval=5,
                          **kwargs):
    """
    Creates and saves an animated line plot for a 1D field over time.

    Parameters:
    - field: 2D array of shape (time_steps, num_points), field values.
    - x_coordinates: 1D array of shape (num_points,), x-coordinates.
    - time_range: 1D array of shape (time_steps,), time steps for animation.
    - save_path: str, directory where animation should be saved.
    - file_name: str, name of the output file (without extension).
    - file_format: str, 'mp4' (default) or 'gif'.
    - fig_dimensions: tuple, figure size (width, height).
    - ylim: tuple, y-axis limits (ymin, ymax).
    """
    vmin, vmax = field.min(), field.max()  # Get global min and max
    if ylim is None:
        vdiff = np.abs(vmin - vmax)
        ylim = (vmin - 0.025*vdiff, vmax + 0.025*vdiff)
    
    fig, ax = plt.subplots(figsize=fig_dimensions)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("Field Value")
    line, = ax.plot([], [], "-o", **kwargs)
    
    def update(frame):
        ax.clear()
        ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("Field Value")
        ax.set_title(f"Time: {time_range[frame]:.3f}")
        ax.plot(x_coordinates, field[frame, :], **kwargs)
        
    ani = FuncAnimation(fig, update, frames=len(time_range), interval=interval, blit=False)
    
    plt.close(fig)
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_extension = file_format.lower()
        full_path = os.path.join(save_path, f"{file_name}.{file_extension}")

        if file_extension == "mp4":
            writer = FFMpegWriter(fps=30)
            ani.save(full_path, writer=writer)
        elif file_extension == "gif":
            writer = PillowWriter(fps=30)
            ani.save(full_path, writer=writer)
        else:
            raise ValueError("Unsupported file format. Use 'mp4' or 'gif'.")

        print(f"Animation saved to: {full_path}")
    else:
        plt.show()

def plot_animate_field_2D(field, 
                          basis_coordinates, 
                          time_range,
                          save_path=None,
                          file_name="animation",
                          file_format="mp4", 
                          fig_dimensions=(6, 6), 
                          c_levels=20, 
                          cov=False,
                          **kwargs):
    """
    Creates and saves an animated contour plot for a 2D field over time.

    Parameters:
    - field: 2D array of shape (time_steps, num_points), field values.
    - basis_coordinates: 2D array of shape (num_points, 2), (x, y) coordinates.
    - time_range: 1D array of shape (time_steps,), time steps for animation.
    - save_path: str, directory where animation should be saved.
    - file_name: str, name of the output file (without extension).
    - file_format: str, 'mp4' (default) or 'gif'.
    - fig_dimensions: tuple, figure size (width, height).
    - c_levels: int, number of contour levels.
    """
    if cov:
        field = np.einsum("ijj->ij", field)

    vmin, vmax = field.min(), field.max()  # Get global min and max
    
    fig, ax = plt.subplots(figsize=fig_dimensions)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)  # Adjust cmap as needed
    sm.set_array([])  # Needed for the colorbar

    # Add colorbar at the bottom
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label("Field Value")

    def update(frame):
        ax.clear()
        ax.set_title(f"Time: {time_range[frame]:.3f}")
        contour = ax.tricontourf(basis_coordinates[:, 0], basis_coordinates[:, 1], 
                                 field[frame, :], levels=c_levels, cmap="viridis", 
                                 norm=norm, **kwargs)
        return contour.collections  # Ensure animation updates correctly

    ani = FuncAnimation(fig, update, frames=len(time_range), interval=5, blit=False)
    
    plt.close(fig)
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_extension = file_format.lower()
        full_path = os.path.join(save_path, f"{file_name}.{file_extension}")

        if file_extension == "mp4":
            writer = FFMpegWriter(fps=30)
            ani.save(full_path, writer=writer)
        elif file_extension == "gif":
            writer = PillowWriter(fps=30)
            ani.save(full_path, writer=writer)
        else:
            raise ValueError("Unsupported file format. Use 'mp4' or 'gif'.")

        print(f"Animation saved to: {full_path}")
    else:
        plt.show()

def plot_field_2D(field, basis_coordinates, fig_dimensions = FIGSIZE_DIMENSIONS_SQUARE, scatter=False, c_levels=CONTOURPLOT_LEVELS_SLOW):
    fig, axes = plt.subplots(figsize=fig_dimensions)
    if scatter:
        scatter = axes.scatter(basis_coordinates[:,0], basis_coordinates[:,1], c=field)
        plt.colorbar(scatter)
    else:
        surf = axes.tricontourf(basis_coordinates[:, 0], basis_coordinates[:, 1], field, levels=c_levels)
        plt.colorbar(surf)
    plt.show()

def plot_field_1D(field, basis_coordinates, fig_dimensions = FIGSIZE_DIMENSIONS_RECTANGLE):
    fig, axes = plt.subplots(figsize = fig_dimensions)
    line = axes.plot(basis_coordinates, field)
    plt.show()

def plot_snapshot_field_1D(field, 
                           basis_coordinates, 
                           time_range, 
                           num_plots=STATIC_PLOT_NUM, 
                           fig_area=FIGSIZE_AREA,
                           autoscale=True,
                           scatter=False,
                           **kwargs):
    """
    Plots multiple snapshots of a 1D field at selected time points.
    """
    # if time_range is less than num_plots, rescale
    if len(time_range) < STATIC_PLOT_NUM:
        num_plots = len(time_range)

    n_rows = (num_plots + STATIC_PLOT_ROW_MAX_1D - 1) // STATIC_PLOT_ROW_MAX_1D if num_plots > STATIC_PLOT_ROW_MAX_1D else 1
    n_cols = min(num_plots, STATIC_PLOT_ROW_MAX_1D)

    x_figsize = np.sqrt(fig_area) * np.sqrt(n_cols / n_rows) * np.sqrt(4/3)
    y_figsize = (3/4)*(fig_area / x_figsize)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(x_figsize, y_figsize))
    vmin, vmax = field.min(), field.max()
    vdiff = np.abs(vmin - vmax)
    ylim = (vmin - 0.025*vdiff, vmax + 0.025*vdiff)
    indices = np.linspace(0, len(time_range)-1, num_plots, dtype=int)
    labels = [fr"$t={time_range[i]:.4f}$" for i in indices]

    axes = np.array(axes).flatten()
    line_plots = []

    for i, ax in enumerate(axes[:num_plots]):
        if scatter:
            line = ax.scatter(basis_coordinates, field[indices[i],:], **kwargs)
            line_plots.append(line)
        else:
            line = ax.plot(basis_coordinates, field[indices[i],:], **kwargs)
            line_plots.append(line)
        ax.set_title(labels[i])

        if autoscale:
            ax.set_ylim(*ylim)
            if i < (n_rows - 1) * n_cols:
                ax.set_xticklabels([])
            if i % n_cols != 0:
                ax.set_yticklabels([])

    if autoscale:
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

    fig.tight_layout()
    # plt.subplots_adjust(bottom=0.25)
    plt.show()

def plot_snapshot_field_2D(field, 
                           basis_coordinates, 
                           time_range, 
                           num_plots=STATIC_PLOT_NUM, 
                           scatter = False, 
                           fig_area=FIGSIZE_AREA, 
                           c_levels=CONTOURPLOT_LEVELS_SLOW,
                           cov = False,
                           lims = None,
                           **kwargs):
    """
    Plots multiple snapshots of a 2D field at selected time points.
    """
    # if time_range is less than num_plots, rescale
    if len(time_range) < STATIC_PLOT_NUM:
        num_plots = len(time_range)

    n_rows = (num_plots + STATIC_PLOT_ROW_MAX_2D - 1) // STATIC_PLOT_ROW_MAX_2D if num_plots > STATIC_PLOT_ROW_MAX_2D else 1
    n_cols = min(num_plots, STATIC_PLOT_ROW_MAX_2D)

    x_figsize = np.sqrt(fig_area) * np.sqrt(n_cols / n_rows)
    y_figsize = fig_area / x_figsize

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(x_figsize, y_figsize))
    indices = np.linspace(0, len(time_range)-1, num_plots, dtype=int)
    labels = [fr"$t={time_range[i]:.4f}$" for i in indices]

    axes = np.array(axes).flatten()
    vmin, vmax = field.min(), field.max()
    contour_plots = []
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    if cov:
        field = np.einsum("ijj->ij", field)

    for i, ax in enumerate(axes[:num_plots]):
        if scatter:
            cont = ax.scatter(basis_coordinates[:, 0], basis_coordinates[:, 1], c=field[indices[i], :], cmap=cmap, norm=norm, **kwargs)
        else:
            cont = ax.tricontourf(basis_coordinates[:, 0], basis_coordinates[:, 1], field[indices[i], :], levels=c_levels, vmin=vmin, vmax=vmax, norm=norm, **kwargs)
        contour_plots.append(cont)
        ax.set_title(labels[i])
        if lims:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])

        if i < (n_rows - 1) * n_cols:
            ax.set_xticklabels([])
        if i % n_cols != 0:
            ax.set_yticklabels([])

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal').ax.set_xlabel("Field Value")

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()

def plot_3D_field_2D(field, basis_coordinates, time_step, fig_dimensions=FIGSIZE_DIMENSIONS_SQUARE, **kwargs):
    """
    Plots a 3D surface plot of a 2D field at a specific time step.
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=fig_dimensions)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(basis_coordinates[:, 0], basis_coordinates[:, 1], field[time_step, :], cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

# def plot_time_series_with_error_bars(
#     time_series_list, 
#     time_series_variances_list=None, 
#     time_range=None, 
#     alpha=0.2, 
#     title=None, 
#     labels=None,
#     force_zero=False, 
#     fig_dimensions=FIGSIZE_DIMENSIONS_SQUARE,
#     bars=None,
#     floors=None):

#     if time_range is None:
#         raise ValueError('No time-range specified.')
    
#     if not isinstance(time_series_list, list):
#         time_series_list = [time_series_list]
    
#     if time_series_variances_list is not None:
#         if not isinstance(time_series_variances_list, list):
#             time_series_variances_list = [time_series_variances_list]
#         if len(time_series_list) != len(time_series_variances_list):
#             raise ValueError("Number of time series must match number of variance inputs.")
    
#     num_series = len(time_series_list)
#     colors = cm.viridis(np.linspace(0, 1, num_series))
    
#     fig, ax = plt.subplots(figsize=fig_dimensions)

#     ts_max = np.max(time_series_list)
#     ts_min = np.min(time_series_list)
#     ts_range = ts_max - ts_min
#     ts_bonus = 0.05*ts_range
#     ts_bot = ts_min - ts_bonus
#     ts_top = ts_max + ts_bonus


#     xs_max = np.max(time_range)
#     xs_min = np.min(time_range)
#     xs_range = xs_max - xs_min
#     xs_bonus = 0.05*xs_range
#     xs_bot = xs_min - xs_bonus
#     xs_top = xs_max + xs_bonus

#     for i, (ts, color) in enumerate(zip(time_series_list, colors)):
#         means = ts.flatten()
#         ax.plot(time_range, means, color=color, label=labels[i] if labels else None)
        
#         if time_series_variances_list is not None:
#             stds = np.sqrt(time_series_variances_list[i].flatten())
#             if force_zero:
#                 l_bound = (means - 2 * stds)
#                 l_bound[l_bound < 0] = 0
#                 ax.fill_between(time_range, l_bound, means + 2 * stds, color=color, alpha=alpha)
#             else:
#                 ax.fill_between(time_range, means - 2 * stds, means + 2 * stds, color=color, alpha=alpha)
#     if title:
#         ax.set_title(title)
#     if bars is not None:
#         ax.vlines(bars, ts_bot, ts_top, colors='black', label='Measurements')
#     if floors is not None:
#         ax.hlines(floors, xs_bot, xs_top, colors='orange', label='Truth', linestyle='dashed')
#     if labels:
#         ax.legend()

    
#     plt.show()


def plot_time_series_with_error_bars(
    time_series_list,
    time_series_variances_list=None,
    time_range=None,
    alpha=0.2,
    title=None,
    labels=None,
    force_zero=False,
    fig_dimensions=FIGSIZE_DIMENSIONS_SQUARE,
    bars=None,
    floors=None,
    # New argument to accept a list of hatch styles
    hatch_patterns=None):

    if time_range is None:
        raise ValueError('No time-range specified.')

    if not isinstance(time_series_list, list):
        time_series_list = [time_series_list]

    if time_series_variances_list is not None:
        if not isinstance(time_series_variances_list, list):
            time_series_variances_list = [time_series_variances_list]
        if len(time_series_list) != len(time_series_variances_list):
            raise ValueError("Number of time series must match number of variance inputs.")

    # New: Define a default set of hatch patterns to cycle through
    if hatch_patterns is None:
        hatch_patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    num_series = len(time_series_list)
    colors = cm.viridis(np.linspace(0, 1, num_series))

    fig, ax = plt.subplots(figsize=fig_dimensions)

    # This section for setting plot limits can be simplified
    # You can let matplotlib auto-scale or set it after plotting
    # For now, we will leave it as is.
    ts_max = np.max(time_series_list)
    ts_min = np.min(time_series_list)
    ts_range = ts_max - ts_min
    ts_bonus = 0.05*ts_range
    ts_bot = ts_min - ts_bonus
    ts_top = ts_max + ts_bonus

    xs_max = np.max(time_range)
    xs_min = np.min(time_range)
    xs_range = xs_max - xs_min
    xs_bonus = 0.05*xs_range
    xs_bot = xs_min - xs_bonus
    xs_top = xs_max + xs_bonus

    for i, (ts, color) in enumerate(zip(time_series_list, colors)):
        means = ts.flatten()
        ax.plot(time_range, means, color=color, label=labels[i] if labels else f'Series {i+1}')

        if time_series_variances_list is not None:
            stds = np.sqrt(time_series_variances_list[i].flatten())
            # New: Select a hatch pattern for the current series
            current_hatch = hatch_patterns[i % len(hatch_patterns)]

            if force_zero:
                l_bound = (means - 2 * stds)
                l_bound[l_bound < 0] = 0
                # Modified: Added hatch and edgecolor parameters
                ax.fill_between(time_range, l_bound, means + 2 * stds,
                                color=color, alpha=alpha,
                                hatch=current_hatch, edgecolor=color)
            else:
                # Modified: Added hatch and edgecolor parameters
                ax.fill_between(time_range, means - 2 * stds, means + 2 * stds,
                                color=color, alpha=alpha,
                                hatch=current_hatch, edgecolor=color)
    if title:
        ax.set_title(title)
    if bars is not None:
        ax.vlines(bars, ts_bot, ts_top, colors='black', label='Measurements')
    if floors is not None:
        ax.hlines(floors, xs_bot, xs_top, colors='orange', label='Truth', linestyle='dashed')

    # Important: The legend needs to be handled carefully with fill_between
    # The plot lines already have labels, so this should work correctly.
    if labels or num_series > 1:
        ax.legend()

    plt.show()