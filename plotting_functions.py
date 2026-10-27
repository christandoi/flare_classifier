import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from TS_fitter import make_grids

BAND_COLORS = {"95 GHz": "orangered", "150 GHz": "dodgerblue"}
METHOD_STYLES = {
    "ts_positive_fwhm_pixel_45": {
        "label": "all pipeline cuts, TS > 45",
        "color": "black",
        "marker": "o",
        "open": False,
    },
    "ts_positive_fwhm_pixel_27": {
        "label": "all pipeline cuts, TS > 27",
        "color": "black",
        "marker": "o",
        "open": True,
    },
    "ts_positive_fwhm_45": {
        "label": "no pixel cut, TS > 45",
        "color": "mediumseagreen",
        "marker": "s",
        "open": False,
    },
    "ts_positive_fwhm_27": {
        "label": "no pixel cut, TS > 27",
        "color": "mediumseagreen",
        "marker": "s",
        "open": True,
    },
    "logistic": {
        "label": "logistic regression",
        "color": "slateblue",
        "marker": "D",
        "open": False,
    },
    "gbm": {
        "label": "gradient-boosted trees",
        "color": "orchid",
        "marker": "^",
        "open": False,
    },
    "cnn": {"label": "1D CNN", "color": "crimson", "marker": "v", "open": False},
}


def plot_lightcurve_fit(
    ax, time_arr, flux_arr, err_arr, t_ref, fit, title=None, legend=False
):
    """
    Draws one window's light curve and best-fit Gaussian on an existing axis.
    Data are points with error bars (no connecting lines); only the fitted
    model is drawn as a line. Masked points (err = inf) are skipped.

    Arguments
    ---------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    time_arr, flux_arr, err_arr : array_like
        Time, flux and error arrays as formatted by format_data_fits().
    t_ref : float
        Reference time (MJD) returned by format_data_fits().
    fit : tuple
        Output of ts_fit(): (TS, t0, fwhm, amps, amp_errs, surface_map).
    title : str, optional
        Panel title.
    legend : bool
        Whether to add a band legend to this axis.
    """
    TS, t0, fwhm, amps, amp_errs, _ = fit
    good = np.isfinite(err_arr)
    for band, (label, c) in enumerate(BAND_COLORS.items()):
        g = good[band]
        ax.errorbar(
            time_arr[g] + t_ref,
            flux_arr[band, g],
            yerr=err_arr[band, g],
            c=c,
            fmt="o",
            ms=3,
            capsize=2,
            label=label,
        )
    ylims = ax.get_ylim()

    # dense model grid, with extra points around t0 so short flares are resolved
    model_times = np.union1d(
        np.linspace(time_arr.min(), time_arr.max(), 2000),
        np.linspace(t0 - 3 * fwhm, t0 + 3 * fwhm, 200),
    )
    profile = np.exp(-4 * np.log(2) * (model_times - t0) ** 2 / fwhm**2)
    for band, c in enumerate(BAND_COLORS.values()):
        ax.plot(model_times + t_ref, amps[band] * profile, c=c, lw=0.75)

    ax.axhline(0, c="k", ls="--", lw=0.5, zorder=-10)
    ax.set_ylim(ylims)
    ax.set_xlim(time_arr.min() + t_ref, time_arr.max() + t_ref)
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    if title is not None:
        ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(fontsize=8)


def plot_fit(
    time_arr,
    flux_arr,
    err_arr,
    t_ref,
    fit,
    title=None,
    filename=None,
    show_surface=True,
):
    """
    Plots one light curve with its best-fit Gaussian (plot_lightcurve_fit())
    and, optionally, the TS surface map next to it.

    Arguments
    ---------
    time_arr, flux_arr, err_arr : array_like
        Time, flux and error arrays as formatted by format_data_fits().
    t_ref : float
        Reference time (MJD) returned by format_data_fits().
    fit : tuple
        Output of ts_fit(): (TS, t0, fwhm, amps, amp_errs, surface_map).
    title : str, optional
        Figure title; defaults to the fitted TS and FWHM.
    filename : str, optional
        If given, saves the figure to this path.
    show_surface : bool
        Whether to draw the TS surface map. It's skipped automatically when
        ts_fit() returned no surface.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    TS, t0, fwhm, amps, amp_errs, surface_map = fit
    show_surface = show_surface and surface_map is not None

    if show_surface:
        fig, (ax0, ax1) = plt.subplots(
            1,
            2,
            figsize=(15, 4.5),
            gridspec_kw={"width_ratios": [2, 1], "wspace": 0.25},
        )
    else:
        fig, ax0 = plt.subplots(figsize=(10, 4.5))
    fig.suptitle(title or f"TS: {TS:.1f} | FWHM: {fwhm * 24:.2f} hr", fontsize=15)

    plot_lightcurve_fit(ax0, time_arr, flux_arr, err_arr, t_ref, fit, legend=True)
    ax0.set_xlabel("MJD")
    ax0.set_ylabel("Flux [mJy]")

    if show_surface:
        t0_grid, fwhm_grid, _ = make_grids(time_arr)
        cax = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.1)
        plot_ts_surface(
            surface_map,
            t0_grid + t_ref,
            fwhm_grid,
            ax=ax1,
            cax=cax,
            best=(t0 + t_ref, fwhm),
        )

    if filename is not None:
        plt.savefig(filename, dpi=150.0, facecolor="white", bbox_inches="tight")
    return fig


def plot_ts_surface(
    surface_map,
    t0_grid,
    fwhm_grid,
    ax,
    cax,
    cmap="viridis",
    best=None,
    dynamic_range=None,
    scale="linear",
):
    """
    Plots the TS surface over the (t0, FWHM) grid with 1-, 2-, and 3-sigma
    contours for two parameters, and optionally marks the best fit.

    Arguments
    ---------
    surface_map : array_like
        TS surface from ts_surface(), shape (len(t0_grid), len(fwhm_grid)).
    t0_grid, fwhm_grid : array_like
        Grids the surface was evaluated on (t0 in MJD if already offset by t_ref).
    ax, cax : matplotlib.axes.Axes
        Axes for the surface and its colorbar.
    cmap : str
        Colormap name. Defaults to 'viridis'.
    best : 2-tuple, optional
        (t0, fwhm) of the best fit, marked with a red star.
    dynamic_range : float, optional
        Show only TS values within this range of the maximum.
    scale : str
        'linear' or 'log' color scaling. Log scaling clips TS <= 1.
    """
    vmax = surface_map.max()
    vmin = surface_map.min() if dynamic_range is None else vmax - dynamic_range
    if scale == "log":
        color_norm = colors.LogNorm(vmin=max(vmin, 1), vmax=vmax)
    else:
        color_norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # surface_map has shape (len(t0_grid), len(fwhm_grid)); transpose puts t0 on x
    mesh = ax.pcolormesh(
        t0_grid, fwhm_grid, surface_map.T, shading="auto", cmap=cmap, norm=color_norm
    )

    # delta TS for 1-, 2-, 3-sigma with two free parameters
    levels = [vmax - 11.83, vmax - 6.18, vmax - 2.30]
    contour_labels = {
        levels[0]: r"3-$\sigma$ (11.83 TS)",
        levels[1]: r"2-$\sigma$ (6.18 TS)",
        levels[2]: r"1-$\sigma$ (2.30 TS)",
    }
    # thick black under thin white so contours show on any color; label once
    ax.contour(
        t0_grid, fwhm_grid, surface_map.T, levels=levels, colors="k", linewidths=3
    )
    contours = ax.contour(
        t0_grid, fwhm_grid, surface_map.T, levels=levels, colors="white", linewidths=1
    )
    labels = ax.clabel(
        contours, inline=True, fmt=contour_labels, fontsize=10, colors="white"
    )
    for label in labels:
        label.set_path_effects(
            [path_effects.withStroke(linewidth=3, foreground="black")]
        )

    if best is not None:
        t0, fwhm = best
        ax.plot(t0, fwhm, "r*", ms=12)
    ax.set_yscale("log")
    ax.set_xlabel("MJD")
    ax.set_ylabel("FWHM [days]")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    plt.colorbar(mesh, cax=cax, label="TS")
