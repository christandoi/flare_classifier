import numpy as np
import polars as pl
import warnings
from astropy.units import UnitsWarning
from astropy.table import Table


def read_lightcurves(path):
    """
    Reads the four-year SPT-3G flare star light curves (Tandoi et al. 2024,
    Zenodo 10.5281/zenodo.11372033, spt3g_flare_lightcurves_2019_2022.fits).

    The FITS table has one row per Gaia star, with the time series stored as
    variable-length arrays. This unpacks it into a long-format DataFrame with
    one row per observation, which makes grouping by star and cutting time
    windows straightforward. Fluxes are measured at the Gaia position of each
    star, not at the pixel the flare was detected in.

    Column names are changed from e.g. '95_flux' to 'flux_95' so they are
    valid Python identifiers (df.flux_95 works, df.95_flux does not).

    Arguments
    ---------
    path : str
        Path to spt3g_flare_lightcurves_2019_2022.fits.

    Returns
    -------
    lightcurves : polars.DataFrame
        One row per observation, sorted by gaia_id then mjd, with columns:
        spt_id, gaia_id (str), ra, dec (deg), mjd (days), flux_95, err_95,
        flux_150, err_150 (mJy).
    """

    def to_str(value):
        # depending on the astropy version, string columns come back as
        # bytes or str, padded with trailing spaces
        if isinstance(value, bytes):
            value = value.decode("ascii")
        return str(value).strip()

    # the file labels the mjd column with unit 'MJD', which is not a valid
    # FITS unit; units are dropped below anyway, so silence the warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        table = Table.read(path)

    stars = []
    for row in table:
        star = pl.DataFrame(
            {
                "spt_id": to_str(row["spt_id"]),
                "gaia_id": to_str(row["gaia_id"]),
                # ra/dec are stored as strings in the FITS file
                "ra": float(to_str(row["ra"])),
                "dec": float(to_str(row["dec"])),
                # convert the big-endian FITS arrays to native byte order
                "mjd": np.asarray(row["mjd"], dtype=float),
                "flux_95": np.asarray(row["95_flux"], dtype=float),
                "err_95": np.asarray(row["95_err"], dtype=float),
                "flux_150": np.asarray(row["150_flux"], dtype=float),
                "err_150": np.asarray(row["150_err"], dtype=float),
            }
        )
        stars.append(star)

    lightcurves = pl.concat(stars)
    lightcurves = lightcurves.sort(["gaia_id", "mjd"])
    return lightcurves


def format_data_fits(
    lightcurves, gaia_id, t_start, window=12.0, mask_mjds=None, mask_halfwidth=1.0
):
    """
    Cuts one star's light curve to a single window and formats it for the TS
    fitter, matching the output of format_data_spt3g() for a single pixel.
    Masks bad data the same way: flux set to 0 and error set to inf, so those
    points carry zero weight in the fit.

    Arguments
    ---------
    lightcurves : polars.DataFrame
        Output of read_lightcurves().
    gaia_id : str
        Gaia DR3 source ID of the star.
    t_start : float
        Start of the window, in MJD.
    window : float
        Duration of the window, in days. Defaults to 12.
    mask_mjds : array_like, optional
        Absolute MJDs of other known events on this star (e.g. other catalog
        flares). Observations within mask_halfwidth of any of them are masked,
        so the fit cannot lock onto a different flare in the same window.
    mask_halfwidth : float
        Half-width of each masked region, in days. Defaults to 1.

    Returns
    -------
    time_arr : array_like
        Timestamps in days relative to t_ref, shape (n_obs,).
    flux_arr, err_arr : array_like
        Flux and error (mJy) for both bands, shape (2, n_obs), with rows
        (95 GHz, 150 GHz).
    t_ref : float
        Reference time (MJD) to convert back to absolute time. Set to t_start
        so that e.g. the fitted t0 converts as t_ref + t0.
    """
    star = lightcurves.filter(
        (pl.col("gaia_id") == gaia_id)
        & (pl.col("mjd") >= t_start)
        & (pl.col("mjd") < t_start + window)
    ).sort("mjd")

    t_ref = t_start
    time_arr = star["mjd"].to_numpy() - t_ref
    flux_arr = np.stack([star["flux_95"].to_numpy(), star["flux_150"].to_numpy()])
    err_arr = np.stack([star["err_95"].to_numpy(), star["err_150"].to_numpy()])
    # mask bad data
    bad_data = ~np.isfinite(flux_arr) | ~np.isfinite(err_arr) | (err_arr <= 0)

    # mask other known events on this star
    if mask_mjds is not None and len(mask_mjds) > 0:
        abs_time = time_arr + t_ref
        near_other = (
            np.abs(abs_time[:, None] - np.asarray(mask_mjds)[None, :]) <= mask_halfwidth
        ).any(axis=1)
        bad_data |= near_other[None, :]  # mask both bands

    flux_arr = np.where(bad_data, 0.0, flux_arr)
    err_arr = np.where(bad_data, np.inf, err_arr)
    return time_arr, flux_arr, err_arr, t_ref


def window_starts(event_mjd, anchor, window=12.0, stride=6.0):
    """
    Start times of the sliding windows (length `window`, step `stride`,
    beginning at `anchor`) that contain event_mjd. With window = 2 * stride,
    every time falls in exactly two windows.
    """
    k = np.floor((event_mjd - anchor) / stride)
    starts = anchor + stride * np.arange(k - window / stride + 1, k + 1)
    return starts[starts + window > event_mjd]
