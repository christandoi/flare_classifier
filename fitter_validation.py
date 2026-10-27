import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import TS_fitter as tsf
import read_spt3g_fits as spt3g
import plotting_functions as pltfun


def run_validation(
    catalog, lightcurves, anchor, window=12.0, min_good=6, plot_dir=None
):
    """
    Refits every catalog event with the pipeline's sliding windows (length
    `window`, step window/2, starting at `anchor`), keeping the higher-TS of
    the two windows that contain each event. Other catalog flares on the same
    star are masked.

    Arguments
    ---------
    catalog : polars.DataFrame
        Flare catalog, with dr3_source_id read as a string and the published
        TS as ts_published.
    lightcurves : polars.DataFrame
        Output of read_lightcurves().
    anchor : float
        Start time (MJD) of the sliding window grid.
    window : float
        Window length in days. Defaults to 12.
    min_good : int
        Minimum number of good observations for a window to be fit.
    plot_dir : str, optional
        If given, saves a plot_fit() figure per event there.

    Returns
    -------
    validation : polars.DataFrame
        One row per event that had at least one fittable window.
    """
    event_mjds = dict(catalog.group_by("dr3_source_id").agg("mjd").iter_rows())
    rows = []

    for event in tqdm(catalog.iter_rows(named=True), total=catalog.height, leave=False):
        gaia_id = event["dr3_source_id"]
        event_mjd = event["mjd"]
        # spt_id has spaces and '+/-'; build a filename-safe tag
        spt_name = event["spt_id"].replace("SPT-SV ", "").replace(" ", "")
        tag = f"{spt_name}_{event_mjd:.2f}"
        other_mjds = [m for m in event_mjds[gaia_id] if m != event_mjd]

        # fit both sliding windows containing the event, keep the higher TS
        best = None
        for t_start in spt3g.window_starts(event_mjd, anchor, window, window / 2):
            time_arr, flux_arr, err_arr, t_ref = spt3g.format_data_fits(
                lightcurves, gaia_id, t_start, window, mask_mjds=other_mjds
            )
            if np.isfinite(err_arr).all(axis=0).sum() < min_good:
                continue
            fit = tsf.ts_fit(time_arr, flux_arr, err_arr, window)
            if best is None or fit[0] > best[0][0]:
                best = (fit, time_arr, flux_arr, err_arr, t_ref)

        if best is None:
            print(tag, "no window with enough good points, skipped")
            continue
        fit, time_arr, flux_arr, err_arr, t_ref = best
        TS, t0, fwhm, amps, amp_errs, surface_map = fit

        if plot_dir is not None:
            fig = pltfun.plot_fit(
                time_arr,
                flux_arr,
                err_arr,
                t_ref,
                fit,
                title=(
                    f"{event['spt_id']} | MJD {event_mjd:.2f} | "
                    f"published TS: {event['ts_published']:.1f} | "
                    f"refit TS: {TS:.1f} | FWHM: {fwhm * 24:.2f} hr"
                ),
                filename=f"{plot_dir}/{tag}_ts_fit.png",
            )
            plt.close(fig)  # many open figures will eat memory otherwise

        # is the detection observation itself in the public light curve?
        star_mjd = lightcurves.filter(pl.col("gaia_id") == gaia_id)["mjd"].to_numpy()
        peak_obs_offset_min = np.min(np.abs(star_mjd - event_mjd)) * 24 * 60

        rows.append(
            {
                "spt_id": event["spt_id"],
                "gaia_id": gaia_id,
                "mjd": event_mjd,
                "ts_published": event["ts_published"],
                "ts_refit": float(TS),
                "t0_mjd": float(t0 + t_ref),
                "fwhm_hr": float(fwhm * 24),
                "amp_95": float(amps[0]),
                "amp_150": float(amps[1]),
                "n_obs": len(time_arr),
                "n_masked_other": int(np.isinf(err_arr[0]).sum()),
                "t0_offset_d": float(t0 + t_ref - event_mjd),
                "window_start": float(t_ref),
                "has_peak_obs": bool(peak_obs_offset_min < 15),
            }
        )

    return pl.DataFrame(rows)


def validation_score(validation, t0_max_offset=1.5):
    """
    Summarizes how well a run_validation() result reproduces the published
    TS, for comparing window-grid anchors. Events whose detection observation
    is missing from the public light curve, or whose fitted t0 is far from the
    catalog epoch, are left out.

    Arguments
    ---------
    validation : polars.DataFrame
        Output of run_validation().
    t0_max_offset : float
        Largest allowed |fitted t0 - catalog MJD|, in days.

    Returns
    -------
    n_events : int
        Number of events used.
    median_abs_log_ratio : float
        Median |log(refit TS / published TS)|; 0 is a perfect match.
    """
    clean = validation.filter(
        pl.col("has_peak_obs") & (pl.col("t0_offset_d").abs() < t0_max_offset)
    )
    log_ratio = np.log(clean["ts_refit"].to_numpy() / clean["ts_published"].to_numpy())
    return clean.height, np.median(np.abs(log_ratio))
