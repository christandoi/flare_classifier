import numpy as np
import polars as pl
from tqdm.auto import tqdm

import read_spt3g_fits as spt3g
import TS_fitter as tsf


def star_windows(
    mjd, flare_mjds, anchor, window=12.0, stride=6.0, min_good=6, flare_margin=1.0
):
    """
    Finds the sliding windows (length `window`, step `stride`, grid starting at
    `anchor`) for one star's light curve, keeping windows with at least
    `min_good` observations, and counts the catalog flares in each.

    Window edges match format_data_fits(): an observation is in a window if
    window_start <= mjd < window_start + window.

    Arguments
    ---------
    mjd : array_like
        Sorted observation times (MJD) of one star.
    flare_mjds : array_like
        Catalog flare times (MJD) of the same star; may be empty.
    anchor : float
        Start time (MJD) of the window grid.
    window, stride : float
        Window length and step, in days. Defaults to 12 and 6.
    min_good : int
        Minimum number of observations for a window to be kept.
    flare_margin : float
        A flare counts toward a window if its MJD falls within
        [window_start - flare_margin, window_start + window + flare_margin).

    Returns
    -------
    starts : array_like
        Start times (MJD) of the kept windows.
    n_obs : array_like
        Number of observations in each kept window.
    span_d : array_like
        Time between the first and last observation in each window, in days.
    n_flares : array_like
        Number of catalog flares in each window, including the margin.
    """
    mjd = np.asarray(mjd, dtype=float)
    flare_mjds = np.sort(np.asarray(flare_mjds, dtype=float))

    # every grid window that could contain at least one observation; the -1
    # keeps the window that starts before the first observation
    k_min = np.floor((mjd[0] - anchor) / stride) - 1
    k_max = np.floor((mjd[-1] - anchor) / stride)
    starts = anchor + stride * np.arange(k_min, k_max + 1)

    # index range of the observations inside each window, in one vectorized step
    first = np.searchsorted(mjd, starts, side="left")
    last = np.searchsorted(mjd, starts + window, side="left")
    n_obs = last - first

    keep = n_obs >= min_good
    starts, first, last, n_obs = starts[keep], first[keep], last[keep], n_obs[keep]
    span_d = mjd[last - 1] - mjd[first]

    # catalog flares falling in each window (with margin)
    n_flares = np.searchsorted(
        flare_mjds, starts + window + flare_margin, side="left"
    ) - np.searchsorted(flare_mjds, starts - flare_margin, side="left")
    return starts, n_obs, span_d, n_flares


def enumerate_windows(
    lightcurves, catalog, anchor, window=12.0, stride=6.0, min_good=6, flare_margin=1.0
):
    """
    Lists every sliding window for each star (see star_windows()) and flags the
    windows that contain a catalog flare. Quiescent windows (contaminated ==
    False) are the negatives and the injection hosts for the classifier.

    Arguments
    ---------
    lightcurves : polars.DataFrame
        Output of read_lightcurves().
    catalog : polars.DataFrame
        Flare catalog, with dr3_source_id read as a string.
    anchor : float
        Start time (MJD) of the window grid; use fixed_anchor.
    window, stride : float
        Window length and step, in days. Defaults to 12 and 6.
    min_good : int
        Minimum number of observations for a window to be kept.
    flare_margin : float
        Margin (days) around each catalog flare when flagging windows.

    Returns
    -------
    windows : polars.DataFrame
        One row per kept window, sorted by gaia_id then window_start, with
        columns: gaia_id, window_start (MJD), n_obs, span_d (days),
        n_flares, contaminated (n_flares > 0).
    """
    flares_by_star = dict(catalog.group_by("dr3_source_id").agg("mjd").iter_rows())

    tables = []
    for (gaia_id,), star in lightcurves.group_by("gaia_id"):
        mjd = star["mjd"].sort().to_numpy()
        starts, n_obs, span_d, n_flares = star_windows(
            mjd,
            flares_by_star.get(gaia_id, []),
            anchor,
            window,
            stride,
            min_good,
            flare_margin,
        )
        tables.append(
            pl.DataFrame(
                {
                    "gaia_id": [gaia_id] * len(starts),
                    "window_start": starts,
                    "n_obs": n_obs,
                    "span_d": span_d,
                    "n_flares": n_flares,
                }
            )
        )

    windows = (
        pl.concat(tables)
        .with_columns((pl.col("n_flares") > 0).alias("contaminated"))
        .sort("gaia_id", "window_start")
    )
    return windows


def window_features(time_arr, flux_arr, err_arr, fit, window=12.0):
    """
    Computes the classifier features for one window from its data and its TS
    fit. Used for both quiescent and injected windows, so that negatives and
    positives go through exactly the same processing.

    Arguments
    ---------
    time_arr, flux_arr, err_arr : array_like
        Time, flux and error arrays as formatted by format_data_fits().
        Masked points have flux 0 and error inf, so their SNR is 0.
    fit : tuple
        Output of ts_fit(): (TS, t0, fwhm, amps, amp_errs, surface_map).
    window : float
        Window length in days, used to detect fits at the FWHM upper bound.

    Returns
    -------
    features : dict
        ts, fwhm_d, t0_rel_d (fit peak time relative to the window start;
        diagnostic, not a classifier feature), snr_amp_95, snr_amp_150
        (fitted amplitude / its error), fwhm_at_bound, n_good, span_d,
        null_chi2_red (reduced chi^2 of a zero-flux model over both bands),
        n_3sigma_both (observations above 3 sigma in both bands), peak_snr
        (highest single-observation SNR, bands combined).
    """
    TS, t0, fwhm, amps, amp_errs, _ = fit
    good = np.isfinite(err_arr).all(axis=0)
    snr = flux_arr / err_arr  # (2, n_obs); masked points give 0
    n_good = int(good.sum())

    return {
        "ts": float(TS),
        "fwhm_d": float(fwhm),
        "t0_rel_d": float(t0),
        "snr_amp_95": float(amps[0] / amp_errs[0]),
        "snr_amp_150": float(amps[1] / amp_errs[1]),
        "fwhm_at_bound": bool(np.isclose(fwhm, 3 * window, rtol=1e-3)),
        "n_good": n_good,
        "span_d": float(time_arr[good].max() - time_arr[good].min()),
        "null_chi2_red": float((snr[:, good] ** 2).sum() / (2 * n_good)),
        "n_3sigma_both": int(((snr[0] > 3) & (snr[1] > 3)).sum()),
        "peak_snr": float(((snr[0] + snr[1]) / np.sqrt(2)).max()),
    }


def fit_windows(windows, lightcurves, window=12.0):
    """
    Fits the TS for every window in `windows` and computes its features with
    window_features(). Run on the quiescent windows this gives the null (noise
    only) distribution of TS and the classifier negatives.

    Arguments
    ---------
    windows : polars.DataFrame
        Rows of enumerate_windows() to fit (e.g. filtered to contaminated ==
        False).
    lightcurves : polars.DataFrame
        Output of read_lightcurves().
    window : float
        Window length in days. Defaults to 12.

    Returns
    -------
    fits : polars.DataFrame
        One row per input window: gaia_id, window_start, then the
        window_features() columns.
    """
    rows = []
    for row in tqdm(windows.iter_rows(named=True), total=windows.height):
        time_arr, flux_arr, err_arr, t_ref = spt3g.format_data_fits(
            lightcurves, row["gaia_id"], row["window_start"], window
        )
        fit = tsf.ts_fit(time_arr, flux_arr, err_arr, window)
        rows.append(
            {
                "gaia_id": row["gaia_id"],
                "window_start": row["window_start"],
                **window_features(time_arr, flux_arr, err_arr, fit, window),
            }
        )
    return pl.DataFrame(rows)


def averaged_gaussian(
    obs_mjd, t0, fwhm, amp, cross_offset=50 / 1440, cross_duration=30 / 1440
):
    """
    Gaussian flare profile averaged over the time each observation actually
    spends on the source, rather than sampled at the observation's MJD. The
    public light-curve MJD is the observation start; single-scan timestamps
    show a source is crossed over ~30 min starting ~50 min later.

    Uses the analytic integral of a Gaussian (error functions), so very short
    flares are diluted correctly: a 30 min flare inside a 30 min crossing
    contributes much less than its peak amplitude.

    Arguments
    ---------
    obs_mjd : array_like
        Observation start times (MJD, or any time in days, consistent with t0).
    t0 : float
        Flare peak time, same units as obs_mjd.
    fwhm : float
        Flare FWHM, in days.
    amp : float
        Flare peak amplitude (mJy).
    cross_offset, cross_duration : float
        Start of the source crossing relative to obs_mjd, and its length,
        in days. Default 50 min and 30 min.

    Returns
    -------
    flux : array_like
        Average flare flux (mJy) during each observation's crossing.
    """
    from scipy.special import erf

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    a = np.asarray(obs_mjd) + cross_offset - t0
    b = a + cross_duration
    integral = (
        sigma
        * np.sqrt(np.pi / 2)
        * (erf(b / (np.sqrt(2) * sigma)) - erf(a / (np.sqrt(2) * sigma)))
    )
    return amp * integral / cross_duration


def draw_injections(
    hosts,
    n_injections,
    alpha_pool,
    seed=0,
    window=12.0,
    fwhm_range=(30 / 1440, 10.0),
    snr_range=(1.0, 100.0),
):
    """
    Draws random flare parameters and host windows for injection.

    Arguments
    ---------
    hosts : polars.DataFrame
        Candidate host windows (gaia_id, window_start), e.g. the non-ambiguous
        quiescent windows. Sampled uniformly with replacement.
    n_injections : int
        Number of injections to draw.
    alpha_pool : array_like
        Spectral indices (S ~ nu^alpha, 95-150 GHz) to resample from, e.g. the
        catalog's spectral_index_95_150 column.
    seed : int
        Random seed, for reproducibility.
    window : float
        Window length in days; t0 is drawn uniformly within it.
    fwhm_range : 2-tuple
        Min and max FWHM in days, drawn log-uniform. Default 30 min to 10 d.
    snr_range : 2-tuple
        Min and max peak SNR at 150 GHz, drawn log-uniform. The 150 GHz peak
        amplitude is snr_peak times the host window's median 150 GHz error.

    Returns
    -------
    injections : polars.DataFrame
        One row per injection: inj_id, gaia_id, window_start, inj_t0_rel_d
        (peak time relative to window start), inj_fwhm_d, inj_snr_peak,
        inj_alpha.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, hosts.height, n_injections)
    log_uniform = lambda lo, hi: np.exp(
        rng.uniform(np.log(lo), np.log(hi), n_injections)
    )

    return pl.DataFrame(
        {
            "inj_id": np.arange(n_injections),
            "gaia_id": hosts["gaia_id"].gather(idx),
            "window_start": hosts["window_start"].gather(idx),
            "inj_t0_rel_d": rng.uniform(0, window, n_injections),
            "inj_fwhm_d": log_uniform(*fwhm_range),
            "inj_snr_peak": log_uniform(*snr_range),
            "inj_alpha": rng.choice(np.asarray(alpha_pool, dtype=float), n_injections),
        }
    )


def fit_injections(injections, lightcurves, window=12.0, normalization="band_150"):
    """
    Injects each flare from draw_injections() into its host window's real
    data, then fits and computes features exactly as fit_windows() does for
    quiescent windows. The host's own noise is kept; only signal is added,
    and only to unmasked observations.

    Arguments
    ---------
    injections : polars.DataFrame
        Output of draw_injections().
    lightcurves : polars.DataFrame
        Output of read_lightcurves().
    window : float
        Window length in days. Defaults to 12.
    normalization : str
        How inj_snr_peak sets the flare's brightness.
        'band_150' (default, used for the main injections): inj_snr_peak is
        the peak SNR at 150 GHz; the 95 GHz amplitude follows from the
        spectral index, so changing alpha changes only the 95 GHz flux.
        'combined': inj_snr_peak is the peak SNR of both bands combined in
        quadrature, sqrt(snr_95**2 + snr_150**2); both amplitudes are scaled
        so changing alpha changes the spectral shape at fixed total
        significance. Used for the fixed-alpha stress test.

    Returns
    -------
    fits : polars.DataFrame
        The injection parameters plus inj_amp_95, inj_amp_150 (peak, mJy),
        inj_max_obs_snr_150 (largest averaged injected signal in any one
        observation / its error at 150 GHz), inj_max_obs_snr_combined (the
        same with both bands combined in quadrature), and the
        window_features() columns.
    """
    rows = []
    for inj in tqdm(injections.iter_rows(named=True), total=injections.height):
        time_arr, flux_arr, err_arr, t_ref = spt3g.format_data_fits(
            lightcurves, inj["gaia_id"], inj["window_start"], window
        )
        good = np.isfinite(err_arr)
        median_err_95 = np.median(err_arr[0][good[0]])
        median_err_150 = np.median(err_arr[1][good[1]])
        amp_ratio = (95 / 150) ** inj["inj_alpha"]  # amp_95 / amp_150

        if normalization == "band_150":
            amp_150 = inj["inj_snr_peak"] * median_err_150
        elif normalization == "combined":
            # combined peak SNR for amp_150 = 1 mJy, then scale to the requested SNR
            combined_snr_per_mjy = np.hypot(
                amp_ratio / median_err_95, 1 / median_err_150
            )
            amp_150 = inj["inj_snr_peak"] / combined_snr_per_mjy
        else:
            raise ValueError(f"unknown normalization '{normalization}'")
        amp_95 = amp_150 * amp_ratio
        signal = np.stack(
            [
                averaged_gaussian(
                    time_arr, inj["inj_t0_rel_d"], inj["inj_fwhm_d"], amp_95
                ),
                averaged_gaussian(
                    time_arr, inj["inj_t0_rel_d"], inj["inj_fwhm_d"], amp_150
                ),
            ]
        )
        flux_inj = np.where(good, flux_arr + signal, flux_arr)

        fit = tsf.ts_fit(time_arr, flux_inj, err_arr, window)
        rows.append(
            {
                **inj,
                "inj_amp_95": float(amp_95),
                "inj_amp_150": float(amp_150),
                "inj_max_obs_snr_150": float(
                    np.max(np.where(good[1], signal[1] / err_arr[1], 0))
                ),
                "inj_max_obs_snr_combined": float(
                    np.max(
                        np.where(
                            good.all(axis=0),
                            np.hypot(signal[0] / err_arr[0], signal[1] / err_arr[1]),
                            0,
                        )
                    )
                ),
                **window_features(time_arr, flux_inj, err_arr, fit, window),
            }
        )
    return pl.DataFrame(rows)


def observed_signal_snr(injections, lightcurves, window=12.0):
    """
    Recomputes the observed (noiseless) SNR of injected flares from their
    stored parameters, without refitting: the largest injected signal in any
    one observation divided by that observation's error, at 150 GHz and with
    both bands combined in quadrature. Same definitions as fit_injections(),
    so inj_max_obs_snr_150 should reproduce the stored column exactly.

    Arguments
    ---------
    injections : polars.DataFrame
        Injections with gaia_id, window_start, inj_t0_rel_d, inj_fwhm_d,
        inj_amp_95 and inj_amp_150 (e.g. the output of fit_injections()).
    lightcurves : polars.DataFrame
        Output of read_lightcurves().
    window : float
        Window length in days. Defaults to 12.

    Returns
    -------
    snr : polars.DataFrame
        inj_max_obs_snr_150 and inj_max_obs_snr_combined, one row per
        injection, in the input order.
    """
    rows = []
    for inj in tqdm(injections.iter_rows(named=True), total=injections.height):
        time_arr, flux_arr, err_arr, t_ref = spt3g.format_data_fits(
            lightcurves, inj["gaia_id"], inj["window_start"], window
        )
        good = np.isfinite(err_arr)
        signal = np.stack(
            [
                averaged_gaussian(
                    time_arr, inj["inj_t0_rel_d"], inj["inj_fwhm_d"], inj["inj_amp_95"]
                ),
                averaged_gaussian(
                    time_arr, inj["inj_t0_rel_d"], inj["inj_fwhm_d"], inj["inj_amp_150"]
                ),
            ]
        )
        rows.append(
            {
                "inj_max_obs_snr_150": float(
                    np.max(np.where(good[1], signal[1] / err_arr[1], 0))
                ),
                "inj_max_obs_snr_combined": float(
                    np.max(
                        np.where(
                            good.all(axis=0),
                            np.hypot(signal[0] / err_arr[0], signal[1] / err_arr[1]),
                            0,
                        )
                    )
                ),
            }
        )
    return pl.DataFrame(rows)
