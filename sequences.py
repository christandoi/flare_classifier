import numpy as np
import polars as pl
from tqdm.auto import tqdm

import read_spt3g_fits as spt3g
import TS_fitter as tsf
from dataset import averaged_gaussian
from load_db import query_df


# input channels of each sequence, in array order
CHANNELS = ["snr_95", "snr_150", "t_rel_scaled", "gap_scaled", "mask"]


def fetch_window_observations(conn):
    """
    Gets every window's observations from the database in one query, then
    formats each window exactly as format_data_fits() does for the TS
    fitter: times relative to window_start, and bad data (non-finite flux or
    error, or error <= 0) set to flux 0 and error inf, per band.

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection to the flare database.

    Returns
    -------
    windows : dict
        window_id -> (time_arr, flux_arr, err_arr), with time_arr of shape
        (n_obs,) in days since window_start, and flux_arr, err_arr of shape
        (2, n_obs) with rows (95 GHz, 150 GHz), as from format_data_fits().
    """
    obs = query_df(
        conn,
        """
        SELECT
            w.window_id,
            o.mjd - w.window_start AS t_rel_d,
            o.flux_95, o.err_95, o.flux_150, o.err_150
        FROM windows w
        CROSS JOIN window_grid g
        JOIN observations o
            ON o.gaia_id = w.gaia_id
            AND o.mjd >= w.window_start
            AND o.mjd < w.window_start + g.window_d
        ORDER BY w.window_id, o.mjd
    """,
    )

    window_ids = obs["window_id"].to_numpy()
    time_all = obs["t_rel_d"].to_numpy()
    flux_all = np.stack([obs["flux_95"].to_numpy(), obs["flux_150"].to_numpy()])
    err_all = np.stack([obs["err_95"].to_numpy(), obs["err_150"].to_numpy()])

    # same masking as format_data_fits()
    bad_data = ~np.isfinite(flux_all) | ~np.isfinite(err_all) | (err_all <= 0)
    flux_all = np.where(bad_data, 0.0, flux_all)
    err_all = np.where(bad_data, np.inf, err_all)

    # rows are sorted by window_id, so each window is one contiguous block
    unique_ids, first = np.unique(window_ids, return_index=True)
    last = np.append(first[1:], len(window_ids))
    return {
        int(wid): (time_all[i:j], flux_all[:, i:j], err_all[:, i:j])
        for wid, i, j in zip(unique_ids, first, last)
    }


def add_injection(time_arr, flux_arr, err_arr, inj):
    """
    Adds an injected flare to a window's data, exactly as fit_injections()
    did: the observation-averaged Gaussian from averaged_gaussian(), added
    only to unmasked observations, with the host's real noise kept.

    Arguments
    ---------
    time_arr, flux_arr, err_arr : array_like
        One window's data, as returned by fetch_window_observations().
    inj : dict
        Injection parameters: inj_t0_rel_d, inj_fwhm_d, inj_amp_95,
        inj_amp_150 (as stored in the injections table).

    Returns
    -------
    flux_inj : array_like
        Flux with the flare added, shape (2, n_obs).
    """
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
    return np.where(good, flux_arr + signal, flux_arr)


def to_sequence(time_arr, flux_arr, err_arr, max_len, window=12.0):
    """
    Converts one window into the CNN input: one column per observation, in
    time order, padded with zeros to max_len. Channels (rows), see CHANNELS:
      snr_95, snr_150 : flux / error per band (0 for masked points)
      t_rel_scaled    : time since window start / window
      gap_scaled      : time since the previous observation / window
                        (0 for the first observation)
      mask            : 1 if the observation has good data in at least one
                        band, 0 for fully masked observations and padding

    Arguments
    ---------
    time_arr, flux_arr, err_arr : array_like
        One window's data (flux may include an injected flare).
    max_len : int
        Padded sequence length; must be >= the number of observations.
    window : float
        Window length in days, used to scale the time channels.

    Returns
    -------
    sequence : array_like
        Array of shape (5, max_len), float32.
    """
    n_obs = len(time_arr)
    snr = flux_arr / err_arr  # masked points: 0 / inf = 0
    gap = np.diff(time_arr, prepend=time_arr[0])

    sequence = np.zeros((len(CHANNELS), max_len), dtype=np.float32)
    sequence[0, :n_obs] = snr[0]
    sequence[1, :n_obs] = snr[1]
    sequence[2, :n_obs] = time_arr / window
    sequence[3, :n_obs] = gap / window
    sequence[4, :n_obs] = np.isfinite(err_arr).any(axis=0)
    return sequence


def build_sequences(rows, windows, max_len=None, window=12.0):
    """
    Builds the CNN input array for a table of rows (e.g. the dataset view).
    Rows with injection parameters get their flare added back to the host
    window's data first; rows without (inj_amp_150 null) use the raw data.

    Arguments
    ---------
    rows : polars.DataFrame
        One row per example, with window_id and the injection columns
        inj_t0_rel_d, inj_fwhm_d, inj_amp_95, inj_amp_150 (null for
        quiescent windows).
    windows : dict
        Output of fetch_window_observations().
    max_len : int, optional
        Padded sequence length. Defaults to the longest window in `rows`.
    window : float
        Window length in days.

    Returns
    -------
    window_sequences : array_like
        Sequences, shape (n_rows, 5, max_len), float32, in the order of `rows`.
    """
    if max_len is None:
        max_len = max(len(windows[wid][0]) for wid in rows["window_id"].unique())

    window_sequences = np.zeros((rows.height, len(CHANNELS), max_len), dtype=np.float32)
    for k, row in enumerate(tqdm(rows.iter_rows(named=True), total=rows.height)):
        time_arr, flux_arr, err_arr = windows[row["window_id"]]
        if row["inj_amp_150"] is not None:
            flux_arr = add_injection(time_arr, flux_arr, err_arr, row)
        window_sequences[k] = to_sequence(time_arr, flux_arr, err_arr, max_len, window)
    return window_sequences


def check_rebuild(rows, windows, n_check=100, seed=100, window=12.0):
    """
    Checks that the rebuilt data are what the TS fitter saw: refits a random
    sample of rows with ts_fit() and compares with the stored TS. Agreement
    to numerical precision means the CNN trains on the same light curves as
    the feature-based models.

    Arguments
    ---------
    rows : polars.DataFrame
        Rows with window_id, ts (stored fit) and the injection columns, e.g.
        the dataset view.
    windows : dict
        Output of fetch_window_observations().
    n_check : int
        Rows to refit per source (quiescent and injection).
    seed : int
        Random seed for the sample.
    window : float
        Window length in days.

    Returns
    -------
    check : polars.DataFrame
        source, window_id, inj_id, ts (stored), ts_rebuilt, abs_diff.
    """
    sample = pl.concat(
        [
            group.sample(min(n_check, group.height), seed=seed)
            for _, group in rows.group_by("source", maintain_order=True)
        ]
    )

    ts_rebuilt = []
    for row in tqdm(sample.iter_rows(named=True), total=sample.height):
        time_arr, flux_arr, err_arr = windows[row["window_id"]]
        if row["inj_amp_150"] is not None:
            flux_arr = add_injection(time_arr, flux_arr, err_arr, row)
        TS = tsf.ts_fit(time_arr, flux_arr, err_arr, window)[0]
        ts_rebuilt.append(float(TS))

    return (
        sample.select("source", "window_id", "inj_id", "ts")
        .with_columns(pl.Series("ts_rebuilt", ts_rebuilt))
        .with_columns((pl.col("ts_rebuilt") - pl.col("ts")).abs().alias("abs_diff"))
    )


def event_sequences(
    conn, windows, max_len, mask_halfwidth=1.0, min_good=6, window=12.0
):
    """
    CNN inputs for the real flares (the catalog plus Guns et al. 2026 flare
    5, i.e. every row of catalog_flares), built the same way as the
    real-event test of the feature-based models: every grid window that
    contains the event (spt3g.window_starts()), with observations within
    mask_halfwidth of the star's other known flares masked (as
    format_data_fits(mask_mjds=...) does), keeping windows with at least
    min_good observations good in both bands.

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection to the flare database.
    windows : dict
        Output of fetch_window_observations().
    max_len : int
        Padded sequence length (use the training sequences' length).
    mask_halfwidth : float
        Half-width (days) of the masked region around other flares.
    min_good : int
        Minimum observations good in both bands for a window to be kept.
    window : float
        Window length in days.

    Returns
    -------
    event_rows : polars.DataFrame
        One row per event window: gaia_id, spt_id, mjd, origin,
        window_start, window_id.
    event_window_sequences : array_like
        Shape (n_rows, 5, max_len), in the order of event_rows.
    """
    events = query_df(
        conn,
        """
        SELECT gaia_id, COALESCE(spt_id, source || ' flare') AS spt_id, mjd, source AS origin
        FROM catalog_flares
        ORDER BY flare_id
    """,
    )
    anchor, stride_d = conn.execute(
        "SELECT anchor, stride_d FROM window_grid"
    ).fetchone()
    window_ids = {
        (gaia_id, window_index): window_id
        for window_id, gaia_id, window_index in conn.execute(
            "SELECT window_id, gaia_id, window_index FROM windows"
        ).fetchall()
    }
    known_mjds = dict(events.group_by("gaia_id").agg("mjd").iter_rows())

    rows, sequences = [], []
    for event in tqdm(events.iter_rows(named=True), total=events.height):
        other_mjds = np.array(
            [m for m in known_mjds[event["gaia_id"]] if m != event["mjd"]]
        )
        for t_start in spt3g.window_starts(event["mjd"], anchor):
            window_index = int(round((t_start - anchor) / stride_d))
            window_id = window_ids.get((event["gaia_id"], window_index))
            if window_id is None:  # fewer than min_obs observations, never built
                continue
            time_arr, flux_arr, err_arr = windows[window_id]

            # mask the star's other known flares, both bands
            if len(other_mjds) > 0:
                near_other = (
                    np.abs(time_arr[:, None] - (other_mjds - t_start)[None, :])
                    <= mask_halfwidth
                ).any(axis=1)
                flux_arr = np.where(near_other, 0.0, flux_arr)
                err_arr = np.where(near_other, np.inf, err_arr)

            if np.isfinite(err_arr).all(axis=0).sum() < min_good:
                continue
            rows.append({**event, "window_start": t_start, "window_id": window_id})
            sequences.append(to_sequence(time_arr, flux_arr, err_arr, max_len, window))

    return pl.DataFrame(rows), np.stack(sequences)
