import numpy as np
from scipy.optimize import minimize


def format_data(map_frames, cutout_size=None):
    """
    Take in map frames of transient events (nominally 41x41 thumbnails
    from already processed light curves within a lightcurve file) and
    formats the data to pass through to the new TS fitter implemented
    using scipy.optimize.minimize rather than the C++ method featuring
    the GNU Scientific Library implementation of the Nelder-Mead
    algorithm. Masks bad data and mandates the two bands must have
    the same timestamps.

    Arguments
    ---------
    map_frames : list of frames
        Map frames (90 and 150 GHz) of observations within lightcurve
        of interest.
    cutout_size : int, optional
        Half-width of thumbnail size, in pixels, if you do not want
        to run the fitter over the full size input maps.

    Returns
    -------
    time_arr, flux_arr, err_ar : array_like
        Arrays containing the timestamps (in days), flux (in mJy) and
        errors (mJy) for both bands.
    t_ref : float
        The reference time of the event to convert back to an absolute
        value, as time_arr is given in units relative to the start of
        the light curve.
    """
    from spt3g import core, maps, std_processing

    time90 = []
    time150 = []
    flux90 = []
    flux150 = []
    err90 = []
    err150 = []

    for fr in map_frames:
        maps.RemoveWeights(fr, zero_nans=True)
        flux_map = np.array(fr["T"]) / core.G3Units.mJy
        with np.errstate(divide="ignore"):
            err_map = 1 / np.sqrt(np.array(fr["Wunpol"].TT)) / core.G3Units.mJy
        match fr["Id"]:
            case "90GHz":
                mjd = std_processing.obsid_to_g3time(fr["ObservationID"]).mjd
                time90.append(mjd)
                flux90.append(flux_map)
                err90.append(err_map)

            case "150GHz":
                mjd = std_processing.obsid_to_g3time(fr["ObservationID"]).mjd
                time150.append(mjd)
                flux150.append(flux_map)
                err150.append(err_map)

    # make sure everything is in chronological order and as numpy arrays
    time90_arr = np.asarray(time90)
    time150_arr = np.asarray(time150)
    assert np.array_equal(time90_arr, time150_arr), "times do not match"
    time_arr = time90_arr.copy()
    t_ref = time_arr.min()  # so we can convert back to absolute time

    order = np.argsort(time_arr)
    time_arr = time_arr[order] - t_ref  # change time from absolute to relative
    f90_arr = np.asarray(flux90)
    f150_arr = np.asarray(flux150)
    err90_arr = np.asarray(err90)
    err150_arr = np.asarray(err150)

    flux_arr = np.stack([f90_arr, f150_arr])[:, order]
    err_arr = np.stack([err90_arr, err150_arr])[:, order]

    # optionally keep only a box around the center, e.g. cutout_size=5 gives 11x11
    if cutout_size is not None:
        center_y, center_x = flux_arr.shape[2] // 2, flux_arr.shape[3] // 2
        box_y = slice(center_y - cutout_size, center_y + cutout_size + 1)
        box_x = slice(center_x - cutout_size, center_x + cutout_size + 1)
        flux_arr = flux_arr[..., box_y, box_x]
        err_arr = err_arr[..., box_y, box_x]

    # mask bad data
    bad_data = ~np.isfinite(flux_arr) | ~np.isfinite(err_arr) | (err_arr <= 0)
    flux_arr = np.where(bad_data, 0.0, flux_arr)
    err_arr = np.where(bad_data, np.inf, err_arr)
    return time_arr, flux_arr, err_arr, t_ref


def make_grids(time_arr, window=12.0, min_fwhm=5 / 60 / 24, max_fwhm=None):
    """
    Initializes the grids for the two parameters (flare peak time, FWHM)
    to be used for the TS fitter.

    Arguments
    ---------
    time_arr : array_like
        Array of timestamps corresponding to the flux and error data
        being passed into the fitter.
    window : float
        Maximum duration of light curve to be fit, in days. Defaults
        to 12.
    min_fwhm : float
        Minimum value of FWHM for the parameter search, in days. Defaults
        to 5 minutes.
    max_fwhm : float, optional
        Maximum value of FWHM for the parameter search, in days. When using
        the default value of None this corresponds to a FWHM of window*3 days.

    Returns
    -------
    t0_grid, fwhm_grid : array_like
        2D grids of values for the peak time and FWHM parameters
        to be searched over for a best fit.
    t0_step : float
        Time step value of our flare peak time search grid.
    """
    max_fwhm = 3 * window if max_fwhm is None else max_fwhm
    t0_steps = np.linspace(time_arr.min(), time_arr.max(), round(window * 24))
    t0_grid = np.union1d(t0_steps, time_arr)
    fwhm_grid = np.geomspace(min_fwhm, max_fwhm, 40)
    t0_step = t0_steps[1] - t0_steps[0]
    return t0_grid, fwhm_grid, t0_step


def ts_surface(time_arr, flux_arr, err_arr, t0_grid, fwhm_grid, window=12.0):
    """
    Takes in time, flux, error arrays along with grids of flare onsets (t0)
    and durations (fwhm), and additional flare width penalty. Uses these to
    create time offset and model profile grids. Calculates a delta chisq
    surface map over the 2D grid of flare parameters to be used for a test
    statistic fit.

    Arguments
    ---------
    time_arr, flux_arr, err_ar : array_like
        Arrays containing the timestamps (in days), flux (in mJy) and
        errors (mJy) for both bands. Taken from format_data().
    t0_grid, fwhm_grid : int
        Half-width of thumbnail size, in pixels, if you do not want
        to run the fitter over the full size input maps.
    window : float
        Maximum duration of light curve to be fit, in days. Defaults
        to 12.

    Returns
    -------
    surface_map : array_like
        TS surface map over our flare parameter grid (t0, fwhm) showing the TS
        value of each fit.
    """

    time_offset = time_arr[None, None, :] - t0_grid[:, None, None]
    # FWHM expression of gaussian instead of sigma, since we want to return FWHM
    model_profiles = np.exp(
        -(4 * np.log(2)) * time_offset**2 / fwhm_grid[None, :, None] ** 2
    )
    dchi2_grid = np.zeros((len(t0_grid), len(fwhm_grid)))

    # least-squares fitting over the two bands
    for band in range(2):
        inverse_variance = 1 / err_arr[band] ** 2
        weighted_model = model_profiles @ (flux_arr[band] * inverse_variance)
        model_normalization = (model_profiles**2) @ inverse_variance + 1e-12
        dchi2_grid += weighted_model**2 / model_normalization

    # include flare width penalty term
    surface_map = dchi2_grid + 2 * np.log(fwhm_grid[None, :] / window)

    return surface_map


def peak_amplitudes(time_arr, flux_arr, err_arr, t0, fwhm):
    """
    Finds the peak amplitude of the best-fit Gaussian, since we are only
    fitting for t0 and FWHM.

    Arguments
    ---------
    time_arr, flux_arr, err_arr : array_like
        Time, flux and error arrays as formatted by format_data().
    t0, fwhm : float
        The peak time and FWHM of our Gaussian best-fit.

    Returns
    -------
    amps, amp_errs : array_like
        Peak amplitude and error of the best-fit Gaussians in both frequency
        bands.
    """
    profile = np.exp(-(4 * np.log(2)) * (time_arr - t0) ** 2 / fwhm**2)
    amps = np.zeros(2)
    amp_errs = np.zeros(2)
    for band in range(2):
        inverse_variance = 1 / err_arr[band] ** 2
        norm = (profile**2) @ inverse_variance + 1e-12
        amps[band] = (profile @ (flux_arr[band] * inverse_variance)) / norm
        amp_errs[band] = 1 / np.sqrt(norm)

    return amps, amp_errs


def ts_fit(time_arr, flux_arr, err_arr, window=12.0, start=None):
    """
    Takes in time, flux, and error arrays for a given pixel's light curve and returns
    the best-fit (highest TS) Gaussian for the potential event. Before running the
    Nelder-Mead simplex aglorithm, initalizes the parameters by doing a coarse grid search
    to reduce fit time. Optionally takes best-fit parameters for the purpose of
    comparing neighboring pixels to the known solution, as pixels should share the
    same temporal properties of an event.

    Arguments
    ---------
    time_arr, flux_arr, err_arr : array_like
        Time, flux and error arrays as formatted by format_data().
    window : float
        Maximum duration of light curve to be fit, in days. Defaults
        to 12.
    start :  2-tuple, optional
        (t0, fwhm) of a known solution, used for refining the search on neighboring
        pixels.

    Returns
    -------
    TS, t0, fwhm :  float
        Test Statistic and 2-parameter solution of best-fit Gaussian for the light
        curve's potential event.
    amplitudes, amplitude_errors : array_like
        Peak amplitudes and errors for the best-fit Gaussian in both frequency bands.
    surface_map : array_like
        TS surface map over our flare parameter grid (t0, fwhm) showing the TS
        value of each fit.
    """
    min_fwhm = 5 / 60 / 24  # 5 min in days
    max_fwhm = window * 3  # explained below
    t0_grid, fwhm_grid, t0_step = make_grids(time_arr, window, min_fwhm, max_fwhm)

    if start is None:
        surface_map = ts_surface(
            time_arr, flux_arr, err_arr, t0_grid, fwhm_grid, window
        )

        if not np.isfinite(surface_map).all():
            raise ValueError("check flux/err for NaN or zero errors")

        # find grid coordinates with highest dchi2
        best_t0_idx, best_fwhm_idx = np.unravel_index(
            np.argmax(surface_map), surface_map.shape
        )
        best_t0 = t0_grid[best_t0_idx]
        best_log_fwhm = np.log(fwhm_grid[best_fwhm_idx])

    else:
        # start the simplex from a known solution (e.g. brightest pixel's fit)
        surface_map = None
        best_t0, best_log_fwhm = start

    # first we'll put bounds on the parameter search. for t0 we'll extend the
    # parameters by 6 days in either direction (half the window search). fwhm
    # is a little bit trickier: a meaningful FWHM outside the range of the window
    # (12 days) will depend on SNR of the points, i.e. fwhm_max = (window/2) * sqrt(4*ln(2)*SNR)
    # we could go through our years of data and come up with the optimal SNR
    # for typical events, which should be fixed for all searches to not invalidate
    # the results. instead we'll use a reasonable upper limit FWHM of 36 days (3x window)
    # which corresponds to a SNR of ~13.
    #### IMPORTANT to note: we don't actually care about the physical duration of the
    # flare, we're only using it for a best fit. going beyond that duration probably
    # won't matter as it will already score high enough to extremely obvious
    ### TODO: is the above actually true? we use TS for localization, SNR estimates etc.
    # so we might actually want the most accurate TS possible. find out how much
    # this will change for a very long duration, bright flare. alternatively, use this quick
    # method to find Generic Transients, and long duration analyses can do a more in-depth search.

    extend_times = window / 2
    extend_times = 0  # debug
    lower = [time_arr.min() - (extend_times), np.log(min_fwhm)]
    upper = [time_arr.max() + (extend_times), np.log(max_fwhm)]
    bounds = list(zip(lower, upper))

    # next we'll optimize the step size of the nelder-mead simplex to relate to our parameter grids
    # we need to step inward if a corner leaves the grid bounds, so the simplex
    # will stay non-degenerate
    dt = t0_step if best_t0 + t0_step <= upper[0] else -t0_step
    log_fwhm_step = np.log(fwhm_grid[1] / fwhm_grid[0])
    dfwhm = (
        log_fwhm_step if best_log_fwhm + log_fwhm_step <= upper[1] else -log_fwhm_step
    )
    simplex = np.array(
        [
            [best_t0, best_log_fwhm],
            [best_t0 + dt, best_log_fwhm],
            [best_t0, best_log_fwhm + dfwhm],
        ]
    )

    # define objective function for fine tuning
    # neg_ts finally returns a starting guess for flare parameters, we refine
    # using the nelder-mead optimizer to fine tune our parameter search
    def neg_ts(model_params):
        test_t0, test_log_fwhm = model_params
        return -ts_surface(
            time_arr,
            flux_arr,
            err_arr,
            np.array([test_t0]),
            np.array([np.exp(test_log_fwhm)]),
            window,
        )[0, 0]

    res = minimize(
        neg_ts,
        x0=simplex[0],
        method="Nelder-Mead",
        bounds=bounds,
        options={"initial_simplex": simplex},
    )
    TS = -res.fun
    t0 = res.x[0]
    fwhm = np.exp(res.x[1])
    amplitudes, amplitude_errors = peak_amplitudes(
        time_arr, flux_arr, err_arr, t0, fwhm
    )
    return TS, t0, fwhm, amplitudes, amplitude_errors, surface_map


def ts_fit_map(
    time_arr, flux_arr, err_arr, window=12.0, min_good=6, check_brightest_pixel=False
):
    """
    Runs ts_fit on every pixel of the thumbnail to complete a full TS map.

    Arguments
    ---------
    time_arr, flux_arr, err_arr : array_like
        Time, flux and error arrays as formatted by format_data().
    window : float
        Maximum duration of light curve to be fit, in days. Defaults
        to 12.
    min_good : int
        The minimum number of good data points within the light curve to fit to.
    check_brightest_pixel : bool, optional
        Optional second pass on fitting to each pixel, based on the brightest pixel's
        solution. Currently implemented but not thoroughly tested for how much this helps
        in the "pixelation" issue seen in the GSL Nelder-Mead implementation. When run
        on a 41x41 thumbnail, increases runtime from ~10s to ~15s.

    Returns
    -------
    output : dict
        Dictionary containing 2D arrays of TS, t0, FWHM, amps, errors for each pixel
        corresponding to the input thumbnails. Additionally contains the surface map
        showing the TS values over the parameter grid for the central pixel, used for
        diagnostics.
    """
    ny, nx = flux_arr.shape[2:]
    center_pix = ny // 2
    ts_map = np.full((ny, nx), np.nan)
    t0_map = np.full((ny, nx), np.nan)
    fwhm_map = np.full((ny, nx), np.nan)
    amps_map = np.full((2, ny, nx), np.nan)
    amp_errs_map = np.full((2, ny, nx), np.nan)

    def store(y, x, TS, t0, fwhm, amps, amp_errs):
        ts_map[y, x], t0_map[y, x], fwhm_map[y, x] = TS, t0, fwhm
        amps_map[:, y, x], amp_errs_map[:, y, x] = amps, amp_errs

    # skip pixels without enough usable observations (e.g. zero weight at
    # the map edge)
    pixels = [
        (y, x)
        for y in range(ny)
        for x in range(nx)
        if np.isfinite(err_arr[:, :, y, x]).sum() >= min_good
    ]

    # first pass: every pixel gets its own grid search + nelder-mead polish
    for y, x in pixels:
        TS, t0, fwhm, amps, amp_errs, pixel_surface_map = ts_fit(
            time_arr, flux_arr[:, :, y, x], err_arr[:, :, y, x], window
        )
        if y == x == center_pix:
            center_pix_surface_map = pixel_surface_map
        store(y, x, TS, t0, fwhm, amps, amp_errs)

    # optional second pass: a point source has the same light curve shape in
    # every nearby pixel, so retry every pixel from the brightest pixel's (t0,
    # fwhm) and keep the better fit
    if check_brightest_pixel:
        peak_y, peak_x = np.unravel_index(np.nanargmax(ts_map), ts_map.shape)
        start = (t0_map[peak_y, peak_x], np.log(fwhm_map[peak_y, peak_x]))
        for y, x in pixels:
            TS, t0, fwhm, amps, amp_errs, pixel_surface_map = ts_fit(
                time_arr, flux_arr[:, :, y, x], err_arr[:, :, y, x], window, start=start
            )
            if y == x == center_pix:
                center_pix_surface_map = pixel_surface_map
            if TS > ts_map[y, x]:
                store(y, x, TS, t0, fwhm, amps, amp_errs)

    output = {
        "TestStatisticMap": ts_map,
        "t0": t0_map,
        "FWHM": fwhm_map,
        "amps": amps_map,
        "amp_errs": amp_errs_map,
        "surface_map": center_pix_surface_map,
    }

    return output
