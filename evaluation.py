import numpy as np
import polars as pl
from tqdm.auto import tqdm


def recall_at_fpr(positive_scores, noise_scores, fpr):
    """
    Recall on positives when the detection threshold lets through
    round(fpr * n_noise) of the noise windows. At fpr = 0 the threshold is
    the highest noise score, so no noise window is detected.

    Arguments
    ---------
    positive_scores : array_like
        Scores of the positives (observable injections).
    noise_scores : array_like
        Scores of the noise reference (mirrored negatives).
    fpr : float
        Allowed false-positive rate on the noise reference.

    Returns
    -------
    recall : float
        Fraction of positives scoring strictly above the threshold.
    """
    n_allowed = int(round(fpr * len(noise_scores)))
    threshold = np.sort(noise_scores)[::-1][n_allowed]
    return (positive_scores > threshold).mean()


def recall_matrix(positive_matrix, noise_matrix, fpr_levels):
    """
    recall_at_fpr() for every method (column) at every FPR level.

    Arguments
    ---------
    positive_matrix, noise_matrix : array_like
        Shape (n_rows, n_methods); column j holds method j's scores.
    fpr_levels : list of float
        False-positive rates to evaluate.

    Returns
    -------
    recalls : array_like
        Shape (n_fpr_levels, n_methods).
    """
    return np.array([[recall_at_fpr(positive_matrix[:, j], noise_matrix[:, j], fpr)
                      for j in range(positive_matrix.shape[1])] for fpr in fpr_levels])


def bootstrap_recall(positives, noise, methods, fpr_levels=(0, 1e-4, 1e-3),
                     n_boot=1000, seed=100):
    """
    Recall at fixed noise FPR with bootstrap intervals, resampling whole
    stars (with replacement) so the intervals include star-to-star
    variation. Positives and noise of a sampled star are both taken, and
    every method is evaluated on the same replicates, so differences between
    methods can be bootstrapped as pairs.

    The replicates depend only on the seed and the list of stars, not on
    which methods are passed, so adding a method leaves the other methods'
    results unchanged.

    Arguments
    ---------
    positives : polars.DataFrame
        Positives (observable injections), with gaia_id and one score
        column per method.
    noise : polars.DataFrame
        Noise reference (mirrored negatives), same columns.
    methods : list of str
        Score columns to evaluate.
    fpr_levels : tuple of float
        False-positive rates. Defaults to (0, 1e-4, 1e-3).
    n_boot : int
        Bootstrap replicates. Defaults to 1000.
    seed : int
        Random seed. Defaults to 100.

    Returns
    -------
    full_recall : array_like
        Point estimates, shape (n_fpr_levels, n_methods).
    boot_recall : array_like
        Replicates, shape (n_boot, n_fpr_levels, n_methods).
    """
    stars = np.unique(np.concatenate([positives['gaia_id'].to_numpy(), noise['gaia_id'].to_numpy()]))
    positives_by_star = {g: positives.filter(pl.col('gaia_id') == g).select(methods).to_numpy() for g in stars}
    noise_by_star = {g: noise.filter(pl.col('gaia_id') == g).select(methods).to_numpy() for g in stars}

    full_recall = recall_matrix(np.vstack(list(positives_by_star.values())),
                                np.vstack(list(noise_by_star.values())), fpr_levels)

    rng = np.random.default_rng(seed)
    boot_recall = np.empty((n_boot, len(fpr_levels), len(methods)))
    for b in tqdm(range(n_boot)):
        sampled_stars = rng.choice(stars, size=len(stars), replace=True)
        boot_recall[b] = recall_matrix(np.vstack([positives_by_star[g] for g in sampled_stars]),
                                       np.vstack([noise_by_star[g] for g in sampled_stars]),
                                       fpr_levels)
    return full_recall, boot_recall


def recall_table(full_recall, boot_recall, methods, fpr_levels=(0, 1e-4, 1e-3)):
    """
    Recall with 95% bootstrap intervals, one row per FPR level and method.

    Arguments
    ---------
    full_recall, boot_recall : array_like
        Output of bootstrap_recall().
    methods : list of str
        Score columns, in the order used for bootstrap_recall().
    fpr_levels : tuple of float
        FPR levels, in the order used for bootstrap_recall().

    Returns
    -------
    table : polars.DataFrame
        fpr, method (without the _score suffix), recall, ci_low, ci_high.
    """
    rows = []
    for i, fpr in enumerate(fpr_levels):
        for j, method in enumerate(methods):
            low, high = np.percentile(boot_recall[:, i, j], [2.5, 97.5])
            rows.append({'fpr': fpr, 'method': method.replace('_score', ''),
                         'recall': full_recall[i, j], 'ci_low': low, 'ci_high': high})
    return pl.DataFrame(rows)


def paired_difference(full_recall, boot_recall, methods, method_a, method_b,
                      fpr_levels=(0, 1e-4, 1e-3)):
    """
    Recall difference method_a - method_b with a paired bootstrap interval:
    the difference is taken within each replicate, so star-to-star
    variation shared by both methods cancels.

    Arguments
    ---------
    full_recall, boot_recall : array_like
        Output of bootstrap_recall().
    methods : list of str
        Score columns, in the order used for bootstrap_recall().
    method_a, method_b : str
        The two score columns to compare.
    fpr_levels : tuple of float
        FPR levels, in the order used for bootstrap_recall().

    Returns
    -------
    table : polars.DataFrame
        fpr, difference, ci_low, ci_high, fraction_positive (fraction of
        replicates where method_a has the higher recall).
    """
    a, b = methods.index(method_a), methods.index(method_b)
    rows = []
    for i, fpr in enumerate(fpr_levels):
        difference = boot_recall[:, i, a] - boot_recall[:, i, b]
        low, high = np.percentile(difference, [2.5, 97.5])
        rows.append({'fpr': fpr, 'difference': full_recall[i, a] - full_recall[i, b],
                     'ci_low': low, 'ci_high': high,
                     'fraction_positive': (difference > 0).mean()})
    return pl.DataFrame(rows)


def ts_scores(windows):
    """
    TS as a detection score under three versions of the pipeline's
    selection cuts: TS where the window passes the cuts, -1e9 otherwise
    (-1e9 rather than -inf because some scikit-learn metrics reject inf).

      ts_positive_score            : both fitted amplitudes positive
      ts_positive_fwhm_score       : + FWHM < 10 d
      ts_positive_fwhm_pixel_score : + pixel significance cut (at least one
                                     observation above 3 sigma in both bands);
                                     the full pipeline cuts (old ts_score)

    Arguments
    ---------
    windows : polars.DataFrame
        Windows with ts, fwhm_d, snr_amp_95, snr_amp_150, n_3sigma_both.

    Returns
    -------
    windows : polars.DataFrame
        The input plus the three score columns.
    """
    positive = (pl.col('snr_amp_95') > 0) & (pl.col('snr_amp_150') > 0)
    fwhm = pl.col('fwhm_d') < 10
    pixel = pl.col('n_3sigma_both') >= 1
    return windows.with_columns(
        pl.when(positive).then(pl.col('ts')).otherwise(-1e9).alias('ts_positive_score'),
        pl.when(positive & fwhm).then(pl.col('ts')).otherwise(-1e9).alias('ts_positive_fwhm_score'),
        pl.when(positive & fwhm & pixel).then(pl.col('ts')).otherwise(-1e9).alias('ts_positive_fwhm_pixel_score'),
    )


def bin_series(values, edges, name):
    """
    Labels each value with the bin it falls in, as an ordered categorical
    (polars Enum) so group_by results sort in bin order. Bins are open on the
    left and closed on the right, (a, b], with open-ended bins below the
    first edge and above the last, the same convention as the deprecated
    polars Expr.cut().

    Arguments
    ---------
    values : array_like
        Values to bin (no nulls; filter them out first).
    edges : list of float
        Increasing bin edges.
    name : str
        Name of the returned series.

    Returns
    -------
    bins : polars.Series
        One label per value, e.g. '(1, 3]', '(-inf, 0]' or '(20, inf]'.
    """
    edges = list(edges)
    labels = ([f"(-inf, {edges[0]:g}]"]
              + [f"({low:g}, {high:g}]" for low, high in zip(edges[:-1], edges[1:])]
              + [f"({edges[-1]:g}, inf]"])
    index = np.searchsorted(edges, np.asarray(values, dtype=float), side='left')
    return pl.Series(name, np.array(labels)[index], dtype=pl.Enum(labels))


def binned_completeness(detected, x, edges):
    """
    Fraction detected per bin of x, with binomial standard errors. Bins are
    [a, b) as in np.digitize; values outside the edges are ignored.

    Arguments
    ---------
    detected : array_like of bool
        Whether each injection was detected.
    x : array_like
        Value to bin by (e.g. observed SNR), same length as detected.
    edges : array_like
        Increasing bin edges.

    Returns
    -------
    fraction, error : array_like
        Detected fraction and its binomial standard error per bin (NaN for
        empty bins).
    """
    index = np.digitize(x, edges) - 1
    fraction = np.full(len(edges) - 1, np.nan)
    error = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        in_bin = index == i
        n = in_bin.sum()
        if n > 0:
            fraction[i] = detected[in_bin].mean()
            error[i] = np.sqrt(fraction[i] * (1 - fraction[i]) / n)
    return fraction, error


def snr_at_half(fraction, centers):
    """
    Where binned completeness first reaches 50%, interpolated linearly in
    log(centers) between the two bins that bracket it.

    Arguments
    ---------
    fraction : array_like
        Binned completeness, e.g. from binned_completeness().
    centers : array_like
        Bin centers (geometric centers for log bins).

    Returns
    -------
    half : float
        The 50% crossing, or NaN if completeness never crosses 50%.
    """
    for i in range(len(fraction) - 1):
        if fraction[i] < 0.5 <= fraction[i + 1]:
            weight = (0.5 - fraction[i]) / (fraction[i + 1] - fraction[i])
            return np.exp(np.log(centers[i]) + weight * (np.log(centers[i + 1]) - np.log(centers[i])))
    return np.nan


def binned_difference(detected_a, detected_b, x, edges, min_count=20):
    """
    Completeness difference (method a - method b) per bin of x, on the same
    injections, with a paired standard error: only injections one method
    finds and the other misses add variance, so shared detections and shared
    misses cancel. Bins are [a, b) as in np.digitize.

    Arguments
    ---------
    detected_a, detected_b : array_like of bool
        Whether each injection was detected by method a and by method b.
    x : array_like
        Value to bin by, same length as the detection arrays.
    edges : array_like
        Increasing bin edges (may include -np.inf / np.inf).
    min_count : int
        Bins with fewer injections are returned as NaN. Defaults to 20.

    Returns
    -------
    difference, error : array_like
        Completeness difference and its paired standard error per bin.
    counts : array_like
        Number of injections per bin.
    """
    detected_a, detected_b = np.asarray(detected_a), np.asarray(detected_b)
    index = np.digitize(x, edges) - 1
    n_bins = len(edges) - 1
    difference, error = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        in_bin = index == i
        n = in_bin.sum()
        counts[i] = n
        if n >= min_count:
            a_only = (detected_a[in_bin] & ~detected_b[in_bin]).sum()
            b_only = (~detected_a[in_bin] & detected_b[in_bin]).sum()
            difference[i] = (a_only - b_only) / n
            error[i] = np.sqrt(a_only + b_only - (a_only - b_only) ** 2 / n) / n
    return difference, error, counts
