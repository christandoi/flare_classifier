"""
Unit tests for evaluation.py and the window helper in read_spt3g_fits.py.

Each test uses a tiny hand-made input whose correct answer can be worked out
by hand, so no data files are needed. Run with `pytest` from the repo root.
"""

import numpy as np
import polars as pl
import pytest

import evaluation as ev
import read_spt3g_fits as spt3g


def test_recall_at_fpr_zero_uses_highest_noise_score():
    # threshold = highest noise score (3); positives strictly above it: 3.5 and 4
    positives = np.array([2.5, 3.5, 4.0])
    noise = np.array([0.0, 1.0, 2.0, 3.0])
    assert ev.recall_at_fpr(positives, noise, 0) == pytest.approx(2 / 3)


def test_recall_at_fpr_lets_through_round_fpr_times_n():
    # fpr 0.25 of 4 noise windows lets 1 through: threshold = second highest (2)
    positives = np.array([2.5, 3.5, 4.0])
    noise = np.array([0.0, 1.0, 2.0, 3.0])
    assert ev.recall_at_fpr(positives, noise, 0.25) == pytest.approx(1.0)


def test_recall_matrix_shape_and_values():
    positives = np.array([[2.5, 0.0], [3.5, 0.0], [4.0, 5.0]])
    noise = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    recalls = ev.recall_matrix(positives, noise, [0, 0.25])
    expected = np.array([[2 / 3, 1 / 3], [1.0, 1 / 3]])
    assert recalls.shape == (2, 2)
    np.testing.assert_allclose(recalls, expected)


def test_bootstrap_recall_is_reproducible_and_matches_full_recall():
    positives = pl.DataFrame(
        {'gaia_id': ['a', 'a', 'b', 'c'], 'score': [2.0, 5.0, 4.0, 1.0]}
    )
    noise = pl.DataFrame(
        {'gaia_id': ['a', 'b', 'c', 'c'], 'score': [1.0, 3.0, 0.0, 2.0]}
    )
    full_1, boot_1 = ev.bootstrap_recall(
        positives, noise, ['score'], fpr_levels=(0,), n_boot=20, seed=100
    )
    full_2, boot_2 = ev.bootstrap_recall(
        positives, noise, ['score'], fpr_levels=(0,), n_boot=20, seed=100
    )
    # the point estimate is the recall on all rows: threshold 3, so 5 and 4 pass
    assert full_1[0, 0] == pytest.approx(0.5)
    assert boot_1.shape == (20, 1, 1)
    np.testing.assert_array_equal(boot_1, boot_2)
    np.testing.assert_array_equal(full_1, full_2)


def test_binned_completeness():
    detected = np.array([True, False, True, True])
    x = np.array([0.5, 0.5, 1.5, 1.5])
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    fraction, error = ev.binned_completeness(detected, x, edges)
    np.testing.assert_allclose(fraction[:2], [0.5, 1.0])
    np.testing.assert_allclose(error[:2], [np.sqrt(0.25 / 2), 0.0])
    # the empty last bin is NaN, not zero
    assert np.isnan(fraction[2]) and np.isnan(error[2])


def test_snr_at_half_interpolates_in_log():
    # halfway between 0.2 and 0.8 is halfway in log between 1 and 4, i.e. 2
    half = ev.snr_at_half(np.array([0.2, 0.8]), np.array([1.0, 4.0]))
    assert half == pytest.approx(2.0)


def test_snr_at_half_is_nan_without_a_crossing():
    assert np.isnan(ev.snr_at_half(np.array([0.6, 0.8]), np.array([1.0, 4.0])))


def test_binned_difference_paired_error():
    # a finds 2 that b misses, b finds 0 that a misses: difference 2/4,
    # paired error sqrt(2 + 0 - (2 - 0)**2 / 4) / 4 = 0.25
    detected_a = np.array([True, True, True, False])
    detected_b = np.array([True, False, False, False])
    difference, error, counts = ev.binned_difference(
        detected_a, detected_b, np.zeros(4), [-1.0, 1.0], min_count=1
    )
    assert difference[0] == pytest.approx(0.5)
    assert error[0] == pytest.approx(0.25)
    assert counts[0] == 4


def test_binned_difference_drops_small_bins():
    detected = np.array([True, False, True, False])
    difference, error, counts = ev.binned_difference(
        detected, detected, np.zeros(4), [-1.0, 1.0], min_count=20
    )
    assert np.isnan(difference[0]) and np.isnan(error[0])
    assert counts[0] == 4


def test_ts_scores_applies_cuts_in_order():
    windows = pl.DataFrame(
        {
            'ts': [50.0, 40.0, 30.0, 60.0],
            'fwhm_d': [1.0, 1.0, 12.0, 2.0],
            'snr_amp_95': [3.0, 3.0, 3.0, 3.0],
            'snr_amp_150': [3.0, -3.0, 3.0, 3.0],
            'n_3sigma_both': [1, 1, 1, 0],
        }
    )
    scores = ev.ts_scores(windows)
    # row 0 passes everything; row 1 has a negative amplitude; row 2 fails
    # FWHM < 10 d; row 3 fails only the pixel significance cut
    assert scores['ts_positive_score'].to_list() == [50.0, -1e9, 30.0, 60.0]
    assert scores['ts_positive_fwhm_score'].to_list() == [50.0, -1e9, -1e9, 60.0]
    all_cuts = scores['ts_positive_fwhm_pixel_score'].to_list()
    assert all_cuts == [50.0, -1e9, -1e9, -1e9]


def test_bin_series_is_closed_on_the_right():
    bins = ev.bin_series([0.5, 1.0, 2.0, 5.0], [1, 3], 'bin')
    assert bins.to_list() == ['(-inf, 1]', '(-inf, 1]', '(1, 3]', '(3, inf]']
    assert bins.dtype == pl.Enum(['(-inf, 1]', '(1, 3]', '(3, inf]'])


def test_window_starts_two_windows_per_time():
    # 12-day windows every 6 days from 0: day 10 is in the windows starting at 0 and 6
    np.testing.assert_allclose(spt3g.window_starts(10.0, 0.0), [0.0, 6.0])
