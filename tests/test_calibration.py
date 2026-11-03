import numpy as np
import pytest
from src.calibrate import calibrate_probs, expected_calibration_error, fit_temperature


def test_calibrate_probs_scaling():
    p = np.array([0.7, 0.2, 0.1])

    # T = 1.0 (identity)
    cal_1 = calibrate_probs(p, 1.0)
    assert np.allclose(p, cal_1)

    # T > 1 (softening / increasing entropy)
    cal_warm = calibrate_probs(p, 2.0)
    assert cal_warm[0] < p[0]
    assert np.isclose(cal_warm.sum(), 1.0)

    # T < 1 (sharpening / decreasing entropy)
    cal_cold = calibrate_probs(p, 0.5)
    assert cal_cold[0] > p[0]
    assert np.isclose(cal_cold.sum(), 1.0)


def test_expected_calibration_error():
    # Perfectly calibrated case: confidence = accuracy
    probs = [np.array([0.9, 0.1])] * 9 + [np.array([0.1, 0.9])] * 1
    labels = [0] * 9 + [1] * 1

    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert ece < 0.15


def test_fit_temperature_optimization():
    # Overconfident predictions
    prob_list = [
        np.array([0.95, 0.05]),
        np.array([0.95, 0.05]),
        np.array([0.05, 0.95]),
        np.array([0.95, 0.05]),
    ]
    labels = [0, 1, 1, 0]  # Only 75% accuracy despite 95% confidence

    best_T = fit_temperature(prob_list, labels, bounds=(0.5, 5.0))
    # Optimal T should be > 1 to soften the overconfident probabilities
    assert best_T > 1.0
