"""
Temperature-scaling calibration for letter-readout probabilities.

Fits one temperature T per task on calibration data to minimize
negative log-likelihood (NLL). Also computes ECE (Expected Calibration Error)
and derives precision/recall-balanced decision thresholds.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .readout import TASKS, CommentClassifier, TaskConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calibrate_probs(p: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to probability distribution vector."""
    if T <= 0:
        T = 1.0
    q = p ** (1.0 / T)
    s = q.sum()
    if s <= 0:
        return np.ones_like(p) / len(p)
    return q / s


def fit_temperature(
    prob_list: List[np.ndarray],
    labels: List[int],
    bounds: Tuple[float, float] = (0.3, 8.0),
) -> float:
    """
    Fit optimal calibration temperature on calibration data by minimizing NLL.
    """
    def nll(T: float) -> float:
        total = 0.0
        for p, y in zip(prob_list, labels):
            cal = calibrate_probs(p, T)
            p_y = max(float(cal[y]), 1e-12)
            total -= math.log(p_y)
        return total / len(labels)

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    logger.info("Fitted T=%.4f (val NLL=%.4f)", res.x, res.fun)
    return float(res.x)


def expected_calibration_error(
    probs: List[np.ndarray],
    labels: List[int],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
    if not probs:
        return 0.0

    confidences = np.array([p.max() for p in probs])
    predictions = np.array([p.argmax() for p in probs])
    labels_arr = np.array(labels)
    accuracies = (predictions == labels_arr).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_count = int(mask.sum())
        if bin_count > 0:
            bin_acc = float(accuracies[mask].mean())
            bin_conf = float(confidences[mask].mean())
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def fit_thresholds(
    probs: List[np.ndarray],
    labels: List[int],
    options: List[str],
    target_recall: float = 0.85,
) -> Dict[str, float]:
    """
    Fit per-class thresholds to achieve target recall on calibration data.
    """
    thresholds = {}
    n_classes = len(options)

    for cls_idx in range(n_classes):
        cls_name = options[cls_idx]
        # Calibrated probabilities when the true label is this class
        cls_probs = [float(p[cls_idx]) for p, y in zip(probs, labels) if y == cls_idx]

        if not cls_probs:
            thresholds[cls_name] = 0.45
            continue

        sorted_probs = sorted(cls_probs)
        cutoff_idx = max(0, int((1.0 - target_recall) * len(sorted_probs)))
        thresh_val = round(float(sorted_probs[cutoff_idx]), 4)
        # Cap between 0.25 and 0.80
        thresholds[cls_name] = max(0.25, min(0.80, thresh_val))

    return thresholds


def run_calibration(
    classifier: CommentClassifier,
    calib_csv: str | Path = "data/calib.csv",
    task_name: str = "sentiment",
    config_dir: str | Path = "config",
    max_samples: int | None = None,
) -> Dict[str, float]:
    """
    Execute end-to-end calibration for a single task.
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name}. Available: {list(TASKS.keys())}")

    task = TASKS[task_name]
    df = pd.read_csv(calib_csv)
    if max_samples is not None:
        df = df.iloc[:max_samples].copy()

    # Filter rows that have valid labels
    label_map = {opt: i for i, opt in enumerate(task.options)}
    valid_rows = df[df[task_name].isin(label_map.keys())].copy()

    if len(valid_rows) < 10:
        logger.warning(
            "Task %s has only %d labeled rows in %s. Calibration requires at least 10 rows.",
            task_name,
            len(valid_rows),
            calib_csv,
        )
        return {"temperature": 1.0, "ece_before": 0.0, "ece_after": 0.0}

    logger.info("Calibrating %s on %d samples...", task_name, len(valid_rows))
    prob_list = []
    labels = []

    for _, row in valid_rows.iterrows():
        p = classifier.letter_probs(str(row["text"]), task, use_cache=True)
        prob_list.append(p)
        labels.append(label_map[row[task_name]])

    # Calculate ECE before
    ece_before = expected_calibration_error(prob_list, labels)

    # Fit temperature
    T = fit_temperature(prob_list, labels)

    # Apply calibration
    calibrated_probs = [calibrate_probs(p, T) for p in prob_list]

    # Calculate ECE after
    ece_after = expected_calibration_error(calibrated_probs, labels)

    # Fit thresholds
    thresholds = fit_thresholds(calibrated_probs, labels, task.options)

    # Save to config files
    cfg_dir = Path(config_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    temp_file = cfg_dir / "temperatures.json"
    thresh_file = cfg_dir / "thresholds.json"

    temps = {}
    if temp_file.exists():
        try:
            temps = json.loads(temp_file.read_text(encoding="utf-8"))
        except Exception:
            temps = {}

    threshs = {}
    if thresh_file.exists():
        try:
            threshs = json.loads(thresh_file.read_text(encoding="utf-8"))
        except Exception:
            threshs = {}

    temps[task_name] = {
        "T": round(T, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
    }
    threshs[task_name] = thresholds

    temp_file.write_text(json.dumps(temps, indent=2), encoding="utf-8")
    thresh_file.write_text(json.dumps(threshs, indent=2), encoding="utf-8")

    logger.info(
        "Saved calibration for %s: T=%.3f, ECE: %.4f -> %.4f",
        task_name,
        T,
        ece_before,
        ece_after,
    )

    return {
        "temperature": T,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "thresholds": thresholds,
    }


def calibrate_all_tasks(
    classifier: CommentClassifier,
    calib_csv: str | Path = "data/calib.csv",
    config_dir: str | Path = "config",
    max_samples: int | None = None,
) -> Dict[str, dict]:
    """Calibrate all registered tasks."""
    results = {}
    for task_name in TASKS:
        res = run_calibration(
            classifier,
            calib_csv=calib_csv,
            task_name=task_name,
            config_dir=config_dir,
            max_samples=max_samples,
        )
        results[task_name] = res
    return results


if __name__ == "__main__":
    clf = CommentClassifier()
    calibrate_all_tasks(clf)
