"""
Evaluation module: macro-F1, per-class metrics, confusion matrix,
ECE, reliability scores, and latency benchmarks.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .calibrate import calibrate_probs, expected_calibration_error
from .readout import TASKS, CommentClassifier, TaskConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_task(
    classifier: CommentClassifier,
    data_csv: str | Path,
    task_name: str,
    calibration_T: float = 1.0,
    thresholds: Optional[Dict[str, float]] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate classifier on a dataset for one task.
    """
    task = TASKS[task_name]
    df = pd.read_csv(data_csv)
    if max_samples is not None:
        df = df.iloc[:max_samples].copy()

    label_map = {opt: i for i, opt in enumerate(task.options)}
    valid_rows = df[df[task_name].isin(label_map.keys())].copy()

    if len(valid_rows) == 0:
        logger.warning("No labeled rows for %s in %s", task_name, data_csv)
        return {"task": task_name, "n_samples": 0}

    true_labels = []
    pred_labels = []
    prob_list = []
    latencies = []

    for _, row in valid_rows.iterrows():
        true_idx = label_map[row[task_name]]
        true_labels.append(true_idx)

        t0 = time.perf_counter()
        raw_p = classifier.letter_probs(str(row["text"]), task, use_cache=False)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        cal_p = calibrate_probs(raw_p, calibration_T)
        prob_list.append(cal_p)
        pred_labels.append(int(np.argmax(cal_p)))

    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)
    latencies_arr = np.array(latencies)

    macro_f1 = f1_score(true_arr, pred_arr, average="macro", zero_division=0)
    report = classification_report(
        true_arr,
        pred_arr,
        target_names=[task.options[i] for i in sorted(set(true_labels) | set(pred_labels))],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(true_arr, pred_arr).tolist()
    ece = expected_calibration_error(prob_list, true_labels)

    return {
        "task": task_name,
        "dataset": str(Path(data_csv).name),
        "n_samples": len(valid_rows),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": report,
        "confusion_matrix": cm,
        "ece": round(float(ece), 4),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50) * 1000), 1),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95) * 1000), 1),
        "latency_mean_ms": round(float(latencies_arr.mean() * 1000), 1),
    }


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate a clean markdown report from evaluation results."""
    lines = [
        "# Baseline Evaluation Report",
        "",
        "Evaluation of calibrated one-token letter readout on social media comments.",
        "",
    ]

    for task_name, datasets in results.items():
        lines.append(f"## Task: {task_name.upper()}")
        lines.append("")

        for ds_name, m in datasets.items():
            if m.get("n_samples", 0) == 0:
                continue

            lines.append(f"### Dataset: `{ds_name}` (N = {m['n_samples']})")
            lines.append("")
            lines.append(f"- **Macro-F1**: `{m['macro_f1']:.4f}`")
            lines.append(f"- **ECE**: `{m['ece']:.4f}`")
            lines.append(f"- **Latency (p50 / p95)**: `{m['latency_p50_ms']} ms` / `{m['latency_p95_ms']} ms`")
            lines.append("")

            # Per-class table
            lines.append("| Class | Precision | Recall | F1-Score | Support |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for cls_k, cls_v in m["per_class"].items():
                if cls_k in ("accuracy", "macro avg", "weighted avg"):
                    continue
                lines.append(
                    f"| {cls_k} | {cls_v['precision']:.3f} | {cls_v['recall']:.3f} | {cls_v['f1-score']:.3f} | {cls_v['support']} |"
                )
            lines.append("")

            # Confusion matrix
            lines.append("**Confusion Matrix:**")
            lines.append("```")
            for row in m["confusion_matrix"]:
                lines.append("  ".join(f"{val:4d}" for val in row))
            lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def full_evaluation(
    classifier: CommentClassifier,
    test_csv: str | Path = "data/test.csv",
    hard_csv: str | Path = "data/hard.csv",
    config_dir: str | Path = "config",
    report_dir: str | Path = "reports",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run evaluation across all tasks and datasets.
    """
    config_dir = Path(config_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration parameters
    temps = {}
    threshs = {}
    t_file = config_dir / "temperatures.json"
    th_file = config_dir / "thresholds.json"
    if t_file.exists():
        temps = json.loads(t_file.read_text(encoding="utf-8"))
    if th_file.exists():
        threshs = json.loads(th_file.read_text(encoding="utf-8"))

    results = {}
    for task_name in TASKS:
        t_val = temps.get(task_name, {}).get("T", 1.0)
        th_val = threshs.get(task_name)

        task_res = {}
        if Path(test_csv).exists():
            task_res["test"] = evaluate_task(
                classifier, test_csv, task_name, calibration_T=t_val, thresholds=th_val, max_samples=max_samples
            )
        if Path(hard_csv).exists():
            task_res["hard"] = evaluate_task(
                classifier, hard_csv, task_name, calibration_T=t_val, thresholds=th_val, max_samples=max_samples
            )

        results[task_name] = task_res

    # Save outputs
    (report_dir / "baseline_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_content = generate_markdown_report(results)
    (report_dir / "baseline_results.md").write_text(md_content, encoding="utf-8")

    logger.info("Saved evaluation report to %s", report_dir / "baseline_results.md")
    return results


if __name__ == "__main__":
    clf = CommentClassifier()
    full_evaluation(clf)
