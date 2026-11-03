"""
LLM-Assisted Labeling Assistant.

Uses the CommentClassifier to generate initial candidate labels for:
  - data/calib.csv
  - data/test.csv
  - data/hard.csv

Populates predicted labels and flags low-confidence or hate cases
for targeted human spot-checking.
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .readout import TASKS, CommentClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def auto_label_dataset(
    classifier: CommentClassifier,
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    confidence_threshold: float = 0.65,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Run zero-shot classification across sentiment, hate, and support.
    Populates columns and adds a 'needs_human_review' boolean column.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv or input_csv)

    df = pd.read_csv(input_path)
    if max_samples is not None and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()

    logger.info("Auto-labeling %d rows from %s...", len(df), input_path.name)

    needs_review = []
    sentiments = []
    hates = []
    supports = []
    conf_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Labeling {input_path.stem}"):
        text = str(row["text"])
        res = classifier.analyze(text)

        s_label = res["sentiment"]["label"]
        h_label = res["hate"]["label"]
        sp_label = res["support"]["label"]

        sentiments.append(s_label)
        hates.append(h_label)
        supports.append(sp_label)

        min_conf = min(
            res["sentiment"]["confidence"],
            res["hate"]["confidence"],
            res["support"]["confidence"],
        )
        conf_scores.append(round(min_conf, 4))

        # Flag if any task was low-confidence or hate/offense was triggered
        flag = (min_conf < confidence_threshold) or (h_label != "neither")
        needs_review.append(flag)

    df["sentiment"] = sentiments
    df["hate"] = hates
    df["support"] = supports
    df["min_confidence"] = conf_scores
    df["needs_human_review"] = needs_review

    df.to_csv(output_path, index=False, encoding="utf-8")
    flagged_count = sum(needs_review)
    logger.info(
        "Saved %s: %d / %d rows (%.1f%%) flagged for human review",
        output_path.name,
        flagged_count,
        len(df),
        (flagged_count / len(df)) * 100 if len(df) else 0,
    )
    return df


if __name__ == "__main__":
    clf = CommentClassifier()
    # Label calibration set (400 samples)
    if Path("data/calib.csv").exists():
        auto_label_dataset(clf, "data/calib.csv")
    # Label test set (800 samples)
    if Path("data/test.csv").exists():
        auto_label_dataset(clf, "data/test.csv")
    # Label hard set (100 samples)
    if Path("data/hard.csv").exists():
        auto_label_dataset(clf, "data/hard.csv")
