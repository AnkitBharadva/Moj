"""
Instagram comment data ingestion.

Reads the exported folder structure:
  cockroachjantaparty/
    {post_id}/
      comments.json     — array of comment objects
      post_details.json — post metadata

Outputs anonymized CSVs ready for classification.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_post_details(post_dir: Path) -> dict:
    """Load post metadata."""
    details_file = post_dir / "post_details.json"
    if not details_file.exists():
        return {}
    try:
        with open(details_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", details_file, e)
        return {}


def load_comments(post_dir: Path) -> list[dict]:
    """Load and flatten comments + replies from a post directory."""
    comments_file = post_dir / "comments.json"
    if not comments_file.exists():
        return []

    try:
        with open(comments_file, "r", encoding="utf-8") as f:
            raw_comments = json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", comments_file, e)
        return []

    rows = []
    for c in raw_comments:
        text = str(c.get("text") or "").strip()
        # Top-level comment (Scrub PII: omit username, user_id, profile_pic_url)
        rows.append({
            "comment_id": str(c.get("comment_id")),
            "post_id": post_dir.name,
            "text": text,
            "posted_at": c.get("posted_at"),
            "like_count": int(c.get("like_count") or 0),
            "reply_count": int(c.get("reply_count") or 0),
            "is_reply": False,
            "parent_comment_id": None,
        })

        # Replies
        for r in c.get("replies", []):
            reply_text = str(r.get("text") or "").strip()
            rows.append({
                "comment_id": str(r.get("reply_id")),
                "post_id": post_dir.name,
                "text": reply_text,
                "posted_at": r.get("posted_at"),
                "like_count": int(r.get("like_count") or 0),
                "reply_count": 0,
                "is_reply": True,
                "parent_comment_id": str(c.get("comment_id")),
            })

    return rows


def ingest_all(
    data_dir: str | Path,
    output_dir: str | Path = "data",
    min_text_length: int = 2,
) -> pd.DataFrame:
    """
    Ingest all posts and comments from the Instagram export.

    PII removal:
    - Strips username, user_id, profile_pic_url
    - Only retains: comment_id, post_id, text, posted_at, like_count,
      reply_count, is_reply, parent_comment_id
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    post_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not post_dirs:
        # Fallback to any directory
        post_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    logger.info("Found %d post directories in %s", len(post_dirs), data_dir)

    for post_dir in sorted(post_dirs):
        comments = load_comments(post_dir)
        all_rows.extend(comments)

    df = pd.DataFrame(all_rows)
    logger.info("Raw extracted texts: %d", len(df))

    # Filter out empty or whitespace-only texts
    df = df[df["text"].str.len() >= min_text_length].copy()

    # Drop exact text duplicates to maximize diversity
    df = df.drop_duplicates(subset=["text"]).copy()
    logger.info("Unique comments after filtering: %d", len(df))

    output_file = output_dir / "all_comments.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    logger.info("Saved full dataset to %s", output_file)

    return df


def create_sample_sets(
    df: pd.DataFrame,
    output_dir: str | Path = "data",
    calib_size: int = 400,
    test_size: int = 800,
    hard_size: int = 100,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Create stratified sample sets for calibration, testing, and hard cases.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # Hard set features: Devanagari script, heavy emoji, mentions, short slang
    df["has_devanagari"] = df["text"].str.contains(r"[\u0900-\u097F]", regex=True, na=False)
    df["has_emoji"] = df["text"].str.contains(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U00002600-\U000027BF]",
        regex=True,
        na=False,
    )
    df["has_mention"] = df["text"].str.contains(r"@\w+", regex=True, na=False)
    df["is_short"] = df["text"].str.len() < 25

    hard_candidates = df[
        df["has_devanagari"] | df["has_emoji"] | df["is_short"] | df["has_mention"]
    ]
    hard_n = min(hard_size, len(hard_candidates))
    hard_sample = hard_candidates.sample(n=hard_n, random_state=seed).copy()

    # Pool for calib and test
    remaining = df.drop(hard_sample.index)

    calib_n = min(calib_size, len(remaining))
    calib_sample = remaining.sample(n=calib_n, random_state=seed).copy()
    remaining = remaining.drop(calib_sample.index)

    test_n = min(test_size, len(remaining))
    test_sample = remaining.sample(n=test_n, random_state=seed + 1).copy()

    # Initialize empty label columns for annotation / auto-labeler
    label_cols = ["sentiment", "hate", "support"]
    for s in [calib_sample, test_sample, hard_sample]:
        for col in label_cols:
            s[col] = ""

    sets = {
        "calib": calib_sample,
        "test": test_sample,
        "hard": hard_sample,
    }

    cols = [
        "comment_id",
        "post_id",
        "text",
        "posted_at",
        "like_count",
        "is_reply",
        "sentiment",
        "hate",
        "support",
    ]

    for name, s_df in sets.items():
        out_path = output_dir / f"{name}.csv"
        s_df[cols].to_csv(out_path, index=False, encoding="utf-8")
        logger.info("Saved %s: %d rows", out_path, len(s_df))

    return sets


if __name__ == "__main__":
    df_all = ingest_all("D:/Moj/cockroachjantaparty")
    create_sample_sets(df_all)
