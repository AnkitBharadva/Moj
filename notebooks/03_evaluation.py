# %% [markdown]
# # 03: Comprehensive Model Evaluation & Error Analysis
# 
# Evaluates calibrated model on `test.csv` and `hard.csv`.
# Produces Macro-F1, per-class metrics, confusion matrices, and markdown report.

# %%
import json
import pandas as pd
from src.readout import CommentClassifier
from src.evaluate import full_evaluation

clf = CommentClassifier()
results = full_evaluation(clf, "data/test.csv", "data/hard.csv")

print("Evaluation Summary:")
for task, datasets in results.items():
    print(f"\n=== Task: {task.upper()} ===")
    for ds_name, m in datasets.items():
        if m.get("n_samples", 0) > 0:
            print(f"  [{ds_name}] Macro-F1: {m['macro_f1']:.4f} | ECE: {m['ece']:.4f} | p50 Latency: {m['latency_p50_ms']} ms")

# %% [markdown]
# ### View Generated Markdown Report
# Read `reports/baseline_results.md`.

# %%
from pathlib import Path
rep = Path("reports/baseline_results.md")
if rep.exists():
    print(rep.read_text(encoding="utf-8"))
