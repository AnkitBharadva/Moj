# %% [markdown]
# # 02: Calibration & Reliability Diagram
# 
# Fits temperature scaling $T$ on `calib.csv` and measures Expected Calibration Error (ECE).

# %%
import json
import pandas as pd
import numpy as np
from src.readout import CommentClassifier
from src.calibrate import calibrate_all_tasks, expected_calibration_error

clf = CommentClassifier()
cal_results = calibrate_all_tasks(clf, "data/calib.csv")
print("Calibration results:")
print(json.dumps(cal_results, indent=2))

# %% [markdown]
# ### Check Calibrated Temperatures

# %%
with open("config/temperatures.json") as f:
    temps = json.load(f)
print("Fitted Temperatures:", temps)

with open("config/thresholds.json") as f:
    thresholds = json.load(f)
print("Fitted Thresholds:", thresholds)
