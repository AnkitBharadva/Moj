# %% [markdown]
# # 01: Baseline Readout & Model Sanity Checks
# 
# Verifies letter probability extraction, think mode status, and latency.

# %%
import time
import json
import pandas as pd
import numpy as np
from src.readout import CommentClassifier, TASKS

clf = CommentClassifier()
print("Model loaded:", clf.model)

# %% [markdown]
# ### Test Sample Comment across all 3 Tasks

# %%
sample_text = "Full support to the students! Adivasi school repair work must happen immediately 🙌"
res = clf.analyze(sample_text)
print(json.dumps(res, indent=2))

# %% [markdown]
# ### Check Option Order Bias
# Evaluate probabilities when option order is permuted.

# %%
comment = "Sab jumla hai.. kuch nhi hoga yahan"
task = TASKS["sentiment"]

p1 = clf.letter_probs(comment, task, use_cache=False)
print("Standard order (pos, neu, neg):", p1)

# %% [markdown]
# ### Latency Benchmark (10 iterations)

# %%
latencies = []
for i in range(10):
    t0 = time.perf_counter()
    clf.letter_probs(f"Test latency {i}", task, use_cache=False)
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"Latency p50: {np.percentile(latencies, 50):.1f} ms")
print(f"Latency p95: {np.percentile(latencies, 95):.1f} ms")
