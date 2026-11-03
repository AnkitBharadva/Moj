# Baseline Evaluation Report

Evaluation of calibrated one-token letter readout on social media comments.

## Task: SENTIMENT

### Dataset: `test` (N = 30)

- **Macro-F1**: `1.0000`
- **ECE**: `0.2306`
- **Latency (p50 / p95)**: `231.5 ms` / `277.9 ms`

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| positive | 1.000 | 1.000 | 1.000 | 17.0 |
| neutral | 1.000 | 1.000 | 1.000 | 1.0 |
| negative | 1.000 | 1.000 | 1.000 | 12.0 |

**Confusion Matrix:**
```
  17     0     0
   0     1     0
   0     0    12
```

### Dataset: `hard` (N = 20)

- **Macro-F1**: `1.0000`
- **ECE**: `0.1637`
- **Latency (p50 / p95)**: `230.2 ms` / `358.7 ms`

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| positive | 1.000 | 1.000 | 1.000 | 15.0 |
| neutral | 1.000 | 1.000 | 1.000 | 1.0 |
| negative | 1.000 | 1.000 | 1.000 | 4.0 |

**Confusion Matrix:**
```
  15     0     0
   0     1     0
   0     0     4
```

---

## Task: HATE

### Dataset: `test` (N = 30)

- **Macro-F1**: `1.0000`
- **ECE**: `0.2568`
- **Latency (p50 / p95)**: `224.8 ms` / `268.0 ms`

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| hateful | 1.000 | 1.000 | 1.000 | 11.0 |
| neither | 1.000 | 1.000 | 1.000 | 19.0 |

**Confusion Matrix:**
```
  11     0
   0    19
```

### Dataset: `hard` (N = 20)

- **Macro-F1**: `1.0000`
- **ECE**: `0.1591`
- **Latency (p50 / p95)**: `219.7 ms` / `263.1 ms`

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| hateful | 1.000 | 1.000 | 1.000 | 8.0 |
| neither | 1.000 | 1.000 | 1.000 | 12.0 |

**Confusion Matrix:**
```
   8     0
   0    12
```

---

## Task: SUPPORT

### Dataset: `test` (N = 30)

- **Macro-F1**: `1.0000`
- **ECE**: `0.2925`
- **Latency (p50 / p95)**: `222.5 ms` / `267.4 ms`

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| supports | 1.000 | 1.000 | 1.000 | 22.0 |
| neither | 1.000 | 1.000 | 1.000 | 8.0 |

**Confusion Matrix:**
```
  22     0
   0     8
```

### Dataset: `hard` (N = 20)

- **Macro-F1**: `1.0000`
- **ECE**: `0.1911`
- **Latency (p50 / p95)**: `228.1 ms` / `315.2 ms`

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| supports | 1.000 | 1.000 | 1.000 | 12.0 |
| neither | 1.000 | 1.000 | 1.000 | 8.0 |

**Confusion Matrix:**
```
  12     0
   0     8
```

---
