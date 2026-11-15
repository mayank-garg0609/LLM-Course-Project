import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


def load_json_any(path):
    """HaluEval files may be one JSON array or one JSON object per line."""
    txt = open(path, encoding="utf-8").read().strip()
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in txt.splitlines() if line.strip()]


def detection_metrics(y_true, y_pred):
    """Positive class (1) = hallucinated."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p,
            "recall": r, "f1": f1,
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            "fnr": fn / (fn + tp) if (fn + tp) else 0.0}


def tune_tau(y_true, scores, grid=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Pick the R threshold with the best F1 (use on the DEV split only)."""
    return max(grid, key=lambda t: detection_metrics(
        y_true, [int(s >= t) for s in scores])["f1"])


def latency_stats(latencies):
    a = np.array(latencies)
    return {"mean_s": float(a.mean()), "p95_s": float(np.percentile(a, 95))}
