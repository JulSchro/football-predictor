import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def probability_log_loss(y_true: list[str], probabilities: np.ndarray, labels: list[str]) -> float:
    return float(log_loss(y_true, probabilities, labels=labels))


def brier_score_1x2(y_true: list[str], rows: list[dict]) -> float:
    labels = ["H", "D", "A"]
    total = 0.0
    for actual, row in zip(y_true, rows):
        probs = {
            "H": row["home_win_prob"],
            "D": row["draw_prob"],
            "A": row["away_win_prob"],
        }
        total += sum((probs[label] - (1.0 if actual == label else 0.0)) ** 2 for label in labels)
    return float(total / max(len(y_true), 1))


def calibration_bins(rows: list[dict], bins: int = 5) -> list[dict]:
    buckets: list[list[dict]] = [[] for _ in range(bins)]
    for row in rows:
        confidence = max(row["home_win_prob"], row["draw_prob"], row["away_win_prob"])
        idx = min(int(confidence * bins), bins - 1)
        buckets[idx].append(row)

    output = []
    for idx, bucket in enumerate(buckets):
        if not bucket:
            output.append({"bin": idx + 1, "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        output.append(
            {
                "bin": idx + 1,
                "count": len(bucket),
                "confidence": float(np.mean([max(r["home_win_prob"], r["draw_prob"], r["away_win_prob"]) for r in bucket])),
                "accuracy": float(np.mean([r["actual"] == r["predicted"] for r in bucket])),
            }
        )
    return output


def favorite_breakdown(rows: list[dict], threshold: float = 0.5) -> dict[str, float]:
    favorites = [row for row in rows if max(row["home_win_prob"], row["draw_prob"], row["away_win_prob"]) >= threshold]
    under = [row for row in rows if row not in favorites]
    return {
        "favorite_matches": float(len(favorites)),
        "favorite_accuracy": float(np.mean([r["actual"] == r["predicted"] for r in favorites])) if favorites else 0.0,
        "balanced_matches": float(len(under)),
        "balanced_accuracy": float(np.mean([r["actual"] == r["predicted"] for r in under])) if under else 0.0,
    }
