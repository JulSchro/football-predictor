import pandas as pd

from football_predictor.evaluation.metrics import brier_score_1x2, calibration_bins, favorite_breakdown, probability_log_loss
from football_predictor.prediction.predictor import MatchPredictor


def actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def predicted_outcome(row: dict[str, float]) -> str:
    probabilities = {
        "H": row["home_win_prob"],
        "D": row["draw_prob"],
        "A": row["away_win_prob"],
    }
    return max(probabilities, key=probabilities.get)


def run_backtest(matches: pd.DataFrame, min_train_matches: int = 5) -> tuple[pd.DataFrame, dict[str, float]]:
    played = matches.dropna(subset=["home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    rows: list[dict] = []

    for idx, match in played.iterrows():
        if idx < min_train_matches:
            continue
        history = played.iloc[:idx]
        prediction = MatchPredictor(history).predict(str(match["home_team"]), str(match["away_team"]))
        payload = prediction.model_dump()
        actual = actual_outcome(int(match["home_goals"]), int(match["away_goals"]))
        pred = predicted_outcome(payload)
        rows.append(
            {
                "date": match["date"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "actual": actual,
                "predicted": pred,
                **payload,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results, {
            "matches": 0.0,
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "log_loss": 0.0,
            "brier": 0.0,
            "calibration_error": 0.0,
        }

    confidence = results[["home_win_prob", "draw_prob", "away_win_prob"]].max(axis=1).mean()
    rows = results.to_dict("records")
    probs = results[["home_win_prob", "draw_prob", "away_win_prob"]].to_numpy()
    labels = ["H", "D", "A"]
    bins = calibration_bins(rows)
    calibration_error = sum(abs(item["confidence"] - item["accuracy"]) * item["count"] for item in bins) / len(results)
    metrics = {
        "matches": float(len(results)),
        "accuracy": float((results["actual"] == results["predicted"]).mean()),
        "avg_confidence": float(confidence),
        "log_loss": probability_log_loss(list(results["actual"]), probs, labels),
        "brier": brier_score_1x2(list(results["actual"]), rows),
        "calibration_error": float(calibration_error),
        "calibration_bins": bins,
        "favorite_breakdown": favorite_breakdown(rows),
    }
    return results, metrics
