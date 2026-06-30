from __future__ import annotations

import csv
import json
from math import log, sqrt
from pathlib import Path
from typing import Iterable


OUTCOME_KEYS = ("home", "draw", "away")


def load_backtest_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [normalize_backtest_row(row) for row in csv.DictReader(fh)]


def normalize_backtest_row(row: dict) -> dict:
    normalized = {
        "match_date": row.get("match_date") or row.get("date"),
        "competition": row.get("competition") or "World Cup 2026",
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "model_version": row.get("model_version") or "manual_snapshot",
        "predicted_home_prob": parse_prob(row.get("predicted_home_prob")),
        "predicted_draw_prob": parse_prob(row.get("predicted_draw_prob")),
        "predicted_away_prob": parse_prob(row.get("predicted_away_prob")),
        "predicted_pick": normalize_pick(row.get("predicted_pick")),
        "predicted_scores_json": normalize_scores(row.get("predicted_scores")),
        "predicted_corners": parse_float(row.get("predicted_corners")),
        "predicted_shots_on_target": parse_float(row.get("predicted_shots_on_target")),
        "predicted_cards": parse_float(row.get("predicted_cards")),
        "predicted_over_2_5_prob": parse_prob(row.get("predicted_over_2_5_prob")),
        "predicted_btts_prob": parse_prob(row.get("predicted_btts_prob")),
        "actual_home_goals": parse_int(row.get("actual_home_goals")),
        "actual_away_goals": parse_int(row.get("actual_away_goals")),
        "actual_corners": parse_float(row.get("actual_corners")),
        "actual_shots_on_target": parse_float(row.get("actual_shots_on_target")),
        "actual_cards": parse_float(row.get("actual_cards")),
        "source": row.get("source") or "manual",
        "notes": row.get("notes") or "",
    }
    if not normalized["predicted_pick"]:
        normalized["predicted_pick"] = pick_from_probs(normalized)
    return normalized


def compute_backtest_metrics(rows: Iterable[dict]) -> dict:
    records = [row for row in rows if row.get("actual_home_goals") is not None and row.get("actual_away_goals") is not None]
    evaluated = [add_evaluation(row) for row in records]
    strict = [row for row in evaluated if row["actual_outcome"] and row.get("predicted_pick")]
    prob_rows = [row for row in evaluated if has_probs(row)]

    market_metrics = {
        "corners": regression_metrics(evaluated, "predicted_corners", "actual_corners"),
        "shots_on_target": regression_metrics(evaluated, "predicted_shots_on_target", "actual_shots_on_target"),
        "cards": regression_metrics(evaluated, "predicted_cards", "actual_cards"),
    }
    binary_metrics = {
        "over_2_5": binary_prob_metrics(evaluated, "predicted_over_2_5_prob", "actual_over_2_5"),
        "both_teams_score": binary_prob_metrics(evaluated, "predicted_btts_prob", "actual_btts"),
    }
    return {
        "matches": len(evaluated),
        "strict_accuracy": safe_div(sum(1 for row in strict if row["winner_hit"]), len(strict)),
        "strict_evaluated": len(strict),
        "probability_rows": len(prob_rows),
        "log_loss": multiclass_log_loss(prob_rows),
        "brier": multiclass_brier(prob_rows),
        "exact_score_top2_accuracy": safe_div(
            sum(1 for row in evaluated if row["exact_score_top2_hit"]),
            sum(1 for row in evaluated if row["top_scores"]),
        ),
        "market_metrics": market_metrics,
        "binary_metrics": binary_metrics,
        "biases": {name: values["bias"] for name, values in market_metrics.items()},
        "rows": evaluated,
    }


def add_evaluation(row: dict) -> dict:
    actual = actual_outcome(row)
    scores = parse_scores(row.get("predicted_scores_json"))
    actual_score = f"{row.get('actual_home_goals')}-{row.get('actual_away_goals')}"
    return {
        **row,
        "actual_outcome": actual,
        "actual_score": actual_score,
        "actual_over_2_5": int((row.get("actual_home_goals") or 0) + (row.get("actual_away_goals") or 0) > 2.5),
        "actual_btts": int((row.get("actual_home_goals") or 0) > 0 and (row.get("actual_away_goals") or 0) > 0),
        "top_scores": scores,
        "winner_hit": row.get("predicted_pick") == actual,
        "exact_score_top2_hit": actual_score in [score["score"] for score in scores[:2]],
    }


def regression_metrics(rows: list[dict], predicted_key: str, actual_key: str) -> dict:
    pairs = [(float(row[predicted_key]), float(row[actual_key])) for row in rows if row.get(predicted_key) is not None and row.get(actual_key) is not None]
    if not pairs:
        return {"count": 0, "mae": None, "rmse": None, "bias": None, "direction": "sin datos"}
    errors = [predicted - actual for predicted, actual in pairs]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = sqrt(sum(error * error for error in errors) / len(errors))
    bias = sum(errors) / len(errors)
    return {
        "count": len(pairs),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "direction": bias_direction(bias),
    }


def binary_prob_metrics(rows: list[dict], prob_key: str, actual_key: str) -> dict:
    pairs = [(float(row[prob_key]), int(row[actual_key])) for row in rows if row.get(prob_key) is not None and row.get(actual_key) is not None]
    if not pairs:
        return {"count": 0, "brier": None, "log_loss": None}
    brier = sum((prob - actual) ** 2 for prob, actual in pairs) / len(pairs)
    logloss = -sum(actual * log(clip_prob(prob)) + (1 - actual) * log(clip_prob(1 - prob)) for prob, actual in pairs) / len(pairs)
    return {"count": len(pairs), "brier": round(brier, 4), "log_loss": round(logloss, 4)}


def multiclass_log_loss(rows: list[dict]) -> float | None:
    if not rows:
        return None
    total = 0.0
    for row in rows:
        total -= log(clip_prob(outcome_prob(row, row["actual_outcome"])))
    return round(total / len(rows), 4)


def multiclass_brier(rows: list[dict]) -> float | None:
    if not rows:
        return None
    total = 0.0
    for row in rows:
        actual = row["actual_outcome"]
        total += sum((outcome_prob(row, key) - (1 if key == actual else 0)) ** 2 for key in OUTCOME_KEYS)
    return round(total / len(rows), 4)


def actual_outcome(row: dict) -> str | None:
    home = row.get("actual_home_goals")
    away = row.get("actual_away_goals")
    if home is None or away is None:
        return None
    if home > away:
        return "home"
    if home < away:
        return "away"
    return "draw"


def has_probs(row: dict) -> bool:
    return all(row.get(f"predicted_{key}_prob") is not None for key in OUTCOME_KEYS)


def outcome_prob(row: dict, key: str) -> float:
    return float(row.get(f"predicted_{key}_prob") or 0.0)


def pick_from_probs(row: dict) -> str | None:
    probs = {key: row.get(f"predicted_{key}_prob") for key in OUTCOME_KEYS}
    valid = {key: value for key, value in probs.items() if value is not None}
    if not valid:
        return None
    return max(valid, key=valid.get)


def normalize_pick(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    mapping = {
        "1": "home",
        "local": "home",
        "home": "home",
        "x": "draw",
        "draw": "draw",
        "empate": "draw",
        "2": "away",
        "visitante": "away",
        "away": "away",
    }
    return mapping.get(text, text)


def normalize_scores(value: str | None) -> str:
    if not value:
        return "[]"
    return json.dumps(parse_scores(value), ensure_ascii=False)


def parse_scores(value: str | None) -> list[dict]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        loaded = json.loads(value)
        if isinstance(loaded, list):
            return loaded
    except json.JSONDecodeError:
        pass
    scores = []
    for part in str(value).split("|"):
        if not part.strip():
            continue
        if ":" in part:
            score, prob = part.split(":", 1)
            scores.append({"score": score.strip(), "probability": parse_prob(prob)})
        else:
            scores.append({"score": part.strip(), "probability": None})
    return scores


def parse_prob(value: object) -> float | None:
    number = parse_float(value)
    if number is None:
        return None
    return number / 100 if number > 1 else number


def parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().replace("%", ""))
    except ValueError:
        return None


def parse_int(value: object) -> int | None:
    number = parse_float(value)
    return int(number) if number is not None else None


def safe_div(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def clip_prob(value: float) -> float:
    return min(max(value, 1e-15), 1 - 1e-15)


def bias_direction(value: float) -> str:
    if value > 0.25:
        return "sobreestima"
    if value < -0.25:
        return "subestima"
    return "calibrado"
