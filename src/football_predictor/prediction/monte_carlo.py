from __future__ import annotations

import numpy as np


DEFAULT_LINES = {
    "goals": [0.5, 1.5, 2.5, 3.5, 4.5],
    "corners": [5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
    "cards": [1.5, 2.5, 3.5, 4.5, 5.5],
    "shots_on_target": [4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    "team_corners": [2.5, 3.5, 4.5, 5.5, 6.5],
    "team_shots_on_target": [1.5, 2.5, 3.5, 4.5, 5.5],
}


def simulate_market_distributions(
    prediction: dict,
    markets: dict,
    simulations: int = 20000,
    seed: int = 42,
    match_context: dict | None = None,
) -> dict:
    simulations = max(1000, min(int(simulations or 20000), 100000))
    rng = np.random.default_rng(seed)
    context = _context_adjustments(match_context)

    home_goals_mean = max(0.15, float(markets.get("home_xg_recent") or _implied_team_goals(prediction, "home")))
    away_goals_mean = max(0.15, float(markets.get("away_xg_recent") or _implied_team_goals(prediction, "away")))
    home_goals_mean *= context["goals"]
    away_goals_mean *= context["goals"]

    home_goals = rng.poisson(home_goals_mean, simulations)
    away_goals = rng.poisson(away_goals_mean, simulations)
    total_goals = home_goals + away_goals

    home_corners = _count_sample(rng, float(markets.get("home_corners_expected") or 4.5) * context["corners"], simulations, dispersion=1.45)
    away_corners = _count_sample(rng, float(markets.get("away_corners_expected") or 4.5) * context["corners"], simulations, dispersion=1.45)
    total_corners = home_corners + away_corners

    home_sot = _count_sample(rng, float(markets.get("home_shots_on_target_expected") or 4.0) * context["shots"], simulations, dispersion=1.25)
    away_sot = _count_sample(rng, float(markets.get("away_shots_on_target_expected") or 4.0) * context["shots"], simulations, dispersion=1.25)
    total_sot = home_sot + away_sot

    total_cards = _count_sample(rng, float(markets.get("total_cards_expected") or 4.0) * context["cards"], simulations, dispersion=1.75)

    score_counts: dict[str, int] = {}
    for home, away in zip(home_goals, away_goals):
        key = f"{int(home)}-{int(away)}"
        score_counts[key] = score_counts.get(key, 0) + 1
    top_scores = sorted(score_counts.items(), key=lambda item: item[1], reverse=True)[:8]

    return {
        "simulations": simulations,
        "seed": seed,
        "method": "monte_carlo_market_distribution_v1",
        "context_adjustments": context,
        "goals": _market_summary(total_goals, DEFAULT_LINES["goals"]),
        "corners": _market_summary(total_corners, DEFAULT_LINES["corners"]),
        "cards": _market_summary(total_cards, DEFAULT_LINES["cards"]),
        "shots_on_target": _market_summary(total_sot, DEFAULT_LINES["shots_on_target"]),
        "team_markets": {
            "home": {
                "corners": _market_summary(home_corners, DEFAULT_LINES["team_corners"]),
                "shots_on_target": _market_summary(home_sot, DEFAULT_LINES["team_shots_on_target"]),
            },
            "away": {
                "corners": _market_summary(away_corners, DEFAULT_LINES["team_corners"]),
                "shots_on_target": _market_summary(away_sot, DEFAULT_LINES["team_shots_on_target"]),
            },
        },
        "outcomes": {
            "home_win": _prob(home_goals > away_goals),
            "draw": _prob(home_goals == away_goals),
            "away_win": _prob(home_goals < away_goals),
            "both_teams_score": _prob((home_goals > 0) & (away_goals > 0)),
        },
        "top_scores": [{"score": score, "probability": round(count / simulations, 4)} for score, count in top_scores],
    }


def _market_summary(values: np.ndarray, lines: list[float]) -> dict:
    expected = float(np.mean(values))
    std = float(np.std(values))
    return {
        "expected": round(expected, 2),
        "std": round(std, 2),
        "variance": round(float(np.var(values)), 2),
        "stability": _stability(expected, std),
        "lines": {
            _line_key(line): {
                "over": _prob(values > line),
                "under": _prob(values < line),
                "line_distance": round(abs(expected - line), 2),
            }
            for line in lines
        },
        "percentiles": {
            "p10": round(float(np.percentile(values, 10)), 2),
            "p25": round(float(np.percentile(values, 25)), 2),
            "p50": round(float(np.percentile(values, 50)), 2),
            "p75": round(float(np.percentile(values, 75)), 2),
            "p90": round(float(np.percentile(values, 90)), 2),
        },
    }


def _count_sample(rng: np.random.Generator, mean: float, size: int, dispersion: float) -> np.ndarray:
    mean = max(0.05, float(mean))
    variance = max(mean + mean * dispersion, mean + 0.01)
    if variance <= mean + 0.05:
        return rng.poisson(mean, size)
    n = mean**2 / (variance - mean)
    p = n / (n + mean)
    return rng.negative_binomial(max(n, 0.1), min(max(p, 0.01), 0.99), size)


def _context_adjustments(match_context: dict | None) -> dict[str, float]:
    context = match_context or {}
    goals = 1.0
    corners = 1.0
    cards = 1.0
    shots = 1.0
    if context.get("is_knockout"):
        goals *= 0.94
        cards *= 1.1
    if context.get("is_final"):
        goals *= 0.9
        cards *= 1.12
    if context.get("is_group_stage"):
        corners *= 1.03
    if context.get("is_friendly"):
        cards *= 0.82
        goals *= 1.04
        shots *= 0.96
    pressure = float(context.get("pressure_index") or 50)
    cards *= 1 + max(min((pressure - 50) / 250, 0.18), -0.12)
    return {"goals": round(goals, 4), "corners": round(corners, 4), "cards": round(cards, 4), "shots": round(shots, 4)}


def _implied_team_goals(prediction: dict, side: str) -> float:
    home_prob = float(prediction.get("home_win_prob") or 0.34)
    away_prob = float(prediction.get("away_win_prob") or 0.33)
    over = float(prediction.get("over_2_5_prob") or 0.5)
    total = 1.8 + over * 1.35
    edge = home_prob - away_prob
    share = 0.5 + (edge * 0.22 if side == "home" else -edge * 0.22)
    return total * min(max(share, 0.28), 0.72)


def _stability(expected: float, std: float) -> str:
    if expected <= 0:
        return "baja"
    cv = std / expected
    if cv <= 0.35:
        return "alta"
    if cv <= 0.55:
        return "media"
    return "baja"


def _prob(mask: np.ndarray) -> float:
    return round(float(np.mean(mask)), 4)


def _line_key(line: float) -> str:
    return str(line).replace(".", "_")
