import pandas as pd


def estimate_secondary_markets(stats: pd.DataFrame, home_team: str, away_team: str) -> dict:
    home = team_market_profile(stats, home_team)
    away = team_market_profile(stats, away_team)
    home_corners = blend(home["corners_for"], away["corners_against"])
    away_corners = blend(away["corners_for"], home["corners_against"])
    home_shots = blend(home["shots"], away["shots_against"])
    away_shots = blend(away["shots"], home["shots_against"])
    home_sot = blend(home["shots_on_target"], away["shots_on_target_against"])
    away_sot = blend(away["shots_on_target"], home["shots_on_target_against"])
    total_corners = home_corners + away_corners
    total_cards = home["cards"] + away["cards"]
    total_shots = home_shots + away_shots
    total_sot = home_sot + away_sot

    return {
        "home_corners_expected": round(home_corners, 2),
        "away_corners_expected": round(away_corners, 2),
        "total_corners_expected": round(total_corners, 2),
        "over_8_5_corners_prob": poisson_over(total_corners, 8.5),
        "home_shots_expected": round(home_shots, 2),
        "away_shots_expected": round(away_shots, 2),
        "total_shots_expected": round(total_shots, 2),
        "shots_on_target_expected": round(total_sot, 2),
        "home_shots_on_target_expected": round(home_sot, 2),
        "away_shots_on_target_expected": round(away_sot, 2),
        "total_cards_expected": round(total_cards, 2),
        "over_3_5_cards_prob": poisson_over(total_cards, 3.5),
        "home_xg_recent": round(home["xg"], 2),
        "away_xg_recent": round(away["xg"], 2),
        "home_possession_expected": round(home["possession"], 2),
        "away_possession_expected": round(away["possession"], 2),
        "advanced_samples": {"home": home["sample"], "away": away["sample"]},
    }


def team_market_profile(stats: pd.DataFrame, team: str) -> dict:
    if stats.empty or "team" not in stats:
        return default_profile()
    rows = stats[stats["team"] == team].sort_values("date").tail(10)
    if rows.empty:
        return default_profile()
    against = stats[stats.get("opponent") == team].sort_values("date").tail(10) if "opponent" in stats else pd.DataFrame()
    fouls_cards = weighted_recent(rows, "cards_estimate")
    if fouls_cards is None:
        fouls_cards = (weighted_recent(rows, "fouls") or 12.0) * 0.18
    return {
        "corners_for": weighted_recent(rows, "corners") or 4.5,
        "corners_against": weighted_recent(against, "corners") if not against.empty else 4.5,
        "shots": weighted_recent(rows, "total_shots") or 11.0,
        "shots_against": weighted_recent(against, "total_shots") if not against.empty else 11.0,
        "shots_on_target": weighted_recent(rows, "shots_on_target") or 4.0,
        "shots_on_target_against": weighted_recent(against, "shots_on_target") if not against.empty else 4.0,
        "cards": fouls_cards,
        "fouls": weighted_recent(rows, "fouls") or 12.0,
        "xg": weighted_recent(rows, "xg") or 1.25,
        "possession": weighted_recent(rows, "possession_pct") or 50.0,
        "sample": int(len(rows)),
    }


def default_profile() -> dict:
    return {
        "corners_for": 4.5,
        "corners_against": 4.5,
        "shots": 11.0,
        "shots_against": 11.0,
        "shots_on_target": 4.0,
        "shots_on_target_against": 4.0,
        "cards": 2.1,
        "fouls": 12.0,
        "xg": 1.25,
        "possession": 50.0,
        "sample": 0,
    }


def weighted_recent(rows: pd.DataFrame, column: str) -> float | None:
    if rows.empty or column not in rows:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        return None
    weights = pd.Series(range(1, len(values) + 1), dtype=float, index=values.index)
    return float((values * weights).sum() / weights.sum())


def blend(own: float, opponent_allowed: float | None) -> float:
    if opponent_allowed is None:
        return own
    return own * 0.58 + float(opponent_allowed) * 0.42


def poisson_over(lam: float, line: float) -> float:
    from math import exp, factorial

    threshold = int(line)
    under = sum((lam**k * exp(-lam)) / factorial(k) for k in range(threshold + 1))
    return round(max(0.0, min(1.0, 1 - under)), 4)
