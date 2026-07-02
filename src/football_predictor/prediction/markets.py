import pandas as pd

from football_predictor.prediction.referee_model import apply_referee_card_adjustment


def estimate_secondary_markets(stats: pd.DataFrame, home_team: str, away_team: str, referee_profile: dict | None = None) -> dict:
    home = team_market_profile(stats, home_team)
    away = team_market_profile(stats, away_team)
    home_corners = blend(home["corners_for"], away["corners_against"])
    away_corners = blend(away["corners_for"], home["corners_against"])
    home_shots = blend(home["shots"], away["shots_against"])
    away_shots = blend(away["shots"], home["shots_against"])
    home_sot = blend(home["shots_on_target"], away["shots_on_target_against"])
    away_sot = blend(away["shots_on_target"], home["shots_on_target_against"])
    total_corners = home_corners + away_corners
    team_cards = home["cards"] + away["cards"]
    referee_adjustment = apply_referee_card_adjustment(team_cards, home["sample"] + away["sample"], referee_profile)
    total_cards = referee_adjustment["expected_cards"]
    total_shots = home_shots + away_shots
    total_sot = home_sot + away_sot

    team_markets = team_market_expectations(
        home_team=home_team,
        away_team=away_team,
        home_corners=home_corners,
        away_corners=away_corners,
        home_shots=home_shots,
        away_shots=away_shots,
        home_sot=home_sot,
        away_sot=away_sot,
        home_sample=home["sample"],
        away_sample=away["sample"],
    )
    return {
        "home_corners_expected": round(home_corners, 2),
        "away_corners_expected": round(away_corners, 2),
        "total_corners_expected": round(total_corners, 2),
        "over_8_5_corners_prob": poisson_over(total_corners, 8.5),
        "home_over_3_5_corners_prob": poisson_over(home_corners, 3.5),
        "away_over_3_5_corners_prob": poisson_over(away_corners, 3.5),
        "home_shots_expected": round(home_shots, 2),
        "away_shots_expected": round(away_shots, 2),
        "total_shots_expected": round(total_shots, 2),
        "shots_on_target_expected": round(total_sot, 2),
        "home_shots_on_target_expected": round(home_sot, 2),
        "away_shots_on_target_expected": round(away_sot, 2),
        "home_over_3_5_shots_on_target_prob": poisson_over(home_sot, 3.5),
        "away_over_3_5_shots_on_target_prob": poisson_over(away_sot, 3.5),
        "total_cards_expected": round(total_cards, 2),
        "over_3_5_cards_prob": poisson_over(total_cards, 3.5),
        "referee_cards": {
            **referee_adjustment,
            "profile": referee_profile or {},
        },
        "home_xg_recent": round(home["xg"], 2),
        "away_xg_recent": round(away["xg"], 2),
        "home_possession_expected": round(home["possession"], 2),
        "away_possession_expected": round(away["possession"], 2),
        "team_markets": team_markets,
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


def team_market_expectations(
    home_team: str,
    away_team: str,
    home_corners: float,
    away_corners: float,
    home_shots: float,
    away_shots: float,
    home_sot: float,
    away_sot: float,
    home_sample: int,
    away_sample: int,
) -> dict:
    return {
        "home": {
            "team": home_team,
            "corners_expected": round(home_corners, 2),
            "corners_range": likely_range(home_corners),
            "over_3_5_corners_prob": poisson_over(home_corners, 3.5),
            "over_4_5_corners_prob": poisson_over(home_corners, 4.5),
            "shots_expected": round(home_shots, 2),
            "shots_on_target_expected": round(home_sot, 2),
            "shots_on_target_range": likely_range(home_sot),
            "over_3_5_shots_on_target_prob": poisson_over(home_sot, 3.5),
            "over_4_5_shots_on_target_prob": poisson_over(home_sot, 4.5),
            "sample": home_sample,
            "confidence": sample_confidence(home_sample),
        },
        "away": {
            "team": away_team,
            "corners_expected": round(away_corners, 2),
            "corners_range": likely_range(away_corners),
            "over_3_5_corners_prob": poisson_over(away_corners, 3.5),
            "over_4_5_corners_prob": poisson_over(away_corners, 4.5),
            "shots_expected": round(away_shots, 2),
            "shots_on_target_expected": round(away_sot, 2),
            "shots_on_target_range": likely_range(away_sot),
            "over_3_5_shots_on_target_prob": poisson_over(away_sot, 3.5),
            "over_4_5_shots_on_target_prob": poisson_over(away_sot, 4.5),
            "sample": away_sample,
            "confidence": sample_confidence(away_sample),
        },
    }


def likely_range(expected: float) -> dict:
    lower = max(0, int(expected - max(1.0, expected**0.5)))
    upper = max(lower, int(expected + max(1.0, expected**0.5) + 0.999))
    return {"low": lower, "high": upper, "label": f"{lower}-{upper}"}


def sample_confidence(sample: int) -> str:
    if sample >= 8:
        return "alta"
    if sample >= 4:
        return "media"
    return "baja"
