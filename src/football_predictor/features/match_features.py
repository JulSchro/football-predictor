import pandas as pd

from football_predictor.models.baseline_elo import EloBaseline
from football_predictor.features.team_features import average_home_away_goals, recent_form
from football_predictor.features.team_features import recent_form_by_venue, weighted_recent_form


FEATURE_COLUMNS = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_goal_diff",
    "home_win_rate",
    "home_form_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_goal_diff",
    "away_win_rate",
    "away_form_points",
    "attack_strength_diff",
    "defense_strength_diff",
    "home_venue_goals_for",
    "home_venue_goals_against",
    "home_venue_win_rate",
    "away_venue_goals_for",
    "away_venue_goals_against",
    "away_venue_win_rate",
    "home_weighted_goal_diff",
    "away_weighted_goal_diff",
    "weighted_form_diff",
    "elo_diff",
    "is_home",
]


def build_match_features(df: pd.DataFrame, home_team: str, away_team: str, before_date: str | None = None, n: int = 5) -> dict[str, float]:
    home = recent_form(df, home_team, n=n, before_date=before_date)
    away = recent_form(df, away_team, n=n, before_date=before_date)
    home_venue = recent_form_by_venue(df, home_team, is_home=True, n=n, before_date=before_date)
    away_venue = recent_form_by_venue(df, away_team, is_home=False, n=n, before_date=before_date)
    home_weighted = weighted_recent_form(df, home_team, n=n, before_date=before_date)
    away_weighted = weighted_recent_form(df, away_team, n=n, before_date=before_date)
    league = average_home_away_goals(df)
    history = df[pd.to_datetime(df["date"]) < pd.to_datetime(before_date)] if before_date else df
    elo = EloBaseline().fit(history)

    home_attack = home["avg_goals_for"] / max(league["home_goals_avg"], 0.1)
    away_attack = away["avg_goals_for"] / max(league["away_goals_avg"], 0.1)
    home_defense = home["avg_goals_against"] / max(league["away_goals_avg"], 0.1)
    away_defense = away["avg_goals_against"] / max(league["home_goals_avg"], 0.1)

    return {
        "home_avg_goals_for": home["avg_goals_for"],
        "home_avg_goals_against": home["avg_goals_against"],
        "home_goal_diff": home["goal_diff"],
        "home_win_rate": home["win_rate"],
        "home_form_points": home["form_points"],
        "away_avg_goals_for": away["avg_goals_for"],
        "away_avg_goals_against": away["avg_goals_against"],
        "away_goal_diff": away["goal_diff"],
        "away_win_rate": away["win_rate"],
        "away_form_points": away["form_points"],
        "attack_strength_diff": home_attack - away_attack,
        "defense_strength_diff": away_defense - home_defense,
        "home_venue_goals_for": home_venue["venue_goals_for"],
        "home_venue_goals_against": home_venue["venue_goals_against"],
        "home_venue_win_rate": home_venue["venue_win_rate"],
        "away_venue_goals_for": away_venue["venue_goals_for"],
        "away_venue_goals_against": away_venue["venue_goals_against"],
        "away_venue_win_rate": away_venue["venue_win_rate"],
        "home_weighted_goal_diff": home_weighted["weighted_goal_diff"],
        "away_weighted_goal_diff": away_weighted["weighted_goal_diff"],
        "weighted_form_diff": home_weighted["weighted_form_points"] - away_weighted["weighted_form_points"],
        "elo_diff": elo.rating(home_team) - elo.rating(away_team),
        "is_home": 1.0,
    }


def build_training_matrix(df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    played = df.dropna(subset=["home_goals", "away_goals"]).sort_values("date")

    for _, match in played.iterrows():
        features = build_match_features(
            played,
            str(match["home_team"]),
            str(match["away_team"]),
            before_date=str(match["date"]),
            n=n,
        )
        rows.append(features)
        if match["home_goals"] > match["away_goals"]:
            labels.append("H")
        elif match["home_goals"] < match["away_goals"]:
            labels.append("A")
        else:
            labels.append("D")

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS).fillna(0), pd.Series(labels)
