import pandas as pd


def team_history(df: pd.DataFrame, team: str, before_date: str | None = None) -> pd.DataFrame:
    matches = df.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    if before_date is not None:
        matches = matches[matches["date"] < pd.to_datetime(before_date)]

    home = matches[matches["home_team"] == team].assign(
        goals_for=lambda x: x["home_goals"],
        goals_against=lambda x: x["away_goals"],
        result=lambda x: result_points(x["home_goals"], x["away_goals"]),
        is_home=1,
    )
    away = matches[matches["away_team"] == team].assign(
        goals_for=lambda x: x["away_goals"],
        goals_against=lambda x: x["home_goals"],
        result=lambda x: result_points(x["away_goals"], x["home_goals"]),
        is_home=0,
    )
    cols = ["date", "goals_for", "goals_against", "result", "is_home"]
    return pd.concat([home[cols], away[cols]]).dropna().sort_values("date")


def result_points(goals_for: pd.Series, goals_against: pd.Series) -> pd.Series:
    return (goals_for > goals_against).astype(int) * 3 + (goals_for == goals_against).astype(int)


def recent_form(df: pd.DataFrame, team: str, n: int = 5, before_date: str | None = None) -> dict[str, float]:
    hist = team_history(df, team, before_date).tail(n)
    if hist.empty:
        return {
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
            "goal_diff": 0.0,
            "win_rate": 0.0,
            "form_points": 0.0,
        }

    wins = (hist["result"] == 3).mean()
    return {
        "avg_goals_for": float(hist["goals_for"].mean()),
        "avg_goals_against": float(hist["goals_against"].mean()),
        "goal_diff": float((hist["goals_for"] - hist["goals_against"]).mean()),
        "win_rate": float(wins),
        "form_points": float(hist["result"].mean()),
    }


def recent_form_by_venue(
    df: pd.DataFrame,
    team: str,
    is_home: bool,
    n: int = 5,
    before_date: str | None = None,
) -> dict[str, float]:
    hist = team_history(df, team, before_date)
    hist = hist[hist["is_home"] == int(is_home)].tail(n)
    if hist.empty:
        return {"venue_goals_for": 0.0, "venue_goals_against": 0.0, "venue_win_rate": 0.0}
    return {
        "venue_goals_for": float(hist["goals_for"].mean()),
        "venue_goals_against": float(hist["goals_against"].mean()),
        "venue_win_rate": float((hist["result"] == 3).mean()),
    }


def weighted_recent_form(df: pd.DataFrame, team: str, n: int = 5, before_date: str | None = None) -> dict[str, float]:
    hist = team_history(df, team, before_date).tail(n)
    if hist.empty:
        return {"weighted_goal_diff": 0.0, "weighted_form_points": 0.0}
    weights = pd.Series(range(1, len(hist) + 1), index=hist.index)
    goal_diff = hist["goals_for"] - hist["goals_against"]
    return {
        "weighted_goal_diff": float((goal_diff * weights).sum() / weights.sum()),
        "weighted_form_points": float((hist["result"] * weights).sum() / weights.sum()),
    }


def average_home_away_goals(df: pd.DataFrame) -> dict[str, float]:
    played = df.dropna(subset=["home_goals", "away_goals"])
    if played.empty:
        return {"home_goals_avg": 1.0, "away_goals_avg": 1.0}
    return {
        "home_goals_avg": float(played["home_goals"].mean()),
        "away_goals_avg": float(played["away_goals"].mean()),
    }
