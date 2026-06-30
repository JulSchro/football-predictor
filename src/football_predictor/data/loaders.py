from pathlib import Path

import pandas as pd

from football_predictor.data.validators import Match


REQUIRED_MATCH_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "competition",
    "season",
]


def load_matches_csv(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_MATCH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[REQUIRED_MATCH_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["home_team", "away_team", "competition", "season"]:
        df[col] = df[col].astype(str)
    for col in ["home_goals", "away_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    records = []
    for row in df.to_dict("records"):
        clean = {key: (None if pd.isna(value) else value) for key, value in row.items()}
        records.append(Match(**clean).model_dump(mode="json"))
    return pd.DataFrame(records)
