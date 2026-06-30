from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


INTERNATIONAL_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def download_international_results(path: Path | str, url: str = INTERNATIONAL_RESULTS_URL) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, output)
    return output


def competition_weight(tournament: str) -> float:
    name = tournament.lower()
    if "fifa world cup" in name and "qualification" not in name:
        return 1.2
    if "uefa euro" in name or "copa américa" in name or "copa america" in name:
        return 1.1
    if "qualification" in name or "qualifier" in name:
        return 1.0
    if "nations league" in name:
        return 0.85
    if "friendly" in name:
        return 0.3
    return 0.65


def load_international_results(path: Path | str, start_year: int = 2010) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= start_year].copy()
    df["competition"] = df["tournament"]
    df["season"] = df["date"].dt.year.astype(str)
    df["home_goals"] = df["home_score"]
    df["away_goals"] = df["away_score"]
    df["match_weight"] = df["tournament"].map(competition_weight)
    return df[
        [
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "competition",
            "season",
            "city",
            "country",
            "neutral",
            "match_weight",
        ]
    ].assign(date=lambda x: x["date"].dt.date.astype(str))

