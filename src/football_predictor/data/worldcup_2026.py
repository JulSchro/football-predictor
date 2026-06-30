from pathlib import Path
from urllib.request import urlretrieve
import json

import pandas as pd


OPENFOOTBALL_2026_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
OPENFOOTBALL_WORLD_CUP_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json"


def download_worldcup_2026(path: Path | str, url: str = OPENFOOTBALL_2026_URL) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, output)
    return output


def load_worldcup_2026_json(path: Path | str, include_fixtures: bool = False) -> pd.DataFrame:
    return load_worldcup_json(path, season="2026", include_fixtures=include_fixtures)


def download_worldcup_year(year: int, path: Path | str) -> Path:
    return download_worldcup_2026(path, url=OPENFOOTBALL_WORLD_CUP_URL.format(year=year))


def load_worldcup_json(path: Path | str, season: str, include_fixtures: bool = False) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict] = []

    for match in data.get("matches", []):
        score = match.get("score", {}).get("ft")
        if not include_fixtures and not score:
            continue

        home_goals = int(score[0]) if score else None
        away_goals = int(score[1]) if score else None
        competition = match.get("group") or match.get("round") or "FIFA World Cup 2026"

        rows.append(
            {
                "date": match["date"],
                "home_team": match["team1"],
                "away_team": match["team2"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "competition": competition,
                "season": season,
            }
        )

    return pd.DataFrame(rows)


def load_many_worldcups(paths_by_year: dict[int, Path | str], include_fixtures: bool = False) -> pd.DataFrame:
    frames = [
        load_worldcup_json(path, season=str(year), include_fixtures=include_fixtures)
        for year, path in paths_by_year.items()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
