from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


BASE_URL = "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main"
FILES = {
    "venues": "venues.csv",
    "teams": "teams.csv",
    "players": "squads_and_players.csv",
    "matches": "matches.csv",
    "stats": "match_team_stats.csv",
    "referees": "referees.csv",
}


def download_worldcup_enriched(raw_dir: Path | str) -> dict[str, Path]:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in FILES.items():
        output = raw_path / filename
        urlretrieve(f"{BASE_URL}/{filename}", output)
        paths[key] = output
    return paths


def load_worldcup_enriched(raw_dir: Path | str) -> dict[str, pd.DataFrame]:
    raw_path = Path(raw_dir)
    return {key: pd.read_csv(raw_path / filename) for key, filename in FILES.items()}


def team_lookup(teams: pd.DataFrame) -> dict[int, str]:
    return {int(row.team_id): str(row.team_name) for row in teams.itertuples()}


def summarize_squads(teams: pd.DataFrame, players: pd.DataFrame) -> list[dict]:
    lookup = team_lookup(teams)
    rows = []
    for team_id, group in players.groupby("team_id"):
        values = pd.to_numeric(group["market_value_eur"], errors="coerce").fillna(0)
        ages = (pd.Timestamp("2026-06-11") - pd.to_datetime(group["date_of_birth"], errors="coerce")).dt.days / 365.25
        rows.append(
            {
                "team": lookup.get(int(team_id), str(team_id)),
                "source": "wc2026_enriched_dataset",
                "squad_value_eur": float(values.sum()),
                "squad_size": int(len(group)),
                "avg_age": float(ages.mean()) if not ages.isna().all() else None,
                "raw": {"team_id": int(team_id)},
            }
        )
    return rows


def normalize_venues(venues: pd.DataFrame) -> list[dict]:
    return [
        {
            "source_id": str(row.venue_id),
            "stadium_name": str(row.stadium_name),
            "city": str(row.city),
            "country": str(row.country),
            "capacity": int(row.capacity),
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "altitude_m": float(row.elevation_meters),
            "surface": "natural",
            "roof": None,
            "raw": row._asdict(),
        }
        for row in venues.itertuples(index=False)
    ]


def normalize_players(teams: pd.DataFrame, players: pd.DataFrame) -> list[dict]:
    lookup = team_lookup(teams)
    rows = []
    for row in players.itertuples(index=False):
        rows.append(
            {
                "source_id": str(row.player_id),
                "team": lookup.get(int(row.team_id), str(row.team_id)),
                "player_name": str(row.player_name),
                "position": str(row.position),
                "club_team": str(row.club_team),
                "market_value_eur": float(row.market_value_eur),
                "caps": int(row.caps),
                "date_of_birth": str(row.date_of_birth),
                "height_cm": float(row.height_cm),
                "goals": int(row.goals),
                "raw": row._asdict(),
            }
        )
    return rows


def normalize_advanced_stats(data: dict[str, pd.DataFrame]) -> list[dict]:
    teams = team_lookup(data["teams"])
    matches = data["matches"].set_index("match_id")
    rows = []
    for row in data["stats"].itertuples(index=False):
        match = matches.loc[int(row.match_id)]
        home = teams.get(int(match.home_team_id), str(match.home_team_id))
        away = teams.get(int(match.away_team_id), str(match.away_team_id))
        team = teams.get(int(row.team_id), str(row.team_id))
        opponent = away if team == home else home
        rows.append(
            {
                "source_match_id": str(row.match_id),
                "team": team,
                "opponent": opponent,
                "date": str(match.date),
                "xg": float(match.home_xg if team == home else match.away_xg),
                "possession_pct": int(row.possession_pct),
                "total_shots": int(row.total_shots),
                "shots_on_target": int(row.shots_on_target),
                "corners": int(row.corners),
                "fouls": int(row.fouls),
                "offsides": int(row.offsides),
                "saves": int(row.saves),
                "cards_estimate": None,
                "raw": row._asdict(),
            }
        )
    return rows
