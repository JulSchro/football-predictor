import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    request = Request(url, headers=headers or {})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(2**attempt)
        except URLError as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise last_error or RuntimeError("request failed")


class ApiFootballClient:
    base_url = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (api_key or _env_value("API_FOOTBALL_KEY") or "").strip().strip("\"'")
        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY is required")

    def get(self, endpoint: str, **params: str | int) -> dict:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        return _get_json(url, headers={"x-apisports-key": self.api_key})

    def fixtures(self, league: int, season: int) -> dict:
        return self.get("fixtures", league=league, season=season)

    def fixtures_by_date(self, date: str, league: int | None = None, season: int | None = None) -> dict:
        return self.get("fixtures", date=date, league=league, season=season)

    def fixture_statistics(self, fixture: int) -> dict:
        return self.get("fixtures/statistics", fixture=fixture)

    def fixture_events(self, fixture: int) -> dict:
        return self.get("fixtures/events", fixture=fixture)

    def fixture_lineups(self, fixture: int) -> dict:
        return self.get("fixtures/lineups", fixture=fixture)

    def fixture_players(self, fixture: int) -> dict:
        return self.get("fixtures/players", fixture=fixture)

    def leagues(self, search: str | None = None, country: str | None = None, season: int | None = None) -> dict:
        return self.get("leagues", search=search, country=country, season=season)

    def teams(self, league: int, season: int) -> dict:
        return self.get("teams", league=league, season=season)

    def team_statistics(self, league: int, season: int, team: int) -> dict:
        return self.get("teams/statistics", league=league, season=season, team=team)

    def standings(self, league: int, season: int) -> dict:
        return self.get("standings", league=league, season=season)

    def players(self, league: int, season: int, team: int | None = None, page: int = 1) -> dict:
        return self.get("players", league=league, season=season, team=team, page=page)

    def injuries(self, fixture: int) -> dict:
        return self.get("injuries", fixture=fixture)


class FootballDataOrgClient:
    base_url = "https://api.football-data.org/v4"

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.getenv("FOOTBALL_DATA_ORG_TOKEN")
        if not self.token:
            raise ValueError("FOOTBALL_DATA_ORG_TOKEN is required")

    def get(self, endpoint: str, **params: str | int) -> dict:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        return _get_json(url, headers={"X-Auth-Token": self.token})


class TheSportsDBClient:
    base_url = "https://www.thesportsdb.com/api/v1/json"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("THESPORTSDB_API_KEY", "3")

    def get(self, endpoint: str, **params: str) -> dict:
        query = urlencode({key: value for key, value in params.items() if value})
        url = f"{self.base_url}/{self.api_key}/{endpoint}"
        if query:
            url = f"{url}?{query}"
        return _get_json(url)

    def search_team(self, team: str) -> dict:
        return self.get("searchteams.php", t=team)

    def search_players(self, team: str) -> dict:
        return self.get("searchplayers.php", t=team)


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    dotenv = Path(".env")
    if not dotenv.exists():
        return None
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return None
