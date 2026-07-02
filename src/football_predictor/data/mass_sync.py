from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import sqlite3

from football_predictor.data.api_clients import ApiFootballClient
from football_predictor.data.api_football_sync import (
    sync_api_football_fixture_details_bulk,
    sync_api_football_players,
    sync_api_football_standings,
    sync_api_football_team_statistics,
    sync_competition_season_core,
)
from football_predictor.data.competitions import COMPETITIONS
from football_predictor.database.db import upsert_competition


Progress = Callable[[dict], None]


def run_mass_api_football_sync(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    request_budget: int = 1500,
    seasons_per_league: int = 2,
    max_competitions: int | None = None,
    league_ids: list[int] | None = None,
    max_fixture_details_per_season: int = 80,
    player_request_share: float = 0.35,
    progress: Progress | None = None,
) -> dict:
    targets = [
        item
        for _, item in sorted(
            enumerate(COMPETITIONS),
            key=lambda pair: (pair[1].get("priority", 9), pair[0]),
        )
    ]
    if league_ids:
        allowed = {int(league_id) for league_id in league_ids}
        targets = [item for item in targets if int(item["api_football_league_id"]) in allowed]
    if max_competitions:
        targets = targets[:max_competitions]
    summary: dict[str, Any] = {
        "request_budget": request_budget,
        "requests_used": 0,
        "competitions_processed": 0,
        "seasons_processed": 0,
        "leagues_resolved": [],
        "season_results": [],
        "totals": {
            "fixtures": 0,
            "teams": 0,
            "standings": 0,
            "team_statistics": 0,
            "players": 0,
            "squad_rows": 0,
            "season_stat_rows": 0,
            "fixture_details": 0,
            "advanced_stats": 0,
            "lineups": 0,
            "lineup_players": 0,
            "player_match_stats": 0,
            "referee_history": 0,
        },
        "errors": [],
        "skipped": [],
    }
    for competition in targets:
        if summary["requests_used"] >= request_budget:
            break
        league_id = int(competition["api_football_league_id"])
        for season in resolve_league_seasons(client, league_id, seasons_per_league, summary, progress):
            if summary["requests_used"] >= request_budget:
                break
            season_result = sync_one_competition_season(
                conn,
                client,
                competition,
                season,
                request_budget - summary["requests_used"],
                max_fixture_details_per_season=max_fixture_details_per_season,
                player_request_share=player_request_share,
                progress=progress,
            )
            summary["requests_used"] += int(season_result["requests_used"])
            summary["seasons_processed"] += 1
            summary["season_results"].append(season_result)
            for key, value in season_result["totals"].items():
                summary["totals"][key] = summary["totals"].get(key, 0) + value
            summary["errors"].extend(season_result.get("errors", []))
            conn.commit()
        summary["competitions_processed"] += 1
    return summary


def sync_one_competition_season(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    competition: dict,
    season: int,
    remaining_budget: int,
    max_fixture_details_per_season: int,
    player_request_share: float,
    progress: Progress | None = None,
) -> dict:
    league_id = int(competition["api_football_league_id"])
    league_name = competition["name"]
    result: dict[str, Any] = {
        "league_id": league_id,
        "league_name": league_name,
        "season": season,
        "requests_used": 0,
        "totals": {
            "fixtures": 0,
            "teams": 0,
            "standings": 0,
            "team_statistics": 0,
            "players": 0,
            "squad_rows": 0,
            "season_stat_rows": 0,
            "fixture_details": 0,
            "advanced_stats": 0,
            "lineups": 0,
            "lineup_players": 0,
            "player_match_stats": 0,
            "referee_history": 0,
        },
        "steps": [],
        "errors": [],
    }
    upsert_competition(conn, competition, season=season)
    _emit(progress, result, "core", "start", remaining_budget)
    try:
        core = sync_competition_season_core(conn, client, league_id=league_id, season=season, league_name=league_name, missing_only=True)
        core_requests = int(core.get("requests_used") or 0)
        result["requests_used"] += core_requests
        remaining_budget -= core_requests
        after = core.get("after") or {}
        result["totals"]["fixtures"] += int((after.get("fixtures") or {}).get("records_count") or 0)
        result["totals"]["teams"] += int((after.get("teams") or {}).get("records_count") or 0)
        result["steps"].append({"endpoint": "fixtures+teams", "requests": core_requests, "result": compact_core(core)})
        _emit(progress, result, "core", "done", remaining_budget)
    except Exception as exc:
        result["errors"].append({"endpoint": "core", "error": str(exc)})
        _emit(progress, result, "core", "error", remaining_budget)
        return result

    if remaining_budget <= 0:
        return result

    if not has_rows(conn, "api_football_standings", league_id, season):
        _emit(progress, result, "standings", "start", remaining_budget)
        try:
            standings = sync_api_football_standings(conn, client, league=league_id, season=season)
            result["requests_used"] += 1
            remaining_budget -= 1
            result["totals"]["standings"] += int(standings.get("standings") or 0)
            result["steps"].append({"endpoint": "standings", "requests": 1, "result": standings})
            _emit(progress, result, "standings", "done", remaining_budget)
        except Exception as exc:
            result["errors"].append({"endpoint": "standings", "error": str(exc)})
            _emit(progress, result, "standings", "error", remaining_budget)
    else:
        result["steps"].append({"endpoint": "standings", "skipped": True, "reason": "local_exists"})

    if remaining_budget <= 0:
        return result

    missing_team_stats = missing_team_stat_ids(conn, league_id, season)
    if missing_team_stats:
        team_ids = missing_team_stats[:remaining_budget]
        _emit(progress, result, "team_statistics", "start", remaining_budget)
        try:
            stats = sync_api_football_team_statistics(conn, client, league=league_id, season=season, team_ids=team_ids)
            used = int(stats.get("requests_used") or len(team_ids))
            result["requests_used"] += used
            remaining_budget -= used
            result["totals"]["team_statistics"] += int(stats.get("team_statistics") or 0)
            result["steps"].append({"endpoint": "teams/statistics", "requests": used, "result": stats})
            _emit(progress, result, "team_statistics", "done", remaining_budget)
        except Exception as exc:
            result["errors"].append({"endpoint": "teams/statistics", "error": str(exc)})
            _emit(progress, result, "team_statistics", "error", remaining_budget)
    else:
        result["steps"].append({"endpoint": "teams/statistics", "skipped": True, "reason": "local_complete"})

    if remaining_budget <= 0:
        return result

    if players_need_sync(conn, league_id, season):
        player_budget = max(1, min(remaining_budget, int(remaining_budget * player_request_share)))
        _emit(progress, result, "players", "start", remaining_budget)
        try:
            players = sync_api_football_players(conn, client, league=league_id, season=season, max_requests=player_budget)
            used = int(players.get("requests_used") or 0)
            result["requests_used"] += used
            remaining_budget -= used
            result["totals"]["players"] += int(players.get("players") or 0)
            result["totals"]["squad_rows"] += int(players.get("squad_rows") or 0)
            result["totals"]["season_stat_rows"] += int(players.get("season_stat_rows") or 0)
            result["steps"].append({"endpoint": "players", "requests": used, "result": {k: v for k, v in players.items() if k != "errors"}})
            if players.get("errors"):
                result["errors"].extend({"endpoint": "players", **err} for err in players["errors"][:10])
            _emit(progress, result, "players", "done", remaining_budget)
        except Exception as exc:
            result["errors"].append({"endpoint": "players", "error": str(exc)})
            _emit(progress, result, "players", "error", remaining_budget)
    else:
        result["steps"].append({"endpoint": "players", "skipped": True, "reason": "local_complete"})

    if remaining_budget < 5:
        return result

    fixture_limit = min(max_fixture_details_per_season, remaining_budget // 5)
    if fixture_limit > 0:
        _emit(progress, result, "fixture_details", "start", remaining_budget)
        try:
            details = sync_api_football_fixture_details_bulk(
                conn,
                client,
                league=league_id,
                season=season,
                only_finished=True,
                missing_only=True,
                max_fixtures=fixture_limit,
            )
            used = int(details.get("requests_used_estimate") or 0)
            result["requests_used"] += used
            remaining_budget -= used
            result["totals"]["fixture_details"] += int(details.get("fixtures_processed") or 0)
            for key in ["advanced_stats", "lineups", "lineup_players", "player_match_stats", "referee_history"]:
                result["totals"][key] += int(details.get(key) or 0)
            result["steps"].append({"endpoint": "fixture_details", "requests": used, "result": details})
            if details.get("errors"):
                result["errors"].extend({"endpoint": "fixture_details", **err} for err in details["errors"][:10])
            _emit(progress, result, "fixture_details", "done", remaining_budget)
        except Exception as exc:
            result["errors"].append({"endpoint": "fixture_details", "error": str(exc)})
            _emit(progress, result, "fixture_details", "error", remaining_budget)
    return result


def resolve_league_seasons(
    client: ApiFootballClient,
    league_id: int,
    limit: int,
    summary: dict,
    progress: Progress | None = None,
) -> list[int]:
    try:
        payload = client.get("leagues", id=league_id)
        summary["requests_used"] += 1
        response = payload.get("response") or []
        if not response:
            summary["errors"].append({"endpoint": "leagues", "league_id": league_id, "error": "not_found"})
            return []
        seasons = sorted(
            [int(item["year"]) for item in response[0].get("seasons") or [] if item.get("year") is not None],
            reverse=True,
        )
        current = [int(item["year"]) for item in response[0].get("seasons") or [] if item.get("current") and item.get("year") is not None]
        selected = []
        for year in current + seasons:
            if year not in selected and year <= date.today().year:
                selected.append(year)
            if len(selected) >= limit:
                break
        summary["leagues_resolved"].append(
            {
                "league_id": league_id,
                "name": (response[0].get("league") or {}).get("name"),
                "seasons": selected,
            }
        )
        _emit(progress, {"league_id": league_id, "season": None, "requests_used": 1}, "leagues", "done", None)
        return selected
    except Exception as exc:
        summary["errors"].append({"endpoint": "leagues", "league_id": league_id, "error": str(exc)})
        _emit(progress, {"league_id": league_id, "season": None, "requests_used": 1}, "leagues", "error", None)
        return []


def has_rows(conn: sqlite3.Connection, table: str, league_id: int, season: int) -> bool:
    return bool(
        conn.execute(
            f"SELECT 1 FROM {table} WHERE league_id = ? AND season = ? LIMIT 1",
            (league_id, season),
        ).fetchone()
    )


def missing_team_stat_ids(conn: sqlite3.Connection, league_id: int, season: int) -> list[int]:
    return [
        int(row["team_id"])
        for row in conn.execute(
            """
            SELECT ts.team_id
            FROM api_football_team_seasons ts
            LEFT JOIN api_football_team_statistics st
              ON st.league_id = ts.league_id AND st.season = ts.season AND st.team_id = ts.team_id
            WHERE ts.league_id = ? AND ts.season = ? AND st.id IS NULL
            ORDER BY ts.team_name
            """,
            (league_id, season),
        ).fetchall()
    ]


def players_need_sync(conn: sqlite3.Connection, league_id: int, season: int) -> bool:
    inventory = conn.execute(
        """
        SELECT status
        FROM sync_inventory
        WHERE provider = 'api_football'
          AND entity_type = 'competition_season'
          AND league_id = ?
          AND season = ?
          AND data_type = 'players'
        """,
        (league_id, season),
    ).fetchone()
    if inventory and inventory["status"] == "complete":
        return False
    return bool(
        conn.execute(
            """
            SELECT 1
            FROM api_football_team_seasons ts
            WHERE ts.league_id = ? AND ts.season = ?
              AND NOT EXISTS (
                SELECT 1 FROM player_season_stats ps
                WHERE ps.competition_id = ts.league_id
                  AND ps.season = ts.season
                  AND ps.team_name = ts.team_name
              )
            LIMIT 1
            """,
            (league_id, season),
        ).fetchone()
    )


def compact_core(core: dict) -> dict:
    return {
        "requests_used": core.get("requests_used"),
        "fixtures": ((core.get("after") or {}).get("fixtures") or {}).get("records_count"),
        "teams": ((core.get("after") or {}).get("teams") or {}).get("records_count"),
        "actions": [
            {key: value for key, value in action.items() if key in {"data_type", "skipped", "reason", "fixtures", "teams", "api_results"}}
            for action in core.get("actions", [])
        ],
    }


def _emit(progress: Progress | None, season_result: dict, endpoint: str, status: str, remaining_budget: int | None) -> None:
    if progress:
        progress(
            {
                "league_id": season_result.get("league_id"),
                "league_name": season_result.get("league_name"),
                "season": season_result.get("season"),
                "endpoint": endpoint,
                "status": status,
                "requests_used": season_result.get("requests_used", 0),
                "remaining_budget": remaining_budget,
            }
        )
