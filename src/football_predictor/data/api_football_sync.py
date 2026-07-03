from __future__ import annotations

import json
from typing import Any

import sqlite3

from football_predictor.data.api_clients import ApiFootballClient
from football_predictor.database.db import (
    get_sync_inventory,
    find_venue,
    insert_matches,
    upsert_advanced_stats,
    upsert_api_football_fixture,
    upsert_api_football_fixture_lineup,
    upsert_api_football_fixture_player,
    upsert_api_football_league_coverage,
    upsert_api_football_standing,
    upsert_api_football_team_statistics,
    upsert_api_football_team_season,
    upsert_match_context,
    upsert_player,
    upsert_player_availability,
    upsert_player_match_stats,
    upsert_player_season_stats,
    upsert_referee_match_history,
    upsert_sync_inventory,
    upsert_team_squad,
    upsert_teams,
    parse_referee_name,
    rebuild_all_referee_metrics,
)
from football_predictor.prediction.match_context import normalize_competition_context


FINISHED_STATUSES = {"FT", "AET", "PEN"}


def normalize_league_coverage(item: dict[str, Any]) -> list[dict]:
    league = item.get("league") or {}
    country = item.get("country") or {}
    rows = []
    for season in item.get("seasons") or []:
        coverage = season.get("coverage") or {}
        fixtures = coverage.get("fixtures") or {}
        standings = coverage.get("standings")
        rows.append(
            {
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "league_type": league.get("type"),
                "country": country.get("name"),
                "country_code": country.get("code"),
                "season": season.get("year"),
                "current": int(bool(season.get("current"))) if season.get("current") is not None else None,
                "start_date": season.get("start"),
                "end_date": season.get("end"),
                "fixtures_events": _coverage_bool(fixtures.get("events")),
                "fixtures_lineups": _coverage_bool(fixtures.get("lineups")),
                "fixtures_statistics_fixtures": _coverage_bool(fixtures.get("statistics_fixtures", fixtures.get("statistics-fixtures"))),
                "fixtures_statistics_players": _coverage_bool(fixtures.get("statistics_players", fixtures.get("statistics-players"))),
                "standings": _coverage_bool(standings),
                "players": _coverage_bool(coverage.get("players")),
                "top_scorers": _coverage_bool(coverage.get("top_scorers")),
                "top_assists": _coverage_bool(coverage.get("top_assists")),
                "top_cards": _coverage_bool(coverage.get("top_cards")),
                "injuries": _coverage_bool(coverage.get("injuries")),
                "predictions": _coverage_bool(coverage.get("predictions")),
                "odds": _coverage_bool(coverage.get("odds")),
                "raw": {"league": league, "country": country, "season": season},
            }
        )
    return [row for row in rows if row["league_id"] is not None and row["season"] is not None]


def normalize_fixture(row: dict[str, Any]) -> dict:
    fixture = row.get("fixture", {})
    league = row.get("league", {})
    teams = row.get("teams", {})
    venue = fixture.get("venue") or {}
    status = fixture.get("status") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    return {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("date"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "season": league.get("season"),
        "home_team": home.get("name"),
        "away_team": away.get("name"),
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "status_short": status.get("short"),
        "venue_name": venue.get("name"),
        "venue_city": venue.get("city"),
        "raw": row,
    }


def fixture_to_match_row(row: dict[str, Any]) -> dict | None:
    normalized = normalize_fixture(row)
    goals = row.get("goals") or {}
    if normalized["status_short"] not in FINISHED_STATUSES:
        return None
    if goals.get("home") is None or goals.get("away") is None:
        return None
    return {
        "date": str(normalized["date"] or "")[:10],
        "home_team": normalized["home_team"],
        "away_team": normalized["away_team"],
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "competition": normalized["league_name"],
        "season": str(normalized["season"] or ""),
    }


def fixture_to_context_row(row: dict[str, Any], conn: sqlite3.Connection | None = None) -> dict | None:
    normalized = normalize_fixture(row)
    if normalized["fixture_id"] is None:
        return None
    league = row.get("league") or {}
    fixture = row.get("fixture") or {}
    venue = fixture.get("venue") or {}
    matched_venue = find_venue(conn, venue.get("name"), venue.get("city")) if conn is not None else None
    context = normalize_competition_context(normalized["league_name"], league.get("round"))
    return {
        "match_key": f"api-football:{normalized['fixture_id']}",
        "date": str(normalized["date"] or "")[:10] or None,
        "home_team": normalized["home_team"],
        "away_team": normalized["away_team"],
        "venue_id": matched_venue.get("id") if matched_venue else None,
        "stadium_name": (matched_venue.get("stadium_name") if matched_venue else venue.get("name")),
        "city": (matched_venue.get("city") if matched_venue else venue.get("city")),
        "country": matched_venue.get("country") if matched_venue else None,
        "neutral": None,
        "competition_weight": round(float(context["pressure_index"]) / 100, 3),
        "stage": context["stage"],
        "raw": {
            "provider": "api-football",
            "fixture_id": normalized["fixture_id"],
            "league_id": normalized["league_id"],
            "league_name": normalized["league_name"],
            "league_round": league.get("round"),
            "context": context,
        },
    }


def normalize_fixture_statistics(
    fixture_id: int,
    fixture_row: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict]:
    fixture = normalize_fixture(fixture_row)
    response = payload.get("response") or []
    teams = {fixture["home_team"]: fixture["away_team"], fixture["away_team"]: fixture["home_team"]}
    rows = []
    for item in response:
        team_name = (item.get("team") or {}).get("name")
        if not team_name:
            continue
        values = {_normal_key(stat.get("type")): _to_number(stat.get("value")) for stat in item.get("statistics", [])}
        yellow = values.get("yellow_cards")
        red = values.get("red_cards")
        cards = None
        if yellow is not None or red is not None:
            cards = (yellow or 0) + (red or 0) * 2
        rows.append(
            {
                "source_match_id": f"api-football:{fixture_id}",
                "team": team_name,
                "opponent": teams.get(team_name),
                "date": str(fixture["date"] or "")[:10],
                "xg": values.get("expected_goals") or values.get("xg"),
                "possession_pct": values.get("ball_possession"),
                "total_shots": values.get("total_shots"),
                "shots_on_target": values.get("shots_on_goal"),
                "shots_off_target": values.get("shots_off_goal"),
                "blocked_shots": values.get("blocked_shots"),
                "corners": values.get("corner_kicks"),
                "fouls": values.get("fouls"),
                "offsides": values.get("offsides"),
                "saves": values.get("goalkeeper_saves"),
                "yellow_cards": yellow,
                "red_cards": red,
                "cards_estimate": cards,
                "passes_total": values.get("total_passes"),
                "passes_accurate": values.get("passes_accurate"),
                "pass_accuracy_pct": values.get("passes_%"),
                "attacks": values.get("attacks"),
                "dangerous_attacks": values.get("dangerous_attacks"),
                "raw": item,
            }
        )
    return rows


def normalize_referee_history(
    fixture_row: dict[str, Any],
    stats_rows: list[dict],
    events_payload: dict[str, Any],
) -> dict | None:
    fixture = normalize_fixture(fixture_row)
    fixture_info = fixture_row.get("fixture") or {}
    league = fixture_row.get("league") or {}
    referee_name, referee_country = parse_referee_name(fixture_info.get("referee"))
    if not referee_name or not fixture["fixture_id"]:
        return None
    home_team = fixture["home_team"]
    away_team = fixture["away_team"]
    stats_by_team = {row.get("team"): row for row in stats_rows}
    home_stats = stats_by_team.get(home_team, {})
    away_stats = stats_by_team.get(away_team, {})
    events = events_payload.get("response") or []
    card_counts = _event_card_counts(events, home_team, away_team)
    home_yellow = _first_number(home_stats.get("yellow_cards"), card_counts["home_yellow"])
    away_yellow = _first_number(away_stats.get("yellow_cards"), card_counts["away_yellow"])
    home_red = _first_number(home_stats.get("red_cards"), card_counts["home_red"])
    away_red = _first_number(away_stats.get("red_cards"), card_counts["away_red"])
    home_cards = _first_number(home_stats.get("cards_estimate"), (home_yellow or 0) + (home_red or 0) * 2)
    away_cards = _first_number(away_stats.get("cards_estimate"), (away_yellow or 0) + (away_red or 0) * 2)
    home_fouls = _first_number(home_stats.get("fouls"))
    away_fouls = _first_number(away_stats.get("fouls"))
    return {
        "fixture_id": fixture["fixture_id"],
        "referee_name": referee_name,
        "referee_country": referee_country,
        "date": str(fixture["date"] or "")[:10],
        "league_id": fixture["league_id"],
        "league_name": fixture["league_name"],
        "season": fixture["season"],
        "round": league.get("round"),
        "home_team": home_team,
        "away_team": away_team,
        "home_yellow": home_yellow,
        "away_yellow": away_yellow,
        "home_red": home_red,
        "away_red": away_red,
        "home_cards": home_cards,
        "away_cards": away_cards,
        "total_yellow": _sum_optional(home_yellow, away_yellow),
        "total_red": _sum_optional(home_red, away_red),
        "total_cards": _sum_optional(home_cards, away_cards),
        "home_fouls": home_fouls,
        "away_fouls": away_fouls,
        "total_fouls": _sum_optional(home_fouls, away_fouls),
        "penalties": _event_penalties(events),
        "raw_events": events_payload,
        "raw_fixture": fixture_row,
    }


def normalize_injuries(payload: dict[str, Any]) -> list[dict]:
    rows = []
    for item in payload.get("response") or []:
        player = item.get("player") or {}
        team = item.get("team") or {}
        fixture = item.get("fixture") or {}
        rows.append(
            {
                "source": "api-football",
                "fixture_id": fixture.get("id"),
                "team": team.get("name"),
                "player_name": player.get("name"),
                "reason": player.get("reason") or item.get("reason"),
                "status": player.get("type") or item.get("type"),
                "raw": item,
            }
        )
    return [row for row in rows if row["team"] and row["player_name"]]


def normalize_team_season(row: dict[str, Any], league_id: int, season: int) -> dict:
    team = row.get("team") or {}
    venue = row.get("venue") or {}
    return {
        "league_id": league_id,
        "season": season,
        "team_id": team.get("id"),
        "team_name": team.get("name"),
        "country": team.get("country"),
        "founded": team.get("founded"),
        "national": int(bool(team.get("national"))) if team.get("national") is not None else None,
        "venue_name": venue.get("name"),
        "venue_city": venue.get("city"),
        "raw": row,
    }


def normalize_team_statistics(payload: dict[str, Any], league_id: int, season: int, team_id: int) -> dict | None:
    response = payload.get("response") or {}
    team = response.get("team") or {}
    league = response.get("league") or {}
    fixtures = response.get("fixtures") or {}
    goals = response.get("goals") or {}
    clean_sheet = response.get("clean_sheet") or {}
    failed_to_score = response.get("failed_to_score") or {}
    if not team.get("id") and not team_id:
        return None
    return {
        "league_id": league.get("id") or league_id,
        "league_name": league.get("name"),
        "season": league.get("season") or season,
        "team_id": team.get("id") or team_id,
        "team_name": team.get("name") or str(team_id),
        "played": _nested(fixtures, "played", "total"),
        "wins": _nested(fixtures, "wins", "total"),
        "draws": _nested(fixtures, "draws", "total"),
        "losses": _nested(fixtures, "loses", "total"),
        "goals_for": _nested(goals, "for", "total", "total"),
        "goals_against": _nested(goals, "against", "total", "total"),
        "home_played": _nested(fixtures, "played", "home"),
        "home_wins": _nested(fixtures, "wins", "home"),
        "away_played": _nested(fixtures, "played", "away"),
        "away_wins": _nested(fixtures, "wins", "away"),
        "clean_sheets": clean_sheet.get("total"),
        "failed_to_score": failed_to_score.get("total"),
        "form": response.get("form"),
        "raw": response,
    }


def normalize_standings(payload: dict[str, Any], league_id: int, season: int) -> list[dict]:
    rows = []
    for block in payload.get("response") or []:
        league = block.get("league") or {}
        for group in league.get("standings") or []:
            for item in group or []:
                team = item.get("team") or {}
                all_stats = item.get("all") or {}
                goals = all_stats.get("goals") or {}
                if not team.get("id"):
                    continue
                rows.append(
                    {
                        "league_id": league.get("id") or league_id,
                        "league_name": league.get("name"),
                        "season": league.get("season") or season,
                        "group_name": item.get("group"),
                        "rank": item.get("rank"),
                        "team_id": team.get("id"),
                        "team_name": team.get("name"),
                        "points": item.get("points"),
                        "goals_diff": item.get("goalsDiff"),
                        "form": item.get("form"),
                        "status": item.get("status"),
                        "description": item.get("description"),
                        "played": all_stats.get("played"),
                        "wins": all_stats.get("win"),
                        "draws": all_stats.get("draw"),
                        "losses": all_stats.get("lose"),
                        "goals_for": goals.get("for"),
                        "goals_against": goals.get("against"),
                        "raw": item,
                    }
                )
    return rows


def normalize_lineups(fixture_id: int, payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    lineup_rows = []
    player_rows = []
    for item in payload.get("response") or []:
        team = item.get("team") or {}
        coach = item.get("coach") or {}
        team_id = team.get("id")
        team_name = team.get("name")
        if not team_id or not team_name:
            continue
        lineup_rows.append(
            {
                "fixture_id": fixture_id,
                "team_id": team_id,
                "team_name": team_name,
                "formation": item.get("formation"),
                "coach_id": coach.get("id"),
                "coach_name": coach.get("name"),
                "raw": item,
            }
        )
        for is_starting, collection_name in [(1, "startXI"), (0, "substitutes")]:
            for entry in item.get(collection_name) or []:
                player = entry.get("player") or {}
                if not player.get("name"):
                    continue
                player_rows.append(
                    {
                        "fixture_id": fixture_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player.get("id"),
                        "player_name": player.get("name"),
                        "number": player.get("number"),
                        "position": player.get("pos"),
                        "grid": player.get("grid"),
                        "is_starting": is_starting,
                        "raw": player,
                    }
                )
    return lineup_rows, player_rows


def normalize_player_statistics(payload: dict[str, Any], league_id: int, season: int) -> tuple[list[dict], list[dict], list[dict]]:
    players = []
    squads = []
    season_stats = []
    for item in payload.get("response") or []:
        player = item.get("player") or {}
        stats_list = item.get("statistics") or []
        player_id = player.get("id")
        name = player.get("name")
        if not player_id or not name:
            continue
        player_row = {
            "api_player_id": player_id,
            "name": name,
            "firstname": player.get("firstname"),
            "lastname": player.get("lastname"),
            "birth_date": _nested(player, "birth", "date"),
            "age": player.get("age"),
            "nationality": player.get("nationality"),
            "height": player.get("height"),
            "weight": player.get("weight"),
            "preferred_position": None,
            "photo_url": player.get("photo"),
            "raw": player,
        }
        players.append(player_row)
        for stats in stats_list:
            team = stats.get("team") or {}
            league = stats.get("league") or {}
            games = stats.get("games") or {}
            goals = stats.get("goals") or {}
            shots = stats.get("shots") or {}
            passes = stats.get("passes") or {}
            tackles = stats.get("tackles") or {}
            duels = stats.get("duels") or {}
            cards = stats.get("cards") or {}
            substitutes = stats.get("substitutes") or {}
            team_id = team.get("id")
            team_name = team.get("name")
            if not team_name:
                continue
            competition_id = league.get("id") or league_id
            competition_name = league.get("name")
            row_season = league.get("season") or season
            position = games.get("position")
            player_row["preferred_position"] = player_row["preferred_position"] or position
            squads.append(
                {
                    "api_team_id": team_id,
                    "team_name": team_name,
                    "player_id": player_id,
                    "api_player_id": player_id,
                    "competition_id": competition_id,
                    "competition_name": competition_name,
                    "season": row_season,
                    "squad_number": None,
                    "position": position,
                    "is_active": 1,
                    "joined_at": None,
                    "left_at": None,
                    "source": "api_football",
                    "raw": {"player": player, "statistics": stats},
                }
            )
            season_stats.append(
                {
                    "api_player_id": player_id,
                    "player_id": player_id,
                    "api_team_id": team_id,
                    "team_name": team_name,
                    "competition_id": competition_id,
                    "competition_name": competition_name,
                    "season": row_season,
                    "appearances": games.get("appearences"),
                    "lineups": games.get("lineups"),
                    "minutes": games.get("minutes"),
                    "goals": goals.get("total"),
                    "assists": goals.get("assists"),
                    "shots": shots.get("total"),
                    "shots_on_target": shots.get("on"),
                    "passes": passes.get("total"),
                    "key_passes": passes.get("key"),
                    "pass_accuracy": _to_number(passes.get("accuracy")),
                    "tackles": tackles.get("total"),
                    "interceptions": tackles.get("interceptions"),
                    "duels_won": duels.get("won"),
                    "yellow_cards": cards.get("yellow"),
                    "red_cards": cards.get("red"),
                    "rating": _to_number(games.get("rating")),
                    "source": "api_football",
                    "raw": {"player": player, "statistics": stats, "substitutes": substitutes},
                }
            )
    return players, squads, season_stats


def normalize_fixture_player_statistics(fixture_id: int, payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    players = []
    match_stats = []
    for team_block in payload.get("response") or []:
        team = team_block.get("team") or {}
        team_id = team.get("id")
        team_name = team.get("name")
        if not team_name:
            continue
        for item in team_block.get("players") or []:
            player = item.get("player") or {}
            stats_list = item.get("statistics") or []
            api_player_id = player.get("id")
            player_name = player.get("name")
            if not api_player_id or not player_name:
                continue
            players.append(
                {
                    "api_player_id": api_player_id,
                    "name": player_name,
                    "photo_url": player.get("photo"),
                    "raw": player,
                }
            )
            for stats in stats_list:
                games = stats.get("games") or {}
                shots = stats.get("shots") or {}
                goals = stats.get("goals") or {}
                passes = stats.get("passes") or {}
                tackles = stats.get("tackles") or {}
                duels = stats.get("duels") or {}
                cards = stats.get("cards") or {}
                match_stats.append(
                    {
                        "fixture_id": fixture_id,
                        "match_id": None,
                        "api_team_id": team_id,
                        "team_name": team_name,
                        "api_player_id": api_player_id,
                        "player_id": api_player_id,
                        "position": games.get("position"),
                        "is_starter": 0 if games.get("substitute") else 1,
                        "minutes": games.get("minutes"),
                        "goals": goals.get("total"),
                        "assists": goals.get("assists"),
                        "shots": shots.get("total"),
                        "shots_on_target": shots.get("on"),
                        "passes": passes.get("total"),
                        "key_passes": passes.get("key"),
                        "tackles": tackles.get("total"),
                        "interceptions": tackles.get("interceptions"),
                        "duels_won": duels.get("won"),
                        "yellow_cards": cards.get("yellow"),
                        "red_cards": cards.get("red"),
                        "rating": _to_number(games.get("rating")),
                        "source": "api_football",
                        "raw": {"team": team, "player": player, "statistics": stats},
                    }
                )
    return players, match_stats


def sync_api_football_fixtures(conn: sqlite3.Connection, client: ApiFootballClient, league: int, season: int) -> dict:
    payload = client.fixtures(league=league, season=season)
    return _store_fixture_payload(conn, payload)


def sync_api_football_fixtures_by_date(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    date: str,
    league: int | None = None,
    season: int | None = None,
    sync_finished_details: bool = True,
    max_finished_details: int = 50,
) -> dict:
    payload = client.fixtures_by_date(date=date, league=league, season=season)
    result = _store_fixture_payload(conn, payload)
    detail_totals = {
        "finished_fixtures_detected": 0,
        "fixture_details_synced": 0,
        "advanced_stats": 0,
        "availability_rows": 0,
        "lineups": 0,
        "lineup_players": 0,
        "player_match_stats": 0,
        "referee_history": 0,
        "detail_errors": [],
    }
    if sync_finished_details:
        finished = [
            row
            for row in payload.get("response") or []
            if ((row.get("fixture") or {}).get("status") or {}).get("short") in {"FT", "AET", "PEN"}
        ]
        detail_totals["finished_fixtures_detected"] = len(finished)
        for row in finished[: max(0, max_finished_details)]:
            fixture_id = ((row.get("fixture") or {}).get("id"))
            if fixture_id is None:
                continue
            try:
                details = sync_api_football_fixture_details(conn, client, int(fixture_id), row)
            except Exception as exc:  # keep the daily sync alive; report the bad fixture.
                detail_totals["detail_errors"].append({"fixture_id": fixture_id, "error": str(exc)})
                continue
            detail_totals["fixture_details_synced"] += 1
            for key in [
                "advanced_stats",
                "availability_rows",
                "lineups",
                "lineup_players",
                "player_match_stats",
                "referee_history",
            ]:
                detail_totals[key] += int(details.get(key) or 0)
    result["date"] = date
    result["league"] = league
    result["season"] = season
    result["finished_details"] = detail_totals
    return result


def _store_fixture_payload(conn: sqlite3.Connection, payload: dict) -> dict:
    fixtures = payload.get("response") or []
    match_rows = []
    teams = []
    for row in fixtures:
        normalized = normalize_fixture(row)
        if normalized["fixture_id"] is None:
            continue
        upsert_api_football_fixture(conn, normalized)
        context_row = fixture_to_context_row(row, conn)
        if context_row:
            upsert_match_context(conn, context_row)
        teams.extend([normalized["home_team"], normalized["away_team"]])
        match = fixture_to_match_row(row)
        if match:
            match_rows.append(match)
    upsert_teams(conn, [team for team in teams if team])
    inserted = insert_matches(conn, match_rows) if match_rows else 0
    return {
        "fixtures": len(fixtures),
        "finished_matches_inserted": inserted,
        "api_results": payload.get("results"),
        "api_errors": payload.get("errors") or {},
        "api_paging": payload.get("paging") or {},
    }


def local_competition_season_status(conn: sqlite3.Connection, league_id: int, season: int, league_name: str | None = None) -> dict:
    fixture_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM api_football_fixtures
        WHERE league_id = ? AND season = ?
        """,
        (league_id, season),
    ).fetchone()[0]
    finished_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM api_football_fixtures
        WHERE league_id = ? AND season = ? AND status_short IN ('FT', 'AET', 'PEN')
        """,
        (league_id, season),
    ).fetchone()[0]
    team_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM api_football_team_seasons
        WHERE league_id = ? AND season = ?
        """,
        (league_id, season),
    ).fetchone()[0]
    context_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM match_context mc
        JOIN api_football_fixtures f ON mc.match_key = 'api-football:' || f.fixture_id
        WHERE f.league_id = ? AND f.season = ?
        """,
        (league_id, season),
    ).fetchone()[0]
    inventory = {
        row["data_type"]: dict(row)
        for row in conn.execute(
            """
            SELECT *
            FROM sync_inventory
            WHERE provider = 'api_football'
              AND entity_type = 'competition_season'
              AND league_id = ?
              AND season = ?
            """,
            (league_id, season),
        ).fetchall()
    }
    return {
        "league_id": league_id,
        "league_name": league_name,
        "season": season,
        "fixtures": {
            "status": _local_status(fixture_count, inventory.get("fixtures")),
            "records_count": fixture_count,
            "inventory": inventory.get("fixtures"),
        },
        "finished_results": {
            "status": "complete" if finished_count else "missing",
            "records_count": finished_count,
        },
        "teams": {
            "status": _local_status(team_count, inventory.get("teams")),
            "records_count": team_count,
            "inventory": inventory.get("teams"),
        },
        "match_context": {
            "status": "complete" if context_count and fixture_count and context_count >= fixture_count else _local_status(context_count, None),
            "records_count": context_count,
        },
    }


def sync_competition_season_core(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    league_id: int,
    season: int,
    league_name: str | None = None,
    missing_only: bool = True,
) -> dict:
    before = local_competition_season_status(conn, league_id, season, league_name)
    requests_used = 0
    results: dict[str, Any] = {"before": before, "actions": []}

    if not missing_only or before["fixtures"]["status"] != "complete":
        fixture_result = sync_api_football_fixtures(conn, client, league=league_id, season=season)
        requests_used += 1
        upsert_sync_inventory(
            conn,
            {
                "provider": "api_football",
                "entity_type": "competition_season",
                "league_id": league_id,
                "league_name": league_name or _first_league_name(conn, league_id, season),
                "season": season,
                "data_type": "fixtures",
                "status": "complete" if fixture_result["fixtures"] else "missing",
                "records_count": fixture_result["fixtures"],
                "expected_count": fixture_result["fixtures"] or None,
                "request_cost": 1,
                "expires_at": None,
                "error_message": json.dumps(fixture_result.get("api_errors") or {}) if fixture_result.get("api_errors") else None,
                "raw": fixture_result,
            },
        )
        results["actions"].append({"data_type": "fixtures", **fixture_result})
    else:
        results["actions"].append({"data_type": "fixtures", "skipped": True, "reason": "local_complete"})

    after_fixtures = local_competition_season_status(conn, league_id, season, league_name)
    if not missing_only or after_fixtures["teams"]["status"] != "complete":
        teams_payload = client.teams(league=league_id, season=season)
        requests_used += 1
        team_rows = []
        for item in teams_payload.get("response") or []:
            normalized = normalize_team_season(item, league_id, season)
            if normalized["team_id"] is None or not normalized["team_name"]:
                continue
            upsert_api_football_team_season(conn, normalized)
            team_rows.append(normalized)
        upsert_teams(conn, [row["team_name"] for row in team_rows])
        upsert_sync_inventory(
            conn,
            {
                "provider": "api_football",
                "entity_type": "competition_season",
                "league_id": league_id,
                "league_name": league_name or _first_league_name(conn, league_id, season),
                "season": season,
                "data_type": "teams",
                "status": "complete" if team_rows else "missing",
                "records_count": len(team_rows),
                "expected_count": len(team_rows) or None,
                "request_cost": 1,
                "expires_at": None,
                "error_message": json.dumps(teams_payload.get("errors") or {}) if teams_payload.get("errors") else None,
                "raw": {"api_results": teams_payload.get("results"), "api_errors": teams_payload.get("errors") or {}},
            },
        )
        results["actions"].append(
            {
                "data_type": "teams",
                "teams": len(team_rows),
                "api_results": teams_payload.get("results"),
                "api_errors": teams_payload.get("errors") or {},
            }
        )
    else:
        results["actions"].append({"data_type": "teams", "skipped": True, "reason": "local_complete"})

    context_result = backfill_match_context_from_api_fixtures(conn, league_id=league_id, season=season)
    results["actions"].append({"data_type": "match_context", **context_result})
    results["after"] = local_competition_season_status(conn, league_id, season, league_name)
    results["requests_used"] = requests_used
    return results


def backfill_match_context_from_api_fixtures(
    conn: sqlite3.Connection,
    league_id: int | None = None,
    season: int | None = None,
) -> dict:
    clauses = []
    params: list[int] = []
    if league_id is not None:
        clauses.append("league_id = ?")
        params.append(league_id)
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT raw_json
        FROM api_football_fixtures
        {where_sql}
        """,
        tuple(params),
    ).fetchall()
    upserted = 0
    for row in rows:
        raw = json.loads(row["raw_json"] or "{}")
        context_row = fixture_to_context_row(raw)
        if not context_row:
            continue
        upsert_match_context(conn, context_row)
        upserted += 1
    return {"contexts_upserted": upserted}


def sync_api_football_standings(conn: sqlite3.Connection, client: ApiFootballClient, league: int, season: int) -> dict:
    payload = client.standings(league=league, season=season)
    rows = normalize_standings(payload, league_id=league, season=season)
    for row in rows:
        upsert_api_football_standing(conn, row)
    return {"standings": len(rows), "api_results": payload.get("results"), "api_errors": payload.get("errors") or {}}


def sync_api_football_team_statistics(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    league: int,
    season: int,
    team_ids: list[int] | None = None,
    limit: int | None = None,
) -> dict:
    if team_ids is None:
        team_ids = [
            int(row["team_id"])
            for row in conn.execute(
                """
                SELECT team_id
                FROM api_football_team_seasons
                WHERE league_id = ? AND season = ?
                ORDER BY team_name
                """,
                (league, season),
            ).fetchall()
        ]
    if limit is not None:
        team_ids = team_ids[:limit]
    saved = 0
    errors = []
    for team_id in team_ids:
        payload = client.team_statistics(league=league, season=season, team=team_id)
        row = normalize_team_statistics(payload, league_id=league, season=season, team_id=team_id)
        if row:
            upsert_api_football_team_statistics(conn, row)
            saved += 1
        if payload.get("errors"):
            errors.append({"team_id": team_id, "errors": payload.get("errors")})
    return {"team_statistics": saved, "requests_used": len(team_ids), "errors": errors}


def sync_api_football_players(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    league: int,
    season: int,
    team_ids: list[int] | None = None,
    max_requests: int = 100,
) -> dict:
    if team_ids is None:
        team_ids = [
            int(row["team_id"])
            for row in conn.execute(
                """
                SELECT team_id
                FROM api_football_team_seasons
                WHERE league_id = ? AND season = ?
                ORDER BY team_name
                """,
                (league, season),
            ).fetchall()
        ]
    requests_used = 0
    saved_players = 0
    saved_squads = 0
    saved_stats = 0
    errors = []
    teams_done = 0
    for team_id in team_ids:
        if requests_used >= max_requests:
            break
        page = 1
        team_had_payload = False
        while requests_used < max_requests:
            payload = client.players(league=league, season=season, team=team_id, page=page)
            requests_used += 1
            team_had_payload = True
            if payload.get("errors"):
                errors.append({"team_id": team_id, "page": page, "errors": payload.get("errors")})
            players, squads, stats_rows = normalize_player_statistics(payload, league_id=league, season=season)
            player_id_map = {}
            for player in players:
                internal_id = upsert_player(conn, player)
                player_id_map[player["api_player_id"]] = internal_id
                saved_players += 1
            for squad in squads:
                api_player_id = squad.get("api_player_id")
                internal_id = player_id_map.get(api_player_id)
                if internal_id is None:
                    continue
                upsert_team_squad(conn, {**squad, "player_id": internal_id})
                saved_squads += 1
            for stats in stats_rows:
                api_player_id = stats.get("api_player_id")
                internal_id = player_id_map.get(api_player_id)
                if internal_id is None:
                    continue
                upsert_player_season_stats(conn, {**stats, "player_id": internal_id})
                saved_stats += 1
            paging = payload.get("paging") or {}
            total_pages = int(paging.get("total") or page)
            if page >= total_pages or not payload.get("response"):
                break
            page += 1
        if team_had_payload:
            teams_done += 1
    upsert_sync_inventory(
        conn,
        {
            "provider": "api_football",
            "entity_type": "competition_season",
            "league_id": league,
            "league_name": _first_league_name(conn, league, season),
            "season": season,
            "data_type": "players",
            "status": "complete" if teams_done and requests_used < max_requests else "partial",
            "records_count": saved_players,
            "expected_count": None,
            "request_cost": requests_used,
            "expires_at": None,
            "error_message": json.dumps(errors) if errors else None,
            "raw": {"teams_done": teams_done, "teams_total": len(team_ids), "errors": errors[:10]},
        },
    )
    return {
        "players": saved_players,
        "squad_rows": saved_squads,
        "season_stat_rows": saved_stats,
        "teams_done": teams_done,
        "teams_total": len(team_ids),
        "requests_used": requests_used,
        "max_requests": max_requests,
        "errors": errors,
    }


def sync_api_football_league_coverage(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    league_ids: list[int] | None = None,
    season: int | None = None,
    search: str | None = None,
    country: str | None = None,
) -> dict:
    payloads = []
    requests_used = 0
    errors = []
    if league_ids:
        for league_id in league_ids:
            payload = client.get("leagues", id=league_id, season=season)
            payloads.append(payload)
            requests_used += 1
            if payload.get("errors"):
                errors.append({"league_id": league_id, "errors": payload.get("errors")})
    else:
        payload = client.leagues(search=search, country=country, season=season)
        payloads.append(payload)
        requests_used += 1
        if payload.get("errors"):
            errors.append({"errors": payload.get("errors")})

    saved = 0
    league_count = 0
    feature_counts = {
        "fixtures_events": 0,
        "fixtures_lineups": 0,
        "fixtures_statistics_fixtures": 0,
        "fixtures_statistics_players": 0,
        "standings": 0,
        "players": 0,
        "injuries": 0,
        "odds": 0,
    }
    for payload in payloads:
        response = payload.get("response") or []
        league_count += len(response)
        for item in response:
            for row in normalize_league_coverage(item):
                upsert_api_football_league_coverage(conn, row)
                saved += 1
                for feature in feature_counts:
                    feature_counts[feature] += int(bool(row.get(feature)))
    return {
        "requests_used": requests_used,
        "leagues": league_count,
        "coverage_rows": saved,
        "feature_counts": feature_counts,
        "errors": errors,
    }


def sync_api_football_fixture_lineups(conn: sqlite3.Connection, client: ApiFootballClient, fixture_id: int) -> dict:
    payload = client.fixture_lineups(fixture=fixture_id)
    lineups, players = normalize_lineups(fixture_id, payload)
    for row in lineups:
        upsert_api_football_fixture_lineup(conn, row)
    for row in players:
        upsert_api_football_fixture_player(conn, row)
    return {
        "lineups": len(lineups),
        "lineup_players": len(players),
        "api_results": payload.get("results"),
        "api_errors": payload.get("errors") or {},
    }


def sync_api_football_fixture_player_stats(conn: sqlite3.Connection, client: ApiFootballClient, fixture_id: int) -> dict:
    payload = client.fixture_players(fixture=fixture_id)
    players, stats_rows = normalize_fixture_player_statistics(fixture_id, payload)
    player_id_map = {}
    for player in players:
        player_id_map[player["api_player_id"]] = upsert_player(conn, player)
    saved = 0
    for row in stats_rows:
        internal_id = player_id_map.get(row.get("api_player_id"))
        if internal_id is None:
            continue
        upsert_player_match_stats(conn, {**row, "player_id": internal_id})
        saved += 1
    return {
        "player_match_stats": saved,
        "api_results": payload.get("results"),
        "api_errors": payload.get("errors") or {},
    }


def _local_status(records_count: int, inventory_row: dict | None) -> str:
    if inventory_row and inventory_row.get("status") == "complete" and records_count > 0:
        return "complete"
    if records_count > 0:
        return "partial"
    return "missing"


def _first_league_name(conn: sqlite3.Connection, league_id: int, season: int) -> str | None:
    row = conn.execute(
        """
        SELECT league_name
        FROM api_football_fixtures
        WHERE league_id = ? AND season = ? AND league_name IS NOT NULL
        LIMIT 1
        """,
        (league_id, season),
    ).fetchone()
    return str(row["league_name"]) if row else None


def sync_api_football_fixture_details(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    fixture_id: int,
    fixture_row: dict[str, Any],
) -> dict:
    stats_payload = client.fixture_statistics(fixture=fixture_id)
    stats_rows = normalize_fixture_statistics(fixture_id, fixture_row, stats_payload)
    for row in stats_rows:
        upsert_advanced_stats(conn, row)

    events_payload = client.fixture_events(fixture=fixture_id)
    referee_history = normalize_referee_history(fixture_row, stats_rows, events_payload)
    if referee_history:
        upsert_referee_match_history(conn, referee_history)

    injuries_payload = client.injuries(fixture=fixture_id)
    injury_rows = normalize_injuries(injuries_payload)
    for row in injury_rows:
        upsert_player_availability(conn, row)

    lineup_result = sync_api_football_fixture_lineups(conn, client, fixture_id)
    player_stats_result = sync_api_football_fixture_player_stats(conn, client, fixture_id)

    return {
        "advanced_stats": len(stats_rows),
        "availability_rows": len(injury_rows),
        "referee_history": 1 if referee_history else 0,
        **lineup_result,
        **player_stats_result,
    }


def sync_api_football_fixture_details_bulk(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    league: int | None = None,
    season: int | None = None,
    only_finished: bool = True,
    missing_only: bool = True,
    max_fixtures: int = 100,
) -> dict:
    clauses = []
    params: list[int | str] = []
    if league is not None:
        clauses.append("league_id = ?")
        params.append(league)
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    if only_finished:
        clauses.append("status_short IN ('FT', 'AET', 'PEN')")
    if missing_only:
        clauses.append(
            """
            (
                NOT EXISTS (
                    SELECT 1 FROM match_team_advanced_stats s
                    WHERE s.source_match_id = 'api-football:' || api_football_fixtures.fixture_id
                )
                OR NOT EXISTS (
                    SELECT 1 FROM player_match_stats p
                    WHERE p.fixture_id = api_football_fixtures.fixture_id
                )
                OR NOT EXISTS (
                    SELECT 1 FROM referee_match_history r
                    WHERE r.fixture_id = api_football_fixtures.fixture_id
                )
            )
            """
        )
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT fixture_id, raw_json
        FROM api_football_fixtures
        {where_sql}
        ORDER BY date ASC, fixture_id ASC
        LIMIT ?
        """,
        (*params, max_fixtures),
    ).fetchall()
    totals = {
        "fixtures_processed": 0,
        "advanced_stats": 0,
        "availability_rows": 0,
        "lineups": 0,
        "lineup_players": 0,
        "player_match_stats": 0,
        "referee_history": 0,
        "requests_used_estimate": 0,
        "errors": [],
    }
    for row in rows:
        fixture_id = int(row["fixture_id"])
        raw = json.loads(row["raw_json"] or "{}")
        result = sync_api_football_fixture_details(conn, client, fixture_id, raw)
        totals["fixtures_processed"] += 1
        totals["requests_used_estimate"] += 5
        for key in ["advanced_stats", "availability_rows", "lineups", "lineup_players", "player_match_stats", "referee_history"]:
            totals[key] += int(result.get(key) or 0)
        api_errors = [
            result.get("api_errors"),
        ]
        if any(api_errors):
            totals["errors"].append({"fixture_id": fixture_id, "errors": api_errors})
    totals["referee_metrics_rebuilt"] = rebuild_all_referee_metrics(conn) if totals["referee_history"] else 0
    return totals


def sync_api_football_market_stats_bulk(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    league: int | None = None,
    season: int | None = None,
    only_finished: bool = True,
    max_fixtures: int = 100,
) -> dict:
    clauses = [
        """
        NOT EXISTS (
            SELECT 1 FROM match_team_advanced_stats s
            WHERE s.source_match_id = 'api-football:' || api_football_fixtures.fixture_id
        )
        """
    ]
    params: list[int | str] = []
    if league is not None:
        clauses.append("league_id = ?")
        params.append(league)
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    if only_finished:
        clauses.append("status_short IN ('FT', 'AET', 'PEN')")
    rows = conn.execute(
        f"""
        SELECT fixture_id, raw_json
        FROM api_football_fixtures
        WHERE {' AND '.join(clauses)}
        ORDER BY date ASC, fixture_id ASC
        LIMIT ?
        """,
        (*params, max_fixtures),
    ).fetchall()
    totals = {
        "fixtures_processed": 0,
        "advanced_stats": 0,
        "requests_used": 0,
        "skipped_missing_fixture_raw": 0,
        "errors": [],
    }
    for row in rows:
        fixture_id = int(row["fixture_id"])
        fixture_row = json.loads(row["raw_json"] or "{}")
        if not fixture_row:
            totals["skipped_missing_fixture_raw"] += 1
            continue
        try:
            payload = client.fixture_statistics(fixture=fixture_id)
            totals["requests_used"] += 1
            stats_rows = normalize_fixture_statistics(fixture_id, fixture_row, payload)
            for stats_row in stats_rows:
                upsert_advanced_stats(conn, stats_row)
            totals["fixtures_processed"] += 1
            totals["advanced_stats"] += len(stats_rows)
            if payload.get("errors"):
                totals["errors"].append({"fixture_id": fixture_id, "errors": payload.get("errors")})
        except Exception as exc:
            totals["errors"].append({"fixture_id": fixture_id, "error": str(exc)})
    return totals


def sync_api_football_fixture_details_by_id(
    conn: sqlite3.Connection,
    client: ApiFootballClient,
    fixture_id: int,
) -> dict:
    fixture_payload = client.get("fixtures", id=fixture_id)
    fixtures = fixture_payload.get("response") or []
    if not fixtures:
        return {
            "advanced_stats": 0,
            "availability_rows": 0,
            "error": "fixture_not_found",
            "api_errors": fixture_payload.get("errors") or {},
            "api_results": fixture_payload.get("results"),
        }
    normalized = normalize_fixture(fixtures[0])
    upsert_api_football_fixture(conn, normalized)
    return sync_api_football_fixture_details(conn, client, fixture_id, fixtures[0])


def _normal_key(value: str | None) -> str:
    return (value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coverage_bool(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _first_number(*values: Any) -> int | float | None:
    for value in values:
        number = _to_number(value)
        if number is not None:
            return int(number) if float(number).is_integer() else number
    return None


def _sum_optional(*values: int | float | None) -> int | float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    total = sum(float(value) for value in clean)
    return int(total) if total.is_integer() else total


def _event_card_counts(events: list[dict], home_team: str | None, away_team: str | None) -> dict[str, int]:
    counts = {"home_yellow": 0, "away_yellow": 0, "home_red": 0, "away_red": 0}
    for event in events:
        if str(event.get("type") or "").lower() != "card":
            continue
        team_name = ((event.get("team") or {}).get("name") or "").strip()
        side = "home" if team_name == home_team else "away" if team_name == away_team else None
        if not side:
            continue
        detail = str(event.get("detail") or "").lower()
        if "yellow" in detail:
            counts[f"{side}_yellow"] += 1
        elif "red" in detail:
            counts[f"{side}_red"] += 1
    return counts


def _event_penalties(events: list[dict]) -> int:
    count = 0
    for event in events:
        detail = str(event.get("detail") or "").lower()
        comments = str(event.get("comments") or "").lower()
        event_type = str(event.get("type") or "").lower()
        if "penalty" in detail or ("penalty" in comments and event_type in {"goal", "var"}):
            if "shootout" not in detail:
                count += 1
    return count


def _nested(data: dict, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
