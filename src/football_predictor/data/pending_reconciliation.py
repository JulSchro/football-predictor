from __future__ import annotations

from datetime import datetime, timedelta
import re
import json
import sqlite3
import time
import unicodedata
from typing import Any, Callable

from football_predictor.data.api_clients import ApiFootballClient
from football_predictor.data.api_football_sync import (
    normalize_fixture,
    sync_api_football_fixture_details,
    sync_api_football_fixtures_by_date,
)
from football_predictor.database.db import (
    actual_market_stats_for_fixture,
    reconcile_prediction_backtests_for_fixture,
    upsert_api_football_fixture,
)

FINISHED_STATUSES = {"FT", "AET", "PEN"}
ACTIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}


def run_pending_fixture_reconciliation(
    conn: sqlite3.Connection,
    client: ApiFootballClient | None = None,
    sync_api: bool = False,
    max_fixtures: int = 100,
    retry_hours: int = 6,
    league: int | None = None,
    season: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    started = time.perf_counter()
    client = client if sync_api else None
    today = datetime.now().date().isoformat()

    pending_before = _pending_snapshot_count(conn, today)
    aliases = _alias_map(conn)
    unmatched_dates = _pending_unmatched_dates(conn, today, league=league, season=season)
    date_sync_results = []
    if client and unmatched_dates:
        for target in unmatched_dates:
            _emit(progress, {"event": "sync_unmatched_date", "date": target})
            date_sync_results.append(
                sync_api_football_fixtures_by_date(
                    conn,
                    client,
                    date=target,
                    league=league,
                    season=season,
                    sync_finished_details=False,
                )
            )

    fixtures = _pending_fixture_groups(conn, today, max_fixtures=max_fixtures, league=league, season=season, aliases=aliases)
    totals = {
        "fixtures_pending_found": len(fixtures),
        "fixtures_processed": 0,
        "fixtures_completed": 0,
        "fixtures_skipped_live": 0,
        "fixtures_skipped_retry": 0,
        "snapshots_updated": 0,
        "markets_updated": 0,
        "corners_updated": 0,
        "shots_on_target_updated": 0,
        "cards_updated": 0,
        "errors_found": 0,
        "fixtures_without_stats": 0,
        "requests_estimated": 0,
    }
    reports = []
    _emit(progress, {"event": "found", "fixtures": len(fixtures), "snapshots": pending_before})

    for item in fixtures:
        fixture_id = int(item["fixture_id"])
        fixture_started = time.perf_counter()
        _emit(progress, {"event": "processing", "fixture_id": fixture_id, "home_team": item["home_team"], "away_team": item["away_team"]})
        cache = _sync_status(conn, fixture_id)
        if _retry_blocked(cache):
            totals["fixtures_skipped_retry"] += 1
            reports.append(_fixture_report(item, "retry_wait", cache=cache, seconds=time.perf_counter() - fixture_started))
            continue

        status_short = str(item["status_short"] or "")
        raw_fixture = json.loads(item["raw_json"] or "{}")
        if client and status_short not in FINISHED_STATUSES:
            refreshed = _refresh_fixture_status(conn, client, fixture_id)
            totals["requests_estimated"] += 1
            if refreshed:
                raw_fixture = refreshed
                status_short = str(((refreshed.get("fixture") or {}).get("status") or {}).get("short") or status_short)

        if status_short in ACTIVE_STATUSES or status_short not in FINISHED_STATUSES:
            _upsert_sync_status(
                conn,
                fixture_id,
                status_short=status_short,
                stats_completed=False,
                retry_after=_retry_after(retry_hours),
                snapshots_updated=0,
                raw={"reason": "fixture_not_finished"},
            )
            totals["fixtures_skipped_live"] += 1
            reports.append(_fixture_report(item, "not_finished", status_short=status_short, seconds=time.perf_counter() - fixture_started))
            continue

        totals["fixtures_processed"] += 1
        market_stats = actual_market_stats_for_fixture(conn, fixture_id)
        detail_result: dict[str, Any] = {}
        if client and not market_stats["complete"] and not _cache_completed(cache):
            _emit(progress, {"event": "download_details", "fixture_id": fixture_id})
            try:
                detail_result = sync_api_football_fixture_details(conn, client, fixture_id, raw_fixture)
                totals["requests_estimated"] += 5
            except Exception as exc:
                totals["errors_found"] += 1
                _upsert_sync_status(
                    conn,
                    fixture_id,
                    status_short=status_short,
                    stats_completed=False,
                    retry_after=_retry_after(retry_hours),
                    error_message=str(exc),
                    snapshots_updated=0,
                    raw={"error": str(exc)},
                )
                reports.append(_fixture_report(item, "error", status_short=status_short, error=str(exc), seconds=time.perf_counter() - fixture_started))
                continue
            market_stats = actual_market_stats_for_fixture(conn, fixture_id)

        reconciliation = reconcile_prediction_backtests_for_fixture(conn, fixture_id)
        updated_markets = reconciliation.get("markets_updated") or []
        snapshots_updated = int(reconciliation.get("snapshots_updated") or 0)
        stats_completed = bool(market_stats["complete"])
        if stats_completed:
            totals["fixtures_completed"] += 1
        else:
            totals["fixtures_without_stats"] += 1

        totals["snapshots_updated"] += snapshots_updated
        totals["markets_updated"] += snapshots_updated * len(updated_markets)
        totals["corners_updated"] += snapshots_updated if "corners" in updated_markets else 0
        totals["shots_on_target_updated"] += snapshots_updated if "shots_on_target" in updated_markets else 0
        totals["cards_updated"] += snapshots_updated if "cards" in updated_markets else 0
        _upsert_sync_status(
            conn,
            fixture_id,
            status_short=status_short,
            stats_completed=stats_completed,
            events_completed=bool(detail_result.get("referee_history") or detail_result == {}),
            lineups_completed=bool(detail_result.get("lineups") or detail_result == {}),
            player_stats_completed=bool(detail_result.get("player_match_stats") or detail_result == {}),
            referee_completed=bool(detail_result.get("referee_history") or detail_result == {}),
            retry_after=None if stats_completed or not client else _retry_after(retry_hours),
            snapshots_updated=snapshots_updated,
            raw={"detail_result": detail_result, "reconciliation": reconciliation},
        )
        reports.append({**reconciliation, "seconds": round(time.perf_counter() - fixture_started, 3)})
        _emit(progress, {"event": "fixture_done", "fixture_id": fixture_id, "stats_complete": stats_completed, "snapshots_updated": snapshots_updated})

    pending_after = _pending_snapshot_count(conn, today)
    remaining = _remaining_finished_pending(conn, today, aliases=aliases)
    elapsed = time.perf_counter() - started
    return {
        **totals,
        "pending_before": pending_before,
        "pending_after": pending_after,
        "unmatched_dates_synced": date_sync_results,
        "remaining_finished_pending": remaining,
        "validation_ok": len(remaining) == 0,
        "time_total_seconds": round(elapsed, 3),
        "time_avg_fixture_seconds": round(elapsed / max(1, totals["fixtures_processed"]), 3),
        "parallelism": 1,
        "reports": reports,
    }


def _pending_snapshot_count(conn: sqlite3.Connection, today: str) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_backtests
            WHERE match_date <= ?
              AND (
                actual_home_goals IS NULL OR actual_away_goals IS NULL
                OR actual_corners IS NULL OR actual_shots_on_target IS NULL OR actual_cards IS NULL
              )
            """,
            (today,),
        ).fetchone()[0]
    )


def _pending_unmatched_dates(conn: sqlite3.Connection, today: str, league: int | None = None, season: int | None = None) -> list[str]:
    aliases = _alias_map(conn)
    snapshots = _pending_snapshot_rows(conn, today, league=league)
    dates = []
    for snapshot in snapshots:
        fixtures = _fixture_rows_for_date(conn, str(snapshot["match_date"]), league=league, season=season)
        if not any(_same_match(snapshot, fixture, aliases) for fixture in fixtures):
            dates.append(str(snapshot["match_date"]))
    return sorted(set(dates))[:20]


def _pending_fixture_groups(
    conn: sqlite3.Connection,
    today: str,
    max_fixtures: int,
    league: int | None = None,
    season: int | None = None,
    aliases: dict[str, str] | None = None,
) -> list[sqlite3.Row]:
    aliases = aliases or _alias_map(conn)
    snapshots = _pending_snapshot_rows(conn, today, league=league)
    grouped: dict[int, dict[str, Any]] = {}
    for snapshot in snapshots:
        fixtures = _fixture_rows_for_date(conn, str(snapshot["match_date"]), league=league, season=season)
        for fixture in fixtures:
            if not _same_match(snapshot, fixture, aliases):
                continue
            fixture_id = int(fixture["fixture_id"])
            if fixture_id not in grouped:
                grouped[fixture_id] = {**dict(fixture), "snapshots": 0}
            grouped[fixture_id]["snapshots"] += 1
            break
        if len(grouped) >= max_fixtures:
            break
    return [_DictRow(row) for row in grouped.values()]


def _remaining_finished_pending(conn: sqlite3.Connection, today: str, aliases: dict[str, str] | None = None) -> list[dict]:
    aliases = aliases or _alias_map(conn)
    snapshots = _pending_snapshot_rows(conn, today)
    output = []
    for snapshot in snapshots:
        fixtures = _fixture_rows_for_date(conn, str(snapshot["match_date"]))
        fixture = next((fixture for fixture in fixtures if fixture["status_short"] in FINISHED_STATUSES and _same_match(snapshot, fixture, aliases)), None)
        if not fixture:
            continue
        missing = [
            name
            for name, field in [
                ("home_goals", "actual_home_goals"),
                ("away_goals", "actual_away_goals"),
                ("corners", "actual_corners"),
                ("shots_on_target", "actual_shots_on_target"),
                ("cards", "actual_cards"),
            ]
            if snapshot[field] is None
        ]
        output.append({**dict(snapshot), "fixture_id": fixture["fixture_id"], "status_short": fixture["status_short"], "missing": missing})
        if len(output) >= 25:
            break
    return output


class _DictRow(dict):
    def __getattr__(self, name: str) -> Any:
        return self[name]


def _pending_snapshot_rows(conn: sqlite3.Connection, today: str, league: int | None = None) -> list[sqlite3.Row]:
    clauses = [
        "p.match_date <= ?",
        """
        (
            p.actual_home_goals IS NULL OR p.actual_away_goals IS NULL
            OR p.actual_corners IS NULL OR p.actual_shots_on_target IS NULL OR p.actual_cards IS NULL
        )
        """,
    ]
    params: list[Any] = [today]
    if league is not None:
        clauses.append("p.competition IN (SELECT name FROM competitions WHERE api_football_league_id = ?)")
        params.append(league)
    return conn.execute(
        f"""
        SELECT p.id, p.match_date, p.competition, p.home_team, p.away_team,
               p.actual_home_goals, p.actual_away_goals, p.actual_corners,
               p.actual_shots_on_target, p.actual_cards
        FROM prediction_backtests p
        WHERE {' AND '.join(clauses)}
        ORDER BY p.match_date ASC, p.id ASC
        """,
        tuple(params),
    ).fetchall()


def _fixture_rows_for_date(conn: sqlite3.Connection, target: str, league: int | None = None, season: int | None = None) -> list[sqlite3.Row]:
    clauses = ["substr(date, 1, 10) = ?"]
    params: list[Any] = [target]
    if league is not None:
        clauses.append("league_id = ?")
        params.append(league)
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    return conn.execute(
        f"""
        SELECT fixture_id, date, league_id, league_name, season, home_team, away_team, status_short, raw_json
        FROM api_football_fixtures
        WHERE {' AND '.join(clauses)}
        ORDER BY date ASC, fixture_id ASC
        """,
        tuple(params),
    ).fetchall()


def _same_match(snapshot: sqlite3.Row, fixture: sqlite3.Row, aliases: dict[str, str]) -> bool:
    return (
        _canonical(snapshot["home_team"], aliases) == _canonical(fixture["home_team"], aliases)
        and _canonical(snapshot["away_team"], aliases) == _canonical(fixture["away_team"], aliases)
    )


def _alias_map(conn: sqlite3.Connection) -> dict[str, str]:
    aliases: dict[str, str] = {}
    rows = conn.execute("SELECT canonical_name, alias FROM team_name_aliases").fetchall()
    for row in rows:
        canonical = _normalize_name(row["canonical_name"])
        aliases[_normalize_name(row["canonical_name"])] = canonical
        aliases[_normalize_name(row["alias"])] = canonical
    return aliases


def _canonical(name: str | None, aliases: dict[str, str]) -> str:
    normalized = _normalize_name(name)
    return aliases.get(normalized, normalized)


def _normalize_name(name: str | None) -> str:
    value = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _refresh_fixture_status(conn: sqlite3.Connection, client: ApiFootballClient, fixture_id: int) -> dict | None:
    payload = client.get("fixtures", id=fixture_id)
    fixtures = payload.get("response") or []
    if not fixtures:
        return None
    normalized = normalize_fixture(fixtures[0])
    upsert_api_football_fixture(conn, normalized)
    return fixtures[0]


def _sync_status(conn: sqlite3.Connection, fixture_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM fixture_detail_sync_status WHERE fixture_id = ?", (fixture_id,)).fetchone()
    return dict(row) if row else None


def _upsert_sync_status(
    conn: sqlite3.Connection,
    fixture_id: int,
    status_short: str | None,
    stats_completed: bool,
    events_completed: bool = False,
    lineups_completed: bool = False,
    player_stats_completed: bool = False,
    referee_completed: bool = False,
    snapshots_updated: int = 0,
    retry_after: str | None = None,
    error_message: str | None = None,
    raw: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO fixture_detail_sync_status
        (fixture_id, status_short, last_checked, stats_completed, events_completed, lineups_completed,
         player_stats_completed, referee_completed, snapshots_updated, retry_after, error_message, raw_json, updated_at)
        VALUES
        (:fixture_id, :status_short, CURRENT_TIMESTAMP, :stats_completed, :events_completed, :lineups_completed,
         :player_stats_completed, :referee_completed, :snapshots_updated, :retry_after, :error_message, :raw_json,
         CURRENT_TIMESTAMP)
        ON CONFLICT(fixture_id) DO UPDATE SET
            status_short = excluded.status_short,
            last_checked = excluded.last_checked,
            stats_completed = excluded.stats_completed,
            events_completed = excluded.events_completed,
            lineups_completed = excluded.lineups_completed,
            player_stats_completed = excluded.player_stats_completed,
            referee_completed = excluded.referee_completed,
            snapshots_updated = excluded.snapshots_updated,
            retry_after = excluded.retry_after,
            error_message = excluded.error_message,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            "fixture_id": fixture_id,
            "status_short": status_short,
            "stats_completed": int(stats_completed),
            "events_completed": int(events_completed),
            "lineups_completed": int(lineups_completed),
            "player_stats_completed": int(player_stats_completed),
            "referee_completed": int(referee_completed),
            "snapshots_updated": snapshots_updated,
            "retry_after": retry_after,
            "error_message": error_message,
            "raw_json": json.dumps(raw or {}),
        },
    )


def _cache_completed(cache: dict | None) -> bool:
    return bool(cache and cache.get("stats_completed"))


def _retry_blocked(cache: dict | None) -> bool:
    retry_after = cache.get("retry_after") if cache else None
    if not retry_after or cache.get("stats_completed"):
        return False
    raw = {}
    try:
        raw = json.loads(cache.get("raw_json") or "{}")
    except (TypeError, ValueError):
        raw = {}
    attempted_api = bool(raw.get("error") or raw.get("reason") == "fixture_not_finished" or raw.get("detail_result"))
    if not attempted_api:
        return False
    try:
        return datetime.fromisoformat(str(retry_after)) > datetime.now()
    except ValueError:
        return False


def _retry_after(hours: int) -> str:
    return (datetime.now() + timedelta(hours=max(1, hours))).isoformat(timespec="seconds")


def _fixture_report(row: sqlite3.Row, reason: str, **extra: Any) -> dict:
    return {
        "fixture_id": int(row["fixture_id"]),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "status": extra.pop("status_short", row["status_short"]),
        "snapshots": int(row["snapshots"] or 0),
        "reason": reason,
        **extra,
    }


def _emit(progress: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if progress:
        progress(event)
