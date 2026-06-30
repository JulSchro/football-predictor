from datetime import date, timedelta
import json
from pathlib import Path
import time

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from football_predictor.config import settings
from football_predictor.data.api_clients import ApiFootballClient
from football_predictor.data.api_football_sync import (
    sync_api_football_fixture_details,
    sync_api_football_fixture_details_by_id,
    sync_api_football_fixtures_by_date,
    sync_api_football_standings,
    sync_api_football_team_statistics,
)
from football_predictor.data.competitions import competition_catalog
from football_predictor.data.coverage import data_coverage
from football_predictor.database.db import (
    advanced_summary_by_team,
    api_lineup_summary_by_team,
    api_standings_summary_by_team,
    api_team_strength_summary_by_team,
    availability_summary_by_team,
    player_quality_summary_by_team,
    connect,
    delete_pending_prediction_backtest,
    get_match_context,
    get_team_external_metrics,
    get_venue,
    init_db,
    insert_experiment,
    insert_daily_job,
    insert_prediction,
    list_experiments,
    list_pending_prediction_backtests,
    list_prediction_backtest_summaries,
    list_prediction_backtests,
    list_teams,
    list_venues,
    update_prediction_backtest_result,
    upsert_prediction_backtest,
)
from football_predictor.evaluation.backtest import run_backtest
from football_predictor.evaluation.prediction_backtest import add_evaluation, compute_backtest_metrics, parse_scores
from football_predictor.prediction.factor_model import build_team_profile
from football_predictor.prediction.calibration import apply_market_calibration
from football_predictor.prediction.markets import estimate_secondary_markets
from football_predictor.prediction.player_markets import estimate_player_markets
from football_predictor.prediction.predictor import MatchPredictor
from football_predictor.prediction.simulator import simulate_advanced_match, simulate_fixture_list, simulate_match


def _market_error(predicted: float | None, actual: float | None) -> dict | None:
    if predicted is None or actual is None:
        return None
    error = float(predicted) - float(actual)
    return {
        "predicted": predicted,
        "actual": actual,
        "error": round(error, 4),
        "absolute_error": round(abs(error), 4),
        "direction": "sobreestima" if error > 0 else "subestima" if error < 0 else "exacto",
    }


def _snapshot_payload(
    prediction: dict,
    simulation: dict,
    markets: dict,
    match_context: dict | None,
    data_quality: dict | None,
    fixture: dict | None = None,
    betting_picks: list[dict] | None = None,
    player_markets: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "model_prediction": prediction,
        "simulation": {
            "mode": simulation.get("mode"),
            "simulations": simulation.get("simulations"),
            "outcomes": simulation.get("simulation"),
            "top_scores": simulation.get("top_scores", []),
            "factor_edge": simulation.get("factor_edge"),
            "venue_edge": simulation.get("venue_edge"),
            "scenario_mix": simulation.get("scenario_mix"),
            "factor_cards": simulation.get("factor_cards", []),
            "sensitivity": simulation.get("sensitivity", []),
            "profiles": simulation.get("profiles"),
        },
        "secondary_markets": markets,
        "match_context": match_context,
        "data_quality": data_quality,
        "fixture": fixture,
        "betting_picks": betting_picks or [],
        "player_markets": player_markets or {},
    }


def _date_window(start: str, range_days: int) -> list[str]:
    base = date.fromisoformat(start)
    days = max(1, min(int(range_days or 1), 7))
    return [(base + timedelta(days=offset)).isoformat() for offset in range(days)]


def _confidence_label(probability: float, edge: float = 0.0) -> str:
    if probability >= 0.68 or (probability >= 0.58 and edge >= 0.16):
        return "alta"
    if probability >= 0.52 or edge >= 0.1:
        return "media"
    return "baja"


def _betting_picks(
    home_team: str,
    away_team: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    over_2_5_prob: float,
    btts_prob: float,
    markets: dict,
) -> list[dict]:
    outcomes = [
        ("home", home_team, float(home_prob)),
        ("draw", "Empate", float(draw_prob)),
        ("away", away_team, float(away_prob)),
    ]
    ranked = sorted(outcomes, key=lambda item: item[2], reverse=True)
    top_key, top_label, top_prob = ranked[0]
    edge = top_prob - ranked[1][2]
    picks = [
        {
            "market": "1X2",
            "pick": top_label if top_key != "draw" else "Empate",
            "probability": round(top_prob, 4),
            "confidence": _confidence_label(top_prob, edge),
            "risk": "medio" if top_key == "draw" else "normal",
        }
    ]
    if top_key == "home":
        dc_prob = home_prob + draw_prob
        picks.append({"market": "Doble oportunidad", "pick": f"{home_team} o empate", "probability": round(dc_prob, 4), "confidence": _confidence_label(dc_prob, 0.12), "risk": "bajo"})
    elif top_key == "away":
        dc_prob = away_prob + draw_prob
        picks.append({"market": "Doble oportunidad", "pick": f"{away_team} o empate", "probability": round(dc_prob, 4), "confidence": _confidence_label(dc_prob, 0.12), "risk": "bajo"})
    if over_2_5_prob >= 0.54:
        picks.append({"market": "Goles", "pick": "Over 2.5", "probability": round(float(over_2_5_prob), 4), "confidence": _confidence_label(float(over_2_5_prob), 0.08), "risk": "medio"})
    elif over_2_5_prob <= 0.44:
        picks.append({"market": "Goles", "pick": "Under 2.5", "probability": round(1 - float(over_2_5_prob), 4), "confidence": _confidence_label(1 - float(over_2_5_prob), 0.08), "risk": "medio"})
    if btts_prob >= 0.55:
        picks.append({"market": "Ambos anotan", "pick": "Si", "probability": round(float(btts_prob), 4), "confidence": _confidence_label(float(btts_prob), 0.08), "risk": "medio"})
    elif btts_prob <= 0.43:
        picks.append({"market": "Ambos anotan", "pick": "No", "probability": round(1 - float(btts_prob), 4), "confidence": _confidence_label(1 - float(btts_prob), 0.08), "risk": "medio"})
    corners_prob = markets.get("over_8_5_corners_prob")
    if corners_prob is not None and float(corners_prob) >= 0.56:
        picks.append({"market": "Corners", "pick": "Over 8.5", "probability": round(float(corners_prob), 4), "confidence": _confidence_label(float(corners_prob), 0.08), "risk": "medio"})
    cards_prob = markets.get("over_3_5_cards_prob")
    if cards_prob is not None and float(cards_prob) >= 0.56:
        picks.append({"market": "Tarjetas", "pick": "Over 3.5", "probability": round(float(cards_prob), 4), "confidence": _confidence_label(float(cards_prob), 0.08), "risk": "alto"})
    return sorted(picks, key=lambda item: item["probability"], reverse=True)[:5]


def create_app(db_path: Path | str | None = None) -> FastAPI:
    app = FastAPI(title="FootballPredictor")
    resolved_db_path = Path(db_path) if db_path is not None else settings.db_path
    cache: dict[str, tuple[float, object]] = {}

    def cached(key: str, ttl_seconds: float, factory):
        now = time.monotonic()
        item = cache.get(key)
        if item and now - item[0] < ttl_seconds:
            return item[1]
        value = factory()
        cache[key] = (now, value)
        return value

    def clear_cache() -> None:
        cache.clear()

    def load_matches() -> pd.DataFrame:
        def factory() -> pd.DataFrame:
            init_db(resolved_db_path)
            with connect(resolved_db_path) as conn:
                return pd.read_sql_query("SELECT * FROM matches ORDER BY date", conn)

        return cached("matches_df", 60, factory)

    def load_team_metrics() -> dict[str, dict]:
        def factory() -> dict[str, dict]:
            with connect(resolved_db_path) as conn:
                teams = list_teams(conn)
                external_rows = conn.execute(
                    """
                    SELECT *
                    FROM team_external_metrics
                    ORDER BY updated_at DESC
                    """
                ).fetchall()
                external: dict[str, dict] = {}
                for row in external_rows:
                    data = dict(row)
                    team = str(data["team"])
                    target = external.setdefault(team, {})
                    raw = json.loads(data.get("raw_json") or "{}")
                    if "raw" not in target and raw:
                        target["raw"] = raw
                    for key in ["fifa_rank", "fifa_points", "squad_value_eur", "squad_size", "avg_age"]:
                        if target.get(key) is None and data.get(key) is not None:
                            target[key] = data[key]
                advanced = advanced_summary_by_team(conn)
                availability = availability_summary_by_team(conn)
                api_strength = api_team_strength_summary_by_team(conn)
                api_standings = api_standings_summary_by_team(conn)
                api_lineups = api_lineup_summary_by_team(conn)
                player_quality = player_quality_summary_by_team(conn)
                return {
                    team: {
                        **external.get(team, {}),
                        **advanced.get(team, {}),
                        **availability.get(team, {}),
                        **api_strength.get(team, {}),
                        **api_standings.get(team, {}),
                        **api_lineups.get(team, {}),
                        **player_quality.get(team, {}),
                    }
                    for team in teams
                }

        return cached("team_metrics", 60, factory)

    def load_advanced_stats() -> pd.DataFrame:
        def factory() -> pd.DataFrame:
            init_db(resolved_db_path)
            with connect(resolved_db_path) as conn:
                return pd.read_sql_query("SELECT * FROM match_team_advanced_stats ORDER BY date", conn)

        return cached("advanced_stats_df", 60, factory)

    def load_predictor() -> MatchPredictor:
        return cached("match_predictor", 60, lambda: MatchPredictor(load_matches(), team_metrics=load_team_metrics()))

    def load_factor_profile(team: str) -> dict:
        def factory() -> dict:
            metrics = load_team_metrics()
            return build_team_profile(load_matches(), team, metrics.get(team, {}))

        return cached(f"factor_profile:{team.lower()}", 60, factory)

    def load_player_markets(home_team: str, away_team: str, markets: dict) -> dict:
        key_parts = [
            home_team.lower(),
            away_team.lower(),
            str(markets.get("home_shots_on_target_expected")),
            str(markets.get("away_shots_on_target_expected")),
            str(markets.get("home_xg_recent")),
            str(markets.get("away_xg_recent")),
        ]

        def factory() -> dict:
            with connect(resolved_db_path) as conn:
                return estimate_player_markets(conn, home_team, away_team, markets)

        return cached(f"player_markets:{'|'.join(key_parts)}", 60, factory)

    def load_prediction_backtest_metrics(limit: int = 500) -> dict:
        def factory() -> dict:
            with connect(resolved_db_path) as conn:
                rows = list_prediction_backtests(conn, limit=limit)
            return compute_backtest_metrics(rows)

        return cached(f"backtest_metrics:{limit}", 30, factory)

    def load_match_context(home_team: str, away_team: str, match_date: str | None = None) -> dict | None:
        with connect(resolved_db_path) as conn:
            return get_match_context(conn, home_team, away_team, match_date=match_date)

    def configured_league_ids(conn) -> set[int]:
        rows = conn.execute(
            """
            SELECT enabled, api_football_league_id
            FROM competitions
            WHERE api_football_league_id IS NOT NULL
            """
        ).fetchall()
        if rows:
            return {int(row["api_football_league_id"]) for row in rows if row["enabled"]}
        return {int(row["api_football_league_id"]) for row in competition_catalog()["competitions"] if row.get("api_football_league_id")}

    def fixture_rows_for_date(
        conn,
        target: str,
        configured_only: bool = True,
        league: int | None = None,
        season: int | None = None,
    ) -> list[dict]:
        clauses = [
            "substr(date, 1, 10) = ?",
            "home_team IS NOT NULL",
            "away_team IS NOT NULL",
            "status_short NOT IN ('CANC', 'PST', 'ABD', 'AWD', 'WO')",
        ]
        params: list[str | int] = [target]
        if league is not None:
            clauses.append("league_id = ?")
            params.append(league)
        if season is not None:
            clauses.append("season = ?")
            params.append(season)
        rows = conn.execute(
            f"""
            SELECT fixture_id, date, league_id, league_name, season, home_team, away_team,
                   status_short, venue_name, venue_city
            FROM api_football_fixtures
            WHERE {' AND '.join(clauses)}
            ORDER BY date ASC, league_name, home_team
            """,
            tuple(params),
        ).fetchall()
        fixtures = [dict(row) for row in rows]
        if not configured_only:
            return fixtures
        enabled_ids = configured_league_ids(conn)
        return [row for row in fixtures if row.get("league_id") in enabled_ids]

    def fixture_detail_rows_for_date(
        conn,
        target: str,
        league: int | None = None,
        season: int | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        clauses = ["substr(date, 1, 10) = ?", "fixture_id IS NOT NULL"]
        params: list[str | int] = [target]
        if league is not None:
            clauses.append("league_id = ?")
            params.append(league)
        if season is not None:
            clauses.append("season = ?")
            params.append(season)
        if limit is not None and limit <= 0:
            return []
        limit_sql = " LIMIT ?" if limit else ""
        if limit:
            params.append(max(1, limit))
        rows = conn.execute(
            f"""
            SELECT fixture_id, date, league_id, league_name, season, home_team, away_team,
                   status_short, raw_json
            FROM api_football_fixtures
            WHERE {' AND '.join(clauses)}
            ORDER BY
                CASE WHEN status_short IN ('NS', 'TBD') THEN 0 ELSE 1 END,
                date ASC
            {limit_sql}
            """,
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]

    def create_daily_prediction_snapshots(
        target: str,
        simulations: int = 5000,
        configured_only: bool = True,
        league: int | None = None,
        season: int | None = None,
        range_days: int = 1,
    ) -> dict:
        init_db(resolved_db_path)
        matches = load_matches()
        team_metrics = load_team_metrics()
        advanced_stats = load_advanced_stats()
        with connect(resolved_db_path) as quality_conn:
            quality_snapshot = data_coverage(quality_conn)
        created = 0
        skipped = 0
        snapshots = []
        dates = _date_window(target, range_days)
        with connect(resolved_db_path) as conn:
            fixtures = []
            for item_date in dates:
                fixtures.extend(
                    fixture_rows_for_date(
                        conn,
                        item_date,
                        configured_only=configured_only,
                        league=league,
                        season=season,
                    )
                )
            existing_metrics = compute_backtest_metrics(list_prediction_backtests(conn))
            predictor = MatchPredictor(matches, team_metrics=team_metrics)
            profile_cache: dict[str, dict] = {}
            def get_profile(team: str) -> dict:
                if team not in profile_cache:
                    profile_cache[team] = build_team_profile(matches, team, team_metrics.get(team, {}))
                return profile_cache[team]

            for fixture in fixtures:
                home = str(fixture["home_team"])
                away = str(fixture["away_team"])
                match_date = str(fixture.get("date") or target)[:10]
                try:
                    prediction = predictor.predict(home, away).model_dump()
                    match_context = get_match_context(conn, home, away, match_date=match_date)
                    simulation = simulate_advanced_match(
                        matches,
                        team_metrics,
                        home,
                        away,
                        simulations=simulations,
                        match_context=match_context,
                        predictor=predictor,
                        home_profile=get_profile(home),
                        away_profile=get_profile(away),
                    )
                except Exception as exc:
                    skipped += 1
                    snapshots.append({"home_team": home, "away_team": away, "error": str(exc)})
                    continue
                markets = apply_market_calibration(estimate_secondary_markets(advanced_stats, home, away), existing_metrics)
                player_markets = estimate_player_markets(conn, home, away, markets)
                simulated_probs = simulation.get("simulation", {})
                home_prob = simulated_probs.get("home_win", prediction["home_win_prob"])
                draw_prob = simulated_probs.get("draw", prediction["draw_prob"])
                away_prob = simulated_probs.get("away_win", prediction["away_win_prob"])
                probs = {"home": home_prob, "draw": draw_prob, "away": away_prob}
                betting_picks = _betting_picks(
                    home,
                    away,
                    home_prob,
                    draw_prob,
                    away_prob,
                    prediction["over_2_5_prob"],
                    prediction["both_teams_score_prob"],
                    markets,
                )
                row = {
                    "match_date": match_date,
                    "competition": fixture["league_name"] or "Unknown",
                    "home_team": home,
                    "away_team": away,
                    "model_version": "auto_daily_snapshot",
                    "predicted_home_prob": home_prob,
                    "predicted_draw_prob": draw_prob,
                    "predicted_away_prob": away_prob,
                    "predicted_pick": max(probs, key=probs.get),
                    "predicted_scores_json": json.dumps(simulation.get("top_scores", [])[:5]),
                    "predicted_corners": markets.get("total_corners_expected"),
                    "predicted_shots_on_target": markets.get("shots_on_target_expected"),
                    "predicted_cards": markets.get("total_cards_expected"),
                    "predicted_over_2_5_prob": prediction["over_2_5_prob"],
                    "predicted_btts_prob": prediction["both_teams_score_prob"],
                    "actual_home_goals": None,
                    "actual_away_goals": None,
                    "actual_corners": None,
                    "actual_shots_on_target": None,
                    "actual_cards": None,
                    "source": "local_daily_job",
                    "notes": f"Snapshot automatico local. Contexto: {(match_context or {}).get('stage_label', 'sin contexto')}",
                    "snapshot": _snapshot_payload(
                        prediction,
                        simulation,
                        markets,
                        match_context,
                        quality_snapshot,
                        fixture=fixture,
                        betting_picks=betting_picks,
                        player_markets=player_markets,
                    ),
                }
                upsert_prediction_backtest(conn, row)
                snapshots.append({**row, "context": (match_context or {}).get("stage_label"), "betting_picks": betting_picks, "player_markets": player_markets})
                created += 1
            insert_daily_job(
                conn,
                {
                    "job_date": target,
                    "competition": "all",
                    "status": "predictions_created",
                    "fixtures_found": len(fixtures),
                    "predictions_created": created,
                    "results_updated": 0,
                    "errors": [row for row in snapshots if row.get("error")],
                },
            )
            conn.commit()
            clear_cache()
        return {"date": target, "dates": dates, "fixtures": len(fixtures), "predictions_created": created, "skipped": skipped, "snapshots": snapshots}

    def actual_market_stats_for_fixture(conn, fixture_id: int) -> dict:
        row = conn.execute(
            """
            SELECT
                SUM(corners) AS actual_corners,
                SUM(shots_on_target) AS actual_shots_on_target,
                SUM(
                    CASE
                        WHEN cards_estimate IS NOT NULL THEN cards_estimate
                        ELSE COALESCE(yellow_cards, 0) + COALESCE(red_cards, 0) * 2
                    END
                ) AS actual_cards
            FROM match_team_advanced_stats
            WHERE source_match_id = ?
            """,
            (f"api-football:{fixture_id}",),
        ).fetchone()
        if not row:
            return {"actual_corners": None, "actual_shots_on_target": None, "actual_cards": None}
        return {
            "actual_corners": row["actual_corners"],
            "actual_shots_on_target": row["actual_shots_on_target"],
            "actual_cards": row["actual_cards"],
        }

    def update_results_from_local_fixtures(target: str) -> dict:
        init_db(resolved_db_path)
        updated = 0
        with connect(resolved_db_path) as conn:
            fixtures = conn.execute(
                """
                SELECT fixture_id, date, league_name, home_team, away_team, status_short, raw_json
                FROM api_football_fixtures
                WHERE substr(date, 1, 10) = ?
                  AND status_short IN ('FT', 'AET', 'PEN')
                """,
                (target,),
            ).fetchall()
            for fixture in fixtures:
                raw = json.loads(fixture["raw_json"] or "{}")
                goals = raw.get("goals") or {}
                if goals.get("home") is None or goals.get("away") is None:
                    continue
                market_stats = actual_market_stats_for_fixture(conn, int(fixture["fixture_id"]))
                rows = conn.execute(
                    """
                    SELECT id
                    FROM prediction_backtests
                    WHERE match_date = ?
                      AND home_team = ?
                      AND away_team = ?
                      AND (actual_home_goals IS NULL OR actual_away_goals IS NULL)
                    """,
                    (target, fixture["home_team"], fixture["away_team"]),
                ).fetchall()
                for row in rows:
                    update_prediction_backtest_result(
                        conn,
                        int(row["id"]),
                        {
                            "actual_home_goals": int(goals["home"]),
                            "actual_away_goals": int(goals["away"]),
                            **market_stats,
                            "notes": "Resultado actualizado desde fixture local/API",
                        },
                    )
                    updated += 1
            insert_daily_job(
                conn,
                {
                    "job_date": target,
                    "competition": "all",
                    "status": "results_updated",
                    "fixtures_found": len(fixtures),
                    "predictions_created": 0,
                    "results_updated": updated,
                    "errors": [],
                },
            )
            conn.commit()
            clear_cache()
            metrics = compute_backtest_metrics(list_prediction_backtests(conn, limit=500))
        return {"date": target, "finished_fixtures": len(fixtures), "results_updated": updated, "metrics": {key: value for key, value in metrics.items() if key != "rows"}}

    def reconcile_pending_results(
        max_dates: int = 7,
        sync_api: bool = False,
        league: int | None = None,
        season: int | None = None,
    ) -> dict:
        init_db(resolved_db_path)
        today = date.today().isoformat()
        with connect(resolved_db_path) as conn:
            pending_before = conn.execute(
                """
                SELECT COUNT(*)
                FROM prediction_backtests
                WHERE (actual_home_goals IS NULL OR actual_away_goals IS NULL)
                  AND match_date <= ?
                """,
                (today,),
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT DISTINCT match_date
                FROM prediction_backtests
                WHERE (actual_home_goals IS NULL OR actual_away_goals IS NULL)
                  AND match_date <= ?
                ORDER BY match_date ASC
                LIMIT ?
                """,
                (today, max(1, max_dates)),
            ).fetchall()
        sync_results = []
        update_results = []
        if sync_api:
            client = ApiFootballClient()
        for row in rows:
            target = str(row["match_date"])
            if sync_api:
                with connect(resolved_db_path) as conn:
                    synced = sync_api_football_fixtures_by_date(conn, client, date=target, league=league, season=season)
                    conn.commit()
                    clear_cache()
                sync_results.append({"date": target, **synced})
            update_results.append(update_results_from_local_fixtures(target))
        with connect(resolved_db_path) as conn:
            pending_after = conn.execute(
                """
                SELECT COUNT(*)
                FROM prediction_backtests
                WHERE (actual_home_goals IS NULL OR actual_away_goals IS NULL)
                  AND match_date <= ?
                """,
                (today,),
            ).fetchone()[0]
        return {
            "dates_checked": [str(row["match_date"]) for row in rows],
            "pending_before": int(pending_before),
            "pending_after": int(pending_after),
            "results_updated": sum(int(result.get("results_updated", 0)) for result in update_results),
            "sync_api": sync_api,
            "sync_results": sync_results,
            "updates": update_results,
        }

    static_dir = Path(__file__).with_name("static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/styles.css")
    def styles() -> FileResponse:
        return FileResponse(static_dir / "styles.css")

    @app.get("/api/summary")
    def summary() -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            counts = {
                "teams": conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0],
                "matches": conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
                "predictions": conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
                "models": conn.execute("SELECT COUNT(*) FROM model_artifacts").fetchone()[0],
                "latest_match_date": conn.execute("SELECT MAX(date) FROM matches").fetchone()[0],
            }
        return counts

    @app.get("/api/data-quality")
    def quality() -> dict:
        def factory() -> dict:
            init_db(resolved_db_path)
            with connect(resolved_db_path) as conn:
                return data_coverage(conn)

        return cached("data_quality", 30, factory)

    @app.get("/api/competitions")
    def competitions() -> dict:
        return competition_catalog()

    @app.get("/api/experiments")
    def experiments(limit: int = Query(default=10, ge=1, le=100)) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            return {"experiments": list_experiments(conn, limit=limit)}

    @app.get("/api/teams")
    def teams() -> dict[str, list[str]]:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            return {"teams": list_teams(conn)}

    @app.get("/api/venues")
    def venues() -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            return {"venues": list_venues(conn)}

    @app.get("/api/matches")
    def matches(limit: int = Query(default=20, ge=1, le=200)) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            rows = conn.execute(
                """
                SELECT date, home_team, away_team, home_goals, away_goals, competition, season
                FROM matches
                ORDER BY date DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {"matches": [dict(row) for row in rows]}

    @app.get("/api/data-explorer/competitions")
    def data_explorer_competitions() -> dict:
        def factory() -> dict:
            init_db(resolved_db_path)
            with connect(resolved_db_path) as conn:
                rows = conn.execute(
                    """
                    WITH combined AS (
                        SELECT competition AS competition, season AS season, date AS match_date,
                               CASE WHEN home_goals IS NOT NULL AND away_goals IS NOT NULL THEN 1 ELSE 0 END AS finished,
                               0 AS upcoming,
                               'historical' AS source
                        FROM matches
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM api_football_fixtures f
                            WHERE matches.date = substr(f.date, 1, 10)
                              AND lower(matches.home_team) = lower(f.home_team)
                              AND lower(matches.away_team) = lower(f.away_team)
                              AND lower(matches.competition) = lower(f.league_name)
                              AND CAST(matches.season AS TEXT) = CAST(f.season AS TEXT)
                        )
                        UNION ALL
                        SELECT league_name AS competition, CAST(season AS TEXT) AS season, substr(date, 1, 10) AS match_date,
                               CASE WHEN status_short IN ('FT', 'AET', 'PEN') THEN 1 ELSE 0 END AS finished,
                               CASE WHEN status_short NOT IN ('FT', 'AET', 'PEN', 'CANC', 'PST') THEN 1 ELSE 0 END AS upcoming,
                               'api-football' AS source
                        FROM api_football_fixtures
                    )
                    SELECT COALESCE(competition, 'Unknown') AS competition,
                           COALESCE(season, '-') AS season,
                           COUNT(*) AS rows,
                           SUM(finished) AS finished,
                           SUM(upcoming) AS upcoming,
                           MIN(match_date) AS first_date,
                           MAX(match_date) AS last_date,
                           GROUP_CONCAT(DISTINCT source) AS sources
                    FROM combined
                    WHERE competition IS NOT NULL
                    GROUP BY competition, season
                    ORDER BY rows DESC, competition
                    LIMIT 200
                    """
                ).fetchall()
            return {"competitions": [dict(row) for row in rows]}

        return cached("data_explorer_competitions", 60, factory)

    @app.get("/api/data-explorer/fixtures")
    def data_explorer_fixtures(
        search: str | None = Query(default=None),
        competition: str | None = Query(default=None),
        status: str | None = Query(default=None),
        date_from: str | None = Query(default=None),
        date_to: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=300),
    ) -> dict:
        clauses = []
        params: list[str | int] = []
        if search:
            clauses.append("(home_team LIKE ? OR away_team LIKE ? OR competition LIKE ? OR venue_name LIKE ?)")
            term = f"%{search}%"
            params.extend([term, term, term, term])
        if competition:
            clauses.append("competition = ?")
            params.append(competition)
        if status and status != "all":
            if status == "finished":
                clauses.append("status_group = 'finished'")
            elif status == "upcoming":
                clauses.append("status_group = 'upcoming'")
            elif status == "live":
                clauses.append("status_group = 'live'")
            else:
                clauses.append("status = ?")
                params.append(status)
        if date_from:
            clauses.append("match_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("match_date <= ?")
            params.append(date_to)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            WITH combined AS (
                SELECT CAST(id AS TEXT) AS id,
                       date AS match_date,
                       competition AS competition,
                       season AS season,
                       home_team AS home_team,
                       away_team AS away_team,
                       CASE WHEN home_goals IS NULL OR away_goals IS NULL THEN NULL ELSE home_goals || '-' || away_goals END AS score,
                       CASE WHEN home_goals IS NOT NULL AND away_goals IS NOT NULL THEN 'FT' ELSE 'HIST' END AS status,
                       CASE WHEN home_goals IS NOT NULL AND away_goals IS NOT NULL THEN 'finished' ELSE 'unknown' END AS status_group,
                       NULL AS venue_name,
                       NULL AS venue_city,
                       'historical' AS source,
                       NULL AS api_fixture_id
                FROM matches
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM api_football_fixtures f
                    WHERE matches.date = substr(f.date, 1, 10)
                      AND lower(matches.home_team) = lower(f.home_team)
                      AND lower(matches.away_team) = lower(f.away_team)
                      AND lower(matches.competition) = lower(f.league_name)
                      AND CAST(matches.season AS TEXT) = CAST(f.season AS TEXT)
                )
                UNION ALL
                SELECT CAST(fixture_id AS TEXT) AS id,
                       substr(date, 1, 10) AS match_date,
                       league_name AS competition,
                       CAST(season AS TEXT) AS season,
                       home_team AS home_team,
                       away_team AS away_team,
                       CASE
                           WHEN json_extract(raw_json, '$.goals.home') IS NULL OR json_extract(raw_json, '$.goals.away') IS NULL
                           THEN NULL
                           ELSE json_extract(raw_json, '$.goals.home') || '-' || json_extract(raw_json, '$.goals.away')
                       END AS score,
                       status_short AS status,
                       CASE
                           WHEN status_short IN ('FT', 'AET', 'PEN') THEN 'finished'
                           WHEN status_short IN ('1H', 'HT', '2H', 'ET', 'BT', 'P', 'SUSP', 'INT') THEN 'live'
                           WHEN status_short IN ('CANC', 'PST', 'ABD', 'AWD', 'WO') THEN 'inactive'
                           ELSE 'upcoming'
                       END AS status_group,
                       venue_name AS venue_name,
                       venue_city AS venue_city,
                       'api-football' AS source,
                       fixture_id AS api_fixture_id
                FROM api_football_fixtures
            )
            SELECT *
            FROM combined
            {where_sql}
            ORDER BY match_date DESC, competition, home_team
            LIMIT ?
        """
        params.append(limit)
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            totals = conn.execute(
                """
                SELECT
                    (
                        SELECT COUNT(*)
                        FROM matches m
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM api_football_fixtures f
                            WHERE m.date = substr(f.date, 1, 10)
                              AND lower(m.home_team) = lower(f.home_team)
                              AND lower(m.away_team) = lower(f.away_team)
                              AND lower(m.competition) = lower(f.league_name)
                              AND CAST(m.season AS TEXT) = CAST(f.season AS TEXT)
                        )
                    ) AS historical_matches,
                    (SELECT COUNT(*) FROM api_football_fixtures) AS api_fixtures,
                    (SELECT COUNT(DISTINCT league_name) FROM api_football_fixtures WHERE league_name IS NOT NULL) AS api_competitions,
                    (SELECT COUNT(*) FROM api_football_fixtures WHERE status_short IN ('FT', 'AET', 'PEN')) AS api_finished,
                    (SELECT COUNT(*) FROM api_football_fixtures WHERE status_short NOT IN ('FT', 'AET', 'PEN', 'CANC', 'PST')) AS api_upcoming
                """
            ).fetchone()
        return {"fixtures": [dict(row) for row in rows], "totals": dict(totals)}

    @app.get("/api/backtest")
    def backtest(min_train_matches: int = Query(default=8, ge=1, le=100), max_matches: int = Query(default=250, ge=50, le=2000)) -> dict:
        matches = load_matches().tail(max_matches)
        results, metrics = run_backtest(matches, min_train_matches=min_train_matches)
        recent = results.tail(12).to_dict("records") if not results.empty else []
        return {"metrics": metrics, "recent": recent}

    @app.get("/api/prediction-backtests")
    def prediction_backtests(limit: int = Query(default=100, ge=1, le=500)) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            rows = list_prediction_backtest_summaries(conn, limit=limit)
        metrics = compute_backtest_metrics(rows)
        return {
            "metrics": {key: value for key, value in metrics.items() if key != "rows"},
            "rows": metrics["rows"][:limit],
        }

    @app.get("/api/prediction-history")
    def prediction_history(limit: int = Query(default=200, ge=1, le=1000)) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            rows = list_prediction_backtest_summaries(conn, limit=limit)
        history = []
        for row in rows:
            item = dict(row)
            item["status"] = "evaluated" if item.get("actual_home_goals") is not None and item.get("actual_away_goals") is not None else "pending"
            item["top_scores"] = parse_scores(item.get("predicted_scores_json"))
            if item["status"] == "evaluated":
                item.update(add_evaluation(item))
            history.append(item)
        return {
            "rows": history,
            "counts": {
                "total": len(history),
                "pending": sum(1 for row in history if row["status"] == "pending"),
                "evaluated": sum(1 for row in history if row["status"] == "evaluated"),
            },
        }

    @app.get("/api/prediction-history/{prediction_id}")
    def prediction_history_detail(prediction_id: int) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM prediction_backtests
                WHERE id = ?
                """,
                (prediction_id,),
            ).fetchone()
            if not row:
                return {"error": "prediction_not_found"}
            item = dict(row)
            context = get_match_context(conn, item["home_team"], item["away_team"], match_date=item["match_date"])
        snapshot = json.loads(item.get("snapshot_json") or "{}")
        item["status"] = "evaluated" if item.get("actual_home_goals") is not None and item.get("actual_away_goals") is not None else "pending"
        item["top_scores"] = parse_scores(item.get("predicted_scores_json"))
        item["snapshot"] = snapshot
        item["match_context"] = snapshot.get("match_context") or context
        if item["status"] == "evaluated":
            item.update(add_evaluation(item))
            item["market_errors"] = {
                "corners": _market_error(item.get("predicted_corners"), item.get("actual_corners")),
                "shots_on_target": _market_error(item.get("predicted_shots_on_target"), item.get("actual_shots_on_target")),
                "cards": _market_error(item.get("predicted_cards"), item.get("actual_cards")),
            }
        return item

    @app.get("/api/tracking/pending")
    def tracking_pending() -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            return {"pending": list_pending_prediction_backtests(conn)}

    @app.delete("/api/tracking/snapshot/{backtest_id}")
    def delete_tracking_snapshot(backtest_id: int) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            deleted = delete_pending_prediction_backtest(conn, backtest_id)
            conn.commit()
            clear_cache()
            pending = list_pending_prediction_backtests(conn)
        return {"deleted": deleted, "pending": pending}

    @app.post("/api/tracking/snapshot")
    def save_tracking_snapshot(request: TrackingSnapshotRequest) -> dict:
        matches = load_matches()
        team_metrics = load_team_metrics()
        predictor = load_predictor()
        prediction = predictor.predict(request.home_team, request.away_team)
        with connect(resolved_db_path) as conn:
            venue = get_venue(conn, request.venue_id)
            match_context = get_match_context(conn, request.home_team, request.away_team, request.match_date)
        simulation = simulate_advanced_match(
            matches,
            team_metrics,
            request.home_team,
            request.away_team,
            simulations=request.simulations,
            mode=request.mode,
            venue=venue,
            match_context=match_context,
            predictor=predictor,
            home_profile=load_factor_profile(request.home_team),
            away_profile=load_factor_profile(request.away_team),
        )
        markets = apply_market_calibration(
            estimate_secondary_markets(load_advanced_stats(), request.home_team, request.away_team),
            load_prediction_backtest_metrics(),
        )
        player_markets = load_player_markets(request.home_team, request.away_team, markets)
        with connect(resolved_db_path) as quality_conn:
            quality_snapshot = data_coverage(quality_conn)
        probs = prediction.model_dump()
        simulated_probs = simulation.get("simulation", {})
        home_prob = simulated_probs.get("home_win", probs["home_win_prob"])
        draw_prob = simulated_probs.get("draw", probs["draw_prob"])
        away_prob = simulated_probs.get("away_win", probs["away_win_prob"])
        scores = simulation.get("top_scores", [])[:5]
        row = {
            "match_date": request.match_date,
            "competition": request.competition,
            "home_team": request.home_team,
            "away_team": request.away_team,
            "model_version": request.model_version,
            "predicted_home_prob": home_prob,
            "predicted_draw_prob": draw_prob,
            "predicted_away_prob": away_prob,
            "predicted_pick": max(
                {"home": home_prob, "draw": draw_prob, "away": away_prob},
                key={"home": home_prob, "draw": draw_prob, "away": away_prob}.get,
            ),
            "predicted_scores_json": __import__("json").dumps(scores),
            "predicted_corners": markets.get("total_corners_expected"),
            "predicted_shots_on_target": markets.get("shots_on_target_expected"),
            "predicted_cards": markets.get("total_cards_expected"),
            "predicted_over_2_5_prob": probs["over_2_5_prob"],
            "predicted_btts_prob": probs["both_teams_score_prob"],
            "actual_home_goals": None,
            "actual_away_goals": None,
            "actual_corners": None,
            "actual_shots_on_target": None,
            "actual_cards": None,
            "source": "ui_snapshot",
            "notes": request.notes or "",
            "snapshot": _snapshot_payload(
                probs,
                simulation,
                markets,
                match_context,
                quality_snapshot,
                fixture=None,
                player_markets=player_markets,
            ),
        }
        with connect(resolved_db_path) as conn:
            upsert_prediction_backtest(conn, row)
            conn.commit()
            clear_cache()
            pending = list_pending_prediction_backtests(conn)
        return {"saved": True, "snapshot": row, "pending": pending}

    @app.post("/api/tracking/result/{backtest_id}")
    def save_tracking_result(backtest_id: int, request: TrackingResultRequest) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            update_prediction_backtest_result(conn, backtest_id, request.model_dump())
            conn.commit()
            clear_cache()
            rows = list_prediction_backtests(conn, limit=500)
        metrics = compute_backtest_metrics(rows)
        return {"saved": True, "metrics": {key: value for key, value in metrics.items() if key != "rows"}}

    @app.get("/api/operations/today")
    def today_operations(target_date: str | None = Query(default=None), range_days: int = Query(default=3)) -> dict:
        target = target_date or date.today().isoformat()
        dates = _date_window(target, range_days)
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            fixtures = conn.execute(
                f"""
                SELECT fixture_id, date, league_id, league_name, season, home_team, away_team,
                       status_short, venue_name, venue_city
                FROM api_football_fixtures
                WHERE substr(date, 1, 10) IN ({','.join('?' for _ in dates)})
                ORDER BY date ASC, league_name, home_team
                """,
                tuple(dates),
            ).fetchall()
            pending_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM prediction_backtests
                WHERE actual_home_goals IS NULL OR actual_away_goals IS NULL
                """
            ).fetchone()[0]
            competition_rows = conn.execute(
                """
                SELECT enabled, priority, api_football_league_id
                FROM competitions
                WHERE api_football_league_id IS NOT NULL
                """
            ).fetchall()

        if competition_rows:
            enabled = [dict(row) for row in competition_rows if row["enabled"]]
            priority_one = [row for row in enabled if row["priority"] == 1]
        else:
            catalog = competition_catalog()["competitions"]
            enabled = [row for row in catalog if row.get("api_football_league_id")]
            priority_one = [row for row in enabled if row.get("priority") == 1]

        enabled_ids = {row["api_football_league_id"] for row in enabled}
        priority_ids = {row["api_football_league_id"] for row in priority_one}
        fixture_rows = []
        for row in fixtures:
            item = dict(row)
            item["configured"] = item["league_id"] in enabled_ids
            item["priority"] = item["league_id"] in priority_ids
            fixture_rows.append(item)
        fixture_rows.sort(key=lambda row: (not row["priority"], not row["configured"], row["date"] or "", row["league_name"] or ""))
        relevant_count = sum(1 for row in fixture_rows if row["configured"])
        priority_fixture_count = sum(1 for row in fixture_rows if row["priority"])
        return {
            "date": target,
            "dates": dates,
            "fixtures": fixture_rows,
            "counts": {
                "fixtures_today": len(fixture_rows),
                "configured_fixtures_today": relevant_count,
                "priority_fixtures_today": priority_fixture_count,
                "pending_snapshots": pending_count,
                "enabled_competitions": len(enabled),
                "priority_competitions": len(priority_one),
            },
            "api_budget": {
                "free_daily_limit": 100,
                "fixtures_scan_priority": 1,
                "fixtures_scan_all": 1,
                "details_minimum": priority_fixture_count * 2,
                "details_full": relevant_count * 3,
                "estimated_minimum": 1 + priority_fixture_count * 2,
                "estimated_full": 1 + relevant_count * 3,
            },
        }

    @app.post("/api/operations/sync-today")
    def sync_today_from_api(request: SyncTodayRequest) -> dict:
        target = request.date or date.today().isoformat()
        init_db(resolved_db_path)
        client = ApiFootballClient()
        dates = _date_window(target, request.range_days)
        results = []
        with connect(resolved_db_path) as conn:
            for item_date in dates:
                results.append(
                    {
                        "date": item_date,
                        **sync_api_football_fixtures_by_date(
                            conn,
                            client,
                            date=item_date,
                            league=request.league,
                            season=request.season,
                        ),
                    }
                )
            insert_daily_job(
                conn,
                {
                    "job_date": target,
                    "competition": str(request.league or "all"),
                    "status": "api_football_synced",
                    "fixtures_found": sum(item["fixtures"] for item in results),
                    "predictions_created": 0,
                    "results_updated": sum(item["finished_matches_inserted"] for item in results),
                    "errors": [item for item in results if item.get("api_errors")],
                },
            )
            conn.commit()
            clear_cache()
        return {
            "date": target,
            "dates": dates,
            "fixtures": sum(item["fixtures"] for item in results),
            "finished_matches_inserted": sum(item["finished_matches_inserted"] for item in results),
            "daily": results,
        }

    @app.post("/api/operations/sync-standings")
    def sync_standings_from_api(request: SeasonEnrichmentRequest) -> dict:
        init_db(resolved_db_path)
        client = ApiFootballClient()
        with connect(resolved_db_path) as conn:
            result = sync_api_football_standings(conn, client, league=request.league, season=request.season)
            conn.commit()
            clear_cache()
        return result

    @app.post("/api/operations/sync-team-stats")
    def sync_team_stats_from_api(request: SeasonEnrichmentRequest) -> dict:
        init_db(resolved_db_path)
        client = ApiFootballClient()
        with connect(resolved_db_path) as conn:
            result = sync_api_football_team_statistics(
                conn,
                client,
                league=request.league,
                season=request.season,
                limit=request.team_limit,
            )
            conn.commit()
            clear_cache()
        return result

    @app.post("/api/operations/sync-fixture-details")
    def sync_fixture_details_from_api(request: FixtureDetailsRequest) -> dict:
        init_db(resolved_db_path)
        client = ApiFootballClient()
        with connect(resolved_db_path) as conn:
            result = sync_api_football_fixture_details_by_id(conn, client, fixture_id=request.fixture_id)
            conn.commit()
            clear_cache()
        return result

    @app.post("/api/operations/sync-enrichment")
    def sync_enrichment_from_api(request: DailyEnrichmentRequest) -> dict:
        target = request.date or date.today().isoformat()
        init_db(resolved_db_path)
        client = ApiFootballClient()
        detail_results = []
        with connect(resolved_db_path) as conn:
            standings = sync_api_football_standings(conn, client, league=request.league, season=request.season)
            team_stats = sync_api_football_team_statistics(
                conn,
                client,
                league=request.league,
                season=request.season,
                limit=request.team_limit,
            )
            fixtures = fixture_detail_rows_for_date(
                conn,
                target,
                league=request.league,
                season=request.season,
                limit=request.fixture_limit,
            )
            for fixture in fixtures:
                raw = json.loads(fixture.get("raw_json") or "{}")
                if raw:
                    details = sync_api_football_fixture_details(conn, client, int(fixture["fixture_id"]), raw)
                else:
                    details = sync_api_football_fixture_details_by_id(conn, client, fixture_id=int(fixture["fixture_id"]))
                detail_results.append(
                    {
                        "fixture_id": fixture["fixture_id"],
                        "home_team": fixture["home_team"],
                        "away_team": fixture["away_team"],
                        **details,
                    }
                )
            conn.commit()
            clear_cache()
        return {
            "date": target,
            "league": request.league,
            "season": request.season,
            "standings": standings,
            "team_stats": team_stats,
            "fixture_details": detail_results,
            "requests_estimated": 1 + int(team_stats.get("requests_used", 0)) + len(detail_results) * 3,
        }

    @app.post("/api/operations/predict-today")
    def predict_today_from_fixtures(request: DailyPredictionRequest) -> dict:
        target = request.date or date.today().isoformat()
        return create_daily_prediction_snapshots(
            target,
            simulations=request.simulations,
            configured_only=request.configured_only,
            league=request.league,
            season=request.season,
            range_days=request.range_days,
        )

    @app.post("/api/operations/update-results")
    def update_results_from_fixtures(request: DailyResultsRequest) -> dict:
        target = request.date or date.today().isoformat()
        dates = _date_window(target, request.range_days)
        updates = [update_results_from_local_fixtures(item_date) for item_date in dates]
        return {
            "date": target,
            "dates": dates,
            "finished_fixtures": sum(item["finished_fixtures"] for item in updates),
            "results_updated": sum(item["results_updated"] for item in updates),
            "daily": updates,
            "metrics": updates[-1].get("metrics", {}) if updates else {},
        }

    @app.post("/api/operations/reconcile-pending-results")
    def reconcile_pending_results_from_fixtures(request: PendingReconciliationRequest) -> dict:
        return reconcile_pending_results(
            max_dates=request.max_dates,
            sync_api=request.sync_api,
            league=request.league,
            season=request.season,
        )

    @app.get("/api/operations/betting-board")
    def betting_board(target_date: str | None = Query(default=None), range_days: int = Query(default=3)) -> dict:
        target = target_date or date.today().isoformat()
        dates = _date_window(target, range_days)
        with connect(resolved_db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM prediction_backtests
                WHERE match_date IN ({','.join('?' for _ in dates)})
                ORDER BY match_date ASC, competition, home_team
                """,
                tuple(dates),
            ).fetchall()
        board = []
        for row in rows:
            item = dict(row)
            snapshot = json.loads(item.get("snapshot_json") or "{}")
            picks = snapshot.get("betting_picks") or _betting_picks(
                item["home_team"],
                item["away_team"],
                item.get("predicted_home_prob") or 0,
                item.get("predicted_draw_prob") or 0,
                item.get("predicted_away_prob") or 0,
                item.get("predicted_over_2_5_prob") or 0,
                item.get("predicted_btts_prob") or 0,
                {},
            )
            best_pick = picks[0] if picks else None
            board.append(
                {
                    "id": item["id"],
                    "match_date": item["match_date"],
                    "competition": item["competition"],
                    "home_team": item["home_team"],
                    "away_team": item["away_team"],
                    "status": "evaluated" if item.get("actual_home_goals") is not None and item.get("actual_away_goals") is not None else "pending",
                    "best_pick": best_pick,
                    "picks": picks,
                }
            )
        return {"date": target, "dates": dates, "rows": board}

    @app.post("/api/operations/run-daily")
    def run_daily_pipeline(request: DailyPipelineRequest) -> dict:
        target = request.date or date.today().isoformat()
        sync_result = None
        if request.sync_first:
            client = ApiFootballClient()
            with connect(resolved_db_path) as conn:
                daily_sync = []
                for item_date in _date_window(target, request.range_days):
                    daily_sync.append(
                        {
                            "date": item_date,
                            **sync_api_football_fixtures_by_date(
                                conn,
                                client,
                                date=item_date,
                                league=request.league,
                                season=request.season,
                            ),
                        }
                    )
                sync_result = {
                    "fixtures": sum(item["fixtures"] for item in daily_sync),
                    "finished_matches_inserted": sum(item["finished_matches_inserted"] for item in daily_sync),
                    "daily": daily_sync,
                }
                conn.commit()
                clear_cache()
        prediction_result = create_daily_prediction_snapshots(
            target,
            simulations=request.simulations,
            configured_only=request.configured_only,
            league=request.league,
            season=request.season,
            range_days=request.range_days,
        )
        updates = [update_results_from_local_fixtures(item_date) for item_date in _date_window(target, request.range_days)]
        results_update = {
            "finished_fixtures": sum(item["finished_fixtures"] for item in updates),
            "results_updated": sum(item["results_updated"] for item in updates),
            "daily": updates,
        }
        pending_reconciliation = reconcile_pending_results(max_dates=7, sync_api=False)
        return {
            "date": target,
            "sync": sync_result,
            "predictions": prediction_result,
            "results": results_update,
            "pending_reconciliation": pending_reconciliation,
        }

    @app.post("/api/predict")
    def predict(request: PredictionRequest) -> dict:
        prediction = load_predictor().predict(request.home_team, request.away_team)
        payload = prediction.model_dump()
        markets = estimate_secondary_markets(load_advanced_stats(), request.home_team, request.away_team)
        payload["player_markets"] = load_player_markets(request.home_team, request.away_team, markets)
        if request.save:
            with connect(resolved_db_path) as conn:
                payload["id"] = insert_prediction(conn, payload)
                conn.commit()
                clear_cache()
        return payload

    @app.post("/api/simulate")
    def simulate(request: SimulationRequest) -> dict:
        return simulate_match(load_predictor(), request.home_team, request.away_team, simulations=request.simulations)

    @app.post("/api/simulate/advanced")
    def simulate_advanced(request: SimulationRequest) -> dict:
        with connect(resolved_db_path) as conn:
            venue = get_venue(conn, request.venue_id)
            match_context = get_match_context(conn, request.home_team, request.away_team)
        result = simulate_advanced_match(
            load_matches(),
            load_team_metrics(),
            request.home_team,
            request.away_team,
            simulations=request.simulations,
            mode=request.mode,
            venue=venue,
            match_context=match_context,
            predictor=load_predictor(),
            home_profile=load_factor_profile(request.home_team),
            away_profile=load_factor_profile(request.away_team),
        )
        result["secondary_markets"] = apply_market_calibration(
            estimate_secondary_markets(load_advanced_stats(), request.home_team, request.away_team),
            load_prediction_backtest_metrics(),
        )
        result["player_markets"] = load_player_markets(request.home_team, request.away_team, result["secondary_markets"])
        with connect(resolved_db_path) as conn:
            insert_experiment(
                conn,
                name=f"{request.home_team} vs {request.away_team}",
                kind="simulation",
                config=request.model_dump(),
                metrics=result.get("simulation", {}),
            )
            conn.commit()
        return result

    @app.post("/api/simulate/fixtures")
    def simulate_fixtures(request: FixtureSimulationRequest) -> dict:
        return {
            "fixtures": simulate_fixture_list(
                load_matches(),
                load_team_metrics(),
                [fixture.model_dump() for fixture in request.fixtures],
                simulations=request.simulations,
                mode=request.mode,
                predictor=load_predictor(),
            )
        }

    @app.get("/api/team-profile/{team}")
    def team_profile(team: str) -> dict:
        return load_factor_profile(team)

    return app


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    save: bool = False


class SimulationRequest(BaseModel):
    home_team: str
    away_team: str
    simulations: int = 10000
    mode: str = "hybrid"
    venue_id: int | None = None


class TrackingSnapshotRequest(BaseModel):
    match_date: str
    competition: str = "Manual"
    home_team: str
    away_team: str
    simulations: int = 10000
    mode: str = "hybrid"
    venue_id: int | None = None
    model_version: str = "ui_manual_snapshot"
    notes: str | None = None


class TrackingResultRequest(BaseModel):
    actual_home_goals: int
    actual_away_goals: int
    actual_corners: float | None = None
    actual_shots_on_target: float | None = None
    actual_cards: float | None = None
    notes: str | None = None


class SyncTodayRequest(BaseModel):
    date: str | None = None
    league: int | None = None
    season: int | None = None
    range_days: int = 1


class SeasonEnrichmentRequest(BaseModel):
    league: int
    season: int
    team_limit: int | None = 8


class FixtureDetailsRequest(BaseModel):
    fixture_id: int


class DailyEnrichmentRequest(BaseModel):
    date: str | None = None
    league: int
    season: int
    team_limit: int | None = 8
    fixture_limit: int | None = 2


class DailyPredictionRequest(BaseModel):
    date: str | None = None
    league: int | None = None
    season: int | None = None
    simulations: int = 5000
    configured_only: bool = True
    range_days: int = 1


class DailyResultsRequest(BaseModel):
    date: str | None = None
    range_days: int = 1


class PendingReconciliationRequest(BaseModel):
    max_dates: int = 7
    sync_api: bool = False
    league: int | None = None
    season: int | None = None


class DailyPipelineRequest(BaseModel):
    date: str | None = None
    league: int | None = None
    season: int | None = None
    simulations: int = 5000
    configured_only: bool = True
    sync_first: bool = True
    range_days: int = 1


class FixtureRequest(BaseModel):
    home_team: str
    away_team: str


class FixtureSimulationRequest(BaseModel):
    fixtures: list[FixtureRequest]
    simulations: int = 5000
    mode: str = "hybrid"


app = create_app()


