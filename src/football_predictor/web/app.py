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
from football_predictor.data.pending_reconciliation import run_pending_fixture_reconciliation
from football_predictor.database.db import (
    advanced_summary_by_team,
    api_lineup_summary_by_team,
    api_standings_summary_by_team,
    api_team_strength_summary_by_team,
    availability_summary_by_team,
    player_quality_summary_by_team,
    connect,
    delete_prediction_backtest,
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
    reconcile_prediction_backtests_for_date,
    update_prediction_backtest_result,
    upsert_prediction_backtest,
)
from football_predictor.evaluation.backtest import run_backtest
from football_predictor.evaluation.prediction_backtest import add_evaluation, compute_backtest_metrics, parse_scores
from football_predictor.prediction.factor_model import build_team_profile
from football_predictor.prediction.calibration import apply_market_calibration
from football_predictor.prediction.markets import estimate_secondary_markets
from football_predictor.prediction.monte_carlo import simulate_market_distributions
from football_predictor.prediction.pick_engine import generate_betting_picks, generate_pick_report
from football_predictor.prediction.player_markets import estimate_player_markets
from football_predictor.prediction.referee_model import referee_card_profile
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


def _prediction_lifecycle(row: dict) -> dict:
    has_score = row.get("actual_home_goals") is not None and row.get("actual_away_goals") is not None
    market_fields = {
        "corners": row.get("actual_corners"),
        "shots_on_target": row.get("actual_shots_on_target"),
        "cards": row.get("actual_cards"),
    }
    missing_markets = [name for name, value in market_fields.items() if value is None]
    match_date = str(row.get("match_date") or "")[:10]
    is_past = False
    try:
        is_past = bool(match_date) and date.fromisoformat(match_date) < date.today()
    except ValueError:
        is_past = False
    if has_score and not missing_markets:
        lifecycle_status = "stats_complete"
        label = "Stats completas"
    elif has_score:
        lifecycle_status = "score_updated"
        label = "Marcador actualizado"
    elif is_past:
        lifecycle_status = "result_late"
        label = "Resultado pendiente atrasado"
    else:
        lifecycle_status = "awaiting_result"
        label = "Esperando resultado"
    return {
        "status": "evaluated" if has_score else "pending",
        "lifecycle_status": lifecycle_status,
        "lifecycle_label": label,
        "missing_actuals": ([] if has_score else ["score"]) + ([] if not has_score else missing_markets),
        "is_limbo": lifecycle_status == "result_late" or (has_score and bool(missing_markets)),
    }


def _with_prediction_lifecycle(rows: list[dict]) -> list[dict]:
    items = []
    for row in rows:
        item = dict(row)
        item.update(_prediction_lifecycle(item))
        items.append(item)
    return items


def _value_changed(current: object, new: object) -> bool:
    if current is None and new is None:
        return False
    if current is None or new is None:
        return True
    try:
        return abs(float(current) - float(new)) > 1e-9
    except (TypeError, ValueError):
        return current != new


def _snapshot_payload(
    prediction: dict,
    simulation: dict,
    markets: dict,
    match_context: dict | None,
    data_quality: dict | None,
    fixture: dict | None = None,
    betting_picks: list[dict] | None = None,
    player_markets: dict | None = None,
    market_distribution: dict | None = None,
    pick_report: dict | None = None,
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
        "market_distribution": market_distribution or {},
        "match_context": match_context,
        "data_quality": data_quality,
        "fixture": fixture,
        "betting_picks": betting_picks or [],
        "pick_report": pick_report or {},
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
    market_distribution: dict | None = None,
) -> list[dict]:
    return generate_betting_picks(
        home_team,
        away_team,
        home_prob,
        draw_prob,
        away_prob,
        over_2_5_prob,
        btts_prob,
        markets,
        market_distribution,
    )


def _betting_pick_report(
    home_team: str,
    away_team: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    over_2_5_prob: float,
    btts_prob: float,
    markets: dict,
    market_distribution: dict | None = None,
) -> dict:
    return generate_pick_report(
        home_team,
        away_team,
        home_prob,
        draw_prob,
        away_prob,
        over_2_5_prob,
        btts_prob,
        markets,
        market_distribution,
    )
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
    team_markets = markets.get("team_markets") or {}
    for side, label in (("home", home_team), ("away", away_team)):
        team = team_markets.get(side) or {}
        corners_team_prob = team.get("over_3_5_corners_prob")
        if corners_team_prob is not None and float(corners_team_prob) >= 0.58:
            picks.append(
                {
                    "market": f"Corners {label}",
                    "pick": "Over 3.5",
                    "probability": round(float(corners_team_prob), 4),
                    "confidence": _confidence_label(float(corners_team_prob), 0.08),
                    "risk": "medio",
                }
            )
        shots_team_prob = team.get("over_3_5_shots_on_target_prob")
        if shots_team_prob is not None and float(shots_team_prob) >= 0.58:
            picks.append(
                {
                    "market": f"Tiros puerta {label}",
                    "pick": "Over 3.5",
                    "probability": round(float(shots_team_prob), 4),
                    "confidence": _confidence_label(float(shots_team_prob), 0.08),
                    "risk": "medio",
                }
            )
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

    def load_referee_profile(fixture_id: int | None) -> dict | None:
        if fixture_id is None:
            return None

        def factory() -> dict | None:
            with connect(resolved_db_path) as conn:
                return referee_card_profile(conn, fixture_id)

        return cached(f"referee_profile:{fixture_id}", 60, factory)

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
                referee_profile = referee_card_profile(conn, int(fixture["fixture_id"])) if fixture.get("fixture_id") else None
                markets = apply_market_calibration(estimate_secondary_markets(advanced_stats, home, away, referee_profile), existing_metrics)
                market_distribution = simulate_market_distributions(
                    prediction,
                    markets,
                    simulations=simulations,
                    match_context=match_context,
                )
                player_markets = estimate_player_markets(conn, home, away, markets)
                simulated_probs = simulation.get("simulation", {})
                home_prob = simulated_probs.get("home_win", prediction["home_win_prob"])
                draw_prob = simulated_probs.get("draw", prediction["draw_prob"])
                away_prob = simulated_probs.get("away_win", prediction["away_win_prob"])
                probs = {"home": home_prob, "draw": draw_prob, "away": away_prob}
                pick_report = _betting_pick_report(
                    home,
                    away,
                    home_prob,
                    draw_prob,
                    away_prob,
                    prediction["over_2_5_prob"],
                    prediction["both_teams_score_prob"],
                    markets,
                    market_distribution,
                )
                betting_picks = pick_report["recommended"]
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
                        pick_report=pick_report,
                        player_markets=player_markets,
                        market_distribution=market_distribution,
                    ),
                }
                upsert_prediction_backtest(conn, row)
                snapshots.append({**row, "context": (match_context or {}).get("stage_label"), "betting_picks": betting_picks, "pick_report": pick_report, "player_markets": player_markets, "market_distribution": market_distribution})
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
                COUNT(*) AS team_rows,
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
        empty = {"actual_corners": None, "actual_shots_on_target": None, "actual_cards": None}
        if not row or row["team_rows"] < 2:
            return {**empty, "complete": False}
        return {
            "actual_corners": row["actual_corners"],
            "actual_shots_on_target": row["actual_shots_on_target"],
            "actual_cards": row["actual_cards"],
            "complete": True,
        }

    def update_results_from_local_fixtures(target: str) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            result = reconcile_prediction_backtests_for_date(conn, target)
            insert_daily_job(
                conn,
                {
                    "job_date": target,
                    "competition": "all",
                    "status": "results_updated",
                    "fixtures_found": result["finished_fixtures"],
                    "predictions_created": 0,
                    "results_updated": result["results_updated"],
                    "errors": [],
                },
            )
            conn.commit()
            clear_cache()
            metrics = compute_backtest_metrics(list_prediction_backtests(conn, limit=500))
        return {**result, "metrics": {key: value for key, value in metrics.items() if key != "rows"}}

    def reconcile_pending_results(
        max_dates: int = 7,
        sync_api: bool = False,
        league: int | None = None,
        season: int | None = None,
        max_fixtures: int = 100,
        retry_hours: int = 6,
    ) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            result = run_pending_fixture_reconciliation(
                conn,
                client=ApiFootballClient() if sync_api else None,
                sync_api=sync_api,
                max_fixtures=max_fixtures,
                retry_hours=retry_hours,
                league=league,
                season=season,
            )
            insert_daily_job(
                conn,
                {
                    "job_date": date.today().isoformat(),
                    "competition": str(league or "all"),
                    "status": "pending_fixture_reconciliation",
                    "fixtures_found": result["fixtures_pending_found"],
                    "predictions_created": 0,
                    "results_updated": result["snapshots_updated"],
                    "errors": result.get("remaining_finished_pending", []),
                },
            )
            conn.commit()
            clear_cache()
        result["max_dates_legacy"] = max_dates
        result["sync_api"] = sync_api
        return result

    static_dir = Path(__file__).with_name("static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            conn.execute("SELECT 1").fetchone()
        return {"status": "ok"}

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

    @app.get("/api/data-explorer/overview")
    def data_explorer_overview() -> dict:
        def factory() -> dict:
            init_db(resolved_db_path)
            with connect(resolved_db_path) as conn:
                row = conn.execute(
                    """
                    SELECT
                        (SELECT COUNT(DISTINCT competition) FROM matches WHERE competition IS NOT NULL) AS historical_competitions,
                        (SELECT COUNT(DISTINCT league_name) FROM api_football_fixtures WHERE league_name IS NOT NULL) AS api_competitions,
                        (SELECT COUNT(*) FROM api_football_fixtures) AS api_fixtures,
                        (SELECT COUNT(*) FROM matches) AS historical_matches,
                        (SELECT COUNT(*) FROM teams) AS teams,
                        (SELECT COUNT(*) FROM players) AS players,
                        (SELECT COUNT(*) FROM venues) AS venues,
                        (SELECT COUNT(*) FROM referees) AS referees,
                        (SELECT COUNT(*) FROM player_season_stats) AS player_stats,
                        (SELECT COUNT(*) FROM player_match_stats) AS player_match_stats,
                        (SELECT COUNT(*) FROM match_team_advanced_stats) AS advanced_stats,
                        (SELECT COUNT(*) FROM api_football_fixture_lineups) AS fixture_lineups,
                        (SELECT COUNT(*) FROM api_football_league_coverage) AS league_coverage,
                        (SELECT COUNT(*) FROM prediction_backtests) AS predictions,
                        (
                            SELECT MAX(sync_marker)
                            FROM (
                                SELECT MAX(date) AS sync_marker FROM api_football_fixtures
                                UNION ALL SELECT MAX(updated_at) FROM player_season_stats
                                UNION ALL SELECT MAX(updated_at) FROM api_football_league_coverage
                                UNION ALL SELECT MAX(created_at) FROM daily_jobs
                            )
                        ) AS last_sync
                    """
                ).fetchone()
                coverage_rows = conn.execute(
                    """
                    SELECT league_id, league_name, country, season, current,
                           fixtures_events, fixtures_lineups, fixtures_statistics_fixtures,
                           fixtures_statistics_players, standings, players, injuries,
                           predictions, odds, updated_at
                    FROM api_football_league_coverage
                    ORDER BY current DESC, updated_at DESC, league_name
                    LIMIT 80
                    """
                ).fetchall()
            payload = dict(row)
            payload["total_competitions"] = max(payload["historical_competitions"], payload["api_competitions"])
            payload["total_matches"] = payload["historical_matches"] + payload["api_fixtures"]
            payload["stored_statistics"] = (
                payload["player_stats"]
                + payload["player_match_stats"]
                + payload["advanced_stats"]
                + payload["fixture_lineups"]
            )
            payload["coverage"] = [dict(item) for item in coverage_rows]
            return payload

        return cached("data_explorer_overview", 60, factory)

    @app.get("/api/data-explorer/fixtures")
    def data_explorer_fixtures(
        search: str | None = Query(default=None),
        competition: str | None = Query(default=None),
        status: str | None = Query(default=None),
        date_from: str | None = Query(default=None),
        date_to: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=300),
    ) -> dict:
        api_clauses = []
        api_params: list[str | int] = []
        historical_clauses = []
        historical_params: list[str | int] = []
        if search:
            term = f"%{search}%"
            api_clauses.append("(home_team LIKE ? OR away_team LIKE ? OR league_name LIKE ? OR venue_name LIKE ? OR CAST(fixture_id AS TEXT) LIKE ?)")
            api_params.extend([term, term, term, term, term])
            historical_clauses.append("(home_team LIKE ? OR away_team LIKE ? OR competition LIKE ? OR CAST(id AS TEXT) LIKE ?)")
            historical_params.extend([term, term, term, term])
        if competition:
            api_clauses.append("league_name = ?")
            api_params.append(competition)
            historical_clauses.append("competition = ?")
            historical_params.append(competition)
        if status and status != "all":
            if status == "finished":
                api_clauses.append("status_short IN ('FT', 'AET', 'PEN')")
                historical_clauses.append("home_goals IS NOT NULL AND away_goals IS NOT NULL")
            elif status == "upcoming":
                api_clauses.append("status_short NOT IN ('FT', 'AET', 'PEN', 'CANC', 'PST')")
                historical_clauses.append("1 = 0")
            elif status == "live":
                api_clauses.append("status_short IN ('1H', 'HT', '2H', 'ET', 'BT', 'P', 'SUSP', 'INT')")
                historical_clauses.append("1 = 0")
            else:
                api_clauses.append("status_short = ?")
                api_params.append(status)
                historical_clauses.append("1 = 0")
        if date_from:
            api_clauses.append("substr(date, 1, 10) >= ?")
            api_params.append(date_from)
            historical_clauses.append("date >= ?")
            historical_params.append(date_from)
        if date_to:
            api_clauses.append("substr(date, 1, 10) <= ?")
            api_params.append(date_to)
            historical_clauses.append("date <= ?")
            historical_params.append(date_to)

        api_where = f"WHERE {' AND '.join(api_clauses)}" if api_clauses else ""
        historical_where = f"WHERE {' AND '.join(historical_clauses)}" if historical_clauses else ""
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            api_rows = conn.execute(
                f"""
                SELECT CAST(fixture_id AS TEXT) AS id,
                       substr(date, 1, 10) AS match_date,
                       league_name AS competition,
                       CAST(season AS TEXT) AS season,
                       home_team,
                       away_team,
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
                       venue_name,
                       venue_city,
                       'api-football' AS source,
                       fixture_id AS api_fixture_id
                FROM api_football_fixtures
                {api_where}
                ORDER BY date DESC, league_name, home_team
                LIMIT ?
                """,
                tuple([*api_params, limit]),
            ).fetchall()
            rows = [dict(row) for row in api_rows]
            if len(rows) < limit and status not in {"upcoming", "live"}:
                historical_rows = conn.execute(
                    f"""
                    SELECT CAST(id AS TEXT) AS id,
                           date AS match_date,
                           competition,
                           season,
                           home_team,
                           away_team,
                           CASE WHEN home_goals IS NULL OR away_goals IS NULL THEN NULL ELSE home_goals || '-' || away_goals END AS score,
                           CASE WHEN home_goals IS NOT NULL AND away_goals IS NOT NULL THEN 'FT' ELSE 'HIST' END AS status,
                           CASE WHEN home_goals IS NOT NULL AND away_goals IS NOT NULL THEN 'finished' ELSE 'unknown' END AS status_group,
                           NULL AS venue_name,
                           NULL AS venue_city,
                           'historical' AS source,
                           NULL AS api_fixture_id
                    FROM matches
                    {historical_where}
                    ORDER BY date DESC, competition, home_team
                    LIMIT ?
                    """,
                    tuple([*historical_params, limit - len(rows)]),
                ).fetchall()
                rows.extend(dict(row) for row in historical_rows)
            rows = sorted(rows, key=lambda item: (item.get("match_date") or "", item.get("competition") or ""), reverse=True)[:limit]
            totals = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM matches) AS historical_matches,
                    (SELECT COUNT(*) FROM api_football_fixtures) AS api_fixtures,
                    (SELECT COUNT(DISTINCT league_name) FROM api_football_fixtures WHERE league_name IS NOT NULL) AS api_competitions,
                    (SELECT COUNT(*) FROM api_football_fixtures WHERE status_short IN ('FT', 'AET', 'PEN')) AS api_finished,
                    (SELECT COUNT(*) FROM api_football_fixtures WHERE status_short NOT IN ('FT', 'AET', 'PEN', 'CANC', 'PST')) AS api_upcoming
                """
            ).fetchone()
        return {"fixtures": rows, "totals": dict(totals)}

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
            item.update(_prediction_lifecycle(item))
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
                "stats_complete": sum(1 for row in history if row["lifecycle_status"] == "stats_complete"),
                "score_updated": sum(1 for row in history if row["lifecycle_status"] == "score_updated"),
                "limbo": sum(1 for row in history if row["is_limbo"]),
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
        item.update(_prediction_lifecycle(item))
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

    @app.delete("/api/prediction-history/{prediction_id}")
    def delete_prediction_history_item(prediction_id: int) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            deleted = delete_prediction_backtest(conn, prediction_id)
            conn.commit()
            clear_cache()
        return {"deleted": deleted}

    @app.get("/api/tracking/pending")
    def tracking_pending() -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            pending = _with_prediction_lifecycle(list_pending_prediction_backtests(conn))
            return {"pending": pending}

    @app.delete("/api/tracking/snapshot/{backtest_id}")
    def delete_tracking_snapshot(backtest_id: int) -> dict:
        init_db(resolved_db_path)
        with connect(resolved_db_path) as conn:
            deleted = delete_pending_prediction_backtest(conn, backtest_id)
            conn.commit()
            clear_cache()
            pending = _with_prediction_lifecycle(list_pending_prediction_backtests(conn))
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
        probs = prediction.model_dump()
        market_distribution = simulate_market_distributions(
            probs,
            markets,
            simulations=request.simulations,
            match_context=match_context,
        )
        player_markets = load_player_markets(request.home_team, request.away_team, markets)
        with connect(resolved_db_path) as quality_conn:
            quality_snapshot = data_coverage(quality_conn)
        simulated_probs = simulation.get("simulation", {})
        home_prob = simulated_probs.get("home_win", probs["home_win_prob"])
        draw_prob = simulated_probs.get("draw", probs["draw_prob"])
        away_prob = simulated_probs.get("away_win", probs["away_win_prob"])
        pick_report = _betting_pick_report(
            request.home_team,
            request.away_team,
            home_prob,
            draw_prob,
            away_prob,
            probs["over_2_5_prob"],
            probs["both_teams_score_prob"],
            markets,
            market_distribution,
        )
        betting_picks = pick_report["recommended"]
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
                betting_picks=betting_picks,
                pick_report=pick_report,
                player_markets=player_markets,
                market_distribution=market_distribution,
            ),
        }
        with connect(resolved_db_path) as conn:
            upsert_prediction_backtest(conn, row)
            conn.commit()
            clear_cache()
            pending = _with_prediction_lifecycle(list_pending_prediction_backtests(conn))
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
                sync_result = sync_api_football_fixtures_by_date(
                    conn,
                    client,
                    date=item_date,
                    league=request.league,
                    season=request.season,
                )
                reconciliation = reconcile_prediction_backtests_for_date(conn, item_date)
                results.append({"date": item_date, **sync_result, "reconciliation": reconciliation})
            insert_daily_job(
                conn,
                {
                    "job_date": target,
                    "competition": str(request.league or "all"),
                    "status": "api_football_synced",
                    "fixtures_found": sum(item["fixtures"] for item in results),
                    "predictions_created": 0,
                    "results_updated": sum(item["reconciliation"]["results_updated"] for item in results),
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
            "results_updated": sum(item["reconciliation"]["results_updated"] for item in results),
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
            max_fixtures=request.max_fixtures,
            retry_hours=request.retry_hours,
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
                    sync_result_for_date = sync_api_football_fixtures_by_date(
                        conn,
                        client,
                        date=item_date,
                        league=request.league,
                        season=request.season,
                    )
                    reconciliation = reconcile_prediction_backtests_for_date(conn, item_date)
                    daily_sync.append({"date": item_date, **sync_result_for_date, "reconciliation": reconciliation})
                sync_result = {
                    "fixtures": sum(item["fixtures"] for item in daily_sync),
                    "finished_matches_inserted": sum(item["finished_matches_inserted"] for item in daily_sync),
                    "results_updated": sum(item["reconciliation"]["results_updated"] for item in daily_sync),
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
        market_distribution = simulate_market_distributions(payload, markets)
        payload["secondary_markets"] = markets
        payload["market_distribution"] = market_distribution
        payload["player_markets"] = load_player_markets(request.home_team, request.away_team, markets)
        payload["pick_report"] = _betting_pick_report(
            request.home_team,
            request.away_team,
            payload["home_win_prob"],
            payload["draw_prob"],
            payload["away_win_prob"],
            payload["over_2_5_prob"],
            payload["both_teams_score_prob"],
            markets,
            market_distribution,
        )
        payload["betting_picks"] = payload["pick_report"]["recommended"]
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
        result["market_distribution"] = simulate_market_distributions(
            result.get("model_prediction", {}),
            result["secondary_markets"],
            simulations=request.simulations,
            match_context=match_context,
        )
        result["player_markets"] = load_player_markets(request.home_team, request.away_team, result["secondary_markets"])
        sim_probs = result.get("simulation", {})
        model_prediction = result.get("model_prediction", {})
        result["pick_report"] = _betting_pick_report(
            request.home_team,
            request.away_team,
            sim_probs.get("home_win", model_prediction.get("home_win_prob", 0)),
            sim_probs.get("draw", model_prediction.get("draw_prob", 0)),
            sim_probs.get("away_win", model_prediction.get("away_win_prob", 0)),
            model_prediction.get("over_2_5_prob", 0),
            model_prediction.get("both_teams_score_prob", 0),
            result["secondary_markets"],
            result["market_distribution"],
        )
        result["betting_picks"] = result["pick_report"]["recommended"]
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
    max_fixtures: int = 100
    retry_hours: int = 6


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


