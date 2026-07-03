from pathlib import Path
import json
import pickle

import pandas as pd
import typer

from football_predictor.config import settings
from football_predictor.data.api_clients import ApiFootballClient
from football_predictor.data.api_football_sync import (
    backfill_match_context_from_api_fixtures,
    local_competition_season_status,
    sync_api_football_standings,
    sync_api_football_team_statistics,
    sync_api_football_players,
    sync_api_football_league_coverage,
    sync_api_football_fixture_details_bulk,
    sync_competition_season_core,
    sync_api_football_fixture_details_by_id,
    sync_api_football_fixtures_by_date,
    sync_api_football_fixtures,
)
from football_predictor.data.competitions import COMPETITIONS
from football_predictor.data.fifa_rankings import fetch_fifa_mens_rankings
from football_predictor.data.international_history import download_international_results, load_international_results
from football_predictor.data.loaders import load_matches_csv
from football_predictor.data.quality_plan import alias_rows, requirement_rows
from football_predictor.data.mass_sync import run_mass_api_football_sync
from football_predictor.data.pending_reconciliation import run_pending_fixture_reconciliation
from football_predictor.data.worldcup_enriched import (
    download_worldcup_enriched,
    load_worldcup_enriched,
    normalize_advanced_stats,
    normalize_players,
    normalize_venues,
    summarize_squads,
)
from football_predictor.data.worldcup_2026 import download_worldcup_2026, download_worldcup_year, load_worldcup_2026_json, load_worldcup_json
from football_predictor.database.db import (
    connect,
    count_matches_shadowed_by_api_football,
    delete_matches_shadowed_by_api_football,
    insert_experiment,
    init_db,
    insert_matches,
    insert_model_artifact,
    insert_prediction,
    list_teams,
    get_match_context,
    get_team_external_metrics,
    advanced_summary_by_team,
    api_lineup_summary_by_team,
    api_standings_summary_by_team,
    api_team_strength_summary_by_team,
    availability_summary_by_team,
    player_quality_summary_by_team,
    upsert_team_external_metrics,
    upsert_teams,
    upsert_advanced_stats,
    upsert_squad_player,
    upsert_venue,
    list_prediction_backtests,
    upsert_prediction_backtest,
    upsert_competition,
    list_competitions,
    insert_daily_job,
    reconcile_prediction_backtests_for_date,
    list_sync_inventory,
    backfill_match_context_venues,
    upsert_team_alias,
    upsert_data_source_requirement,
)
from football_predictor.data.coverage import data_coverage
from football_predictor.evaluation.backtest import run_backtest
from football_predictor.evaluation.prediction_backtest import compute_backtest_metrics, load_backtest_file
from football_predictor.models.ml_model import MatchOutcomeModel
from football_predictor.models.baseline_poisson import PoissonBaseline
from football_predictor.prediction.predictor import MatchPredictor
from football_predictor.prediction.markets import estimate_secondary_markets
from football_predictor.prediction.player_markets import estimate_player_markets
from football_predictor.prediction.calibration import apply_market_calibration
from football_predictor.prediction.simulator import simulate_advanced_match, simulate_match


app = typer.Typer(help="Herramienta base para prediccion de futbol.")


def _load_matches_from_db(db_path: Path | str | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM matches ORDER BY date", conn)


def _load_team_metrics(db_path: Path | str | None = None) -> dict[str, dict]:
    with connect(db_path) as conn:
        teams = list_teams(conn)
        advanced = advanced_summary_by_team(conn)
        availability = availability_summary_by_team(conn)
        api_strength = api_team_strength_summary_by_team(conn)
        api_standings = api_standings_summary_by_team(conn)
        api_lineups = api_lineup_summary_by_team(conn)
        player_quality = player_quality_summary_by_team(conn)
        metrics = {}
        for team in teams:
            metrics[team] = {
                **get_team_external_metrics(conn, team),
                **advanced.get(team, {}),
                **availability.get(team, {}),
                **api_strength.get(team, {}),
                **api_standings.get(team, {}),
                **api_lineups.get(team, {}),
                **player_quality.get(team, {}),
            }
        return metrics


def _load_advanced_stats_from_db(db_path: Path | str | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM match_team_advanced_stats ORDER BY date", conn)


@app.command("init-db")
def init_database(db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    typer.echo(f"Base SQLite lista: {db_path}")


@app.command("load-csv")
def load_csv(csv_path: Path, db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    df = load_matches_csv(csv_path)
    rows = df.to_dict("records")
    teams = list(df["home_team"]) + list(df["away_team"])
    with connect(db_path) as conn:
        upsert_teams(conn, teams)
        inserted = insert_matches(conn, rows)
        conn.commit()
    typer.echo(f"Partidos cargados: {inserted}")


@app.command("import-worldcup-2026")
def import_worldcup_2026(
    db_path: Path = settings.db_path,
    raw_path: Path = Path("data/raw/worldcup_2026_openfootball.json"),
    refresh: bool = typer.Option(True, help="Descargar de nuevo la fuente publica."),
    include_fixtures: bool = typer.Option(False, help="Incluir partidos futuros sin marcador."),
) -> None:
    init_db(db_path)
    if refresh or not raw_path.exists():
        download_worldcup_2026(raw_path)

    df = load_worldcup_2026_json(raw_path, include_fixtures=include_fixtures)
    rows = df.to_dict("records")
    teams = list(df["home_team"]) + list(df["away_team"]) if not df.empty else []

    with connect(db_path) as conn:
        upsert_teams(conn, teams)
        inserted = insert_matches(conn, rows)
        conn.commit()

    typer.echo(f"Partidos Mundial 2026 cargados: {inserted}")
    typer.echo(f"Partidos leidos desde fuente: {len(df)}")


@app.command("import-worldcups")
def import_worldcups(
    years: str = typer.Option("2014,2018,2022,2026", help="Anios separados por coma."),
    db_path: Path = settings.db_path,
    raw_dir: Path = Path("data/raw/openfootball_worldcups"),
) -> None:
    init_db(db_path)
    total_inserted = 0
    total_rows = 0
    for year_text in years.split(","):
        year = int(year_text.strip())
        raw_path = raw_dir / f"worldcup_{year}.json"
        download_worldcup_year(year, raw_path)
        df = load_worldcup_json(raw_path, season=str(year), include_fixtures=False)
        rows = df.to_dict("records")
        teams = list(df["home_team"]) + list(df["away_team"]) if not df.empty else []
        with connect(db_path) as conn:
            upsert_teams(conn, teams)
            inserted = insert_matches(conn, rows)
            conn.commit()
        total_inserted += inserted
        total_rows += len(df)
    typer.echo(f"Partidos historicos insertados: {total_inserted}")
    typer.echo(f"Partidos historicos leidos: {total_rows}")


@app.command("import-international-history")
def import_international_history(
    db_path: Path = settings.db_path,
    raw_path: Path = Path("data/raw/international_results.csv"),
    start_year: int = 2010,
) -> None:
    init_db(db_path)
    download_international_results(raw_path)
    df = load_international_results(raw_path, start_year=start_year)
    rows = df.to_dict("records")
    teams = list(df["home_team"]) + list(df["away_team"]) if not df.empty else []
    with connect(db_path) as conn:
        upsert_teams(conn, teams)
        inserted = insert_matches(conn, rows)
        conn.commit()
    typer.echo(f"Partidos internacionales leidos: {len(df)}")
    typer.echo(f"Partidos internacionales insertados: {inserted}")


@app.command("import-wc2026-enriched")
def import_wc2026_enriched(
    db_path: Path = settings.db_path,
    raw_dir: Path = Path("data/raw/wc2026_enriched"),
) -> None:
    init_db(db_path)
    download_worldcup_enriched(raw_dir)
    data = load_worldcup_enriched(raw_dir)
    with connect(db_path) as conn:
        for venue in normalize_venues(data["venues"]):
            upsert_venue(conn, venue)
        for metrics in summarize_squads(data["teams"], data["players"]):
            upsert_team_external_metrics(conn, metrics)
        for player in normalize_players(data["teams"], data["players"]):
            upsert_squad_player(conn, player)
        for stats in normalize_advanced_stats(data):
            upsert_advanced_stats(conn, stats)
        conn.commit()
    typer.echo("Dataset enriquecido WC2026 importado")
    typer.echo(f"Sedes: {len(data['venues'])}")
    typer.echo(f"Jugadores: {len(data['players'])}")
    typer.echo(f"Stats equipo-partido: {len(data['stats'])}")


@app.command("train-baseline")
def train_baseline(db_path: Path = settings.db_path, output: Path = Path("models/poisson_baseline.pkl")) -> None:
    init_db(db_path)
    df = _load_matches_from_db(db_path)
    model = PoissonBaseline().fit(df)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(model, fh)
    trained_until = str(df["date"].max()) if not df.empty else None
    with connect(db_path) as conn:
        insert_model_artifact(conn, "poisson_baseline", "poisson", str(output), trained_until)
        conn.commit()
    typer.echo(f"Baseline guardado: {output}")


@app.command("train-ml")
def train_ml(
    db_path: Path = settings.db_path,
    output: Path = Path("models/match_outcome_logistic.pkl"),
    model_type: str = "logistic",
) -> None:
    init_db(db_path)
    df = _load_matches_from_db(db_path)
    model = MatchOutcomeModel(model_type=model_type).fit(df)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(model, fh)
    trained_until = str(df["date"].max()) if not df.empty else None
    with connect(db_path) as conn:
        insert_model_artifact(conn, f"match_outcome_{model_type}", model_type, str(output), trained_until)
        conn.commit()
    typer.echo(f"Modelo ML guardado: {output}")


@app.command("backtest")
def backtest(
    db_path: Path = settings.db_path,
    min_train_matches: int = 5,
    output: Path | None = Path("data/processed/backtest_predictions.csv"),
) -> None:
    df = _load_matches_from_db(db_path)
    results, metrics = run_backtest(df, min_train_matches=min_train_matches)
    if output is not None and not results.empty:
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
    with connect(db_path) as conn:
        insert_experiment(
            conn,
            name="backtest",
            kind="evaluation",
            config={"min_train_matches": min_train_matches},
            metrics=metrics,
            artifact_path=str(output) if output else None,
        )
        conn.commit()
    typer.echo(json.dumps(metrics, indent=2))


@app.command("import-prediction-backtest")
def import_prediction_backtest(
    path: Path,
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    rows = load_backtest_file(path)
    with connect(db_path) as conn:
        for row in rows:
            upsert_prediction_backtest(conn, row)
        conn.commit()
    typer.echo(f"Backtests importados: {len(rows)}")


@app.command("prediction-backtest-report")
def prediction_backtest_report(db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        metrics = compute_backtest_metrics(list_prediction_backtests(conn))
    typer.echo(json.dumps({key: value for key, value in metrics.items() if key != "rows"}, indent=2))


@app.command("sync-competition-catalog")
def sync_competition_catalog(
    season: int | None = typer.Option(None, help="Temporada activa opcional."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        for competition in COMPETITIONS:
            upsert_competition(conn, competition, season=season)
        conn.commit()
    typer.echo(f"Competiciones preparadas: {len(COMPETITIONS)}")


@app.command("show-competitions")
def show_competitions(db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = list_competitions(conn)
    typer.echo(json.dumps(rows, indent=2))


@app.command("today-fixtures")
def today_fixtures(
    date: str | None = typer.Option(None, help="Fecha YYYY-MM-DD. Por defecto usa hoy."),
    db_path: Path = settings.db_path,
    configured_only: bool = typer.Option(True, help="Mostrar solo competiciones configuradas."),
) -> None:
    from datetime import date as date_type

    target = date or date_type.today().isoformat()
    init_db(db_path)
    with connect(db_path) as conn:
        if configured_only:
            rows = conn.execute(
                """
                SELECT f.fixture_id, f.date, f.league_name, f.season, f.home_team, f.away_team, f.status_short
                FROM api_football_fixtures f
                JOIN competitions c ON c.api_football_league_id = f.league_id AND c.enabled = 1
                WHERE substr(f.date, 1, 10) = ?
                ORDER BY c.priority ASC, f.date
                """,
                (target,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT fixture_id, date, league_name, season, home_team, away_team, status_short
                FROM api_football_fixtures
                WHERE substr(date, 1, 10) = ?
                ORDER BY date
                """,
                (target,),
            ).fetchall()
        insert_daily_job(
            conn,
            {
                "job_date": target,
                "competition": "all",
                "status": "fixtures_checked",
                "fixtures_found": len(rows),
                "predictions_created": 0,
                "results_updated": 0,
                "errors": [],
            },
        )
        conn.commit()
    typer.echo(json.dumps({"date": target, "fixtures": [dict(row) for row in rows]}, indent=2))


@app.command("predict-today")
def predict_today(
    date: str | None = typer.Option(None, help="Fecha YYYY-MM-DD. Por defecto usa hoy."),
    db_path: Path = settings.db_path,
    simulations: int = typer.Option(5000, min=1000, max=50000),
    configured_only: bool = typer.Option(True, help="Predecir solo competiciones configuradas."),
) -> None:
    from datetime import date as date_type

    target = date or date_type.today().isoformat()
    init_db(db_path)
    matches = _load_matches_from_db(db_path)
    team_metrics = _load_team_metrics(db_path)
    advanced_stats = _load_advanced_stats_from_db(db_path)
    with connect(db_path) as conn:
        if configured_only:
            fixtures = conn.execute(
                """
                SELECT f.date, f.league_name, f.home_team, f.away_team
                FROM api_football_fixtures f
                JOIN competitions c ON c.api_football_league_id = f.league_id AND c.enabled = 1
                WHERE substr(f.date, 1, 10) = ?
                  AND f.home_team IS NOT NULL
                  AND f.away_team IS NOT NULL
                  AND f.status_short NOT IN ('CANC', 'PST')
                ORDER BY c.priority ASC, f.date
                """,
                (target,),
            ).fetchall()
        else:
            fixtures = conn.execute(
                """
                SELECT date, league_name, home_team, away_team
                FROM api_football_fixtures
                WHERE substr(date, 1, 10) = ?
                  AND home_team IS NOT NULL
                  AND away_team IS NOT NULL
                  AND status_short NOT IN ('CANC', 'PST')
                ORDER BY date
                """,
                (target,),
            ).fetchall()
        existing_metrics = compute_backtest_metrics(list_prediction_backtests(conn))
        created = 0
        for fixture in fixtures:
            home = str(fixture["home_team"])
            away = str(fixture["away_team"])
            prediction = MatchPredictor(matches, team_metrics=team_metrics).predict(home, away).model_dump()
            match_context = get_match_context(conn, home, away, match_date=target)
            simulation = simulate_advanced_match(
                matches,
                team_metrics,
                home,
                away,
                simulations=simulations,
                match_context=match_context,
            )
            markets = apply_market_calibration(estimate_secondary_markets(advanced_stats, home, away), existing_metrics)
            simulated_probs = simulation.get("simulation", {})
            home_prob = simulated_probs.get("home_win", prediction["home_win_prob"])
            draw_prob = simulated_probs.get("draw", prediction["draw_prob"])
            away_prob = simulated_probs.get("away_win", prediction["away_win_prob"])
            probs = {"home": home_prob, "draw": draw_prob, "away": away_prob}
            upsert_prediction_backtest(
                conn,
                {
                    "match_date": target,
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
                    "notes": "Snapshot automatico local",
                },
            )
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
                "errors": [],
            },
        )
        conn.commit()
    typer.echo(json.dumps({"date": target, "fixtures": len(fixtures), "predictions_created": created}, indent=2))


@app.command("update-results-from-fixtures")
def update_results_from_fixtures(
    date: str | None = typer.Option(None, help="Fecha YYYY-MM-DD. Por defecto usa hoy."),
    db_path: Path = settings.db_path,
) -> None:
    from datetime import date as date_type

    target = date or date_type.today().isoformat()
    init_db(db_path)
    with connect(db_path) as conn:
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
    typer.echo(json.dumps(result, indent=2))


@app.command("rescue-pending-results")
def rescue_pending_results(
    sync_api: bool = typer.Option(True, help="Consultar API-Football para fixtures pendientes."),
    max_fixtures: int = typer.Option(100, min=1, max=500, help="Maximo de fixtures pendientes a procesar."),
    retry_hours: int = typer.Option(6, min=1, max=48, help="Horas antes de reintentar fixtures sin stats."),
    league: int | None = typer.Option(None, help="Opcional: limitar a una liga/competicion."),
    season: int | None = typer.Option(None, help="Opcional: limitar a temporada API-Football."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        result = run_pending_fixture_reconciliation(
            conn,
            client=ApiFootballClient() if sync_api else None,
            sync_api=sync_api,
            max_fixtures=max_fixtures,
            retry_hours=retry_hours,
            league=league,
            season=season,
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("show-teams")
def show_teams(db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        for team in list_teams(conn):
            typer.echo(team)


@app.command("data-quality")
def show_data_quality(db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        typer.echo(json.dumps(data_coverage(conn), indent=2))


@app.command("prepare-data-quality")
def prepare_data_quality(db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        for row in alias_rows():
            upsert_team_alias(conn, row)
        for row in requirement_rows():
            upsert_data_source_requirement(conn, row)
        conn.commit()
    typer.echo(f"Alias cargados: {len(alias_rows())}")
    typer.echo(f"Requisitos de fuentes cargados: {len(requirement_rows())}")


@app.command("data-quality-report")
def data_quality_report(db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        report = data_coverage(conn)
    compact = {
        "score": report["score"],
        "readiness": report.get("readiness", {}),
        "top_competitions": report.get("competition_coverage", [])[:10],
        "groups": report["groups"],
    }
    typer.echo(json.dumps(compact, indent=2))


@app.command("market-coverage")
def market_coverage_report(db_path: Path = settings.db_path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        report = data_coverage(conn).get("market_coverage", {})
    typer.echo(json.dumps(report, indent=2))


@app.command("sync-fifa-ranking")
def sync_fifa_ranking(db_path: Path = settings.db_path, count: int = 250) -> None:
    init_db(db_path)
    rankings = fetch_fifa_mens_rankings(count=count)
    with connect(db_path) as conn:
        known_teams = set(list_teams(conn))
        matched = 0
        for row in rankings:
            upsert_team_external_metrics(conn, row)
            if row["team"] in known_teams:
                matched += 1
        conn.commit()
    typer.echo(f"Rankings FIFA sincronizados: {len(rankings)}")
    typer.echo(f"Equipos del proyecto con match exacto: {matched}")


@app.command("sync-api-football-fixtures")
def sync_api_fixtures(
    league: int = typer.Option(..., help="ID de liga/competicion en API-Football."),
    season: int = typer.Option(..., help="Temporada en API-Football."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()
    with connect(db_path) as conn:
        result = sync_api_football_fixtures(conn, client, league=league, season=season)
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("sync-api-football-coverage")
def sync_api_coverage(
    league: list[int] | None = typer.Option(None, "--league", help="ID de liga/competicion. Puede repetirse."),
    season: int | None = typer.Option(None, help="Temporada opcional."),
    search: str | None = typer.Option(None, help="Buscar ligas por texto si no se pasa --league."),
    country: str | None = typer.Option(None, help="Filtrar por pais si no se pasa --league."),
    catalog: bool = typer.Option(True, help="Usar catalogo local de competiciones prioritarias si no se pasa --league/search/country."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()
    league_ids = league
    if not league_ids and catalog and not search and not country:
        league_ids = [int(item["api_football_league_id"]) for item in COMPETITIONS if item.get("api_football_league_id")]
    with connect(db_path) as conn:
        result = sync_api_football_league_coverage(
            conn,
            client,
            league_ids=league_ids,
            season=season,
            search=search,
            country=country,
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("sync-api-football-standings")
def sync_api_standings(
    league: int = typer.Option(..., help="ID de liga/competicion en API-Football."),
    season: int = typer.Option(..., help="Temporada."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()
    with connect(db_path) as conn:
        result = sync_api_football_standings(conn, client, league=league, season=season)
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("sync-api-football-team-stats")
def sync_api_team_stats(
    league: int = typer.Option(..., help="ID de liga/competicion en API-Football."),
    season: int = typer.Option(..., help="Temporada."),
    team: int | None = typer.Option(None, help="Opcional: ID de equipo."),
    limit: int | None = typer.Option(None, help="Opcional: limitar equipos para ahorrar requests."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()
    with connect(db_path) as conn:
        result = sync_api_football_team_statistics(
            conn,
            client,
            league=league,
            season=season,
            team_ids=[team] if team else None,
            limit=limit,
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("sync-api-football-players")
def sync_api_players(
    league: int = typer.Option(..., help="ID de liga/competicion en API-Football."),
    season: int = typer.Option(..., help="Temporada."),
    team: int | None = typer.Option(None, help="Opcional: ID de equipo."),
    max_requests: int = typer.Option(250, min=1, max=7500, help="Presupuesto maximo de requests para jugadores."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()
    with connect(db_path) as conn:
        result = sync_api_football_players(
            conn,
            client,
            league=league,
            season=season,
            team_ids=[team] if team else None,
            max_requests=max_requests,
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("backfill-match-context")
def backfill_match_context(
    league: int | None = typer.Option(None, help="Opcional: ID de liga/competicion."),
    season: int | None = typer.Option(None, help="Opcional: temporada."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        result = backfill_match_context_from_api_fixtures(conn, league_id=league, season=season)
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("backfill-venue-matching")
def backfill_venue_matching(
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        result = backfill_match_context_venues(conn)
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("sync-api-football-today")
def sync_api_today(
    date: str | None = typer.Option(None, help="Fecha YYYY-MM-DD. Por defecto usa hoy."),
    league: int | None = typer.Option(None, help="Opcional: limitar a una liga/competicion."),
    season: int | None = typer.Option(None, help="Opcional: temporada de API-Football."),
    max_finished_details: int = typer.Option(50, min=0, max=200, help="Maximo de fixtures terminados a enriquecer."),
    skip_finished_details: bool = typer.Option(False, help="Solo sincronizar fixtures/resultados, sin stats completas."),
    db_path: Path = settings.db_path,
) -> None:
    from datetime import date as date_type

    target = date or date_type.today().isoformat()
    init_db(db_path)
    client = ApiFootballClient()
    with connect(db_path) as conn:
        result = sync_api_football_fixtures_by_date(
            conn,
            client,
            date=target,
            league=league,
            season=season,
            sync_finished_details=not skip_finished_details,
            max_finished_details=max_finished_details,
        )
        reconciliation = reconcile_prediction_backtests_for_date(conn, target)
        result["reconciliation"] = reconciliation
        insert_daily_job(
            conn,
            {
                "job_date": target,
                "competition": str(league or "all"),
                "status": "api_football_synced",
                "fixtures_found": result["fixtures"],
                "predictions_created": 0,
                "results_updated": reconciliation["results_updated"],
                "errors": result.get("api_errors") or {},
            },
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("sync-inventory-status")
def sync_inventory_status(
    league: int = typer.Option(..., help="ID de liga/competicion en API-Football."),
    season: int = typer.Option(..., help="Temporada."),
    name: str | None = typer.Option(None, help="Nombre legible opcional."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        status = local_competition_season_status(conn, league_id=league, season=season, league_name=name)
        inventory = list_sync_inventory(conn, league_id=league, season=season)
    typer.echo(json.dumps({"status": status, "inventory": inventory}, indent=2))


@app.command("dedupe-api-football-matches")
def dedupe_api_football_matches(
    db_path: Path = settings.db_path,
    dry_run: bool = typer.Option(False, help="Solo contar duplicados, no borrar."),
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        duplicates = count_matches_shadowed_by_api_football(conn)
        deleted = 0
        if not dry_run:
            deleted = delete_matches_shadowed_by_api_football(conn)
            conn.commit()
    typer.echo(json.dumps({"duplicates": duplicates, "deleted": deleted, "dry_run": dry_run}, indent=2))


@app.command("sync-competition-season")
def sync_competition_season(
    league: int = typer.Option(..., help="ID de liga/competicion en API-Football."),
    season: int = typer.Option(..., help="Temporada."),
    name: str | None = typer.Option(None, help="Nombre legible opcional."),
    force: bool = typer.Option(False, help="Forzar descarga aunque el inventario diga completo."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()
    with connect(db_path) as conn:
        result = sync_competition_season_core(
            conn,
            client,
            league_id=league,
            season=season,
            league_name=name,
            missing_only=not force,
        )
        insert_daily_job(
            conn,
            {
                "job_date": str(season),
                "competition": name or str(league),
                "status": "competition_season_synced",
                "fixtures_found": result["after"]["fixtures"]["records_count"],
                "predictions_created": 0,
                "results_updated": result["after"]["finished_results"]["records_count"],
                "errors": [action.get("api_errors") for action in result["actions"] if action.get("api_errors")],
            },
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("search-api-football-leagues")
def search_api_leagues(
    search: str | None = typer.Option(None, help="Texto a buscar, por ejemplo World Cup."),
    country: str | None = typer.Option(None, help="Pais, si aplica."),
    season: int | None = typer.Option(None, help="Temporada, si aplica."),
    limit: int = typer.Option(20, min=1, max=100),
) -> None:
    client = ApiFootballClient()
    payload = client.leagues(search=search, country=country, season=season)
    rows = []
    for item in (payload.get("response") or [])[:limit]:
        league = item.get("league") or {}
        country_data = item.get("country") or {}
        seasons = item.get("seasons") or []
        rows.append(
            {
                "league_id": league.get("id"),
                "name": league.get("name"),
                "type": league.get("type"),
                "country": country_data.get("name"),
                "seasons": [season_row.get("year") for season_row in seasons[-5:]],
            }
        )
    typer.echo(
        json.dumps(
            {
                "results": payload.get("results"),
                "errors": payload.get("errors") or {},
                "leagues": rows,
            },
            indent=2,
        )
    )


@app.command("sync-api-football-fixture-details")
def sync_api_fixture_details(
    fixture_id: int = typer.Option(..., help="ID del fixture en API-Football."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()
    with connect(db_path) as conn:
        result = sync_api_football_fixture_details_by_id(conn, client, fixture_id)
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("sync-api-football-fixture-details-bulk")
def sync_api_fixture_details_bulk(
    league: int | None = typer.Option(None, help="Opcional: ID de liga/competicion."),
    season: int | None = typer.Option(None, help="Opcional: temporada."),
    max_fixtures: int = typer.Option(100, min=1, max=2000, help="Maximo de fixtures a enriquecer."),
    include_unfinished: bool = typer.Option(False, help="Tambien intentar partidos no finalizados."),
    force: bool = typer.Option(False, help="Reintentar aunque ya existan estadisticas."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()
    with connect(db_path) as conn:
        result = sync_api_football_fixture_details_bulk(
            conn,
            client,
            league=league,
            season=season,
            only_finished=not include_unfinished,
            missing_only=not force,
            max_fixtures=max_fixtures,
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("sync-api-football-massive")
def sync_api_football_massive(
    request_budget: int = typer.Option(1500, min=1, max=7500, help="Presupuesto estimado de requests."),
    seasons_per_league: int = typer.Option(2, min=1, max=5, help="Temporadas por liga segun API-Football."),
    max_competitions: int | None = typer.Option(None, min=1, help="Limitar numero de competiciones prioritarias."),
    league_id: list[int] | None = typer.Option(None, "--league-id", help="Procesar solo estos IDs de liga API-Football."),
    max_fixture_details_per_season: int = typer.Option(80, min=0, max=500, help="Maximo de fixtures caros por liga/temporada."),
    player_request_share: float = typer.Option(0.35, min=0.0, max=1.0, help="Fraccion del presupuesto de cada temporada para jugadores."),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    client = ApiFootballClient()

    def progress(event: dict) -> None:
        typer.echo(
            "[api-football] "
            f"{event.get('league_name') or event.get('league_id')} "
            f"{event.get('season') or ''} "
            f"{event['endpoint']}={event['status']} "
            f"used={event.get('requests_used')} "
            f"remaining={event.get('remaining_budget')}"
        )

    with connect(db_path) as conn:
        result = run_mass_api_football_sync(
            conn,
            client,
            request_budget=request_budget,
            seasons_per_league=seasons_per_league,
            max_competitions=max_competitions,
            league_ids=league_id,
            max_fixture_details_per_season=max_fixture_details_per_season,
            player_request_share=player_request_share,
            progress=progress,
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("predict")
def predict(
    home: str = typer.Option(...),
    away: str = typer.Option(...),
    db_path: Path = settings.db_path,
    save: bool = typer.Option(False, help="Guardar la prediccion en SQLite."),
) -> None:
    init_db(db_path)
    df = _load_matches_from_db(db_path)
    prediction = MatchPredictor(df, team_metrics=_load_team_metrics(db_path)).predict(home, away)
    if save:
        with connect(db_path) as conn:
            prediction_id = insert_prediction(conn, prediction.model_dump())
            conn.commit()
        typer.echo(f"Prediccion guardada con id: {prediction_id}")
    typer.echo(prediction.model_dump_json(indent=2))


@app.command("player-markets")
def player_markets(
    home: str = typer.Option(...),
    away: str = typer.Option(...),
    db_path: Path = settings.db_path,
) -> None:
    init_db(db_path)
    advanced_stats = _load_advanced_stats_from_db(db_path)
    markets = estimate_secondary_markets(advanced_stats, home, away)
    with connect(db_path) as conn:
        result = estimate_player_markets(conn, home, away, markets)
    typer.echo(json.dumps(result, indent=2))


@app.command("simulate")
def simulate(
    home: str = typer.Option(...),
    away: str = typer.Option(...),
    simulations: int = typer.Option(10000, min=1000, max=200000),
    mode: str = typer.Option("hybrid", help="classic, hybrid o poker."),
    db_path: Path = settings.db_path,
) -> None:
    df = _load_matches_from_db(db_path)
    metrics = _load_team_metrics(db_path)
    if mode == "classic":
        predictor = MatchPredictor(df, team_metrics=metrics)
        result = simulate_match(predictor, home, away, simulations=simulations)
    else:
        with connect(db_path) as conn:
            match_context = get_match_context(conn, home, away)
        result = simulate_advanced_match(df, metrics, home, away, simulations=simulations, mode=mode, match_context=match_context)
    with connect(db_path) as conn:
        insert_experiment(
            conn,
            name=f"{home} vs {away}",
            kind="simulation",
            config={"home": home, "away": away, "simulations": simulations, "mode": mode},
            metrics=result.get("simulation", {}),
        )
        conn.commit()
    typer.echo(json.dumps(result, indent=2))


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    db_path: Path = settings.db_path,
) -> None:
    import uvicorn

    from football_predictor.web.app import create_app

    typer.echo(f"Interfaz disponible en http://{host}:{port}")
    uvicorn.run(create_app(db_path), host=host, port=port)


if __name__ == "__main__":
    app()
