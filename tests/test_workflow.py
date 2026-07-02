from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner
from fastapi.testclient import TestClient

from football_predictor.cli import app
from football_predictor.database.db import (
    connect,
    init_db,
    insert_matches,
    insert_prediction,
    upsert_team_alias,
    upsert_advanced_stats,
    upsert_api_football_fixture,
    upsert_prediction_backtest,
)
from football_predictor.data.coverage import data_coverage
from football_predictor.evaluation.backtest import run_backtest
from football_predictor.models.ml_model import MatchOutcomeModel
from football_predictor.prediction.predictor import MatchPredictor
from football_predictor.web.app import create_app


runner = CliRunner()


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2024-01-01", "A", "B", 2, 1, "League", "2024"],
            ["2024-01-08", "C", "D", 0, 1, "League", "2024"],
            ["2024-01-15", "A", "C", 1, 1, "League", "2024"],
            ["2024-01-22", "B", "D", 3, 0, "League", "2024"],
            ["2024-01-29", "D", "A", 0, 2, "League", "2024"],
            ["2024-02-05", "C", "B", 2, 0, "League", "2024"],
            ["2024-02-12", "A", "D", 2, 2, "League", "2024"],
            ["2024-02-19", "B", "C", 1, 2, "League", "2024"],
        ],
        columns=["date", "home_team", "away_team", "home_goals", "away_goals", "competition", "season"],
    )


def write_csv(path: Path) -> None:
    sample_matches().to_csv(path, index=False)


def test_cli_load_predict_save_and_show_teams(tmp_path: Path) -> None:
    db_path = tmp_path / "football.sqlite"
    csv_path = tmp_path / "matches.csv"
    write_csv(csv_path)

    assert runner.invoke(app, ["init-db", "--db-path", str(db_path)]).exit_code == 0
    assert runner.invoke(app, ["load-csv", str(csv_path), "--db-path", str(db_path)]).exit_code == 0

    teams = runner.invoke(app, ["show-teams", "--db-path", str(db_path)])
    assert teams.exit_code == 0
    assert "A" in teams.stdout

    prediction = runner.invoke(app, ["predict", "--home", "A", "--away", "B", "--db-path", str(db_path), "--save"])
    assert prediction.exit_code == 0
    assert "Prediccion guardada" in prediction.stdout


def test_prediction_persistence_directly(tmp_path: Path) -> None:
    db_path = tmp_path / "football.sqlite"
    init_db(db_path)
    pred = MatchPredictor(sample_matches()).predict("A", "B")
    with connect(db_path) as conn:
        prediction_id = insert_prediction(conn, pred.model_dump())
        count = conn.execute("SELECT COUNT(*) AS total FROM predictions").fetchone()["total"]
    assert prediction_id == 1
    assert count == 1


def test_backtest_returns_metrics() -> None:
    results, metrics = run_backtest(sample_matches(), min_train_matches=3)
    assert len(results) == 5
    assert 0 <= metrics["accuracy"] <= 1
    assert "log_loss" in metrics
    assert "brier" in metrics


def test_ml_model_trains_and_predicts() -> None:
    model = MatchOutcomeModel().fit(sample_matches())
    probs = model.predict_proba("A", "B")
    assert abs(sum(probs.values()) - 1.0) < 0.001


def test_data_coverage_reports_groups(tmp_path: Path) -> None:
    db_path = tmp_path / "football.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        coverage = data_coverage(conn)
    assert "core_results" in coverage["groups"]
    assert 0 <= coverage["score"] <= 1


def test_data_explorer_filters_fixtures(tmp_path: Path) -> None:
    db_path = tmp_path / "football.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        insert_matches(conn, sample_matches().to_dict("records"))
        upsert_api_football_fixture(
            conn,
            {
                "fixture_id": 99,
                "date": "2026-06-29T20:00:00+00:00",
                "league_id": 1,
                "league_name": "World Cup",
                "season": 2026,
                "home_team": "Brazil",
                "away_team": "Japan",
                "home_team_id": 1,
                "away_team_id": 2,
                "status_short": "NS",
                "venue_name": "Test Stadium",
                "venue_city": "Test City",
                "raw": {},
            },
        )
        conn.commit()

    client = TestClient(create_app(db_path))
    response = client.get("/api/data-explorer/fixtures", params={"search": "Brazil", "status": "upcoming"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["totals"]["api_fixtures"] == 1
    assert payload["fixtures"][0]["home_team"] == "Brazil"
    assert payload["fixtures"][0]["status_group"] == "upcoming"


def test_daily_prediction_cycle_from_local_fixtures(tmp_path: Path) -> None:
    db_path = tmp_path / "daily.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        insert_matches(conn, sample_matches().to_dict("records"))
        upsert_api_football_fixture(
            conn,
            {
                "fixture_id": 101,
                "date": "2026-06-29T18:00:00+00:00",
                "league_id": 1,
                "league_name": "World Cup",
                "season": 2026,
                "home_team": "A",
                "away_team": "B",
                "home_team_id": 1,
                "away_team_id": 2,
                "status_short": "NS",
                "venue_name": "Test Stadium",
                "venue_city": "Test City",
                "raw": {"goals": {"home": None, "away": None}},
            },
        )
        conn.commit()

    client = TestClient(create_app(db_path))
    predicted = client.post(
        "/api/operations/predict-today",
        json={"date": "2026-06-29", "simulations": 1000, "configured_only": True},
    )
    assert predicted.status_code == 200
    assert predicted.json()["predictions_created"] == 1

    history = client.get("/api/prediction-history", params={"limit": 10}).json()
    assert history["counts"]["pending"] == 1
    assert history["counts"]["limbo"] == 1
    detail = client.get(f"/api/prediction-history/{history['rows'][0]['id']}").json()
    assert detail["home_team"] == "A"
    assert detail["status"] == "pending"
    assert detail["lifecycle_status"] == "result_late"
    assert "top_scores" in detail
    assert "market_distribution" in detail["snapshot"]
    assert "corners" in detail["snapshot"]["market_distribution"]
    assert "pick_report" in detail["snapshot"]
    assert "discarded" in detail["snapshot"]["pick_report"]
    assert all("tier" in pick for pick in detail["snapshot"]["betting_picks"])

    with connect(db_path) as conn:
        upsert_api_football_fixture(
            conn,
            {
                "fixture_id": 101,
                "date": "2026-06-29T18:00:00+00:00",
                "league_id": 1,
                "league_name": "World Cup",
                "season": 2026,
                "home_team": "A",
                "away_team": "B",
                "home_team_id": 1,
                "away_team_id": 2,
                "status_short": "FT",
                "venue_name": "Test Stadium",
                "venue_city": "Test City",
                "raw": {"goals": {"home": 2, "away": 1}},
            },
        )
        conn.commit()

    updated = client.post("/api/operations/update-results", json={"date": "2026-06-29"})
    assert updated.status_code == 200
    assert updated.json()["results_updated"] == 1
    assert updated.json()["metrics"]["matches"] == 1
    history_after = client.get("/api/prediction-history", params={"limit": 10}).json()
    assert history_after["rows"][0]["competition"] == "World Cup 2026"
    assert history_after["rows"][0]["lifecycle_status"] == "score_updated"
    assert history_after["counts"]["score_updated"] == 1


def test_reconcile_pending_results_closes_stale_snapshot_from_local_fixture(tmp_path: Path) -> None:
    db_path = tmp_path / "reconcile.sqlite"
    init_db(db_path)
    match_date = (date.today() - timedelta(days=1)).isoformat()
    with connect(db_path) as conn:
        upsert_prediction_backtest(
            conn,
            {
                "match_date": match_date,
                "competition": "World Cup",
                "home_team": "A",
                "away_team": "B",
                "model_version": "test_snapshot",
                "predicted_home_prob": 0.55,
                "predicted_draw_prob": 0.25,
                "predicted_away_prob": 0.20,
                "predicted_pick": "home",
                "predicted_scores_json": '[{"score": "2-1", "probability": 0.2}]',
                "predicted_corners": 9.0,
                "predicted_shots_on_target": 8.0,
                "predicted_cards": 3.5,
                "predicted_over_2_5_prob": 0.55,
                "predicted_btts_prob": 0.50,
                "actual_home_goals": None,
                "actual_away_goals": None,
                "actual_corners": None,
                "actual_shots_on_target": None,
                "actual_cards": None,
                "source": "test",
                "notes": None,
                "snapshot": {},
            },
        )
        upsert_api_football_fixture(
            conn,
            {
                "fixture_id": 202,
                "date": f"{match_date}T18:00:00+00:00",
                "league_id": 1,
                "league_name": "World Cup",
                "season": 2026,
                "home_team": "A",
                "away_team": "B",
                "home_team_id": 1,
                "away_team_id": 2,
                "status_short": "FT",
                "venue_name": "Test Stadium",
                "venue_city": "Test City",
                "raw": {"goals": {"home": 3, "away": 1}},
            },
        )
        for team, opponent, corners, shots, cards in [("A", "B", 6, 5, 2), ("B", "A", 4, 3, 1)]:
            upsert_advanced_stats(
                conn,
                {
                    "source_match_id": "api-football:202",
                    "team": team,
                    "opponent": opponent,
                    "date": match_date,
                    "xg": 1.0,
                    "possession_pct": 50,
                    "total_shots": 10,
                    "shots_on_target": shots,
                    "shots_off_target": 3,
                    "blocked_shots": 2,
                    "corners": corners,
                    "fouls": 12,
                    "offsides": 1,
                    "saves": 2,
                    "yellow_cards": cards,
                    "red_cards": 0,
                    "cards_estimate": cards,
                    "passes_total": 400,
                    "passes_accurate": 340,
                    "pass_accuracy_pct": 85,
                    "attacks": 90,
                    "dangerous_attacks": 40,
                    "raw": {},
                },
            )
        conn.commit()

    client = TestClient(create_app(db_path))
    response = client.post("/api/operations/reconcile-pending-results", json={"max_dates": 7, "sync_api": False})
    payload = response.json()

    assert response.status_code == 200
    assert payload["fixtures_pending_found"] == 1
    assert payload["snapshots_updated"] == 1
    assert payload["pending_before"] == 1
    assert payload["pending_after"] == 0
    assert payload["fixtures_completed"] == 1
    assert payload["corners_updated"] == 1
    assert payload["shots_on_target_updated"] == 1
    assert payload["cards_updated"] == 1

    detail = client.get("/api/prediction-history", params={"limit": 1}).json()["rows"][0]
    assert detail["actual_home_goals"] == 3
    assert detail["actual_away_goals"] == 1
    assert detail["actual_corners"] == 10
    assert detail["actual_shots_on_target"] == 8
    assert detail["actual_cards"] == 3

    with connect(db_path) as conn:
        cache = conn.execute("SELECT stats_completed, snapshots_updated FROM fixture_detail_sync_status WHERE fixture_id = 202").fetchone()
    assert dict(cache) == {"stats_completed": 1, "snapshots_updated": 1}


def test_update_results_reconciles_stale_market_stats_from_api_fixture(tmp_path: Path) -> None:
    db_path = tmp_path / "reconcile_markets.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        upsert_prediction_backtest(
            conn,
            {
                "match_date": "2026-06-29",
                "competition": "World Cup",
                "home_team": "Brazil",
                "away_team": "Japan",
                "model_version": "test_snapshot",
                "predicted_home_prob": 0.60,
                "predicted_draw_prob": 0.25,
                "predicted_away_prob": 0.15,
                "predicted_pick": "home",
                "predicted_scores_json": '[{"score": "2-1", "probability": 0.2}]',
                "predicted_corners": 6.2,
                "predicted_shots_on_target": 9.0,
                "predicted_cards": 2.3,
                "predicted_over_2_5_prob": 0.55,
                "predicted_btts_prob": 0.50,
                "actual_home_goals": 2,
                "actual_away_goals": 1,
                "actual_corners": 4,
                "actual_shots_on_target": 7,
                "actual_cards": 4,
                "source": "test",
                "notes": None,
                "snapshot": {},
            },
        )
        upsert_api_football_fixture(
            conn,
            {
                "fixture_id": 1562344,
                "date": "2026-06-29T17:00:00+00:00",
                "league_id": 1,
                "league_name": "World Cup",
                "season": 2026,
                "home_team": "Brazil",
                "away_team": "Japan",
                "home_team_id": 6,
                "away_team_id": 12,
                "status_short": "FT",
                "venue_name": "Test Stadium",
                "venue_city": "Test City",
                "raw": {"goals": {"home": 2, "away": 1}},
            },
        )
        for team, opponent, corners, shots, yellow in [("Brazil", "Japan", 6, 7, 2), ("Japan", "Brazil", 2, 2, 3)]:
            upsert_advanced_stats(
                conn,
                {
                    "source_match_id": "api-football:1562344",
                    "team": team,
                    "opponent": opponent,
                    "date": "2026-06-29",
                    "xg": 1.0,
                    "possession_pct": 50,
                    "total_shots": 10,
                    "shots_on_target": shots,
                    "shots_off_target": 3,
                    "blocked_shots": 2,
                    "corners": corners,
                    "fouls": 10,
                    "offsides": 1,
                    "saves": 2,
                    "yellow_cards": yellow,
                    "red_cards": None,
                    "cards_estimate": yellow,
                    "passes_total": 400,
                    "passes_accurate": 340,
                    "pass_accuracy_pct": 85,
                    "attacks": 90,
                    "dangerous_attacks": 40,
                    "raw": {},
                },
            )
        conn.commit()

    client = TestClient(create_app(db_path))
    response = client.post("/api/operations/update-results", json={"date": "2026-06-29"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["results_updated"] == 1

    detail = client.get("/api/prediction-history", params={"limit": 1}).json()["rows"][0]
    assert detail["actual_home_goals"] == 2
    assert detail["actual_away_goals"] == 1
    assert detail["actual_corners"] == 8
    assert detail["actual_shots_on_target"] == 9
    assert detail["actual_cards"] == 5


def test_reconcile_pending_results_uses_team_aliases_for_fixture_matching(tmp_path: Path) -> None:
    db_path = tmp_path / "alias_reconcile.sqlite"
    init_db(db_path)
    match_date = (date.today() - timedelta(days=1)).isoformat()
    with connect(db_path) as conn:
        upsert_team_alias(conn, {"canonical_name": "DR Congo", "alias": "DR Congo", "source": "test", "confidence": 1.0})
        upsert_team_alias(conn, {"canonical_name": "DR Congo", "alias": "Congo DR", "source": "test", "confidence": 1.0})
        upsert_prediction_backtest(
            conn,
            {
                "match_date": match_date,
                "competition": "World Cup",
                "home_team": "DR Congo",
                "away_team": "Uzbekistan",
                "model_version": "alias_snapshot",
                "predicted_home_prob": 0.55,
                "predicted_draw_prob": 0.25,
                "predicted_away_prob": 0.20,
                "predicted_pick": "home",
                "predicted_scores_json": "[]",
                "predicted_corners": 7.0,
                "predicted_shots_on_target": 5.0,
                "predicted_cards": 4.0,
                "predicted_over_2_5_prob": 0.50,
                "predicted_btts_prob": 0.50,
                "actual_home_goals": None,
                "actual_away_goals": None,
                "actual_corners": None,
                "actual_shots_on_target": None,
                "actual_cards": None,
                "source": "test",
                "notes": None,
                "snapshot": {},
            },
        )
        upsert_api_football_fixture(
            conn,
            {
                "fixture_id": 303,
                "date": f"{match_date}T18:00:00+00:00",
                "league_id": 1,
                "league_name": "World Cup",
                "season": 2026,
                "home_team": "Congo DR",
                "away_team": "Uzbekistan",
                "home_team_id": 1,
                "away_team_id": 2,
                "status_short": "FT",
                "venue_name": "Test Stadium",
                "venue_city": "Test City",
                "raw": {"goals": {"home": 3, "away": 1}},
            },
        )
        for team, opponent, corners, shots, cards in [("Congo DR", "Uzbekistan", 4, 3, 2), ("Uzbekistan", "Congo DR", 2, 2, 3)]:
            upsert_advanced_stats(
                conn,
                {
                    "source_match_id": "api-football:303",
                    "team": team,
                    "opponent": opponent,
                    "date": match_date,
                    "xg": 1.0,
                    "possession_pct": 50,
                    "total_shots": 10,
                    "shots_on_target": shots,
                    "shots_off_target": 3,
                    "blocked_shots": 2,
                    "corners": corners,
                    "fouls": 11,
                    "offsides": 1,
                    "saves": 2,
                    "yellow_cards": cards,
                    "red_cards": 0,
                    "cards_estimate": cards,
                    "passes_total": 400,
                    "passes_accurate": 340,
                    "pass_accuracy_pct": 85,
                    "attacks": 90,
                    "dangerous_attacks": 40,
                    "raw": {},
                },
            )
        conn.commit()

    client = TestClient(create_app(db_path))
    payload = client.post("/api/operations/reconcile-pending-results", json={"sync_api": False}).json()

    assert payload["fixtures_pending_found"] == 1
    assert payload["snapshots_updated"] == 1
    detail = client.get("/api/prediction-history", params={"limit": 1}).json()["rows"][0]
    assert detail["actual_home_goals"] == 3
    assert detail["actual_away_goals"] == 1
    assert detail["actual_corners"] == 6
    assert detail["actual_shots_on_target"] == 5
    assert detail["actual_cards"] == 5
