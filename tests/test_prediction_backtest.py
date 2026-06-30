from pathlib import Path

from football_predictor.database.db import (
    connect,
    delete_pending_prediction_backtest,
    init_db,
    update_prediction_backtest_result,
    upsert_prediction_backtest,
)
from football_predictor.evaluation.prediction_backtest import compute_backtest_metrics, load_backtest_file
from football_predictor.prediction.calibration import apply_market_calibration


def test_prediction_backtest_metrics_detect_hits_and_biases() -> None:
    rows = [
        {
            "match_date": "2026-06-27",
            "competition": "World Cup 2026",
            "home_team": "A",
            "away_team": "B",
            "model_version": "test",
            "predicted_home_prob": 0.1,
            "predicted_draw_prob": 0.2,
            "predicted_away_prob": 0.7,
            "predicted_pick": "away",
            "predicted_scores_json": '[{"score": "0-1"}, {"score": "0-2"}]',
            "predicted_corners": 9.0,
            "predicted_shots_on_target": 8.0,
            "predicted_cards": 4.5,
            "predicted_over_2_5_prob": 0.4,
            "predicted_btts_prob": 0.3,
            "actual_home_goals": 0,
            "actual_away_goals": 2,
            "actual_corners": 10,
            "actual_shots_on_target": 8,
            "actual_cards": 2,
        }
    ]

    metrics = compute_backtest_metrics(rows)

    assert metrics["strict_accuracy"] == 1.0
    assert metrics["exact_score_top2_accuracy"] == 1.0
    assert metrics["market_metrics"]["cards"]["direction"] == "sobreestima"
    assert metrics["log_loss"] is not None
    assert metrics["brier"] is not None


def test_load_backtest_file_normalizes_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "backtest.csv"
    csv_path.write_text(
        "\n".join(
            [
                "match_date,competition,home_team,away_team,model_version,predicted_home_prob,predicted_draw_prob,predicted_away_prob,predicted_pick,predicted_scores,predicted_corners,predicted_shots_on_target,predicted_cards,predicted_over_2_5_prob,predicted_btts_prob,actual_home_goals,actual_away_goals,actual_corners,actual_shots_on_target,actual_cards,source,notes",
                "2026-06-27,World Cup,A,B,test,60,20,20,home,1-0:22|2-0:12,7,5,3,55,40,1,0,8,4,2,manual,ok",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_backtest_file(csv_path)

    assert rows[0]["predicted_home_prob"] == 0.6
    assert rows[0]["predicted_scores_json"].startswith("[")
    assert rows[0]["predicted_pick"] == "home"


def test_market_calibration_reduces_positive_bias() -> None:
    markets = {"total_cards_expected": 4.8, "total_corners_expected": 9.0}
    metrics = {
        "market_metrics": {
            "cards": {"bias": 1.8},
            "corners": {"bias": 0.5},
        }
    }

    calibrated = apply_market_calibration(markets, metrics)

    assert calibrated["total_cards_expected"] == 3.0
    assert calibrated["total_corners_expected"] == 8.5
    assert calibrated["calibration"]["applied"] is True


def test_delete_pending_prediction_backtest_keeps_evaluated_history(tmp_path: Path) -> None:
    db_path = tmp_path / "tracking.sqlite"
    init_db(db_path)
    row = {
        "match_date": "2026-06-28",
        "competition": "Manual",
        "home_team": "Mexico",
        "away_team": "France",
        "model_version": "test",
        "predicted_home_prob": 0.2,
        "predicted_draw_prob": 0.25,
        "predicted_away_prob": 0.55,
        "predicted_pick": "away",
        "predicted_scores_json": "[]",
        "predicted_corners": 8.0,
        "predicted_shots_on_target": 7.0,
        "predicted_cards": 3.0,
        "predicted_over_2_5_prob": 0.5,
        "predicted_btts_prob": 0.45,
        "actual_home_goals": None,
        "actual_away_goals": None,
        "actual_corners": None,
        "actual_shots_on_target": None,
        "actual_cards": None,
        "source": "test",
        "notes": "",
    }

    with connect(db_path) as conn:
        upsert_prediction_backtest(conn, row)
        pending_id = conn.execute("SELECT id FROM prediction_backtests").fetchone()[0]
        assert delete_pending_prediction_backtest(conn, pending_id) == 1
        upsert_prediction_backtest(conn, row)
        evaluated_id = conn.execute("SELECT id FROM prediction_backtests").fetchone()[0]
        update_prediction_backtest_result(
            conn,
            evaluated_id,
            {"actual_home_goals": 0, "actual_away_goals": 2, "actual_corners": None, "actual_shots_on_target": None, "actual_cards": None, "notes": None},
        )
        assert delete_pending_prediction_backtest(conn, evaluated_id) == 0
