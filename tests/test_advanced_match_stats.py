import pandas as pd

from football_predictor.database.db import advanced_summary_by_team, connect, init_db, upsert_advanced_stats
from football_predictor.prediction.markets import estimate_secondary_markets


def test_advanced_stats_store_extra_api_fields_and_summarize_against(tmp_path) -> None:
    db_path = tmp_path / "advanced.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        upsert_advanced_stats(
            conn,
            {
                "source_match_id": "api-football:1",
                "team": "Mexico",
                "opponent": "France",
                "date": "2026-06-11",
                "xg": 1.2,
                "possession_pct": 54,
                "total_shots": 12,
                "shots_on_target": 5,
                "shots_off_target": 4,
                "blocked_shots": 3,
                "corners": 6,
                "fouls": 11,
                "offsides": 1,
                "saves": 3,
                "yellow_cards": 2,
                "red_cards": 0,
                "cards_estimate": 2,
                "passes_total": 455,
                "passes_accurate": 390,
                "pass_accuracy_pct": 86,
                "attacks": 100,
                "dangerous_attacks": 52,
                "raw": {},
            },
        )
        upsert_advanced_stats(
            conn,
            {
                "source_match_id": "api-football:1",
                "team": "France",
                "opponent": "Mexico",
                "date": "2026-06-11",
                "xg": 1.7,
                "possession_pct": 46,
                "total_shots": 14,
                "shots_on_target": 6,
                "shots_off_target": 5,
                "blocked_shots": 3,
                "corners": 4,
                "fouls": 13,
                "offsides": 2,
                "saves": 4,
                "yellow_cards": 1,
                "red_cards": 0,
                "cards_estimate": 1,
                "passes_total": 410,
                "passes_accurate": 340,
                "pass_accuracy_pct": 83,
                "attacks": 94,
                "dangerous_attacks": 47,
                "raw": {},
            },
        )

        summary = advanced_summary_by_team(conn)

    assert summary["Mexico"]["corners_for"] == 6
    assert summary["Mexico"]["corners_against"] == 4
    assert summary["Mexico"]["shots_on_target_against"] == 6
    assert summary["Mexico"]["pass_accuracy_for"] == 86


def test_secondary_markets_use_recent_stats_and_opponent_allowed() -> None:
    stats = pd.DataFrame(
        [
            {"date": "2026-06-01", "team": "Mexico", "opponent": "France", "corners": 4, "total_shots": 9, "shots_on_target": 3, "cards_estimate": 2, "fouls": 10, "xg": 1.0, "possession_pct": 48},
            {"date": "2026-06-02", "team": "France", "opponent": "Mexico", "corners": 8, "total_shots": 16, "shots_on_target": 7, "cards_estimate": 1, "fouls": 9, "xg": 2.2, "possession_pct": 57},
            {"date": "2026-06-10", "team": "Mexico", "opponent": "Brazil", "corners": 7, "total_shots": 15, "shots_on_target": 6, "cards_estimate": 3, "fouls": 14, "xg": 1.6, "possession_pct": 52},
            {"date": "2026-06-10", "team": "Brazil", "opponent": "Mexico", "corners": 5, "total_shots": 11, "shots_on_target": 4, "cards_estimate": 2, "fouls": 12, "xg": 1.1, "possession_pct": 48},
        ]
    )

    markets = estimate_secondary_markets(stats, "Mexico", "France")

    assert markets["home_corners_expected"] > 5
    assert markets["shots_on_target_expected"] > 8
    assert markets["advanced_samples"]["home"] == 2
