from pathlib import Path

from football_predictor.data.coverage import data_coverage
from football_predictor.data.quality_plan import alias_rows, requirement_rows
from football_predictor.database.db import (
    connect,
    init_db,
    upsert_advanced_stats,
    upsert_competition,
    upsert_data_source_requirement,
    upsert_api_football_fixture,
    upsert_team_alias,
)


def test_data_quality_includes_readiness_and_competition_coverage(tmp_path: Path) -> None:
    db_path = tmp_path / "quality.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        for row in alias_rows()[:3]:
            upsert_team_alias(conn, row)
        for row in requirement_rows()[:3]:
            upsert_data_source_requirement(conn, row)
        upsert_competition(
            conn,
            {
                "name": "World Cup",
                "region": "Selecciones",
                "country": "World",
                "api_football_league_id": 1,
                "priority": 1,
                "status": "partial",
            },
            season=2026,
        )
        conn.commit()
        report = data_coverage(conn)

    assert report["readiness"]["total"] == 3
    assert report["groups"]["name_mapping"]["available"] is True
    assert report["competition_coverage"][0]["name"] == "World Cup"


def test_data_quality_includes_market_coverage(tmp_path: Path) -> None:
    db_path = tmp_path / "quality.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        upsert_api_football_fixture(
            conn,
            {
                "fixture_id": 1,
                "date": "2026-06-11T20:00:00+00:00",
                "league_id": 1,
                "league_name": "World Cup",
                "season": 2026,
                "home_team": "Mexico",
                "away_team": "France",
                "home_team_id": 10,
                "away_team_id": 20,
                "status_short": "FT",
                "venue_name": "Estadio Azteca",
                "venue_city": "Mexico City",
                "raw": {},
            },
        )
        for team, corners in [("Mexico", 6), ("France", 2)]:
            upsert_advanced_stats(
                conn,
                {
                    "source_match_id": "api-football:1",
                    "team": team,
                    "opponent": "France" if team == "Mexico" else "Mexico",
                    "date": "2026-06-11",
                    "xg": 1.2,
                    "possession_pct": 50,
                    "total_shots": 10,
                    "shots_on_target": 4,
                    "shots_off_target": None,
                    "blocked_shots": None,
                    "corners": corners,
                    "fouls": 12,
                    "offsides": None,
                    "saves": None,
                    "yellow_cards": 2,
                    "red_cards": 0,
                    "cards_estimate": 2,
                    "passes_total": None,
                    "passes_accurate": None,
                    "pass_accuracy_pct": None,
                    "attacks": None,
                    "dangerous_attacks": None,
                    "raw": {},
                },
            )
        conn.commit()
        report = data_coverage(conn)

    market_report = report["market_coverage"]
    assert market_report["matches_with_any_stats"] == 1
    assert market_report["markets"]["corners"]["sample_matches"] == 1
    assert market_report["markets"]["cards"]["strength"] == "weak"
    assert market_report["by_competition"][0]["competition"] == "World Cup 2026"
