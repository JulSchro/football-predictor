from pathlib import Path

from football_predictor.data.coverage import data_coverage
from football_predictor.data.quality_plan import alias_rows, requirement_rows
from football_predictor.database.db import (
    connect,
    init_db,
    upsert_competition,
    upsert_data_source_requirement,
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
