from pathlib import Path

from football_predictor.database.db import (
    connect,
    init_db,
    upsert_api_football_fixture_player,
    upsert_player,
    upsert_player_availability,
    upsert_player_form_snapshot,
    upsert_player_match_stats,
    upsert_player_season_stats,
    upsert_team_squad,
    player_quality_summary_by_team,
)


def test_player_tables_are_created(tmp_path: Path) -> None:
    db_path = tmp_path / "players.sqlite"
    init_db(db_path)

    expected = {
        "players",
        "team_squads",
        "player_season_stats",
        "player_match_stats",
        "lineups",
        "player_form_snapshots",
        "player_availability",
    }
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ({})
            """.format(",".join("?" for _ in expected)),
            tuple(expected),
        ).fetchall()

    assert {row["name"] for row in rows} == expected


def test_player_core_upserts_store_squad_stats_availability_and_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "player_core.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        player_id = upsert_player(
            conn,
            {
                "api_player_id": 10,
                "name": "Starter A",
                "birth_date": "2000-01-01",
                "nationality": "Mexico",
                "preferred_position": "F",
            },
        )
        upsert_team_squad(
            conn,
            {
                "team_name": "Mexico",
                "api_team_id": 100,
                "player_id": player_id,
                "competition_id": 1,
                "competition_name": "World Cup",
                "season": 2026,
                "squad_number": 9,
                "position": "F",
                "is_active": 1,
                "joined_at": None,
                "left_at": None,
            },
        )
        upsert_player_season_stats(
            conn,
            {
                "player_id": player_id,
                "team_id": None,
                "api_team_id": 100,
                "team_name": "Mexico",
                "competition_id": 1,
                "competition_name": "World Cup",
                "season": 2026,
                "appearances": 3,
                "lineups": 3,
                "minutes": 270,
                "goals": 2,
                "assists": 1,
                "shots": 8,
                "shots_on_target": 4,
                "passes": 90,
                "key_passes": 5,
                "pass_accuracy": 82.5,
                "tackles": 1,
                "interceptions": 0,
                "duels_won": 9,
                "yellow_cards": 1,
                "red_cards": 0,
                "rating": 7.2,
            },
        )
        upsert_player_match_stats(
            conn,
            {
                "match_id": None,
                "fixture_id": 123,
                "team_id": None,
                "api_team_id": 100,
                "team_name": "Mexico",
                "player_id": player_id,
                "position": "F",
                "is_starter": 1,
                "minutes": 90,
                "goals": 1,
                "assists": 0,
                "shots": 3,
                "shots_on_target": 2,
                "passes": 30,
                "key_passes": 1,
                "tackles": 0,
                "interceptions": 0,
                "duels_won": 3,
                "yellow_cards": 0,
                "red_cards": 0,
                "rating": 7.5,
            },
        )
        upsert_player_availability(
            conn,
            {
                "source": "api_football",
                "fixture_id": 124,
                "player_id": player_id,
                "api_player_id": 10,
                "team": "Mexico",
                "player_name": "Starter A",
                "reason": "Rotation risk",
                "status": "doubtful",
                "importance_score": 0.8,
            },
        )
        upsert_player_form_snapshot(
            conn,
            {
                "match_id": None,
                "fixture_id": 124,
                "team_id": None,
                "api_team_id": 100,
                "team_name": "Mexico",
                "player_id": player_id,
                "minutes_last_5": 420,
                "goals_last_5": 3,
                "assists_last_5": 1,
                "rating_last_5": 7.1,
                "starts_last_5": 5,
                "fatigue_score": 0.65,
                "form_score": 0.78,
                "availability_score": 0.7,
                "importance_score": 0.8,
            },
        )
        conn.commit()

        counts = {
            name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in [
                "players",
                "team_squads",
                "player_season_stats",
                "player_match_stats",
                "player_availability",
                "player_form_snapshots",
            ]
        }

    assert counts == {
        "players": 1,
        "team_squads": 1,
        "player_season_stats": 1,
        "player_match_stats": 1,
        "player_availability": 1,
        "player_form_snapshots": 1,
    }


def test_api_fixture_player_is_mirrored_into_normalized_lineups(tmp_path: Path) -> None:
    db_path = tmp_path / "lineups.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        upsert_api_football_fixture_player(
            conn,
            {
                "fixture_id": 123,
                "team_id": 100,
                "team_name": "Mexico",
                "player_id": 10,
                "player_name": "Starter A",
                "number": 9,
                "position": "F",
                "grid": "1:1",
                "is_starting": 1,
            },
        )
        conn.commit()
        player = conn.execute("SELECT api_player_id, name FROM players").fetchone()
        lineup = conn.execute("SELECT fixture_id, team_name, role, position FROM lineups").fetchone()

    assert dict(player) == {"api_player_id": 10, "name": "Starter A"}
    assert dict(lineup) == {"fixture_id": 123, "team_name": "Mexico", "role": "starter", "position": "F"}


def test_player_quality_summary_exposes_model_ready_features(tmp_path: Path) -> None:
    db_path = tmp_path / "player_quality.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        player_id = upsert_player(conn, {"api_player_id": 10, "name": "Starter A"})
        upsert_player_season_stats(
            conn,
            {
                "player_id": player_id,
                "team_id": None,
                "api_team_id": 100,
                "team_name": "Mexico",
                "competition_id": 1,
                "competition_name": "World Cup",
                "season": 2026,
                "appearances": 20,
                "lineups": 18,
                "minutes": 1800,
                "goals": 12,
                "assists": 5,
                "shots": 44,
                "shots_on_target": 20,
                "passes": 500,
                "key_passes": 30,
                "pass_accuracy": 82,
                "tackles": 6,
                "interceptions": 2,
                "duels_won": 80,
                "yellow_cards": 2,
                "red_cards": 0,
                "rating": 7.3,
            },
        )
        conn.commit()
        summary = player_quality_summary_by_team(conn)

    mexico = summary["Mexico"]
    assert mexico["squad_players_tracked"] == 1
    assert mexico["player_rating_score"] > 50
    assert mexico["player_attack_output_score"] > 50
