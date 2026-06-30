from pathlib import Path

from football_predictor.database.db import connect, init_db, upsert_player, upsert_player_season_stats
from football_predictor.prediction.player_markets import estimate_player_markets, position_attack_boost


def add_player(conn, team: str, api_id: int, name: str, position: str, minutes: int, goals: int, shots_on_target: int, rating: float) -> None:
    player_id = upsert_player(conn, {"api_player_id": api_id, "name": name, "preferred_position": position})
    upsert_player_season_stats(
        conn,
        {
            "player_id": player_id,
            "team_id": None,
            "api_team_id": None,
            "team_name": team,
            "competition_id": 1,
            "competition_name": "League",
            "season": 2026,
            "appearances": 10,
            "lineups": 8,
            "minutes": minutes,
            "goals": goals,
            "assists": 1,
            "shots": shots_on_target * 2,
            "shots_on_target": shots_on_target,
            "passes": 100,
            "key_passes": 4,
            "pass_accuracy": 80,
            "tackles": 1,
            "interceptions": 0,
            "duels_won": 10,
            "yellow_cards": 1,
            "red_cards": 0,
            "rating": rating,
        },
    )


def test_player_markets_rank_scorers_and_shots_on_target(tmp_path: Path) -> None:
    db_path = tmp_path / "player_markets.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        add_player(conn, "A", 1, "Striker A", "Attacker", 1200, 12, 24, 7.4)
        add_player(conn, "A", 2, "Mid A", "Midfielder", 1300, 3, 8, 6.9)
        add_player(conn, "B", 3, "Striker B", "Attacker", 1100, 7, 16, 7.1)
        add_player(conn, "B", 4, "Def B", "Defender", 1500, 1, 3, 6.8)
        conn.commit()
        result = estimate_player_markets(
            conn,
            "A",
            "B",
            {
                "home_shots_on_target_expected": 5.2,
                "away_shots_on_target_expected": 3.4,
                "home_xg_recent": 1.8,
                "away_xg_recent": 1.0,
            },
        )

    assert result["home_team"]["expected_shots_on_target"] == 5.2
    assert result["top_scorers"][0]["player_name"] == "Striker A"
    assert result["top_shots_on_target"][0]["player_name"] == "Striker A"
    assert result["top_scorers"][0]["score_prob"] > result["top_scorers"][-1]["score_prob"]


def test_position_attack_boost_does_not_treat_defender_as_forward() -> None:
    assert position_attack_boost("Attacker") > position_attack_boost("Defender")
    assert position_attack_boost("F") > position_attack_boost("D")
    assert position_attack_boost("Goalkeeper") < position_attack_boost("Defender")
