from football_predictor.data.api_football_sync import (
    fixture_to_context_row,
    fixture_to_match_row,
    normalize_lineups,
    normalize_fixture_player_statistics,
    normalize_fixture,
    normalize_fixture_statistics,
    normalize_league_coverage,
    normalize_referee_history,
    normalize_injuries,
    normalize_standings,
    normalize_player_statistics,
    normalize_team_statistics,
    sync_api_football_players,
    sync_api_football_league_coverage,
    sync_competition_season_core,
    sync_api_football_fixtures_by_date,
)
from football_predictor.database.db import (
    connect,
    count_matches_shadowed_by_api_football,
    delete_matches_shadowed_by_api_football,
    init_db,
    insert_matches,
)


def sample_fixture() -> dict:
    return {
        "fixture": {
            "id": 123,
            "referee": "Maurizio Mariani, Italy",
            "date": "2026-06-11T20:00:00+00:00",
            "venue": {"name": "Estadio Azteca", "city": "Mexico City"},
            "status": {"short": "FT"},
        },
        "league": {"id": 1, "name": "World Cup", "season": 2026, "round": "Group Stage - 1"},
        "teams": {
            "home": {"id": 10, "name": "Mexico"},
            "away": {"id": 20, "name": "France"},
        },
        "goals": {"home": 1, "away": 2},
    }


def test_normalize_fixture_extracts_core_fields() -> None:
    row = normalize_fixture(sample_fixture())

    assert row["fixture_id"] == 123
    assert row["home_team"] == "Mexico"
    assert row["away_team"] == "France"
    assert row["venue_name"] == "Estadio Azteca"


def test_fixture_to_match_row_uses_only_finished_matches() -> None:
    row = fixture_to_match_row(sample_fixture())

    assert row is not None
    assert row["date"] == "2026-06-11"
    assert row["home_goals"] == 1
    assert row["competition"] == "World Cup"


def test_fixture_to_context_row_classifies_group_stage() -> None:
    row = fixture_to_context_row(sample_fixture())

    assert row is not None
    assert row["stage"] == "group_stage"
    assert row["competition_weight"] > 0.5
    assert row["raw"]["context"]["is_group_stage"] is True


def test_normalize_fixture_statistics_maps_api_types() -> None:
    payload = {
        "response": [
            {
                "team": {"name": "Mexico"},
                "statistics": [
                    {"type": "Expected Goals", "value": "1.4"},
                    {"type": "Ball Possession", "value": "54%"},
                    {"type": "Total Shots", "value": 12},
                    {"type": "Shots on Goal", "value": 5},
                    {"type": "Shots off Goal", "value": 4},
                    {"type": "Blocked Shots", "value": 3},
                    {"type": "Corner Kicks", "value": 6},
                    {"type": "Total passes", "value": 455},
                    {"type": "Passes accurate", "value": 390},
                    {"type": "Passes %", "value": "86%"},
                    {"type": "Yellow Cards", "value": 2},
                    {"type": "Red Cards", "value": 1},
                    {"type": "Dangerous Attacks", "value": 52},
                ],
            },
            {
                "team": {"name": "France"},
                "statistics": [{"type": "Expected Goals", "value": "2.1"}],
            },
        ]
    }

    rows = normalize_fixture_statistics(123, sample_fixture(), payload)

    mexico = rows[0]
    assert len(rows) == 2
    assert mexico["xg"] == 1.4
    assert mexico["possession_pct"] == 54
    assert mexico["shots_on_target"] == 5
    assert mexico["shots_off_target"] == 4
    assert mexico["blocked_shots"] == 3
    assert mexico["passes_total"] == 455
    assert mexico["passes_accurate"] == 390
    assert mexico["pass_accuracy_pct"] == 86
    assert mexico["dangerous_attacks"] == 52
    assert mexico["cards_estimate"] == 4


def test_normalize_referee_history_uses_fixture_stats_and_events() -> None:
    stats_rows = [
        {"team": "Mexico", "yellow_cards": 2, "red_cards": 0, "cards_estimate": 2, "fouls": 11},
        {"team": "France", "yellow_cards": 3, "red_cards": 1, "cards_estimate": 5, "fouls": 13},
    ]
    events = {
        "response": [
            {"team": {"name": "Mexico"}, "type": "Card", "detail": "Yellow Card", "comments": "Foul"},
            {"team": {"name": "France"}, "type": "Goal", "detail": "Penalty", "comments": None},
        ]
    }

    row = normalize_referee_history(sample_fixture(), stats_rows, events)

    assert row["referee_name"] == "Maurizio Mariani"
    assert row["referee_country"] == "Italy"
    assert row["total_cards"] == 7
    assert row["total_fouls"] == 24
    assert row["penalties"] == 1


def test_normalize_injuries_extracts_player_availability() -> None:
    payload = {
        "response": [
            {
                "player": {"name": "Player A", "reason": "Injury", "type": "Missing"},
                "team": {"name": "Mexico"},
                "fixture": {"id": 123},
            }
        ]
    }

    rows = normalize_injuries(payload)

    assert rows[0]["team"] == "Mexico"
    assert rows[0]["player_name"] == "Player A"
    assert rows[0]["fixture_id"] == 123


def test_normalize_standings_extracts_table_rows() -> None:
    payload = {
        "response": [
            {
                "league": {
                    "id": 1,
                    "name": "World Cup",
                    "season": 2026,
                    "standings": [
                        [
                            {
                                "rank": 1,
                                "team": {"id": 10, "name": "Mexico"},
                                "points": 6,
                                "goalsDiff": 3,
                                "group": "Group A",
                                "form": "WW",
                                "status": "same",
                                "description": "Qualified",
                                "all": {"played": 2, "win": 2, "draw": 0, "lose": 0, "goals": {"for": 4, "against": 1}},
                            }
                        ]
                    ],
                }
            }
        ]
    }

    rows = normalize_standings(payload, league_id=1, season=2026)

    assert rows[0]["team_name"] == "Mexico"
    assert rows[0]["group_name"] == "Group A"
    assert rows[0]["points"] == 6


def test_normalize_team_statistics_extracts_summary() -> None:
    payload = {
        "response": {
            "team": {"id": 10, "name": "Mexico"},
            "league": {"id": 1, "name": "World Cup", "season": 2026},
            "form": "WWDL",
            "fixtures": {"played": {"total": 4, "home": 2, "away": 2}, "wins": {"total": 2, "home": 2, "away": 0}, "draws": {"total": 1}, "loses": {"total": 1}},
            "goals": {"for": {"total": {"total": 6}}, "against": {"total": {"total": 4}}},
            "clean_sheet": {"total": 1},
            "failed_to_score": {"total": 0},
        }
    }

    row = normalize_team_statistics(payload, league_id=1, season=2026, team_id=10)

    assert row is not None
    assert row["played"] == 4
    assert row["goals_for"] == 6
    assert row["home_wins"] == 2


def test_normalize_player_statistics_extracts_player_squad_and_stats() -> None:
    payload = {
        "response": [
            {
                "player": {
                    "id": 7,
                    "name": "Player Seven",
                    "firstname": "Player",
                    "lastname": "Seven",
                    "age": 24,
                    "birth": {"date": "2002-01-01"},
                    "nationality": "Mexico",
                    "height": "180 cm",
                    "weight": "75 kg",
                    "photo": "https://example.test/player.png",
                },
                "statistics": [
                    {
                        "team": {"id": 10, "name": "Mexico"},
                        "league": {"id": 1, "name": "World Cup", "season": 2026},
                        "games": {"appearences": 4, "lineups": 3, "minutes": 300, "position": "Attacker", "rating": "7.12"},
                        "shots": {"total": 9, "on": 5},
                        "goals": {"total": 2, "assists": 1},
                        "passes": {"total": 100, "key": 8, "accuracy": "82"},
                        "tackles": {"total": 3, "interceptions": 1},
                        "duels": {"won": 14},
                        "cards": {"yellow": 1, "red": 0},
                    }
                ],
            }
        ]
    }

    players, squads, stats = normalize_player_statistics(payload, league_id=1, season=2026)

    assert players[0]["api_player_id"] == 7
    assert players[0]["preferred_position"] == "Attacker"
    assert squads[0]["team_name"] == "Mexico"
    assert stats[0]["minutes"] == 300
    assert stats[0]["rating"] == 7.12


def test_normalize_league_coverage_extracts_feature_flags() -> None:
    rows = normalize_league_coverage(
        {
            "league": {"id": 39, "name": "Premier League", "type": "League"},
            "country": {"name": "England", "code": "GB"},
            "seasons": [
                {
                    "year": 2025,
                    "current": True,
                    "start": "2025-08-01",
                    "end": "2026-05-30",
                    "coverage": {
                        "fixtures": {
                            "events": True,
                            "lineups": True,
                            "statistics_fixtures": True,
                            "statistics_players": False,
                        },
                        "standings": True,
                        "players": True,
                        "injuries": True,
                        "odds": False,
                    },
                }
            ],
        }
    )

    assert rows[0]["league_id"] == 39
    assert rows[0]["season"] == 2025
    assert rows[0]["fixtures_events"] == 1
    assert rows[0]["fixtures_statistics_players"] == 0
    assert rows[0]["odds"] == 0


def test_normalize_lineups_extracts_starters_and_subs() -> None:
    payload = {
        "response": [
            {
                "team": {"id": 10, "name": "Mexico"},
                "coach": {"id": 99, "name": "Coach A"},
                "formation": "4-3-3",
                "startXI": [{"player": {"id": 1, "name": "Starter", "number": 9, "pos": "F", "grid": "1:1"}}],
                "substitutes": [{"player": {"id": 2, "name": "Sub", "number": 12, "pos": "M"}}],
            }
        ]
    }

    lineups, players = normalize_lineups(123, payload)

    assert lineups[0]["formation"] == "4-3-3"
    assert len(players) == 2
    assert players[0]["is_starting"] == 1
    assert players[1]["is_starting"] == 0


def test_normalize_fixture_player_statistics_extracts_match_stats() -> None:
    payload = {
        "response": [
            {
                "team": {"id": 10, "name": "Mexico"},
                "players": [
                    {
                        "player": {"id": 7, "name": "Player Seven"},
                        "statistics": [
                            {
                                "games": {"minutes": 90, "position": "F", "rating": "7.5", "substitute": False},
                                "shots": {"total": 4, "on": 2},
                                "goals": {"total": 1, "assists": 1},
                                "passes": {"total": 33, "key": 2},
                                "tackles": {"total": 1, "interceptions": 0},
                                "duels": {"won": 5},
                                "cards": {"yellow": 1, "red": 0},
                            }
                        ],
                    }
                ],
            }
        ]
    }

    players, stats = normalize_fixture_player_statistics(123, payload)

    assert players[0]["api_player_id"] == 7
    assert stats[0]["fixture_id"] == 123
    assert stats[0]["team_name"] == "Mexico"
    assert stats[0]["shots_on_target"] == 2
    assert stats[0]["rating"] == 7.5


class FakeApiFootballClient:
    calls = 0

    def get(self, endpoint: str, **params) -> dict:
        self.calls += 1
        assert endpoint == "leagues"
        assert params.get("id") == 1
        return {
            "response": [
                {
                    "league": {"id": 1, "name": "World Cup", "type": "Cup"},
                    "country": {"name": "World", "code": None},
                    "seasons": [
                        {
                            "year": 2026,
                            "current": True,
                            "coverage": {
                                "fixtures": {"events": True, "lineups": True, "statistics_fixtures": True, "statistics_players": True},
                                "standings": True,
                                "players": True,
                                "injuries": True,
                                "odds": True,
                            },
                        }
                    ],
                }
            ],
            "results": 1,
            "errors": {},
        }

    def fixtures(self, league: int, season: int) -> dict:
        self.calls += 1
        assert league == 1
        assert season == 2026
        return {"response": [sample_fixture()], "results": 1, "errors": {}, "paging": {"current": 1, "total": 1}}

    def fixtures_by_date(self, date: str, league: int | None = None, season: int | None = None) -> dict:
        self.calls += 1
        assert date == "2026-06-11"
        assert league == 1
        assert season == 2026
        return {"response": [sample_fixture()], "results": 1, "errors": {}, "paging": {"current": 1, "total": 1}}

    def teams(self, league: int, season: int) -> dict:
        self.calls += 1
        assert league == 1
        assert season == 2026
        return {
            "response": [
                {
                    "team": {"id": 10, "name": "Mexico", "country": "Mexico", "founded": 1927, "national": True},
                    "venue": {"name": "Estadio Azteca", "city": "Mexico City"},
                }
            ],
            "results": 1,
            "errors": {},
        }

    def players(self, league: int, season: int, team: int | None = None, page: int = 1) -> dict:
        self.calls += 1
        assert league == 1
        assert season == 2026
        assert team == 10
        assert page == 1
        return {
            "response": [
                {
                    "player": {"id": 7, "name": "Player Seven", "birth": {"date": "2002-01-01"}, "nationality": "Mexico"},
                    "statistics": [
                        {
                            "team": {"id": 10, "name": "Mexico"},
                            "league": {"id": 1, "name": "World Cup", "season": 2026},
                            "games": {"appearences": 1, "lineups": 1, "minutes": 90, "position": "Attacker", "rating": "7.0"},
                            "goals": {"total": 1, "assists": 0},
                        }
                    ],
                }
            ],
            "results": 1,
            "errors": {},
            "paging": {"current": 1, "total": 1},
        }

    def fixture_statistics(self, fixture: int) -> dict:
        self.calls += 1
        assert fixture == 123
        return {
            "response": [
                {
                    "team": {"name": "Mexico"},
                    "statistics": [
                        {"type": "Corner Kicks", "value": 6},
                        {"type": "Shots on Goal", "value": 5},
                        {"type": "Total Shots", "value": 12},
                        {"type": "Fouls", "value": 11},
                        {"type": "Yellow Cards", "value": 2},
                        {"type": "Red Cards", "value": 0},
                    ],
                },
                {
                    "team": {"name": "France"},
                    "statistics": [
                        {"type": "Corner Kicks", "value": 2},
                        {"type": "Shots on Goal", "value": 4},
                        {"type": "Total Shots", "value": 9},
                        {"type": "Fouls", "value": 13},
                        {"type": "Yellow Cards", "value": 1},
                        {"type": "Red Cards", "value": 0},
                    ],
                },
            ],
            "results": 2,
            "errors": {},
        }

    def fixture_events(self, fixture: int) -> dict:
        self.calls += 1
        assert fixture == 123
        return {"response": [], "results": 0, "errors": {}}

    def injuries(self, fixture: int) -> dict:
        self.calls += 1
        assert fixture == 123
        return {"response": [], "results": 0, "errors": {}}

    def fixture_lineups(self, fixture: int) -> dict:
        self.calls += 1
        assert fixture == 123
        return {"response": [], "results": 0, "errors": {}}

    def fixture_players(self, fixture: int) -> dict:
        self.calls += 1
        assert fixture == 123
        return {"response": [], "results": 0, "errors": {}}


def test_sync_fixtures_by_date_stores_api_payload(tmp_path) -> None:
    db_path = tmp_path / "api.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        result = sync_api_football_fixtures_by_date(conn, FakeApiFootballClient(), date="2026-06-11", league=1, season=2026)
        stored = conn.execute("SELECT COUNT(*) FROM api_football_fixtures").fetchone()[0]
        contexts = conn.execute("SELECT stage FROM match_context").fetchall()
        match = conn.execute("SELECT home_team, away_team FROM matches").fetchone()
        stats = conn.execute(
            """
            SELECT SUM(corners) AS corners, SUM(shots_on_target) AS shots_on_target, SUM(yellow_cards) AS yellow_cards
            FROM match_team_advanced_stats
            WHERE source_match_id = 'api-football:123'
            """
        ).fetchone()

    assert result["fixtures"] == 1
    assert result["finished_matches_inserted"] == 1
    assert result["finished_details"]["fixture_details_synced"] == 1
    assert result["finished_details"]["advanced_stats"] == 2
    assert stored == 1
    assert [dict(row) for row in contexts] == [{"stage": "group_stage"}]
    assert dict(match) == {"home_team": "Mexico", "away_team": "France"}
    assert dict(stats) == {"corners": 8, "shots_on_target": 9, "yellow_cards": 3}


def test_sync_competition_season_skips_when_inventory_is_complete(tmp_path) -> None:
    db_path = tmp_path / "inventory.sqlite"
    init_db(db_path)
    fake_client = FakeApiFootballClient()

    with connect(db_path) as conn:
        first = sync_competition_season_core(conn, fake_client, league_id=1, season=2026, league_name="World Cup")
        second = sync_competition_season_core(conn, fake_client, league_id=1, season=2026, league_name="World Cup")
        inventory_rows = conn.execute("SELECT data_type, status FROM sync_inventory ORDER BY data_type").fetchall()

    assert first["requests_used"] == 2
    assert second["requests_used"] == 0
    assert second["actions"][0]["skipped"] is True
    assert [dict(row) for row in inventory_rows] == [
        {"data_type": "fixtures", "status": "complete"},
        {"data_type": "teams", "status": "complete"},
    ]


def test_sync_api_football_players_stores_normalized_player_data(tmp_path) -> None:
    db_path = tmp_path / "players_api.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        result = sync_competition_season_core(conn, FakeApiFootballClient(), league_id=1, season=2026, league_name="World Cup")
        assert result["requests_used"] == 2
        player_result = sync_api_football_players(conn, FakeApiFootballClient(), league=1, season=2026, max_requests=10)
        player = conn.execute("SELECT api_player_id, name FROM players").fetchone()
        squad = conn.execute("SELECT team_name, season FROM team_squads").fetchone()
        stats = conn.execute("SELECT minutes, goals FROM player_season_stats").fetchone()

    assert player_result["players"] == 1
    assert dict(player) == {"api_player_id": 7, "name": "Player Seven"}
    assert dict(squad) == {"team_name": "Mexico", "season": 2026}
    assert dict(stats) == {"minutes": 90, "goals": 1}


def test_sync_api_football_league_coverage_stores_flags(tmp_path) -> None:
    db_path = tmp_path / "coverage_api.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        result = sync_api_football_league_coverage(conn, FakeApiFootballClient(), league_ids=[1], season=2026)
        row = conn.execute(
            """
            SELECT league_name, season, fixtures_lineups, players, injuries, odds
            FROM api_football_league_coverage
            WHERE league_id = 1 AND season = 2026
            """
        ).fetchone()

    assert result["coverage_rows"] == 1
    assert dict(row) == {
        "league_name": "World Cup",
        "season": 2026,
        "fixtures_lineups": 1,
        "players": 1,
        "injuries": 1,
        "odds": 1,
    }


def test_delete_matches_shadowed_by_api_football_keeps_unique_historical(tmp_path) -> None:
    db_path = tmp_path / "dedupe.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        insert_matches(
            conn,
            [
                {
                    "date": "2026-06-11",
                    "home_team": "Mexico",
                    "away_team": "France",
                    "home_goals": 1,
                    "away_goals": 2,
                    "competition": "World Cup",
                    "season": "2026",
                },
                {
                    "date": "2025-01-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 0,
                    "away_goals": 0,
                    "competition": "Friendly",
                    "season": "2025",
                },
            ],
        )
        from football_predictor.database.db import upsert_api_football_fixture

        upsert_api_football_fixture(conn, normalize_fixture(sample_fixture()))
        assert count_matches_shadowed_by_api_football(conn) == 1
        assert delete_matches_shadowed_by_api_football(conn) == 1
        remaining = conn.execute("SELECT home_team, away_team, competition FROM matches").fetchall()

    assert [dict(row) for row in remaining] == [{"home_team": "A", "away_team": "B", "competition": "Friendly"}]
