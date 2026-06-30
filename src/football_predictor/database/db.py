from pathlib import Path
import json
import sqlite3

from football_predictor.config import settings


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        ensure_schema_compatibility(conn)


def ensure_schema_compatibility(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(prediction_backtests)").fetchall()}
    if "snapshot_json" not in columns:
        conn.execute("ALTER TABLE prediction_backtests ADD COLUMN snapshot_json TEXT")
    availability_columns = {row["name"] for row in conn.execute("PRAGMA table_info(player_availability)").fetchall()}
    availability_additions = {
        "player_id": "INTEGER",
        "api_player_id": "INTEGER",
        "team_id": "INTEGER",
        "api_team_id": "INTEGER",
        "expected_return": "TEXT",
        "importance_score": "REAL",
    }
    for column, column_type in availability_additions.items():
        if column not in availability_columns:
            conn.execute(f"ALTER TABLE player_availability ADD COLUMN {column} {column_type}")
    advanced_columns = {row["name"] for row in conn.execute("PRAGMA table_info(match_team_advanced_stats)").fetchall()}
    advanced_additions = {
        "shots_off_target": "INTEGER",
        "blocked_shots": "INTEGER",
        "yellow_cards": "INTEGER",
        "red_cards": "INTEGER",
        "passes_total": "INTEGER",
        "passes_accurate": "INTEGER",
        "pass_accuracy_pct": "REAL",
        "attacks": "INTEGER",
        "dangerous_attacks": "INTEGER",
    }
    for column, column_type in advanced_additions.items():
        if column not in advanced_columns:
            conn.execute(f"ALTER TABLE match_team_advanced_stats ADD COLUMN {column} {column_type}")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS api_football_team_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER NOT NULL,
            league_name TEXT,
            season INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            played INTEGER,
            wins INTEGER,
            draws INTEGER,
            losses INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            home_played INTEGER,
            home_wins INTEGER,
            away_played INTEGER,
            away_wins INTEGER,
            clean_sheets INTEGER,
            failed_to_score INTEGER,
            form TEXT,
            raw_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(league_id, season, team_id)
        );

        CREATE TABLE IF NOT EXISTS api_football_standings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER NOT NULL,
            league_name TEXT,
            season INTEGER NOT NULL,
            group_name TEXT,
            rank INTEGER,
            team_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            points INTEGER,
            goals_diff INTEGER,
            form TEXT,
            status TEXT,
            description TEXT,
            played INTEGER,
            wins INTEGER,
            draws INTEGER,
            losses INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            raw_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(league_id, season, group_name, team_id)
        );

        CREATE TABLE IF NOT EXISTS api_football_fixture_lineups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            formation TEXT,
            coach_id INTEGER,
            coach_name TEXT,
            raw_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(fixture_id, team_id)
        );

        CREATE TABLE IF NOT EXISTS api_football_fixture_players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            player_id INTEGER,
            player_name TEXT NOT NULL,
            number INTEGER,
            position TEXT,
            grid TEXT,
            is_starting INTEGER NOT NULL,
            raw_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(fixture_id, team_id, player_id, player_name, is_starting)
        );

        CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
        CREATE INDEX IF NOT EXISTS idx_team_squads_team_season ON team_squads(team_name, season);
        CREATE INDEX IF NOT EXISTS idx_matches_date_id ON matches(date DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_matches_competition_season ON matches(competition, season);
        CREATE INDEX IF NOT EXISTS idx_matches_teams_date ON matches(home_team, away_team, date);
        CREATE INDEX IF NOT EXISTS idx_api_fixtures_date ON api_football_fixtures(date);
        CREATE INDEX IF NOT EXISTS idx_api_fixtures_league_season ON api_football_fixtures(league_id, season);
        CREATE INDEX IF NOT EXISTS idx_api_fixtures_status ON api_football_fixtures(status_short);
        CREATE INDEX IF NOT EXISTS idx_api_fixtures_match_lookup ON api_football_fixtures(home_team, away_team, league_name, season);
        CREATE INDEX IF NOT EXISTS idx_api_fixtures_dedupe_lookup ON api_football_fixtures(
            substr(date, 1, 10),
            lower(home_team),
            lower(away_team),
            lower(league_name),
            CAST(season AS TEXT)
        );
        CREATE INDEX IF NOT EXISTS idx_prediction_backtests_date_id ON prediction_backtests(match_date DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_prediction_backtests_pending ON prediction_backtests(actual_home_goals, actual_away_goals, match_date);
        CREATE INDEX IF NOT EXISTS idx_prediction_backtests_match ON prediction_backtests(match_date, home_team, away_team);
        CREATE INDEX IF NOT EXISTS idx_match_context_match ON match_context(home_team, away_team, date);
        CREATE INDEX IF NOT EXISTS idx_advanced_stats_team_date ON match_team_advanced_stats(team, date);
        CREATE INDEX IF NOT EXISTS idx_advanced_stats_opponent ON match_team_advanced_stats(opponent);
        CREATE INDEX IF NOT EXISTS idx_player_season_stats_team ON player_season_stats(team_name, season);
        CREATE INDEX IF NOT EXISTS idx_player_season_stats_team_nocase ON player_season_stats(team_name COLLATE NOCASE, season);
        CREATE INDEX IF NOT EXISTS idx_player_season_stats_player_team ON player_season_stats(player_id, team_name);
        CREATE INDEX IF NOT EXISTS idx_player_match_stats_fixture ON player_match_stats(fixture_id);
        CREATE INDEX IF NOT EXISTS idx_lineups_fixture_team ON lineups(fixture_id, team_name);
        CREATE INDEX IF NOT EXISTS idx_player_availability_fixture_team ON player_availability(fixture_id, team);
        CREATE INDEX IF NOT EXISTS idx_player_availability_team_player ON player_availability(team, player_id);
        CREATE INDEX IF NOT EXISTS idx_player_availability_team_nocase ON player_availability(team COLLATE NOCASE, player_id);
        """
    )


def upsert_teams(conn: sqlite3.Connection, team_names: list[str]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO teams(name) VALUES (?)",
        [(name,) for name in sorted(set(team_names))],
    )


def insert_matches(conn: sqlite3.Connection, rows: list[dict]) -> int:
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO matches
        (date, home_team, away_team, home_goals, away_goals, competition, season)
        VALUES (:date, :home_team, :away_team, :home_goals, :away_goals, :competition, :season)
        """,
        rows,
    )
    return conn.total_changes - before


def insert_prediction(conn: sqlite3.Connection, prediction: dict, match_id: int | None = None) -> int:
    cursor = conn.execute(
        """
        INSERT INTO predictions
        (match_id, home_team, away_team, model_name, home_win_prob, draw_prob, away_win_prob,
         most_likely_score, over_2_5_prob, both_teams_score_prob)
        VALUES
        (:match_id, :home_team, :away_team, :model_name, :home_win_prob, :draw_prob, :away_win_prob,
         :most_likely_score, :over_2_5_prob, :both_teams_score_prob)
        """,
        {**prediction, "match_id": match_id},
    )
    return int(cursor.lastrowid)


def insert_model_artifact(
    conn: sqlite3.Connection,
    model_name: str,
    model_type: str,
    artifact_path: str,
    trained_until: str | None = None,
    metrics_json: str | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO model_artifacts
        (model_name, model_type, artifact_path, trained_until, metrics_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (model_name, model_type, artifact_path, trained_until, metrics_json),
    )
    return int(cursor.lastrowid)


def list_teams(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM teams ORDER BY name").fetchall()
    return [str(row["name"]) for row in rows]


def upsert_team_external_metrics(conn: sqlite3.Connection, metrics: dict) -> None:
    conn.execute(
        """
        INSERT INTO team_external_metrics
        (team, source, fifa_rank, fifa_points, squad_value_eur, squad_size, avg_age, raw_json, updated_at)
        VALUES
        (:team, :source, :fifa_rank, :fifa_points, :squad_value_eur, :squad_size, :avg_age, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(team, source) DO UPDATE SET
            fifa_rank = excluded.fifa_rank,
            fifa_points = excluded.fifa_points,
            squad_value_eur = excluded.squad_value_eur,
            squad_size = excluded.squad_size,
            avg_age = excluded.avg_age,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            "team": metrics["team"],
            "source": metrics["source"],
            "fifa_rank": metrics.get("fifa_rank"),
            "fifa_points": metrics.get("fifa_points"),
            "squad_value_eur": metrics.get("squad_value_eur"),
            "squad_size": metrics.get("squad_size"),
            "avg_age": metrics.get("avg_age"),
            "raw_json": json.dumps(metrics.get("raw", {})),
        },
    )


def get_team_external_metrics(conn: sqlite3.Connection, team: str) -> dict:
    rows = conn.execute(
        """
        SELECT * FROM team_external_metrics
        WHERE team = ?
        ORDER BY updated_at DESC
        """,
        (team,),
    ).fetchall()
    merged: dict = {}
    for row in rows:
        data = dict(row)
        raw = json.loads(data.get("raw_json") or "{}")
        if "raw" not in merged and raw:
            merged["raw"] = raw
        for key in ["fifa_rank", "fifa_points", "squad_value_eur", "squad_size", "avg_age"]:
            if merged.get(key) is None and data.get(key) is not None:
                merged[key] = data[key]
    return merged


def insert_experiment(
    conn: sqlite3.Connection,
    name: str,
    kind: str,
    config: dict,
    metrics: dict | None = None,
    artifact_path: str | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO experiments (name, kind, config_json, metrics_json, artifact_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            name,
            kind,
            json.dumps(config),
            json.dumps(metrics or {}),
            artifact_path,
        ),
    )
    return int(cursor.lastrowid)


def list_experiments(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, name, kind, config_json, metrics_json, artifact_path, created_at
        FROM experiments
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_venue(conn: sqlite3.Connection, venue: dict) -> None:
    conn.execute(
        """
        INSERT INTO venues
        (source_id, stadium_name, city, country, capacity, latitude, longitude, altitude_m, surface, roof, raw_json)
        VALUES (:source_id, :stadium_name, :city, :country, :capacity, :latitude, :longitude, :altitude_m, :surface, :roof, :raw_json)
        ON CONFLICT(stadium_name, city) DO UPDATE SET
            source_id = excluded.source_id,
            country = excluded.country,
            capacity = excluded.capacity,
            latitude = excluded.latitude,
            longitude = excluded.longitude,
            altitude_m = excluded.altitude_m,
            surface = excluded.surface,
            roof = excluded.roof,
            raw_json = excluded.raw_json
        """,
        {**venue, "raw_json": json.dumps(venue.get("raw", {}))},
    )


def upsert_squad_player(conn: sqlite3.Connection, player: dict) -> None:
    conn.execute(
        """
        INSERT INTO squad_players
        (source_id, team, player_name, position, club_team, market_value_eur, caps, date_of_birth, height_cm, goals, raw_json)
        VALUES
        (:source_id, :team, :player_name, :position, :club_team, :market_value_eur, :caps, :date_of_birth, :height_cm, :goals, :raw_json)
        ON CONFLICT(team, player_name) DO UPDATE SET
            position = excluded.position,
            club_team = excluded.club_team,
            market_value_eur = excluded.market_value_eur,
            caps = excluded.caps,
            date_of_birth = excluded.date_of_birth,
            height_cm = excluded.height_cm,
            goals = excluded.goals,
            raw_json = excluded.raw_json
        """,
        {**player, "raw_json": json.dumps(player.get("raw", {}))},
    )


def _get_or_create_team_id(conn: sqlite3.Connection, team_name: str | None) -> int | None:
    if not team_name:
        return None
    conn.execute("INSERT OR IGNORE INTO teams(name) VALUES (?)", (team_name,))
    row = conn.execute("SELECT id FROM teams WHERE name = ?", (team_name,)).fetchone()
    return int(row["id"]) if row else None


def upsert_player(conn: sqlite3.Connection, player: dict) -> int:
    api_player_id = player.get("api_player_id") or player.get("player_id")
    name = player.get("name") or player.get("player_name")
    if not name:
        raise ValueError("player name is required")
    existing = None
    if api_player_id is not None:
        existing = conn.execute("SELECT id FROM players WHERE api_player_id = ?", (api_player_id,)).fetchone()
    if existing is None:
        existing = conn.execute(
            """
            SELECT id
            FROM players
            WHERE lower(name) = lower(?)
              AND COALESCE(birth_date, '') = COALESCE(?, '')
              AND COALESCE(nationality, '') = COALESCE(?, '')
            """,
            (name, player.get("birth_date"), player.get("nationality")),
        ).fetchone()
    payload = {
        "api_player_id": api_player_id,
        "name": name,
        "firstname": player.get("firstname"),
        "lastname": player.get("lastname"),
        "birth_date": player.get("birth_date"),
        "age": player.get("age"),
        "nationality": player.get("nationality"),
        "height": player.get("height"),
        "weight": player.get("weight"),
        "preferred_position": player.get("preferred_position") or player.get("position"),
        "photo_url": player.get("photo_url") or player.get("photo"),
        "raw_json": json.dumps(player.get("raw", {})),
    }
    if existing:
        conn.execute(
            """
            UPDATE players
            SET api_player_id = COALESCE(:api_player_id, api_player_id),
                name = :name,
                firstname = COALESCE(:firstname, firstname),
                lastname = COALESCE(:lastname, lastname),
                birth_date = COALESCE(:birth_date, birth_date),
                age = COALESCE(:age, age),
                nationality = COALESCE(:nationality, nationality),
                height = COALESCE(:height, height),
                weight = COALESCE(:weight, weight),
                preferred_position = COALESCE(:preferred_position, preferred_position),
                photo_url = COALESCE(:photo_url, photo_url),
                raw_json = CASE WHEN :raw_json != '{}' THEN :raw_json ELSE raw_json END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :id
            """,
            {**payload, "id": existing["id"]},
        )
        return int(existing["id"])
    cursor = conn.execute(
        """
        INSERT INTO players
        (api_player_id, name, firstname, lastname, birth_date, age, nationality,
         height, weight, preferred_position, photo_url, raw_json, updated_at)
        VALUES
        (:api_player_id, :name, :firstname, :lastname, :birth_date, :age, :nationality,
         :height, :weight, :preferred_position, :photo_url, :raw_json, CURRENT_TIMESTAMP)
        """,
        payload,
    )
    return int(cursor.lastrowid)


def upsert_team_squad(conn: sqlite3.Connection, row: dict) -> None:
    team_name = row["team_name"]
    player_id = row["player_id"]
    team_id = row.get("team_id") or _get_or_create_team_id(conn, team_name)
    conn.execute(
        """
        INSERT INTO team_squads
        (team_id, api_team_id, team_name, player_id, competition_id, competition_name, season,
         squad_number, position, is_active, joined_at, left_at, source, raw_json, updated_at)
        VALUES
        (:team_id, :api_team_id, :team_name, :player_id, :competition_id, :competition_name, :season,
         :squad_number, :position, :is_active, :joined_at, :left_at, :source, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(team_name, player_id, competition_id, season) DO UPDATE SET
            team_id = excluded.team_id,
            api_team_id = excluded.api_team_id,
            competition_name = excluded.competition_name,
            squad_number = excluded.squad_number,
            position = excluded.position,
            is_active = excluded.is_active,
            joined_at = excluded.joined_at,
            left_at = excluded.left_at,
            source = excluded.source,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            **row,
            "team_id": team_id,
            "source": row.get("source", "api_football"),
            "is_active": int(row.get("is_active", 1)),
            "raw_json": json.dumps(row.get("raw", {})),
        },
    )


def upsert_player_season_stats(conn: sqlite3.Connection, row: dict) -> None:
    team_name = row["team_name"]
    conn.execute(
        """
        INSERT INTO player_season_stats
        (player_id, team_id, api_team_id, team_name, competition_id, competition_name, season,
         appearances, lineups, minutes, goals, assists, shots, shots_on_target, passes,
         key_passes, pass_accuracy, tackles, interceptions, duels_won, yellow_cards,
         red_cards, rating, source, raw_json, updated_at)
        VALUES
        (:player_id, :team_id, :api_team_id, :team_name, :competition_id, :competition_name, :season,
         :appearances, :lineups, :minutes, :goals, :assists, :shots, :shots_on_target, :passes,
         :key_passes, :pass_accuracy, :tackles, :interceptions, :duels_won, :yellow_cards,
         :red_cards, :rating, :source, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(player_id, team_name, competition_id, season, source) DO UPDATE SET
            team_id = excluded.team_id,
            api_team_id = excluded.api_team_id,
            competition_name = excluded.competition_name,
            appearances = excluded.appearances,
            lineups = excluded.lineups,
            minutes = excluded.minutes,
            goals = excluded.goals,
            assists = excluded.assists,
            shots = excluded.shots,
            shots_on_target = excluded.shots_on_target,
            passes = excluded.passes,
            key_passes = excluded.key_passes,
            pass_accuracy = excluded.pass_accuracy,
            tackles = excluded.tackles,
            interceptions = excluded.interceptions,
            duels_won = excluded.duels_won,
            yellow_cards = excluded.yellow_cards,
            red_cards = excluded.red_cards,
            rating = excluded.rating,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            **row,
            "team_id": row.get("team_id") or _get_or_create_team_id(conn, team_name),
            "source": row.get("source", "api_football"),
            "raw_json": json.dumps(row.get("raw", {})),
        },
    )


def upsert_player_match_stats(conn: sqlite3.Connection, row: dict) -> None:
    team_name = row["team_name"]
    conn.execute(
        """
        INSERT INTO player_match_stats
        (match_id, fixture_id, team_id, api_team_id, team_name, player_id, position, is_starter,
         minutes, goals, assists, shots, shots_on_target, passes, key_passes, tackles,
         interceptions, duels_won, yellow_cards, red_cards, rating, source, raw_json, updated_at)
        VALUES
        (:match_id, :fixture_id, :team_id, :api_team_id, :team_name, :player_id, :position, :is_starter,
         :minutes, :goals, :assists, :shots, :shots_on_target, :passes, :key_passes, :tackles,
         :interceptions, :duels_won, :yellow_cards, :red_cards, :rating, :source, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(fixture_id, team_name, player_id, source) DO UPDATE SET
            match_id = excluded.match_id,
            team_id = excluded.team_id,
            api_team_id = excluded.api_team_id,
            position = excluded.position,
            is_starter = excluded.is_starter,
            minutes = excluded.minutes,
            goals = excluded.goals,
            assists = excluded.assists,
            shots = excluded.shots,
            shots_on_target = excluded.shots_on_target,
            passes = excluded.passes,
            key_passes = excluded.key_passes,
            tackles = excluded.tackles,
            interceptions = excluded.interceptions,
            duels_won = excluded.duels_won,
            yellow_cards = excluded.yellow_cards,
            red_cards = excluded.red_cards,
            rating = excluded.rating,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            **row,
            "team_id": row.get("team_id") or _get_or_create_team_id(conn, team_name),
            "is_starter": int(row.get("is_starter", 0)),
            "source": row.get("source", "api_football"),
            "raw_json": json.dumps(row.get("raw", {})),
        },
    )


def upsert_lineup_entry(conn: sqlite3.Connection, row: dict) -> None:
    team_name = row["team_name"]
    conn.execute(
        """
        INSERT INTO lineups
        (match_id, fixture_id, team_id, api_team_id, team_name, player_id, api_player_id,
         role, position, formation_position, shirt_number, source, raw_json, updated_at)
        VALUES
        (:match_id, :fixture_id, :team_id, :api_team_id, :team_name, :player_id, :api_player_id,
         :role, :position, :formation_position, :shirt_number, :source, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(fixture_id, team_name, player_id, role, source) DO UPDATE SET
            match_id = excluded.match_id,
            team_id = excluded.team_id,
            api_team_id = excluded.api_team_id,
            api_player_id = excluded.api_player_id,
            position = excluded.position,
            formation_position = excluded.formation_position,
            shirt_number = excluded.shirt_number,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            **row,
            "team_id": row.get("team_id") or _get_or_create_team_id(conn, team_name),
            "source": row.get("source", "api_football"),
            "raw_json": json.dumps(row.get("raw", {})),
        },
    )


def upsert_player_form_snapshot(conn: sqlite3.Connection, row: dict) -> None:
    team_name = row["team_name"]
    conn.execute(
        """
        INSERT INTO player_form_snapshots
        (match_id, fixture_id, team_id, api_team_id, team_name, player_id,
         minutes_last_5, goals_last_5, assists_last_5, rating_last_5, starts_last_5,
         fatigue_score, form_score, availability_score, importance_score, raw_json)
        VALUES
        (:match_id, :fixture_id, :team_id, :api_team_id, :team_name, :player_id,
         :minutes_last_5, :goals_last_5, :assists_last_5, :rating_last_5, :starts_last_5,
         :fatigue_score, :form_score, :availability_score, :importance_score, :raw_json)
        ON CONFLICT(fixture_id, team_name, player_id) DO UPDATE SET
            match_id = excluded.match_id,
            team_id = excluded.team_id,
            api_team_id = excluded.api_team_id,
            minutes_last_5 = excluded.minutes_last_5,
            goals_last_5 = excluded.goals_last_5,
            assists_last_5 = excluded.assists_last_5,
            rating_last_5 = excluded.rating_last_5,
            starts_last_5 = excluded.starts_last_5,
            fatigue_score = excluded.fatigue_score,
            form_score = excluded.form_score,
            availability_score = excluded.availability_score,
            importance_score = excluded.importance_score,
            raw_json = excluded.raw_json
        """,
        {
            **row,
            "team_id": row.get("team_id") or _get_or_create_team_id(conn, team_name),
            "raw_json": json.dumps(row.get("raw", {})),
        },
    )


def upsert_advanced_stats(conn: sqlite3.Connection, stats: dict) -> None:
    conn.execute(
        """
        INSERT INTO match_team_advanced_stats
        (source_match_id, team, opponent, date, xg, possession_pct, total_shots, shots_on_target,
         shots_off_target, blocked_shots, corners, fouls, offsides, saves, yellow_cards, red_cards,
         cards_estimate, passes_total, passes_accurate, pass_accuracy_pct, attacks, dangerous_attacks,
         raw_json)
        VALUES
        (:source_match_id, :team, :opponent, :date, :xg, :possession_pct, :total_shots, :shots_on_target,
         :shots_off_target, :blocked_shots, :corners, :fouls, :offsides, :saves, :yellow_cards, :red_cards,
         :cards_estimate, :passes_total, :passes_accurate, :pass_accuracy_pct, :attacks, :dangerous_attacks,
         :raw_json)
        ON CONFLICT(source_match_id, team) DO UPDATE SET
            opponent = excluded.opponent,
            date = excluded.date,
            xg = excluded.xg,
            possession_pct = excluded.possession_pct,
            total_shots = excluded.total_shots,
            shots_on_target = excluded.shots_on_target,
            shots_off_target = excluded.shots_off_target,
            blocked_shots = excluded.blocked_shots,
            corners = excluded.corners,
            fouls = excluded.fouls,
            offsides = excluded.offsides,
            saves = excluded.saves,
            yellow_cards = excluded.yellow_cards,
            red_cards = excluded.red_cards,
            cards_estimate = excluded.cards_estimate,
            passes_total = excluded.passes_total,
            passes_accurate = excluded.passes_accurate,
            pass_accuracy_pct = excluded.pass_accuracy_pct,
            attacks = excluded.attacks,
            dangerous_attacks = excluded.dangerous_attacks,
            raw_json = excluded.raw_json
        """,
        {**stats, "raw_json": json.dumps(stats.get("raw", {}))},
    )


def upsert_match_context(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO match_context
        (match_key, date, home_team, away_team, venue_id, stadium_name, city, country,
         neutral, competition_weight, stage, raw_json)
        VALUES
        (:match_key, :date, :home_team, :away_team, :venue_id, :stadium_name, :city, :country,
         :neutral, :competition_weight, :stage, :raw_json)
        ON CONFLICT(match_key) DO UPDATE SET
            date = excluded.date,
            home_team = excluded.home_team,
            away_team = excluded.away_team,
            venue_id = excluded.venue_id,
            stadium_name = excluded.stadium_name,
            city = excluded.city,
            country = excluded.country,
            neutral = excluded.neutral,
            competition_weight = excluded.competition_weight,
            stage = excluded.stage,
            raw_json = excluded.raw_json
        """,
        {**row, "raw_json": json.dumps(row.get("raw", {}))},
    )


def get_match_context(
    conn: sqlite3.Connection,
    home_team: str,
    away_team: str,
    match_date: str | None = None,
) -> dict | None:
    clauses = ["lower(home_team) = lower(?)", "lower(away_team) = lower(?)"]
    params: list[str] = [home_team, away_team]
    if match_date:
        clauses.append("date = ?")
        params.append(match_date)
    where_sql = " AND ".join(clauses)
    row = conn.execute(
        f"""
        SELECT *
        FROM match_context
        WHERE {where_sql}
        ORDER BY
            CASE WHEN date >= date('now') THEN 0 ELSE 1 END,
            ABS(julianday(COALESCE(date, date('now'))) - julianday(date('now'))) ASC,
            id DESC
        LIMIT 1
        """,
        tuple(params),
    ).fetchone()
    if not row:
        return None
    data = dict(row)
    raw = json.loads(data.get("raw_json") or "{}")
    context = raw.get("context") if isinstance(raw, dict) else None
    return {**data, **(context or {}), "raw": raw}


def upsert_api_football_fixture(conn: sqlite3.Connection, fixture: dict) -> None:
    conn.execute(
        """
        INSERT INTO api_football_fixtures
        (fixture_id, date, league_id, league_name, season, home_team, away_team,
         home_team_id, away_team_id, status_short, venue_name, venue_city, raw_json, updated_at)
        VALUES
        (:fixture_id, :date, :league_id, :league_name, :season, :home_team, :away_team,
         :home_team_id, :away_team_id, :status_short, :venue_name, :venue_city, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(fixture_id) DO UPDATE SET
            date = excluded.date,
            league_id = excluded.league_id,
            league_name = excluded.league_name,
            season = excluded.season,
            home_team = excluded.home_team,
            away_team = excluded.away_team,
            home_team_id = excluded.home_team_id,
            away_team_id = excluded.away_team_id,
            status_short = excluded.status_short,
            venue_name = excluded.venue_name,
            venue_city = excluded.venue_city,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {**fixture, "raw_json": json.dumps(fixture.get("raw", {}))},
    )


def upsert_api_football_team_season(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO api_football_team_seasons
        (league_id, season, team_id, team_name, country, founded, national,
         venue_name, venue_city, raw_json, updated_at)
        VALUES
        (:league_id, :season, :team_id, :team_name, :country, :founded, :national,
         :venue_name, :venue_city, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(league_id, season, team_id) DO UPDATE SET
            team_name = excluded.team_name,
            country = excluded.country,
            founded = excluded.founded,
            national = excluded.national,
            venue_name = excluded.venue_name,
            venue_city = excluded.venue_city,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {**row, "raw_json": json.dumps(row.get("raw", {}))},
    )


def upsert_api_football_team_statistics(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO api_football_team_statistics
        (league_id, league_name, season, team_id, team_name, played, wins, draws, losses,
         goals_for, goals_against, home_played, home_wins, away_played, away_wins,
         clean_sheets, failed_to_score, form, raw_json, updated_at)
        VALUES
        (:league_id, :league_name, :season, :team_id, :team_name, :played, :wins, :draws, :losses,
         :goals_for, :goals_against, :home_played, :home_wins, :away_played, :away_wins,
         :clean_sheets, :failed_to_score, :form, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(league_id, season, team_id) DO UPDATE SET
            league_name = excluded.league_name,
            team_name = excluded.team_name,
            played = excluded.played,
            wins = excluded.wins,
            draws = excluded.draws,
            losses = excluded.losses,
            goals_for = excluded.goals_for,
            goals_against = excluded.goals_against,
            home_played = excluded.home_played,
            home_wins = excluded.home_wins,
            away_played = excluded.away_played,
            away_wins = excluded.away_wins,
            clean_sheets = excluded.clean_sheets,
            failed_to_score = excluded.failed_to_score,
            form = excluded.form,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {**row, "raw_json": json.dumps(row.get("raw", {}))},
    )


def upsert_api_football_standing(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO api_football_standings
        (league_id, league_name, season, group_name, rank, team_id, team_name, points,
         goals_diff, form, status, description, played, wins, draws, losses,
         goals_for, goals_against, raw_json, updated_at)
        VALUES
        (:league_id, :league_name, :season, :group_name, :rank, :team_id, :team_name, :points,
         :goals_diff, :form, :status, :description, :played, :wins, :draws, :losses,
         :goals_for, :goals_against, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(league_id, season, group_name, team_id) DO UPDATE SET
            league_name = excluded.league_name,
            rank = excluded.rank,
            team_name = excluded.team_name,
            points = excluded.points,
            goals_diff = excluded.goals_diff,
            form = excluded.form,
            status = excluded.status,
            description = excluded.description,
            played = excluded.played,
            wins = excluded.wins,
            draws = excluded.draws,
            losses = excluded.losses,
            goals_for = excluded.goals_for,
            goals_against = excluded.goals_against,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {**row, "raw_json": json.dumps(row.get("raw", {}))},
    )


def upsert_api_football_fixture_lineup(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO api_football_fixture_lineups
        (fixture_id, team_id, team_name, formation, coach_id, coach_name, raw_json, updated_at)
        VALUES
        (:fixture_id, :team_id, :team_name, :formation, :coach_id, :coach_name, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(fixture_id, team_id) DO UPDATE SET
            team_name = excluded.team_name,
            formation = excluded.formation,
            coach_id = excluded.coach_id,
            coach_name = excluded.coach_name,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {**row, "raw_json": json.dumps(row.get("raw", {}))},
    )


def upsert_api_football_fixture_player(conn: sqlite3.Connection, row: dict) -> None:
    internal_player_id = upsert_player(
        conn,
        {
            "api_player_id": row.get("player_id"),
            "name": row.get("player_name"),
            "preferred_position": row.get("position"),
            "raw": row.get("raw", {}),
        },
    )
    upsert_lineup_entry(
        conn,
        {
            "match_id": None,
            "fixture_id": row.get("fixture_id"),
            "team_id": None,
            "api_team_id": row.get("team_id"),
            "team_name": row.get("team_name"),
            "player_id": internal_player_id,
            "api_player_id": row.get("player_id"),
            "role": "starter" if int(row.get("is_starting") or 0) else "substitute",
            "position": row.get("position"),
            "formation_position": row.get("grid"),
            "shirt_number": row.get("number"),
            "source": "api_football",
            "raw": row.get("raw", {}),
        },
    )
    conn.execute(
        """
        INSERT INTO api_football_fixture_players
        (fixture_id, team_id, team_name, player_id, player_name, number, position, grid, is_starting, raw_json, updated_at)
        VALUES
        (:fixture_id, :team_id, :team_name, :player_id, :player_name, :number, :position, :grid, :is_starting, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(fixture_id, team_id, player_id, player_name, is_starting) DO UPDATE SET
            team_name = excluded.team_name,
            number = excluded.number,
            position = excluded.position,
            grid = excluded.grid,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {**row, "raw_json": json.dumps(row.get("raw", {}))},
    )


def upsert_sync_inventory(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO sync_inventory
        (provider, entity_type, league_id, league_name, season, data_type, status,
         records_count, expected_count, request_cost, last_synced_at, expires_at,
         error_message, raw_json, updated_at)
        VALUES
        (:provider, :entity_type, :league_id, :league_name, :season, :data_type, :status,
         :records_count, :expected_count, :request_cost, CURRENT_TIMESTAMP, :expires_at,
         :error_message, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(provider, entity_type, league_id, season, data_type) DO UPDATE SET
            league_name = excluded.league_name,
            status = excluded.status,
            records_count = excluded.records_count,
            expected_count = excluded.expected_count,
            request_cost = excluded.request_cost,
            last_synced_at = excluded.last_synced_at,
            expires_at = excluded.expires_at,
            error_message = excluded.error_message,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {**row, "raw_json": json.dumps(row.get("raw", {}))},
    )


def get_sync_inventory(
    conn: sqlite3.Connection,
    provider: str,
    entity_type: str,
    league_id: int,
    season: int,
    data_type: str,
) -> dict | None:
    row = conn.execute(
        """
        SELECT *
        FROM sync_inventory
        WHERE provider = ?
          AND entity_type = ?
          AND league_id = ?
          AND season = ?
          AND data_type = ?
        """,
        (provider, entity_type, league_id, season, data_type),
    ).fetchone()
    return dict(row) if row else None


def list_sync_inventory(conn: sqlite3.Connection, league_id: int | None = None, season: int | None = None) -> list[dict]:
    clauses = []
    params: list[int] = []
    if league_id is not None:
        clauses.append("league_id = ?")
        params.append(league_id)
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM sync_inventory
        {where_sql}
        ORDER BY league_name, season DESC, data_type
        """,
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_player_availability(conn: sqlite3.Connection, row: dict) -> None:
    internal_player_id = row.get("player_id")
    if internal_player_id is None and row.get("player_name"):
        internal_player_id = upsert_player(
            conn,
            {
                "api_player_id": row.get("api_player_id"),
                "name": row.get("player_name"),
                "raw": row.get("raw", {}),
            },
        )
    conn.execute(
        """
        INSERT OR IGNORE INTO player_availability
        (source, fixture_id, player_id, api_player_id, team_id, api_team_id, team, player_name,
         reason, status, expected_return, importance_score, raw_json)
        VALUES
        (:source, :fixture_id, :player_id, :api_player_id, :team_id, :api_team_id, :team, :player_name,
         :reason, :status, :expected_return, :importance_score, :raw_json)
        """,
        {
            **row,
            "player_id": internal_player_id,
            "team_id": row.get("team_id") or _get_or_create_team_id(conn, row.get("team")),
            "api_player_id": row.get("api_player_id"),
            "api_team_id": row.get("api_team_id"),
            "expected_return": row.get("expected_return"),
            "importance_score": row.get("importance_score"),
            "raw_json": json.dumps(row.get("raw", {})),
        },
    )


def advanced_summary_by_team(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT team,
               AVG(xg) AS xg_for,
               AVG(opp_xg) AS xg_against,
               AVG(total_shots) AS shots_for,
               AVG(opp_total_shots) AS shots_against,
               AVG(shots_on_target) AS shots_on_target_for,
               AVG(opp_shots_on_target) AS shots_on_target_against,
               AVG(corners) AS corners_for,
               AVG(opp_corners) AS corners_against,
               AVG(fouls) AS fouls_for,
               AVG(cards_estimate) AS cards_for,
               AVG(possession_pct) AS possession_for,
               AVG(pass_accuracy_pct) AS pass_accuracy_for,
               AVG(dangerous_attacks) AS dangerous_attacks_for,
               COUNT(*) AS advanced_matches
        FROM (
            SELECT mine.*,
                   opp.xg AS opp_xg,
                   opp.total_shots AS opp_total_shots,
                   opp.shots_on_target AS opp_shots_on_target,
                   opp.corners AS opp_corners
            FROM match_team_advanced_stats mine
            LEFT JOIN match_team_advanced_stats opp
              ON opp.source_match_id = mine.source_match_id
             AND opp.team = mine.opponent
        )
        GROUP BY team
        """
    ).fetchall()
    return {str(row["team"]): dict(row) for row in rows}


def availability_summary_by_team(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT pa.team,
               COUNT(*) AS unavailable_players,
               AVG(COALESCE(pa.importance_score, pss.rating, 6.0)) AS unavailable_avg_importance,
               SUM(
                   CASE
                       WHEN pa.importance_score IS NOT NULL THEN pa.importance_score
                       WHEN pss.rating IS NOT NULL THEN MAX(0, pss.rating - 5.5)
                       ELSE 1.0
                   END
               ) AS missing_player_importance,
               SUM(COALESCE(pss.goals, 0)) AS unavailable_goals,
               SUM(COALESCE(pss.assists, 0)) AS unavailable_assists,
               SUM(COALESCE(pss.minutes, 0)) AS unavailable_minutes
        FROM player_availability pa
        LEFT JOIN player_season_stats pss
          ON pss.player_id = pa.player_id
         AND lower(pss.team_name) = lower(pa.team)
        GROUP BY pa.team
        """
    ).fetchall()
    return {str(row["team"]): dict(row) for row in rows}


def api_team_strength_summary_by_team(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT team_name,
               played AS api_played,
               wins AS api_wins,
               draws AS api_draws,
               losses AS api_losses,
               goals_for AS api_goals_for,
               goals_against AS api_goals_against,
               home_played AS api_home_played,
               home_wins AS api_home_wins,
               away_played AS api_away_played,
               away_wins AS api_away_wins,
               clean_sheets AS api_clean_sheets,
               failed_to_score AS api_failed_to_score,
               form AS api_form
        FROM api_football_team_statistics
        WHERE team_name IS NOT NULL
        """
    ).fetchall()
    output: dict[str, dict] = {}
    for row in rows:
        data = dict(row)
        played = float(data.get("api_played") or 0)
        wins = float(data.get("api_wins") or 0)
        draws = float(data.get("api_draws") or 0)
        goals_for = float(data.get("api_goals_for") or 0)
        goals_against = float(data.get("api_goals_against") or 0)
        data["api_win_rate"] = wins / played if played else None
        data["api_points_per_match"] = (wins * 3 + draws) / played if played else None
        data["api_goal_diff_per_match"] = (goals_for - goals_against) / played if played else None
        data["api_goals_for_per_match"] = goals_for / played if played else None
        data["api_goals_against_per_match"] = goals_against / played if played else None
        data["api_form_score"] = _form_score(data.get("api_form"))
        output[str(data["team_name"])] = data
    return output


def api_standings_summary_by_team(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT team_name,
               MIN(rank) AS standing_rank,
               MAX(points) AS standing_points,
               MAX(goals_diff) AS standing_goals_diff,
               MAX(played) AS standing_played,
               MAX(form) AS standing_form,
               MAX(description) AS standing_description
        FROM api_football_standings
        WHERE team_name IS NOT NULL
        GROUP BY team_name
        """
    ).fetchall()
    output: dict[str, dict] = {}
    for row in rows:
        data = dict(row)
        played = float(data.get("standing_played") or 0)
        points = float(data.get("standing_points") or 0)
        data["standing_points_per_match"] = points / played if played else None
        data["standing_form_score"] = _form_score(data.get("standing_form"))
        output[str(data["team_name"])] = data
    return output


def api_lineup_summary_by_team(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT team_name,
               COUNT(DISTINCT fixture_id) AS lineup_fixtures,
               COUNT(DISTINCT player_name) AS lineup_distinct_players,
               SUM(CASE WHEN is_starting = 1 THEN 1 ELSE 0 END) AS lineup_starters,
               SUM(CASE WHEN is_starting = 0 THEN 1 ELSE 0 END) AS lineup_substitutes
        FROM api_football_fixture_players
        WHERE team_name IS NOT NULL
        GROUP BY team_name
        """
    ).fetchall()
    return {str(row["team_name"]): dict(row) for row in rows}


def player_quality_summary_by_team(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT team_name,
               COUNT(DISTINCT player_id) AS squad_players_tracked,
               AVG(rating) AS player_avg_rating,
               AVG(CASE WHEN minutes IS NOT NULL THEN minutes END) AS player_avg_minutes,
               SUM(COALESCE(minutes, 0)) AS player_total_minutes,
               SUM(COALESCE(goals, 0)) AS player_total_goals,
               SUM(COALESCE(assists, 0)) AS player_total_assists,
               SUM(COALESCE(shots_on_target, 0)) AS player_total_shots_on_target,
               SUM(COALESCE(yellow_cards, 0)) AS player_total_yellow_cards,
               SUM(COALESCE(red_cards, 0)) AS player_total_red_cards,
               SUM(CASE WHEN COALESCE(minutes, 0) >= 900 THEN 1 ELSE 0 END) AS regular_players,
               SUM(CASE WHEN COALESCE(rating, 0) >= 7.0 THEN 1 ELSE 0 END) AS high_rating_players
        FROM player_season_stats
        WHERE team_name IS NOT NULL
        GROUP BY team_name
        """
    ).fetchall()
    output: dict[str, dict] = {}
    for row in rows:
        data = dict(row)
        tracked = float(data.get("squad_players_tracked") or 0)
        regular = float(data.get("regular_players") or 0)
        minutes = float(data.get("player_total_minutes") or 0)
        goals = float(data.get("player_total_goals") or 0)
        assists = float(data.get("player_total_assists") or 0)
        rating = data.get("player_avg_rating")
        data["player_rating_score"] = clamp_db(35 + float(rating) * 7.5) if rating is not None else None
        data["player_depth_score"] = clamp_db(30 + tracked * 2.2 + regular * 1.4) if tracked else None
        data["player_attack_output_score"] = clamp_db(42 + (goals + assists * 0.7) / max(tracked, 1) * 8) if tracked else None
        data["player_minutes_per_player"] = minutes / tracked if tracked else None
        output[str(data["team_name"])] = data
    return output


def clamp_db(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _form_score(form: str | None) -> float | None:
    if not form:
        return None
    values = {"W": 3.0, "D": 1.0, "L": 0.0}
    chars = [char for char in str(form).upper() if char in values]
    if not chars:
        return None
    weights = list(range(1, len(chars) + 1))
    weighted = sum(values[char] * weight for char, weight in zip(chars, weights)) / sum(weights)
    return round((weighted / 3.0) * 100, 3)


def upsert_prediction_backtest(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO prediction_backtests
        (match_date, competition, home_team, away_team, model_version,
         predicted_home_prob, predicted_draw_prob, predicted_away_prob, predicted_pick,
         predicted_scores_json, predicted_corners, predicted_shots_on_target, predicted_cards,
         predicted_over_2_5_prob, predicted_btts_prob,
         actual_home_goals, actual_away_goals, actual_corners, actual_shots_on_target,
         actual_cards, source, notes, snapshot_json)
        VALUES
        (:match_date, :competition, :home_team, :away_team, :model_version,
         :predicted_home_prob, :predicted_draw_prob, :predicted_away_prob, :predicted_pick,
         :predicted_scores_json, :predicted_corners, :predicted_shots_on_target, :predicted_cards,
         :predicted_over_2_5_prob, :predicted_btts_prob,
         :actual_home_goals, :actual_away_goals, :actual_corners, :actual_shots_on_target,
         :actual_cards, :source, :notes, :snapshot_json)
        ON CONFLICT(match_date, home_team, away_team, model_version) DO UPDATE SET
            competition = excluded.competition,
            predicted_home_prob = excluded.predicted_home_prob,
            predicted_draw_prob = excluded.predicted_draw_prob,
            predicted_away_prob = excluded.predicted_away_prob,
            predicted_pick = excluded.predicted_pick,
            predicted_scores_json = excluded.predicted_scores_json,
            predicted_corners = excluded.predicted_corners,
            predicted_shots_on_target = excluded.predicted_shots_on_target,
            predicted_cards = excluded.predicted_cards,
            predicted_over_2_5_prob = excluded.predicted_over_2_5_prob,
            predicted_btts_prob = excluded.predicted_btts_prob,
            actual_home_goals = excluded.actual_home_goals,
            actual_away_goals = excluded.actual_away_goals,
            actual_corners = excluded.actual_corners,
            actual_shots_on_target = excluded.actual_shots_on_target,
            actual_cards = excluded.actual_cards,
            source = excluded.source,
            notes = excluded.notes,
            snapshot_json = excluded.snapshot_json
        """,
        {**row, "snapshot_json": json.dumps(row.get("snapshot", {})) if row.get("snapshot") is not None else row.get("snapshot_json")},
    )


def list_prediction_backtests(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    sql = """
        SELECT *
        FROM prediction_backtests
        ORDER BY match_date DESC, id DESC
    """
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def list_prediction_backtest_summaries(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    sql = """
        SELECT id, match_date, competition, home_team, away_team, model_version,
               predicted_home_prob, predicted_draw_prob, predicted_away_prob, predicted_pick,
               predicted_scores_json, predicted_corners, predicted_shots_on_target, predicted_cards,
               predicted_over_2_5_prob, predicted_btts_prob,
               actual_home_goals, actual_away_goals, actual_corners, actual_shots_on_target,
               actual_cards, source, notes, created_at
        FROM prediction_backtests
        ORDER BY match_date DESC, id DESC
    """
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def list_pending_prediction_backtests(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, match_date, competition, home_team, away_team, model_version,
               predicted_home_prob, predicted_draw_prob, predicted_away_prob, predicted_pick,
               predicted_scores_json, predicted_corners, predicted_shots_on_target, predicted_cards,
               predicted_over_2_5_prob, predicted_btts_prob,
               actual_home_goals, actual_away_goals, actual_corners, actual_shots_on_target,
               actual_cards, source, notes, created_at
        FROM prediction_backtests
        WHERE actual_home_goals IS NULL OR actual_away_goals IS NULL
        ORDER BY match_date ASC, id ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def delete_pending_prediction_backtest(conn: sqlite3.Connection, backtest_id: int) -> int:
    cursor = conn.execute(
        """
        DELETE FROM prediction_backtests
        WHERE id = ?
          AND (actual_home_goals IS NULL OR actual_away_goals IS NULL)
        """,
        (backtest_id,),
    )
    return int(cursor.rowcount)


def count_matches_shadowed_by_api_football(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM matches m
            WHERE EXISTS (
                SELECT 1
                FROM api_football_fixtures f
                WHERE m.date = substr(f.date, 1, 10)
                  AND lower(m.home_team) = lower(f.home_team)
                  AND lower(m.away_team) = lower(f.away_team)
                  AND lower(m.competition) = lower(f.league_name)
                  AND CAST(m.season AS TEXT) = CAST(f.season AS TEXT)
            )
            """
        ).fetchone()[0]
    )


def delete_matches_shadowed_by_api_football(conn: sqlite3.Connection) -> int:
    before = conn.total_changes
    conn.execute(
        """
        DELETE FROM matches
        WHERE id IN (
            SELECT m.id
            FROM matches m
            JOIN api_football_fixtures f
              ON m.date = substr(f.date, 1, 10)
             AND lower(m.home_team) = lower(f.home_team)
             AND lower(m.away_team) = lower(f.away_team)
             AND lower(m.competition) = lower(f.league_name)
             AND CAST(m.season AS TEXT) = CAST(f.season AS TEXT)
        )
        """
    )
    return conn.total_changes - before


def update_prediction_backtest_result(conn: sqlite3.Connection, backtest_id: int, result: dict) -> None:
    conn.execute(
        """
        UPDATE prediction_backtests
        SET actual_home_goals = :actual_home_goals,
            actual_away_goals = :actual_away_goals,
            actual_corners = :actual_corners,
            actual_shots_on_target = :actual_shots_on_target,
            actual_cards = :actual_cards,
            notes = COALESCE(:notes, notes)
        WHERE id = :id
        """,
        {**result, "id": backtest_id},
    )


def upsert_competition(conn: sqlite3.Connection, competition: dict, season: int | None = None) -> None:
    conn.execute(
        """
        INSERT INTO competitions
        (name, region, country, api_football_league_id, season, enabled, priority, status, updated_at)
        VALUES
        (:name, :region, :country, :api_football_league_id, :season, :enabled, :priority, :status, CURRENT_TIMESTAMP)
        ON CONFLICT(name, season) DO UPDATE SET
            region = excluded.region,
            country = excluded.country,
            api_football_league_id = excluded.api_football_league_id,
            enabled = excluded.enabled,
            priority = excluded.priority,
            status = excluded.status,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            "name": competition["name"],
            "region": competition["region"],
            "country": competition.get("country"),
            "api_football_league_id": competition.get("api_football_league_id"),
            "season": season,
            "enabled": int(competition.get("enabled", 1)),
            "priority": competition.get("priority", 3),
            "status": competition.get("status", "planned"),
        },
    )


def list_competitions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT *
        FROM competitions
        ORDER BY enabled DESC, priority ASC, region, name
        """
    ).fetchall()
    return [dict(row) for row in rows]


def insert_daily_job(conn: sqlite3.Connection, job: dict) -> int:
    cursor = conn.execute(
        """
        INSERT INTO daily_jobs
        (job_date, competition, status, fixtures_found, predictions_created, results_updated, errors_json)
        VALUES
        (:job_date, :competition, :status, :fixtures_found, :predictions_created, :results_updated, :errors_json)
        """,
        {**job, "errors_json": json.dumps(job.get("errors", []))},
    )
    return int(cursor.lastrowid)


def upsert_team_alias(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO team_name_aliases
        (canonical_name, alias, source, confidence, updated_at)
        VALUES (:canonical_name, :alias, :source, :confidence, CURRENT_TIMESTAMP)
        ON CONFLICT(alias) DO UPDATE SET
            canonical_name = excluded.canonical_name,
            source = excluded.source,
            confidence = excluded.confidence,
            updated_at = CURRENT_TIMESTAMP
        """,
        row,
    )


def upsert_data_source_requirement(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO data_source_requirements
        (category, label, source_name, endpoint, status, priority, unlocks_score, notes)
        VALUES (:category, :label, :source_name, :endpoint, :status, :priority, :unlocks_score, :notes)
        ON CONFLICT(category, label, source_name) DO UPDATE SET
            endpoint = excluded.endpoint,
            status = excluded.status,
            priority = excluded.priority,
            unlocks_score = excluded.unlocks_score,
            notes = excluded.notes
        """,
        {
            **row,
            "notes": row.get("notes"),
        },
    )


def list_data_source_requirements(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT *
        FROM data_source_requirements
        ORDER BY priority ASC, category, label
        """
    ).fetchall()
    return [dict(row) for row in rows]


def alias_summary(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*) AS aliases,
               COUNT(DISTINCT canonical_name) AS canonical_teams
        FROM team_name_aliases
        """
    ).fetchone()
    return dict(row)


def list_venues(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, stadium_name, city, country, capacity, latitude, longitude, altitude_m, surface, roof
        FROM venues
        ORDER BY country, city, stadium_name
        """
    ).fetchall()
    return [dict(row) for row in rows]


def get_venue(conn: sqlite3.Connection, venue_id: int | None) -> dict | None:
    if venue_id is None:
        return None
    row = conn.execute(
        """
        SELECT id, stadium_name, city, country, capacity, latitude, longitude, altitude_m, surface, roof
        FROM venues WHERE id = ?
        """,
        (venue_id,),
    ).fetchone()
    return dict(row) if row else None
