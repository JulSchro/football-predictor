from pathlib import Path
import json
import re
import sqlite3
import unicodedata

from football_predictor.config import settings


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def normalize_competition_name(name: str | None, season: int | str | None = None) -> str | None:
    if name is None:
        return None
    value = str(name).strip()
    if value.lower() in {"world cup", "fifa world cup"} and str(season or "") == "2026":
        return "World Cup 2026"
    return value


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
    conn.execute(
        """
        UPDATE prediction_backtests
        SET competition = 'World Cup 2026'
        WHERE lower(competition) IN ('world cup', 'fifa world cup')
          AND substr(match_date, 1, 4) = '2026'
        """
    )
    conn.execute(
        """
        UPDATE api_football_fixtures
        SET league_name = 'World Cup 2026'
        WHERE lower(league_name) IN ('world cup', 'fifa world cup')
          AND CAST(season AS TEXT) = '2026'
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_football_league_coverage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id INTEGER NOT NULL,
            league_name TEXT,
            league_type TEXT,
            country TEXT,
            country_code TEXT,
            season INTEGER NOT NULL,
            current INTEGER,
            start_date TEXT,
            end_date TEXT,
            fixtures_events INTEGER,
            fixtures_lineups INTEGER,
            fixtures_statistics_fixtures INTEGER,
            fixtures_statistics_players INTEGER,
            standings INTEGER,
            players INTEGER,
            top_scorers INTEGER,
            top_assists INTEGER,
            top_cards INTEGER,
            injuries INTEGER,
            predictions INTEGER,
            odds INTEGER,
            raw_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(league_id, season)
        )
        """
    )
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

        CREATE TABLE IF NOT EXISTS fixture_detail_sync_status (
            fixture_id INTEGER PRIMARY KEY,
            status_short TEXT,
            last_checked TEXT,
            stats_completed INTEGER NOT NULL DEFAULT 0,
            events_completed INTEGER NOT NULL DEFAULT 0,
            lineups_completed INTEGER NOT NULL DEFAULT 0,
            player_stats_completed INTEGER NOT NULL DEFAULT 0,
            referee_completed INTEGER NOT NULL DEFAULT 0,
            snapshots_updated INTEGER NOT NULL DEFAULT 0,
            retry_after TEXT,
            error_message TEXT,
            raw_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS referees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            country TEXT,
            source TEXT NOT NULL DEFAULT 'api_football',
            raw_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, country, source)
        );

        CREATE TABLE IF NOT EXISTS referee_match_history (
            fixture_id INTEGER PRIMARY KEY,
            referee_id INTEGER NOT NULL,
            referee_name TEXT NOT NULL,
            referee_country TEXT,
            date TEXT,
            league_id INTEGER,
            league_name TEXT,
            season INTEGER,
            round TEXT,
            home_team TEXT,
            away_team TEXT,
            home_yellow INTEGER,
            away_yellow INTEGER,
            home_red INTEGER,
            away_red INTEGER,
            home_cards REAL,
            away_cards REAL,
            total_yellow INTEGER,
            total_red INTEGER,
            total_cards REAL,
            home_fouls INTEGER,
            away_fouls INTEGER,
            total_fouls INTEGER,
            penalties INTEGER,
            raw_events_json TEXT,
            raw_fixture_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(referee_id) REFERENCES referees(id)
        );

        CREATE TABLE IF NOT EXISTS referee_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            referee_id INTEGER NOT NULL,
            scope_type TEXT NOT NULL,
            scope_value TEXT NOT NULL,
            matches INTEGER NOT NULL,
            avg_yellow REAL,
            avg_red REAL,
            avg_cards REAL,
            avg_home_cards REAL,
            avg_away_cards REAL,
            home_away_diff REAL,
            avg_fouls REAL,
            penalty_rate REAL,
            std_cards REAL,
            p25_cards REAL,
            p50_cards REAL,
            p75_cards REAL,
            last5_avg_cards REAL,
            last10_avg_cards REAL,
            recent_trend REAL,
            consistency REAL,
            strictness_percentile REAL,
            strictness_label TEXT,
            home_bias_label TEXT,
            away_bias_label TEXT,
            profile_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(referee_id) REFERENCES referees(id),
            UNIQUE(referee_id, scope_type, scope_value)
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
        CREATE INDEX IF NOT EXISTS idx_fixture_detail_sync_retry ON fixture_detail_sync_status(stats_completed, retry_after);
        CREATE INDEX IF NOT EXISTS idx_prediction_backtests_date_id ON prediction_backtests(match_date DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_prediction_backtests_pending ON prediction_backtests(actual_home_goals, actual_away_goals, match_date);
        CREATE INDEX IF NOT EXISTS idx_prediction_backtests_match ON prediction_backtests(match_date, home_team, away_team);
        CREATE INDEX IF NOT EXISTS idx_match_context_match ON match_context(home_team, away_team, date);
        CREATE INDEX IF NOT EXISTS idx_advanced_stats_team_date ON match_team_advanced_stats(team, date);
        CREATE INDEX IF NOT EXISTS idx_advanced_stats_opponent ON match_team_advanced_stats(opponent);
        CREATE INDEX IF NOT EXISTS idx_referee_history_referee_date ON referee_match_history(referee_id, date);
        CREATE INDEX IF NOT EXISTS idx_referee_history_league_season ON referee_match_history(league_id, season);
        CREATE INDEX IF NOT EXISTS idx_referee_metrics_referee_scope ON referee_metrics(referee_id, scope_type, scope_value);
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


def _venue_key(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_value = re.sub(r"\([^)]*\)", " ", ascii_value)
    ascii_value = re.sub(r"(stadium|stadio|estadio|stadion)\b", " ", ascii_value)
    ascii_value = re.sub(r"\b(stadium|stadio|estadio|stadion|arena|park|field|the)\b", " ", ascii_value)
    ascii_value = re.sub(r"[^a-z0-9]+", " ", ascii_value)
    words = [word for word in ascii_value.split() if word not in {"de", "del", "do", "da", "dos", "das"}]
    return " ".join(words)


def _venue_aliases(value: str | None) -> list[str]:
    if not value:
        return []
    aliases = [value]
    aliases.extend(re.findall(r"\(([^)]+)\)", str(value)))
    cleaned = re.sub(r"\([^)]*\)", " ", str(value)).strip()
    if cleaned:
        aliases.append(cleaned)
    keys = []
    for alias in aliases:
        key = _venue_key(alias)
        if key and key not in keys:
            keys.append(key)
    return keys


def find_venue(conn: sqlite3.Connection, stadium_name: str | None, city: str | None = None, country: str | None = None) -> dict | None:
    if not stadium_name:
        return None
    clauses = ["stadium_name IS NOT NULL"]
    params: list[str] = []
    if city:
        clauses.append("(lower(city) = lower(?) OR city IS NULL)")
        params.append(city)
    if country:
        clauses.append("(lower(country) = lower(?) OR country IS NULL)")
        params.append(country)
    rows = conn.execute(
        f"""
        SELECT id, source_id, stadium_name, city, country, capacity, latitude, longitude, altitude_m, surface, roof
        FROM venues
        WHERE {" AND ".join(clauses)}
        """,
        tuple(params),
    ).fetchall()
    target_keys = _venue_aliases(stadium_name)
    city_key = _venue_key(city)
    best: tuple[int, dict] | None = None
    for row in rows:
        item = dict(row)
        venue_keys = _venue_aliases(item.get("stadium_name"))
        if not set(target_keys).intersection(venue_keys):
            continue
        score = 10
        if city_key and _venue_key(item.get("city")) == city_key:
            score += 5
        if item.get("capacity"):
            score += 1
        if best is None or score > best[0]:
            best = (score, item)
    return best[1] if best else None


def _best_venue_from_rows(venues: list[dict], stadium_name: str | None, city: str | None = None, country: str | None = None) -> dict | None:
    target_keys = _venue_aliases(stadium_name)
    if not target_keys:
        return None
    city_key = _venue_key(city)
    country_key = _venue_key(country)
    best: tuple[int, dict] | None = None
    for venue in venues:
        has_exact_key = bool(set(target_keys).intersection(venue["_keys"]))
        has_same_city = bool(city_key and venue["_city_key"] == city_key)
        has_safe_partial = has_same_city and any(
            len(target_key) >= 5
            and len(venue_key) >= 5
            and (target_key in venue_key or venue_key in target_key)
            for target_key in target_keys
            for venue_key in venue["_keys"]
        )
        if not has_exact_key and not has_safe_partial:
            continue
        score = 10
        if has_same_city:
            score += 5
        if country_key and venue["_country_key"] == country_key:
            score += 2
        if venue.get("capacity"):
            score += 1
        if best is None or score > best[0]:
            best = (score, venue)
    return best[1] if best else None


def backfill_match_context_venues(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        """
        SELECT id, stadium_name, city, country
        FROM match_context
        WHERE stadium_name IS NOT NULL
          AND stadium_name != ''
          AND venue_id IS NULL
        """
    ).fetchall()
    venue_rows = conn.execute(
        """
        SELECT id, source_id, stadium_name, city, country, capacity, latitude, longitude, altitude_m, surface, roof
        FROM venues
        WHERE stadium_name IS NOT NULL
        """
    ).fetchall()
    venues = []
    for row in venue_rows:
        item = dict(row)
        item["_keys"] = _venue_aliases(item.get("stadium_name"))
        item["_city_key"] = _venue_key(item.get("city"))
        item["_country_key"] = _venue_key(item.get("country"))
        venues.append(item)
    matched = 0
    unmatched = 0
    for row in rows:
        venue = _best_venue_from_rows(venues, row["stadium_name"], row["city"], row["country"])
        if not venue:
            unmatched += 1
            continue
        conn.execute(
            """
            UPDATE match_context
            SET venue_id = ?,
                stadium_name = COALESCE(?, stadium_name),
                city = COALESCE(?, city),
                country = COALESCE(?, country)
            WHERE id = ?
            """,
            (venue["id"], venue["stadium_name"], venue["city"], venue["country"], row["id"]),
        )
        matched += 1
    return {"contexts_checked": len(rows), "venues_loaded": len(venues), "matched": matched, "unmatched": unmatched}


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
    if existing is None and api_player_id is not None:
        existing = conn.execute("SELECT id FROM players WHERE api_player_id = ?", (api_player_id,)).fetchone()
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
            SET api_player_id = CASE
                    WHEN api_player_id IS NULL
                     AND :api_player_id IS NOT NULL
                     AND NOT EXISTS (SELECT 1 FROM players WHERE api_player_id = :api_player_id AND id != :id)
                    THEN :api_player_id
                    ELSE api_player_id
                END,
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
    try:
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
    except sqlite3.IntegrityError:
        existing = conn.execute(
            """
            SELECT id
            FROM players
            WHERE name = ?
              AND COALESCE(birth_date, '') = COALESCE(?, '')
              AND COALESCE(nationality, '') = COALESCE(?, '')
            """,
            (name, player.get("birth_date"), player.get("nationality")),
        ).fetchone()
        if existing:
            return int(existing["id"])
        raise
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
    payload = {
        **fixture,
        "league_name": normalize_competition_name(fixture.get("league_name"), fixture.get("season")),
    }
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
        {**payload, "raw_json": json.dumps(payload.get("raw", {}))},
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


def upsert_api_football_league_coverage(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO api_football_league_coverage (
            league_id, league_name, league_type, country, country_code, season, current,
            start_date, end_date, fixtures_events, fixtures_lineups,
            fixtures_statistics_fixtures, fixtures_statistics_players, standings, players,
            top_scorers, top_assists, top_cards, injuries, predictions, odds, raw_json,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(league_id, season) DO UPDATE SET
            league_name = excluded.league_name,
            league_type = excluded.league_type,
            country = excluded.country,
            country_code = excluded.country_code,
            current = excluded.current,
            start_date = excluded.start_date,
            end_date = excluded.end_date,
            fixtures_events = excluded.fixtures_events,
            fixtures_lineups = excluded.fixtures_lineups,
            fixtures_statistics_fixtures = excluded.fixtures_statistics_fixtures,
            fixtures_statistics_players = excluded.fixtures_statistics_players,
            standings = excluded.standings,
            players = excluded.players,
            top_scorers = excluded.top_scorers,
            top_assists = excluded.top_assists,
            top_cards = excluded.top_cards,
            injuries = excluded.injuries,
            predictions = excluded.predictions,
            odds = excluded.odds,
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            row.get("league_id"),
            row.get("league_name"),
            row.get("league_type"),
            row.get("country"),
            row.get("country_code"),
            row.get("season"),
            row.get("current"),
            row.get("start_date"),
            row.get("end_date"),
            row.get("fixtures_events"),
            row.get("fixtures_lineups"),
            row.get("fixtures_statistics_fixtures"),
            row.get("fixtures_statistics_players"),
            row.get("standings"),
            row.get("players"),
            row.get("top_scorers"),
            row.get("top_assists"),
            row.get("top_cards"),
            row.get("injuries"),
            row.get("predictions"),
            row.get("odds"),
            json.dumps(row.get("raw") or {}, ensure_ascii=False),
        ),
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
    payload = {
        **row,
        "competition": normalize_competition_name(row.get("competition"), str(row.get("match_date") or "")[:4]),
    }
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
            actual_home_goals = COALESCE(excluded.actual_home_goals, prediction_backtests.actual_home_goals),
            actual_away_goals = COALESCE(excluded.actual_away_goals, prediction_backtests.actual_away_goals),
            actual_corners = COALESCE(excluded.actual_corners, prediction_backtests.actual_corners),
            actual_shots_on_target = COALESCE(excluded.actual_shots_on_target, prediction_backtests.actual_shots_on_target),
            actual_cards = COALESCE(excluded.actual_cards, prediction_backtests.actual_cards),
            source = excluded.source,
            notes = excluded.notes,
            snapshot_json = excluded.snapshot_json
        """,
        {
            **payload,
            "snapshot_json": json.dumps(payload.get("snapshot", {})) if payload.get("snapshot") is not None else payload.get("snapshot_json"),
        },
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


def delete_prediction_backtest(conn: sqlite3.Connection, backtest_id: int) -> int:
    cursor = conn.execute(
        """
        DELETE FROM prediction_backtests
        WHERE id = ?
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
                  AND CAST(m.season AS TEXT) = CAST(f.season AS TEXT)
                  AND (
                    lower(m.competition) = lower(f.league_name)
                    OR (
                        CAST(m.season AS TEXT) = '2026'
                        AND lower(m.competition) IN ('world cup', 'fifa world cup', 'world cup 2026')
                        AND lower(f.league_name) IN ('world cup', 'fifa world cup', 'world cup 2026')
                    )
                  )
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
             AND CAST(m.season AS TEXT) = CAST(f.season AS TEXT)
             AND (
                lower(m.competition) = lower(f.league_name)
                OR (
                    CAST(m.season AS TEXT) = '2026'
                    AND lower(m.competition) IN ('world cup', 'fifa world cup', 'world cup 2026')
                    AND lower(f.league_name) IN ('world cup', 'fifa world cup', 'world cup 2026')
                )
             )
        )
        """
    )
    return conn.total_changes - before


def actual_market_stats_for_fixture(conn: sqlite3.Connection, fixture_id: int) -> dict:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS team_rows,
            SUM(corners) AS actual_corners,
            SUM(shots_on_target) AS actual_shots_on_target,
            SUM(total_shots) AS actual_total_shots,
            SUM(fouls) AS actual_fouls,
            SUM(passes_total) AS actual_passes,
            SUM(
                CASE
                    WHEN cards_estimate IS NOT NULL THEN cards_estimate
                    ELSE COALESCE(yellow_cards, 0) + COALESCE(red_cards, 0) * 2
                END
            ) AS actual_cards
        FROM match_team_advanced_stats
        WHERE source_match_id = ?
        """,
        (f"api-football:{fixture_id}",),
    ).fetchone()
    empty = {
        "actual_corners": None,
        "actual_shots_on_target": None,
        "actual_total_shots": None,
        "actual_fouls": None,
        "actual_passes": None,
        "actual_cards": None,
        "complete": False,
    }
    if not row or int(row["team_rows"] or 0) < 2:
        return empty
    return {
        "actual_corners": row["actual_corners"],
        "actual_shots_on_target": row["actual_shots_on_target"],
        "actual_total_shots": row["actual_total_shots"],
        "actual_fouls": row["actual_fouls"],
        "actual_passes": row["actual_passes"],
        "actual_cards": row["actual_cards"],
        "complete": True,
    }


def _prediction_value_changed(current: object, new: object) -> bool:
    if current is None and new is None:
        return False
    if current is None or new is None:
        return True
    try:
        return abs(float(current) - float(new)) > 1e-9
    except (TypeError, ValueError):
        return current != new


def reconcile_prediction_backtests_for_fixture(conn: sqlite3.Connection, fixture_id: int) -> dict:
    fixture = conn.execute(
        """
        SELECT fixture_id, date, league_name, home_team, away_team, status_short, raw_json
        FROM api_football_fixtures
        WHERE fixture_id = ?
        """,
        (fixture_id,),
    ).fetchone()
    if not fixture:
        return {
            "fixture_id": fixture_id,
            "home_team": None,
            "away_team": None,
            "status": "missing_fixture",
            "goals": {"home": None, "away": None},
            "stats_complete": False,
            "snapshots": 0,
            "snapshots_updated": 0,
            "markets_updated": [],
            "missing_markets": ["fixture"],
            "reason": "fixture_not_found",
        }
    target = str(fixture["date"] or "")[:10]
    raw = json.loads(fixture["raw_json"] or "{}")
    goals = raw.get("goals") or {}
    if goals.get("home") is None or goals.get("away") is None:
        return {
            "fixture_id": int(fixture["fixture_id"]),
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "status": fixture["status_short"],
            "goals": {"home": None, "away": None},
            "stats_complete": False,
            "snapshots": 0,
            "snapshots_updated": 0,
            "markets_updated": [],
            "missing_markets": ["goals"],
            "reason": "goals_not_available",
        }
    market_stats = actual_market_stats_for_fixture(conn, int(fixture["fixture_id"]))
    candidate_rows = conn.execute(
        """
        SELECT id, actual_home_goals, actual_away_goals, actual_corners, actual_shots_on_target, actual_cards
             , home_team, away_team
        FROM prediction_backtests
        WHERE match_date = ?
        """,
        (target,),
    ).fetchall()
    aliases = _team_alias_map(conn)
    rows = [
        row
        for row in candidate_rows
        if _team_canonical(row["home_team"], aliases) == _team_canonical(fixture["home_team"], aliases)
        and _team_canonical(row["away_team"], aliases) == _team_canonical(fixture["away_team"], aliases)
    ]
    markets_updated: list[str] = []
    missing_markets: list[str] = []
    if market_stats["complete"]:
        for field, market_name in [
            ("actual_corners", "corners"),
            ("actual_shots_on_target", "shots_on_target"),
            ("actual_cards", "cards"),
        ]:
            if market_stats.get(field) is None:
                missing_markets.append(market_name)
            else:
                markets_updated.append(market_name)
    else:
        missing_markets.extend(["corners", "shots_on_target", "cards"])

    updated = 0
    for row in rows:
        result = {
            "actual_home_goals": int(goals["home"]),
            "actual_away_goals": int(goals["away"]),
            "actual_corners": row["actual_corners"],
            "actual_shots_on_target": row["actual_shots_on_target"],
            "actual_cards": row["actual_cards"],
            "notes": "Resultado actualizado; pendiente de estadisticas API-Football",
        }
        if market_stats["complete"]:
            result.update(
                {
                    "actual_corners": market_stats["actual_corners"],
                    "actual_shots_on_target": market_stats["actual_shots_on_target"],
                    "actual_cards": market_stats["actual_cards"],
                    "notes": "Resultado reconciliado con estadisticas API-Football",
                }
            )
        changed = any(
            _prediction_value_changed(row[field], result[field])
            for field in (
                "actual_home_goals",
                "actual_away_goals",
                "actual_corners",
                "actual_shots_on_target",
                "actual_cards",
            )
        )
        if changed:
            update_prediction_backtest_result(conn, int(row["id"]), result)
            updated += 1

    return {
        "fixture_id": int(fixture["fixture_id"]),
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "status": fixture["status_short"],
        "goals": {"home": goals.get("home"), "away": goals.get("away")},
        "stats_complete": bool(market_stats["complete"]),
        "corners": market_stats.get("actual_corners"),
        "shots_on_target": market_stats.get("actual_shots_on_target"),
        "total_shots": market_stats.get("actual_total_shots"),
        "fouls": market_stats.get("actual_fouls"),
        "passes": market_stats.get("actual_passes"),
        "cards": market_stats.get("actual_cards"),
        "snapshots": len(rows),
        "snapshots_updated": updated,
        "markets_updated": markets_updated,
        "missing_markets": missing_markets,
        "reason": None if market_stats["complete"] else "stats_pending",
    }


def reconcile_prediction_backtests_for_date(conn: sqlite3.Connection, target: str) -> dict:
    fixtures = conn.execute(
        """
        SELECT fixture_id
        FROM api_football_fixtures
        WHERE substr(date, 1, 10) = ?
          AND status_short IN ('FT', 'AET', 'PEN')
        """,
        (target,),
    ).fetchall()
    stats_pending = 0
    snapshots_checked = 0
    fixture_reports = []
    for fixture in fixtures:
        report = reconcile_prediction_backtests_for_fixture(conn, int(fixture["fixture_id"]))
        snapshots_checked += int(report.get("snapshots") or 0)
        stats_pending += 0 if report.get("stats_complete") else int(report.get("snapshots") or 0)
        fixture_reports.append(report)
    return {
        "date": target,
        "finished_fixtures": len(fixtures),
        "snapshots_checked": snapshots_checked,
        "results_updated": sum(int(report.get("snapshots_updated") or 0) for report in fixture_reports),
        "stats_pending_snapshots": stats_pending,
        "fixtures": fixture_reports,
    }


def _team_alias_map(conn: sqlite3.Connection) -> dict[str, str]:
    aliases: dict[str, str] = {}
    try:
        rows = conn.execute("SELECT canonical_name, alias FROM team_name_aliases").fetchall()
    except sqlite3.OperationalError:
        rows = []
    for row in rows:
        canonical = _normalize_team_name(row["canonical_name"])
        aliases[_normalize_team_name(row["canonical_name"])] = canonical
        aliases[_normalize_team_name(row["alias"])] = canonical
    return aliases


def _team_canonical(name: str | None, aliases: dict[str, str]) -> str:
    normalized = _normalize_team_name(name)
    return aliases.get(normalized, normalized)


def _normalize_team_name(name: str | None) -> str:
    value = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def update_prediction_backtest_result(conn: sqlite3.Connection, backtest_id: int, result: dict) -> None:
    conn.execute(
        """
        UPDATE prediction_backtests
        SET actual_home_goals = COALESCE(:actual_home_goals, actual_home_goals),
            actual_away_goals = COALESCE(:actual_away_goals, actual_away_goals),
            actual_corners = COALESCE(:actual_corners, actual_corners),
            actual_shots_on_target = COALESCE(:actual_shots_on_target, actual_shots_on_target),
            actual_cards = COALESCE(:actual_cards, actual_cards),
            notes = COALESCE(:notes, notes)
        WHERE id = :id
        """,
        {**result, "id": backtest_id},
    )


def upsert_referee(conn: sqlite3.Connection, referee: dict) -> int:
    conn.execute(
        """
        INSERT INTO referees
        (name, country, source, raw_json, updated_at)
        VALUES (:name, :country, :source, :raw_json, CURRENT_TIMESTAMP)
        ON CONFLICT(name, country, source) DO UPDATE SET
            raw_json = excluded.raw_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            "name": referee["name"],
            "country": referee.get("country"),
            "source": referee.get("source", "api_football"),
            "raw_json": json.dumps(referee.get("raw", {})),
        },
    )
    row = conn.execute(
        "SELECT id FROM referees WHERE name = ? AND COALESCE(country, '') = COALESCE(?, '') AND source = ?",
        (referee["name"], referee.get("country"), referee.get("source", "api_football")),
    ).fetchone()
    return int(row["id"])


def upsert_referee_match_history(conn: sqlite3.Connection, history: dict) -> None:
    referee_id = upsert_referee(
        conn,
        {
            "name": history["referee_name"],
            "country": history.get("referee_country"),
            "source": "api_football",
            "raw": {"fixture_id": history["fixture_id"]},
        },
    )
    conn.execute(
        """
        INSERT INTO referee_match_history
        (fixture_id, referee_id, referee_name, referee_country, date, league_id, league_name, season, round,
         home_team, away_team, home_yellow, away_yellow, home_red, away_red, home_cards, away_cards,
         total_yellow, total_red, total_cards, home_fouls, away_fouls, total_fouls, penalties,
         raw_events_json, raw_fixture_json, updated_at)
        VALUES
        (:fixture_id, :referee_id, :referee_name, :referee_country, :date, :league_id, :league_name, :season, :round,
         :home_team, :away_team, :home_yellow, :away_yellow, :home_red, :away_red, :home_cards, :away_cards,
         :total_yellow, :total_red, :total_cards, :home_fouls, :away_fouls, :total_fouls, :penalties,
         :raw_events_json, :raw_fixture_json, CURRENT_TIMESTAMP)
        ON CONFLICT(fixture_id) DO UPDATE SET
            referee_id = excluded.referee_id,
            referee_name = excluded.referee_name,
            referee_country = excluded.referee_country,
            date = excluded.date,
            league_id = excluded.league_id,
            league_name = excluded.league_name,
            season = excluded.season,
            round = excluded.round,
            home_team = excluded.home_team,
            away_team = excluded.away_team,
            home_yellow = excluded.home_yellow,
            away_yellow = excluded.away_yellow,
            home_red = excluded.home_red,
            away_red = excluded.away_red,
            home_cards = excluded.home_cards,
            away_cards = excluded.away_cards,
            total_yellow = excluded.total_yellow,
            total_red = excluded.total_red,
            total_cards = excluded.total_cards,
            home_fouls = excluded.home_fouls,
            away_fouls = excluded.away_fouls,
            total_fouls = excluded.total_fouls,
            penalties = excluded.penalties,
            raw_events_json = excluded.raw_events_json,
            raw_fixture_json = excluded.raw_fixture_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        {
            **history,
            "referee_id": referee_id,
            "raw_events_json": json.dumps(history.get("raw_events", {})),
            "raw_fixture_json": json.dumps(history.get("raw_fixture", {})),
        },
    )
    rebuild_referee_metrics(conn, referee_id)


def rebuild_referee_metrics(conn: sqlite3.Connection, referee_id: int) -> None:
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT *
            FROM referee_match_history
            WHERE referee_id = ? AND total_cards IS NOT NULL
            ORDER BY date
            """,
            (referee_id,),
        ).fetchall()
    ]
    if not rows:
        return
    scopes = [("global", "all", rows)]
    for league_id in sorted({row["league_id"] for row in rows if row.get("league_id") is not None}):
        scopes.append(("league", str(league_id), [row for row in rows if row.get("league_id") == league_id]))
    for season in sorted({row["season"] for row in rows if row.get("season") is not None}):
        scopes.append(("season", str(season), [row for row in rows if row.get("season") == season]))
    global_avgs = [
        float(row["avg_cards"])
        for row in conn.execute(
            """
            SELECT referee_id, AVG(total_cards) AS avg_cards, COUNT(*) AS matches
            FROM referee_match_history
            WHERE total_cards IS NOT NULL
            GROUP BY referee_id
            HAVING matches >= 1
            """
        ).fetchall()
        if row["avg_cards"] is not None
    ]
    for scope_type, scope_value, scope_rows in scopes:
        metrics = _referee_metrics_from_rows(scope_rows, global_avgs)
        conn.execute(
            """
            INSERT INTO referee_metrics
            (referee_id, scope_type, scope_value, matches, avg_yellow, avg_red, avg_cards, avg_home_cards,
             avg_away_cards, home_away_diff, avg_fouls, penalty_rate, std_cards, p25_cards, p50_cards,
             p75_cards, last5_avg_cards, last10_avg_cards, recent_trend, consistency, strictness_percentile,
             strictness_label, home_bias_label, away_bias_label, profile_json, updated_at)
            VALUES
            (:referee_id, :scope_type, :scope_value, :matches, :avg_yellow, :avg_red, :avg_cards, :avg_home_cards,
             :avg_away_cards, :home_away_diff, :avg_fouls, :penalty_rate, :std_cards, :p25_cards, :p50_cards,
             :p75_cards, :last5_avg_cards, :last10_avg_cards, :recent_trend, :consistency, :strictness_percentile,
             :strictness_label, :home_bias_label, :away_bias_label, :profile_json, CURRENT_TIMESTAMP)
            ON CONFLICT(referee_id, scope_type, scope_value) DO UPDATE SET
                matches = excluded.matches,
                avg_yellow = excluded.avg_yellow,
                avg_red = excluded.avg_red,
                avg_cards = excluded.avg_cards,
                avg_home_cards = excluded.avg_home_cards,
                avg_away_cards = excluded.avg_away_cards,
                home_away_diff = excluded.home_away_diff,
                avg_fouls = excluded.avg_fouls,
                penalty_rate = excluded.penalty_rate,
                std_cards = excluded.std_cards,
                p25_cards = excluded.p25_cards,
                p50_cards = excluded.p50_cards,
                p75_cards = excluded.p75_cards,
                last5_avg_cards = excluded.last5_avg_cards,
                last10_avg_cards = excluded.last10_avg_cards,
                recent_trend = excluded.recent_trend,
                consistency = excluded.consistency,
                strictness_percentile = excluded.strictness_percentile,
                strictness_label = excluded.strictness_label,
                home_bias_label = excluded.home_bias_label,
                away_bias_label = excluded.away_bias_label,
                profile_json = excluded.profile_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            {
                **metrics,
                "referee_id": referee_id,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "profile_json": json.dumps(metrics["profile"]),
            },
        )


def rebuild_all_referee_metrics(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT id FROM referees ORDER BY id").fetchall()
    for row in rows:
        rebuild_referee_metrics(conn, int(row["id"]))
    return len(rows)


def get_referee_profile_for_fixture(conn: sqlite3.Connection, fixture_id: int | None) -> dict | None:
    if fixture_id is None:
        return None
    row = conn.execute(
        """
        SELECT h.fixture_id, h.referee_id, h.referee_name, h.referee_country, h.league_id, h.league_name, h.season,
               m.*
        FROM referee_match_history h
        JOIN referee_metrics m ON m.referee_id = h.referee_id AND m.scope_type = 'global' AND m.scope_value = 'all'
        WHERE h.fixture_id = ?
        """,
        (fixture_id,),
    ).fetchone()
    if not row:
        fixture = conn.execute("SELECT raw_json FROM api_football_fixtures WHERE fixture_id = ?", (fixture_id,)).fetchone()
        if not fixture:
            return None
        raw = json.loads(fixture["raw_json"] or "{}")
        referee_name, referee_country = parse_referee_name((raw.get("fixture") or {}).get("referee"))
        if not referee_name:
            return None
        return {"referee_name": referee_name, "referee_country": referee_country, "matches": 0, "available": False}
    item = dict(row)
    item["profile"] = json.loads(item.get("profile_json") or "{}")
    item["available"] = True
    return item


def parse_referee_name(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return None, None
    return parts[0], parts[1] if len(parts) > 1 else None


def _referee_metrics_from_rows(rows: list[dict], global_avgs: list[float]) -> dict:
    cards = [float(row["total_cards"]) for row in rows if row.get("total_cards") is not None]
    yellows = [float(row["total_yellow"]) for row in rows if row.get("total_yellow") is not None]
    reds = [float(row["total_red"]) for row in rows if row.get("total_red") is not None]
    home_cards = [float(row["home_cards"]) for row in rows if row.get("home_cards") is not None]
    away_cards = [float(row["away_cards"]) for row in rows if row.get("away_cards") is not None]
    fouls = [float(row["total_fouls"]) for row in rows if row.get("total_fouls") is not None]
    penalties = [float(row["penalties"] or 0) for row in rows]
    avg_cards = _mean(cards)
    home_avg = _mean(home_cards)
    away_avg = _mean(away_cards)
    percentile = _percentile_rank(avg_cards, global_avgs)
    strictness = _strictness_label(percentile)
    home_diff = (home_avg or 0) - (away_avg or 0) if home_avg is not None and away_avg is not None else None
    metrics = {
        "matches": len(cards),
        "avg_yellow": _mean(yellows),
        "avg_red": _mean(reds),
        "avg_cards": avg_cards,
        "avg_home_cards": home_avg,
        "avg_away_cards": away_avg,
        "home_away_diff": home_diff,
        "avg_fouls": _mean(fouls),
        "penalty_rate": _mean(penalties),
        "std_cards": _std(cards),
        "p25_cards": _quantile(cards, 0.25),
        "p50_cards": _quantile(cards, 0.50),
        "p75_cards": _quantile(cards, 0.75),
        "last5_avg_cards": _mean(cards[-5:]),
        "last10_avg_cards": _mean(cards[-10:]),
        "recent_trend": (_mean(cards[-5:]) - _mean(cards[:-5])) if len(cards) > 5 and _mean(cards[:-5]) is not None else None,
        "consistency": _consistency(cards),
        "strictness_percentile": percentile,
        "strictness_label": strictness,
        "home_bias_label": _bias_label(home_diff, "local"),
        "away_bias_label": _bias_label(-(home_diff or 0), "visitante") if home_diff is not None else "sin muestra",
    }
    metrics["profile"] = {
        "classification": strictness,
        "card_distribution": {
            "p25": metrics["p25_cards"],
            "p50": metrics["p50_cards"],
            "p75": metrics["p75_cards"],
        },
        "recent": {
            "last5_avg_cards": metrics["last5_avg_cards"],
            "last10_avg_cards": metrics["last10_avg_cards"],
            "trend": metrics["recent_trend"],
        },
        "bias": {
            "home_away_diff": metrics["home_away_diff"],
            "home_label": metrics["home_bias_label"],
            "away_label": metrics["away_bias_label"],
        },
    }
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in metrics.items()}


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _std(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return 0.0 if clean else None
    mean = sum(clean) / len(clean)
    return (sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)) ** 0.5


def _quantile(values: list[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    pos = (len(clean) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(clean) - 1)
    frac = pos - lower
    return clean[lower] * (1 - frac) + clean[upper] * frac


def _consistency(values: list[float]) -> float | None:
    mean = _mean(values)
    std = _std(values)
    if mean is None or std is None:
        return None
    if mean == 0:
        return 1.0 if std == 0 else 0.0
    return max(0.0, min(1.0, 1 - (std / mean)))


def _percentile_rank(value: float | None, distribution: list[float]) -> float | None:
    clean = sorted(float(item) for item in distribution if item is not None)
    if value is None or not clean:
        return None
    below = sum(1 for item in clean if item <= value)
    return below / len(clean)


def _strictness_label(percentile: float | None) -> str:
    if percentile is None:
        return "sin muestra suficiente"
    if percentile <= 0.2:
        return "muy permisivo"
    if percentile <= 0.4:
        return "permisivo"
    if percentile <= 0.6:
        return "neutral"
    if percentile <= 0.8:
        return "estricto"
    return "muy estricto"


def _bias_label(diff: float | None, side: str) -> str:
    if diff is None:
        return "sin muestra"
    if diff > 0:
        return f"mas tarjetas al {side}"
    if diff < 0:
        return f"menos tarjetas al {side}"
    return "neutral"


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
