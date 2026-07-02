CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    country TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    competition TEXT NOT NULL,
    season TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, home_team, away_team, competition, season)
);

CREATE TABLE IF NOT EXISTS team_match_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team TEXT NOT NULL,
    is_home INTEGER NOT NULL,
    goals_for INTEGER NOT NULL,
    goals_against INTEGER NOT NULL,
    shots INTEGER,
    shots_on_target INTEGER,
    possession REAL,
    FOREIGN KEY(match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    model_name TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    draw_prob REAL NOT NULL,
    away_win_prob REAL NOT NULL,
    most_likely_score TEXT NOT NULL,
    over_2_5_prob REAL NOT NULL,
    both_teams_score_prob REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    trained_until TEXT,
    metrics_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS team_external_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    source TEXT NOT NULL,
    fifa_rank INTEGER,
    fifa_points REAL,
    squad_value_eur REAL,
    squad_size INTEGER,
    avg_age REAL,
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team, source)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT,
    artifact_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS venues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT,
    stadium_name TEXT NOT NULL,
    city TEXT,
    country TEXT,
    capacity INTEGER,
    latitude REAL,
    longitude REAL,
    altitude_m REAL,
    surface TEXT,
    roof TEXT,
    raw_json TEXT,
    UNIQUE(stadium_name, city)
);

CREATE TABLE IF NOT EXISTS squad_players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT,
    team TEXT NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT,
    club_team TEXT,
    market_value_eur REAL,
    caps INTEGER,
    date_of_birth TEXT,
    height_cm REAL,
    goals INTEGER,
    raw_json TEXT,
    UNIQUE(team, player_name)
);

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_player_id INTEGER,
    name TEXT NOT NULL,
    firstname TEXT,
    lastname TEXT,
    birth_date TEXT,
    age INTEGER,
    nationality TEXT,
    height TEXT,
    weight TEXT,
    preferred_position TEXT,
    photo_url TEXT,
    raw_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(api_player_id),
    UNIQUE(name, birth_date, nationality)
);

CREATE TABLE IF NOT EXISTS team_squads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER,
    api_team_id INTEGER,
    team_name TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    competition_id INTEGER,
    competition_name TEXT,
    season INTEGER,
    squad_number INTEGER,
    position TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    joined_at TEXT,
    left_at TEXT,
    source TEXT NOT NULL DEFAULT 'api_football',
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(team_id) REFERENCES teams(id),
    FOREIGN KEY(player_id) REFERENCES players(id),
    UNIQUE(team_name, player_id, competition_id, season)
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER,
    api_team_id INTEGER,
    team_name TEXT NOT NULL,
    competition_id INTEGER,
    competition_name TEXT,
    season INTEGER NOT NULL,
    appearances INTEGER,
    lineups INTEGER,
    minutes INTEGER,
    goals INTEGER,
    assists INTEGER,
    shots INTEGER,
    shots_on_target INTEGER,
    passes INTEGER,
    key_passes INTEGER,
    pass_accuracy REAL,
    tackles INTEGER,
    interceptions INTEGER,
    duels_won INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    rating REAL,
    source TEXT NOT NULL DEFAULT 'api_football',
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(player_id) REFERENCES players(id),
    UNIQUE(player_id, team_name, competition_id, season, source)
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    fixture_id INTEGER,
    team_id INTEGER,
    api_team_id INTEGER,
    team_name TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    position TEXT,
    is_starter INTEGER NOT NULL DEFAULT 0,
    minutes INTEGER,
    goals INTEGER,
    assists INTEGER,
    shots INTEGER,
    shots_on_target INTEGER,
    passes INTEGER,
    key_passes INTEGER,
    tackles INTEGER,
    interceptions INTEGER,
    duels_won INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    rating REAL,
    source TEXT NOT NULL DEFAULT 'api_football',
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(id),
    FOREIGN KEY(player_id) REFERENCES players(id),
    UNIQUE(fixture_id, team_name, player_id, source)
);

CREATE TABLE IF NOT EXISTS lineups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    fixture_id INTEGER,
    team_id INTEGER,
    api_team_id INTEGER,
    team_name TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    api_player_id INTEGER,
    role TEXT NOT NULL,
    position TEXT,
    formation_position TEXT,
    shirt_number INTEGER,
    source TEXT NOT NULL DEFAULT 'api_football',
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(id),
    FOREIGN KEY(player_id) REFERENCES players(id),
    UNIQUE(fixture_id, team_name, player_id, role, source)
);

CREATE TABLE IF NOT EXISTS player_form_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    fixture_id INTEGER,
    team_id INTEGER,
    api_team_id INTEGER,
    team_name TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    minutes_last_5 INTEGER,
    goals_last_5 INTEGER,
    assists_last_5 INTEGER,
    rating_last_5 REAL,
    starts_last_5 INTEGER,
    fatigue_score REAL,
    form_score REAL,
    availability_score REAL,
    importance_score REAL,
    raw_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(id),
    FOREIGN KEY(player_id) REFERENCES players(id),
    UNIQUE(fixture_id, team_name, player_id)
);

CREATE TABLE IF NOT EXISTS match_team_advanced_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_match_id TEXT,
    team TEXT NOT NULL,
    opponent TEXT,
    date TEXT,
    xg REAL,
    possession_pct REAL,
    total_shots INTEGER,
    shots_on_target INTEGER,
    shots_off_target INTEGER,
    blocked_shots INTEGER,
    corners INTEGER,
    fouls INTEGER,
    offsides INTEGER,
    saves INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    cards_estimate REAL,
    passes_total INTEGER,
    passes_accurate INTEGER,
    pass_accuracy_pct REAL,
    attacks INTEGER,
    dangerous_attacks INTEGER,
    raw_json TEXT,
    UNIQUE(source_match_id, team)
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

CREATE TABLE IF NOT EXISTS match_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_key TEXT NOT NULL UNIQUE,
    date TEXT,
    home_team TEXT,
    away_team TEXT,
    venue_id INTEGER,
    stadium_name TEXT,
    city TEXT,
    country TEXT,
    neutral INTEGER,
    competition_weight REAL,
    stage TEXT,
    raw_json TEXT,
    FOREIGN KEY(venue_id) REFERENCES venues(id)
);

CREATE TABLE IF NOT EXISTS api_football_fixtures (
    fixture_id INTEGER PRIMARY KEY,
    date TEXT,
    league_id INTEGER,
    league_name TEXT,
    season INTEGER,
    home_team TEXT,
    away_team TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    status_short TEXT,
    venue_name TEXT,
    venue_city TEXT,
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_football_team_seasons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    country TEXT,
    founded INTEGER,
    national INTEGER,
    venue_name TEXT,
    venue_city TEXT,
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(league_id, season, team_id)
);

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

CREATE TABLE IF NOT EXISTS sync_inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    league_id INTEGER,
    league_name TEXT,
    season INTEGER,
    data_type TEXT NOT NULL,
    status TEXT NOT NULL,
    records_count INTEGER NOT NULL DEFAULT 0,
    expected_count INTEGER,
    request_cost INTEGER NOT NULL DEFAULT 0,
    last_synced_at TEXT,
    expires_at TEXT,
    error_message TEXT,
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, entity_type, league_id, season, data_type)
);

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

CREATE TABLE IF NOT EXISTS player_availability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    fixture_id INTEGER,
    player_id INTEGER,
    api_player_id INTEGER,
    team_id INTEGER,
    api_team_id INTEGER,
    team TEXT NOT NULL,
    player_name TEXT NOT NULL,
    reason TEXT,
    status TEXT,
    expected_return TEXT,
    importance_score REAL,
    raw_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(player_id) REFERENCES players(id),
    UNIQUE(source, fixture_id, team, player_name, reason)
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

CREATE TABLE IF NOT EXISTS prediction_backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT NOT NULL,
    competition TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_home_prob REAL,
    predicted_draw_prob REAL,
    predicted_away_prob REAL,
    predicted_pick TEXT,
    predicted_scores_json TEXT,
    predicted_corners REAL,
    predicted_shots_on_target REAL,
    predicted_cards REAL,
    predicted_over_2_5_prob REAL,
    predicted_btts_prob REAL,
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    actual_corners REAL,
    actual_shots_on_target REAL,
    actual_cards REAL,
    source TEXT,
    notes TEXT,
    snapshot_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_date, home_team, away_team, model_version)
);

CREATE INDEX IF NOT EXISTS idx_prediction_backtests_date_id ON prediction_backtests(match_date DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_backtests_pending ON prediction_backtests(actual_home_goals, actual_away_goals, match_date);
CREATE INDEX IF NOT EXISTS idx_prediction_backtests_match ON prediction_backtests(match_date, home_team, away_team);

CREATE TABLE IF NOT EXISTS competitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    country TEXT,
    api_football_league_id INTEGER,
    season INTEGER,
    enabled INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'planned',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, season)
);

CREATE TABLE IF NOT EXISTS daily_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_date TEXT NOT NULL,
    competition TEXT,
    status TEXT NOT NULL,
    fixtures_found INTEGER NOT NULL DEFAULT 0,
    predictions_created INTEGER NOT NULL DEFAULT 0,
    results_updated INTEGER NOT NULL DEFAULT 0,
    errors_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS team_name_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    alias TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL DEFAULT 'manual',
    confidence REAL NOT NULL DEFAULT 1.0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_source_requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    label TEXT NOT NULL,
    source_name TEXT NOT NULL,
    endpoint TEXT,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 3,
    unlocks_score REAL NOT NULL DEFAULT 0,
    notes TEXT,
    UNIQUE(category, label, source_name)
);
