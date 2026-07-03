from __future__ import annotations

import sqlite3


REQUIRED_DATA_GROUPS = {
    "core_results": ["matches", "teams", "goals", "competition"],
    "team_strength": ["elo", "glicko_proxy", "own_ranking", "fifa_rank", "squad_value"],
    "recent_form": ["last_5", "last_10", "last_20", "trend", "home_away_split"],
    "players": ["squads", "minutes", "ratings", "injuries", "suspensions"],
    "advanced_stats": ["xg", "xga", "ppda", "field_tilt", "progressive_actions"],
    "context": ["stage", "must_win", "travel", "rest", "coach"],
    "environment": ["weather", "stadium", "referee", "odds", "closing_line"],
}


def data_coverage(conn: sqlite3.Connection) -> dict:
    matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    external = conn.execute("SELECT COUNT(*) FROM team_external_metrics").fetchone()[0]
    fifa = conn.execute("SELECT COUNT(*) FROM team_external_metrics WHERE fifa_rank IS NOT NULL").fetchone()[0]
    predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    experiments = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    players = conn.execute("SELECT COUNT(*) FROM squad_players").fetchone()[0]
    availability = conn.execute("SELECT COUNT(*) FROM player_availability").fetchone()[0]
    venues = conn.execute("SELECT COUNT(*) FROM venues").fetchone()[0]
    advanced = conn.execute("SELECT COUNT(*) FROM match_team_advanced_stats").fetchone()[0]
    market_stats = conn.execute(
        """
        SELECT
            SUM(CASE WHEN corners IS NOT NULL THEN 1 ELSE 0 END) AS corners,
            SUM(CASE WHEN shots_on_target IS NOT NULL THEN 1 ELSE 0 END) AS shots_on_target,
            SUM(CASE WHEN total_shots IS NOT NULL THEN 1 ELSE 0 END) AS total_shots,
            SUM(CASE WHEN cards_estimate IS NOT NULL OR yellow_cards IS NOT NULL THEN 1 ELSE 0 END) AS cards,
            SUM(CASE WHEN possession_pct IS NOT NULL THEN 1 ELSE 0 END) AS possession
        FROM match_team_advanced_stats
        """
    ).fetchone()
    api_fixtures = conn.execute("SELECT COUNT(*) FROM api_football_fixtures").fetchone()[0]
    api_team_stats = safe_count(conn, "api_football_team_statistics")
    api_standings = safe_count(conn, "api_football_standings")
    api_lineups = safe_count(conn, "api_football_fixture_lineups")
    api_lineup_players = safe_count(conn, "api_football_fixture_players")
    api_coverage_rows = safe_count(conn, "api_football_league_coverage")
    aliases = safe_count(conn, "team_name_aliases")
    requirements = source_readiness(conn)
    markets = market_coverage(conn)

    groups = {
        "core_results": coverage_item("Datos de partidos", matches > 0 or api_fixtures > 0, "real", matches + api_fixtures),
        "team_strength": coverage_item("Fuerza de equipos", (matches > 0 and fifa > 0) or api_team_stats > 0, "real", fifa + api_team_stats),
        "recent_form": coverage_item("Forma reciente", matches > 0, "real", matches),
        "players": coverage_item("Jugadores", players > 0 or availability > 0 or api_lineup_players > 0, "mixed", players + availability + api_lineup_players),
        "advanced_stats": {
            **coverage_item("Stats de partido", advanced > 0, "real" if advanced >= 40 else "mixed", advanced),
            "details": {
                "corners": int(market_stats["corners"] or 0),
                "shots_on_target": int(market_stats["shots_on_target"] or 0),
                "total_shots": int(market_stats["total_shots"] or 0),
                "cards": int(market_stats["cards"] or 0),
                "possession": int(market_stats["possession"] or 0),
            },
        },
        "context": coverage_item("Contexto/calendario", matches > 0 or api_standings > 0, "mixed" if api_standings > 0 else "proxy", matches + api_standings),
        "api_enrichment": coverage_item("Enriquecimiento API", api_team_stats > 0 or api_standings > 0 or api_lineups > 0, "real", api_team_stats + api_standings + api_lineups),
        "environment": coverage_item("Sedes/estadio", venues > 0, "mixed", venues),
        "name_mapping": coverage_item("Mapeo de nombres", aliases > 0, "real", aliases),
        "api_readiness": coverage_item("Preparacion API", requirements["ready_count"] > 0 or api_coverage_rows > 0, "mixed", requirements["total"] + api_coverage_rows),
        "predictions": coverage_item("Predicciones guardadas", predictions > 0, "real", predictions),
        "experiments": coverage_item("Experimentos", experiments > 0, "real", experiments),
    }
    score = sum(item["weight"] for item in groups.values()) / len(groups)
    return {
        "score": round(score, 3),
        "groups": groups,
        "required_groups": REQUIRED_DATA_GROUPS,
        "readiness": requirements,
        "competition_coverage": competition_coverage(conn),
        "market_coverage": markets,
    }


def coverage_item(label: str, available: bool, status: str, count: int) -> dict:
    if status == "real" and available:
        weight = 1.0
    elif status == "mixed" and available:
        weight = 0.65
    elif status == "proxy" and available:
        weight = 0.45
    else:
        weight = 0.0
    return {"label": label, "available": available, "status": status, "count": int(count), "weight": weight}


def safe_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def _market_strength(ratio: float, sample_matches: int) -> str:
    if sample_matches >= 100 and ratio >= 0.75:
        return "strong"
    if sample_matches >= 40 and ratio >= 0.45:
        return "medium"
    if sample_matches > 0:
        return "weak"
    return "missing"


def _market_item(label: str, rows: sqlite3.Row, key: str, total_matches: int) -> dict:
    team_rows = int(rows[key] or 0)
    sample_matches = team_rows // 2
    ratio = sample_matches / total_matches if total_matches else 0.0
    return {
        "label": label,
        "team_rows": team_rows,
        "sample_matches": sample_matches,
        "coverage_ratio": round(ratio, 3),
        "strength": _market_strength(ratio, sample_matches),
    }


def market_coverage(conn: sqlite3.Connection, limit: int = 20) -> dict:
    total_matches = safe_count(conn, "api_football_fixtures") or safe_count(conn, "matches")
    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS team_rows,
            COUNT(DISTINCT source_match_id) AS matches_with_any_stats,
            SUM(CASE WHEN corners IS NOT NULL THEN 1 ELSE 0 END) AS corners,
            SUM(CASE WHEN shots_on_target IS NOT NULL THEN 1 ELSE 0 END) AS shots_on_target,
            SUM(CASE WHEN total_shots IS NOT NULL THEN 1 ELSE 0 END) AS total_shots,
            SUM(CASE WHEN yellow_cards IS NOT NULL OR red_cards IS NOT NULL OR cards_estimate IS NOT NULL THEN 1 ELSE 0 END) AS cards,
            SUM(CASE WHEN xg IS NOT NULL THEN 1 ELSE 0 END) AS xg,
            SUM(CASE WHEN possession_pct IS NOT NULL THEN 1 ELSE 0 END) AS possession,
            SUM(CASE WHEN fouls IS NOT NULL THEN 1 ELSE 0 END) AS fouls
        FROM match_team_advanced_stats
        """
    ).fetchone()
    by_competition = conn.execute(
        """
        SELECT
            COALESCE(f.league_name, 'Unknown') AS competition,
            COUNT(DISTINCT s.source_match_id) AS matches_with_any_stats,
            COUNT(DISTINCT CASE WHEN s.corners IS NOT NULL THEN s.source_match_id END) AS corners_matches,
            COUNT(DISTINCT CASE WHEN s.shots_on_target IS NOT NULL THEN s.source_match_id END) AS shots_on_target_matches,
            COUNT(DISTINCT CASE WHEN s.yellow_cards IS NOT NULL OR s.red_cards IS NOT NULL OR s.cards_estimate IS NOT NULL THEN s.source_match_id END) AS cards_matches,
            COUNT(DISTINCT CASE WHEN s.xg IS NOT NULL THEN s.source_match_id END) AS xg_matches
        FROM match_team_advanced_stats s
        LEFT JOIN api_football_fixtures f
            ON s.source_match_id = 'api-football:' || f.fixture_id
        GROUP BY COALESCE(f.league_name, 'Unknown')
        ORDER BY matches_with_any_stats DESC, competition
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    markets = {
        "corners": _market_item("Corners", totals, "corners", total_matches),
        "shots_on_target": _market_item("Tiros a puerta", totals, "shots_on_target", total_matches),
        "total_shots": _market_item("Tiros totales", totals, "total_shots", total_matches),
        "cards": _market_item("Tarjetas", totals, "cards", total_matches),
        "xg": _market_item("xG", totals, "xg", total_matches),
        "possession": _market_item("Posesion", totals, "possession", total_matches),
        "fouls": _market_item("Faltas", totals, "fouls", total_matches),
    }
    return {
        "total_matches_reference": int(total_matches or 0),
        "matches_with_any_stats": int(totals["matches_with_any_stats"] or 0),
        "team_stat_rows": int(totals["team_rows"] or 0),
        "markets": markets,
        "by_competition": [dict(row) for row in by_competition],
    }


def source_readiness(conn: sqlite3.Connection) -> dict:
    try:
        rows = conn.execute("SELECT status, COUNT(*) AS total FROM data_source_requirements GROUP BY status").fetchall()
    except sqlite3.OperationalError:
        rows = []
    by_status = {str(row["status"]): int(row["total"]) for row in rows}
    total = sum(by_status.values())
    ready = by_status.get("ready", 0) + by_status.get("partial", 0)
    return {
        "total": total,
        "ready_count": ready,
        "ready_ratio": round(ready / total, 3) if total else 0.0,
        "by_status": by_status,
    }


def competition_coverage(conn: sqlite3.Connection) -> list[dict]:
    try:
        competitions = conn.execute(
            """
            SELECT name, region, country, api_football_league_id, status, priority
            FROM competitions
            WHERE enabled = 1
            ORDER BY priority ASC, region, name
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    rows = []
    for competition in competitions:
        name = str(competition["name"])
        api_id = competition["api_football_league_id"]
        match_count = conn.execute("SELECT COUNT(*) FROM matches WHERE competition = ?", (name,)).fetchone()[0]
        fixture_count = 0
        if api_id is not None:
            fixture_count = conn.execute("SELECT COUNT(*) FROM api_football_fixtures WHERE league_id = ?", (api_id,)).fetchone()[0]
        has_real = match_count > 0 or fixture_count > 0
        score = 0.15
        if has_real:
            score += 0.45
        if str(competition["status"]) == "partial":
            score += 0.15
        if fixture_count > 0:
            score += 0.15
        rows.append(
            {
                "name": name,
                "region": competition["region"],
                "country": competition["country"],
                "api_football_league_id": api_id,
                "status": competition["status"],
                "matches": int(match_count),
                "fixtures": int(fixture_count),
                "coverage_score": round(min(score, 1.0), 3),
            }
        )
    return rows
