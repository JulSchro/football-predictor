from __future__ import annotations

from math import exp
import sqlite3


def estimate_player_markets(
    conn: sqlite3.Connection,
    home_team: str,
    away_team: str,
    secondary_markets: dict | None = None,
    home_goal_expectancy: float | None = None,
    away_goal_expectancy: float | None = None,
) -> dict:
    markets = secondary_markets or {}
    home_sot = float(markets.get("home_shots_on_target_expected") or 4.0)
    away_sot = float(markets.get("away_shots_on_target_expected") or 4.0)
    home_goals = float(home_goal_expectancy or markets.get("home_xg_recent") or max(home_sot * 0.32, 0.8))
    away_goals = float(away_goal_expectancy or markets.get("away_xg_recent") or max(away_sot * 0.32, 0.8))
    home_players = team_player_markets(conn, home_team, home_goals, home_sot)
    away_players = team_player_markets(conn, away_team, away_goals, away_sot)
    all_scorers = sorted(home_players["players"] + away_players["players"], key=lambda row: row["score_prob"], reverse=True)
    all_sot = sorted(home_players["players"] + away_players["players"], key=lambda row: row["shot_on_target_prob"], reverse=True)
    return {
        "home_team": {"team": home_team, **home_players},
        "away_team": {"team": away_team, **away_players},
        "top_scorers": all_scorers[:8],
        "top_shots_on_target": all_sot[:8],
        "data_notes": data_notes(home_players, away_players),
    }


def team_player_markets(conn: sqlite3.Connection, team: str, team_goal_expectancy: float, team_sot_expectancy: float) -> dict:
    rows = player_rows(conn, team)
    if not rows:
        return {
            "expected_goals": round(team_goal_expectancy, 2),
            "expected_shots_on_target": round(team_sot_expectancy, 2),
            "players": [],
            "sample_players": 0,
            "coverage": "sin datos",
        }
    unavailable = unavailable_player_ids(conn, team)
    scored = [score_player(dict(row), unavailable) for row in rows]
    scored = [row for row in scored if row["availability"] > 0.05]
    scorer_total = sum(row["scorer_weight"] for row in scored) or 1.0
    sot_total = sum(row["sot_weight"] for row in scored) or 1.0
    output = []
    for row in scored:
        goal_lambda = max(team_goal_expectancy, 0.05) * row["scorer_weight"] / scorer_total
        sot_lambda = max(team_sot_expectancy, 0.05) * row["sot_weight"] / sot_total
        output.append(
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "team": team,
                "position": row.get("position"),
                "score_prob": round(prob_at_least_one(goal_lambda), 4),
                "shot_on_target_prob": round(prob_at_least_one(sot_lambda), 4),
                "expected_goals": round(goal_lambda, 3),
                "expected_shots_on_target": round(sot_lambda, 3),
                "confidence": confidence_label(row["confidence_base"]),
                "minutes": row["minutes"],
                "goals": row["goals"],
                "shots_on_target": row["shots_on_target"],
                "rating": row["rating"],
                "availability": round(row["availability"], 2),
            }
        )
    output.sort(key=lambda item: item["score_prob"], reverse=True)
    return {
        "expected_goals": round(team_goal_expectancy, 2),
        "expected_shots_on_target": round(team_sot_expectancy, 2),
        "players": output[:12],
        "sample_players": len(rows),
        "coverage": "alta" if len(rows) >= 16 else "media" if len(rows) >= 8 else "baja",
    }


def player_rows(conn: sqlite3.Connection, team: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT p.id AS player_id,
               p.name AS player_name,
               p.preferred_position AS position,
               SUM(COALESCE(pss.minutes, 0)) AS minutes,
               SUM(COALESCE(pss.goals, 0)) AS goals,
               SUM(COALESCE(pss.assists, 0)) AS assists,
               SUM(COALESCE(pss.shots, 0)) AS shots,
               SUM(COALESCE(pss.shots_on_target, 0)) AS shots_on_target,
               AVG(pss.rating) AS rating,
               COUNT(*) AS stat_rows
        FROM player_season_stats pss
        JOIN players p ON p.id = pss.player_id
        WHERE pss.team_name = ? COLLATE NOCASE
        GROUP BY p.id, p.name, p.preferred_position
        HAVING SUM(COALESCE(pss.minutes, 0)) > 0
        ORDER BY minutes DESC, rating DESC
        """,
        (team,),
    ).fetchall()


def unavailable_player_ids(conn: sqlite3.Connection, team: str) -> set[int]:
    rows = conn.execute(
        """
        SELECT DISTINCT player_id
        FROM player_availability
        WHERE team = ? COLLATE NOCASE
          AND player_id IS NOT NULL
          AND lower(COALESCE(status, '')) NOT IN ('available', 'fit')
        """,
        (team,),
    ).fetchall()
    return {int(row["player_id"]) for row in rows}


def score_player(row: dict, unavailable: set[int]) -> dict:
    minutes = float(row.get("minutes") or 0)
    goals = float(row.get("goals") or 0)
    assists = float(row.get("assists") or 0)
    shots = float(row.get("shots") or 0)
    shots_on_target = float(row.get("shots_on_target") or 0)
    rating = float(row.get("rating") or 6.4)
    per90_base = max(minutes / 90, 1.0)
    goals_p90 = goals / per90_base
    shots_p90 = shots / per90_base
    sot_p90 = shots_on_target / per90_base
    starts_proxy = min(minutes / 900, 1.0)
    position_boost = position_attack_boost(row.get("position"))
    availability = 0.0 if int(row["player_id"]) in unavailable else 1.0
    rating_boost = max(rating - 6.2, 0) * 0.12
    scorer_weight = (
        0.18
        + goals_p90 * 0.9
        + sot_p90 * 0.35
        + shots_p90 * 0.08
        + assists / per90_base * 0.05
        + rating_boost
        + starts_proxy * 0.18
    ) * position_boost * availability
    sot_weight = (
        0.22
        + sot_p90 * 0.85
        + shots_p90 * 0.22
        + goals_p90 * 0.18
        + rating_boost
        + starts_proxy * 0.15
    ) * position_boost * availability
    confidence = min(1.0, minutes / 1200) * 0.65 + min(1.0, float(row.get("stat_rows") or 1) / 3) * 0.35
    return {
        **row,
        "minutes": int(minutes),
        "goals": int(goals),
        "shots_on_target": int(shots_on_target),
        "rating": round(rating, 2),
        "scorer_weight": max(scorer_weight, 0.01),
        "sot_weight": max(sot_weight, 0.01),
        "availability": availability,
        "confidence_base": confidence,
    }


def position_attack_boost(position: str | None) -> float:
    text = (position or "").strip().lower()
    if text in {"f", "fw", "st", "cf", "lw", "rw"} or any(token in text for token in ["attack", "forward", "striker", "winger"]):
        return 1.18
    if text in {"m", "mf", "cm", "am", "dm"} or "midfield" in text:
        return 0.88
    if text in {"d", "df", "cb", "lb", "rb"} or any(token in text for token in ["defender", "back"]):
        return 0.45
    if text in {"g", "gk"} or "goalkeeper" in text:
        return 0.03
    return 0.75


def prob_at_least_one(lam: float) -> float:
    return max(0.0, min(0.95, 1 - exp(-max(lam, 0.0))))


def confidence_label(value: float) -> str:
    if value >= 0.72:
        return "alta"
    if value >= 0.42:
        return "media"
    return "baja"


def data_notes(home: dict, away: dict) -> list[str]:
    notes = []
    if home["coverage"] != "alta" or away["coverage"] != "alta":
        notes.append("Mercados de jugador con cobertura limitada para al menos un equipo.")
    notes.append("Estimacion basada en temporada, minutos, rating, tiros y disponibilidad local.")
    notes.append("Mejora cuando hay alineaciones confirmadas y stats recientes por jugador.")
    return notes
