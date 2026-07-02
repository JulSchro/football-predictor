from __future__ import annotations

import sqlite3

from football_predictor.database.db import get_referee_profile_for_fixture


def referee_card_profile(conn: sqlite3.Connection, fixture_id: int | None) -> dict | None:
    profile = get_referee_profile_for_fixture(conn, fixture_id)
    if not profile:
        return None
    return {
        "available": bool(profile.get("available")),
        "fixture_id": fixture_id,
        "referee_name": profile.get("referee_name"),
        "referee_country": profile.get("referee_country"),
        "matches": int(profile.get("matches") or 0),
        "avg_cards": profile.get("avg_cards"),
        "avg_yellow": profile.get("avg_yellow"),
        "avg_red": profile.get("avg_red"),
        "avg_home_cards": profile.get("avg_home_cards"),
        "avg_away_cards": profile.get("avg_away_cards"),
        "home_away_diff": profile.get("home_away_diff"),
        "avg_fouls": profile.get("avg_fouls"),
        "penalty_rate": profile.get("penalty_rate"),
        "std_cards": profile.get("std_cards"),
        "last5_avg_cards": profile.get("last5_avg_cards"),
        "last10_avg_cards": profile.get("last10_avg_cards"),
        "recent_trend": profile.get("recent_trend"),
        "consistency": profile.get("consistency"),
        "strictness_percentile": profile.get("strictness_percentile"),
        "strictness_label": profile.get("strictness_label"),
        "home_bias_label": profile.get("home_bias_label"),
        "away_bias_label": profile.get("away_bias_label"),
        "profile": profile.get("profile") or {},
    }


def apply_referee_card_adjustment(team_expected_cards: float, team_sample: int, referee_profile: dict | None) -> dict:
    if not referee_profile or not referee_profile.get("available") or referee_profile.get("avg_cards") is None:
        return {
            "expected_cards": round(team_expected_cards, 2),
            "applied": False,
            "reason": "sin historial suficiente de arbitro",
        }
    referee_matches = int(referee_profile.get("matches") or 0)
    team_weight = max(int(team_sample or 0), 0)
    total_weight = referee_matches + team_weight
    if total_weight <= 0:
        return {
            "expected_cards": round(team_expected_cards, 2),
            "applied": False,
            "reason": "sin muestra ponderable",
        }
    referee_expected = float(referee_profile["avg_cards"])
    adjusted = (float(team_expected_cards) * team_weight + referee_expected * referee_matches) / total_weight
    return {
        "expected_cards": round(max(0.0, adjusted), 2),
        "applied": True,
        "team_baseline_cards": round(float(team_expected_cards), 2),
        "referee_expected_cards": round(referee_expected, 2),
        "team_sample_weight": team_weight,
        "referee_sample_weight": referee_matches,
        "adjustment": round(adjusted - float(team_expected_cards), 4),
        "explanations": referee_explanations(referee_profile),
    }


def referee_explanations(profile: dict) -> list[str]:
    name = profile.get("referee_name") or "El arbitro"
    explanations = []
    if profile.get("avg_cards") is not None:
        explanations.append(f"{name} promedia {float(profile['avg_cards']):.2f} tarjetas por partido.")
    if profile.get("last10_avg_cards") is not None:
        explanations.append(f"En sus ultimos 10 partidos su media fue {float(profile['last10_avg_cards']):.2f}.")
    if profile.get("recent_trend") is not None:
        trend = float(profile["recent_trend"])
        direction = "al alza" if trend > 0 else "a la baja" if trend < 0 else "estable"
        explanations.append(f"Su tendencia reciente esta {direction} ({trend:+.2f}).")
    if profile.get("home_away_diff") is not None:
        diff = float(profile["home_away_diff"])
        side = "locales" if diff > 0 else "visitantes" if diff < 0 else "ningun lado"
        explanations.append(f"Distribucion local/visitante: {abs(diff):.2f} tarjetas de diferencia hacia {side}.")
    if profile.get("strictness_label"):
        explanations.append(f"Indice de severidad: {profile['strictness_label']}.")
    return explanations
