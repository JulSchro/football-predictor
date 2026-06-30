from __future__ import annotations


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def normalize_competition_context(competition: str | None = None, round_name: str | None = None) -> dict:
    competition_text = (competition or "").strip()
    round_text = (round_name or "").strip()
    blob = f"{competition_text} {round_text}".lower()

    competition_type = _competition_type(blob)
    stage = _stage(blob, competition_type)
    values = _base_values(competition_type, stage)
    is_two_leg = _contains_any(blob, ["1st leg", "2nd leg", "first leg", "second leg", "ida", "vuelta"])
    is_return_leg = _contains_any(blob, ["2nd leg", "second leg", "vuelta"])
    if is_return_leg:
        values.update(
            draw_acceptance=max(float(values["draw_acceptance"]), 58.0),
            conservatism=float(values["conservatism"]) + 4,
            pressure_index=float(values["pressure_index"]) + 4,
        )

    return {
        "competition": competition_text or None,
        "round": round_text or None,
        "competition_type": competition_type,
        "stage": stage,
        "stage_label": _stage_label(stage),
        "is_friendly": competition_type == "friendly",
        "is_group_stage": stage == "group_stage",
        "is_knockout": stage in {"round_of_32", "round_of_16", "quarter_final", "semi_final", "final"},
        "is_two_leg": is_two_leg,
        "is_return_leg": is_return_leg,
        "is_extra_time_possible": stage in {"round_of_32", "round_of_16", "quarter_final", "semi_final", "final"},
        "is_final": stage == "final",
        "is_league": competition_type == "league",
        **values,
    }


def apply_context_to_outcomes(probabilities: dict[str, float], context: dict | None) -> dict[str, float]:
    if not context:
        return probabilities

    home = float(probabilities["home_win"])
    draw = float(probabilities["draw"])
    away = float(probabilities["away_win"])

    draw_delta = (float(context.get("draw_acceptance", 50)) - 50) * 0.0012
    draw_delta += (float(context.get("conservatism", 50)) - 50) * 0.0008
    draw_delta -= (float(context.get("volatility", 50)) - 50) * 0.0005
    draw_delta = max(min(draw_delta, 0.055), -0.055)

    if context.get("is_friendly"):
        home, draw, away = _compress_to_uniform(home, draw, away, amount=0.08)
    if context.get("is_return_leg"):
        draw_delta += 0.018
    if context.get("is_extra_time_possible"):
        draw_delta += 0.008

    draw = max(0.08, min(0.42, draw + draw_delta))
    remaining = max(0.01, 1 - draw)
    side_total = max(home + away, 0.01)
    home = remaining * home / side_total
    away = remaining * away / side_total
    return _normalize({"home_win": home, "draw": draw, "away_win": away})


def context_factor_card(context: dict | None) -> dict | None:
    if not context:
        return None
    pressure = float(context.get("pressure_index", 50))
    rotation = float(context.get("rotation_risk", 50))
    edge = pressure - rotation
    return {
        "name": "Contexto",
        "home": round(pressure, 2),
        "away": round(rotation, 2),
        "edge": round(edge, 2),
    }


def context_sensitivity(context: dict | None) -> dict | None:
    if not context:
        return None
    pressure = float(context.get("pressure_index", 50))
    rotation = float(context.get("rotation_risk", 50))
    draw_acceptance = float(context.get("draw_acceptance", 50))
    impact = min((abs(pressure - 50) + abs(rotation - 50) + abs(draw_acceptance - 50)) / 300 * 0.14, 0.14)
    return {
        "factor": "Contexto competitivo",
        "direction": str(context.get("stage_label") or context.get("stage") or "neutral"),
        "edge": round(pressure - rotation, 2),
        "estimated_probability_impact": round(impact, 4),
        "baseline_factor_edge": 0.0,
    }


def _competition_type(blob: str) -> str:
    if "friendly" in blob or "friendlies" in blob or "amistoso" in blob:
        return "friendly"
    if "world cup" in blob or "mundial" in blob:
        return "world_cup"
    if "champions" in blob or "libertadores" in blob or "sudamericana" in blob:
        return "continental"
    if "cup" in blob or "copa" in blob or "fa cup" in blob:
        return "cup"
    if "qualifier" in blob or "qualification" in blob or "nations league" in blob:
        return "international"
    return "league"


def _contains_any(blob: str, patterns: list[str]) -> bool:
    return any(pattern in blob for pattern in patterns)


def _stage(blob: str, competition_type: str) -> str:
    if competition_type == "friendly":
        return "friendly"
    if "final" in blob and "semi" not in blob and "quarter" not in blob:
        return "final"
    if "semi" in blob:
        return "semi_final"
    if "quarter" in blob or "4th finals" in blob:
        return "quarter_final"
    if "round of 16" in blob or "8th finals" in blob or "last 16" in blob:
        return "round_of_16"
    if "round of 32" in blob or "16th finals" in blob:
        return "round_of_32"
    if "group" in blob or "groups" in blob:
        return "group_stage"
    if competition_type == "league":
        return "league"
    return "unknown"


def _base_values(competition_type: str, stage: str) -> dict:
    values = {
        "pressure_index": 50.0,
        "rotation_risk": 35.0,
        "draw_acceptance": 45.0,
        "conservatism": 50.0,
        "volatility": 50.0,
    }
    if competition_type == "friendly":
        values.update(pressure_index=24.0, rotation_risk=78.0, draw_acceptance=52.0, conservatism=38.0, volatility=66.0)
    elif stage == "group_stage":
        values.update(pressure_index=58.0, rotation_risk=34.0, draw_acceptance=57.0, conservatism=54.0, volatility=47.0)
    elif stage == "round_of_32":
        values.update(pressure_index=70.0, rotation_risk=22.0, draw_acceptance=35.0, conservatism=61.0, volatility=54.0)
    elif stage == "round_of_16":
        values.update(pressure_index=76.0, rotation_risk=18.0, draw_acceptance=31.0, conservatism=64.0, volatility=56.0)
    elif stage == "quarter_final":
        values.update(pressure_index=82.0, rotation_risk=15.0, draw_acceptance=28.0, conservatism=67.0, volatility=58.0)
    elif stage == "semi_final":
        values.update(pressure_index=87.0, rotation_risk=12.0, draw_acceptance=25.0, conservatism=69.0, volatility=60.0)
    elif stage == "final":
        values.update(pressure_index=92.0, rotation_risk=10.0, draw_acceptance=22.0, conservatism=72.0, volatility=62.0)
    elif competition_type in {"cup", "continental", "world_cup", "international"}:
        values.update(pressure_index=64.0, rotation_risk=28.0, draw_acceptance=42.0, conservatism=58.0, volatility=52.0)
    return values


def _stage_label(stage: str) -> str:
    labels = {
        "friendly": "Amistoso",
        "group_stage": "Fase de grupos",
        "round_of_32": "Ronda de 32",
        "round_of_16": "Octavos",
        "quarter_final": "Cuartos",
        "semi_final": "Semifinal",
        "final": "Final",
        "league": "Liga",
        "unknown": "Sin clasificar",
    }
    return labels.get(stage, stage)


def _compress_to_uniform(home: float, draw: float, away: float, amount: float) -> tuple[float, float, float]:
    target = 1 / 3
    return (
        home * (1 - amount) + target * amount,
        draw * (1 - amount) + target * amount,
        away * (1 - amount) + target * amount,
    )


def _normalize(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
    return {key: value / total for key, value in probabilities.items()}
