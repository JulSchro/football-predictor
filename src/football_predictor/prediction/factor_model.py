from math import exp

import pandas as pd

from football_predictor.features.team_features import recent_form, recent_form_by_venue, team_history
from football_predictor.models.baseline_elo import EloBaseline
from football_predictor.prediction.predictor import fifa_seed_ratings


PENDING_GROUPS = {
    "players": [
        "minutes",
        "goals",
        "assists",
        "xg",
        "xa",
        "form",
        "fatigue",
        "age",
        "rating",
        "injuries",
        "suspensions",
        "position_quality",
        "chemistry",
    ],
    "advanced": [
        "xg",
        "xga",
        "ppda",
        "field_tilt",
        "expected_threat",
        "deep_completions",
        "progressive_passes",
        "progressive_carries",
        "pressures",
        "recoveries",
        "touches_in_area",
    ],
    "context": [
        "must_win",
        "draw_is_enough",
        "final",
        "knockout",
        "derby",
        "coach_debut",
        "press_pressure",
        "internal_conflict",
    ],
    "environment": [
        "temperature",
        "rain",
        "wind",
        "altitude",
        "humidity",
        "grass_type",
        "attendance",
        "referee_cards",
        "referee_penalties",
        "odds",
        "closing_line",
    ],
}


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))


def form_sequence(history: pd.DataFrame, n: int) -> str:
    tail = history.tail(n)
    chars = []
    for _, row in tail.iterrows():
        if row["result"] == 3:
            chars.append("W")
        elif row["result"] == 1:
            chars.append("D")
        else:
            chars.append("L")
    return "".join(chars)


def trend_score(history: pd.DataFrame, n: int) -> float:
    tail = history.tail(n)
    if len(tail) < 2:
        return 50.0
    points = tail["result"].astype(float).reset_index(drop=True)
    weights = pd.Series(range(1, len(points) + 1), dtype=float)
    weighted = float((points * weights).sum() / weights.sum())
    plain = float(points.mean())
    return clamp(50 + (weighted - plain) * 18)


def consistency_score(history: pd.DataFrame, n: int = 10) -> float:
    tail = history.tail(n)
    if len(tail) < 2:
        return 50.0
    goal_diff = tail["goals_for"] - tail["goals_against"]
    return clamp(100 - float(goal_diff.std(ddof=0)) * 18)


def days_rest_score(history: pd.DataFrame) -> float:
    if len(history) < 2:
        return 60.0
    last_dates = pd.to_datetime(history["date"]).tail(2).to_list()
    days = max((last_dates[-1] - last_dates[-2]).days, 0)
    if days >= 7:
        return 80.0
    if days >= 5:
        return 65.0
    if days >= 3:
        return 48.0
    return 32.0


def congestion_penalty(history: pd.DataFrame) -> float:
    if history.empty:
        return 0.0
    matches_7 = matches_in_window(history, 7)
    matches_14 = matches_in_window(history, 14)
    matches_30 = matches_in_window(history, 30)
    return clamp(max(0, matches_7 - 1) * 7 + max(0, matches_14 - 3) * 3 + max(0, matches_30 - 6) * 1.5, 0, 24)


def own_ranking_score(elo_rating: float, form_points: float, goal_diff: float) -> float:
    return clamp(50 + (elo_rating - 1500) / 8 + form_points * 8 + goal_diff * 6)


def external_strength(metrics: dict) -> dict[str, float | None]:
    fifa_rank = metrics.get("fifa_rank")
    fifa_points = metrics.get("fifa_points")
    squad_value = metrics.get("squad_value_eur")
    avg_age = metrics.get("avg_age")
    squad_size = metrics.get("squad_size")

    return {
        "fifa_rank_score": clamp(100 - (float(fifa_rank) - 1) * 1.4) if fifa_rank else None,
        "fifa_points_score": clamp((float(fifa_points) - 1200) / 8) if fifa_points else None,
        "squad_value_score": clamp(35 + (float(squad_value) ** 0.18) / 1.8) if squad_value else None,
        "age_balance_score": clamp(100 - abs(float(avg_age) - 27.5) * 8) if avg_age else None,
        "bench_depth_score": clamp(float(squad_size) * 2.7) if squad_size else None,
    }


def opponent_adjusted_form(matches: pd.DataFrame, team: str, n: int = 10) -> dict[str, float]:
    if matches.empty:
        return {"score": 50.0, "momentum": 50.0, "schedule_difficulty": 50.0}
    elo = EloBaseline().fit(matches)
    rows = []
    for _, row in matches.sort_values("date").iterrows():
        if row["home_team"] == team:
            opponent = row["away_team"]
            goals_for = row["home_goals"]
            goals_against = row["away_goals"]
        elif row["away_team"] == team:
            opponent = row["home_team"]
            goals_for = row["away_goals"]
            goals_against = row["home_goals"]
        else:
            continue
        if pd.isna(goals_for) or pd.isna(goals_against):
            continue
        points = 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
        opponent_rating = elo.rating(str(opponent))
        opponent_score = clamp(50 + (opponent_rating - 1500) / 7)
        adjusted_result = 50 + (points - 1.3) * 13 + (float(goals_for) - float(goals_against)) * 4 + (opponent_score - 50) * 0.22
        rows.append({"adjusted": clamp(adjusted_result), "opponent_score": opponent_score})
    tail = rows[-n:]
    if not tail:
        return {"score": 50.0, "momentum": 50.0, "schedule_difficulty": 50.0}
    weights = list(range(1, len(tail) + 1))
    weighted_score = sum(item["adjusted"] * weight for item, weight in zip(tail, weights)) / sum(weights)
    plain_score = sum(item["adjusted"] for item in tail) / len(tail)
    schedule = sum(item["opponent_score"] for item in tail) / len(tail)
    return {
        "score": round(weighted_score, 2),
        "momentum": round(clamp(50 + (weighted_score - plain_score) * 1.2), 2),
        "schedule_difficulty": round(schedule, 2),
    }


def missing_player_importance(metrics: dict, external: dict, lineup_depth: float = 0.0) -> float:
    unavailable = float(metrics.get("unavailable_players") or 0)
    explicit_importance = metrics.get("missing_player_importance")
    if explicit_importance is not None:
        return clamp(float(explicit_importance), 0, 45)
    star_multiplier = 1.0
    if external.get("squad_value_score") and external["squad_value_score"] > 70:
        star_multiplier += 0.25
    if lineup_depth and lineup_depth < 16:
        star_multiplier += 0.18
    return clamp(unavailable * 4.2 * star_multiplier, 0, 42)


def style_profile(metrics: dict) -> dict[str, float]:
    possession = metrics.get("possession_for")
    pass_accuracy = metrics.get("pass_accuracy_for")
    shots = metrics.get("shots_for")
    shots_on_target = metrics.get("shots_on_target_for")
    corners = metrics.get("corners_for")
    fouls = metrics.get("fouls_for")
    cards = metrics.get("cards_for")
    dangerous = metrics.get("dangerous_attacks_for")
    possession_control = weighted_mean(
        [
            clamp(float(possession)) if possession is not None else None,
            clamp(20 + float(pass_accuracy) * 0.9) if pass_accuracy is not None else None,
        ],
        default=50.0,
    )
    shot_quality = clamp(42 + (float(shots_on_target or 0) / max(float(shots or 1), 1)) * 75) if shots is not None or shots_on_target is not None else 50.0
    directness = weighted_mean(
        [
            clamp(35 + float(shots or 0) * 2.0) if shots is not None else None,
            clamp(35 + float(dangerous or 0) * 0.55) if dangerous is not None else None,
        ],
        default=50.0,
    )
    set_piece_threat = clamp(42 + float(corners) * 4.2) if corners is not None else 50.0
    discipline = weighted_mean(
        [
            clamp(82 - float(fouls) * 1.4) if fouls is not None else None,
            clamp(86 - float(cards) * 9) if cards is not None else None,
        ],
        default=50.0,
    )
    return {
        "possession_control": round(possession_control, 2),
        "shot_quality": round(shot_quality, 2),
        "directness": round(directness, 2),
        "set_piece_threat": round(set_piece_threat, 2),
        "discipline": round(discipline, 2),
    }


def build_team_profile(matches: pd.DataFrame, team: str, metrics: dict | None = None) -> dict:
    metrics = metrics or {}
    played = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    history = team_history(played, team)
    elo = EloBaseline(initial_ratings=fifa_seed_ratings({team: metrics})).fit(played)
    elo_rating = elo.rating(team)
    form5 = recent_form(played, team, n=5)
    form10 = recent_form(played, team, n=10)
    form20 = recent_form(played, team, n=20)
    opponent_adjusted = opponent_adjusted_form(played, team, n=10)
    home = recent_form_by_venue(played, team, is_home=True, n=20)
    away = recent_form_by_venue(played, team, is_home=False, n=20)
    ext = external_strength(metrics)
    style = style_profile(metrics)

    glicko_proxy = clamp(50 + (elo_rating - 1500) / 7 - max(0, 8 - len(history)) * 2.5)
    ranking_score = own_ranking_score(elo_rating, form10["form_points"], form10["goal_diff"])
    sample_trust = min(len(history) / 12, 1.0)
    fifa_component = weighted_mean([ext.get("fifa_rank_score"), ext.get("fifa_points_score")], default=50.0)
    value_component = weighted_mean([ext.get("squad_value_score"), ext.get("bench_depth_score"), ext.get("age_balance_score")], default=50.0)
    player_quality_component = weighted_mean(
        [
            metrics.get("player_rating_score"),
            metrics.get("player_depth_score"),
            metrics.get("player_attack_output_score"),
        ],
        default=None,
    )
    elo_component = clamp(50 + (elo_rating - 1500) / 8)
    form_component = weighted_mean([ranking_score, opponent_adjusted["score"]], default=ranking_score)
    api_component = api_strength_score(metrics)
    api_trust = min(float(metrics.get("api_played") or metrics.get("standing_played") or 0) / 6, 1.0)
    strength_score = (
        fifa_component * 0.42
        + elo_component * 0.28
        + value_component * 0.14
        + (form_component * sample_trust + 50 * (1 - sample_trust)) * 0.16
    )
    if api_component is not None:
        strength_score = strength_score * (1 - api_trust * 0.28) + api_component * api_trust * 0.28
    if player_quality_component is not None:
        strength_score = strength_score * 0.86 + float(player_quality_component) * 0.14
    api_form = weighted_mean([metrics.get("api_form_score"), metrics.get("standing_form_score")], default=None)
    momentum = trend_score(history, 5) * 0.45 + trend_score(history, 10) * 0.35 + opponent_adjusted["momentum"] * 0.20
    if api_form is not None:
        momentum = momentum * (1 - api_trust * 0.35) + api_form * api_trust * 0.35
    attack_raw = clamp(50 + form10["avg_goals_for"] * 16 + form10["goal_diff"] * 6)
    defense_raw = clamp(72 - form10["avg_goals_against"] * 14 + form10["goal_diff"] * 4)
    attack = attack_raw * sample_trust + 55 * (1 - sample_trust)
    defense = defense_raw * sample_trust + 55 * (1 - sample_trust)
    season_attack = api_attack_score(metrics)
    season_defense = api_defense_score(metrics)
    if season_attack is not None:
        attack = attack * (1 - api_trust * 0.28) + season_attack * api_trust * 0.28
    if season_defense is not None:
        defense = defense * (1 - api_trust * 0.28) + season_defense * api_trust * 0.28
    if metrics.get("player_attack_output_score") is not None:
        attack = attack * 0.84 + float(metrics["player_attack_output_score"]) * 0.16
    if metrics.get("player_rating_score") is not None:
        defense = defense * 0.91 + float(metrics["player_rating_score"]) * 0.09
    advanced = advanced_strength(metrics)
    advanced_trust = min(float(metrics.get("advanced_matches") or 0) / 8, 1.0)
    attack = attack * (1 - advanced_trust * 0.35) + advanced["attack"] * advanced_trust * 0.35
    defense = defense * (1 - advanced_trust * 0.35) + advanced["defense"] * advanced_trust * 0.35
    home_advantage = clamp(50 + home["venue_win_rate"] * 35 + home["venue_goals_for"] * 5)
    away_resilience = clamp(50 + away["venue_win_rate"] * 35 + away["venue_goals_for"] * 5)
    if metrics.get("api_home_played"):
        home_advantage = home_advantage * 0.72 + clamp(45 + float(metrics.get("api_home_wins") or 0) / max(float(metrics.get("api_home_played") or 1), 1) * 45) * 0.28
    if metrics.get("api_away_played"):
        away_resilience = away_resilience * 0.72 + clamp(45 + float(metrics.get("api_away_wins") or 0) / max(float(metrics.get("api_away_played") or 1), 1) * 45) * 0.28
    fatigue = days_rest_score(history)
    congestion = congestion_penalty(history)
    consistency = consistency_score(history, 10)
    unavailable_players = float(metrics.get("unavailable_players") or 0)
    missing_importance = missing_player_importance(metrics, ext, lineup_depth=float(metrics.get("lineup_distinct_players") or 0))
    lineup_depth = float(metrics.get("lineup_distinct_players") or 0)
    tracked_depth_score = metrics.get("player_depth_score")
    lineup_depth_score = weighted_mean(
        [
            clamp(45 + lineup_depth * 1.8) if lineup_depth else None,
            float(tracked_depth_score) if tracked_depth_score is not None else None,
        ],
        default=None,
    )
    minutes_per_player = metrics.get("player_minutes_per_player")
    player_fatigue_penalty = clamp((float(minutes_per_player) - 1400) / 70, 0, 18) if minutes_per_player is not None else 0.0
    availability = clamp(
        50
        + (ext.get("bench_depth_score") or 50) * 0.25
        + (ext.get("age_balance_score") or 50) * 0.25
        + (lineup_depth_score or 50) * 0.15
        + float(metrics.get("player_rating_score") or 50) * 0.10
        - missing_importance
    )
    tactical_reliability = weighted_mean(
        [style["possession_control"], style["shot_quality"], style["discipline"], consistency],
        default=50.0,
    )
    pressure_resilience = clamp(
        consistency * 0.35
        + availability * 0.25
        + (100 - congestion) * 0.20
        + weighted_mean([ext.get("fifa_points_score"), strength_score], default=strength_score) * 0.20
    )

    generated = {
        "attack_strength": round(attack, 2),
        "defense_strength": round(defense, 2),
        "momentum_score": round(momentum, 2),
        "pressure_index": round(pressure_resilience, 2),
        "fatigue_index": round(clamp(100 - fatigue + congestion), 2),
        "player_fatigue_index": round(player_fatigue_penalty, 2),
        "confidence_score": round((strength_score + momentum + consistency) / 3, 2),
        "recent_improvement": round(trend_score(history, 5), 2),
        "consistency_score": round(consistency, 2),
        "squad_stability": round(lineup_depth_score or 50.0, 2),
        "coach_stability": 50.0,
        "home_advantage_score": round(home_advantage, 2),
        "away_resilience_score": round(away_resilience, 2),
        "goal_expectancy": round(max(form10["avg_goals_for"], 0.2), 2),
        "expected_momentum": round(momentum, 2),
        "injury_impact": round(missing_importance, 2),
        "player_availability_score": round(availability, 2),
        "player_quality_score": round(float(player_quality_component), 2) if player_quality_component is not None else 50.0,
        "player_attack_output": round(float(metrics.get("player_attack_output_score") or 50), 2),
        "squad_depth_score": round(lineup_depth_score or 50.0, 2),
        "opponent_adjusted_form": round(opponent_adjusted["score"], 2),
        "schedule_difficulty": round(opponent_adjusted["schedule_difficulty"], 2),
        "tactical_reliability": round(tactical_reliability, 2),
        "style_control": round(style["possession_control"], 2),
        "style_directness": round(style["directness"], 2),
        "set_piece_threat": round(style["set_piece_threat"], 2),
        "discipline_score": round(style["discipline"], 2),
    }

    return {
        "team": team,
        "sample_size": int(len(history)),
        "strength": {
            "score": round(strength_score, 2),
            "elo": round(elo_rating, 2),
            "glicko_proxy": round(glicko_proxy, 2),
            "own_ranking": round(ranking_score, 2),
            "fifa_rank_score": ext.get("fifa_rank_score"),
            "squad_value_score": ext.get("squad_value_score"),
            "coach_quality": 50.0,
            "coach_stability": 50.0,
            "project_years": 50.0,
            "advanced_trust": round(advanced_trust, 2),
            "api_strength_score": round(api_component, 2) if api_component is not None else None,
            "api_trust": round(api_trust, 2),
            "standing_rank": metrics.get("standing_rank"),
            "standing_points_per_match": metrics.get("standing_points_per_match"),
            "player_quality_score": round(float(player_quality_component), 2) if player_quality_component is not None else None,
        },
        "form": {
            "last_5": {**form5, "sequence": form_sequence(history, 5), "trend": round(trend_score(history, 5), 2)},
            "last_10": {**form10, "sequence": form_sequence(history, 10), "trend": round(trend_score(history, 10), 2)},
            "last_20": {**form20, "sequence": form_sequence(history, 20), "trend": round(trend_score(history, 20), 2)},
            "opponent_adjusted": opponent_adjusted,
        },
        "venue": {
            "home": home,
            "away": away,
            "travel_distance_score": 50.0,
            "travel_time_score": 50.0,
        },
        "fatigue_calendar": {
            "rest_score": round(fatigue, 2),
            "matches_last_7": int(matches_in_window(history, 7)),
            "matches_last_14": int(matches_in_window(history, 14)),
            "matches_last_30": int(matches_in_window(history, 30)),
            "congestion_penalty": round(congestion, 2),
            "rotation_score": 50.0,
        },
        "style": style,
        "generated": generated,
        "pending_data": PENDING_GROUPS,
    }


def advanced_strength(metrics: dict) -> dict[str, float]:
    xg_for = metrics.get("xg_for")
    xg_against = metrics.get("xg_against")
    shots = metrics.get("shots_for")
    shots_against = metrics.get("shots_against")
    shots_on_target = metrics.get("shots_on_target_for")
    shots_on_target_against = metrics.get("shots_on_target_against")
    corners = metrics.get("corners_for")
    corners_against = metrics.get("corners_against")
    fouls = metrics.get("fouls_for")
    possession = metrics.get("possession_for")
    pass_accuracy = metrics.get("pass_accuracy_for")
    dangerous_attacks = metrics.get("dangerous_attacks_for")
    attack_values = [
        clamp(42 + float(xg_for) * 24) if xg_for is not None else None,
        clamp(35 + float(shots_on_target) * 8) if shots_on_target is not None else None,
        clamp(35 + float(shots) * 2.2) if shots is not None else None,
        clamp(42 + float(corners) * 4) if corners is not None else None,
        clamp(35 + float(dangerous_attacks) * 0.65) if dangerous_attacks is not None else None,
        clamp(30 + float(possession) * 0.75) if possession is not None else None,
        clamp(20 + float(pass_accuracy) * 0.9) if pass_accuracy is not None else None,
    ]
    defense_values = [
        clamp(82 - float(xg_against) * 22) if xg_against is not None else None,
        clamp(78 - float(shots_against) * 2.0) if shots_against is not None else None,
        clamp(82 - float(shots_on_target_against) * 6.5) if shots_on_target_against is not None else None,
        clamp(72 - float(corners_against) * 3.4) if corners_against is not None else None,
        clamp(72 - float(fouls) * 1.3) if fouls is not None else None,
    ]
    return {
        "attack": weighted_mean(attack_values, default=55.0),
        "defense": weighted_mean(defense_values, default=55.0),
    }


def api_strength_score(metrics: dict) -> float | None:
    values = []
    if metrics.get("api_points_per_match") is not None:
        values.append(clamp(30 + float(metrics["api_points_per_match"]) * 22))
    if metrics.get("api_win_rate") is not None:
        values.append(clamp(35 + float(metrics["api_win_rate"]) * 55))
    if metrics.get("api_goal_diff_per_match") is not None:
        values.append(clamp(50 + float(metrics["api_goal_diff_per_match"]) * 18))
    if metrics.get("standing_rank") is not None:
        values.append(clamp(100 - (float(metrics["standing_rank"]) - 1) * 6))
    if metrics.get("standing_points_per_match") is not None:
        values.append(clamp(30 + float(metrics["standing_points_per_match"]) * 22))
    if not values:
        return None
    return sum(values) / len(values)


def api_attack_score(metrics: dict) -> float | None:
    values = []
    if metrics.get("api_goals_for_per_match") is not None:
        values.append(clamp(38 + float(metrics["api_goals_for_per_match"]) * 22))
    if metrics.get("api_goal_diff_per_match") is not None:
        values.append(clamp(52 + float(metrics["api_goal_diff_per_match"]) * 12))
    return sum(values) / len(values) if values else None


def api_defense_score(metrics: dict) -> float | None:
    values = []
    if metrics.get("api_goals_against_per_match") is not None:
        values.append(clamp(82 - float(metrics["api_goals_against_per_match"]) * 18))
    if metrics.get("api_clean_sheets") is not None and metrics.get("api_played"):
        values.append(clamp(48 + float(metrics["api_clean_sheets"]) / max(float(metrics["api_played"]), 1) * 48))
    if metrics.get("api_goal_diff_per_match") is not None:
        values.append(clamp(55 + float(metrics["api_goal_diff_per_match"]) * 10))
    return sum(values) / len(values) if values else None


def matches_in_window(history: pd.DataFrame, days: int) -> int:
    if history.empty:
        return 0
    dates = pd.to_datetime(history["date"])
    latest = dates.max()
    return int((dates >= latest - pd.Timedelta(days=days)).sum())


def weighted_mean(values: list[float | None], default: float = 50.0) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else default


def profile_edge(home_profile: dict, away_profile: dict) -> float:
    home = home_profile["generated"]
    away = away_profile["generated"]
    raw = (
        (home_profile["strength"]["score"] - away_profile["strength"]["score"]) * 0.004
        + (home["momentum_score"] - away["momentum_score"]) * 0.0025
        + (home["attack_strength"] - away["defense_strength"]) * 0.0018
        + (home["home_advantage_score"] - away["away_resilience_score"]) * 0.002
        + (away["fatigue_index"] - home["fatigue_index"]) * 0.0015
        + (home["player_availability_score"] - away["player_availability_score"]) * 0.0015
        + (home["player_quality_score"] - away["player_quality_score"]) * 0.0014
        + (home["squad_depth_score"] - away["squad_depth_score"]) * 0.0008
        + (away["player_fatigue_index"] - home["player_fatigue_index"]) * 0.001
        + (home["opponent_adjusted_form"] - away["opponent_adjusted_form"]) * 0.0013
        + (home["tactical_reliability"] - away["tactical_reliability"]) * 0.0012
        + (home["discipline_score"] - away["discipline_score"]) * 0.0007
    )
    return max(min(raw, 0.21), -0.21)


def factor_cards(home_profile: dict, away_profile: dict) -> list[dict]:
    cards = [
        ("Fuerza", home_profile["strength"]["score"], away_profile["strength"]["score"]),
        ("Momento", home_profile["generated"]["momentum_score"], away_profile["generated"]["momentum_score"]),
        ("Ataque vs defensa", home_profile["generated"]["attack_strength"], away_profile["generated"]["defense_strength"]),
        ("Localia", home_profile["generated"]["home_advantage_score"], away_profile["generated"]["away_resilience_score"]),
        ("Fatiga", 100 - home_profile["generated"]["fatigue_index"], 100 - away_profile["generated"]["fatigue_index"]),
        ("Disponibilidad", home_profile["generated"]["player_availability_score"], away_profile["generated"]["player_availability_score"]),
        ("Calidad jugadores", home_profile["generated"]["player_quality_score"], away_profile["generated"]["player_quality_score"]),
        ("Profundidad", home_profile["generated"]["squad_depth_score"], away_profile["generated"]["squad_depth_score"]),
        ("Forma ajustada", home_profile["generated"]["opponent_adjusted_form"], away_profile["generated"]["opponent_adjusted_form"]),
        ("Estilo", home_profile["generated"]["tactical_reliability"], away_profile["generated"]["tactical_reliability"]),
        ("Disciplina", home_profile["generated"]["discipline_score"], away_profile["generated"]["discipline_score"]),
    ]
    return [
        {"name": name, "home": round(home, 2), "away": round(away, 2), "edge": round(home - away, 2)}
        for name, home, away in cards
    ]
