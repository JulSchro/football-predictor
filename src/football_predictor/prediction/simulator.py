import random

import pandas as pd

from football_predictor.prediction.factor_model import build_team_profile, factor_cards, profile_edge
from football_predictor.prediction.match_context import apply_context_to_outcomes, context_factor_card, context_sensitivity
from football_predictor.prediction.predictor import MatchPredictor
from football_predictor.prediction.venue_model import venue_edge


def simulate_match(
    predictor: MatchPredictor,
    home_team: str,
    away_team: str,
    simulations: int = 10000,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    prediction = predictor.predict(home_team, away_team)
    matrix = predictor.poisson.predict_score_matrix(home_team, away_team)
    scores = list(matrix.keys())
    weights = list(matrix.values())

    outcomes = {"home_win": 0, "draw": 0, "away_win": 0, "over_2_5": 0, "both_teams_score": 0}
    score_counts: dict[str, int] = {}

    for _ in range(simulations):
        home_goals, away_goals = rng.choices(scores, weights=weights, k=1)[0]
        score_key = f"{home_goals}-{away_goals}"
        score_counts[score_key] = score_counts.get(score_key, 0) + 1
        if home_goals > away_goals:
            outcomes["home_win"] += 1
        elif home_goals < away_goals:
            outcomes["away_win"] += 1
        else:
            outcomes["draw"] += 1
        if home_goals + away_goals > 2.5:
            outcomes["over_2_5"] += 1
        if home_goals > 0 and away_goals > 0:
            outcomes["both_teams_score"] += 1

    top_scores = sorted(score_counts.items(), key=lambda item: item[1], reverse=True)[:8]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "simulations": simulations,
        "model_prediction": prediction.model_dump(),
        "simulation": {key: value / simulations for key, value in outcomes.items()},
        "top_scores": [{"score": score, "probability": count / simulations} for score, count in top_scores],
    }


def simulate_advanced_match(
    matches: pd.DataFrame,
    team_metrics: dict[str, dict],
    home_team: str,
    away_team: str,
    simulations: int = 10000,
    mode: str = "hybrid",
    seed: int = 42,
    venue: dict | None = None,
    match_context: dict | None = None,
    predictor: MatchPredictor | None = None,
    home_profile: dict | None = None,
    away_profile: dict | None = None,
) -> dict:
    rng = random.Random(seed)
    predictor = predictor or MatchPredictor(matches, team_metrics=team_metrics)
    base_prediction = predictor.predict(home_team, away_team)
    home_profile = home_profile or build_team_profile(matches, home_team, team_metrics.get(home_team, {}))
    away_profile = away_profile or build_team_profile(matches, away_team, team_metrics.get(away_team, {}))
    factor_edge = profile_edge(home_profile, away_profile)
    ground_edge = venue_edge(home_team, away_team, venue, team_metrics)
    edge = max(min(factor_edge + ground_edge, 0.22), -0.22)
    matrix = predictor.poisson.predict_score_matrix(home_team, away_team)
    scores = list(matrix.keys())
    weights = list(matrix.values())
    score_choices = {
        "home_win": filter_scores_by_outcome(scores, "home_win"),
        "draw": filter_scores_by_outcome(scores, "draw"),
        "away_win": filter_scores_by_outcome(scores, "away_win"),
    }
    score_weights = {outcome: [matrix[score] for score in outcome_scores] for outcome, outcome_scores in score_choices.items()}
    outcome_probabilities = adjusted_outcome_probabilities(base_prediction.model_dump(), edge)
    outcome_probabilities = apply_context_to_outcomes(outcome_probabilities, match_context)

    outcomes = {"home_win": 0, "draw": 0, "away_win": 0, "over_2_5": 0, "both_teams_score": 0}
    score_counts: dict[str, int] = {}
    scenario_counts = {"strength": 0, "form": 0, "fatigue": 0, "availability": 0, "context": 0}

    for _ in range(simulations):
        outcome = rng.choices(
            ["home_win", "draw", "away_win"],
            weights=[outcome_probabilities["home_win"], outcome_probabilities["draw"], outcome_probabilities["away_win"]],
            k=1,
        )[0]
        home_goals, away_goals = rng.choices(score_choices[outcome], weights=score_weights[outcome], k=1)[0]
        draw_edge = edge + poker_factor_noise(rng, mode)
        scenario_counts[dominant_scenario(rng, draw_edge)] += 1

        score_key = f"{home_goals}-{away_goals}"
        score_counts[score_key] = score_counts.get(score_key, 0) + 1
        if home_goals > away_goals:
            outcomes["home_win"] += 1
        elif home_goals < away_goals:
            outcomes["away_win"] += 1
        else:
            outcomes["draw"] += 1
        if home_goals + away_goals > 2.5:
            outcomes["over_2_5"] += 1
        if home_goals > 0 and away_goals > 0:
            outcomes["both_teams_score"] += 1

    top_scores = sorted(score_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "mode": mode,
        "simulations": simulations,
        "factor_edge": edge,
        "venue_edge": ground_edge,
        "model_prediction": base_prediction.model_dump(),
        "simulation": {key: value / simulations for key, value in outcomes.items()},
        "top_scores": [{"score": score, "probability": count / simulations} for score, count in top_scores],
        "factor_cards": _factor_cards_with_context(home_profile, away_profile, match_context),
        "profiles": {"home": home_profile, "away": away_profile},
        "scenario_mix": {key: value / simulations for key, value in scenario_counts.items()},
        "sensitivity": sensitivity_report(home_profile, away_profile, match_context),
        "match_context": match_context,
    }


def sensitivity_report(home_profile: dict, away_profile: dict, match_context: dict | None = None) -> list[dict]:
    base = profile_edge(home_profile, away_profile)
    cards = factor_cards(home_profile, away_profile)
    output = []
    for card in cards:
        impact = abs(card["edge"]) / 100
        direction = "home" if card["edge"] > 0 else "away" if card["edge"] < 0 else "neutral"
        output.append(
            {
                "factor": card["name"],
                "direction": direction,
                "edge": card["edge"],
                "estimated_probability_impact": round(min(impact * 0.32, 0.14), 4),
                "baseline_factor_edge": round(base, 4),
            }
        )
    context_row = context_sensitivity(match_context)
    if context_row:
        output.append(context_row)
    return sorted(output, key=lambda item: item["estimated_probability_impact"], reverse=True)


def _factor_cards_with_context(home_profile: dict, away_profile: dict, match_context: dict | None) -> list[dict]:
    cards = factor_cards(home_profile, away_profile)
    context_card = context_factor_card(match_context)
    if context_card:
        cards.append(context_card)
    return cards


def simulate_fixture_list(
    matches: pd.DataFrame,
    team_metrics: dict[str, dict],
    fixtures: list[dict],
    simulations: int = 5000,
    mode: str = "hybrid",
    predictor: MatchPredictor | None = None,
    profile_cache: dict[str, dict] | None = None,
) -> list[dict]:
    results = []
    predictor = predictor or MatchPredictor(matches, team_metrics=team_metrics)
    profile_cache = profile_cache or {}
    def get_profile(team: str) -> dict:
        if team not in profile_cache:
            profile_cache[team] = build_team_profile(matches, team, team_metrics.get(team, {}))
        return profile_cache[team]

    for fixture in fixtures:
        result = simulate_advanced_match(
            matches,
            team_metrics,
            fixture["home_team"],
            fixture["away_team"],
            simulations=simulations,
            mode=mode,
            venue=fixture.get("venue"),
            match_context=fixture.get("match_context"),
            predictor=predictor,
            home_profile=get_profile(fixture["home_team"]),
            away_profile=get_profile(fixture["away_team"]),
        )
        sim = result["simulation"]
        winner = "home_team" if sim["home_win"] >= sim["away_win"] else "away_team"
        results.append(
            {
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "projected_winner": fixture[winner],
                "home_win": sim["home_win"],
                "draw": sim["draw"],
                "away_win": sim["away_win"],
                "top_score": result["top_scores"][0] if result["top_scores"] else None,
            }
        )
    return results


def poker_factor_noise(rng: random.Random, mode: str) -> float:
    if mode == "classic":
        return 0.0
    volatility = 0.035 if mode == "hybrid" else 0.065
    cards = [
        rng.gauss(0, volatility),
        rng.gauss(0, volatility * 0.8),
        rng.gauss(0, volatility * 0.6),
        rng.choice([-1, 1]) * rng.random() * volatility,
    ]
    return sum(cards) / len(cards)


def adjusted_outcome_probabilities(prediction: dict, edge: float) -> dict[str, float]:
    home = max(prediction["home_win_prob"] + edge, 0.03)
    draw = max(prediction["draw_prob"] - abs(edge) * 0.15, 0.08)
    away = max(prediction["away_win_prob"] - edge, 0.03)
    total = home + draw + away
    return {"home_win": home / total, "draw": draw / total, "away_win": away / total}


def filter_scores_by_outcome(scores: list[tuple[int, int]], outcome: str) -> list[tuple[int, int]]:
    if outcome == "home_win":
        filtered = [score for score in scores if score[0] > score[1]]
    elif outcome == "away_win":
        filtered = [score for score in scores if score[0] < score[1]]
    else:
        filtered = [score for score in scores if score[0] == score[1]]
    return filtered or scores


def dominant_scenario(rng: random.Random, edge: float) -> str:
    if abs(edge) > 0.12:
        return "strength"
    return rng.choices(
        ["form", "fatigue", "availability", "context"],
        weights=[0.34, 0.24, 0.24, 0.18],
        k=1,
    )[0]
