import pandas as pd

from football_predictor.data.validators import Match
from football_predictor.prediction.adjustments import apply_external_metrics_adjustment
from football_predictor.prediction.predictor import MatchPredictor
from football_predictor.prediction.factor_model import build_team_profile
from football_predictor.prediction.match_context import normalize_competition_context
from football_predictor.prediction.simulator import simulate_advanced_match, simulate_fixture_list, simulate_match
from football_predictor.models.ensemble import EnsemblePredictor


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2024-01-01", "A", "B", 2, 1, "League", "2024"],
            ["2024-01-08", "B", "C", 1, 1, "League", "2024"],
            ["2024-01-15", "C", "A", 0, 2, "League", "2024"],
        ],
        columns=["date", "home_team", "away_team", "home_goals", "away_goals", "competition", "season"],
    )


def test_match_validator_rejects_same_team() -> None:
    try:
        Match(
            date="2024-01-01",
            home_team="A",
            away_team="A",
            home_goals=1,
            away_goals=0,
            competition="League",
            season="2024",
        )
    except ValueError:
        assert True
    else:
        assert False


def test_predictor_returns_expected_payload() -> None:
    prediction = MatchPredictor(sample_matches()).predict("A", "B")
    assert prediction.home_team == "A"
    assert prediction.away_team == "B"
    assert prediction.most_likely_score
    assert abs(prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob - 1.0) < 0.001


def test_external_metrics_adjustment_keeps_probabilities_valid() -> None:
    probs = {"home_win_prob": 0.4, "draw_prob": 0.3, "away_win_prob": 0.3}
    adjusted = apply_external_metrics_adjustment(
        probs,
        {"fifa_rank": 1, "fifa_points": 1900, "squad_value_eur": 900_000_000},
        {"fifa_rank": 40, "fifa_points": 1500, "squad_value_eur": 120_000_000},
    )
    assert adjusted["home_win_prob"] > probs["home_win_prob"]
    assert abs(sum(adjusted.values()) - 1.0) < 0.001


def test_api_metrics_adjustment_changes_probabilities() -> None:
    probs = {"home_win_prob": 0.4, "draw_prob": 0.3, "away_win_prob": 0.3}
    adjusted = apply_external_metrics_adjustment(
        probs,
        {"api_points_per_match": 2.5, "api_goal_diff_per_match": 1.2, "standing_rank": 1, "unavailable_players": 0},
        {"api_points_per_match": 0.8, "api_goal_diff_per_match": -0.6, "standing_rank": 8, "unavailable_players": 3},
    )
    assert adjusted["home_win_prob"] > probs["home_win_prob"]
    assert abs(sum(adjusted.values()) - 1.0) < 0.001


def test_player_metrics_adjustment_changes_probabilities() -> None:
    probs = {"home_win_prob": 0.4, "draw_prob": 0.3, "away_win_prob": 0.3}
    adjusted = apply_external_metrics_adjustment(
        probs,
        {"player_rating_score": 82, "player_depth_score": 78, "player_attack_output_score": 74, "missing_player_importance": 0},
        {"player_rating_score": 68, "player_depth_score": 55, "player_attack_output_score": 58, "missing_player_importance": 5},
    )
    assert adjusted["home_win_prob"] > probs["home_win_prob"]
    assert abs(sum(adjusted.values()) - 1.0) < 0.001


def test_simulator_returns_distribution() -> None:
    predictor = MatchPredictor(sample_matches())
    result = simulate_match(predictor, "A", "B", simulations=1000)
    assert result["simulations"] == 1000
    assert len(result["top_scores"]) > 0
    assert 0 <= result["simulation"]["home_win"] <= 1


def test_advanced_simulator_returns_profiles_and_factor_cards() -> None:
    result = simulate_advanced_match(sample_matches(), {}, "A", "B", simulations=1000, mode="poker")
    assert result["mode"] == "poker"
    assert len(result["factor_cards"]) > 0
    assert len(result["sensitivity"]) > 0
    assert result["profiles"]["home"]["team"] == "A"
    assert 0 <= result["simulation"]["draw"] <= 1


def test_match_context_classifies_competitive_stage() -> None:
    group = normalize_competition_context("World Cup", "Group Stage - 2")
    final = normalize_competition_context("World Cup", "Final")
    friendly = normalize_competition_context("Friendlies", "Friendlies 1")

    assert group["is_group_stage"] is True
    assert final["is_knockout"] is True
    assert final["pressure_index"] > group["pressure_index"]
    assert friendly["is_friendly"] is True
    assert friendly["rotation_risk"] > final["rotation_risk"]
    second_leg = normalize_competition_context("Champions League", "Semi-finals - 2nd leg")
    assert second_leg["is_return_leg"] is True
    assert second_leg["draw_acceptance"] >= 58


def test_advanced_simulator_includes_match_context() -> None:
    context = normalize_competition_context("World Cup", "Round of 16")
    result = simulate_advanced_match(sample_matches(), {}, "A", "B", simulations=1000, match_context=context)

    assert result["match_context"]["stage"] == "round_of_16"
    assert any(card["name"] == "Contexto" for card in result["factor_cards"])
    assert any(row["factor"] == "Contexto competitivo" for row in result["sensitivity"])


def test_team_profile_exposes_requested_factor_groups() -> None:
    profile = build_team_profile(sample_matches(), "A")
    assert "strength" in profile
    assert "last_5" in profile["form"]
    assert "players" in profile["pending_data"]
    assert "opponent_adjusted_form" in profile["generated"]
    assert "tactical_reliability" in profile["generated"]
    assert "style" in profile


def test_team_profile_uses_api_enrichment_metrics() -> None:
    profile = build_team_profile(
        sample_matches(),
        "A",
        {
            "api_played": 6,
            "api_points_per_match": 2.4,
            "api_win_rate": 0.75,
            "api_goal_diff_per_match": 1.2,
            "api_goals_for_per_match": 2.1,
            "api_goals_against_per_match": 0.7,
            "api_form_score": 85,
            "standing_rank": 1,
            "standing_points_per_match": 2.4,
            "lineup_distinct_players": 20,
            "possession_for": 58,
            "pass_accuracy_for": 87,
            "shots_for": 15,
            "shots_on_target_for": 6,
            "corners_for": 7,
            "fouls_for": 9,
            "cards_for": 1.5,
            "player_rating_score": 82,
            "player_depth_score": 76,
            "player_attack_output_score": 74,
            "player_minutes_per_player": 1200,
        },
    )
    assert profile["strength"]["api_strength_score"] is not None
    assert profile["strength"]["api_trust"] == 1.0
    assert profile["generated"]["squad_stability"] > 50
    assert profile["generated"]["tactical_reliability"] > 50
    assert profile["style"]["set_piece_threat"] > 50
    assert profile["generated"]["player_quality_score"] > 50
    assert profile["generated"]["squad_depth_score"] > 50


def test_ensemble_predictor_returns_probabilities() -> None:
    prediction = EnsemblePredictor().fit(sample_matches()).predict("A", "B")
    assert prediction.model_name in {"ensemble_poisson_elo_ml", "poisson_elo"}
    assert abs(prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob - 1.0) < 0.001


def test_fixture_list_simulation_returns_winners() -> None:
    result = simulate_fixture_list(
        sample_matches(),
        {},
        [{"home_team": "A", "away_team": "B"}, {"home_team": "B", "away_team": "C"}],
        simulations=1000,
    )
    assert len(result) == 2
    assert "projected_winner" in result[0]
