from football_predictor.prediction.monte_carlo import simulate_market_distributions
from football_predictor.prediction.pick_engine import generate_betting_picks, generate_pick_report


def test_market_monte_carlo_returns_line_distribution() -> None:
    prediction = {
        "home_win_prob": 0.52,
        "draw_prob": 0.26,
        "away_win_prob": 0.22,
        "over_2_5_prob": 0.54,
        "both_teams_score_prob": 0.48,
    }
    markets = {
        "home_xg_recent": 1.6,
        "away_xg_recent": 0.9,
        "home_corners_expected": 4.4,
        "away_corners_expected": 3.6,
        "total_cards_expected": 3.1,
        "home_shots_on_target_expected": 4.6,
        "away_shots_on_target_expected": 3.1,
    }

    result = simulate_market_distributions(prediction, markets, simulations=3000, seed=7)

    assert result["corners"]["expected"] > 6
    assert "8_5" in result["corners"]["lines"]
    assert 0 <= result["corners"]["lines"]["8_5"]["over"] <= 1
    assert "home" in result["team_markets"]


def test_pick_engine_avoids_corner_line_when_expected_is_too_close() -> None:
    prediction = {
        "home_win_prob": 0.62,
        "draw_prob": 0.23,
        "away_win_prob": 0.15,
        "over_2_5_prob": 0.5,
        "both_teams_score_prob": 0.46,
    }
    markets = {
        "home_xg_recent": 1.4,
        "away_xg_recent": 0.8,
        "home_corners_expected": 4.4,
        "away_corners_expected": 3.6,
        "total_corners_expected": 8.0,
        "total_cards_expected": 2.8,
        "home_shots_on_target_expected": 4.2,
        "away_shots_on_target_expected": 2.7,
    }
    distribution = simulate_market_distributions(prediction, markets, simulations=5000, seed=11)

    picks = generate_betting_picks(
        "A",
        "B",
        0.62,
        0.23,
        0.15,
        0.5,
        0.46,
        markets,
        distribution,
        max_picks=10,
    )

    assert not any(pick["market"] == "Corners" and pick["pick"] == "Over 8.5" for pick in picks)
    assert any(pick["market"] == "Doble oportunidad" and pick["tier"] == "seguro" for pick in picks)
    assert all("tier" in pick for pick in picks)


def test_pick_report_explains_discarded_and_no_bet_markets() -> None:
    prediction = {
        "home_win_prob": 0.58,
        "draw_prob": 0.25,
        "away_win_prob": 0.17,
        "over_2_5_prob": 0.51,
        "both_teams_score_prob": 0.48,
    }
    markets = {
        "home_xg_recent": 1.2,
        "away_xg_recent": 0.9,
        "home_corners_expected": 4.2,
        "away_corners_expected": 3.8,
        "total_cards_expected": 2.7,
        "home_shots_on_target_expected": 3.6,
        "away_shots_on_target_expected": 2.8,
    }
    distribution = simulate_market_distributions(prediction, markets, simulations=4000, seed=13)

    report = generate_pick_report("A", "B", 0.58, 0.25, 0.17, 0.51, 0.48, markets, distribution)

    assert report["recommended"]
    assert report["discarded"]
    assert isinstance(report["no_bet"], list)
    assert any(item.get("discard_reasons") for item in report["discarded"])
    assert report["summary"]["method"] == "monte_carlo_pick_report_v1"
