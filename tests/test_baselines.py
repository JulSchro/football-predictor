import pandas as pd

from football_predictor.models.baseline_elo import EloBaseline
from football_predictor.models.baseline_poisson import PoissonBaseline, poisson_pmf


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2024-01-01", "A", "B", 2, 1, "League", "2024"],
            ["2024-01-08", "B", "A", 0, 1, "League", "2024"],
            ["2024-01-15", "A", "C", 1, 1, "League", "2024"],
        ],
        columns=["date", "home_team", "away_team", "home_goals", "away_goals", "competition", "season"],
    )


def test_poisson_pmf_positive() -> None:
    assert poisson_pmf(1, 1.2) > 0


def test_poisson_prediction_probabilities_sum_to_one() -> None:
    prediction = PoissonBaseline().fit(sample_matches()).predict("A", "B")
    total = prediction["home_win_prob"] + prediction["draw_prob"] + prediction["away_win_prob"]
    assert abs(float(total) - 1.0) < 0.001
    assert "-" in str(prediction["most_likely_score"])


def test_elo_updates_ratings_and_predicts() -> None:
    elo = EloBaseline().fit(sample_matches())
    probs = elo.predict_1x2("A", "B")
    assert elo.rating("A") > elo.initial_rating
    assert abs(sum(probs.values()) - 1.0) < 0.001

