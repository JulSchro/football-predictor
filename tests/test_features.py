import pandas as pd

from football_predictor.features.match_features import FEATURE_COLUMNS, build_match_features, build_training_matrix
from football_predictor.features.team_features import recent_form


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2024-01-01", "A", "B", 2, 1, "League", "2024"],
            ["2024-01-08", "C", "A", 1, 1, "League", "2024"],
            ["2024-01-15", "B", "C", 0, 2, "League", "2024"],
            ["2024-01-22", "A", "C", 3, 0, "League", "2024"],
        ],
        columns=["date", "home_team", "away_team", "home_goals", "away_goals", "competition", "season"],
    )


def test_recent_form_returns_expected_keys() -> None:
    form = recent_form(sample_matches(), "A", n=3)
    assert form["avg_goals_for"] == 2.0
    assert form["win_rate"] == 2 / 3


def test_build_match_features_has_stable_columns() -> None:
    features = build_match_features(sample_matches(), "A", "B")
    assert list(features.keys()) == FEATURE_COLUMNS
    assert features["is_home"] == 1.0
    assert "elo_diff" in features
    assert "weighted_form_diff" in features


def test_build_training_matrix_creates_labels() -> None:
    x, y = build_training_matrix(sample_matches())
    assert x.shape == (4, len(FEATURE_COLUMNS))
    assert set(y) == {"H", "D", "A"}
