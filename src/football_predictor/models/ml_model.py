import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_predictor.features.match_features import FEATURE_COLUMNS, build_match_features, build_training_matrix


class MatchOutcomeModel:
    def __init__(self, model_type: str = "logistic") -> None:
        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500)),
                ]
            )
        self.classes_: list[str] = []
        self.history: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame) -> "MatchOutcomeModel":
        x, y = build_training_matrix(matches)
        if y.nunique() < 2:
            raise ValueError("Need at least two outcome classes to train the ML model")
        self.model.fit(x, y)
        self.classes_ = list(self.model.classes_)
        self.history = matches.copy()
        return self

    def predict_proba(self, home_team: str, away_team: str) -> dict[str, float]:
        if self.history is None:
            raise ValueError("Model is not fitted")
        features = build_match_features(self.history, home_team, away_team)
        x = pd.DataFrame([features], columns=FEATURE_COLUMNS).fillna(0)
        probabilities = self.model.predict_proba(x)[0]
        mapped = dict(zip(self.classes_, probabilities))
        return {
            "home_win_prob": float(mapped.get("H", 0.0)),
            "draw_prob": float(mapped.get("D", 0.0)),
            "away_win_prob": float(mapped.get("A", 0.0)),
        }
