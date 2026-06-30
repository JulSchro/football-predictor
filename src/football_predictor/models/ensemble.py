import pandas as pd

from football_predictor.data.validators import Prediction
from football_predictor.models.ml_model import MatchOutcomeModel
from football_predictor.prediction.predictor import MatchPredictor


class EnsemblePredictor:
    def __init__(self, ml_weight: float = 0.35) -> None:
        self.ml_weight = ml_weight
        self.base: MatchPredictor | None = None
        self.ml: MatchOutcomeModel | None = None
        self.has_ml = False

    def fit(self, matches: pd.DataFrame, team_metrics: dict[str, dict] | None = None) -> "EnsemblePredictor":
        self.base = MatchPredictor(matches, team_metrics=team_metrics)
        try:
            self.ml = MatchOutcomeModel().fit(matches)
            self.has_ml = True
        except ValueError:
            self.ml = None
            self.has_ml = False
        return self

    def predict(self, home_team: str, away_team: str) -> Prediction:
        if self.base is None:
            raise ValueError("EnsemblePredictor is not fitted")
        base = self.base.predict(home_team, away_team)
        if not self.has_ml or self.ml is None:
            return base

        ml = self.ml.predict_proba(home_team, away_team)
        base_weight = 1.0 - self.ml_weight
        home = base.home_win_prob * base_weight + ml["home_win_prob"] * self.ml_weight
        draw = base.draw_prob * base_weight + ml["draw_prob"] * self.ml_weight
        away = base.away_win_prob * base_weight + ml["away_win_prob"] * self.ml_weight
        total = home + draw + away
        return Prediction(
            home_team=home_team,
            away_team=away_team,
            model_name="ensemble_poisson_elo_ml",
            home_win_prob=home / total,
            draw_prob=draw / total,
            away_win_prob=away / total,
            most_likely_score=base.most_likely_score,
            over_2_5_prob=base.over_2_5_prob,
            both_teams_score_prob=base.both_teams_score_prob,
        )
