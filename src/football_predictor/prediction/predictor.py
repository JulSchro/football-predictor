import pandas as pd

from football_predictor.data.validators import Prediction
from football_predictor.models.baseline_elo import EloBaseline
from football_predictor.models.baseline_poisson import PoissonBaseline
from football_predictor.prediction.adjustments import apply_external_metrics_adjustment


class MatchPredictor:
    def __init__(self, matches: pd.DataFrame | None = None, team_metrics: dict[str, dict] | None = None) -> None:
        self.matches = matches if matches is not None else pd.DataFrame()
        self.team_metrics = team_metrics or {}
        self.poisson = PoissonBaseline().fit(self.matches)
        self.elo = EloBaseline(initial_ratings=fifa_seed_ratings(self.team_metrics)).fit(self.matches)

    def predict(self, home_team: str, away_team: str, model_name: str = "poisson_elo") -> Prediction:
        poisson = self.poisson.predict(home_team, away_team)
        elo = self.elo.predict_1x2(home_team, away_team)

        home = (float(poisson["home_win_prob"]) + elo["home_win_prob"]) / 2
        draw = (float(poisson["draw_prob"]) + elo["draw_prob"]) / 2
        away = (float(poisson["away_win_prob"]) + elo["away_win_prob"]) / 2
        probabilities = apply_external_metrics_adjustment(
            {"home_win_prob": home, "draw_prob": draw, "away_win_prob": away},
            self.team_metrics.get(home_team, {}),
            self.team_metrics.get(away_team, {}),
        )

        return Prediction(
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
            home_win_prob=probabilities["home_win_prob"],
            draw_prob=probabilities["draw_prob"],
            away_win_prob=probabilities["away_win_prob"],
            most_likely_score=str(poisson["most_likely_score"]),
            over_2_5_prob=float(poisson["over_2_5_prob"]),
            both_teams_score_prob=float(poisson["both_teams_score_prob"]),
        )


def fifa_seed_ratings(team_metrics: dict[str, dict]) -> dict[str, float]:
    ratings = {}
    for team, metrics in team_metrics.items():
        rank = metrics.get("fifa_rank")
        points = metrics.get("fifa_points")
        if points:
            ratings[team] = 1300 + min(max(float(points) - 1200, 0), 750) * 0.75
        elif rank:
            ratings[team] = 1850 - min(float(rank), 150) * 4
    return ratings
