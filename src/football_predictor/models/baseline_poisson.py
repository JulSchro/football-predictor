from math import exp, factorial

import pandas as pd

from football_predictor.features.team_features import average_home_away_goals, recent_form


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(lam, 0.05)
    return (lam**k * exp(-lam)) / factorial(k)


class PoissonBaseline:
    def __init__(self, max_goals: int = 6) -> None:
        self.max_goals = max_goals
        self.history: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame) -> "PoissonBaseline":
        self.history = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        return self

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        if self.history is None or self.history.empty:
            return 1.2, 1.0

        league = average_home_away_goals(self.history)
        home = recent_form(self.history, home_team)
        away = recent_form(self.history, away_team)

        home_xg = (home["avg_goals_for"] + away["avg_goals_against"] + league["home_goals_avg"]) / 3
        away_xg = (away["avg_goals_for"] + home["avg_goals_against"] + league["away_goals_avg"]) / 3
        return max(home_xg, 0.2), max(away_xg, 0.2)

    def predict_score_matrix(self, home_team: str, away_team: str) -> dict[tuple[int, int], float]:
        home_xg, away_xg = self.expected_goals(home_team, away_team)
        scores = {}
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                scores[(h, a)] = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)
        total = sum(scores.values())
        return {score: prob / total for score, prob in scores.items()}

    def predict(self, home_team: str, away_team: str) -> dict[str, float | str]:
        matrix = self.predict_score_matrix(home_team, away_team)
        home_win = sum(p for (h, a), p in matrix.items() if h > a)
        draw = sum(p for (h, a), p in matrix.items() if h == a)
        away_win = sum(p for (h, a), p in matrix.items() if h < a)
        best_score = max(matrix, key=matrix.get)
        return {
            "home_win_prob": home_win,
            "draw_prob": draw,
            "away_win_prob": away_win,
            "most_likely_score": f"{best_score[0]}-{best_score[1]}",
            "over_2_5_prob": sum(p for (h, a), p in matrix.items() if h + a > 2.5),
            "both_teams_score_prob": sum(p for (h, a), p in matrix.items() if h > 0 and a > 0),
        }

