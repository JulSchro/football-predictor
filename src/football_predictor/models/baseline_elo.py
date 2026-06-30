import pandas as pd


class EloBaseline:
    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 24.0,
        home_advantage: float = 60.0,
        initial_ratings: dict[str, float] | None = None,
    ) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_ratings = initial_ratings or {}
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_ratings.get(team, self.initial_rating))

    def expected_home(self, home_team: str, away_team: str) -> float:
        home_rating = self.rating(home_team) + self.home_advantage
        away_rating = self.rating(away_team)
        return 1 / (1 + 10 ** ((away_rating - home_rating) / 400))

    def fit(self, matches: pd.DataFrame) -> "EloBaseline":
        played = matches.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
        for _, row in played.iterrows():
            home = str(row["home_team"])
            away = str(row["away_team"])
            expected = self.expected_home(home, away)
            actual = 1.0 if row["home_goals"] > row["away_goals"] else 0.0 if row["home_goals"] < row["away_goals"] else 0.5
            change = self.k_factor * (actual - expected)
            self.ratings[home] = self.rating(home) + change
            self.ratings[away] = self.rating(away) - change
        return self

    def predict_1x2(self, home_team: str, away_team: str, draw_base: float = 0.26) -> dict[str, float]:
        home_strength = self.expected_home(home_team, away_team)
        draw = draw_base
        remaining = 1 - draw
        return {
            "home_win_prob": home_strength * remaining,
            "draw_prob": draw,
            "away_win_prob": (1 - home_strength) * remaining,
        }
