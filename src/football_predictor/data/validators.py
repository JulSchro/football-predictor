from datetime import date
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Team(BaseModel):
    name: str = Field(min_length=1)
    country: str | None = None


class Match(BaseModel):
    date: date
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    home_goals: int | None = Field(default=None, ge=0)
    away_goals: int | None = Field(default=None, ge=0)
    competition: str = Field(min_length=1)
    season: str = Field(min_length=1)

    @model_validator(mode="after")
    def different_teams(self) -> "Match":
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must be different")
        return self


class TeamMatchStats(BaseModel):
    match_id: int
    team: str = Field(min_length=1)
    is_home: bool
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    shots: int | None = Field(default=None, ge=0)
    shots_on_target: int | None = Field(default=None, ge=0)
    possession: float | None = Field(default=None, ge=0, le=100)


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    home_team: str
    away_team: str
    model_name: str
    home_win_prob: float = Field(ge=0, le=1)
    draw_prob: float = Field(ge=0, le=1)
    away_win_prob: float = Field(ge=0, le=1)
    most_likely_score: str
    over_2_5_prob: float = Field(ge=0, le=1)
    both_teams_score_prob: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def probabilities_sum(self) -> "Prediction":
        total = self.home_win_prob + self.draw_prob + self.away_win_prob
        if abs(total - 1.0) > 0.02:
            raise ValueError("1X2 probabilities must sum to 1")
        return self

