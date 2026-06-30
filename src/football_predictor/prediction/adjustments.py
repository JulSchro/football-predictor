from math import log10


def apply_external_metrics_adjustment(probabilities: dict[str, float], home_metrics: dict, away_metrics: dict) -> dict[str, float]:
    edge = 0.0

    if home_metrics.get("fifa_rank") and away_metrics.get("fifa_rank"):
        edge += (float(away_metrics["fifa_rank"]) - float(home_metrics["fifa_rank"])) * 0.0025

    if home_metrics.get("fifa_points") and away_metrics.get("fifa_points"):
        edge += (float(home_metrics["fifa_points"]) - float(away_metrics["fifa_points"])) * 0.0008

    if home_metrics.get("squad_value_eur") and away_metrics.get("squad_value_eur"):
        home_value = max(float(home_metrics["squad_value_eur"]), 1.0)
        away_value = max(float(away_metrics["squad_value_eur"]), 1.0)
        edge += log10(home_value / away_value) * 0.08

    if home_metrics.get("api_points_per_match") is not None and away_metrics.get("api_points_per_match") is not None:
        edge += (float(home_metrics["api_points_per_match"]) - float(away_metrics["api_points_per_match"])) * 0.025

    if home_metrics.get("api_goal_diff_per_match") is not None and away_metrics.get("api_goal_diff_per_match") is not None:
        edge += (float(home_metrics["api_goal_diff_per_match"]) - float(away_metrics["api_goal_diff_per_match"])) * 0.018

    if home_metrics.get("standing_rank") is not None and away_metrics.get("standing_rank") is not None:
        edge += (float(away_metrics["standing_rank"]) - float(home_metrics["standing_rank"])) * 0.004

    if home_metrics.get("unavailable_players") is not None and away_metrics.get("unavailable_players") is not None:
        edge += (float(away_metrics["unavailable_players"]) - float(home_metrics["unavailable_players"])) * 0.006

    if home_metrics.get("missing_player_importance") is not None and away_metrics.get("missing_player_importance") is not None:
        edge += (float(away_metrics["missing_player_importance"]) - float(home_metrics["missing_player_importance"])) * 0.004

    if home_metrics.get("player_rating_score") is not None and away_metrics.get("player_rating_score") is not None:
        edge += (float(home_metrics["player_rating_score"]) - float(away_metrics["player_rating_score"])) * 0.0012

    if home_metrics.get("player_depth_score") is not None and away_metrics.get("player_depth_score") is not None:
        edge += (float(home_metrics["player_depth_score"]) - float(away_metrics["player_depth_score"])) * 0.0008

    if home_metrics.get("player_attack_output_score") is not None and away_metrics.get("player_attack_output_score") is not None:
        edge += (float(home_metrics["player_attack_output_score"]) - float(away_metrics["player_attack_output_score"])) * 0.001

    edge = max(min(edge, 0.16), -0.16)
    adjusted = probabilities.copy()
    adjusted["home_win_prob"] = max(adjusted["home_win_prob"] + edge, 0.02)
    adjusted["away_win_prob"] = max(adjusted["away_win_prob"] - edge, 0.02)

    total = adjusted["home_win_prob"] + adjusted["draw_prob"] + adjusted["away_win_prob"]
    return {key: value / total for key, value in adjusted.items()}
