from __future__ import annotations

from football_predictor.prediction.markets import likely_range, poisson_over


MARKET_MAP = {
    "corners": "total_corners_expected",
    "shots_on_target": "shots_on_target_expected",
    "cards": "total_cards_expected",
}


def calibration_adjustments(backtest_metrics: dict) -> dict[str, float]:
    markets = backtest_metrics.get("market_metrics") or {}
    adjustments = {}
    for name, market_key in MARKET_MAP.items():
        bias = (markets.get(name) or {}).get("bias")
        if bias is not None:
            adjustments[market_key] = -float(bias)
    return adjustments


def apply_market_calibration(markets: dict, backtest_metrics: dict) -> dict:
    calibrated = dict(markets)
    adjustments = calibration_adjustments(backtest_metrics)
    for key, adjustment in adjustments.items():
        if calibrated.get(key) is not None:
            previous = float(calibrated[key])
            calibrated[key] = round(max(0.0, float(calibrated[key]) + adjustment), 2)
            _scale_team_market(calibrated, key, previous, float(calibrated[key]))
    calibrated["calibration"] = {
        "applied": bool(adjustments),
        "adjustments": {key: round(value, 4) for key, value in adjustments.items()},
    }
    return calibrated


def _scale_team_market(markets: dict, total_key: str, previous_total: float, new_total: float) -> None:
    if previous_total <= 0:
        return
    scale = new_total / previous_total
    if total_key == "total_corners_expected":
        fields = ("home_corners_expected", "away_corners_expected")
        nested_field = "corners_expected"
    elif total_key == "shots_on_target_expected":
        fields = ("home_shots_on_target_expected", "away_shots_on_target_expected")
        nested_field = "shots_on_target_expected"
    else:
        return
    for field in fields:
        if markets.get(field) is not None:
            markets[field] = round(max(0.0, float(markets[field]) * scale), 2)
    team_markets = markets.get("team_markets") or {}
    for side in ("home", "away"):
        side_data = team_markets.get(side) or {}
        if side_data.get(nested_field) is not None:
            value = round(max(0.0, float(side_data[nested_field]) * scale), 2)
            side_data[nested_field] = value
            if nested_field == "corners_expected":
                side_data["corners_range"] = likely_range(value)
                side_data["over_3_5_corners_prob"] = poisson_over(value, 3.5)
                side_data["over_4_5_corners_prob"] = poisson_over(value, 4.5)
            elif nested_field == "shots_on_target_expected":
                side_data["shots_on_target_range"] = likely_range(value)
                side_data["over_3_5_shots_on_target_prob"] = poisson_over(value, 3.5)
                side_data["over_4_5_shots_on_target_prob"] = poisson_over(value, 4.5)
