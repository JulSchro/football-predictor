from __future__ import annotations


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
            calibrated[key] = round(max(0.0, float(calibrated[key]) + adjustment), 2)
    calibrated["calibration"] = {
        "applied": bool(adjustments),
        "adjustments": {key: round(value, 4) for key, value in adjustments.items()},
    }
    return calibrated
