from __future__ import annotations


def generate_betting_picks(
    home_team: str,
    away_team: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    over_2_5_prob: float,
    btts_prob: float,
    markets: dict,
    market_distribution: dict | None = None,
    max_picks: int = 7,
) -> list[dict]:
    return generate_pick_report(
        home_team,
        away_team,
        home_prob,
        draw_prob,
        away_prob,
        over_2_5_prob,
        btts_prob,
        markets,
        market_distribution,
        max_picks=max_picks,
    )["recommended"]


def generate_pick_report(
    home_team: str,
    away_team: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    over_2_5_prob: float,
    btts_prob: float,
    markets: dict,
    market_distribution: dict | None = None,
    max_picks: int = 7,
) -> dict:
    picks: list[dict] = []
    picks.extend(_outcome_picks(home_team, away_team, home_prob, draw_prob, away_prob))
    discarded: list[dict] = []
    distribution = market_distribution or {}

    goals = distribution.get("goals")
    if goals:
        selected, rejected = _line_selection("Goles", goals, lines=[1.5, 2.5, 3.5], min_distance=0.65)
        picks.extend(selected)
        discarded.extend(rejected)
    else:
        picks.extend(_fallback_goal_picks(over_2_5_prob, btts_prob))

    if distribution.get("corners"):
        selected, rejected = _line_selection("Corners", distribution["corners"], lines=[5.5, 6.5, 7.5, 8.5, 9.5], min_distance=0.85)
        picks.extend(selected)
        discarded.extend(rejected)
    elif markets.get("over_8_5_corners_prob") is not None:
        picks.append(_build_pick("Corners", "Over 8.5", float(markets["over_8_5_corners_prob"]), "medio", source="legacy"))

    if distribution.get("cards"):
        selected, rejected = _line_selection("Tarjetas", distribution["cards"], lines=[1.5, 2.5, 3.5, 4.5], min_distance=0.75)
        picks.extend(selected)
        discarded.extend(rejected)
    elif markets.get("over_3_5_cards_prob") is not None:
        picks.append(_build_pick("Tarjetas", "Over 3.5", float(markets["over_3_5_cards_prob"]), "alto", source="legacy"))

    team_distribution = distribution.get("team_markets") or {}
    for side, label in (("home", home_team), ("away", away_team)):
        team = team_distribution.get(side) or {}
        if team.get("corners"):
            selected, rejected = _line_selection(f"Corners {label}", team["corners"], lines=[2.5, 3.5, 4.5, 5.5], min_distance=0.65)
            picks.extend(selected)
            discarded.extend(rejected)
        if team.get("shots_on_target"):
            selected, rejected = _line_selection(f"Tiros puerta {label}", team["shots_on_target"], lines=[1.5, 2.5, 3.5, 4.5], min_distance=0.55)
            picks.extend(selected)
            discarded.extend(rejected)

    if not goals:
        picks.extend(_btts_picks(btts_prob))
    else:
        btts_distribution = distribution.get("outcomes", {}).get("both_teams_score")
        if btts_distribution is not None:
            picks.extend(_binary_picks("Ambos anotan", "Si", "No", float(btts_distribution)))

    clean = [pick for pick in picks if not pick.get("discarded")]
    clean.sort(key=lambda item: (_tier_rank(item["tier"]), item["probability"]), reverse=True)
    discarded.sort(key=lambda item: (item["market"], -item["probability"]))
    return {
        "recommended": clean[:max_picks],
        "discarded": discarded[:18],
        "no_bet": _no_bet_rows(clean, discarded),
        "summary": {
            "recommended": len(clean[:max_picks]),
            "discarded": len(discarded),
            "method": "monte_carlo_pick_report_v1" if distribution else "legacy_pick_report_v1",
        },
    }


def _outcome_picks(home_team: str, away_team: str, home_prob: float, draw_prob: float, away_prob: float) -> list[dict]:
    outcomes = [
        ("home", home_team, float(home_prob)),
        ("draw", "Empate", float(draw_prob)),
        ("away", away_team, float(away_prob)),
    ]
    ranked = sorted(outcomes, key=lambda item: item[2], reverse=True)
    top_key, top_label, top_prob = ranked[0]
    edge = top_prob - ranked[1][2]
    picks = [_build_pick("1X2", top_label if top_key != "draw" else "Empate", top_prob, "normal", edge=edge)]
    if top_key == "home":
        picks.append(_build_pick("Doble oportunidad", f"{home_team} o empate", float(home_prob) + float(draw_prob), "bajo", edge=0.12))
    elif top_key == "away":
        picks.append(_build_pick("Doble oportunidad", f"{away_team} o empate", float(away_prob) + float(draw_prob), "bajo", edge=0.12))
    return picks


def _line_picks(market: str, summary: dict, lines: list[float], min_distance: float) -> list[dict]:
    selected, _ = _line_selection(market, summary, lines, min_distance)
    return selected


def _line_selection(market: str, summary: dict, lines: list[float], min_distance: float) -> tuple[list[dict], list[dict]]:
    candidates = []
    expected = float(summary.get("expected") or 0)
    stability = summary.get("stability") or "baja"
    for line in lines:
        data = (summary.get("lines") or {}).get(_line_key(line)) or {}
        over = data.get("over")
        under = data.get("under")
        if over is not None:
            candidates.append(_market_pick(market, f"Over {line}", float(over), expected, line, stability, min_distance, "over"))
        if under is not None:
            candidates.append(_market_pick(market, f"Under {line}", float(under), expected, line, stability, min_distance, "under"))
    viable = [pick for pick in candidates if not pick["discarded"]]
    rejected = [pick for pick in candidates if pick["discarded"]]
    if not viable:
        return [], rejected
    viable.sort(key=lambda item: (_tier_rank(item["tier"]), item["probability"], item["line_distance"]), reverse=True)
    selected = viable[:1]
    selected_keys = {(item["market"], item["pick"]) for item in selected}
    rejected.extend([item for item in viable[1:] if (item["market"], item["pick"]) not in selected_keys])
    return selected, rejected


def _market_pick(
    market: str,
    pick: str,
    probability: float,
    expected: float,
    line: float,
    stability: str,
    min_distance: float,
    direction: str,
) -> dict:
    distance = abs(expected - line)
    wrong_side = (direction == "over" and expected <= line) or (direction == "under" and expected >= line)
    reasons = []
    if wrong_side:
        reasons.append("linea_del_lado_equivocado")
    if distance < min_distance:
        reasons.append("linea_demasiado_pegada")
    if probability < 0.52:
        reasons.append("probabilidad_baja")
    if stability == "baja" and probability < 0.68:
        reasons.append("mercado_inestable")
    tier = _tier(probability, stability, distance, min_distance)
    return {
        **_build_pick(market, pick, probability, _risk_from_tier(tier), edge=max(0.0, distance / max(expected, 1.0)), tier=tier, source="monte_carlo"),
        "expected": round(expected, 2),
        "line": line,
        "line_distance": round(distance, 2),
        "stability": stability,
        "discarded": bool(reasons),
        "discard_reasons": reasons,
    }


def _fallback_goal_picks(over_2_5_prob: float, btts_prob: float) -> list[dict]:
    picks = []
    if over_2_5_prob >= 0.58:
        picks.append(_build_pick("Goles", "Over 2.5", float(over_2_5_prob), "medio", source="legacy"))
    elif over_2_5_prob <= 0.42:
        picks.append(_build_pick("Goles", "Under 2.5", 1 - float(over_2_5_prob), "medio", source="legacy"))
    picks.extend(_btts_picks(btts_prob))
    return picks


def _btts_picks(btts_prob: float) -> list[dict]:
    return _binary_picks("Ambos anotan", "Si", "No", float(btts_prob))


def _binary_picks(market: str, yes_label: str, no_label: str, probability: float) -> list[dict]:
    if probability >= 0.58:
        return [_build_pick(market, yes_label, probability, "medio")]
    if probability <= 0.42:
        return [_build_pick(market, no_label, 1 - probability, "medio")]
    return []


def _build_pick(
    market: str,
    pick: str,
    probability: float,
    risk: str,
    edge: float = 0.0,
    tier: str | None = None,
    source: str = "model",
) -> dict:
    probability = round(max(0.0, min(float(probability), 1.0)), 4)
    tier = tier or _tier(probability, "media", edge, 0.08)
    return {
        "market": market,
        "pick": pick,
        "probability": probability,
        "confidence": _confidence_label(probability, edge),
        "risk": risk,
        "tier": tier,
        "source": source,
    }


def _tier(probability: float, stability: str, distance: float, min_distance: float) -> str:
    if probability >= 0.7 and stability in {"alta", "media"} and distance >= min_distance:
        return "seguro"
    if probability >= 0.6 and distance >= min_distance:
        return "valor"
    return "arriesgado"


def _risk_from_tier(tier: str) -> str:
    if tier == "seguro":
        return "bajo"
    if tier == "valor":
        return "medio"
    return "alto"


def _confidence_label(probability: float, edge: float = 0.0) -> str:
    if probability >= 0.7 or (probability >= 0.62 and edge >= 0.18):
        return "alta"
    if probability >= 0.58 or edge >= 0.1:
        return "media"
    return "baja"


def _tier_rank(tier: str) -> int:
    return {"arriesgado": 1, "valor": 2, "seguro": 3}.get(tier, 0)


def _no_bet_rows(recommended: list[dict], discarded: list[dict]) -> list[dict]:
    rows = []
    recommended_markets = {item["market"] for item in recommended}
    for market in ["Goles", "Corners", "Tarjetas"]:
        if market in recommended_markets:
            continue
        market_discards = [item for item in discarded if item["market"] == market]
        if not market_discards:
            continue
        top = sorted(market_discards, key=lambda item: item["probability"], reverse=True)[0]
        rows.append(
            {
                "market": market,
                "decision": "No bet",
                "best_candidate": top["pick"],
                "probability": top["probability"],
                "reasons": top.get("discard_reasons", []),
                "expected": top.get("expected"),
                "line": top.get("line"),
                "stability": top.get("stability"),
            }
        )
    return rows


def _line_key(line: float) -> str:
    return str(line).replace(".", "_")
