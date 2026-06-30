HOST_COUNTRY_CODES = {
    "Mexico": "MEX",
    "United States": "USA",
    "Canada": "CAN",
}


def venue_edge(home_team: str, away_team: str, venue: dict | None, team_metrics: dict[str, dict]) -> float:
    if not venue:
        return 0.0

    country = venue.get("country")
    altitude = float(venue.get("altitude_m") or 0)
    home_code = team_metrics.get(home_team, {}).get("raw", {}).get("country_code")
    away_code = team_metrics.get(away_team, {}).get("raw", {}).get("country_code")

    edge = 0.0
    if home_code and country == home_code:
        edge += 0.07
    if away_code and country == away_code:
        edge -= 0.07

    if home_team == "Mexico" and country == "MEX":
        edge += 0.05
    elif home_team == "Mexico" and country == "USA":
        edge += 0.025
    if away_team == "Mexico" and country == "MEX":
        edge -= 0.05
    elif away_team == "Mexico" and country == "USA":
        edge -= 0.025

    if altitude >= 1500:
        if home_team == "Mexico":
            edge += 0.025
        if away_team == "Mexico":
            edge -= 0.025

    return max(min(edge, 0.12), -0.12)

