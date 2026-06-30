COMPETITIONS = [
    {"region": "Europa", "name": "Premier League", "country": "England", "api_football_league_id": 39, "priority": 1, "status": "planned"},
    {"region": "Europa", "name": "LaLiga", "country": "Spain", "api_football_league_id": 140, "priority": 1, "status": "planned"},
    {"region": "Europa", "name": "Serie A", "country": "Italy", "api_football_league_id": 135, "priority": 1, "status": "planned"},
    {"region": "Europa", "name": "Bundesliga", "country": "Germany", "api_football_league_id": 78, "priority": 1, "status": "planned"},
    {"region": "Europa", "name": "Ligue 1", "country": "France", "api_football_league_id": 61, "priority": 2, "status": "planned"},
    {"region": "Europa", "name": "Champions League", "country": "Europe", "api_football_league_id": 2, "priority": 1, "status": "planned"},
    {"region": "Europa", "name": "Europa League", "country": "Europe", "api_football_league_id": 3, "priority": 2, "status": "planned"},
    {"region": "Europa", "name": "Conference League", "country": "Europe", "api_football_league_id": 848, "priority": 3, "status": "planned"},
    {"region": "America", "name": "Liga MX", "country": "Mexico", "api_football_league_id": 262, "priority": 1, "status": "planned"},
    {"region": "America", "name": "MLS", "country": "USA", "api_football_league_id": 253, "priority": 2, "status": "planned"},
    {"region": "America", "name": "Brasileirao", "country": "Brazil", "api_football_league_id": 71, "priority": 1, "status": "planned"},
    {"region": "America", "name": "Primera Division Argentina", "country": "Argentina", "api_football_league_id": 128, "priority": 1, "status": "planned"},
    {"region": "America", "name": "Copa Libertadores", "country": "South America", "api_football_league_id": 13, "priority": 1, "status": "planned"},
    {"region": "America", "name": "Copa Sudamericana", "country": "South America", "api_football_league_id": 11, "priority": 2, "status": "planned"},
    {"region": "America", "name": "CONCACAF Champions Cup", "country": "North America", "api_football_league_id": 16, "priority": 2, "status": "planned"},
    {"region": "Selecciones", "name": "World Cup", "country": "World", "api_football_league_id": 1, "priority": 1, "status": "partial"},
    {"region": "Selecciones", "name": "Copa America", "country": "South America", "api_football_league_id": 9, "priority": 1, "status": "planned"},
    {"region": "Selecciones", "name": "Euro", "country": "Europe", "api_football_league_id": 4, "priority": 1, "status": "planned"},
    {"region": "Selecciones", "name": "CONCACAF Gold Cup", "country": "North America", "api_football_league_id": 22, "priority": 2, "status": "planned"},
    {"region": "Selecciones", "name": "World Cup Qualification CONMEBOL", "country": "South America", "api_football_league_id": 34, "priority": 1, "status": "planned"},
    {"region": "Selecciones", "name": "World Cup Qualification CONCACAF", "country": "North America", "api_football_league_id": 31, "priority": 1, "status": "planned"},
]


def competition_catalog() -> dict:
    regions: dict[str, list[dict]] = {}
    for competition in COMPETITIONS:
        regions.setdefault(competition["region"], []).append(competition)
    return {
        "competitions": COMPETITIONS,
        "regions": regions,
        "totals": {
            "competitions": len(COMPETITIONS),
            "regions": len(regions),
            "planned": sum(1 for item in COMPETITIONS if item["status"] == "planned"),
            "partial": sum(1 for item in COMPETITIONS if item["status"] == "partial"),
        },
    }
