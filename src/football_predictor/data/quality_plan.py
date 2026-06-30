TEAM_ALIASES = {
    "Mexico": ["México", "MEX"],
    "United States": ["USA", "United States of America", "USMNT"],
    "England": ["ENG"],
    "France": ["FRA"],
    "Spain": ["España", "ESP"],
    "Germany": ["Alemania", "GER"],
    "Argentina": ["ARG"],
    "Brazil": ["Brasil", "BRA"],
    "Portugal": ["POR"],
    "Croatia": ["Hrvatska", "CRO"],
    "Netherlands": ["Holland", "Países Bajos", "NED"],
    "DR Congo": ["RD Congo", "Congo DR", "Democratic Republic of the Congo"],
    "Korea Republic": ["South Korea", "Corea del Sur"],
    "Iran": ["IR Iran"],
    "Saudi Arabia": ["Saudi", "Arabia Saudita"],
    "Czech Republic": ["Czechia"],
    "Türkiye": ["Turkey", "Turkiye"],
    "Ivory Coast": ["Côte d'Ivoire", "Cote d'Ivoire"],
    "Cape Verde": ["Cabo Verde"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina", "Bosnia"],
}


DATA_REQUIREMENTS = [
    {"category": "fixtures", "label": "Fixtures del dia", "source_name": "API-Football", "endpoint": "fixtures", "status": "ready", "priority": 1, "unlocks_score": 0.08},
    {"category": "results", "label": "Resultados finales", "source_name": "API-Football", "endpoint": "fixtures", "status": "ready", "priority": 1, "unlocks_score": 0.08},
    {"category": "advanced", "label": "Estadisticas por fixture", "source_name": "API-Football", "endpoint": "fixtures/statistics", "status": "ready", "priority": 1, "unlocks_score": 0.12},
    {"category": "players", "label": "Lesiones", "source_name": "API-Football", "endpoint": "injuries", "status": "ready", "priority": 1, "unlocks_score": 0.08},
    {"category": "players", "label": "Alineaciones", "source_name": "API-Football", "endpoint": "fixtures/lineups", "status": "planned", "priority": 1, "unlocks_score": 0.1},
    {"category": "players", "label": "Plantillas y jugadores", "source_name": "API-Football", "endpoint": "players", "status": "planned", "priority": 2, "unlocks_score": 0.08},
    {"category": "context", "label": "Eventos del partido", "source_name": "API-Football", "endpoint": "fixtures/events", "status": "planned", "priority": 2, "unlocks_score": 0.04},
    {"category": "context", "label": "Clasificacion/tabla", "source_name": "API-Football", "endpoint": "standings", "status": "planned", "priority": 2, "unlocks_score": 0.04},
    {"category": "environment", "label": "Sedes", "source_name": "API-Football/local", "endpoint": "venues", "status": "partial", "priority": 2, "unlocks_score": 0.04},
    {"category": "environment", "label": "Arbitro", "source_name": "API-Football", "endpoint": "fixtures", "status": "planned", "priority": 3, "unlocks_score": 0.04},
    {"category": "environment", "label": "Clima", "source_name": "Weather API", "endpoint": "weather", "status": "external_pending", "priority": 3, "unlocks_score": 0.04},
    {"category": "odds", "label": "Cuotas prepartido", "source_name": "API-Football/odds", "endpoint": "odds", "status": "external_pending", "priority": 3, "unlocks_score": 0.08},
    {"category": "odds", "label": "Closing line", "source_name": "Odds provider", "endpoint": "closing", "status": "external_pending", "priority": 3, "unlocks_score": 0.08},
]


def alias_rows() -> list[dict]:
    rows = []
    for canonical, aliases in TEAM_ALIASES.items():
        rows.append({"canonical_name": canonical, "alias": canonical, "source": "canonical", "confidence": 1.0})
        for alias in aliases:
            rows.append({"canonical_name": canonical, "alias": alias, "source": "manual_seed", "confidence": 0.95})
    return rows


def requirement_rows() -> list[dict]:
    return [dict(row) for row in DATA_REQUIREMENTS]
