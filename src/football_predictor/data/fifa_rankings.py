from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json


FIFA_RANKINGS_URL = "https://api.fifa.com/api/v3/rankings"


TEAM_NAME_ALIASES = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
}


def fetch_fifa_mens_rankings(count: int = 250) -> list[dict]:
    query = urlencode({"gender": 1, "count": count})
    request = Request(f"{FIFA_RANKINGS_URL}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [normalize_ranking_row(row) for row in payload.get("Results", [])]


def normalize_ranking_row(row: dict) -> dict:
    name = next((item["Description"] for item in row.get("TeamName", []) if item.get("Locale") == "en-GB"), None)
    name = name or row.get("TeamName", [{}])[0].get("Description")
    canonical = TEAM_NAME_ALIASES.get(name, name)
    return {
        "team": canonical,
        "source": "fifa_official_rankings",
        "fifa_rank": row.get("Rank"),
        "fifa_points": row.get("DecimalTotalPoints") or row.get("TotalPoints"),
        "raw": {
            "fifa_team_name": name,
            "country_code": row.get("IdCountry"),
            "previous_rank": row.get("PrevRank"),
            "previous_points": row.get("DecimalPrevPoints") or row.get("PrevPoints"),
            "published_at": row.get("PubDate"),
            "next_published_at": row.get("NextPubDate"),
            "ranking_movement": row.get("RankingMovement"),
        },
    }

