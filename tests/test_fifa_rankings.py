from football_predictor.data.fifa_rankings import normalize_ranking_row


def test_normalize_fifa_ranking_row() -> None:
    row = {
        "TeamName": [{"Locale": "en-GB", "Description": "Argentina"}],
        "Rank": 1,
        "DecimalTotalPoints": 1877.27,
        "IdCountry": "ARG",
        "PrevRank": 3,
        "DecimalPrevPoints": 1867.25,
        "PubDate": "2026-06-19T00:00:00Z",
    }

    normalized = normalize_ranking_row(row)

    assert normalized["team"] == "Argentina"
    assert normalized["fifa_rank"] == 1
    assert normalized["fifa_points"] == 1877.27
    assert normalized["source"] == "fifa_official_rankings"

