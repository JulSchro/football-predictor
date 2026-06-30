import json
from pathlib import Path

from football_predictor.data.worldcup_2026 import load_many_worldcups, load_worldcup_2026_json, load_worldcup_json


def test_load_worldcup_2026_json_completed_only(tmp_path: Path) -> None:
    raw = tmp_path / "worldcup.json"
    raw.write_text(
        json.dumps(
            {
                "matches": [
                    {
                        "round": "Matchday 1",
                        "date": "2026-06-11",
                        "team1": "Mexico",
                        "team2": "South Africa",
                        "score": {"ft": [2, 1]},
                        "group": "Group A",
                    },
                    {
                        "round": "Matchday 2",
                        "date": "2026-06-12",
                        "team1": "Canada",
                        "team2": "Qatar",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    df = load_worldcup_2026_json(raw)

    assert len(df) == 1
    assert df.iloc[0]["home_team"] == "Mexico"
    assert df.iloc[0]["home_goals"] == 2
    assert df.iloc[0]["competition"] == "Group A"


def test_load_worldcup_2026_json_can_include_fixtures(tmp_path: Path) -> None:
    raw = tmp_path / "worldcup.json"
    raw.write_text(
        json.dumps({"matches": [{"round": "R1", "date": "2026-06-12", "team1": "A", "team2": "B"}]}),
        encoding="utf-8",
    )

    df = load_worldcup_2026_json(raw, include_fixtures=True)

    assert len(df) == 1
    assert df.iloc[0]["home_goals"] is None


def test_load_many_worldcups_combines_years(tmp_path: Path) -> None:
    raw_2022 = tmp_path / "worldcup_2022.json"
    raw_2026 = tmp_path / "worldcup_2026.json"
    payload = {"matches": [{"round": "R1", "date": "2022-11-20", "team1": "A", "team2": "B", "score": {"ft": [1, 0]}}]}
    raw_2022.write_text(json.dumps(payload), encoding="utf-8")
    raw_2026.write_text(json.dumps(payload), encoding="utf-8")

    df = load_many_worldcups({2022: raw_2022, 2026: raw_2026})

    assert len(df) == 2
    assert set(df["season"]) == {"2022", "2026"}
