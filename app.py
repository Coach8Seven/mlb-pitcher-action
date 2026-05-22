import os
from datetime import datetime

import requests
from flask import Flask, jsonify, request


MLB_BASE = "https://statsapi.mlb.com/api/v1"

app = Flask(__name__)


def _get_json(url, params=None):
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _num(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _find_pitcher(name):
    data = _get_json(
        f"{MLB_BASE}/people/search",
        {"names": name, "sportId": 1, "activeStatus": "Y"},
    )
    people = data.get("people", [])
    if not people:
        data = _get_json(f"{MLB_BASE}/people/search", {"names": name, "sportId": 1})
        people = data.get("people", [])
    if not people:
        return None

    pitchers = [
        person
        for person in people
        if person.get("primaryPosition", {}).get("abbreviation") == "P"
    ]
    return (pitchers or people)[0]


def _pitcher_game_log(person_id, season):
    hydrate = (
        "stats(group=[pitching],type=[gameLog],"
        f"season={season},sportId=1,gameType=[R])"
    )
    data = _get_json(
        f"{MLB_BASE}/people/{person_id}",
        {"hydrate": hydrate, "season": season, "sportId": 1},
    )
    stats = data.get("people", [{}])[0].get("stats", [])
    for block in stats:
        group = block.get("group", {}).get("displayName", "").lower()
        stat_type = block.get("type", {}).get("displayName", "").lower()
        if group == "pitching" and "game" in stat_type:
            return block.get("splits", [])
    return []


def _is_start(split):
    stat = split.get("stat", {})
    games_started = _num(stat.get("gamesStarted"))
    if games_started:
        return True
    # Some game logs omit gamesStarted. Treat larger workloads as starts,
    # but avoid one-inning relief appearances.
    innings = str(stat.get("inningsPitched", "0"))
    whole = _num(innings.split(".")[0])
    return whole >= 3


def _game_pk(split):
    game = split.get("game") or {}
    if game.get("gamePk"):
        return game["gamePk"]
    link = game.get("link", "")
    parts = [part for part in link.split("/") if part.isdigit()]
    return _num(parts[-1]) if parts else None


def _pitch_count_from_boxscore(game_pk, person_id):
    if not game_pk:
        return None
    data = _get_json(f"{MLB_BASE}/game/{game_pk}/boxscore")
    key = f"ID{person_id}"
    for side in ("home", "away"):
        player = data.get("teams", {}).get(side, {}).get("players", {}).get(key)
        if not player:
            continue
        pitching = player.get("stats", {}).get("pitching", {})
        for field in ("numberOfPitches", "pitchesThrown"):
            if pitching.get(field) is not None:
                return _num(pitching.get(field), None)
    return None


def _format_start(split, person_id):
    stat = split.get("stat", {})
    game = split.get("game", {})
    game_pk = _game_pk(split)
    opponent = split.get("opponent", {}).get("name")
    strikeouts = _num(stat.get("strikeOuts"))
    pitch_count = stat.get("numberOfPitches")
    if pitch_count is None:
        pitch_count = _pitch_count_from_boxscore(game_pk, person_id)

    return {
        "date": split.get("date"),
        "gamePk": game_pk,
        "opponent": opponent,
        "game": game.get("description"),
        "inningsPitched": stat.get("inningsPitched"),
        "pitches": _num(pitch_count, None) if pitch_count is not None else None,
        "strikeouts": strikeouts,
        "earnedRuns": _num(stat.get("earnedRuns"), None),
        "walks": _num(stat.get("baseOnBalls"), None),
        "hits": _num(stat.get("hits"), None),
        "source": "MLB Stats API gameLog + boxscore",
    }


def _summary(starts, line):
    def hit_rate(items):
        if line is None or not items:
            return None
        hits = sum(1 for item in items if item["strikeouts"] > line)
        return {
            "hits": hits,
            "sample": len(items),
            "rate": round(hits / len(items), 3),
            "line": line,
        }

    def average(items, key):
        vals = [item[key] for item in items if item.get(key) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 1)

    last5 = starts[:5]
    last10 = starts[:10]
    return {
        "last5": {
            "averageStrikeouts": average(last5, "strikeouts"),
            "averagePitches": average(last5, "pitches"),
            "hitRateVsLine": hit_rate(last5),
        },
        "last10": {
            "averageStrikeouts": average(last10, "strikeouts"),
            "averagePitches": average(last10, "pitches"),
            "hitRateVsLine": hit_rate(last10),
        },
    }


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "mlb-pitcher-action"})


@app.get("/pitcher-last-starts")
def pitcher_last_starts():
    pitcher_name = request.args.get("pitcher_name", "").strip()
    season = request.args.get("season") or str(datetime.now().year)
    limit = min(max(_num(request.args.get("limit"), 10), 1), 15)
    line_raw = request.args.get("line")
    line = float(line_raw) if line_raw not in (None, "") else None

    if not pitcher_name:
        return jsonify({"error": "pitcher_name is required"}), 400

    pitcher = _find_pitcher(pitcher_name)
    if not pitcher:
        return jsonify({"error": f"No pitcher found for '{pitcher_name}'"}), 404

    splits = _pitcher_game_log(pitcher["id"], season)
    starts = [_format_start(split, pitcher["id"]) for split in splits if _is_start(split)]
    starts = sorted(starts, key=lambda item: item.get("date") or "", reverse=True)[:limit]

    return jsonify(
        {
            "pitcher": {
                "id": pitcher["id"],
                "name": pitcher.get("fullName"),
                "throws": pitcher.get("pitchHand", {}).get("code"),
                "team": pitcher.get("currentTeam", {}).get("name"),
            },
            "season": season,
            "requestedLimit": limit,
            "startsReturned": len(starts),
            "starts": starts,
            "summary": _summary(starts, line),
            "notes": [
                "Pitch counts come from MLB Stats API boxscore when not present in the pitcher game log.",
                "Hit rate uses strikeouts greater than the supplied line, matching an Over prop.",
            ],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

