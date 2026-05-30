import os
from datetime import datetime

import requests
from flask import Flask, jsonify, request


MLB_BASE = "https://statsapi.mlb.com/api/v1"
MLB_LIVE = "https://statsapi.mlb.com/api/v1.1"

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


def _avg(values, digits=1):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), digits)


def _pct(part, whole):
    if not whole:
        return None
    return round(part / whole, 3)


def _is_whiff(event):
    call = (event.get("details") or {}).get("call") or {}
    code = str(call.get("code") or (event.get("details") or {}).get("code") or "")
    description = str(call.get("description") or (event.get("details") or {}).get("description") or "")
    return code in {"S", "W"} or "swinging strike" in description.lower()


def _is_called_strike(event):
    call = (event.get("details") or {}).get("call") or {}
    code = str(call.get("code") or (event.get("details") or {}).get("code") or "")
    description = str(call.get("description") or (event.get("details") or {}).get("description") or "")
    return code == "C" or description.lower() == "called strike"


def _pitch_mix_for_game(game_pk, person_id):
    if not game_pk:
        return None

    data = _get_json(f"{MLB_LIVE}/game/{game_pk}/feed/live")
    pitch_types = {}
    total_pitches = 0

    for play in data.get("liveData", {}).get("plays", {}).get("allPlays", []):
        pitcher_id = play.get("matchup", {}).get("pitcher", {}).get("id")
        if pitcher_id != person_id:
            continue

        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue

            details = event.get("details") or {}
            pitch_type = details.get("type") or {}
            code = pitch_type.get("code") or "UNK"
            description = pitch_type.get("description") or "Unknown"
            pitch_data = event.get("pitchData") or {}
            breaks = pitch_data.get("breaks") or {}
            speed = pitch_data.get("startSpeed")
            spin_rate = breaks.get("spinRate")

            total_pitches += 1
            bucket = pitch_types.setdefault(
                code,
                {
                    "code": code,
                    "description": description,
                    "count": 0,
                    "velocities": [],
                    "spinRates": [],
                    "whiffs": 0,
                    "calledStrikes": 0,
                },
            )
            bucket["count"] += 1
            if speed is not None:
                bucket["velocities"].append(float(speed))
            if spin_rate is not None:
                bucket["spinRates"].append(float(spin_rate))
            if _is_whiff(event):
                bucket["whiffs"] += 1
            if _is_called_strike(event):
                bucket["calledStrikes"] += 1

    if not total_pitches:
        return {
            "available": False,
            "source": "MLB Stats API live feed",
            "reason": "No pitch-level events found for pitcher in this game.",
        }

    pitches = []
    for bucket in pitch_types.values():
        pitches.append(
            {
                "code": bucket["code"],
                "description": bucket["description"],
                "count": bucket["count"],
                "usage": _pct(bucket["count"], total_pitches),
                "averageVelocity": _avg(bucket["velocities"]),
                "maxVelocity": round(max(bucket["velocities"]), 1) if bucket["velocities"] else None,
                "averageSpinRate": _avg(bucket["spinRates"], 0),
                "whiffRate": _pct(bucket["whiffs"], bucket["count"]),
                "calledStrikeWhiffRate": _pct(bucket["whiffs"] + bucket["calledStrikes"], bucket["count"]),
            }
        )

    return {
        "available": True,
        "source": "MLB Stats API live feed",
        "totalTrackedPitches": total_pitches,
        "pitches": sorted(pitches, key=lambda item: item["count"], reverse=True),
    }


def _pitch_mix_trend(starts):
    by_type = {}
    starts_with_mix = [start for start in starts if start.get("pitchMix", {}).get("available")]

    for index, start in enumerate(starts_with_mix):
        for pitch in start["pitchMix"].get("pitches", []):
            code = pitch["code"]
            bucket = by_type.setdefault(
                code,
                {
                    "code": code,
                    "description": pitch["description"],
                    "counts": [],
                    "usage": [],
                    "velocity": [],
                    "whiffRate": [],
                    "calledStrikeWhiffRate": [],
                    "latest": None,
                    "last3": [],
                },
            )
            bucket["counts"].append(pitch["count"])
            bucket["usage"].append(pitch["usage"])
            bucket["velocity"].append(pitch["averageVelocity"])
            bucket["whiffRate"].append(pitch["whiffRate"])
            bucket["calledStrikeWhiffRate"].append(pitch["calledStrikeWhiffRate"])
            if index == 0:
                bucket["latest"] = pitch
            if index < 3:
                bucket["last3"].append(pitch)

    pitches = []
    for bucket in by_type.values():
        latest = bucket["latest"] or {}
        last3 = bucket["last3"]
        sample_usage = _avg(bucket["usage"], 3)
        sample_velocity = _avg(bucket["velocity"])
        last3_usage = _avg([pitch.get("usage") for pitch in last3], 3)
        last3_velocity = _avg([pitch.get("averageVelocity") for pitch in last3])
        pitches.append(
            {
                "code": bucket["code"],
                "description": bucket["description"],
                "startsTracked": len(bucket["counts"]),
                "latestUsage": latest.get("usage"),
                "last3Usage": last3_usage,
                "sampleUsage": sample_usage,
                "usageChangeLatestVsSample": (
                    round(latest["usage"] - sample_usage, 3)
                    if latest.get("usage") is not None and sample_usage is not None
                    else None
                ),
                "latestAverageVelocity": latest.get("averageVelocity"),
                "last3AverageVelocity": last3_velocity,
                "sampleAverageVelocity": sample_velocity,
                "velocityChangeLatestVsSample": (
                    round(latest["averageVelocity"] - sample_velocity, 1)
                    if latest.get("averageVelocity") is not None and sample_velocity is not None
                    else None
                ),
                "sampleWhiffRate": _avg(bucket["whiffRate"], 3),
                "sampleCalledStrikeWhiffRate": _avg(bucket["calledStrikeWhiffRate"], 3),
            }
        )

    return {
        "available": bool(pitches),
        "source": "MLB Stats API live feed",
        "startsTracked": len(starts_with_mix),
        "pitches": sorted(
            pitches,
            key=lambda item: item.get("sampleUsage") or 0,
            reverse=True,
        ),
        "notes": [
            "Usage values are decimals, so 0.412 means 41.2% usage.",
            "Velocity change compares the latest start to the returned sample average.",
            "Whiff and CSW rates are pitch-level rates within the returned starts.",
        ],
    }


def _format_start(split, person_id, include_pitch_mix=False):
    stat = split.get("stat", {})
    game = split.get("game", {})
    game_pk = _game_pk(split)
    opponent = split.get("opponent", {}).get("name")
    strikeouts = _num(stat.get("strikeOuts"))
    pitch_count = stat.get("numberOfPitches")
    if pitch_count is None:
        pitch_count = _pitch_count_from_boxscore(game_pk, person_id)

    start = {
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
    if include_pitch_mix:
        try:
            start["pitchMix"] = _pitch_mix_for_game(game_pk, person_id)
        except requests.RequestException:
            start["pitchMix"] = {
                "available": False,
                "source": "MLB Stats API live feed",
                "reason": "Pitch-level feed unavailable for this game.",
            }
    return start


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
    include_pitch_mix = str(request.args.get("include_pitch_mix", "true")).lower() not in {
        "0",
        "false",
        "no",
    }

    if not pitcher_name:
        return jsonify({"error": "pitcher_name is required"}), 400

    pitcher = _find_pitcher(pitcher_name)
    if not pitcher:
        return jsonify({"error": f"No pitcher found for '{pitcher_name}'"}), 404

    splits = _pitcher_game_log(pitcher["id"], season)
    start_splits = sorted(
        [split for split in splits if _is_start(split)],
        key=lambda item: item.get("date") or "",
        reverse=True,
    )[:limit]
    starts = [
        _format_start(split, pitcher["id"], include_pitch_mix=include_pitch_mix)
        for split in start_splits
    ]

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
            "pitchMixTrend": _pitch_mix_trend(starts) if include_pitch_mix else None,
            "notes": [
                "Pitch counts come from MLB Stats API boxscore when not present in the pitcher game log.",
                "Hit rate uses strikeouts greater than the supplied line, matching an Over prop.",
                "Pitch mix and velocity come from MLB Stats API live-feed pitch-level data when include_pitch_mix=true.",
            ],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
