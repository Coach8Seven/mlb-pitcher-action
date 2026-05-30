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
