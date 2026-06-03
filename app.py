import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import ceil, floor, sqrt

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


def _float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _avg(values, digits=1):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), digits)


def _stddev(values, digits=2):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return round(sqrt(variance), digits)


def _pct(part, whole):
    if not whole:
        return None
    return round(part / whole, 3)


def _innings_to_outs(value):
    try:
        whole, _, remainder = str(value or "0").partition(".")
        return int(whole) * 3 + int(remainder or 0)
    except (TypeError, ValueError):
        return 0


def _outs_to_innings(outs):
    return round(outs / 3, 1) if outs is not None else None


def _first_stat(data):
    blocks = data.get("stats", [])
    if not blocks:
        return {}
    splits = blocks[0].get("splits", [])
    if not splits:
        return {}
    return splits[0].get("stat", {})


def _pitcher_person(person_id):
    people = _get_json(f"{MLB_BASE}/people/{person_id}").get("people", [])
    return people[0] if people else {}


def _pitcher_season_stat(person_id, season):
    data = _get_json(
        f"{MLB_BASE}/people/{person_id}/stats",
        {"stats": "season", "group": "pitching", "season": season, "sportIds": 1},
    )
    return _first_stat(data)


def _team_hitting_split(team_id, pitcher_hand, season):
    split_code = "vl" if pitcher_hand == "L" else "vr"
    data = _get_json(
        f"{MLB_BASE}/teams/{team_id}/stats",
        {
            "stats": "statSplits",
            "group": "hitting",
            "season": season,
            "sitCodes": split_code,
        },
    )
    stat = _first_stat(data)
    plate_appearances = _num(stat.get("plateAppearances"), None)
    strikeouts = _num(stat.get("strikeOuts"), None)
    return {
        "split": "vs Left" if pitcher_hand == "L" else "vs Right",
        "plateAppearances": plate_appearances,
        "strikeouts": strikeouts,
        "strikeoutRate": _pct(strikeouts, plate_appearances),
        "avg": _float(stat.get("avg")),
        "obp": _float(stat.get("obp")),
        "ops": _float(stat.get("ops")),
    }


def _recent_start_workload(person_id, season, limit=5):
    splits = sorted(
        [split for split in _pitcher_game_log(person_id, season) if _is_start(split)],
        key=lambda item: item.get("date") or "",
        reverse=True,
    )[:limit]
    starts = []
    for split in splits:
        stat = split.get("stat", {})
        starts.append(
            {
                "date": split.get("date"),
                "opponent": split.get("opponent", {}).get("name"),
                "inningsPitched": stat.get("inningsPitched"),
                "pitches": _num(stat.get("numberOfPitches"), None),
                "battersFaced": _num(stat.get("battersFaced"), None),
                "strikeouts": _num(stat.get("strikeOuts"), None),
            }
        )
    return _workload_summary(starts)


def _normalize_score(value, low, high):
    if value is None or high <= low:
        return 0
    return min(max((value - low) / (high - low), 0), 1)


def _clamp(value, low, high):
    return min(max(value, low), high)


def _screening_score(pitcher_k_rate, opponent_k_rate, recent):
    recent_bf = recent.get("averageBattersFaced")
    recent_pitches = recent.get("averagePitches")
    recent_ks = recent.get("averageStrikeouts")
    score = (
        0.30 * _normalize_score(pitcher_k_rate, 0.15, 0.33)
        + 0.25 * _normalize_score(opponent_k_rate, 0.18, 0.30)
        + 0.15 * _normalize_score(recent_bf, 18, 27)
        + 0.15 * _normalize_score(recent_pitches, 70, 100)
        + 0.15 * _normalize_score(recent_ks, 3, 8)
    )
    return round(score * 100, 1)


def _workload_summary(starts):
    return {
        "startsTracked": len(starts),
        "averageOuts": _avg(
            [_innings_to_outs(start["inningsPitched"]) for start in starts],
            1,
        ),
        "averageInningsDecimal": _avg(
            [_innings_to_outs(start["inningsPitched"]) / 3 for start in starts],
            1,
        ),
        "averagePitches": _avg([start["pitches"] for start in starts]),
        "averageBattersFaced": _avg([start["battersFaced"] for start in starts]),
        "averageStrikeouts": _avg([start["strikeouts"] for start in starts]),
        "strikeoutRate": _pct(
            sum(_num(start.get("strikeouts"), 0) for start in starts),
            sum(_num(start.get("battersFaced"), 0) for start in starts),
        ),
        "battersFacedStdDev": _stddev([start["battersFaced"] for start in starts]),
        "strikeoutsStdDev": _stddev([start["strikeouts"] for start in starts]),
        "starts": starts,
    }


def _weighted_available(items, digits=1):
    available = [(value, weight) for value, weight in items if value is not None]
    weight_total = sum(weight for _, weight in available)
    if not available or not weight_total:
        return None
    return round(sum(value * weight for value, weight in available) / weight_total, digits)


def _expected_batters_faced(season, recent_last5, recent_last10):
    season_starts = season.get("starts")
    season_bf = season.get("battersFaced")
    season_bf_per_start = (
        round(season_bf / season_starts, 1)
        if season_bf is not None and season_starts
        else None
    )
    last5_bf = recent_last5.get("averageBattersFaced")
    last10_bf = recent_last10.get("averageBattersFaced")
    base = _weighted_available(
        [
            (last5_bf, 0.50),
            (last10_bf, 0.30),
            (season_bf_per_start, 0.20),
        ],
        1,
    )

    if base is None:
        return {
            "expectedBattersFaced": None,
            "baseExpectedBattersFaced": None,
            "seasonBattersFacedPerStart": season_bf_per_start,
            "adjustments": [],
            "notes": ["Expected BF unavailable because workload data is missing."],
        }

    adjustments = []
    expected = base
    last5_pitches = recent_last5.get("averagePitches")
    last10_pitches = recent_last10.get("averagePitches")
    last10_bf = recent_last10.get("averageBattersFaced")
    pitches_per_bf = (
        round(last10_pitches / last10_bf, 2)
        if last10_pitches is not None and last10_bf
        else None
    )

    if last5_pitches is not None:
        if last5_pitches >= 94:
            adjustments.append({"reason": "stable high recent pitch count", "value": 0.3})
        elif last5_pitches < 75:
            adjustments.append({"reason": "light recent pitch count", "value": -1.0})
        elif last5_pitches < 84:
            adjustments.append({"reason": "below-average recent pitch count", "value": -0.5})

    if pitches_per_bf is not None:
        if pitches_per_bf >= 4.45:
            adjustments.append({"reason": "inefficient pitches per batter faced", "value": -0.8})
        elif pitches_per_bf >= 4.25:
            adjustments.append({"reason": "mild efficiency drag", "value": -0.4})
        elif pitches_per_bf <= 3.75:
            adjustments.append({"reason": "efficient recent BF conversion", "value": 0.3})

    bb_rate = season.get("bbRate")
    if bb_rate is not None:
        if bb_rate >= 0.105:
            adjustments.append({"reason": "high walk-rate workload risk", "value": -0.8})
        elif bb_rate >= 0.09:
            adjustments.append({"reason": "walk-rate efficiency concern", "value": -0.4})
        elif bb_rate <= 0.055:
            adjustments.append({"reason": "low walk-rate efficiency support", "value": 0.2})

    short_recent_starts = sum(
        1
        for start in recent_last5.get("starts", [])
        if start.get("battersFaced") is not None and start["battersFaced"] < 20
    )
    if short_recent_starts >= 2:
        adjustments.append({"reason": "multiple recent short starts", "value": -0.7})

    for adjustment in adjustments:
        expected += adjustment["value"]

    expected = round(_clamp(expected, 12, 30), 1)
    return {
        "expectedBattersFaced": expected,
        "baseExpectedBattersFaced": base,
        "seasonBattersFacedPerStart": season_bf_per_start,
        "pitchesPerBatterFacedLast10": pitches_per_bf,
        "adjustments": adjustments,
        "notes": [
            "Base Expected BF uses 50% last 5 BF, 30% last 10 BF, and 20% season BF/start.",
            "Workload can improve ranking only through Expected Ks; low-K pitchers are not promoted for volume alone.",
        ],
    }


def _estimated_k_probability(season, opponent_split, recent_last10):
    base_k_rate = season.get("kRate")
    if base_k_rate is None:
        return {
            "estimatedKRate": None,
            "baseKRate": None,
            "components": {},
            "notes": ["Estimated K% unavailable because season K rate is missing."],
        }

    opponent_k_rate = opponent_split.get("strikeoutRate")
    matchup_adjustment = (
        round(_clamp((opponent_k_rate - 0.22) * 0.6, -0.03, 0.03), 3)
        if opponent_k_rate is not None
        else 0
    )
    recent_k_rate = recent_last10.get("strikeoutRate")
    recent_form_adjustment = (
        round(_clamp((recent_k_rate - base_k_rate) * 0.35, -0.02, 0.02), 3)
        if recent_k_rate is not None
        else 0
    )
    # Pitch-level whiff/CSW and velocity are reviewed in Stage 2 to avoid
    # making the screenshot screen too slow and noisy.
    pitch_quality_adjustment = 0

    estimated = round(
        _clamp(
            base_k_rate
            + matchup_adjustment
            + recent_form_adjustment
            + pitch_quality_adjustment,
            0.08,
            0.42,
        ),
        3,
    )
    return {
        "estimatedKRate": estimated,
        "baseKRate": base_k_rate,
        "components": {
            "opponentMatchupAdjustment": matchup_adjustment,
            "recentFormAdjustment": recent_form_adjustment,
            "pitchQualityAdjustment": pitch_quality_adjustment,
            "redFlagPenalty": 0,
        },
        "recentKRateLast10Starts": recent_k_rate,
        "notes": [
            "Estimated K% starts with pitcher season K%; matchup and recent form only adjust it.",
            "Opponent adjustment is capped from -3% to +3%. Recent-form adjustment is capped from -2% to +2%.",
            "Pitch-quality adjustment is deferred to Stage 2 pitch mix/velocity review.",
        ],
    }


def _expected_strikeout_range(expected_ks, confidence_label, recent_last10):
    if expected_ks is None:
        return None
    width = {
        "Low": 2.2,
        "Medium": 1.9,
        "Medium+": 1.7,
        "High": 1.5,
        "High+": 1.4,
    }.get(confidence_label, 1.9)
    volatility = recent_last10.get("strikeoutsStdDev")
    if volatility is not None and volatility >= 2.4:
        width += 0.5
    elif volatility is not None and volatility <= 1.2:
        width -= 0.2
    low = max(0, floor(expected_ks - width))
    high = max(low, ceil(expected_ks + width))
    return {"low": low, "high": high, "display": f"{low}-{high}"}


def _confidence_label(season, opponent_split, recent_last5, recent_last10, expected_bf, estimated_k_rate):
    points = 0.0
    k_rate = season.get("kRate")
    opponent_k_rate = opponent_split.get("strikeoutRate")
    starts_tracked = recent_last10.get("startsTracked") or 0
    last5_pitches = recent_last5.get("averagePitches")
    bf_stddev = recent_last10.get("battersFacedStdDev")
    bb_rate = season.get("bbRate")

    if k_rate is not None:
        if k_rate >= 0.28:
            points += 2.0
        elif k_rate >= 0.25:
            points += 1.4
        elif k_rate >= 0.22:
            points += 0.7
        elif k_rate < 0.18:
            points -= 1.0

    if opponent_k_rate is not None:
        if opponent_k_rate >= 0.25:
            points += 1.0
        elif opponent_k_rate >= 0.23:
            points += 0.5
        elif opponent_k_rate < 0.20:
            points -= 0.8

    if expected_bf is not None:
        if expected_bf >= 24:
            points += 1.0
        elif expected_bf >= 22:
            points += 0.5
        elif expected_bf < 20:
            points -= 1.0

    if starts_tracked >= 8:
        points += 0.5
    elif starts_tracked < 5:
        points -= 0.8

    if last5_pitches is not None and last5_pitches < 80:
        points -= 0.8
    if bf_stddev is not None and bf_stddev >= 4:
        points -= 0.5
    if bb_rate is not None and bb_rate >= 0.105:
        points -= 0.7
    if estimated_k_rate is not None and estimated_k_rate >= 0.29:
        points += 0.5

    if points >= 3.2:
        label = "High"
    elif points >= 2.1:
        label = "Medium+"
    elif points >= 0.8:
        label = "Medium"
    else:
        label = "Low"

    return {
        "label": label,
        "score": round(points, 1),
        "notes": [
            "High+ is applied only after ranking when a High profile separates from the board.",
            "Confidence is research confidence, not bet confidence.",
        ],
    }


def _pitcher_screening_profile(person_id, opponent_team_id, season):
    person = _pitcher_person(person_id)
    season_stat = _pitcher_season_stat(person_id, season)
    hand = person.get("pitchHand", {}).get("code")
    opponent_split = _team_hitting_split(opponent_team_id, hand, season)
    batters_faced = _num(season_stat.get("battersFaced"), None)
    strikeouts = _num(season_stat.get("strikeOuts"), None)
    walks = _num(season_stat.get("baseOnBalls"), None)
    k_rate = _pct(strikeouts, batters_faced)
    bb_rate = _pct(walks, batters_faced)
    season_payload = {
        "starts": _num(season_stat.get("gamesStarted"), None),
        "inningsPitched": season_stat.get("inningsPitched"),
        "pitches": _num(season_stat.get("numberOfPitches"), None),
        "battersFaced": batters_faced,
        "strikeouts": strikeouts,
        "kRate": k_rate,
        "bbRate": bb_rate,
        "strikeoutsPer9": _float(season_stat.get("strikeoutsPer9Inn")),
        "pitchesPerInning": _float(season_stat.get("pitchesPerInning")),
    }
    recent_last10 = _recent_start_workload(person_id, season, limit=10)
    recent_last5 = _workload_summary(recent_last10.get("starts", [])[:5])
    expected_bf = _expected_batters_faced(
        season_payload,
        recent_last5,
        recent_last10,
    )
    k_probability = _estimated_k_probability(
        season_payload,
        opponent_split,
        recent_last10,
    )
    expected_batters_faced = expected_bf.get("expectedBattersFaced")
    estimated_k_rate = k_probability.get("estimatedKRate")
    expected_ks = (
        round(expected_batters_faced * estimated_k_rate, 1)
        if expected_batters_faced is not None and estimated_k_rate is not None
        else None
    )
    confidence = _confidence_label(
        season_payload,
        opponent_split,
        recent_last5,
        recent_last10,
        expected_batters_faced,
        estimated_k_rate,
    )
    expected_range = _expected_strikeout_range(
        expected_ks,
        confidence.get("label"),
        recent_last10,
    )

    return {
        "id": person_id,
        "name": person.get("fullName"),
        "throws": hand,
        "season": season_payload,
        "recentLast5": recent_last5,
        "recentLast10": recent_last10,
        "opponentHittingSplit": opponent_split,
        "researchPriorityScore": _screening_score(
            k_rate,
            opponent_split.get("strikeoutRate"),
            recent_last5,
        ),
        "expectedKModel": {
            "estimatedKRate": estimated_k_rate,
            "expectedBattersFaced": expected_batters_faced,
            "expectedStrikeouts": expected_ks,
            "expectedStrikeoutRange": expected_range,
            "estimatedKProbability": k_probability,
            "expectedBattersFacedModel": expected_bf,
            "confidence": confidence,
            "lineupStatus": "Unavailable from helper",
            "lineupConfidenceRule": "Projected lineups can support research, but projected or unavailable lineups should reduce final confidence.",
        },
        "scoreNotes": [
            "Research-priority score is a screening aid, not a bet recommendation.",
            "Score weights pitcher K rate, opponent K rate vs handedness, and recent workload.",
            "Expected Ks is the preferred Stage 1 ranking field when available.",
            "Lineups, news, weather, prop line, and price still require later checks.",
        ],
    }


def _schedule(date):
    data = _get_json(
        f"{MLB_BASE}/schedule",
        {"sportId": 1, "date": date, "hydrate": "probablePitcher,team,venue"},
    )
    dates = data.get("dates", [])
    return dates[0].get("games", []) if dates else []


def _normalize_team_code(value):
    aliases = {"CHW": "CWS", "WAS": "WSH", "AZ": "ARI"}
    code = str(value or "").strip().upper()
    return aliases.get(code, code)


def _requested_matchups(raw):
    if not raw:
        return []
    matchups = []
    for item in str(raw).split(","):
        away, separator, home = item.partition("@")
        if separator and away.strip() and home.strip():
            matchups.append(
                (_normalize_team_code(away), _normalize_team_code(home))
            )
    return matchups


def _format_screening_game(game, season):
    away = game.get("teams", {}).get("away", {})
    home = game.get("teams", {}).get("home", {})
    away_code = _normalize_team_code(away.get("team", {}).get("abbreviation"))
    home_code = _normalize_team_code(home.get("team", {}).get("abbreviation"))
    sides = [("away", away, home), ("home", home, away)]
    pitchers = []

    for side, team_side, opponent_side in sides:
        team_code = _normalize_team_code(team_side.get("team", {}).get("abbreviation"))
        opponent_code = _normalize_team_code(opponent_side.get("team", {}).get("abbreviation"))
        probable = team_side.get("probablePitcher")
        if not probable:
            pitchers.append(
                {
                    "side": side,
                    "team": team_code,
                    "opponent": opponent_code,
                    "available": False,
                    "reason": "Probable starter is not listed yet.",
                }
            )
            continue

        try:
            profile = _pitcher_screening_profile(
                probable["id"],
                opponent_side.get("team", {}).get("id"),
                season,
            )
            profile.update(
                {
                    "side": side,
                    "team": team_code,
                    "opponent": opponent_code,
                    "available": True,
                }
            )
            pitchers.append(profile)
        except requests.RequestException:
            pitchers.append(
                {
                    "side": side,
                    "team": team_code,
                    "opponent": opponent_code,
                    "id": probable.get("id"),
                    "name": probable.get("fullName"),
                    "available": False,
                    "reason": "MLB Stats API screening data unavailable.",
                }
            )

    available_scores = [
        pitcher["researchPriorityScore"]
        for pitcher in pitchers
        if pitcher.get("researchPriorityScore") is not None
    ]
    return {
        "gamePk": game.get("gamePk"),
        "gameDate": game.get("gameDate"),
        "status": game.get("status", {}).get("detailedState"),
        "matchup": f"{away_code}@{home_code}",
        "awayTeam": away.get("team", {}).get("name"),
        "homeTeam": home.get("team", {}).get("name"),
        "venue": game.get("venue", {}).get("name"),
        "pitchers": pitchers,
        "gameResearchPriorityScore": max(available_scores) if available_scores else None,
    }


def _gap_label(gap):
    if gap is None:
        return None
    if gap < 0.30:
        return "basically tied"
    if gap < 0.60:
        return "small edge"
    if gap < 1.00:
        return "meaningful edge"
    return "major edge"


def _tie_breaker_score(pitcher):
    season = pitcher.get("season", {})
    opponent = pitcher.get("opponentHittingSplit", {})
    recent = pitcher.get("recentLast5", {})
    model = pitcher.get("expectedKModel", {})
    confidence = model.get("confidence", {})
    confidence_score = confidence.get("score") or 0
    score = (
        0.35 * _normalize_score(season.get("kRate"), 0.15, 0.33)
        + 0.25 * _normalize_score(opponent.get("strikeoutRate"), 0.18, 0.30)
        + 0.20 * _normalize_score(recent.get("averagePitches"), 70, 100)
        + 0.20 * _normalize_score(confidence_score, 0, 4)
    )
    return round(score * 100, 1)


def _sort_by_expected_ks_with_ties(pitchers):
    base = sorted(
        pitchers,
        key=lambda pitcher: (
            pitcher.get("expectedStrikeouts")
            if pitcher.get("expectedStrikeouts") is not None
            else -1,
            pitcher.get("tieBreakerScore") or 0,
        ),
        reverse=True,
    )
    clusters = []
    current = []
    anchor = None
    for pitcher in base:
        expected_ks = pitcher.get("expectedStrikeouts")
        if expected_ks is None:
            if current:
                clusters.append(current)
                current = []
                anchor = None
            clusters.append([pitcher])
            continue
        if anchor is None:
            anchor = expected_ks
            current = [pitcher]
            continue
        if anchor - expected_ks < 0.30:
            current.append(pitcher)
        else:
            clusters.append(current)
            anchor = expected_ks
            current = [pitcher]
    if current:
        clusters.append(current)

    ranked = []
    for cluster in clusters:
        if len(cluster) > 1:
            cluster = sorted(
                cluster,
                key=lambda pitcher: pitcher.get("tieBreakerScore") or 0,
                reverse=True,
            )
            for pitcher in cluster:
                pitcher["rankingCluster"] = "Expected-K gap under 0.30; tie-breakers ordered this cluster."
        ranked.extend(cluster)
    return ranked


def _stage1_group(pitcher):
    expected_ks = pitcher.get("expectedStrikeouts")
    estimated_k_rate = pitcher.get("estimatedKRate")
    expected_bf = pitcher.get("expectedBattersFaced")
    confidence = pitcher.get("confidence")

    if expected_ks is None:
        return "PASS FOR NOW"
    if expected_ks >= 5.4 and confidence != "Low":
        return "RESEARCH"
    if (
        expected_ks >= 5.0
        and estimated_k_rate is not None
        and estimated_k_rate >= 0.24
        and expected_bf is not None
        and expected_bf >= 22
        and confidence in {"Medium+", "High", "High+"}
    ):
        return "RESEARCH"
    if expected_ks >= 4.4 or confidence in {"Medium+", "High", "High+"}:
        return "BORDERLINE"
    return "PASS FOR NOW"


def _group_reason(pitcher):
    expected_ks = pitcher.get("expectedStrikeouts")
    estimated_k_rate = pitcher.get("estimatedKRate")
    expected_bf = pitcher.get("expectedBattersFaced")
    opponent_k_rate = pitcher.get("opponentKRateVsHand")
    confidence = pitcher.get("confidence")
    if expected_ks is None:
        return "Missing expected-K estimate."
    if pitcher.get("stage1Group") == "RESEARCH":
        return "Expected Ks, K skill, matchup, and workload are strong enough for pitcher-K screenshots."
    if pitcher.get("stage1Group") == "BORDERLINE":
        concerns = []
        if expected_ks < 5.0:
            concerns.append("expected Ks below primary research group")
        if estimated_k_rate is not None and estimated_k_rate < 0.24:
            concerns.append("estimated K% is not strong")
        if expected_bf is not None and expected_bf < 22:
            concerns.append("expected BF is not strong")
        if opponent_k_rate is not None and opponent_k_rate < 0.22:
            concerns.append("opponent K split is modest")
        return "; ".join(concerns[:2]) or "Interesting but not clean enough before line/price."
    reasons = []
    if estimated_k_rate is not None and estimated_k_rate < 0.21:
        reasons.append("low estimated K%")
    if expected_bf is not None and expected_bf < 21:
        reasons.append("limited expected BF")
    if expected_ks < 4.4:
        reasons.append("expected Ks below screen threshold")
    if confidence == "Low":
        reasons.append("low research confidence")
    return "; ".join(reasons[:2]) or "Does not clear the Stage 1 research threshold."


def _rank_gap_notes(ranked):
    notes = [
        "Expected-K gap guide: under 0.30 = basically tied; 0.30-0.59 = small edge; 0.60-0.99 = meaningful edge; 1.00+ = major edge.",
    ]
    expected_values = [
        pitcher.get("expectedStrikeouts")
        for pitcher in ranked
        if pitcher.get("expectedStrikeouts") is not None
    ]
    if len(expected_values) < 2:
        return notes
    top_gap = round(abs(expected_values[0] - expected_values[1]), 1)
    notes.append(
        f"Ranks 1-2 are separated by {top_gap} expected Ks: {_gap_label(top_gap)}."
    )
    tight_clusters = []
    cluster_start = 0
    cluster_values = []
    for index, expected in enumerate(expected_values):
        proposed = cluster_values + [expected]
        if proposed and max(proposed) - min(proposed) < 0.30:
            cluster_values = proposed
            continue
        if len(cluster_values) > 1:
            tight_clusters.append((cluster_start + 1, index))
        cluster_start = index
        cluster_values = [expected]
    if len(cluster_values) > 1:
        tight_clusters.append((cluster_start + 1, len(expected_values)))
    if tight_clusters:
        notes.append(
            "Tight ranking clusters: "
            + ", ".join(f"Ranks {start}-{end}" for start, end in tight_clusters)
            + " are within 0.30 expected Ks, so tie-breakers matter."
        )
    return notes


def _rank_screening_pitchers(games, limit=10, minimum_score=45):
    pitchers = []
    for game in games:
        for pitcher in game.get("pitchers", []):
            score = pitcher.get("researchPriorityScore")
            if not pitcher.get("available") or score is None:
                continue

            recent = pitcher.get("recentLast5", {})
            recent10 = pitcher.get("recentLast10", {})
            opponent = pitcher.get("opponentHittingSplit", {})
            season = pitcher.get("season", {})
            model = pitcher.get("expectedKModel", {})
            estimated = model.get("estimatedKProbability", {})
            expected_bf_model = model.get("expectedBattersFacedModel", {})
            confidence = model.get("confidence", {})
            expected_range = model.get("expectedStrikeoutRange") or {}
            flat_pitcher = {
                "pitcherId": pitcher.get("id"),
                "pitcher": pitcher.get("name"),
                "team": pitcher.get("team"),
                "opponent": pitcher.get("opponent"),
                "matchup": game.get("matchup"),
                "gameDate": game.get("gameDate"),
                "venue": game.get("venue"),
                "throws": pitcher.get("throws"),
                "researchPriorityScore": score,
                "seasonKRate": season.get("kRate"),
                "seasonStrikeoutsPer9": season.get("strikeoutsPer9"),
                "opponentSplit": opponent.get("split"),
                "opponentKRateVsHand": opponent.get("strikeoutRate"),
                "baseKRate": estimated.get("baseKRate"),
                "opponentMatchupAdjustment": estimated.get("components", {}).get("opponentMatchupAdjustment"),
                "recentFormAdjustment": estimated.get("components", {}).get("recentFormAdjustment"),
                "pitchQualityAdjustment": estimated.get("components", {}).get("pitchQualityAdjustment"),
                "estimatedKRate": model.get("estimatedKRate"),
                "expectedBattersFaced": model.get("expectedBattersFaced"),
                "expectedStrikeouts": model.get("expectedStrikeouts"),
                "expectedStrikeoutRange": expected_range.get("display"),
                "confidence": confidence.get("label"),
                "confidenceScore": confidence.get("score"),
                "lineupStatus": model.get("lineupStatus"),
                "lineupConfidenceRule": model.get("lineupConfidenceRule"),
                "expectedBFBase": expected_bf_model.get("baseExpectedBattersFaced"),
                "expectedBFAdjustments": expected_bf_model.get("adjustments"),
                "seasonBattersFacedPerStart": expected_bf_model.get("seasonBattersFacedPerStart"),
                "pitchesPerBatterFacedLast10": expected_bf_model.get("pitchesPerBatterFacedLast10"),
                "recentLast5AverageKs": recent.get("averageStrikeouts"),
                "recentLast5AveragePitches": recent.get("averagePitches"),
                "recentLast5AverageBattersFaced": recent.get("averageBattersFaced"),
                "recentLast5AverageInningsDecimal": recent.get("averageInningsDecimal"),
                "recentLast10AverageKs": recent10.get("averageStrikeouts"),
                "recentLast10AveragePitches": recent10.get("averagePitches"),
                "recentLast10AverageBattersFaced": recent10.get("averageBattersFaced"),
                "recentLast10AverageInningsDecimal": recent10.get("averageInningsDecimal"),
                "recommendedForDeeperResearch": False,
                "screeningNotes": [
                    "Use Expected Ks as the primary Stage 1 ranking field when available.",
                    "Use measurable screening data only. Reputation or name value is not a reason to research a pitcher.",
                    "This is a research shortlist, not a bet recommendation.",
                    "Bet365 pitcher-K line, price, lineup, news, and weather still require later checks.",
                ],
            }
            flat_pitcher["tieBreakerScore"] = _tie_breaker_score(pitcher)
            pitchers.append(
                flat_pitcher
            )

    ranked = _sort_by_expected_ks_with_ties(pitchers)
    if ranked:
        top_expected = ranked[0].get("expectedStrikeouts")
        second_expected = ranked[1].get("expectedStrikeouts") if len(ranked) > 1 else None
        top_gap = (
            round(top_expected - second_expected, 1)
            if top_expected is not None and second_expected is not None
            else None
        )
        if (
            ranked[0].get("confidence") == "High"
            and ranked[0].get("expectedStrikeouts") is not None
            and ranked[0]["expectedStrikeouts"] >= 6.0
            and (top_gap is None or top_gap >= 0.60 or ranked[0]["expectedStrikeouts"] >= 6.7)
        ):
            ranked[0]["confidence"] = "High+"
            ranked[0]["screeningNotes"].append(
                "High+ means top-tier research priority, not a bet recommendation."
            )

    for index, pitcher in enumerate(ranked):
        previous_expected = ranked[index - 1].get("expectedStrikeouts") if index else None
        current_expected = pitcher.get("expectedStrikeouts")
        gap = (
            round(abs(previous_expected - current_expected), 1)
            if previous_expected is not None and current_expected is not None
            else None
        )
        pitcher["expectedKGapFromPrevious"] = gap
        pitcher["expectedKGapLabel"] = _gap_label(gap) if gap is not None else None
        pitcher["stage1Group"] = _stage1_group(pitcher)
        pitcher["recommendedForDeeperResearch"] = pitcher["stage1Group"] == "RESEARCH"
        pitcher["groupReason"] = _group_reason(pitcher)

    recommended = [
        pitcher
        for pitcher in ranked
        if pitcher.get("recommendedForDeeperResearch")
    ][:limit]
    not_selected = [
        pitcher
        for pitcher in ranked
        if pitcher.get("pitcherId")
        not in {item.get("pitcherId") for item in recommended}
    ]

    for rank, pitcher in enumerate(recommended, start=1):
        pitcher["researchRank"] = rank
    for rank, pitcher in enumerate(not_selected, start=len(recommended) + 1):
        pitcher["screenRank"] = rank

    screening_groups = {
        "research": [pitcher for pitcher in ranked if pitcher.get("stage1Group") == "RESEARCH"],
        "borderline": [pitcher for pitcher in ranked if pitcher.get("stage1Group") == "BORDERLINE"],
        "passForNow": [pitcher for pitcher in ranked if pitcher.get("stage1Group") == "PASS FOR NOW"],
    }
    for group_pitchers in screening_groups.values():
        for rank, pitcher in enumerate(group_pitchers, start=1):
            pitcher["groupRank"] = rank

    return recommended, not_selected, screening_groups, _rank_gap_notes(ranked)


def _compact_pitcher(pitcher):
    return {
        "rank": pitcher.get("groupRank"),
        "overallRank": pitcher.get("researchRank") or pitcher.get("screenRank"),
        "pitcherId": pitcher.get("pitcherId"),
        "pitcher": pitcher.get("pitcher"),
        "matchup": pitcher.get("matchup"),
        "team": pitcher.get("team"),
        "opponent": pitcher.get("opponent"),
        "throws": pitcher.get("throws"),
        "baseKRate": pitcher.get("baseKRate"),
        "opponentKRateVsHand": pitcher.get("opponentKRateVsHand"),
        "opponentMatchupAdjustment": pitcher.get("opponentMatchupAdjustment"),
        "recentFormAdjustment": pitcher.get("recentFormAdjustment"),
        "estimatedKRate": pitcher.get("estimatedKRate"),
        "expectedBattersFaced": pitcher.get("expectedBattersFaced"),
        "expectedStrikeouts": pitcher.get("expectedStrikeouts"),
        "expectedStrikeoutRange": pitcher.get("expectedStrikeoutRange"),
        "confidence": pitcher.get("confidence"),
        "stage1Group": pitcher.get("stage1Group"),
        "expectedKGapFromPrevious": pitcher.get("expectedKGapFromPrevious"),
        "expectedKGapLabel": pitcher.get("expectedKGapLabel"),
        "groupReason": pitcher.get("groupReason"),
        "lineupStatus": pitcher.get("lineupStatus"),
        "recentLast5AverageKs": pitcher.get("recentLast5AverageKs"),
        "recentLast5AveragePitches": pitcher.get("recentLast5AveragePitches"),
        "recentLast5AverageBattersFaced": pitcher.get("recentLast5AverageBattersFaced"),
        "recentLast10AverageKs": pitcher.get("recentLast10AverageKs"),
        "recentLast10AveragePitches": pitcher.get("recentLast10AveragePitches"),
        "recentLast10AverageBattersFaced": pitcher.get("recentLast10AverageBattersFaced"),
        "researchPriorityScore": pitcher.get("researchPriorityScore"),
    }


def _compact_groups(screening_groups):
    return {
        group: [_compact_pitcher(pitcher) for pitcher in pitchers]
        for group, pitchers in screening_groups.items()
    }


def _legacy_pitcher_summary(pitcher):
    return {
        "rank": pitcher.get("rank"),
        "pitcher": pitcher.get("pitcher"),
        "matchup": pitcher.get("matchup"),
        "stage1Group": pitcher.get("stage1Group"),
        "expectedStrikeouts": pitcher.get("expectedStrikeouts"),
        "confidence": pitcher.get("confidence"),
        "groupReason": pitcher.get("groupReason"),
    }


def _format_rate(value):
    return f"{round(value * 100, 1)}%" if value is not None else "N/A"


def _format_pp(value):
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{round(value * 100, 1)} pp"


def _format_number(value, digits=1):
    return f"{round(value, digits):.{digits}f}" if value is not None else "N/A"


def _dashboard_row(pitcher):
    return {
        "rank": pitcher.get("rank"),
        "pitcher": pitcher.get("pitcher"),
        "matchup": pitcher.get("matchup"),
        "hand": pitcher.get("throws") or "N/A",
        "baseK": _format_rate(pitcher.get("baseKRate")),
        "matchupAdj": _format_pp(pitcher.get("opponentMatchupAdjustment")),
        "estK": _format_rate(pitcher.get("estimatedKRate")),
        "expBF": _format_number(pitcher.get("expectedBattersFaced")),
        "expKs": _format_number(pitcher.get("expectedStrikeouts")),
        "range": pitcher.get("expectedStrikeoutRange") or "N/A",
        "confidence": pitcher.get("confidence") or "N/A",
        "gapNote": pitcher.get("expectedKGapLabel") or "top rank",
        "whyConcern": pitcher.get("groupReason") or "N/A",
    }


def _unavailable_pitchers(compact_games):
    unavailable = []
    for game in compact_games:
        for pitcher in game.get("pitchers", []):
            if pitcher.get("available"):
                continue
            unavailable.append(
                {
                    "matchup": game.get("matchup"),
                    "team": pitcher.get("team"),
                    "pitcher": pitcher.get("pitcher") or "Not listed",
                    "reason": pitcher.get("reason") or "Unavailable",
                }
            )
    return unavailable


def _stage1_dashboard(date, requested, unmatched, compact_games, compact_groups, rank_gap_notes):
    research_rows = [_dashboard_row(pitcher) for pitcher in compact_groups.get("research", [])]
    borderline_rows = [_dashboard_row(pitcher) for pitcher in compact_groups.get("borderline", [])]
    pass_rows = [_dashboard_row(pitcher) for pitcher in compact_groups.get("passForNow", [])]
    pitchers_screened = len(research_rows) + len(borderline_rows) + len(pass_rows)
    research_names = [row["pitcher"] for row in research_rows]
    what_next = [
        "No bets yet.",
        "Approve the RESEARCH group before Stage 2.",
        "Send Bet365 pitcher-K screenshots for the RESEARCH group first.",
    ]
    if research_names:
        what_next.extend(
            f"{index}. {name}"
            for index, name in enumerate(research_names, start=1)
        )
    else:
        what_next.append("No RESEARCH pitchers cleared the screen. Consider No Bet Day or ask about BORDERLINE soft-line checks.")
    if borderline_rows:
        what_next.append("Optional: include BORDERLINE pitchers only if you want to check for unusually soft Bet365 lines.")

    return {
        "title": "MLB Pitcher-K Screen",
        "subtitle": "Research only. No bets yet.",
        "slateRead": {
            "date": date,
            "requestedMatchups": [f"{away}@{home}" for away, home in requested],
            "unmatchedRequestedMatchups": unmatched,
            "gamesReturned": len(compact_games),
            "pitchersScreened": pitchers_screened,
            "lineupStatus": "Unavailable from helper unless user/browser confirms projected or confirmed lineups.",
            "pitcherKOddsReviewed": "No",
        },
        "rankGapNotes": rank_gap_notes,
        "tables": {
            "research": {
                "title": "RESEARCH",
                "rows": research_rows,
            },
            "borderline": {
                "title": "BORDERLINE",
                "rows": borderline_rows,
            },
            "passForNow": {
                "title": "PASS FOR NOW",
                "rows": pass_rows,
            },
        },
        "unavailablePitchers": _unavailable_pitchers(compact_games),
        "whatINeedNext": what_next,
        "displayRules": [
            "Use these rows in the returned order.",
            "Copy confidence, matchup, gapNote, and whyConcern exactly.",
            "Do not rename teams, reorder rows, upgrade confidence, or rewrite group reasons.",
            "High+ is research priority only, not a bet.",
        ],
    }


def _compact_game(game):
    pitchers = []
    for pitcher in game.get("pitchers", []):
        if pitcher.get("available"):
            model = pitcher.get("expectedKModel", {})
            pitchers.append(
                {
                    "pitcherId": pitcher.get("id"),
                    "pitcher": pitcher.get("name"),
                    "team": pitcher.get("team"),
                    "opponent": pitcher.get("opponent"),
                    "throws": pitcher.get("throws"),
                    "available": True,
                    "expectedStrikeouts": model.get("expectedStrikeouts"),
                    "stage1Confidence": model.get("confidence", {}).get("label"),
                    "researchPriorityScore": pitcher.get("researchPriorityScore"),
                }
            )
        else:
            pitchers.append(
                {
                    "pitcherId": pitcher.get("id"),
                    "pitcher": pitcher.get("name"),
                    "team": pitcher.get("team"),
                    "opponent": pitcher.get("opponent"),
                    "available": False,
                    "reason": pitcher.get("reason"),
                }
            )

    return {
        "gamePk": game.get("gamePk"),
        "gameDate": game.get("gameDate"),
        "status": game.get("status"),
        "matchup": game.get("matchup"),
        "awayTeam": game.get("awayTeam"),
        "homeTeam": game.get("homeTeam"),
        "venue": game.get("venue"),
        "pitchers": pitchers,
    }


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


@app.get("/game-screening")
def game_screening():
    date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    season = request.args.get("season") or date[:4]
    include_details = str(request.args.get("include_details", "false")).lower() in {
        "1",
        "true",
        "yes",
    }
    requested = _requested_matchups(request.args.get("matchups"))
    games = _schedule(date)

    if requested:
        requested_set = set(requested)
        selected_games = []
        found = set()
        for game in games:
            away = _normalize_team_code(
                game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation")
            )
            home = _normalize_team_code(
                game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation")
            )
            if (away, home) in requested_set:
                selected_games.append(game)
                found.add((away, home))
        unmatched = [
            f"{away}@{home}"
            for away, home in requested
            if (away, home) not in found
        ]
    else:
        selected_games = games
        unmatched = []

    screened = []
    with ThreadPoolExecutor(max_workers=min(max(len(selected_games), 1), 8)) as executor:
        futures = [
            executor.submit(_format_screening_game, game, season)
            for game in selected_games
        ]
        for future in as_completed(futures):
            screened.append(future.result())

    screened = sorted(
        screened,
        key=lambda game: game.get("gameResearchPriorityScore") or 0,
        reverse=True,
    )
    for rank, game in enumerate(screened, start=1):
        game["researchRank"] = rank
    (
        recommended_pitchers,
        not_selected_pitchers,
        screening_groups,
        rank_gap_notes,
    ) = _rank_screening_pitchers(screened)
    compact_groups = _compact_groups(screening_groups)
    compact_games = [_compact_game(game) for game in screened]
    compact_recommended = [
        _legacy_pitcher_summary(pitcher)
        for pitcher in compact_groups.get("research", [])
    ]
    compact_not_selected = [
        _legacy_pitcher_summary(pitcher)
        for pitcher in (
            compact_groups.get("borderline", [])
            + compact_groups.get("passForNow", [])
        )
    ]
    stage1_dashboard = _stage1_dashboard(
        date,
        requested,
        unmatched,
        compact_games,
        compact_groups,
        rank_gap_notes,
    )

    payload = {
        "date": date,
        "season": season,
        "screeningOnly": True,
        "responseMode": "compact",
        "stage1Dashboard": stage1_dashboard,
        "requestedMatchups": [
            f"{away}@{home}" for away, home in requested
        ],
        "unmatchedRequestedMatchups": unmatched,
        "gamesReturned": len(screened),
        "games": compact_games,
        "recommendedPitchers": compact_recommended,
        "notSelectedPitchers": compact_not_selected,
        "screeningGroups": compact_groups,
        "rankGapNotes": rank_gap_notes,
        "notes": [
            "This compact response is designed to fit a full screenshot slate in one action call.",
            "Use stage1Dashboard for the user-facing Stage 1 report.",
            "This endpoint screens games for deeper pitcher-K research. It does not recommend bets.",
            "Expected Ks is the preferred Stage 1 ranking field; the old research-priority score remains as a comparison aid.",
            "Recommended pitchers come from the RESEARCH group, up to a maximum of 10.",
            "Every screened pitcher is returned in screeningGroups: research, borderline, or passForNow.",
            "Expected-K gap guide: under 0.30 = tied; 0.30-0.59 = small edge; 0.60-0.99 = meaningful edge; 1.00+ = major edge.",
            "Use measurable screening data only. Reputation or name value is not a reason to research a pitcher.",
            "Send only screenshot-provided matchups in the matchups query when screening a Bet365 screenshot.",
            "Projected lineups can support research with an uncertainty haircut, but never treat projected lineups as confirmed.",
            "Confirm lineups, injury news, weather, Bet365 pitcher-K lines, and prices before any final bet decision.",
        ],
    }
    if include_details:
        payload["responseMode"] = "detailed"
        payload["debugFullGames"] = screened
        payload["debugFullScreeningGroups"] = screening_groups
        payload["debugFullRecommendedPitchers"] = recommended_pitchers
        payload["debugFullNotSelectedPitchers"] = not_selected_pitchers

    return jsonify(payload)


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
