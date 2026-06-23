import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from math import ceil, floor, sqrt
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, request


MLB_BASE = "https://statsapi.mlb.com/api/v1"
MLB_LIVE = "https://statsapi.mlb.com/api/v1.1"
ATLANTIC_TZ = ZoneInfo("America/Moncton")
BORDERLINE_RULES = {
    "minimumRecentStartsForNormalConfidence": 8,
    "smallSampleThreshold": 5,
    "limitedSampleThreshold": 8,
    "shortPriceCutoff": 1.60,
    "mediumPriceCutoff": 1.75,
    "minimumPrice": 1.40,
    "maximumPrice": 1.95,
    "thinGap": 0.30,
    "meaningfulGap": 0.60,
    "strongGap": 1.00,
    "fullLineMove": 1.00,
    "halfLineMove": 0.50,
    "minimumIndependentSignals": 2,
    "minimumEdgePercentagePoints": 2.0,
    "smallSamplePenalty": 0.12,
    "limitedSamplePenalty": 0.07,
    "projectedLineupPenalty": 0.03,
    "staleDataPenalty": 0.03,
    "pitchQualityUnavailablePenalty": 0.04,
    "weakPitchQualityPenalty": 0.05,
    "thinPricePenalty": 0.04,
}

app = Flask(__name__)


def _get_json(url, params=None):
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _format_game_time(game_date):
    if not game_date:
        return "N/A"
    try:
        parsed = datetime.fromisoformat(str(game_date).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        local = parsed.astimezone(ATLANTIC_TZ)
    except ValueError:
        return str(game_date)
    hour = local.strftime("%I").lstrip("0") or "12"
    return f"{hour}:{local.strftime('%M')} {local.strftime('%p')} Atlantic"


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


def _outs_to_ip_text(outs):
    if outs is None:
        return "N/A"
    whole = int(outs) // 3
    remainder = int(outs) % 3
    return f"{whole}.{remainder}"


def _expected_ip_text(expected_outs):
    if expected_outs is None:
        return "N/A"
    return f"{round(expected_outs / 3, 1):.1f}"


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


def _recent_start_workload(person_id, season, limit=5, before_date=None):
    splits = sorted(
        [
            split
            for split in _pitcher_game_log(person_id, season)
            if _is_start(split)
            and (before_date is None or (split.get("date") or "") < before_date)
        ],
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
    outs = [_innings_to_outs(start["inningsPitched"]) for start in starts]
    pitches = [start["pitches"] for start in starts]
    batters_faced = [start["battersFaced"] for start in starts]
    strikeouts = [start["strikeouts"] for start in starts]
    return {
        "startsTracked": len(starts),
        "averageOuts": _avg(outs, 1),
        "averageInningsDecimal": _avg(
            [out / 3 for out in outs],
            1,
        ),
        "averagePitches": _avg(pitches),
        "averageBattersFaced": _avg(batters_faced),
        "averageStrikeouts": _avg(strikeouts),
        "strikeoutRate": _pct(
            sum(_num(start.get("strikeouts"), 0) for start in starts),
            sum(_num(start.get("battersFaced"), 0) for start in starts),
        ),
        "outsStdDev": _stddev(outs),
        "pitchesStdDev": _stddev(pitches),
        "battersFacedStdDev": _stddev(batters_faced),
        "strikeoutsStdDev": _stddev(strikeouts),
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


def _pitcher_screening_profile(person_id, opponent_team_id, season, before_date=None):
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
    recent_last10 = _recent_start_workload(
        person_id,
        season,
        limit=10,
        before_date=before_date,
    )
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
    official_date = game.get("officialDate") or str(game.get("gameDate") or "")[:10]
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
                before_date=official_date,
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
        "gameTime": _format_game_time(game.get("gameDate")),
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
                "gameTime": game.get("gameTime"),
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
        "gameTime": pitcher.get("gameTime"),
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
        "team": pitcher.get("team") or "N/A",
        "matchup": pitcher.get("matchup"),
        "gameTime": pitcher.get("gameTime") or "N/A",
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
                    "gameTime": game.get("gameTime"),
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
    display_columns = [
        {"key": "rank", "label": "Rank"},
        {"key": "pitcher", "label": "Pitcher"},
        {"key": "team", "label": "Team"},
        {"key": "matchup", "label": "Matchup"},
        {"key": "gameTime", "label": "Game Time"},
        {"key": "hand", "label": "Hand"},
        {"key": "baseK", "label": "Base K%"},
        {"key": "matchupAdj", "label": "Matchup Adj"},
        {"key": "estK", "label": "Est K%"},
        {"key": "expBF", "label": "Exp BF"},
        {"key": "expKs", "label": "Exp Ks"},
        {"key": "range", "label": "Range"},
        {"key": "confidence", "label": "Confidence"},
        {"key": "gapNote", "label": "Gap Note"},
        {"key": "whyConcern", "label": "Why / Concern"},
    ]
    what_next = [
        "No bets yet.",
        "Approve the RESEARCH group before Stage 2.",
        "Send Bet365 pitcher-K screenshots for the RESEARCH group first.",
    ]
    stage2_targets = [
        {
            "rank": row.get("rank"),
            "pitcher": row.get("pitcher"),
            "team": row.get("team"),
            "matchup": row.get("matchup"),
            "gameTime": row.get("gameTime"),
        }
        for row in research_rows[:10]
    ]
    if not research_names:
        what_next.append("No RESEARCH pitchers cleared the screen. Consider No Bet Day or ask about BORDERLINE soft-line checks.")
    borderline_option = (
        "Optional: include BORDERLINE pitchers only if you want to check for unusually soft Bet365 lines."
        if borderline_rows
        else None
    )

    dashboard = {
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
                "displayColumns": display_columns,
                "rows": research_rows,
            },
            "borderline": {
                "title": "BORDERLINE",
                "displayColumns": display_columns,
                "rows": borderline_rows,
            },
            "passForNow": {
                "title": "PASS FOR NOW",
                "displayColumns": display_columns,
                "rows": pass_rows,
            },
        },
        "unavailablePitchers": _unavailable_pitchers(compact_games),
        "stage2ScreenshotTargets": stage2_targets,
        "borderlineOption": borderline_option,
        "whatINeedNext": what_next,
        "displayRules": [
            "Use this section order: Slate Read, rankGapNotes, RESEARCH, BORDERLINE, PASS FOR NOW, What I Need Next.",
            "Use these rows in the returned order.",
            "Use every displayColumns entry for each table, in the returned order.",
            "Use stage1ReportMarkdown exactly when available.",
            "Copy confidence, matchup, gapNote, and whyConcern exactly.",
            "Do not rename teams, reorder rows, upgrade confidence, or rewrite group reasons.",
            "High+ is research priority only, not a bet.",
        ],
    }
    dashboard["stage1ReportMarkdown"] = _stage1_report_markdown(dashboard)
    return dashboard


def _markdown_cell(value):
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers, rows):
    header_line = "| " + " | ".join(_markdown_cell(header) for header in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator] + body)


def _stage1_report_markdown(dashboard):
    slate = dashboard.get("slateRead", {})
    lines = [
        "Slate Read",
        "",
        f"{dashboard.get('title')} - {dashboard.get('subtitle')}",
        "",
        _markdown_table(
            ["Field", "Value"],
            [
                ["Date", slate.get("date")],
                ["Games Returned", slate.get("gamesReturned")],
                ["Pitchers Screened", slate.get("pitchersScreened")],
                ["Pitcher-K Odds Reviewed", slate.get("pitcherKOddsReviewed")],
                ["Lineup Status", slate.get("lineupStatus")],
                [
                    "Unmatched Requested Matchups",
                    ", ".join(slate.get("unmatchedRequestedMatchups") or []) or "None",
                ],
            ],
        ),
    ]

    unavailable = dashboard.get("unavailablePitchers") or []
    if unavailable:
        lines.extend(
            [
                "",
                "Unavailable Pitchers",
                "",
                _markdown_table(
                    ["Matchup", "Game Time", "Team", "Pitcher", "Reason"],
                    [
                        [
                            item.get("matchup"),
                            item.get("gameTime"),
                            item.get("team"),
                            item.get("pitcher"),
                            item.get("reason"),
                        ]
                        for item in unavailable
                    ],
                ),
            ]
        )

    lines.extend(["", "rankGapNotes", ""])
    lines.append(
        _markdown_table(
            ["#", "Note"],
            [
                [index, note]
                for index, note in enumerate(dashboard.get("rankGapNotes") or [], start=1)
            ],
        )
    )

    for table_key in ("research", "borderline", "passForNow"):
        table = dashboard.get("tables", {}).get(table_key, {})
        columns = table.get("displayColumns") or []
        rows = table.get("rows") or []
        headers = [column.get("label") for column in columns]
        keys = [column.get("key") for column in columns]
        lines.extend(["", table.get("title", table_key), ""])
        lines.append(
            _markdown_table(
                headers,
                [[row.get(key) for key in keys] for row in rows],
            )
        )

    lines.extend(["", "What I Need Next", ""])
    for item in dashboard.get("whatINeedNext") or []:
        lines.append(item)
    targets = dashboard.get("stage2ScreenshotTargets") or []
    if targets:
        lines.extend(
            [
                "",
                "Stage 2 Screenshot Targets",
                "",
                _markdown_table(
                    ["Rank", "Pitcher", "Team", "Matchup", "Game Time"],
                    [
                        [
                            target.get("rank"),
                            target.get("pitcher"),
                            target.get("team"),
                            target.get("matchup"),
                            target.get("gameTime"),
                        ]
                        for target in targets
                    ],
                ),
            ]
        )
    if dashboard.get("borderlineOption"):
        lines.extend(["", dashboard["borderlineOption"]])

    return "\n".join(lines)


def _outs_gap_label(gap):
    if gap is None:
        return None
    if gap < 0.75:
        return "basically tied"
    if gap < 1.50:
        return "small edge"
    if gap < 2.50:
        return "meaningful edge"
    return "major edge"


def _expected_outs_model(season, opponent_split, recent_last5, recent_last10):
    season_starts = season.get("starts")
    season_outs = _innings_to_outs(season.get("inningsPitched"))
    season_outs_per_start = (
        round(season_outs / season_starts, 1)
        if season_outs is not None and season_starts
        else None
    )
    last5_outs = recent_last5.get("averageOuts")
    last10_outs = recent_last10.get("averageOuts")
    base = _weighted_available(
        [
            (last5_outs, 0.50),
            (last10_outs, 0.30),
            (season_outs_per_start, 0.20),
        ],
        1,
    )
    if base is None:
        return {
            "expectedOuts": None,
            "baseExpectedOuts": None,
            "seasonOutsPerStart": season_outs_per_start,
            "adjustments": [],
            "notes": ["Expected outs unavailable because workload data is missing."],
        }

    adjustments = []
    expected = base
    last5_pitches = recent_last5.get("averagePitches")
    last5_bf = recent_last5.get("averageBattersFaced")
    last10_pitches = recent_last10.get("averagePitches")
    pitches_per_out = (
        round(last10_pitches / last10_outs, 2)
        if last10_pitches is not None and last10_outs
        else None
    )
    pitches_per_bf = (
        round(last10_pitches / recent_last10.get("averageBattersFaced"), 2)
        if last10_pitches is not None and recent_last10.get("averageBattersFaced")
        else None
    )

    if last5_pitches is not None:
        if last5_pitches >= 96:
            adjustments.append({"reason": "high recent pitch-count leash", "value": 0.7})
        elif last5_pitches >= 90:
            adjustments.append({"reason": "stable recent pitch count", "value": 0.3})
        elif last5_pitches < 80:
            adjustments.append({"reason": "light recent pitch count", "value": -1.0})
        elif last5_pitches < 86:
            adjustments.append({"reason": "below-average recent pitch count", "value": -0.5})

    if last5_bf is not None:
        if last5_bf >= 25:
            adjustments.append({"reason": "stable recent BF volume", "value": 0.4})
        elif last5_bf < 21:
            adjustments.append({"reason": "limited recent BF volume", "value": -0.8})

    if pitches_per_out is not None:
        if pitches_per_out <= 5.0:
            adjustments.append({"reason": "efficient pitches per out", "value": 0.5})
        elif pitches_per_out >= 6.2:
            adjustments.append({"reason": "poor pitches-per-out efficiency", "value": -1.0})
        elif pitches_per_out >= 5.8:
            adjustments.append({"reason": "pitches-per-out drag", "value": -0.6})

    if pitches_per_bf is not None:
        if pitches_per_bf >= 4.45:
            adjustments.append({"reason": "inefficient pitches per batter faced", "value": -0.5})
        elif pitches_per_bf <= 3.75:
            adjustments.append({"reason": "efficient BF conversion", "value": 0.2})

    bb_rate = season.get("bbRate")
    if bb_rate is not None:
        if bb_rate >= 0.105:
            adjustments.append({"reason": "high walk-rate length risk", "value": -0.9})
        elif bb_rate >= 0.09:
            adjustments.append({"reason": "walk-rate efficiency concern", "value": -0.5})
        elif bb_rate <= 0.055:
            adjustments.append({"reason": "low walk-rate length support", "value": 0.2})

    opponent_ops = opponent_split.get("ops")
    opponent_obp = opponent_split.get("obp")
    opponent_k = opponent_split.get("strikeoutRate")
    if opponent_ops is not None:
        if opponent_ops >= 0.760:
            adjustments.append({"reason": "opponent power/on-base pressure", "value": -0.5})
        elif opponent_ops <= 0.680:
            adjustments.append({"reason": "lighter opponent OPS split", "value": 0.2})
    if opponent_obp is not None and opponent_obp >= 0.335:
        adjustments.append({"reason": "opponent OBP can run pitch count", "value": -0.3})
    if opponent_k is not None and opponent_k >= 0.25:
        adjustments.append({"reason": "opponent swing-and-miss helps length", "value": 0.2})

    short_recent = sum(
        1
        for start in recent_last5.get("starts", [])
        if _innings_to_outs(start.get("inningsPitched")) < 15
    )
    if short_recent >= 3:
        adjustments.append({"reason": "three recent starts under 5 IP", "value": -1.4})
    elif short_recent >= 2:
        adjustments.append({"reason": "multiple recent short starts", "value": -0.8})

    for adjustment in adjustments:
        expected += adjustment["value"]

    expected = round(_clamp(expected, 9, 24), 1)
    return {
        "expectedOuts": expected,
        "expectedIP": _expected_ip_text(expected),
        "baseExpectedOuts": base,
        "seasonOutsPerStart": season_outs_per_start,
        "pitchesPerOutLast10": pitches_per_out,
        "pitchesPerBatterFacedLast10": pitches_per_bf,
        "adjustments": adjustments,
        "notes": [
            "Base Expected Outs uses 50% last 5 outs, 30% last 10 outs, and 20% season outs/start.",
            "Adjustments reflect pitch count, BF volume, efficiency, walk risk, opponent pressure, and recent short starts.",
        ],
    }


def _outs_confidence_label(season, recent_last5, recent_last10, expected_outs):
    points = 0.0
    starts_tracked = recent_last10.get("startsTracked") or 0
    last5_outs = recent_last5.get("averageOuts")
    last5_pitches = recent_last5.get("averagePitches")
    last5_bf = recent_last5.get("averageBattersFaced")
    outs_stddev = recent_last10.get("outsStdDev")
    bb_rate = season.get("bbRate")

    if expected_outs is not None:
        if expected_outs >= 18:
            points += 1.2
        elif expected_outs >= 16:
            points += 0.7
        elif expected_outs < 14:
            points -= 1.0

    if last5_outs is not None:
        if last5_outs >= 18:
            points += 1.2
        elif last5_outs >= 16:
            points += 0.7
        elif last5_outs < 14:
            points -= 1.0

    if last5_pitches is not None:
        if last5_pitches >= 94:
            points += 1.0
        elif last5_pitches >= 88:
            points += 0.5
        elif last5_pitches < 80:
            points -= 1.0

    if last5_bf is not None:
        if last5_bf >= 24:
            points += 0.6
        elif last5_bf < 21:
            points -= 0.7

    if starts_tracked >= 8:
        points += 0.5
    elif starts_tracked < 5:
        points -= 0.8

    if outs_stddev is not None:
        if outs_stddev <= 2.0:
            points += 0.4
        elif outs_stddev >= 4.0:
            points -= 0.7

    if bb_rate is not None and bb_rate >= 0.105:
        points -= 0.8
    elif bb_rate is not None and bb_rate <= 0.055:
        points += 0.2

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
            "Confidence is workload confidence for outs research, not bet confidence.",
            "High+ is applied only after ranking when a High profile separates from the board.",
        ],
    }


def _pitcher_outs_screening_profile(person_id, opponent_team_id, season, before_date=None):
    base = _pitcher_screening_profile(
        person_id,
        opponent_team_id,
        season,
        before_date=before_date,
    )
    season_payload = base.get("season", {})
    recent_last5 = base.get("recentLast5", {})
    recent_last10 = base.get("recentLast10", {})
    opponent_split = base.get("opponentHittingSplit", {})
    expected_model = _expected_outs_model(
        season_payload,
        opponent_split,
        recent_last5,
        recent_last10,
    )
    confidence = _outs_confidence_label(
        season_payload,
        recent_last5,
        recent_last10,
        expected_model.get("expectedOuts"),
    )
    expected_outs = expected_model.get("expectedOuts")
    last5_pitches = recent_last5.get("averagePitches")
    last5_outs = recent_last5.get("averageOuts")
    last10_outs = recent_last10.get("averageOuts")
    score = (
        0.35 * _normalize_score(expected_outs, 12, 20)
        + 0.20 * _normalize_score(last5_outs, 12, 20)
        + 0.15 * _normalize_score(last10_outs, 12, 20)
        + 0.15 * _normalize_score(last5_pitches, 75, 100)
        + 0.15 * _normalize_score(confidence.get("score"), 0, 4)
    )
    base["expectedOutsModel"] = {
        "expectedOuts": expected_outs,
        "expectedIP": expected_model.get("expectedIP"),
        "expectedOutsRange": _expected_outs_range(expected_outs, confidence.get("label"), recent_last10),
        "expectedOutsModel": expected_model,
        "confidence": confidence,
        "lineupStatus": "Unavailable from helper",
        "lineupConfidenceRule": "Projected lineups can support research, but projected or unavailable lineups should reduce final confidence.",
    }
    base["outsResearchPriorityScore"] = round(score * 100, 1)
    base["scoreNotes"] = [
        "Outs research-priority score is a screening aid, not a bet recommendation.",
        "Score weights expected outs, recent outs, recent pitch count, and workload confidence.",
        "Lineups, news, weather, outs line, and price still require later checks.",
    ]
    return base


def _expected_outs_range(expected_outs, confidence_label, recent_last10):
    if expected_outs is None:
        return None
    width = {
        "Low": 4.0,
        "Medium": 3.2,
        "Medium+": 2.7,
        "High": 2.3,
        "High+": 2.0,
    }.get(confidence_label, 3.2)
    volatility = recent_last10.get("outsStdDev")
    if volatility is not None and volatility >= 4.0:
        width += 0.8
    elif volatility is not None and volatility <= 2.0:
        width -= 0.4
    low = max(0, floor(expected_outs - width))
    high = max(low, ceil(expected_outs + width))
    return {"low": low, "high": high, "display": f"{low}-{high}"}


def _format_out_screening_game(game, season):
    away = game.get("teams", {}).get("away", {})
    home = game.get("teams", {}).get("home", {})
    away_code = _normalize_team_code(away.get("team", {}).get("abbreviation"))
    home_code = _normalize_team_code(home.get("team", {}).get("abbreviation"))
    official_date = game.get("officialDate") or str(game.get("gameDate") or "")[:10]
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
            profile = _pitcher_outs_screening_profile(
                probable["id"],
                opponent_side.get("team", {}).get("id"),
                season,
                before_date=official_date,
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
                    "reason": "MLB Stats API outs-screening data unavailable.",
                }
            )

    available_scores = [
        pitcher["outsResearchPriorityScore"]
        for pitcher in pitchers
        if pitcher.get("outsResearchPriorityScore") is not None
    ]
    return {
        "gamePk": game.get("gamePk"),
        "gameDate": game.get("gameDate"),
        "gameTime": _format_game_time(game.get("gameDate")),
        "status": game.get("status", {}).get("detailedState"),
        "matchup": f"{away_code}@{home_code}",
        "awayTeam": away.get("team", {}).get("name"),
        "homeTeam": home.get("team", {}).get("name"),
        "venue": game.get("venue", {}).get("name"),
        "pitchers": pitchers,
        "gameResearchPriorityScore": max(available_scores) if available_scores else None,
    }


def _outs_tie_breaker_score(pitcher):
    recent = pitcher.get("recentLast5", {})
    recent10 = pitcher.get("recentLast10", {})
    model = pitcher.get("expectedOutsModel", {})
    confidence = model.get("confidence", {})
    score = (
        0.35 * _normalize_score(model.get("expectedOuts"), 12, 20)
        + 0.25 * _normalize_score(recent.get("averageOuts"), 12, 20)
        + 0.20 * _normalize_score(recent.get("averagePitches"), 75, 100)
        + 0.20 * _normalize_score(confidence.get("score") or 0, 0, 4)
    )
    if recent10.get("outsStdDev") is not None and recent10.get("outsStdDev") <= 2.0:
        score += 0.03
    return round(score * 100, 1)


def _sort_by_expected_outs_with_ties(pitchers):
    base = sorted(
        pitchers,
        key=lambda pitcher: (
            pitcher.get("expectedOuts")
            if pitcher.get("expectedOuts") is not None
            else -1,
            pitcher.get("tieBreakerScore") or 0,
        ),
        reverse=True,
    )
    clusters = []
    current = []
    anchor = None
    for pitcher in base:
        expected_outs = pitcher.get("expectedOuts")
        if expected_outs is None:
            if current:
                clusters.append(current)
                current = []
                anchor = None
            clusters.append([pitcher])
            continue
        if anchor is None:
            anchor = expected_outs
            current = [pitcher]
            continue
        if anchor - expected_outs < 0.75:
            current.append(pitcher)
        else:
            clusters.append(current)
            anchor = expected_outs
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
                pitcher["rankingCluster"] = "Expected-outs gap under 0.75; tie-breakers ordered this cluster."
        ranked.extend(cluster)
    return ranked


def _outs_stage1_group(pitcher):
    expected_outs = pitcher.get("expectedOuts")
    confidence = pitcher.get("workloadConfidence")
    last5_outs = pitcher.get("recentLast5AverageOuts")
    last5_pitches = pitcher.get("recentLast5AveragePitches")

    if expected_outs is None:
        return "PASS FOR NOW"
    if expected_outs >= 16.0 and confidence != "Low":
        return "RESEARCH"
    if (
        expected_outs >= 15.5
        and last5_outs is not None
        and last5_outs >= 15.0
        and last5_pitches is not None
        and last5_pitches >= 88
        and confidence in {"Medium+", "High", "High+"}
    ):
        return "RESEARCH"
    if expected_outs >= 14.5 or confidence in {"Medium+", "High", "High+"}:
        return "BORDERLINE"
    return "PASS FOR NOW"


def _outs_group_reason(pitcher):
    expected_outs = pitcher.get("expectedOuts")
    last5_outs = pitcher.get("recentLast5AverageOuts")
    last5_pitches = pitcher.get("recentLast5AveragePitches")
    confidence = pitcher.get("workloadConfidence")
    efficiency = pitcher.get("pitchesPerOutLast10")

    if expected_outs is None:
        return "Missing expected-outs estimate."
    if pitcher.get("stage1Group") == "RESEARCH":
        return "Expected outs, recent length, pitch count, and workload profile are strong enough for pitcher-outs screenshots."
    if pitcher.get("stage1Group") == "BORDERLINE":
        concerns = []
        if expected_outs < 16:
            concerns.append("expected outs below primary research group")
        if last5_outs is not None and last5_outs < 15.5:
            concerns.append("recent outs are mixed")
        if last5_pitches is not None and last5_pitches < 88:
            concerns.append("recent pitch count is not strong")
        if efficiency is not None and efficiency >= 5.8:
            concerns.append("pitches-per-out efficiency risk")
        if confidence == "Low":
            concerns.append("low workload confidence")
        return "; ".join(concerns[:2]) or "Interesting but needs a soft outs line."
    reasons = []
    if expected_outs < 14.5:
        reasons.append("expected outs below screen threshold")
    if last5_pitches is not None and last5_pitches < 84:
        reasons.append("light recent pitch count")
    if last5_outs is not None and last5_outs < 14:
        reasons.append("short recent outings")
    if confidence == "Low":
        reasons.append("low workload confidence")
    return "; ".join(reasons[:2]) or "Does not clear the Stage 1 outs threshold."


def _outs_rank_gap_notes(ranked):
    notes = [
        "Expected-outs gap guide: under 0.75 = basically tied; 0.75-1.49 = small edge; 1.50-2.49 = meaningful edge; 2.50+ = major edge.",
    ]
    expected_values = [
        pitcher.get("expectedOuts")
        for pitcher in ranked
        if pitcher.get("expectedOuts") is not None
    ]
    if len(expected_values) < 2:
        return notes
    top_gap = round(abs(expected_values[0] - expected_values[1]), 1)
    notes.append(
        f"Ranks 1-2 are separated by {top_gap} expected outs: {_outs_gap_label(top_gap)}."
    )
    tight_clusters = []
    cluster_start = 0
    cluster_values = []
    for index, expected in enumerate(expected_values):
        proposed = cluster_values + [expected]
        if proposed and max(proposed) - min(proposed) < 0.75:
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
            + " are within 0.75 expected outs, so tie-breakers matter."
        )
    return notes


def _rank_out_screening_pitchers(games, limit=10):
    pitchers = []
    for game in games:
        for pitcher in game.get("pitchers", []):
            score = pitcher.get("outsResearchPriorityScore")
            if not pitcher.get("available") or score is None:
                continue

            recent = pitcher.get("recentLast5", {})
            recent10 = pitcher.get("recentLast10", {})
            model = pitcher.get("expectedOutsModel", {})
            expected_model = model.get("expectedOutsModel", {})
            confidence = model.get("confidence", {})
            expected_range = model.get("expectedOutsRange") or {}
            flat_pitcher = {
                "pitcherId": pitcher.get("id"),
                "pitcher": pitcher.get("name"),
                "team": pitcher.get("team"),
                "opponent": pitcher.get("opponent"),
                "matchup": game.get("matchup"),
                "gameDate": game.get("gameDate"),
                "gameTime": game.get("gameTime"),
                "venue": game.get("venue"),
                "throws": pitcher.get("throws"),
                "outsResearchPriorityScore": score,
                "expectedOuts": model.get("expectedOuts"),
                "expectedIP": model.get("expectedIP"),
                "expectedOutsRange": expected_range.get("display"),
                "workloadConfidence": confidence.get("label"),
                "workloadConfidenceScore": confidence.get("score"),
                "lineupStatus": model.get("lineupStatus"),
                "expectedOutsBase": expected_model.get("baseExpectedOuts"),
                "expectedOutsAdjustments": expected_model.get("adjustments"),
                "seasonOutsPerStart": expected_model.get("seasonOutsPerStart"),
                "pitchesPerOutLast10": expected_model.get("pitchesPerOutLast10"),
                "pitchesPerBatterFacedLast10": expected_model.get("pitchesPerBatterFacedLast10"),
                "recentLast5AverageOuts": recent.get("averageOuts"),
                "recentLast5AveragePitches": recent.get("averagePitches"),
                "recentLast5AverageBattersFaced": recent.get("averageBattersFaced"),
                "recentLast10AverageOuts": recent10.get("averageOuts"),
                "recentLast10AveragePitches": recent10.get("averagePitches"),
                "recentLast10AverageBattersFaced": recent10.get("averageBattersFaced"),
                "recommendedForDeeperResearch": False,
                "screeningNotes": [
                    "Use Expected Outs as the primary Stage 1 ranking field.",
                    "Use measurable workload data only. Reputation or name value is not a reason to research a pitcher.",
                    "This is a research shortlist, not a bet recommendation.",
                    "Bet365 pitcher-outs line, price, lineup, news, and weather still require later checks.",
                ],
            }
            flat_pitcher["tieBreakerScore"] = _outs_tie_breaker_score(pitcher)
            pitchers.append(flat_pitcher)

    ranked = _sort_by_expected_outs_with_ties(pitchers)
    if ranked:
        top_expected = ranked[0].get("expectedOuts")
        second_expected = ranked[1].get("expectedOuts") if len(ranked) > 1 else None
        top_gap = (
            round(top_expected - second_expected, 1)
            if top_expected is not None and second_expected is not None
            else None
        )
        if (
            ranked[0].get("workloadConfidence") == "High"
            and ranked[0].get("expectedOuts") is not None
            and ranked[0]["expectedOuts"] >= 18
            and (top_gap is None or top_gap >= 1.5 or ranked[0]["expectedOuts"] >= 19)
        ):
            ranked[0]["workloadConfidence"] = "High+"
            ranked[0]["screeningNotes"].append(
                "High+ means top-tier outs research priority, not a bet recommendation."
            )

    for index, pitcher in enumerate(ranked):
        previous_expected = ranked[index - 1].get("expectedOuts") if index else None
        current_expected = pitcher.get("expectedOuts")
        gap = (
            round(abs(previous_expected - current_expected), 1)
            if previous_expected is not None and current_expected is not None
            else None
        )
        pitcher["expectedOutsGapFromPrevious"] = gap
        pitcher["expectedOutsGapLabel"] = _outs_gap_label(gap) if gap is not None else None
        pitcher["stage1Group"] = _outs_stage1_group(pitcher)
        pitcher["recommendedForDeeperResearch"] = pitcher["stage1Group"] == "RESEARCH"
        pitcher["groupReason"] = _outs_group_reason(pitcher)

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
    return recommended, not_selected, screening_groups, _outs_rank_gap_notes(ranked)


def _compact_out_pitcher(pitcher):
    return {
        "rank": pitcher.get("groupRank"),
        "overallRank": pitcher.get("researchRank") or pitcher.get("screenRank"),
        "pitcherId": pitcher.get("pitcherId"),
        "pitcher": pitcher.get("pitcher"),
        "team": pitcher.get("team"),
        "matchup": pitcher.get("matchup"),
        "gameTime": pitcher.get("gameTime"),
        "throws": pitcher.get("throws"),
        "expectedOuts": pitcher.get("expectedOuts"),
        "expectedIP": pitcher.get("expectedIP"),
        "expectedOutsRange": pitcher.get("expectedOutsRange"),
        "workloadConfidence": pitcher.get("workloadConfidence"),
        "stage1Group": pitcher.get("stage1Group"),
        "expectedOutsGapFromPrevious": pitcher.get("expectedOutsGapFromPrevious"),
        "expectedOutsGapLabel": pitcher.get("expectedOutsGapLabel"),
        "groupReason": pitcher.get("groupReason"),
        "lineupStatus": pitcher.get("lineupStatus"),
        "recentLast5AverageOuts": pitcher.get("recentLast5AverageOuts"),
        "recentLast5AveragePitches": pitcher.get("recentLast5AveragePitches"),
        "recentLast5AverageBattersFaced": pitcher.get("recentLast5AverageBattersFaced"),
        "recentLast10AverageOuts": pitcher.get("recentLast10AverageOuts"),
        "recentLast10AveragePitches": pitcher.get("recentLast10AveragePitches"),
        "recentLast10AverageBattersFaced": pitcher.get("recentLast10AverageBattersFaced"),
        "pitchesPerOutLast10": pitcher.get("pitchesPerOutLast10"),
        "outsResearchPriorityScore": pitcher.get("outsResearchPriorityScore"),
    }


def _compact_out_game(game):
    pitchers = []
    for pitcher in game.get("pitchers", []):
        if pitcher.get("available"):
            model = pitcher.get("expectedOutsModel", {})
            pitchers.append(
                {
                    "pitcherId": pitcher.get("id"),
                    "pitcher": pitcher.get("name"),
                    "team": pitcher.get("team"),
                    "opponent": pitcher.get("opponent"),
                    "throws": pitcher.get("throws"),
                    "available": True,
                    "expectedOuts": model.get("expectedOuts"),
                    "expectedIP": model.get("expectedIP"),
                    "workloadConfidence": model.get("confidence", {}).get("label"),
                    "outsResearchPriorityScore": pitcher.get("outsResearchPriorityScore"),
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
        "gameTime": game.get("gameTime"),
        "status": game.get("status"),
        "matchup": game.get("matchup"),
        "awayTeam": game.get("awayTeam"),
        "homeTeam": game.get("homeTeam"),
        "venue": game.get("venue"),
        "pitchers": pitchers,
    }


def _compact_out_groups(screening_groups):
    return {
        group: [_compact_out_pitcher(pitcher) for pitcher in pitchers]
        for group, pitchers in screening_groups.items()
    }


def _outs_dashboard_row(pitcher):
    return {
        "rank": pitcher.get("rank"),
        "pitcher": pitcher.get("pitcher"),
        "team": pitcher.get("team") or "N/A",
        "matchup": pitcher.get("matchup"),
        "gameTime": pitcher.get("gameTime") or "N/A",
        "hand": pitcher.get("throws") or "N/A",
        "expOuts": _format_number(pitcher.get("expectedOuts")),
        "expIP": pitcher.get("expectedIP") or "N/A",
        "last5AvgOuts": _format_number(pitcher.get("recentLast5AverageOuts")),
        "last10AvgOuts": _format_number(pitcher.get("recentLast10AverageOuts")),
        "last5AvgPitches": _format_number(pitcher.get("recentLast5AveragePitches")),
        "last5AvgBF": _format_number(pitcher.get("recentLast5AverageBattersFaced")),
        "workloadConfidence": pitcher.get("workloadConfidence") or "N/A",
        "gapNote": pitcher.get("expectedOutsGapLabel") or "top rank",
        "whyConcern": pitcher.get("groupReason") or "N/A",
    }


def _outs_stage1_dashboard(date, requested, unmatched, compact_games, compact_groups, rank_gap_notes):
    research_rows = [_outs_dashboard_row(pitcher) for pitcher in compact_groups.get("research", [])]
    borderline_rows = [_outs_dashboard_row(pitcher) for pitcher in compact_groups.get("borderline", [])]
    pass_rows = [_outs_dashboard_row(pitcher) for pitcher in compact_groups.get("passForNow", [])]
    pitchers_screened = len(research_rows) + len(borderline_rows) + len(pass_rows)
    display_columns = [
        {"key": "rank", "label": "Rank"},
        {"key": "pitcher", "label": "Pitcher"},
        {"key": "team", "label": "Team"},
        {"key": "matchup", "label": "Matchup"},
        {"key": "gameTime", "label": "Game Time"},
        {"key": "hand", "label": "Hand"},
        {"key": "expOuts", "label": "Exp Outs"},
        {"key": "expIP", "label": "Exp IP"},
        {"key": "last5AvgOuts", "label": "Last 5 Avg Outs"},
        {"key": "last10AvgOuts", "label": "Last 10 Avg Outs"},
        {"key": "last5AvgPitches", "label": "Last 5 Avg Pitches"},
        {"key": "last5AvgBF", "label": "Last 5 Avg BF"},
        {"key": "workloadConfidence", "label": "Workload Confidence"},
        {"key": "gapNote", "label": "Gap Note"},
        {"key": "whyConcern", "label": "Why / Concern"},
    ]
    what_next = [
        "No bets yet.",
        "Approve, remove, or adjust the Stage 2 Screenshot Targets before Stage 2.",
        "Send Bet365 pitcher-outs screenshots for the Stage 2 Screenshot Targets first.",
    ]
    if not research_rows:
        what_next.append("No RESEARCH pitchers cleared the screen. Consider No Bet Day or ask about BORDERLINE soft-line checks.")
    stage2_targets = [
        {
            "rank": row.get("rank"),
            "pitcher": row.get("pitcher"),
            "team": row.get("team"),
            "matchup": row.get("matchup"),
            "gameTime": row.get("gameTime"),
        }
        for row in research_rows[:10]
    ]
    borderline_option = (
        "Optional: include BORDERLINE pitchers only if you want to check for unusually soft Bet365 outs lines."
        if borderline_rows
        else None
    )
    dashboard = {
        "title": "MLB Pitcher Outs Screen",
        "subtitle": "Research only. No bets yet.",
        "slateRead": {
            "date": date,
            "requestedMatchups": [f"{away}@{home}" for away, home in requested],
            "unmatchedRequestedMatchups": unmatched,
            "gamesReturned": len(compact_games),
            "pitchersScreened": pitchers_screened,
            "lineupStatus": "Unavailable from helper unless user/browser confirms projected or confirmed lineups.",
            "pitcherOutsOddsReviewed": "No",
        },
        "rankGapNotes": rank_gap_notes,
        "tables": {
            "research": {"title": "RESEARCH", "displayColumns": display_columns, "rows": research_rows},
            "borderline": {"title": "BORDERLINE", "displayColumns": display_columns, "rows": borderline_rows},
            "passForNow": {"title": "PASS FOR NOW", "displayColumns": display_columns, "rows": pass_rows},
        },
        "unavailablePitchers": _unavailable_pitchers(compact_games),
        "stage2ScreenshotTargets": stage2_targets,
        "borderlineOption": borderline_option,
        "whatINeedNext": what_next,
        "displayRules": [
            "Use this section order: Slate Read, rankGapNotes, RESEARCH, BORDERLINE, PASS FOR NOW, What I Need Next.",
            "Use these rows in the returned order.",
            "Use every displayColumns entry for each table, in the returned order.",
            "Use outsStage1ReportMarkdown exactly when available.",
            "Do not rename teams, reorder rows, upgrade confidence, or rewrite group reasons.",
            "High+ is research priority only, not a bet.",
        ],
    }
    dashboard["outsStage1ReportMarkdown"] = _outs_stage1_report_markdown(dashboard)
    return dashboard


def _outs_stage1_report_markdown(dashboard):
    slate = dashboard.get("slateRead", {})
    lines = [
        "Slate Read",
        "",
        f"{dashboard.get('title')} - {dashboard.get('subtitle')}",
        "",
        _markdown_table(
            ["Field", "Value"],
            [
                ["Date", slate.get("date")],
                ["Games Returned", slate.get("gamesReturned")],
                ["Pitchers Screened", slate.get("pitchersScreened")],
                ["Pitcher-Outs Odds Reviewed", slate.get("pitcherOutsOddsReviewed")],
                ["Lineup Status", slate.get("lineupStatus")],
                ["Unmatched Requested Matchups", ", ".join(slate.get("unmatchedRequestedMatchups") or []) or "None"],
            ],
        ),
    ]
    unavailable = dashboard.get("unavailablePitchers") or []
    if unavailable:
        lines.extend(
            [
                "",
                "Unavailable Pitchers",
                "",
                _markdown_table(
                    ["Matchup", "Game Time", "Team", "Pitcher", "Reason"],
                    [
                        [item.get("matchup"), item.get("gameTime"), item.get("team"), item.get("pitcher"), item.get("reason")]
                        for item in unavailable
                    ],
                ),
            ]
        )
    lines.extend(["", "rankGapNotes", ""])
    lines.append(_markdown_table(["#", "Note"], [[index, note] for index, note in enumerate(dashboard.get("rankGapNotes") or [], start=1)]))
    for table_key in ("research", "borderline", "passForNow"):
        table = dashboard.get("tables", {}).get(table_key, {})
        columns = table.get("displayColumns") or []
        rows = table.get("rows") or []
        headers = [column.get("label") for column in columns]
        keys = [column.get("key") for column in columns]
        lines.extend(["", table.get("title", table_key), ""])
        lines.append(_markdown_table(headers, [[row.get(key) for key in keys] for row in rows]))
    lines.extend(["", "What I Need Next", ""])
    for item in dashboard.get("whatINeedNext") or []:
        lines.append(item)
    targets = dashboard.get("stage2ScreenshotTargets") or []
    if targets:
        lines.extend(
            [
                "",
                "Stage 2 Screenshot Targets",
                "",
                _markdown_table(
                    ["Rank", "Pitcher", "Team", "Matchup", "Game Time"],
                    [
                        [target.get("rank"), target.get("pitcher"), target.get("team"), target.get("matchup"), target.get("gameTime")]
                        for target in targets
                    ],
                ),
            ]
        )
    if dashboard.get("borderlineOption"):
        lines.extend(["", dashboard["borderlineOption"]])
    return "\n".join(lines)


def _outs_result_vs_line(outs, line, side):
    if outs is None or line is None:
        return "N/A"
    if outs == line:
        return "Push"
    if side == "over":
        return "Yes" if outs > line else "No"
    return "Yes" if outs < line else "No"


def _outs_start_row(split, person_id, line):
    stat = split.get("stat", {})
    game_pk = _game_pk(split)
    pitch_count = stat.get("numberOfPitches")
    if pitch_count is None:
        pitch_count = _pitch_count_from_boxscore(game_pk, person_id)
    outs = _innings_to_outs(stat.get("inningsPitched"))
    return {
        "date": split.get("date"),
        "opponent": split.get("opponent", {}).get("name"),
        "inningsPitched": stat.get("inningsPitched"),
        "outs": outs,
        "pitches": _num(pitch_count, None) if pitch_count is not None else None,
        "battersFaced": _num(stat.get("battersFaced"), None),
        "walks": _num(stat.get("baseOnBalls"), None),
        "earnedRuns": _num(stat.get("earnedRuns"), None),
        "overResult": _outs_result_vs_line(outs, line, "over"),
        "underResult": _outs_result_vs_line(outs, line, "under"),
    }


def _outs_hit_summary(starts, line, sample):
    items = starts[:sample]
    if not items:
        return {
            "sample": 0,
            "averageOuts": None,
            "averageIP": None,
            "averagePitches": None,
            "averageBattersFaced": None,
            "over": "0/0",
            "under": "0/0",
        }
    over_hits = sum(1 for start in items if start.get("outs") is not None and start["outs"] > line)
    under_hits = sum(1 for start in items if start.get("outs") is not None and start["outs"] < line)
    pushes = sum(1 for start in items if start.get("outs") == line)
    suffix = f", {pushes} push" if pushes == 1 else f", {pushes} pushes" if pushes else ""
    avg_outs = _avg([start.get("outs") for start in items])
    return {
        "sample": len(items),
        "averageOuts": avg_outs,
        "averageIP": _expected_ip_text(avg_outs) if avg_outs is not None else None,
        "averagePitches": _avg([start.get("pitches") for start in items]),
        "averageBattersFaced": _avg([start.get("battersFaced") for start in items]),
        "over": f"{over_hits}/{len(items)}{suffix}",
        "under": f"{under_hits}/{len(items)}{suffix}",
    }


def _outs_recent_starts_from_splits(person_id, line, splits):
    starts = [_outs_start_row(split, person_id, line) for split in splits]
    return {
        "starts": starts,
        "last5": _outs_hit_summary(starts, line, 5),
        "last10": _outs_hit_summary(starts, line, 10),
    }


def _outs_decision(candidate, expected_outs, summary):
    line = candidate.get("line")
    over_odds = candidate.get("overOdds")
    under_odds = candidate.get("underOdds")
    gap = round(expected_outs - line, 1) if expected_outs is not None and line is not None else None
    last5 = summary.get("last5", {})
    last10 = summary.get("last10", {})
    avg_pitches = last5.get("averagePitches")
    avg_bf = last5.get("averageBattersFaced")
    over_l10_hits = _num(str(last10.get("over", "0/0")).split("/")[0], 0)
    under_l10_hits = _num(str(last10.get("under", "0/0")).split("/")[0], 0)
    over_in_range = _odds_in_range(over_odds)
    under_in_range = _odds_in_range(under_odds)

    if gap is None:
        return {
            "bestSide": "Monitor Both Sides",
            "status": "Blocked",
            "reason": "Expected outs unavailable, so side cannot be judged cleanly.",
            "gap": gap,
        }

    workload_ok = (
        (avg_pitches is None or avg_pitches >= 86)
        and (avg_bf is None or avg_bf >= 21)
    )
    if gap >= 1.0 and over_in_range and workload_ok and (
        over_l10_hits >= 6 or (over_l10_hits >= 5 and (last5.get("averageOuts") or 0) >= line + 1)
    ):
        return {
            "bestSide": "Over Candidate",
            "status": "Carry Forward",
            "reason": "Expected outs are meaningfully above the line, Over price is in range, and recent workload supports the Over.",
            "gap": gap,
        }
    if gap <= -1.0 and under_in_range and under_l10_hits >= 5:
        return {
            "bestSide": "Under Candidate",
            "status": "Carry Forward",
            "reason": "Expected outs are meaningfully below the line, Under price is in range, and recent length supports the Under.",
            "gap": gap,
        }
    if not over_in_range and not under_in_range:
        return {
            "bestSide": "Pass Both Sides",
            "status": "Pass Both Sides",
            "reason": "Both prices are outside the 1.40-1.95 range.",
            "gap": gap,
        }
    if abs(gap) < 1.0:
        return {
            "bestSide": "Monitor Both Sides",
            "status": "Monitor",
            "reason": "Expected outs are close to the line, so price, lineup, weather, and recheck context matter.",
            "gap": gap,
        }
    return {
        "bestSide": "Monitor Both Sides",
        "status": "Monitor",
        "reason": "There is a directional lean, but price, workload, or recent hit rate is not clean enough yet.",
        "gap": gap,
    }


def _stage2_outs_candidate_profile(candidate, season, games_by_matchup, date):
    game = games_by_matchup.get(candidate.get("matchup"))
    game_time = _format_game_time(game.get("gameDate")) if game else "N/A"
    pitcher = _find_pitcher(candidate["pitcher"])
    if not pitcher:
        return {
            "candidate": candidate,
            "gameTime": game_time,
            "error": f"No pitcher found for {candidate['pitcher']}.",
        }
    opponent_team_id = _opponent_team_id_for_candidate(candidate, games_by_matchup)
    expected_outs = None
    expected_ip = None
    if opponent_team_id:
        profile = _pitcher_outs_screening_profile(
            pitcher["id"],
            opponent_team_id,
            season,
            before_date=date,
        )
        expected_model = profile.get("expectedOutsModel", {})
        expected_outs = expected_model.get("expectedOuts")
        expected_ip = expected_model.get("expectedIP")
    recent_splits = _stage2_recent_start_splits(
        pitcher["id"],
        season,
        limit=10,
        before_date=date,
    )
    recent = _outs_recent_starts_from_splits(pitcher["id"], candidate["line"], recent_splits)
    decision = _outs_decision(candidate, expected_outs, recent)
    return {
        "candidate": candidate,
        "pitcherId": pitcher["id"],
        "pitcher": pitcher.get("fullName") or candidate["pitcher"],
        "team": candidate.get("team"),
        "matchup": candidate.get("matchup"),
        "gameTime": game_time,
        "line": candidate.get("line"),
        "overOdds": candidate.get("overOdds"),
        "underOdds": candidate.get("underOdds"),
        "expectedOuts": expected_outs,
        "expectedIP": expected_ip,
        "gapVsLine": decision.get("gap"),
        "bestSide": decision.get("bestSide"),
        "status": decision.get("status"),
        "reason": decision.get("reason"),
        "recent": recent,
        "error": None,
    }


def _outs_line_read_row(profile):
    candidate = profile.get("candidate", {})
    return [
        profile.get("pitcher") or candidate.get("pitcher"),
        candidate.get("team"),
        candidate.get("matchup"),
        profile.get("gameTime") or "N/A",
        _line(candidate.get("line")),
        _decimal(candidate.get("overOdds")),
        _decimal(candidate.get("underOdds")),
        candidate.get("readStatus", "Readable") if not profile.get("error") else profile.get("error"),
    ]


def _outs_side_row(profile):
    recent = profile.get("recent", {})
    last5 = recent.get("last5", {})
    last10 = recent.get("last10", {})
    return [
        profile.get("pitcher"),
        profile.get("team"),
        profile.get("matchup"),
        profile.get("gameTime") or "N/A",
        _line(profile.get("line")),
        _format_number(profile.get("expectedOuts")),
        profile.get("expectedIP") or "N/A",
        _format_number(profile.get("gapVsLine")),
        _decimal(profile.get("overOdds")),
        _decimal(profile.get("underOdds")),
        f"{last5.get('over')} / {last10.get('over')}",
        f"{last5.get('under')} / {last10.get('under')}",
        _format_number(last5.get("averagePitches")),
        _format_number(last5.get("averageBattersFaced")),
        profile.get("bestSide"),
        profile.get("status"),
        profile.get("reason"),
    ]


def _outs_start_log_table(profile):
    line = profile.get("line")
    rows = []
    for index, start in enumerate(profile.get("recent", {}).get("starts", []), start=1):
        rows.append(
            [
                "Yes" if index <= 5 else "No",
                start.get("date"),
                start.get("opponent"),
                start.get("inningsPitched"),
                start.get("outs"),
                start.get("pitches"),
                start.get("battersFaced"),
                start.get("overResult"),
                start.get("underResult"),
            ]
        )
    return _markdown_table(
        [
            "Last 5?",
            "Date",
            "Opp",
            "IP",
            "Outs",
            "Pitches",
            "BF",
            f"Over {_line(line)}?",
            f"Under {_line(line)}?",
        ],
        rows,
    )


def _outs_summary_table(profile):
    recent = profile.get("recent", {})
    return _markdown_table(
        ["Sample", "Avg Outs", "Avg IP", "Over", "Under", "Avg Pitches", "Avg BF"],
        [
            [
                "Last 5",
                _format_number(recent.get("last5", {}).get("averageOuts")),
                recent.get("last5", {}).get("averageIP") or "N/A",
                recent.get("last5", {}).get("over"),
                recent.get("last5", {}).get("under"),
                _format_number(recent.get("last5", {}).get("averagePitches")),
                _format_number(recent.get("last5", {}).get("averageBattersFaced")),
            ],
            [
                "Last 10",
                _format_number(recent.get("last10", {}).get("averageOuts")),
                recent.get("last10", {}).get("averageIP") or "N/A",
                recent.get("last10", {}).get("over"),
                recent.get("last10", {}).get("under"),
                _format_number(recent.get("last10", {}).get("averagePitches")),
                _format_number(recent.get("last10", {}).get("averageBattersFaced")),
            ],
        ],
    )


def _stage2_outs_report_markdown(date, profiles, parse_errors):
    lines = [
        "Stage 2 Pitcher-Outs Research",
        "",
        "Research only. No final bets yet.",
        "",
        "Bet365 Lines Read",
        "",
        _markdown_table(
            ["Pitcher", "Team", "Matchup", "Game Time", "Outs Line", "Over Odds", "Under Odds", "Read Status"],
            [_outs_line_read_row(profile) for profile in profiles],
        ),
    ]
    if parse_errors:
        lines.extend(
            [
                "",
                "Unreadable / Parsing Issues",
                "",
                _markdown_table(
                    ["#", "Raw", "Issue"],
                    [[error.get("index"), error.get("raw"), error.get("error")] for error in parse_errors],
                ),
            ]
        )
    usable = [profile for profile in profiles if not profile.get("error")]
    lines.extend(
        [
            "",
            "Side Comparison Board",
            "",
            _markdown_table(
                [
                    "Pitcher",
                    "Team",
                    "Matchup",
                    "Game Time",
                    "Outs Line",
                    "Exp Outs",
                    "Exp IP",
                    "Gap vs Line",
                    "Over Odds",
                    "Under Odds",
                    "Over L5/L10",
                    "Under L5/L10",
                    "Avg Pitches",
                    "Avg BF",
                    "Best Side",
                    "Status",
                    "Reason",
                ],
                [_outs_side_row(profile) for profile in usable],
            ),
        ]
    )
    for title, statuses in (
        ("CARRY FORWARD", {"Carry Forward"}),
        ("MONITOR", {"Monitor", "Blocked"}),
        ("PASS BOTH SIDES", {"Pass Both Sides"}),
    ):
        group = [profile for profile in usable if profile.get("status") in statuses]
        lines.extend(["", title, ""])
        if not group:
            lines.append("None")
            continue
        lines.append(
            _markdown_table(
                ["Pitcher", "Team", "Matchup", "Game Time", "Outs Line", "Best Side", "Status", "Reason"],
                [
                    [
                        profile.get("pitcher"),
                        profile.get("team"),
                        profile.get("matchup"),
                        profile.get("gameTime") or "N/A",
                        _line(profile.get("line")),
                        profile.get("bestSide"),
                        profile.get("status"),
                        profile.get("reason"),
                    ]
                    for profile in group
                ],
            )
        )
    lines.extend(["", "Actual Last 10 Start Logs"])
    for profile in usable:
        lines.extend(
            [
                "",
                f"{profile.get('pitcher')} - {profile.get('team')} - {profile.get('matchup')} - {profile.get('gameTime') or 'N/A'}",
                f"Line: {_line(profile.get('line'))} outs | Best Side: {profile.get('bestSide')} | Status: {profile.get('status')}",
                "",
                _outs_start_log_table(profile),
                "",
                _outs_summary_table(profile),
            ]
        )
    lines.extend(
        [
            "",
            "What I Need Next",
            "",
            "Tell me your next available recheck time today.",
            "Example: I can recheck at 9:00 PM Atlantic.",
            "",
            "If you cannot recheck later, say: I cannot recheck later today.",
            "",
            "If you want to proceed now, send updated Bet365 screenshots for the Carry Forward / Monitor candidates.",
        ]
    )
    return "\n".join(lines)


def _stage2_outs_compact_profile(profile):
    if profile.get("error"):
        candidate = profile.get("candidate", {})
        return {
            "pitcher": candidate.get("pitcher"),
            "team": candidate.get("team"),
            "matchup": candidate.get("matchup"),
            "gameTime": profile.get("gameTime") or "N/A",
            "error": profile.get("error"),
        }
    return {
        "pitcher": profile.get("pitcher"),
        "team": profile.get("team"),
        "matchup": profile.get("matchup"),
        "gameTime": profile.get("gameTime"),
        "line": profile.get("line"),
        "overOdds": profile.get("overOdds"),
        "underOdds": profile.get("underOdds"),
        "expectedOuts": profile.get("expectedOuts"),
        "expectedIP": profile.get("expectedIP"),
        "gapVsLine": profile.get("gapVsLine"),
        "bestSide": profile.get("bestSide"),
        "status": profile.get("status"),
        "reason": profile.get("reason"),
        "last5": profile.get("recent", {}).get("last5"),
        "last10": profile.get("recent", {}).get("last10"),
    }


def _parse_stage2_candidates(raw):
    candidates = []
    errors = []
    for index, item in enumerate(str(raw or "").split(";"), start=1):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) < 6:
            errors.append(
                {
                    "index": index,
                    "raw": item,
                    "error": "Expected Pitcher|Team|Matchup|Line|OverOdds|UnderOdds.",
                }
            )
            continue
        line = _float(parts[3])
        over_odds = _float(parts[4])
        under_odds = _float(parts[5])
        if line is None or over_odds is None or under_odds is None:
            errors.append(
                {
                    "index": index,
                    "raw": item,
                    "error": "Line, Over odds, and Under odds must be numeric decimal odds.",
                }
            )
            continue
        stage1_group = _normalize_stage1_group(parts[6] if len(parts) > 6 else None)
        original_line = _float(parts[7]) if len(parts) > 7 and parts[7] else line
        original_over_odds = (
            _float(parts[8]) if len(parts) > 8 and parts[8] else over_odds
        )
        original_under_odds = (
            _float(parts[9]) if len(parts) > 9 and parts[9] else under_odds
        )
        candidates.append(
            {
                "pitcher": parts[0],
                "team": _normalize_team_code(parts[1]),
                "matchup": _normalize_matchup(parts[2]),
                "line": line,
                "overOdds": over_odds,
                "underOdds": under_odds,
                "stage1Group": stage1_group,
                "originalLine": original_line,
                "originalOverOdds": original_over_odds,
                "originalUnderOdds": original_under_odds,
                "readStatus": "Readable",
            }
        )
    return candidates, errors


def _normalize_stage1_group(value):
    normalized = str(value or "").strip().upper().replace(" ", "_")
    if normalized in {"RESEARCH", "BORDERLINE", "UNKNOWN"}:
        return normalized
    if normalized in {"PASS", "PASS_FOR_NOW", "PASS-FOR-NOW"}:
        return "UNKNOWN"
    return "UNKNOWN"


def _normalize_matchup(value):
    away, separator, home = str(value or "").strip().upper().partition("@")
    if not separator:
        return str(value or "").strip().upper()
    return f"{_normalize_team_code(away)}@{_normalize_team_code(home)}"


def _game_lookup(date):
    lookup = {}
    for game in _schedule(date):
        away = game.get("teams", {}).get("away", {}).get("team", {})
        home = game.get("teams", {}).get("home", {}).get("team", {})
        away_code = _normalize_team_code(away.get("abbreviation"))
        home_code = _normalize_team_code(home.get("abbreviation"))
        lookup[f"{away_code}@{home_code}"] = game
    return lookup


def _opponent_team_id_for_candidate(candidate, games_by_matchup):
    game = games_by_matchup.get(candidate.get("matchup"))
    if not game:
        return None
    away = game.get("teams", {}).get("away", {}).get("team", {})
    home = game.get("teams", {}).get("home", {}).get("team", {})
    away_code = _normalize_team_code(away.get("abbreviation"))
    home_code = _normalize_team_code(home.get("abbreviation"))
    team = _normalize_team_code(candidate.get("team"))
    if team == away_code:
        return home.get("id")
    if team == home_code:
        return away.get("id")
    return None


def _decimal(value):
    return f"{value:.2f}" if value is not None else "N/A"


def _line(value):
    if value is None:
        return "N/A"
    return f"{value:.1f}".rstrip("0").rstrip(".") if float(value).is_integer() else f"{value:.1f}"


def _odds_in_range(value):
    return value is not None and 1.40 <= value <= 1.95


def _result_vs_line(strikeouts, line, side):
    if strikeouts is None or line is None:
        return "N/A"
    if strikeouts == line:
        return "Push"
    if side == "over":
        return "Yes" if strikeouts > line else "No"
    return "Yes" if strikeouts < line else "No"


def _stage2_start_row(split, person_id, line):
    stat = split.get("stat", {})
    game_pk = _game_pk(split)
    pitch_count = stat.get("numberOfPitches")
    if pitch_count is None:
        pitch_count = _pitch_count_from_boxscore(game_pk, person_id)
    strikeouts = _num(stat.get("strikeOuts"), None)
    return {
        "date": split.get("date"),
        "opponent": split.get("opponent", {}).get("name"),
        "inningsPitched": stat.get("inningsPitched"),
        "pitches": _num(pitch_count, None) if pitch_count is not None else None,
        "battersFaced": _num(stat.get("battersFaced"), None),
        "strikeouts": strikeouts,
        "walks": _num(stat.get("baseOnBalls"), None),
        "overResult": _result_vs_line(strikeouts, line, "over"),
        "underResult": _result_vs_line(strikeouts, line, "under"),
    }


def _hit_summary(starts, line, sample):
    items = starts[:sample]
    if not items:
        return {
            "sample": 0,
            "averageStrikeouts": None,
            "averagePitches": None,
            "averageBattersFaced": None,
            "over": "0/0",
            "under": "0/0",
        }
    over_hits = sum(1 for start in items if start.get("strikeouts") is not None and start["strikeouts"] > line)
    under_hits = sum(1 for start in items if start.get("strikeouts") is not None and start["strikeouts"] < line)
    pushes = sum(1 for start in items if start.get("strikeouts") == line)
    suffix = f", {pushes} push" if pushes == 1 else f", {pushes} pushes" if pushes else ""
    return {
        "sample": len(items),
        "averageStrikeouts": _avg([start.get("strikeouts") for start in items]),
        "averagePitches": _avg([start.get("pitches") for start in items]),
        "averageBattersFaced": _avg([start.get("battersFaced") for start in items]),
        "over": f"{over_hits}/{len(items)}{suffix}",
        "under": f"{under_hits}/{len(items)}{suffix}",
    }


def _stage2_recent_start_splits(person_id, season, limit=10, before_date=None):
    return sorted(
        [
            split
            for split in _pitcher_game_log(person_id, season)
            if _is_start(split)
            and (before_date is None or (split.get("date") or "") < before_date)
        ],
        key=lambda item: item.get("date") or "",
        reverse=True,
    )[:limit]


def _stage2_recent_starts_from_splits(person_id, line, splits):
    starts = [_stage2_start_row(split, person_id, line) for split in splits]
    return {
        "starts": starts,
        "last5": _hit_summary(starts, line, 5),
        "last10": _hit_summary(starts, line, 10),
    }


def _stage2_recent_starts(person_id, season, line, limit=10, before_date=None):
    splits = _stage2_recent_start_splits(
        person_id,
        season,
        limit=limit,
        before_date=before_date,
    )
    return _stage2_recent_starts_from_splits(person_id, line, splits)


def _pitch_name(pitch):
    if not pitch:
        return "N/A"
    code = pitch.get("code")
    description = pitch.get("description")
    if code and description:
        return f"{description} ({code})"
    return description or code or "N/A"


def _stage2_pitch_quality(person_id, start_splits, sample=3):
    starts = []
    for split in start_splits[:sample]:
        game_pk = _game_pk(split)
        try:
            pitch_mix = _pitch_mix_for_game(game_pk, person_id)
        except requests.RequestException:
            pitch_mix = {
                "available": False,
                "source": "MLB Stats API live feed",
                "reason": "Pitch-level feed unavailable for this game.",
            }
        starts.append({"date": split.get("date"), "pitchMix": pitch_mix})

    trend = _pitch_mix_trend(starts)
    pitches = trend.get("pitches", []) if trend.get("available") else []
    if not pitches:
        return {
            "available": False,
            "startsTracked": 0,
            "reason": "Pitch-level trend unavailable from recent starts.",
        }

    main_pitch = pitches[0]
    swing_miss_pitch = max(
        pitches,
        key=lambda pitch: (
            pitch.get("sampleWhiffRate") or 0,
            pitch.get("sampleUsage") or 0,
        ),
    )
    fastball_codes = {"FF", "FA", "FT", "SI", "FC"}
    fastballs = [pitch for pitch in pitches if pitch.get("code") in fastball_codes]
    fastball = max(
        fastballs,
        key=lambda pitch: pitch.get("sampleUsage") or 0,
    ) if fastballs else None

    flags = []
    velocity_change = fastball.get("velocityChangeLatestVsSample") if fastball else None
    if velocity_change is not None:
        if velocity_change <= -1.0:
            flags.append("fastball velocity down")
        elif velocity_change >= 1.0:
            flags.append("fastball velocity up")
        else:
            flags.append("fastball velocity stable")
    if trend.get("startsTracked", 0) < sample:
        flags.append("limited pitch-level sample")

    return {
        "available": True,
        "startsTracked": trend.get("startsTracked"),
        "mainPitch": main_pitch,
        "swingMissPitch": swing_miss_pitch,
        "fastball": fastball,
        "note": "; ".join(flags) if flags else "No clear velocity flag.",
        "compactPitches": pitches[:3],
    }


def _fraction_hits(value):
    text = str(value or "0/0").split(",")[0].strip()
    hits, separator, sample = text.partition("/")
    if not separator:
        return 0, 0, None
    hit_count = _num(hits, 0)
    sample_count = _num(sample, 0)
    rate = round(hit_count / sample_count, 3) if sample_count else None
    return hit_count, sample_count, rate


def _sample_size_grade(recent):
    sample = len(recent.get("starts", []))
    if sample < BORDERLINE_RULES["smallSampleThreshold"]:
        return "Small"
    if sample < BORDERLINE_RULES["limitedSampleThreshold"]:
        return "Limited"
    return "Normal"


def _projection_gap_grade(gap):
    if gap is None:
        return "unavailable"
    if gap < BORDERLINE_RULES["thinGap"]:
        return "thin"
    if gap < BORDERLINE_RULES["meaningfulGap"]:
        return "small"
    if gap < BORDERLINE_RULES["strongGap"]:
        return "meaningful"
    return "strong"


def _workload_confidence_grade(recent):
    starts = len(recent.get("starts", []))
    last5 = recent.get("last5", {})
    avg_pitches = last5.get("averagePitches")
    avg_bf = last5.get("averageBattersFaced")
    if starts >= 8 and avg_pitches is not None and avg_pitches >= 92 and avg_bf is not None and avg_bf >= 23:
        return "High"
    if starts >= 5 and avg_pitches is not None and avg_pitches >= 86 and avg_bf is not None and avg_bf >= 21:
        return "Medium+"
    if starts >= 5:
        return "Medium"
    return "Low"


def _useful_whiff_pitch(pitch):
    usage = pitch.get("sampleUsage")
    whiff = pitch.get("sampleWhiffRate")
    csw = pitch.get("sampleCalledStrikeWhiffRate")
    if usage is not None and usage < 0.08:
        return False
    return (whiff is not None and whiff >= 0.12) or (csw is not None and csw >= 0.28)


def _breaking_pitch(code):
    return code in {"SL", "ST", "SV", "CU", "KC", "CS", "FS"}


def _pitch_quality_grades(pitch_quality, side):
    if not pitch_quality or not pitch_quality.get("available"):
        return {
            "pitchWhiffGrade": "Unavailable",
            "velocityTrendGrade": "Unavailable",
            "signals": [],
            "flags": ["singleSourceCriticalData"],
        }

    pitches = pitch_quality.get("compactPitches") or []
    useful = [pitch for pitch in pitches if _useful_whiff_pitch(pitch)]
    swing_miss_pitch = pitch_quality.get("swingMissPitch") or {}
    fastball = pitch_quality.get("fastball") or {}
    velocity_change = fastball.get("velocityChangeLatestVsSample")
    breaking_usage_up = any(
        _breaking_pitch(pitch.get("code"))
        and pitch.get("usageChangeLatestVsSample") is not None
        and pitch.get("usageChangeLatestVsSample") >= 0.08
        for pitch in pitches
    )

    flags = []
    signals = []
    if len(useful) >= 2:
        whiff_grade = "Strong"
    elif len(useful) == 1:
        whiff_grade = "Adequate"
        if side == "over":
            flags.append("singlePitchDependency")
    else:
        whiff_grade = "Weak"
        flags.append("noReliableWhiffPitch")

    if velocity_change is None:
        velocity_grade = "Unavailable"
    elif velocity_change <= -1.0:
        velocity_grade = "Down"
        if side == "over":
            flags.append("velocityDrop")
        else:
            signals.append("decliningVelocitySupportsUnder")
    elif velocity_change >= 1.0:
        velocity_grade = "Up"
        if side == "under":
            flags.append("recentVelocitySpikeAgainstUnder")
        else:
            signals.append("stableOrImprovingVelocity")
    else:
        velocity_grade = "Stable"
        if side == "over":
            signals.append("stableOrImprovingVelocity")

    if side == "over":
        if whiff_grade in {"Strong", "Adequate"}:
            signals.append("supportivePitchQualityUpdate")
    else:
        if whiff_grade in {"Strong", "Adequate"}:
            flags.append("strongWhiffQualityAgainstUnder")
        if breaking_usage_up:
            flags.append("breakingBallUsageIncreaseAgainstUnder")
        elif whiff_grade == "Weak" or "velocityDrop" in flags or "decliningVelocitySupportsUnder" in signals:
            signals.append("supportivePitchQualityUpdate")

    if pitch_quality.get("startsTracked", 0) < 3:
        flags.append("singleSourceCriticalData")

    return {
        "pitchWhiffGrade": whiff_grade,
        "velocityTrendGrade": velocity_grade,
        "signals": sorted(set(signals)),
        "flags": sorted(set(flags)),
        "usefulWhiffPitchCount": len(useful),
        "bestWhiffPitch": _pitch_name(swing_miss_pitch),
    }


def _movement_grades(candidate, side):
    line = candidate.get("line")
    original_line = candidate.get("originalLine", line)
    line_delta = (
        round(line - original_line, 1)
        if line is not None and original_line is not None
        else None
    )
    current_price = candidate.get("overOdds") if side == "over" else candidate.get("underOdds")
    original_price = (
        candidate.get("originalOverOdds")
        if side == "over"
        else candidate.get("originalUnderOdds")
    )
    if original_price is None:
        original_price = current_price
    price_delta = (
        round(current_price - original_price, 2)
        if current_price is not None and original_price is not None
        else None
    )

    line_grade = "none"
    if line_delta is not None:
        if side == "over" and line_delta <= -BORDERLINE_RULES["fullLineMove"]:
            line_grade = "favorableFullLineMove"
        elif side == "under" and line_delta >= BORDERLINE_RULES["fullLineMove"]:
            line_grade = "favorableFullLineMove"
        elif side == "over" and line_delta <= -BORDERLINE_RULES["halfLineMove"]:
            line_grade = "favorableHalfLineMove"
        elif side == "under" and line_delta >= BORDERLINE_RULES["halfLineMove"]:
            line_grade = "favorableHalfLineMove"
        elif side == "over" and line_delta >= BORDERLINE_RULES["halfLineMove"]:
            line_grade = "adverseLineMovement"
        elif side == "under" and line_delta <= -BORDERLINE_RULES["halfLineMove"]:
            line_grade = "adverseLineMovement"

    price_grade = "none"
    if price_delta is not None:
        if not _odds_in_range(original_price) and _odds_in_range(current_price):
            price_grade = "movedIntoRange"
        elif price_delta >= 0.08:
            price_grade = "betterPrice"
        elif price_delta <= -0.08:
            price_grade = "adversePriceMovement"

    movement = "No material line or price movement."
    if line_grade == "favorableFullLineMove":
        movement = "Favorable full-line movement; re-evaluate as a changed market."
    elif line_grade == "favorableHalfLineMove":
        movement = "Favorable half-line movement; positive but not enough by itself."
    elif line_grade == "adverseLineMovement":
        movement = "Adverse line movement."
    elif price_grade == "movedIntoRange":
        movement = "Desired side price moved into range."
    elif price_grade == "betterPrice":
        movement = "Desired side price improved."
    elif price_grade == "adversePriceMovement":
        movement = "Desired side price shortened; value may be reduced."

    return {
        "originalLine": original_line,
        "currentLine": line,
        "lineDelta": line_delta,
        "originalSidePrice": original_price,
        "currentSidePrice": current_price,
        "priceDelta": price_delta,
        "lineMovementGrade": line_grade,
        "priceMovementGrade": price_grade,
        "movementInterpretation": movement,
    }


def _price_required_gap(price):
    if price is None:
        return BORDERLINE_RULES["strongGap"]
    if price < BORDERLINE_RULES["shortPriceCutoff"]:
        return BORDERLINE_RULES["strongGap"]
    if price < BORDERLINE_RULES["mediumPriceCutoff"]:
        return BORDERLINE_RULES["meaningfulGap"]
    return BORDERLINE_RULES["thinGap"]


def _uncertainty_penalty(sample_grade, pitch_grade, flags):
    penalty = 0.0
    if sample_grade == "Small":
        penalty += BORDERLINE_RULES["smallSamplePenalty"]
    elif sample_grade == "Limited":
        penalty += BORDERLINE_RULES["limitedSamplePenalty"]
    if pitch_grade == "Unavailable":
        penalty += BORDERLINE_RULES["pitchQualityUnavailablePenalty"]
    elif pitch_grade == "Weak":
        penalty += BORDERLINE_RULES["weakPitchQualityPenalty"]
    if "projectedLineupOnly" in flags:
        penalty += BORDERLINE_RULES["projectedLineupPenalty"]
    if "staleData" in flags:
        penalty += BORDERLINE_RULES["staleDataPenalty"]
    if "shortPriceThinEdge" in flags:
        penalty += BORDERLINE_RULES["thinPricePenalty"]
    return round(min(penalty, 0.35), 3)


def _side_stage2_evaluation(candidate, expected_ks, recent, pitch_quality, side):
    line = candidate.get("line")
    price = candidate.get("overOdds") if side == "over" else candidate.get("underOdds")
    projection_gap = None
    if expected_ks is not None and line is not None:
        projection_gap = round(expected_ks - line, 1) if side == "over" else round(line - expected_ks, 1)
    projection_grade = _projection_gap_grade(projection_gap)
    sample_grade = _sample_size_grade(recent)
    sample = len(recent.get("starts", []))
    last10 = recent.get("last10", {})
    hit_text = last10.get("over") if side == "over" else last10.get("under")
    _, _, hit_rate = _fraction_hits(hit_text)
    pitch_grades = _pitch_quality_grades(pitch_quality, side)
    movement = _movement_grades(candidate, side)
    workload_confidence = _workload_confidence_grade(recent)

    flags = set(pitch_grades.get("flags") or [])
    signals = set(pitch_grades.get("signals") or [])
    critical_blockers = set()

    if sample == 0:
        flags.add("unverifiedRecentStarts")
        critical_blockers.add("unverifiedRecentStarts")
    elif sample < BORDERLINE_RULES["smallSampleThreshold"]:
        flags.add("fewerThanFiveRecentStarts")
    elif sample < BORDERLINE_RULES["minimumRecentStartsForNormalConfidence"]:
        flags.add("incompleteLastTen")

    if not _odds_in_range(price):
        flags.add("priceOutsideRange")
        critical_blockers.add("priceOutsideRange")
    if movement["lineMovementGrade"] == "adverseLineMovement":
        flags.add("adverseLineMovement")
    if movement["priceMovementGrade"] == "adversePriceMovement":
        flags.add("adversePriceMovement")

    if price is not None and price < BORDERLINE_RULES["shortPriceCutoff"]:
        if projection_gap is None or projection_gap < BORDERLINE_RULES["strongGap"]:
            flags.add("shortPriceThinEdge")
        if workload_confidence != "High":
            flags.add("workloadUncertainty")
        if side == "over" and pitch_grades["pitchWhiffGrade"] != "Strong":
            flags.add("noReliableWhiffPitch")

    if side == "over":
        if workload_confidence in {"High", "Medium+"}:
            signals.add("strongerWorkloadConfirmation")
        elif workload_confidence == "Low":
            flags.add("workloadUncertainty")
    else:
        if workload_confidence == "Low":
            signals.add("workloadUncertaintySupportsUnder")
        elif workload_confidence == "High":
            flags.add("strongOutsProjectionAgainstUnder")

    if movement["lineMovementGrade"] == "favorableFullLineMove":
        signals.add("favorableFullLineMovement")
    elif movement["lineMovementGrade"] == "favorableHalfLineMove":
        signals.add("favorableHalfLineMovement")
    if movement["priceMovementGrade"] == "movedIntoRange":
        signals.add("favorablePriceImprovement")
    elif movement["priceMovementGrade"] == "betterPrice":
        signals.add("betterPrice")

    if projection_gap is not None and projection_gap >= _price_required_gap(price):
        signals.add("meaningfulProjectionEdge")

    implied_probability = (round(1 / price, 3) if price else None)
    raw_estimated = None
    if projection_gap is not None:
        hit_component = (hit_rate - 0.5) * 0.16 if hit_rate is not None else 0
        if side == "over":
            workload_component = 0.03 if workload_confidence == "High" else 0.015 if workload_confidence == "Medium+" else -0.02 if workload_confidence == "Low" else 0
            pitch_component = 0.03 if pitch_grades["pitchWhiffGrade"] == "Strong" else 0.015 if pitch_grades["pitchWhiffGrade"] == "Adequate" else -0.03 if pitch_grades["pitchWhiffGrade"] == "Weak" else -0.015
        else:
            workload_component = 0.025 if workload_confidence == "Low" else -0.025 if workload_confidence == "High" else 0
            pitch_component = 0.03 if pitch_grades["pitchWhiffGrade"] == "Weak" else -0.03 if pitch_grades["pitchWhiffGrade"] == "Strong" else -0.015 if pitch_grades["pitchWhiffGrade"] == "Adequate" else 0
        raw_estimated = round(
            _clamp(0.50 + projection_gap * 0.075 + hit_component + workload_component + pitch_component, 0.20, 0.82),
            3,
        )
    penalty = _uncertainty_penalty(sample_grade, pitch_grades["pitchWhiffGrade"], flags)
    adjusted = (
        round(_clamp(raw_estimated - penalty, 0.05, 0.90), 3)
        if raw_estimated is not None
        else None
    )
    edge_points = (
        round((adjusted - implied_probability) * 100, 1)
        if adjusted is not None and implied_probability is not None
        else None
    )
    adjusted_true_odds = round(1 / adjusted, 2) if adjusted else None
    adjusted_edge_qualifies = (
        edge_points is not None
        and edge_points >= BORDERLINE_RULES["minimumEdgePercentagePoints"]
    )
    gap_qualifies = projection_gap is not None and projection_gap >= _price_required_gap(price)

    if sample_grade == "Small" and len(signals - {"meaningfulProjectionEdge"}) < 2:
        flags.add("smallSampleNeedsIndependentSignals")

    return {
        "side": side,
        "stage1Group": candidate.get("stage1Group", "UNKNOWN"),
        "recentStartSample": sample,
        "sampleSizeGrade": sample_grade,
        "projectionGap": projection_gap,
        "projectionGapGrade": projection_grade,
        "pitchWhiffGrade": pitch_grades["pitchWhiffGrade"],
        "velocityTrendGrade": pitch_grades["velocityTrendGrade"],
        "workloadConfidence": workload_confidence,
        "lineupImpactGrade": "Unavailable",
        "uncertaintyPenalty": penalty,
        "borderlinePromotionSignals": sorted(signals),
        "borderlineDowngradeFlags": sorted(flags),
        "criticalBlockers": sorted(critical_blockers),
        "impliedProbability": implied_probability,
        "rawEstimatedTrueProbability": raw_estimated,
        "uncertaintyHaircut": penalty,
        "adjustedEstimatedTrueProbability": adjusted,
        "edgePercentagePoints": edge_points,
        "adjustedTrueOdds": adjusted_true_odds,
        "adjustedEdgeQualifies": adjusted_edge_qualifies,
        "gapQualifies": gap_qualifies,
        **movement,
    }


def _promotion_reason(evaluation, decision):
    blockers = evaluation.get("criticalBlockers") or []
    flags = evaluation.get("borderlineDowngradeFlags") or []
    signals = evaluation.get("borderlinePromotionSignals") or []
    if decision == "Promoted":
        return "Promotion gate passed: adjusted edge qualifies with independent support from " + ", ".join(signals[:3]) + "."
    if blockers:
        return "Critical blocker: " + ", ".join(blockers[:3]) + "."
    if decision == "Monitor":
        return "Directional interest remains, but promotion gate is incomplete."
    if flags:
        return "Rejected by downgrade flags: " + ", ".join(flags[:3]) + "."
    return "Rejected because adjusted edge or independent support is not strong enough."


def _borderline_promotion_audit(candidate, expected_ks, recent, pitch_quality, existing_decision):
    side_evaluations = {
        "over": _side_stage2_evaluation(candidate, expected_ks, recent, pitch_quality, "over"),
        "under": _side_stage2_evaluation(candidate, expected_ks, recent, pitch_quality, "under"),
    }
    ranked = sorted(
        side_evaluations.values(),
        key=lambda item: (
            item.get("edgePercentagePoints") if item.get("edgePercentagePoints") is not None else -999,
            item.get("projectionGap") if item.get("projectionGap") is not None else -999,
        ),
        reverse=True,
    )
    best = ranked[0]
    stage1_group = candidate.get("stage1Group", "UNKNOWN")
    material_signals = [
        signal
        for signal in best.get("borderlinePromotionSignals", [])
        if signal not in {"favorableHalfLineMovement", "betterPrice"}
    ]
    blockers = best.get("criticalBlockers") or []
    flags = best.get("borderlineDowngradeFlags") or []

    if stage1_group != "BORDERLINE":
        decision = "Not Applicable"
        risk_context = best.get("criticalBlockers") or best.get("borderlineDowngradeFlags") or []
        if risk_context:
            reason = (
                "Primary RESEARCH/UNKNOWN candidate; promotion gate not applicable. "
                "Risk context: " + ", ".join(risk_context[:3]) + "."
            )
        else:
            reason = "Primary RESEARCH/UNKNOWN candidate; promotion gate not applicable."
    elif blockers:
        decision = "Rejected"
        reason = _promotion_reason(best, decision)
    elif (
        best.get("adjustedEdgeQualifies")
        and best.get("gapQualifies")
        and len(material_signals) >= 1
        and len(best.get("borderlinePromotionSignals", [])) >= BORDERLINE_RULES["minimumIndependentSignals"]
        and "smallSampleNeedsIndependentSignals" not in flags
    ):
        decision = "Promoted"
        reason = _promotion_reason(best, decision)
    elif best.get("projectionGap") is not None and best.get("projectionGap") >= BORDERLINE_RULES["thinGap"] and not blockers:
        decision = "Monitor"
        reason = _promotion_reason(best, decision)
    else:
        decision = "Rejected"
        reason = _promotion_reason(best, decision)
    return {
        "stage1Group": stage1_group,
        "bestSide": "Over Candidate" if best.get("side") == "over" else "Under Candidate",
        "sideEvaluations": side_evaluations,
        "recentStartSample": best.get("recentStartSample"),
        "sampleSizeGrade": best.get("sampleSizeGrade"),
        "projectionGap": best.get("projectionGap"),
        "projectionGapGrade": best.get("projectionGapGrade"),
        "pitchWhiffGrade": best.get("pitchWhiffGrade"),
        "velocityTrendGrade": best.get("velocityTrendGrade"),
        "workloadConfidence": best.get("workloadConfidence"),
        "lineMovementGrade": best.get("lineMovementGrade"),
        "priceMovementGrade": best.get("priceMovementGrade"),
        "lineupImpactGrade": best.get("lineupImpactGrade"),
        "uncertaintyPenalty": best.get("uncertaintyPenalty"),
        "borderlinePromotionSignals": best.get("borderlinePromotionSignals"),
        "borderlineDowngradeFlags": best.get("borderlineDowngradeFlags"),
        "criticalBlockers": best.get("criticalBlockers"),
        "borderlinePromotionDecision": decision,
        "borderlinePromotionReason": reason,
        "impliedProbability": best.get("impliedProbability"),
        "rawEstimatedTrueProbability": best.get("rawEstimatedTrueProbability"),
        "uncertaintyHaircut": best.get("uncertaintyHaircut"),
        "adjustedEstimatedTrueProbability": best.get("adjustedEstimatedTrueProbability"),
        "edgePercentagePoints": best.get("edgePercentagePoints"),
        "adjustedTrueOdds": best.get("adjustedTrueOdds"),
        "originalLine": best.get("originalLine"),
        "currentLine": best.get("currentLine"),
        "lineDelta": best.get("lineDelta"),
        "originalSidePrice": best.get("originalSidePrice"),
        "currentSidePrice": best.get("currentSidePrice"),
        "priceDelta": best.get("priceDelta"),
        "movementInterpretation": best.get("movementInterpretation"),
    }


def _stage2_decision_with_borderline_gate(candidate, decision, promotion_audit):
    if candidate.get("stage1Group") != "BORDERLINE":
        return decision
    promotion_decision = promotion_audit.get("borderlinePromotionDecision")
    if promotion_decision == "Promoted":
        return {
            "bestSide": promotion_audit.get("bestSide"),
            "status": "Carry Forward",
            "reason": promotion_audit.get("borderlinePromotionReason"),
            "gap": promotion_audit.get("projectionGap"),
        }
    if promotion_decision == "Monitor":
        return {
            "bestSide": promotion_audit.get("bestSide"),
            "status": "Monitor",
            "reason": promotion_audit.get("borderlinePromotionReason"),
            "gap": promotion_audit.get("projectionGap"),
        }
    return {
        "bestSide": promotion_audit.get("bestSide"),
        "status": "Pass Both Sides",
        "reason": promotion_audit.get("borderlinePromotionReason"),
        "gap": promotion_audit.get("projectionGap"),
    }


def _stage2_decision(candidate, expected_ks, summary):
    line = candidate.get("line")
    over_odds = candidate.get("overOdds")
    under_odds = candidate.get("underOdds")
    gap = round(expected_ks - line, 1) if expected_ks is not None and line is not None else None
    last5 = summary.get("last5", {})
    last10 = summary.get("last10", {})
    avg_pitches = last5.get("averagePitches")
    avg_bf = last5.get("averageBattersFaced")
    over_l10_hits = _num(str(last10.get("over", "0/0")).split("/")[0], 0)
    under_l10_hits = _num(str(last10.get("under", "0/0")).split("/")[0], 0)
    over_in_range = _odds_in_range(over_odds)
    under_in_range = _odds_in_range(under_odds)

    if gap is None:
        return {
            "bestSide": "Monitor Both Sides",
            "status": "Blocked",
            "reason": "Expected Ks unavailable, so side cannot be judged cleanly.",
            "gap": gap,
        }

    workload_ok = (
        (avg_pitches is None or avg_pitches >= 84)
        and (avg_bf is None or avg_bf >= 21)
    )
    if gap >= 0.60 and over_in_range and workload_ok and over_l10_hits >= 5:
        return {
            "bestSide": "Over Candidate",
            "status": "Carry Forward",
            "reason": "Expected Ks are meaningfully above the line, Over price is in range, and recent hit rate/workload support the Over.",
            "gap": gap,
        }
    if gap <= -0.60 and under_in_range and under_l10_hits >= 5:
        return {
            "bestSide": "Under Candidate",
            "status": "Carry Forward",
            "reason": "Expected Ks are meaningfully below the line, Under price is in range, and recent hit rate supports the Under.",
            "gap": gap,
        }
    if not over_in_range and not under_in_range:
        return {
            "bestSide": "Pass Both Sides",
            "status": "Pass Both Sides",
            "reason": "Both prices are outside the 1.40-1.95 range.",
            "gap": gap,
        }
    if abs(gap) < 0.60:
        return {
            "bestSide": "Monitor Both Sides",
            "status": "Monitor",
            "reason": "Expected Ks are close to the line, so price, lineup, and recheck context matter.",
            "gap": gap,
        }
    return {
        "bestSide": "Monitor Both Sides",
        "status": "Monitor",
        "reason": "There is a directional lean, but price, recent hit rate, or workload is not clean enough yet.",
        "gap": gap,
    }


def _stage2_candidate_profile(candidate, season, games_by_matchup, date, include_pitch_quality=True):
    game = games_by_matchup.get(candidate.get("matchup"))
    game_time = _format_game_time(game.get("gameDate")) if game else "N/A"
    pitcher = _find_pitcher(candidate["pitcher"])
    if not pitcher:
        return {
            "candidate": candidate,
            "gameTime": game_time,
            "error": f"No pitcher found for {candidate['pitcher']}.",
        }
    opponent_team_id = _opponent_team_id_for_candidate(candidate, games_by_matchup)
    expected_ks = None
    expected_bf = None
    estimated_k_rate = None
    if opponent_team_id:
        profile = _pitcher_screening_profile(
            pitcher["id"],
            opponent_team_id,
            season,
            before_date=date,
        )
        expected_model = profile.get("expectedKModel", {})
        expected_ks = expected_model.get("expectedStrikeouts")
        expected_bf = expected_model.get("expectedBattersFaced")
        estimated_k_rate = expected_model.get("estimatedKRate")

    recent_splits = _stage2_recent_start_splits(
        pitcher["id"],
        season,
        limit=10,
        before_date=date,
    )
    recent = _stage2_recent_starts_from_splits(
        pitcher["id"],
        candidate["line"],
        recent_splits,
    )
    decision = _stage2_decision(candidate, expected_ks, recent)
    pitch_quality = (
        _stage2_pitch_quality(pitcher["id"], recent_splits)
        if include_pitch_quality
        else None
    )
    promotion_audit = _borderline_promotion_audit(
        candidate,
        expected_ks,
        recent,
        pitch_quality,
        decision,
    )
    decision = _stage2_decision_with_borderline_gate(
        candidate,
        decision,
        promotion_audit,
    )
    return {
        "candidate": candidate,
        "pitcherId": pitcher["id"],
        "pitcher": pitcher.get("fullName") or candidate["pitcher"],
        "team": candidate.get("team"),
        "matchup": candidate.get("matchup"),
        "gameTime": game_time,
        "line": candidate.get("line"),
        "overOdds": candidate.get("overOdds"),
        "underOdds": candidate.get("underOdds"),
        "stage1Group": candidate.get("stage1Group", "UNKNOWN"),
        "originalLine": candidate.get("originalLine"),
        "originalOverOdds": candidate.get("originalOverOdds"),
        "originalUnderOdds": candidate.get("originalUnderOdds"),
        "expectedStrikeouts": expected_ks,
        "expectedBattersFaced": expected_bf,
        "estimatedKRate": estimated_k_rate,
        "gapVsLine": decision.get("gap"),
        "bestSide": decision.get("bestSide"),
        "status": decision.get("status"),
        "reason": decision.get("reason"),
        "recent": recent,
        "pitchQuality": pitch_quality,
        "borderlinePromotionAudit": promotion_audit,
        "error": None,
    }


def _stage2_line_read_row(profile):
    candidate = profile.get("candidate", {})
    return [
        profile.get("pitcher") or candidate.get("pitcher"),
        candidate.get("team"),
        candidate.get("matchup"),
        profile.get("gameTime") or "N/A",
        _line(candidate.get("line")),
        _decimal(candidate.get("overOdds")),
        _decimal(candidate.get("underOdds")),
        candidate.get("readStatus", "Readable") if not profile.get("error") else profile.get("error"),
    ]


def _stage2_side_row(profile):
    recent = profile.get("recent", {})
    last5 = recent.get("last5", {})
    last10 = recent.get("last10", {})
    return [
        profile.get("pitcher"),
        profile.get("team"),
        profile.get("matchup"),
        profile.get("gameTime") or "N/A",
        _line(profile.get("line")),
        _format_number(profile.get("expectedStrikeouts")),
        _format_number(profile.get("gapVsLine")),
        _decimal(profile.get("overOdds")),
        _decimal(profile.get("underOdds")),
        f"{last5.get('over')} / {last10.get('over')}",
        f"{last5.get('under')} / {last10.get('under')}",
        profile.get("bestSide"),
        profile.get("status"),
        profile.get("reason"),
    ]


def _stage2_pitch_quality_row(profile):
    quality = profile.get("pitchQuality") or {}
    if not quality.get("available"):
        return [
            profile.get("pitcher"),
            profile.get("team"),
            profile.get("matchup"),
            profile.get("gameTime") or "N/A",
            "Unavailable",
            "Unavailable",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            quality.get("reason") or "Pitch-level trend unavailable.",
        ]

    whiff_pitch = quality.get("swingMissPitch") or {}
    fastball = quality.get("fastball") or {}
    return [
        profile.get("pitcher"),
        profile.get("team"),
        profile.get("matchup"),
        profile.get("gameTime") or "N/A",
        _pitch_name(quality.get("mainPitch")),
        _pitch_name(whiff_pitch),
        _format_rate(whiff_pitch.get("sampleWhiffRate")),
        _format_rate(whiff_pitch.get("sampleCalledStrikeWhiffRate")),
        _format_number(fastball.get("latestAverageVelocity")),
        _format_number(fastball.get("velocityChangeLatestVsSample")),
        quality.get("note"),
    ]


def _stage2_borderline_audit_row(profile):
    audit = profile.get("borderlinePromotionAudit") or {}
    signals = audit.get("borderlinePromotionSignals") or []
    flags = audit.get("borderlineDowngradeFlags") or []
    edge = audit.get("edgePercentagePoints")
    return [
        profile.get("pitcher"),
        audit.get("stage1Group") or profile.get("stage1Group") or "UNKNOWN",
        audit.get("bestSide") or profile.get("bestSide"),
        ", ".join(signals) if signals else "None",
        ", ".join(flags) if flags else "None",
        f"{edge:.1f} pp" if edge is not None else "N/A",
        audit.get("borderlinePromotionDecision") or "Not Applicable",
        audit.get("borderlinePromotionReason") or "N/A",
    ]


def _stage2_compact_profile(profile):
    if profile.get("error"):
        candidate = profile.get("candidate", {})
        return {
            "pitcher": candidate.get("pitcher"),
            "team": candidate.get("team"),
            "matchup": candidate.get("matchup"),
            "gameTime": profile.get("gameTime") or "N/A",
            "error": profile.get("error"),
        }

    quality = profile.get("pitchQuality") or {}
    whiff_pitch = quality.get("swingMissPitch") or {}
    fastball = quality.get("fastball") or {}
    audit = profile.get("borderlinePromotionAudit") or {}
    return {
        "pitcher": profile.get("pitcher"),
        "team": profile.get("team"),
        "matchup": profile.get("matchup"),
        "gameTime": profile.get("gameTime"),
        "line": profile.get("line"),
        "overOdds": profile.get("overOdds"),
        "underOdds": profile.get("underOdds"),
        "stage1Group": profile.get("stage1Group"),
        "recentStartSample": audit.get("recentStartSample"),
        "sampleSizeGrade": audit.get("sampleSizeGrade"),
        "projectionGap": audit.get("projectionGap"),
        "projectionGapGrade": audit.get("projectionGapGrade"),
        "pitchWhiffGrade": audit.get("pitchWhiffGrade"),
        "velocityTrendGrade": audit.get("velocityTrendGrade"),
        "workloadConfidence": audit.get("workloadConfidence"),
        "lineMovementGrade": audit.get("lineMovementGrade"),
        "priceMovementGrade": audit.get("priceMovementGrade"),
        "lineupImpactGrade": audit.get("lineupImpactGrade"),
        "uncertaintyPenalty": audit.get("uncertaintyPenalty"),
        "borderlinePromotionSignals": audit.get("borderlinePromotionSignals"),
        "borderlineDowngradeFlags": audit.get("borderlineDowngradeFlags"),
        "borderlinePromotionDecision": audit.get("borderlinePromotionDecision"),
        "borderlinePromotionReason": audit.get("borderlinePromotionReason"),
        "priceMath": {
            "impliedProbability": audit.get("impliedProbability"),
            "rawEstimatedTrueProbability": audit.get("rawEstimatedTrueProbability"),
            "uncertaintyHaircut": audit.get("uncertaintyHaircut"),
            "adjustedEstimatedTrueProbability": audit.get("adjustedEstimatedTrueProbability"),
            "edgePercentagePoints": audit.get("edgePercentagePoints"),
            "adjustedTrueOdds": audit.get("adjustedTrueOdds"),
        },
        "movement": {
            "originalLine": audit.get("originalLine"),
            "currentLine": audit.get("currentLine"),
            "lineDelta": audit.get("lineDelta"),
            "originalSidePrice": audit.get("originalSidePrice"),
            "currentSidePrice": audit.get("currentSidePrice"),
            "priceDelta": audit.get("priceDelta"),
            "movementInterpretation": audit.get("movementInterpretation"),
        },
        "expectedStrikeouts": profile.get("expectedStrikeouts"),
        "gapVsLine": profile.get("gapVsLine"),
        "bestSide": profile.get("bestSide"),
        "status": profile.get("status"),
        "reason": profile.get("reason"),
        "last5": profile.get("recent", {}).get("last5"),
        "last10": profile.get("recent", {}).get("last10"),
        "pitchQuality": {
            "available": quality.get("available", False),
            "mainPitch": _pitch_name(quality.get("mainPitch")) if quality.get("available") else None,
            "bestWhiffPitch": _pitch_name(whiff_pitch) if quality.get("available") else None,
            "whiffRate": whiff_pitch.get("sampleWhiffRate"),
            "calledStrikeWhiffRate": whiff_pitch.get("sampleCalledStrikeWhiffRate"),
            "fastballVelocity": fastball.get("latestAverageVelocity"),
            "fastballVelocityChange": fastball.get("velocityChangeLatestVsSample"),
            "note": quality.get("note") or quality.get("reason"),
        },
    }


def _stage2_start_log_table(profile):
    line = profile.get("line")
    rows = []
    for index, start in enumerate(profile.get("recent", {}).get("starts", []), start=1):
        rows.append(
            [
                "Yes" if index <= 5 else "No",
                start.get("date"),
                start.get("opponent"),
                start.get("inningsPitched"),
                start.get("pitches"),
                start.get("battersFaced"),
                start.get("strikeouts"),
                start.get("overResult"),
                start.get("underResult"),
            ]
        )
    return _markdown_table(
        [
            "Last 5?",
            "Date",
            "Opp",
            "IP",
            "Pitches",
            "BF",
            "Ks",
            f"Over {_line(line)}?",
            f"Under {_line(line)}?",
        ],
        rows,
    )


def _stage2_summary_table(profile):
    recent = profile.get("recent", {})
    return _markdown_table(
        ["Sample", "Avg Ks", "Over", "Under", "Avg Pitches", "Avg BF"],
        [
            [
                "Last 5",
                _format_number(recent.get("last5", {}).get("averageStrikeouts")),
                recent.get("last5", {}).get("over"),
                recent.get("last5", {}).get("under"),
                _format_number(recent.get("last5", {}).get("averagePitches")),
                _format_number(recent.get("last5", {}).get("averageBattersFaced")),
            ],
            [
                "Last 10",
                _format_number(recent.get("last10", {}).get("averageStrikeouts")),
                recent.get("last10", {}).get("over"),
                recent.get("last10", {}).get("under"),
                _format_number(recent.get("last10", {}).get("averagePitches")),
                _format_number(recent.get("last10", {}).get("averageBattersFaced")),
            ],
        ],
    )


def _stage2_snapshot_table(profiles):
    usable = [profile for profile in profiles if not profile.get("error")]
    status_counts = {
        "Carry Forward": sum(1 for profile in usable if profile.get("status") == "Carry Forward"),
        "Monitor": sum(1 for profile in usable if profile.get("status") in {"Monitor", "Blocked"}),
        "Pass Both Sides": sum(1 for profile in usable if profile.get("status") == "Pass Both Sides"),
    }
    borderline_decisions = {}
    for profile in usable:
        audit = profile.get("borderlinePromotionAudit") or {}
        decision = audit.get("borderlinePromotionDecision")
        if audit.get("stage1Group") == "BORDERLINE" and decision:
            borderline_decisions[decision] = borderline_decisions.get(decision, 0) + 1
    borderline_summary = (
        ", ".join(f"{key}: {value}" for key, value in sorted(borderline_decisions.items()))
        if borderline_decisions
        else "None supplied"
    )
    return _markdown_table(
        ["Item", "Count / Status"],
        [
            ["Candidates reviewed", len(usable)],
            ["Carry Forward", status_counts["Carry Forward"]],
            ["Monitor", status_counts["Monitor"]],
            ["Pass Both Sides", status_counts["Pass Both Sides"]],
            ["Borderline audit", borderline_summary],
        ],
    )


def _stage2_recheck_focus(profile):
    status = profile.get("status")
    audit = profile.get("borderlinePromotionAudit") or {}
    flags = audit.get("borderlineDowngradeFlags") or []
    if status == "Carry Forward":
        return "Confirm starter, lineup, weather, and price."
    if status == "Pass Both Sides":
        return "Only revisit after a material line or price change."
    if "priceOutsideRange" in flags:
        return "Needs price back inside 1.40-1.95."
    if audit.get("stage1Group") == "BORDERLINE":
        return "Needs promotion signal or cleaner risk profile."
    return "Needs lineup, price, or workload confirmation."


def _stage2_priority_recheck_table(profiles):
    usable = [profile for profile in profiles if not profile.get("error")]
    ordered = sorted(
        usable,
        key=lambda profile: {
            "Carry Forward": 0,
            "Monitor": 1,
            "Blocked": 1,
            "Pass Both Sides": 2,
        }.get(profile.get("status"), 3),
    )
    return _markdown_table(
        ["Pitcher", "Side", "Price", "Status", "Stage 1", "Adjusted Edge", "Recheck Focus"],
        [
            [
                profile.get("pitcher"),
                profile.get("bestSide"),
                (
                    _decimal(profile.get("overOdds"))
                    if profile.get("bestSide") == "Over Candidate"
                    else _decimal(profile.get("underOdds"))
                    if profile.get("bestSide") == "Under Candidate"
                    else "N/A"
                ),
                profile.get("status"),
                (profile.get("borderlinePromotionAudit") or {}).get("stage1Group") or profile.get("stage1Group") or "UNKNOWN",
                (
                    f"{(profile.get('borderlinePromotionAudit') or {}).get('edgePercentagePoints'):.1f} pp"
                    if (profile.get("borderlinePromotionAudit") or {}).get("edgePercentagePoints") is not None
                    else "N/A"
                ),
                _stage2_recheck_focus(profile),
            ]
            for profile in ordered
        ],
    )


def _stage2_report_markdown(date, profiles, parse_errors):
    usable = [profile for profile in profiles if not profile.get("error")]
    lines = [
        "Stage 2 Pitcher-K Research",
        "",
        "Research only. No final bets yet.",
        "",
        "At-a-Glance",
        "",
        _stage2_snapshot_table(profiles),
        "",
        "Priority Recheck List",
        "",
        _stage2_priority_recheck_table(usable),
        "",
        "Bet365 Lines Read",
        "",
        _markdown_table(
            ["Pitcher", "Team", "Matchup", "Game Time", "Line", "Over Odds", "Under Odds", "Read Status"],
            [_stage2_line_read_row(profile) for profile in profiles],
        ),
    ]
    if parse_errors:
        lines.extend(
            [
                "",
                "Unreadable / Parsing Issues",
                "",
                _markdown_table(
                    ["#", "Raw", "Issue"],
                    [
                        [error.get("index"), error.get("raw"), error.get("error")]
                        for error in parse_errors
                    ],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Side Comparison Board",
            "",
            _markdown_table(
                [
                    "Pitcher",
                    "Team",
                    "Matchup",
                    "Game Time",
                    "Line",
                    "Exp Ks",
                    "Gap vs Line",
                    "Over Odds",
                    "Under Odds",
                    "Over L5/L10",
                    "Under L5/L10",
                    "Best Side",
                    "Status",
                    "Reason",
                ],
                [_stage2_side_row(profile) for profile in usable],
            ),
            "",
            "Borderline Promotion Audit",
            "",
            _markdown_table(
                [
                    "Pitcher",
                    "Stage 1 Group",
                    "Best Side",
                    "Positive Signals",
                    "Downgrade Flags",
                    "Adjusted Edge",
                    "Promotion Decision",
                    "Reason",
                ],
                [_stage2_borderline_audit_row(profile) for profile in usable],
            ),
            "",
            "Pitch Mix / Velocity Check",
            "",
            _markdown_table(
                [
                    "Pitcher",
                    "Team",
                    "Matchup",
                    "Game Time",
                    "Main Pitch",
                    "Best Whiff Pitch",
                    "Whiff%",
                    "CSW%",
                    "FB Velo",
                    "FB Velo +/-",
                    "Note",
                ],
                [_stage2_pitch_quality_row(profile) for profile in usable],
            ),
        ]
    )

    for title, statuses in (
        ("CARRY FORWARD", {"Carry Forward"}),
        ("MONITOR", {"Monitor", "Blocked"}),
        ("PASS BOTH SIDES", {"Pass Both Sides"}),
    ):
        group = [profile for profile in usable if profile.get("status") in statuses]
        lines.extend(["", title, ""])
        if not group:
            lines.append("None")
            continue
        lines.append(
            _markdown_table(
                ["Pitcher", "Team", "Matchup", "Game Time", "Line", "Best Side", "Status", "Reason"],
                [
                    [
                        profile.get("pitcher"),
                        profile.get("team"),
                        profile.get("matchup"),
                        profile.get("gameTime") or "N/A",
                        _line(profile.get("line")),
                        profile.get("bestSide"),
                        profile.get("status"),
                        profile.get("reason"),
                    ]
                    for profile in group
                ],
            )
        )

    lines.extend(["", "Actual Last 10 Start Logs"])
    for profile in usable:
        lines.extend(
            [
                "",
                f"{profile.get('pitcher')} - {profile.get('team')} - {profile.get('matchup')} - {profile.get('gameTime') or 'N/A'}",
                f"Line: {_line(profile.get('line'))} Ks | Best Side: {profile.get('bestSide')} | Status: {profile.get('status')}",
                "",
                _stage2_start_log_table(profile),
                "",
                _stage2_summary_table(profile),
            ]
        )

    lines.extend(
        [
            "",
            "What I Need Next",
            "",
            "Tell me your next available recheck time today.",
            "Example: I can recheck at 9:00 PM Atlantic.",
            "",
            "If you cannot recheck later, say: I cannot recheck later today.",
            "",
            "If you want to proceed now, send updated Bet365 screenshots for the Carry Forward / Monitor candidates.",
        ]
    )
    return "\n".join(lines)


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
        "gameTime": game.get("gameTime"),
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

    requested_matchups = [
        f"{away}@{home}" for away, home in requested
    ]
    slim_payload = {
        "date": date,
        "season": season,
        "screeningOnly": True,
        "responseMode": "markdown",
        "stage1ReportMarkdown": stage1_dashboard.get("stage1ReportMarkdown"),
        "requestedMatchups": requested_matchups,
        "unmatchedRequestedMatchups": unmatched,
        "gamesReturned": len(screened),
        "pitchersScreened": sum(
            1 for game in compact_games for pitcher in game.get("pitchers", []) if pitcher.get("available")
        ),
        "stage2ScreenshotTargets": stage1_dashboard.get("stage2ScreenshotTargets", []),
        "recommendedPitchers": compact_recommended,
        "notes": [
            "Default response is intentionally slim so ChatGPT actions can handle full screenshot slates reliably.",
            "Display stage1ReportMarkdown exactly for the user-facing Stage 1 report.",
            "Use include_details=true only when debugging or when raw dashboard data is required.",
            "This endpoint screens games for deeper pitcher-K research. It does not recommend bets.",
        ],
    }

    payload = slim_payload
    if include_details:
        payload = {
            **slim_payload,
            "responseMode": "detailed",
            "stage1Dashboard": stage1_dashboard,
            "games": compact_games,
            "notSelectedPitchers": compact_not_selected,
            "screeningGroups": compact_groups,
            "rankGapNotes": rank_gap_notes,
            "debugFullGames": screened,
            "debugFullScreeningGroups": screening_groups,
            "debugFullRecommendedPitchers": recommended_pitchers,
            "debugFullNotSelectedPitchers": not_selected_pitchers,
        }

    return jsonify(payload)


@app.get("/out-screening")
def out_screening():
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
            executor.submit(_format_out_screening_game, game, season)
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
    ) = _rank_out_screening_pitchers(screened)
    compact_groups = _compact_out_groups(screening_groups)
    compact_games = [_compact_out_game(game) for game in screened]
    outs_stage1_dashboard = _outs_stage1_dashboard(
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
        "outsStage1Dashboard": outs_stage1_dashboard,
        "outsStage1ReportMarkdown": outs_stage1_dashboard.get("outsStage1ReportMarkdown"),
        "requestedMatchups": [
            f"{away}@{home}" for away, home in requested
        ],
        "unmatchedRequestedMatchups": unmatched,
        "gamesReturned": len(screened),
        "games": compact_games,
        "recommendedPitchers": [
            _compact_out_pitcher(pitcher)
            for pitcher in recommended_pitchers
        ],
        "notSelectedPitchers": [
            _compact_out_pitcher(pitcher)
            for pitcher in not_selected_pitchers
        ],
        "outScreeningGroups": compact_groups,
        "rankGapNotes": rank_gap_notes,
        "notes": [
            "This endpoint screens games for deeper pitcher-outs research. It does not recommend bets.",
            "Use outsStage1Dashboard for the user-facing Stage 1 report.",
            "Use outsStage1ReportMarkdown exactly when the GPT needs a consistent ready-made Stage 1 report.",
            "Expected Outs is the preferred Stage 1 ranking field.",
            "Recommended pitchers come from the RESEARCH group, up to a maximum of 10.",
            "Every screened pitcher is returned in outScreeningGroups: research, borderline, or passForNow.",
            "Use measurable workload data only. Reputation or name value is not a reason to research a pitcher.",
            "Send only screenshot-provided matchups in the matchups query when screening a Bet365 screenshot.",
            "Projected lineups can support research with an uncertainty haircut, but never treat projected lineups as confirmed.",
            "Confirm lineups, injury news, weather, Bet365 pitcher-outs lines, and prices before any final bet decision.",
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


@app.get("/stage2-outs-research")
def stage2_outs_research():
    date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    season = request.args.get("season") or date[:4]
    raw_candidates = request.args.get("candidates", "")
    include_details = str(
        request.args.get("include_details", "false")
    ).lower() in {"1", "true", "yes"}
    candidates, parse_errors = _parse_stage2_candidates(raw_candidates)

    if not raw_candidates.strip():
        return jsonify({"error": "candidates is required"}), 400

    games_by_matchup = _game_lookup(date)
    indexed_profiles = []
    with ThreadPoolExecutor(max_workers=min(max(len(candidates), 1), 6)) as executor:
        futures = {
            executor.submit(
                _stage2_outs_candidate_profile,
                candidate,
                season,
                games_by_matchup,
                date,
            ): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            indexed_profiles.append((futures[future], future.result()))
    profiles = [
        profile
        for _, profile in sorted(indexed_profiles, key=lambda item: item[0])
    ]
    report = _stage2_outs_report_markdown(date, profiles, parse_errors)

    payload = {
        "date": date,
        "season": season,
        "stage": "Stage 2 Pitcher Outs",
        "screeningOnly": True,
        "responseMode": "compact",
        "inputFormat": "Pitcher|Team|Matchup|OutsLine|OverOdds|UnderOdds;...",
        "candidatesReturned": len(profiles),
        "parseErrors": parse_errors,
        "profiles": [_stage2_outs_compact_profile(profile) for profile in profiles],
        "stage2OutsReportMarkdown": report,
        "notes": [
            "Stage 2 compares the exact Bet365 pitcher-outs line and odds supplied by the user.",
            "This endpoint does not pull Bet365 odds. User screenshots remain the source of truth for line and price.",
            "The report is research only. It does not make final bet recommendations.",
            "Actual last 10 starts are shown for every readable candidate, with Last 5 marked.",
            "Stage 2 evaluates Over and Under equally for pitcher outs.",
            "Stage 2 ends by asking for the user's next available recheck time.",
        ],
    }
    if include_details:
        payload["responseMode"] = "detailed"
        payload["debugFullProfiles"] = profiles

    return jsonify(payload)


@app.get("/stage2-research")
def stage2_research():
    date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    season = request.args.get("season") or date[:4]
    raw_candidates = request.args.get("candidates", "")
    include_pitch_quality = str(
        request.args.get("include_pitch_quality", "true")
    ).lower() not in {"0", "false", "no"}
    include_details = str(
        request.args.get("include_details", "false")
    ).lower() in {"1", "true", "yes"}
    candidates, parse_errors = _parse_stage2_candidates(raw_candidates)

    if not raw_candidates.strip():
        return jsonify({"error": "candidates is required"}), 400

    games_by_matchup = _game_lookup(date)
    indexed_profiles = []
    with ThreadPoolExecutor(max_workers=min(max(len(candidates), 1), 6)) as executor:
        futures = {
            executor.submit(
                _stage2_candidate_profile,
                candidate,
                season,
                games_by_matchup,
                date,
                include_pitch_quality,
            ): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            indexed_profiles.append((futures[future], future.result()))
    profiles = [
        profile
        for _, profile in sorted(indexed_profiles, key=lambda item: item[0])
    ]
    report = _stage2_report_markdown(date, profiles, parse_errors)

    payload = {
        "date": date,
        "season": season,
        "stage": "Stage 2",
        "screeningOnly": True,
        "responseMode": "compact",
        "inputFormat": "Pitcher|Team|Matchup|Line|OverOdds|UnderOdds[|Stage1Group|OriginalLine|OriginalOverOdds|OriginalUnderOdds];...",
        "candidatesReturned": len(profiles),
        "parseErrors": parse_errors,
        "profiles": [_stage2_compact_profile(profile) for profile in profiles],
        "stage2ReportMarkdown": report,
        "notes": [
            "Stage 2 compares the exact Bet365 line and odds supplied by the user.",
            "This endpoint does not pull Bet365 odds. User screenshots remain the source of truth for line and price.",
            "The report is research only. It does not make final bet recommendations.",
            "Actual last 10 starts are shown for every readable candidate, with Last 5 marked.",
            "Pitch mix and velocity are compact checks from recent pitch-level MLB Stats API data when include_pitch_quality=true.",
            "Optional Stage1Group metadata activates the stricter BORDERLINE promotion gate; missing metadata defaults to UNKNOWN.",
            "Stage 2 ends by asking for the user's next available recheck time.",
        ],
    }
    if include_details:
        payload["responseMode"] = "detailed"
        payload["debugFullProfiles"] = profiles

    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
