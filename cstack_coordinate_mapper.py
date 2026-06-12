"""
CSTACK Labs — Module B: 2D Map Coordinate Mapper
================================================
Takes the JSON payload from Module A (cstack_vod_engine.py) and resolves the
named locations mentioned in each tactical failure into normalized (x, y)
coordinates in the 0.0–1.0 range, ready to plot on a 2D radar overlay in the
dashboard.

How it works:
1. A per-map callout table maps human callout names -> normalized coords.
2. A second Gemini pass (cheap, text-only) reads each failure's critique and
   extracts which callout it happened at. This is more robust than regex
   because critiques phrase locations loosely ("near the CT spawn boost").
3. We look the callout up in the table (with simple fuzzy fallback) and stamp
   coordinates onto each failure object.

You can run this fully offline (no second model call) by passing
use_model=False, in which case it does keyword matching against the table.

Setup:
    pip install google-genai
    export GEMINI_API_KEY="your_key"
"""

import os
import json
import difflib

from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.5-flash"

# ---------------------------------------------------------------------------
# Callout tables. Coordinates are normalized 0.0-1.0 where (0,0) is the
# top-left of the radar image and (1,1) is the bottom-right. These are
# approximate anchor points — tune them to match whatever radar PNG your
# dashboard renders. Add maps/callouts as needed.
# ---------------------------------------------------------------------------
CALLOUT_TABLES = {
    "mirage": {
        "t spawn": (0.50, 0.92),
        "ct spawn": (0.55, 0.10),
        "mid": (0.50, 0.50),
        "top mid": (0.50, 0.40),
        "catwalk": (0.62, 0.42),
        "window": (0.55, 0.30),
        "a site": (0.72, 0.25),
        "a ramp": (0.78, 0.45),
        "palace": (0.80, 0.30),
        "ticket": (0.70, 0.20),
        "b site": (0.22, 0.30),
        "b apartments": (0.20, 0.55),
        "market": (0.30, 0.35),
        "van": (0.18, 0.25),
        "underpass": (0.35, 0.70),
        "connector": (0.55, 0.40),
        "jungle": (0.62, 0.32),
        "stairs": (0.66, 0.28),
    },
    "inferno": {
        "t spawn": (0.30, 0.92),
        "ct spawn": (0.55, 0.12),
        "mid": (0.45, 0.50),
        "banana": (0.30, 0.45),
        "b site": (0.28, 0.22),
        "first oranges": (0.32, 0.30),
        "car": (0.28, 0.28),
        "a site": (0.70, 0.30),
        "apartments": (0.55, 0.62),
        "balcony": (0.66, 0.40),
        "pit": (0.78, 0.32),
        "graveyard": (0.62, 0.22),
        "arch": (0.50, 0.42),
        "library": (0.60, 0.30),
    },
    "dust2": {
        "t spawn": (0.50, 0.92),
        "ct spawn": (0.50, 0.10),
        "mid": (0.50, 0.50),
        "mid doors": (0.50, 0.42),
        "catwalk": (0.62, 0.55),
        "lower tunnels": (0.70, 0.70),
        "upper tunnels": (0.72, 0.55),
        "a site": (0.78, 0.30),
        "a long": (0.78, 0.62),
        "long doors": (0.80, 0.72),
        "a short": (0.62, 0.40),
        "goose": (0.74, 0.28),
        "b site": (0.20, 0.28),
        "b tunnels": (0.30, 0.40),
        "b platform": (0.16, 0.24),
        "b window": (0.30, 0.22),
        "car": (0.22, 0.32),
    },
    "ancient": {
        "t spawn": (0.40, 0.90),
        "ct spawn": (0.55, 0.12),
        "mid": (0.48, 0.50),
        "a site": (0.72, 0.30),
        "donut": (0.66, 0.38),
        "b site": (0.25, 0.32),
        "cave": (0.20, 0.55),
        "ramp": (0.35, 0.40),
        "red room": (0.30, 0.25),
    },
    "nuke": {
        "t spawn": (0.20, 0.50),
        "ct spawn": (0.80, 0.45),
        "outside": (0.30, 0.75),
        "a site": (0.60, 0.35),
        "b site": (0.60, 0.60),
        "ramp": (0.50, 0.30),
        "hut": (0.40, 0.40),
        "secret": (0.72, 0.62),
        "lobby": (0.45, 0.20),
    },
}


def _normalize(name: str) -> str:
    return name.strip().lower()


def lookup_callout(map_name: str, callout: str) -> dict | None:
    """Resolve a callout to coords for a given map, with fuzzy fallback."""
    table = CALLOUT_TABLES.get(_normalize(map_name))
    if not table:
        return None

    key = _normalize(callout)
    if key in table:
        x, y = table[key]
        return {"callout": key, "x": x, "y": y, "match": "exact"}

    # Substring match (e.g. "the A site" -> "a site")
    for cand, (x, y) in table.items():
        if cand in key or key in cand:
            return {"callout": cand, "x": x, "y": y, "match": "substring"}

    # Fuzzy fallback
    close = difflib.get_close_matches(key, list(table.keys()), n=1, cutoff=0.6)
    if close:
        x, y = table[close[0]]
        return {"callout": close[0], "x": x, "y": y, "match": "fuzzy"}

    return None


# ---------------------------------------------------------------------------
# Model-assisted callout extraction (optional)
# ---------------------------------------------------------------------------
_EXTRACT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"callout": types.Schema(type=types.Type.STRING)},
    required=["callout"],
)


def _extract_callout_with_model(client, map_name, critique, valid_callouts):
    """Ask the model which known callout a critique refers to."""
    instruction = (
        f"You map Counter-Strike critique text to map callouts. "
        f"Map: {map_name}. Valid callouts: {', '.join(valid_callouts)}. "
        f"Return the single best-matching callout from that list. If none "
        f"clearly applies, return an empty string."
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[critique],
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            response_mime_type="application/json",
            response_schema=_EXTRACT_SCHEMA,
            temperature=0.0,
        ),
    )
    return json.loads(resp.text).get("callout", "")


def enrich_with_coordinates(payload: dict, use_model: bool = True) -> dict:
    """
    Walk a Module A payload and add a 'location' object with x/y coords to
    each tactical failure. Returns the same dict, mutated in place.
    """
    map_name = payload.get("match_metadata", {}).get("map_identified", "")
    table = CALLOUT_TABLES.get(_normalize(map_name), {})

    client = None
    if use_model and table:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)

    for failure in payload.get("tactical_failures", []):
        critique = failure.get("critique", "")
        resolved = None

        if client:
            callout = _extract_callout_with_model(
                client, map_name, critique, list(table.keys())
            )
            if callout:
                resolved = lookup_callout(map_name, callout)

        # Offline fallback: keyword-scan the critique against the table.
        if resolved is None:
            text = _normalize(critique)
            for cand in table:
                if cand in text:
                    resolved = lookup_callout(map_name, cand)
                    break

        failure["location"] = resolved or {
            "callout": None,
            "x": None,
            "y": None,
            "match": "unresolved",
        }

    return payload


if __name__ == "__main__":
    # Demo with a hand-written sample payload (the shape Module A returns).
    sample = {
        "match_metadata": {
            "map_identified": "Mirage",
            "overall_verdict": "Decent aim, poor mid control discipline.",
        },
        "tactical_failures": [
            {
                "timestamp_start": "00:14",
                "timestamp_end": "00:21",
                "failure_archetype": "Uncoordinated mid push",
                "critique": "Two players pushed top mid without a smoke for window, "
                            "giving the CT an uncontested AWP angle.",
                "prescribed_drill": "Practice the standard window smoke from T spawn.",
            },
            {
                "timestamp_start": "01:03",
                "timestamp_end": "01:10",
                "failure_archetype": "Bad trade spacing",
                "critique": "Entry on A ramp was 4 seconds ahead of the trade player, "
                            "so the pick went unpunished.",
                "prescribed_drill": "Run 2-man A execute drills keeping under 1.5s trade gap.",
            },
        ],
    }

    # use_model=False keeps the demo offline; flip to True with a key set.
    enriched = enrich_with_coordinates(sample, use_model=False)
    print(json.dumps(enriched, indent=4))
