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

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    _GENAI_AVAILABLE = False

MODEL_NAME = "gemini-3.5-flash"

# ---------------------------------------------------------------------------
# Callout tables. Coordinates are normalized 0.0-1.0 where (0,0) is the
# top-left of the radar image and (1,1) is the bottom-right. These are
# approximate anchor points — tune them to match whatever radar PNG your
# dashboard renders. Add maps/callouts as needed.
# ---------------------------------------------------------------------------
# Coordinates are derived from the standard A–P / 1–17 referee grid used on
# official callout maps.  Column centres: A=0.03 B=0.09 C=0.16 D=0.22 E=0.28
# F=0.34 G=0.41 H=0.47 I=0.53 J=0.59 K=0.66 L=0.72 M=0.78 N=0.84 O=0.91
# P=0.97.  Row centres follow the same spacing top-to-bottom.
CALLOUT_TABLES = {
    # -----------------------------------------------------------------------
    "ancient": {
        # spawns / sites
        "t spawn":       (0.44, 0.88),
        "ct spawn":      (0.44, 0.13),
        "a site":        (0.19, 0.25),
        "b site":        (0.63, 0.44),
        # A-site approach
        "a main":        (0.06, 0.41),
        "a short":       (0.22, 0.38),
        "plat":          (0.09, 0.22),
        "big box":       (0.16, 0.22),
        "ct":            (0.22, 0.22),
        "triple":        (0.28, 0.25),
        "temple":        (0.31, 0.16),
        "sniper nest":   (0.38, 0.28),
        "tree":          (0.16, 0.34),
        "donut":         (0.22, 0.47),
        "red":           (0.34, 0.34),
        # mid
        "mid":           (0.34, 0.53),
        "top mid":       (0.38, 0.44),
        "lower mid":     (0.31, 0.59),
        "pit":           (0.41, 0.47),
        # B-site approach
        "cave":          (0.50, 0.41),
        "b short":       (0.66, 0.34),
        "alley":         (0.59, 0.25),
        "back alley":    (0.81, 0.22),
        "long":          (0.88, 0.34),
        "wood":          (0.75, 0.47),
        "cat room":      (0.63, 0.53),
        "cat":           (0.47, 0.59),
        "ramp":          (0.84, 0.53),
    },
    # -----------------------------------------------------------------------
    "anubis": {
        # spawns / sites
        "t spawn":       (0.50, 0.94),
        "ct spawn":      (0.56, 0.19),
        "a site":        (0.69, 0.28),
        "b site":        (0.19, 0.50),
        # A-site approach
        "a main":        (0.91, 0.47),
        "heaven":        (0.59, 0.22),
        "back site":     (0.75, 0.22),
        "fountain":      (0.97, 0.31),
        "plateau":       (0.69, 0.34),
        "a-con":         (0.63, 0.41),
        "headshot":      (0.66, 0.50),
        "boat":          (0.72, 0.56),
        "upper":         (0.91, 0.66),
        "stairs":        (0.66, 0.66),
        # mid
        "mid":           (0.47, 0.41),
        "top mid":       (0.47, 0.72),
        "house":         (0.41, 0.50),
        "doors":         (0.47, 0.53),
        "water":         (0.59, 0.59),
        "bridge":        (0.34, 0.66),
        "alley":         (0.59, 0.78),
        # B-site approach
        "b main":        (0.09, 0.66),
        "palace":        (0.34, 0.41),
        "cave":          (0.28, 0.34),
        "sniper":        (0.25, 0.28),
        "street":        (0.22, 0.38),
        "corner":        (0.19, 0.47),
        "ninja":         (0.28, 0.53),
        "pillar":        (0.22, 0.59),
        "b-con":         (0.28, 0.63),
        "ruins":         (0.22, 0.78),
    },
    # -----------------------------------------------------------------------
    "aztec": {
        # spawns / sites
        "t spawn":       (0.38, 0.88),
        "ct spawn":      (0.35, 0.12),
        "a site":        (0.15, 0.35),
        "b site":        (0.65, 0.42),
        # main areas
        "a main":        (0.08, 0.48),
        "house":         (0.40, 0.28),
        "alley":         (0.55, 0.22),
        "tomb":          (0.18, 0.52),
        "middle":        (0.42, 0.52),
        "dig":           (0.55, 0.52),
        "lower b long":  (0.72, 0.55),
        "bridge":        (0.42, 0.62),
        "tunnel":        (0.42, 0.75),
    },
    # -----------------------------------------------------------------------
    "dust2": {
        # spawns / sites
        "t spawn":       (0.50, 0.92),
        "ct spawn":      (0.50, 0.10),
        "a site":        (0.78, 0.30),
        "b site":        (0.20, 0.28),
        # mid
        "mid":           (0.50, 0.50),
        "mid doors":     (0.50, 0.42),
        "catwalk":       (0.62, 0.55),
        # A-site approach
        "a long":        (0.78, 0.62),
        "long doors":    (0.80, 0.72),
        "a short":       (0.62, 0.40),
        "upper tunnels": (0.72, 0.55),
        "lower tunnels": (0.70, 0.70),
        "goose":         (0.74, 0.28),
        # B-site approach
        "b tunnels":     (0.30, 0.40),
        "b platform":    (0.16, 0.24),
        "b window":      (0.30, 0.22),
        "car":           (0.22, 0.32),
    },
    # -----------------------------------------------------------------------
    "inferno": {
        # spawns / sites
        "t spawn":       (0.06, 0.69),
        "ct spawn":      (0.94, 0.38),
        "a site":        (0.75, 0.69),
        "b site":        (0.44, 0.19),
        # B-site area
        "garden":        (0.47, 0.03),
        "dark":          (0.38, 0.09),
        "coffins":       (0.44, 0.09),
        "church":        (0.63, 0.09),
        "pool":          (0.50, 0.16),
        "fountain":      (0.41, 0.22),
        "ct":            (0.56, 0.22),
        "boost":         (0.63, 0.28),
        "speedway":      (0.63, 0.31),
        # banana
        "1st":           (0.47, 0.34),
        "2nd":           (0.41, 0.31),
        "3rd":           (0.34, 0.28),
        "banana":        (0.44, 0.44),
        "car":           (0.47, 0.38),
        "sandbag":       (0.53, 0.44),
        "loggs":         (0.28, 0.53),
        "long corner":   (0.47, 0.53),
        # mid
        "mid":           (0.53, 0.66),
        "underpass":     (0.41, 0.66),
        "second mid":    (0.41, 0.84),
        # A-site approach
        "arch":          (0.59, 0.53),
        "long":          (0.66, 0.59),
        "library":       (0.91, 0.59),
        "moto":          (0.78, 0.59),
        "cubby":         (0.66, 0.63),
        "graveyard":     (0.91, 0.69),
        "pit":           (0.94, 0.78),
        "mini pit":      (0.81, 0.91),
        "balcony":       (0.66, 0.91),
        "halls":         (0.59, 0.91),
        "short":         (0.66, 0.81),
        "patio":         (0.59, 0.84),
        "bedroom":       (0.53, 0.84),
        "boiler":        (0.53, 0.78),
        # T-side
        "ramp":          (0.28, 0.66),
        "t apps":        (0.22, 0.91),
        "apartments":    (0.22, 0.91),
        "bridge":        (0.22, 0.84),
        "back alley":    (0.41, 0.97),
    },
    # -----------------------------------------------------------------------
    "mirage": {
        # spawns / sites
        "t spawn":       (0.94, 0.31),
        "ct spawn":      (0.13, 0.75),
        "a site":        (0.44, 0.81),
        "b site":        (0.13, 0.25),
        # B-site approach
        "b apartments":  (0.25, 0.16),
        "balcony":       (0.13, 0.09),
        "van":           (0.09, 0.16),
        "bench":         (0.03, 0.28),
        "door":          (0.03, 0.41),
        "b window":      (0.09, 0.41),
        "market":        (0.09, 0.47),
        "short":         (0.28, 0.28),
        "corner":        (0.28, 0.22),
        "b stairs":      (0.41, 0.22),
        "ladder":        (0.28, 0.34),
        "cat":           (0.25, 0.31),
        # mid
        "mid":           (0.44, 0.44),
        "top mid":       (0.66, 0.47),
        "catwalk":       (0.41, 0.41),
        "underpass":     (0.41, 0.38),
        "window":        (0.34, 0.47),
        "connector":     (0.47, 0.56),
        "chair":         (0.53, 0.53),
        # T-side / jungle
        "t apps":        (0.75, 0.16),
        "jungle":        (0.34, 0.59),
        "stairs":        (0.47, 0.66),
        "sandwich":      (0.59, 0.63),
        "tetris":        (0.66, 0.66),
        "ramp":          (0.75, 0.66),
        "shadow":        (0.75, 0.72),
        # A-site approach
        "palace":        (0.75, 0.78),
        "triple":        (0.41, 0.78),
        "ct":            (0.34, 0.84),
        "ninja":         (0.47, 0.84),
        "firebox":       (0.56, 0.84),
        "ticket":        (0.41, 0.91),
    },
    # -----------------------------------------------------------------------
    "nuke": {
        # spawns / sites
        "t spawn":       (0.13, 0.59),
        "ct spawn":      (0.88, 0.50),
        "a site":        (0.59, 0.53),
        "b site":        (0.59, 0.59),  # lower level; overlaps A in top-down view
        # upper level (A)
        "ramp":          (0.50, 0.34),
        "boost":         (0.50, 0.44),
        "heaven":        (0.66, 0.47),
        "locker":        (0.72, 0.50),
        "ct box":        (0.81, 0.53),
        "blue box":      (0.72, 0.59),
        "radio":         (0.34, 0.50),
        "lobby":         (0.34, 0.56),
        "hut":           (0.47, 0.59),
        "squeaky":       (0.41, 0.63),
        "silo":          (0.34, 0.66),
        "main":          (0.47, 0.66),
        "outside":       (0.47, 0.72),
        "garage":        (0.75, 0.72),
        "red":           (0.41, 0.78),
        "secret":        (0.59, 0.78),
    },
    # -----------------------------------------------------------------------
    "train": {
        # spawns / sites
        "t spawn":       (0.06, 0.19),
        "ct spawn":      (0.78, 0.81),
        "a site":        (0.78, 0.50),
        "b site":        (0.44, 0.81),
        # A-site approaches
        "a main":        (0.19, 0.38),
        "ivy":           (0.75, 0.28),
        "alley":         (0.50, 0.13),
        "pigeons":       (0.88, 0.13),
        "sandwich":      (0.41, 0.28),
        "dumpster":      (0.34, 0.22),
        "t stairs":      (0.25, 0.28),
        "kitchen":       (0.28, 0.34),
        "hell":          (0.66, 0.34),
        "olof":          (0.34, 0.41),
        "blue":          (0.41, 0.41),
        "green":         (0.50, 0.41),
        "a3":            (0.81, 0.41),
        "red":           (0.41, 0.47),
        "camera":        (0.66, 0.47),
        "ct tunnel":     (0.91, 0.47),
        "brown halls":   (0.06, 0.50),
        "showers":       (0.22, 0.50),
        "ebox":          (0.34, 0.53),
        "bomb train":    (0.59, 0.53),
        "old bomb":      (0.69, 0.53),
        "a2":            (0.78, 0.53),
        "a1":            (0.84, 0.56),
        "heaven":        (0.63, 0.59),
        "pop dog":       (0.19, 0.63),
        "cubby":         (0.81, 0.59),
        "z connector":   (0.59, 0.66),
        "ct stairs":     (0.91, 0.66),
        # B-site approaches
        "b ramp":        (0.22, 0.72),
        "sidewalk":      (0.50, 0.72),
        "b halls":       (0.13, 0.78),
        "white":         (0.38, 0.78),
        "summit":        (0.47, 0.78),
        "b red":         (0.53, 0.78),
        "back site":     (0.69, 0.78),
        "yellow":        (0.63, 0.81),
        "spools":        (0.25, 0.84),
        "oil":           (0.53, 0.84),
        "ladder":        (0.19, 0.88),
        "upper b":       (0.28, 0.88),
        "catwalk":       (0.44, 0.88),
        "headshot":      (0.56, 0.94),
    },
    # -----------------------------------------------------------------------
    "vertigo": {
        # spawns / sites (upper floor)
        "t spawn":       (0.28, 0.75),
        "ct spawn":      (0.53, 0.25),
        "a site":        (0.63, 0.63),
        "b site":        (0.13, 0.25),
        # B-site area
        "back site":     (0.09, 0.16),
        "catwalk":       (0.25, 0.16),
        "cooler":        (0.34, 0.19),
        "ct":            (0.34, 0.22),
        "white box":     (0.09, 0.28),
        "con":           (0.34, 0.28),
        "b stairs":      (0.09, 0.47),
        "wood":          (0.19, 0.38),
        # mid
        "mid":           (0.31, 0.50),
        "t mid":         (0.19, 0.59),
        "ct mid":        (0.41, 0.41),
        "mid boost":     (0.28, 0.41),
        "back door":     (0.53, 0.31),
        "heaven":        (0.66, 0.41),
        "elevator":      (0.50, 0.53),
        "generator":     (0.03, 0.59),
        # A-site approach
        "boost":         (0.72, 0.59),
        "ramp":          (0.50, 0.69),
        "box":           (0.50, 0.66),
        "short":         (0.38, 0.72),
        "ladder":        (0.34, 0.66),
        "t stairs":      (0.09, 0.69),
        "top ramp":      (0.56, 0.78),
        "sand bags":     (0.63, 0.84),
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
_EXTRACT_SCHEMA = (
    types.Schema(
        type=types.Type.OBJECT,
        properties={"callout": types.Schema(type=types.Type.STRING)},
        required=["callout"],
    )
    if _GENAI_AVAILABLE
    else None
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
    if use_model and table and _GENAI_AVAILABLE:
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

        # Offline fallback: keyword-scan the critique, longest key first so
        # short tokens ("ct") cannot shadow more-specific ones ("ct mid").
        if resolved is None:
            text = _normalize(critique)
            for cand in sorted(table, key=len, reverse=True):
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
    demos = [
        {
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
                    "critique": "Entry onto palace was 4 seconds ahead of the trade player "
                                "so the pick went unpunished.",
                    "prescribed_drill": "Run 2-man A execute drills keeping under 1.5 s trade gap.",
                },
            ],
        },
        {
            "match_metadata": {
                "map_identified": "Inferno",
                "overall_verdict": "Strong banana control but A site retakes collapsed.",
            },
            "tactical_failures": [
                {
                    "timestamp_start": "02:11",
                    "timestamp_end": "02:19",
                    "failure_archetype": "Banana over-extension",
                    "critique": "Three Ts held loggs indefinitely and got caught by a CT "
                                "reposition through the arch.",
                    "prescribed_drill": "Set a hard 20-second timer on banana aggression.",
                },
                {
                    "timestamp_start": "03:45",
                    "timestamp_end": "03:52",
                    "failure_archetype": "Graveyard lurk ignored",
                    "critique": "The lurker cut through graveyard completely uncontested "
                                "while the team was focused on the pit.",
                    "prescribed_drill": "Assign a dedicated graveyard watcher on every A execute.",
                },
            ],
        },
        {
            "match_metadata": {
                "map_identified": "Anubis",
                "overall_verdict": "Solid mechanics but mid-control timing was off.",
            },
            "tactical_failures": [
                {
                    "timestamp_start": "00:55",
                    "timestamp_end": "01:03",
                    "failure_archetype": "Uncontested heaven",
                    "critique": "CTs held heaven throughout round 4 and the Ts never "
                                "threw a smoke to neutralise the angle.",
                    "prescribed_drill": "Drill the standard heaven smoke from T main.",
                },
            ],
        },
        {
            "match_metadata": {
                "map_identified": "Vertigo",
                "overall_verdict": "Elevator timings sloppy; mid not contested.",
            },
            "tactical_failures": [
                {
                    "timestamp_start": "01:22",
                    "timestamp_end": "01:30",
                    "failure_archetype": "Mid neglect",
                    "critique": "No player contested ct mid at round start, letting CTs "
                                "set up a dominant mid boost position.",
                    "prescribed_drill": "Send one T to mid boost early every round.",
                },
            ],
        },
    ]

    for sample in demos:
        enriched = enrich_with_coordinates(sample, use_model=False)
        print(json.dumps(enriched, indent=2))
        print()
