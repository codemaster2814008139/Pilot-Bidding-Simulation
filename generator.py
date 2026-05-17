"""
generator.py
------------
Generates synthetic pilot profiles and flight pairings.
Mirrors the JavaScript buildPilots() / buildPairings() logic in the POC,
with deterministic seeding so scenarios are reproducible.

Usage:
    from generator import ScenarioGenerator
    gen = ScenarioGenerator(pilot_seed=1234567, pairing_seed=42)
    pilots   = gen.build_pilots(n=3)
    pairings = gen.build_pairings(n=5, pilots=pilots, base='BOS')
"""

import random
from typing import List, Optional

from models import Leg, Line, Pairing, Pilot, OracleWeights

# ---------------------------------------------------------------------------
# Static airport data
# ---------------------------------------------------------------------------

AIRPORTS = [
    {"code": "JFK", "city": "New York",      "tz": -5},
    {"code": "LAX", "city": "Los Angeles",   "tz": -8},
    {"code": "ORD", "city": "Chicago",       "tz": -6},
    {"code": "DFW", "city": "Dallas",        "tz": -6},
    {"code": "ATL", "city": "Atlanta",       "tz": -5},
    {"code": "DEN", "city": "Denver",        "tz": -7},
    {"code": "SEA", "city": "Seattle",       "tz": -8},
    {"code": "BOS", "city": "Boston",        "tz": -5},
    {"code": "MIA", "city": "Miami",         "tz": -5},
    {"code": "PHX", "city": "Phoenix",       "tz": -7},
    {"code": "MSP", "city": "Minneapolis",   "tz": -6},
    {"code": "DTW", "city": "Detroit",       "tz": -5},
    {"code": "LAS", "city": "Las Vegas",     "tz": -8},
    {"code": "MCO", "city": "Orlando",       "tz": -5},
    {"code": "SFO", "city": "San Francisco", "tz": -8},
    {"code": "CLT", "city": "Charlotte",     "tz": -5},
    {"code": "IAH", "city": "Houston",       "tz": -6},
    {"code": "SLC", "city": "Salt Lake City","tz": -7},
    {"code": "PDX", "city": "Portland",      "tz": -8},
    {"code": "BWI", "city": "Baltimore",     "tz": -5},
]

# Approximate distances in miles between US hubs
DISTANCES = {
    ("BOS", "MIA"): 1258, ("BOS", "SFO"): 2699, ("BOS", "LAX"): 2598,
    ("BOS", "ORD"): 854,  ("BOS", "DFW"): 1560, ("BOS", "ATL"): 1103,
    ("BOS", "DEN"): 1748, ("BOS", "SEA"): 2490, ("BOS", "LAS"): 2380,
    ("BOS", "MCO"): 1114, ("BOS", "IAH"): 1597, ("BOS", "PDX"): 2521,
    ("BOS", "JFK"): 188,  ("BOS", "PHX"): 2300, ("BOS", "MSP"): 1123,
    ("MIA", "SFO"): 2581, ("MIA", "LAX"): 2342, ("MIA", "ORD"): 1197,
    ("MIA", "MCO"): 236,  ("MIA", "DFW"): 1120, ("MIA", "IAH"): 968,
    ("SFO", "LAX"): 337,  ("SFO", "ORD"): 1846, ("SFO", "SEA"): 679,
    ("SFO", "LAS"): 414,  ("SFO", "MCO"): 2441, ("SFO", "PHX"): 651,
    ("LAX", "ORD"): 1744, ("LAX", "LAS"): 236,  ("LAX", "SEA"): 954,
    ("LAX", "PHX"): 370,  ("LAX", "DFW"): 1235,
    ("ORD", "DFW"): 802,  ("ORD", "ATL"): 606,  ("ORD", "MSP"): 334,
    ("DFW", "ATL"): 731,  ("DFW", "IAH"): 224,  ("DFW", "PHX"): 868,
    ("ATL", "MCO"): 403,  ("ATL", "CLT"): 227,
    ("DEN", "LAS"): 748,  ("DEN", "SLC"): 391,  ("DEN", "PHX"): 586,
    ("SEA", "PDX"): 145,  ("SEA", "SLC"): 689,
    ("LAS", "PHX"): 256,  ("LAS", "SLC"): 368,
    ("IAH", "MCO"): 853,  ("PDX", "SLC"): 630,
    ("JFK", "CLT"): 634,  ("MSP", "DTW"): 528,
}

AIRCRAFT = [
    {"type": "B737", "label": "Boeing 737", "speed_mph": 480},
    {"type": "B767", "label": "Boeing 767", "speed_mph": 520},
]

PILOT_NAMES = [
    "Alex Carter", "Jordan Lee", "Sam Rivera",
    "Morgan Kim",  "Taylor Brooks", "Casey Walsh",
    "Dana Okafor", "Reese Tanaka",  "Quinn Patel",
    "Avery Santos",
]

FAMILY_OPTIONS = [
    "Single",
    "Married, no kids",
    "Married, 1 child",
    "Married, 2+ kids",
    "Single parent",
]

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _get_distance(dep: str, arr: str) -> int:
    """Look up distance; try both orderings; fall back to estimate."""
    d = DISTANCES.get((dep, arr)) or DISTANCES.get((arr, dep))
    return d or 800  # fallback estimate


def _block_mins(distance_mi: int, speed_mph: int) -> int:
    """Block time = flight time + 30 min taxi/buffer, rounded to nearest 5 min."""
    raw = round(distance_mi / speed_mph * 60) + 30
    return round(raw / 5) * 5


def _fmt_time(mins_from_midnight: int) -> str:
    mins_from_midnight = ((mins_from_midnight % 1440) + 1440) % 1440
    h, m = divmod(mins_from_midnight, 60)
    ampm = "AM" if h < 12 else "PM"
    h12  = h if 1 <= h <= 12 else (12 if h == 0 else h - 12)
    return f"{h12}:{m:02d} {ampm}"


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """
    Deterministic scenario generator.

    pilot_seed:   controls pilot profile randomness
                  (change to get different pilot profiles)
    pairing_seed: controls pairing route/schedule randomness
                  (change to get different pairings)
    """

    def __init__(self, pilot_seed: int = 1234567, pairing_seed: int = 42):
        self._prng_pilot   = random.Random(pilot_seed)
        self._prng_pairing = random.Random(pairing_seed)

    # ------------------------------------------------------------------
    # Pilots
    # ------------------------------------------------------------------

    def build_pilots(self, n: int = 3, base: str = "BOS") -> List[Pilot]:
        """
        Generate n pilot profiles with auto-derived oracle weights.
        Seniority is assigned 1..n (1 = most senior).
        """
        rng    = self._prng_pilot
        names  = rng.sample(PILOT_NAMES, min(n, len(PILOT_NAMES)))
        pilots = []

        for i, name in enumerate(names):
            age           = rng.randint(26, 58)
            family        = rng.choice(FAMILY_OPTIONS)
            wide_qual     = rng.random() > 0.35
            preferred     = ("B767" if wide_qual and rng.random() > 0.5
                             else "B737")
            qualified     = ["B737", "B767"] if wide_qual else ["B737"]
            base_pay      = rng.randint(180, 270)
            min_rest      = rng.randint(10, 14)

            pilot = Pilot(
                id=i,
                name=name,
                age=age,
                family_status=family,
                seniority=i + 1,
                home_base=base,
                qualified_types=qualified,
                preferred_type=preferred,
                min_rest=min_rest,
                base_pay=base_pay,
            )
            pilot.weights = pilot.auto_weights()
            pilots.append(pilot)

        return pilots

    # ------------------------------------------------------------------
    # Pairings
    # ------------------------------------------------------------------

    def build_pairings(
        self,
        n: int = 5,
        pilots: Optional[List[Pilot]] = None,
        base: str = "BOS",
        max_b767: Optional[int] = None,
    ) -> List[Pairing]:
        """
        Generate n circular pairings starting and ending at `base`.
        Variable legs (2-4) and nights away (1-3).

        Args:
            n:        Number of pairings to generate.
            pilots:   If provided, reference pay is taken from pilots[0].
            base:     Home-base airport code.
            max_b767: If set, at most this many pairings will be B767.
                      All remaining pairings will be B737.  Used to enforce
                      the 17 B737 / 8 B767 ratio when building 25 pairings
                      for line mode.
        """
        rng       = self._prng_pairing
        pairings  = []
        used_keys = set()

        ref_pay = pilots[0].base_pay if pilots else 200
        airport_pool = [a for a in AIRPORTS if a["code"] != base]
        b767_count = 0

        for p_idx in range(n):
            # Enforce B767 cap if specified
            if max_b767 is not None and b767_count >= max_b767:
                aircraft_data = next(a for a in AIRCRAFT if a["type"] == "B737")
                _ = rng.choice(AIRCRAFT)  # consume RNG tick to keep seed deterministic
            else:
                aircraft_data = rng.choice(AIRCRAFT)
            if aircraft_data["type"] == "B767":
                b767_count += 1
            num_legs      = rng.choice([2, 3, 3, 4])   # weighted toward 3
            nights_away   = num_legs - 1
            start_dow     = rng.choice(DOW)

            # Pick intermediate stops (unique per pairing)
            stops = rng.sample(airport_pool, num_legs - 1)
            route = [base] + [s["code"] for s in stops] + [base]

            # Build legs with realistic chained schedule.
            # First departure: morning bank (6:00–9:30 AM, 55%) or
            # afternoon bank (2:00–6:00 PM, 45%), in 5-min slots.
            legs     = []
            cur_day  = 1
            if rng.random() < 0.55:
                cur_mins = round(rng.randint(360, 570) / 5) * 5   # 6:00–9:30 AM
            else:
                cur_mins = round(rng.randint(840, 1080) / 5) * 5  # 2:00–6:00 PM

            for leg_i in range(num_legs):
                dep_code  = route[leg_i]
                arr_code  = route[leg_i + 1]
                dep_ap    = next((a for a in AIRPORTS if a["code"] == dep_code),
                                 {"code": dep_code, "city": dep_code})
                arr_ap    = next((a for a in AIRPORTS if a["code"] == arr_code),
                                 {"code": arr_code, "city": arr_code})
                dist      = _get_distance(dep_code, arr_code)
                blk_mins  = _block_mins(dist, aircraft_data["speed_mph"])

                dep_mins_norm = cur_mins % 1440
                arr_mins_raw  = cur_mins + blk_mins
                arr_day       = cur_day + arr_mins_raw // 1440
                arr_mins_norm = arr_mins_raw % 1440

                legs.append(Leg(
                    dep=dep_code,
                    arr=arr_code,
                    dep_city=dep_ap.get("city", dep_code),
                    arr_city=arr_ap.get("city", arr_code),
                    distance_mi=dist,
                    block_mins=blk_mins,
                    dep_day=cur_day,
                    dep_time=_fmt_time(dep_mins_norm),
                    arr_day=arr_day,
                    arr_time=_fmt_time(arr_mins_norm),
                ))

                # Next leg: overnight hotel departure or same-day turnaround
                is_overnight = leg_i < nights_away
                if is_overnight:
                    cur_day += 1
                    # Hotel departure: 6:00–8:30 AM in 5-min slots
                    cur_mins = round(rng.randint(360, 510) / 5) * 5
                else:
                    # Same-day ground turn: B767 needs more time than B737
                    if aircraft_data["type"] == "B767":
                        turn = round(rng.randint(75, 120) / 5) * 5
                    else:
                        turn = round(rng.randint(50, 90) / 5) * 5
                    cur_mins = arr_mins_raw + turn

            pairing = Pairing(
                id=p_idx + 1,
                aircraft=aircraft_data["type"],
                aircraft_label=aircraft_data["label"],
                base=base,
                legs=legs,
                nights_away=nights_away,
                start_dow=start_dow,
                hotel_quality="Standard",
                min_seniority=999,  # all open to all pilots
            )
            pairings.append(pairing)

        return pairings

    # ------------------------------------------------------------------
    # Lines
    # ------------------------------------------------------------------

    def build_lines(
        self,
        pairings: List[Pairing],
        n_lines: int = 5,
        pairings_per_line: int = 5,
    ) -> List[Line]:
        """
        Group pairings into monthly lines.

        Rules:
        - Each pairing belongs to exactly one line.
        - Uses the seeded pairing PRNG for reproducible shuffling.
        - Raises ValueError if n_lines * pairings_per_line > len(pairings).

        Args:
            pairings:          Full list of pairings to distribute.
            n_lines:           Number of lines to produce.
            pairings_per_line: Pairings per line (must divide evenly).

        Returns:
            List of Line objects numbered 1..n_lines.
        """
        needed = n_lines * pairings_per_line
        if needed > len(pairings):
            raise ValueError(
                f"Need {needed} pairings for {n_lines} lines × {pairings_per_line} "
                f"pairings each, but only {len(pairings)} supplied."
            )

        # Sort B737 pairings before B767 pairings so that earlier lines
        # are all-B737 — this guarantees B737-only pilots always have
        # biddable options, which is a prerequisite for any valid allocation.
        b737_pool = [p for p in pairings[:needed] if p.aircraft == "B737"]
        b767_pool = [p for p in pairings[:needed] if p.aircraft != "B737"]

        # Shuffle within each group with the seeded PRNG
        rng = self._prng_pairing
        for pool in (b737_pool, b767_pool):
            for i in range(len(pool) - 1, 0, -1):
                j = int(rng.random() * (i + 1))
                pool[i], pool[j] = pool[j], pool[i]

        # Rebuild sorted pool: B737 first, then B767
        pool = b737_pool + b767_pool

        lines: List[Line] = []
        for line_idx in range(n_lines):
            start = line_idx * pairings_per_line
            line_pairings = pool[start : start + pairings_per_line]
            lines.append(Line(id=line_idx + 1, pairings=line_pairings))

        return lines
