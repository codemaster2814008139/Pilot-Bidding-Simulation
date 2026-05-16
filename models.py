"""
models.py
---------
Dataclasses for the pilot bidding simulation.
Mirrors the data structures in the HTML POC.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Oracle weights
# ---------------------------------------------------------------------------

@dataclass
class OracleWeights:
    """
    Preference weights used by the oracle scorer.
    Five factors — must sum to 100.

    Derivation from pilot profile (as in the POC):
      - tafb, hotel_nights: derived from family status and age
      - report_time:        fixed at 12 for all pilots
      - aircraft:           35% of remainder
      - credit_pay:         remainder after all other weights
    """
    tafb:         int = 20
    hotel_nights: int = 18
    report_time:  int = 12
    aircraft:     int = 17   # ~35% of remainder
    credit_pay:   int = 33   # fills to 100

    def total(self) -> int:
        return (self.tafb + self.hotel_nights + self.report_time
                + self.aircraft + self.credit_pay)

    def validate(self) -> bool:
        return self.total() == 100


# ---------------------------------------------------------------------------
# Leg (single flight within a pairing)
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    dep:        str   # IATA code e.g. 'BOS'
    arr:        str
    dep_city:   str
    arr_city:   str
    distance_mi: int
    block_mins: int   # block time in minutes
    dep_day:    int   # day within the pairing (1-indexed)
    dep_time:   str   # human-readable e.g. '9:00 AM'
    arr_day:    int
    arr_time:   str

    @property
    def block_hours(self) -> float:
        return round(self.block_mins / 60, 1)


# ---------------------------------------------------------------------------
# Pairing (multi-leg trip, circular — starts and ends at base)
# ---------------------------------------------------------------------------

@dataclass
class Pairing:
    id:             int
    aircraft:       str    # 'B737' or 'B767'
    aircraft_label: str    # 'Boeing 737' or 'Boeing 767'
    base:           str    # home base airport e.g. 'BOS'
    legs:           List[Leg]
    nights_away:    int
    start_dow:      str    # 'Mon', 'Tue', ...
    hotel_quality:  str = 'Standard'  # fixed per POC decision
    min_seniority:  int = 999         # unused — all open to all pilots

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    @property
    def total_block_mins(self) -> int:
        return sum(l.block_mins for l in self.legs)

    @property
    def block_hours(self) -> float:
        return round(self.total_block_mins / 60, 1)

    @property
    def credit_hours(self) -> float:
        """Credit hours = max(actual block, guaranteed minimum)."""
        guarantee = max(4.0, self.num_legs * 0.5)
        return round(max(self.block_hours, guarantee), 1)

    @property
    def tafb(self) -> float:
        """
        Time Away From Base in hours.
        Computed from first departure to last arrival across all days.
        """
        first_dep = self.legs[0].dep_day * 1440 + self._time_to_mins(self.legs[0].dep_time)
        last_arr  = self.legs[-1].arr_day * 1440 + self._time_to_mins(self.legs[-1].arr_time)
        return round((last_arr - first_dep) / 60, 1)

    @property
    def report_time_mins(self) -> int:
        """Report time = 60 minutes before first departure."""
        return self._time_to_mins(self.legs[0].dep_time) - 60

    @property
    def per_diem(self) -> int:
        """Per diem = TAFB × $2.50, rounded to nearest dollar."""
        return round(self.tafb * 2.5)

    @property
    def cities(self) -> List[str]:
        return list(dict.fromkeys([l.dep_city for l in self.legs] + [self.legs[-1].arr_city]))

    def pay_for_pilot(self, base_pay: float) -> int:
        """Total block pay for a specific pilot's hourly rate."""
        return round(self.credit_hours * base_pay)

    def total_trip_value(self, base_pay: float) -> int:
        """Block pay + per diem."""
        return self.pay_for_pilot(base_pay) + self.per_diem

    def is_qualified(self, pilot: 'Pilot') -> bool:
        return self.aircraft in pilot.qualified_types

    @staticmethod
    def _time_to_mins(time_str: str) -> int:
        """Convert '9:00 AM' → minutes from midnight."""
        parts = time_str.split()
        h, m  = map(int, parts[0].split(':'))
        if parts[1] == 'PM' and h != 12:
            h += 12
        if parts[1] == 'AM' and h == 12:
            h = 0
        return h * 60 + m

    def report_time_str(self) -> str:
        mins = ((self.report_time_mins % 1440) + 1440) % 1440
        h, m = divmod(mins, 60)
        ampm = 'AM' if h < 12 else 'PM'
        h12  = h if 1 <= h <= 12 else (12 if h == 0 else h - 12)
        return f"{h12}:{m:02d} {ampm}"


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------

@dataclass
class Pilot:
    id:              int
    name:            str
    age:             int
    family_status:   str   # e.g. 'Single', 'Married, 2+ kids'
    seniority:       int   # 1 = most senior
    home_base:       str   # e.g. 'BOS'
    qualified_types: List[str]  # e.g. ['B737', 'B767']
    preferred_type:  str
    min_rest:        int   # minimum rest hours required
    base_pay:        float # hourly rate in USD
    weights:         OracleWeights = field(default_factory=OracleWeights)

    @property
    def has_kids(self) -> bool:
        return any(k in self.family_status.lower()
                   for k in ('child', 'kids', 'parent'))

    @property
    def is_mid_career(self) -> bool:
        return self.age >= 40

    def auto_weights(self) -> OracleWeights:
        """
        Derive oracle weights from pilot profile.
        Mirrors the JavaScript buildPilots() logic in the POC.
        """
        hotel_w  = 26 if self.has_kids else (16 if self.is_mid_career else 9)
        tafb_w   = 22 if self.has_kids else (15 if self.is_mid_career else 9)
        report_w = 12  # fixed for all pilots (commute removed)
        rem      = 100 - hotel_w - tafb_w - report_w
        aircraft_w  = round(rem * 0.35)
        credit_w    = 100 - hotel_w - tafb_w - report_w - aircraft_w
        return OracleWeights(
            tafb=tafb_w,
            hotel_nights=hotel_w,
            report_time=report_w,
            aircraft=aircraft_w,
            credit_pay=credit_w
        )

    def can_fly(self, pairing: Pairing) -> bool:
        return pairing.aircraft in self.qualified_types


# ---------------------------------------------------------------------------
# Ranked pairing (output of oracle scoring)
# ---------------------------------------------------------------------------

@dataclass
class RankedPairing:
    pairing:   Pairing
    pilot:     Pilot
    qualified: bool
    oracle_score: int
    oracle_rank:  int = 0   # set after sorting

    @property
    def eligible(self) -> bool:
        return self.qualified


# ---------------------------------------------------------------------------
# LLM response (parsed from JSON)
# ---------------------------------------------------------------------------

@dataclass
class LLMPairingRank:
    pairing_id:   int
    rank:         int
    eligible:     bool
    short_reason: str
    pros:         List[str]
    cons:         List[str]


@dataclass
class ScoredPairing:
    """
    Result of an independent scoring call for one pilot+pairing.
    One LLM API call produces one ScoredPairing.
    Scores are sorted externally to produce a ranking.
    """
    pairing:      Pairing
    score:        int         # 0–100
    eligible:     bool
    short_reason: str
    pros:         List[str]
    cons:         List[str]
    call_idx:     int         # index of the API call (for debugging)


@dataclass
class EvalMetrics:
    spearman:     float
    top1_match:   bool
    elig_accuracy: float  # 0.0–1.0
    overall:      int     # composite for logging


@dataclass
class AllocationResult:
    pilot:        Pilot
    pairing:      Optional[Pairing]
    rank_awarded: int    # which rank choice the pilot got (1=first choice)
    bumped_by:    List[str] = field(default_factory=list)  # names of pilots who took higher choices
