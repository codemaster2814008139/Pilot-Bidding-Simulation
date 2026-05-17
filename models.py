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
class LineScoringResult:
    """
    Aggregated result of scoring a single pilot×line combination over n_runs
    independent LLM calls.

    Stability thresholds (applied to score std across runs):
      std < 5   → "stable"   — mean score used directly
      5–10      → "marginal" — mean score used with a warning
      > 10      → "unstable" — pairwise tiebreak used if adjacent scores overlap

    When ranking_method is "pairwise_tiebreak", a head-to-head comparison was
    run against an adjacent line and the result overrides the score ordering.
    """
    line:           Line
    score_runs:     List[int]       # raw score from each run
    score_mean:     float           # mean across runs
    score_std:      float           # std across runs
    stability:      str             # "stable", "marginal", or "unstable"
    eligible:       bool
    short_reason:   str
    pros:           List[str]
    cons:           List[str]
    ranking_method: str = "score"   # "score" or "pairwise_tiebreak"


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
    line:         Optional["Line"] = None  # set when operating in line-bidding mode


# ---------------------------------------------------------------------------
# Line (monthly schedule — group of pairings)
# ---------------------------------------------------------------------------

@dataclass
class Line:
    """
    A pre-assembled monthly schedule consisting of pairings.
    In real airline bidding the line — not the individual pairing — is the
    atomic unit that pilots bid on.
    """
    id:       int
    pairings: List[Pairing]  # ordered list; all pairings in this line

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    @property
    def total_credit_hours(self) -> float:
        """Sum of credit hours across all pairings."""
        return round(sum(p.credit_hours for p in self.pairings), 1)

    @property
    def total_nights_away(self) -> int:
        """Sum of hotel nights across all pairings."""
        return sum(p.nights_away for p in self.pairings)

    @property
    def total_tafb(self) -> float:
        """Sum of TAFB hours across all pairings."""
        return round(sum(p.tafb for p in self.pairings), 1)

    @property
    def total_block_hours(self) -> float:
        """Sum of block hours across all pairings."""
        return round(sum(p.block_hours for p in self.pairings), 1)

    @property
    def total_per_diem(self) -> int:
        """Sum of per-diem dollars across all pairings."""
        return sum(p.per_diem for p in self.pairings)

    @property
    def aircraft_types(self) -> List[str]:
        """Unique aircraft types used, in order of first appearance."""
        seen: List[str] = []
        for p in self.pairings:
            if p.aircraft not in seen:
                seen.append(p.aircraft)
        return seen

    @property
    def start_days(self) -> List[str]:
        """Start days of week for each pairing."""
        return [p.start_dow for p in self.pairings]

    @property
    def cities(self) -> List[str]:
        """All unique cities visited across all pairings."""
        seen: List[str] = []
        for p in self.pairings:
            for c in p.cities:
                if c not in seen:
                    seen.append(c)
        return seen

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def _pairing_day_span(self, pairing: Pairing) -> tuple:
        """
        Estimate the calendar day range for a pairing within the month.

        Convention: pairing with id=i starts at day (i-1)*6 + 1,
        giving ~6-day spacing between consecutive pairings.
        End day = start + nights_away + 1 (one report day + flying days).
        """
        start = (pairing.id - 1) * 6 + 1
        end   = start + pairing.nights_away + 1
        return start, end

    def has_conflicts(self) -> bool:
        """
        True if any two pairings in this line overlap in calendar time.
        Two pairings A and B overlap when A.start <= B.end AND B.start <= A.end.
        """
        spans = [self._pairing_day_span(p) for p in self.pairings]
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a_start, a_end = spans[i]
                b_start, b_end = spans[j]
                if a_start <= b_end and b_start <= a_end:
                    return True
        return False

    # ------------------------------------------------------------------
    # Pilot-specific helpers
    # ------------------------------------------------------------------

    def total_pay_for_pilot(self, pilot: "Pilot") -> int:
        """Sum of block pay (credit hours × base rate) across all pairings."""
        return sum(p.pay_for_pilot(pilot.base_pay) for p in self.pairings)

    def total_trip_value_for_pilot(self, pilot: "Pilot") -> int:
        """Block pay + per diem across all pairings."""
        return self.total_pay_for_pilot(pilot) + self.total_per_diem

    def is_qualified(self, pilot: "Pilot") -> bool:
        """True if the pilot is qualified for every aircraft type in this line."""
        return all(p.is_qualified(pilot) for p in self.pairings)


# ---------------------------------------------------------------------------
# Ranked line (output of oracle line scoring)
# ---------------------------------------------------------------------------

@dataclass
class RankedLine:
    """Oracle ranking result for a single pilot–line combination."""
    line:         Line
    pilot:        Pilot
    qualified:    bool
    oracle_score: int
    oracle_rank:  int = 0   # assigned after sorting


# ---------------------------------------------------------------------------
# LLM line response (parsed from JSON)
# ---------------------------------------------------------------------------

@dataclass
class LLMLineRank:
    """Parsed LLM response for a single line in oracle line-ranking mode."""
    line_id:      int
    rank:         int
    eligible:     bool
    short_reason: str
    pros:         List[str]
    cons:         List[str]
