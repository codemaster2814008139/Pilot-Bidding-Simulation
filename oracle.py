"""
oracle.py
---------
Computes oracle scores and rankings for pilot-pairing combinations.
Mirrors the JavaScript oracleScorePairing() / oracleRankPilot() in the POC.

The oracle converts pairing attributes into 0-100 sub-scores for each
preference factor, then combines them using the pilot's weights.

Important: the oracle is NOT ground truth — it is a deterministic reference
point based on designer-chosen weights and scoring bounds. Agreement between
the LLM and oracle reflects alignment with those weights, not absolute accuracy.
"""

from typing import List, Tuple
from models import Pairing, Pilot, RankedPairing


# ---------------------------------------------------------------------------
# Sub-score functions (each returns 0–100)
# ---------------------------------------------------------------------------

def _tafb_score(tafb_hours: float) -> int:
    """
    Shorter TAFB = better.
    Linear mapping: 80h → 0,  18h → 100.
    Bounds chosen as realistic extremes for 2-4 leg US domestic pairings.
    """
    worst, best = 80.0, 18.0
    score = (worst - tafb_hours) / (worst - best) * 100
    return max(0, min(100, round(score)))


def _hotel_nights_score(nights: int, has_kids: bool) -> int:
    """
    Family pilots prefer fewer nights away; others are less sensitive.
    - has_kids:   0 nights=100, 1=65, 2=30, 3=0  (penalty = 35 per night)
    - no kids:    0 nights=20,  1=60, 2=100 capped (reward = 40 per night)
    """
    if has_kids:
        return max(0, 100 - nights * 35)
    else:
        return min(100, nights * 40 + 20)


def _report_time_score(report_time_mins: int) -> int:
    """
    Later report = better.
    Ideal: report ≥ 7:00 AM (420 min) → 100
    Brutal: report ≤ 4:00 AM (240 min) → 0
    """
    # Normalise to [0, 1440]
    mins = ((report_time_mins % 1440) + 1440) % 1440
    ideal, worst = 420, 240  # minutes from midnight
    if mins >= ideal:
        return 100
    return max(0, round((mins - worst) / (ideal - worst) * 100))


def _aircraft_score(pairing_aircraft: str, preferred_type: str) -> int:
    """
    Binary preference: preferred type → 100, otherwise → 45.
    45 (not 0) reflects that the pilot can still fly — just not their preference.
    """
    return 100 if pairing_aircraft == preferred_type else 45


def _credit_pay_score(credit_hours: float, base_pay: float) -> int:
    """
    Pay score relative to what the pilot would expect at their base rate.
    Anchor: expected = credit_hours × base_pay
    Floor:  85% of expected → score = 0
    Ceil:   125% of expected → score = 100
    (0.85 and 0.4 are arbitrary POC constants — real contracts differ)
    """
    actual_pay   = credit_hours * base_pay
    expected_pay = credit_hours * base_pay   # same rate, so always ~50-60 range
    # More meaningful: compare against other pairings via caller
    # This function just normalises within a plausible range
    floor_pay    = expected_pay * 0.85
    range_pay    = expected_pay * 0.40
    score        = (actual_pay - floor_pay) / range_pay * 100
    return max(0, min(100, round(score)))


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def oracle_score_pairing(pilot: Pilot, pairing: Pairing) -> int:
    """
    Compute oracle score (0–100) for a pilot-pairing combination.

    Returns 0 if the pilot is not qualified for the aircraft type.
    Ineligible pairings are always ranked last.

    Score = weighted sum of five sub-scores / 100
    """
    if not pairing.is_qualified(pilot):
        return 0

    W = pilot.weights

    tafb_s    = _tafb_score(pairing.tafb)
    hotel_s   = _hotel_nights_score(pairing.nights_away, pilot.has_kids)
    report_s  = _report_time_score(pairing.report_time_mins)
    aircraft_s = _aircraft_score(pairing.aircraft, pilot.preferred_type)

    # Credit pay: normalise across pairings for this pilot
    # (simple version — for per-pilot pay comparison see oracle_rank_pilot)
    pay_s = _credit_pay_score(pairing.credit_hours, pilot.base_pay)

    score = (
        tafb_s    * W.tafb        +
        hotel_s   * W.hotel_nights +
        report_s  * W.report_time  +
        aircraft_s * W.aircraft    +
        pay_s     * W.credit_pay
    ) / 100

    return max(0, min(100, round(score)))


def oracle_rank_pilot(pilot: Pilot, pairings: List[Pairing]) -> List[RankedPairing]:
    """
    Score and rank all pairings for a given pilot.

    Rules:
    - Qualified pairings are ranked by oracle score (descending)
    - Unqualified pairings are always ranked last (score = 0)
    - Ties broken by pairing id (stable sort)

    Returns list of RankedPairing sorted best → worst.
    """
    # Score all pairings
    scored = []
    for pair in pairings:
        qualified = pairing.is_qualified(pilot) if (pairing := pair) else False
        score     = oracle_score_pairing(pilot, pair)
        scored.append(RankedPairing(
            pairing=pair,
            pilot=pilot,
            qualified=qualified,
            oracle_score=score,
        ))

    # Sort: qualified by score desc, unqualified last
    scored.sort(key=lambda r: (0 if r.qualified else 1, -r.oracle_score, r.pairing.id))

    # Assign ranks
    for i, r in enumerate(scored):
        r.oracle_rank = i + 1

    return scored


def oracle_rank_all(pilots: List[Pilot], pairings: List[Pairing]) -> dict:
    """
    Compute oracle rankings for all pilots.
    Returns dict: pilot.name → List[RankedPairing]
    """
    return {
        pilot.name: oracle_rank_pilot(pilot, pairings)
        for pilot in pilots
    }


# ---------------------------------------------------------------------------
# Scenario-level normalisation (optional, better pay scoring)
# ---------------------------------------------------------------------------

def normalise_pay_scores(
    pilot: Pilot,
    pairings: List[Pairing]
) -> dict:
    """
    Compute pay scores normalised across pairings for this pilot.
    Best-paying pairing → 100, worst → 0, others interpolated.

    Returns dict: pairing.id → pay_score (0–100)
    """
    pays = {p.id: p.pay_for_pilot(pilot.base_pay) for p in pairings}
    min_pay = min(pays.values())
    max_pay = max(pays.values())
    pay_range = max_pay - min_pay

    if pay_range == 0:
        return {pid: 50 for pid in pays}  # all equal

    return {
        pid: round((pay - min_pay) / pay_range * 100)
        for pid, pay in pays.items()
    }
