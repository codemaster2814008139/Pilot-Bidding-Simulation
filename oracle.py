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
from models import Line, Pairing, Pilot, RankedLine, RankedPairing


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




# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def oracle_score_pairing(pilot: Pilot, pairing: Pairing, pay_score: int = 50) -> int:
    """
    Compute oracle score (0–100) for a pilot-pairing combination.

    Returns 0 if the pilot is not qualified for the aircraft type.
    Ineligible pairings are always ranked last.

    pay_score must be pre-computed by the caller using normalise_pay_scores()
    so that the credit-pay sub-score is meaningful relative to other pairings.
    Defaults to 50 (neutral) when called standalone without context.

    Score = weighted sum of five sub-scores / 100
    """
    if not pairing.is_qualified(pilot):
        return 0

    W = pilot.weights

    tafb_s     = _tafb_score(pairing.tafb)
    hotel_s    = _hotel_nights_score(pairing.nights_away, pilot.has_kids)
    report_s   = _report_time_score(pairing.report_time_mins)
    aircraft_s = _aircraft_score(pairing.aircraft, pilot.preferred_type)

    score = (
        tafb_s     * W.tafb        +
        hotel_s    * W.hotel_nights +
        report_s   * W.report_time  +
        aircraft_s * W.aircraft     +
        pay_score  * W.credit_pay
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
    # Pre-compute normalised pay scores across all pairings for this pilot
    pay_scores = normalise_pay_scores(pilot, pairings)

    scored = []
    for pair in pairings:
        qualified = pair.is_qualified(pilot)
        score     = oracle_score_pairing(pilot, pair, pay_scores.get(pair.id, 50))
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
# Line-level oracle scoring
# ---------------------------------------------------------------------------

# Monthly TAFB bounds: 5 × per-pairing bounds (18h best, 80h worst)
_LINE_TAFB_BEST  = 5 * 18.0   # 90h
_LINE_TAFB_WORST = 5 * 80.0   # 400h

# Expected monthly credit hours used as pay baseline
_EXPECTED_MONTHLY_CREDIT_HOURS = 85.0


def _line_tafb_score(total_tafb: float) -> int:
    """
    Score total monthly TAFB (sum of all 5 pairings).
    Shorter = better.  Linear map: 90h → 100,  400h → 0.
    """
    score = (_LINE_TAFB_WORST - total_tafb) / (_LINE_TAFB_WORST - _LINE_TAFB_BEST) * 100
    return max(0, min(100, round(score)))


def _line_hotel_score(total_nights: int, has_kids: bool) -> int:
    """
    Score total monthly nights away.

    Family pilots have a tight ideal window (8–10 nights).
    Non-family pilots prefer a moderate amount (10–14 nights).
    Outside the ideal range the score falls off linearly.
    """
    if has_kids:
        # Ideal centred on 9 nights; +/-1 per night penalty
        ideal = 9
        score = max(0, 100 - abs(total_nights - ideal) * 15)
    else:
        # Single/mid-career: more nights = more flying (up to a point)
        if total_nights <= 14:
            score = min(100, total_nights * 7)      # 14 nights → 98
        else:
            score = max(0, 100 - (total_nights - 14) * 20)
    return round(score)


def _line_report_score(pairings: List[Pairing]) -> int:
    """
    Score based on average report time across all pairings.
    Reuses the single-pairing report-time sub-scorer.
    """
    if not pairings:
        return 50
    avg = sum(_report_time_score(p.report_time_mins) for p in pairings) / len(pairings)
    return round(avg)


def _line_aircraft_score(pairings: List[Pairing], preferred_type: str) -> int:
    """
    Score = fraction of pairings on the preferred aircraft × 100.
    All preferred → 100; none preferred → 0 (not the same as unqualified).
    """
    if not pairings:
        return 0
    preferred_count = sum(1 for p in pairings if p.aircraft == preferred_type)
    return round(preferred_count / len(pairings) * 100)




def oracle_score_line(pilot: Pilot, line: Line, pay_score: int = 50) -> int:
    """
    Score a monthly line (0–100) for a given pilot.

    The line is evaluated as a whole monthly schedule, not as an average of
    its constituent pairing scores.  Returns 0 if the pilot is not qualified
    for any pairing in the line (ineligible lines always rank last).

    pay_score must be pre-computed by the caller using normalise_line_pay_scores()
    so that the credit-pay sub-score is meaningful relative to other lines.
    Defaults to 50 (neutral) when called standalone without context.

    Sub-scores use the same five weights as pairing scoring (pilot.weights),
    but applied to monthly-level statistics:
      tafb        → total TAFB across all pairings
      hotel_nights → total nights away
      report_time  → average report-time sub-score across all pairings
      aircraft     → fraction of pairings on preferred aircraft type
      credit_pay   → normalised pay rank relative to other lines in the set
    """
    if not line.is_qualified(pilot):
        return 0

    W = pilot.weights

    tafb_s     = _line_tafb_score(line.total_tafb)
    hotel_s    = _line_hotel_score(line.total_nights_away, pilot.has_kids)
    report_s   = _line_report_score(line.pairings)
    aircraft_s = _line_aircraft_score(line.pairings, pilot.preferred_type)

    score = (
        tafb_s     * W.tafb        +
        hotel_s    * W.hotel_nights +
        report_s   * W.report_time  +
        aircraft_s * W.aircraft     +
        pay_score  * W.credit_pay
    ) / 100

    return max(0, min(100, round(score)))


def oracle_rank_lines(pilot: Pilot, lines: List[Line]) -> List[RankedLine]:
    """
    Score and rank all lines for a given pilot.

    Rules:
    - Qualified lines ranked by oracle score (descending).
    - Unqualified lines always ranked last (score = 0).
    - Ties broken by line id (stable sort).

    Returns:
        List of RankedLine sorted best → worst with oracle_rank assigned.
    """
    pay_scores = normalise_line_pay_scores(pilot, lines)

    ranked = []
    for line in lines:
        qualified = line.is_qualified(pilot)
        score     = oracle_score_line(pilot, line, pay_scores.get(line.id, 50))
        ranked.append(RankedLine(
            line=line,
            pilot=pilot,
            qualified=qualified,
            oracle_score=score,
        ))

    ranked.sort(key=lambda r: (0 if r.qualified else 1, -r.oracle_score, r.line.id))

    for i, r in enumerate(ranked):
        r.oracle_rank = i + 1

    return ranked


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


def normalise_line_pay_scores(pilot: Pilot, lines: List[Line]) -> dict:
    """
    Compute pay scores normalised across lines for this pilot.
    Best-paying line → 100, worst → 0, others interpolated.

    Returns dict: line.id → pay_score (0–100)
    """
    pays = {ln.id: ln.total_pay_for_pilot(pilot) for ln in lines}
    min_pay = min(pays.values())
    max_pay = max(pays.values())
    pay_range = max_pay - min_pay

    if pay_range == 0:
        return {lid: 50 for lid in pays}

    return {
        lid: round((pay - min_pay) / pay_range * 100)
        for lid, pay in pays.items()
    }
