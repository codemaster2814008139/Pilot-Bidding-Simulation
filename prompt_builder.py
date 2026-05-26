"""
prompt_builder.py
-----------------
Builds prompts for the two evaluation modes:
  - oracle_prompt(pilot, pairings)   → LLM ranks all pairings for this pilot
  - pairwise_prompt(pilot, pA, pB)   → LLM picks better of two pairings

Mirrors the JavaScript buildPromptFromPairings() / buildPwPrompt() in the POC.
"""

from typing import Dict, List, Optional, Union
from models import Line, Pairing, Pilot


# ---------------------------------------------------------------------------
# Relative-context helpers
# ---------------------------------------------------------------------------

def _pay_label(pay: float, mean_pay: float) -> str:
    """Describe pay relative to the mean of the current prompt's set."""
    pct = (pay - mean_pay) / mean_pay * 100
    if pct > 15:
        return f"highest in this set, +{round(pct)}% above average"
    elif pct > 5:
        return f"above average, +{round(pct)}%"
    elif pct >= -5:
        return "average for this set"
    elif pct >= -15:
        return f"below average, -{round(abs(pct))}%"
    else:
        return f"lowest in this set, -{round(abs(pct))}% below average"


def _tafb_label(tafb: float, all_tafbs: List[float]) -> str:
    """Label TAFB relative to the full set shown in this prompt."""
    base = f"{tafb}h"
    if len(all_tafbs) <= 1:
        return base
    if tafb == max(all_tafbs):
        return f"{base} (longest in this set)"
    if tafb == min(all_tafbs):
        return f"{base} (shortest in this set)"
    return base


def _nights_label(nights: int, all_nights: List[int]) -> str:
    """Label nights-away relative to the full set shown in this prompt."""
    base = f"{nights} night{'s' if nights != 1 else ''}"
    if len(all_nights) <= 1:
        return base
    if nights == max(all_nights):
        return f"{base} (most in this set)"
    if nights == min(all_nights):
        return f"{base} (fewest in this set)"
    return base


def _pairing_ctx(pilot: Pilot, pairings: List[Pairing]) -> Dict:
    """Pre-compute relative-context stats for a list of pairings."""
    pays = [p.pay_for_pilot(pilot.base_pay) for p in pairings]
    return {
        "mean_pay":   sum(pays) / len(pays),
        "all_tafbs":  [p.tafb for p in pairings],
        "all_nights": [p.nights_away for p in pairings],
    }


def _line_ctx(pilot: Pilot, lines: List[Line]) -> Dict:
    """Pre-compute relative-context stats for a list of lines."""
    pays = [line.total_pay_for_pilot(pilot) for line in lines]
    return {
        "mean_pay":   sum(pays) / len(pays),
        "all_tafbs":  [line.total_tafb for line in lines],
        "all_nights": [line.total_nights_away for line in lines],
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pairing_summary(pilot: Pilot, pair: Pairing, ctx: Optional[Dict] = None) -> str:
    """
    Full pairing description for a prompt.
    Includes per-pilot personalised pay figure.
    When ctx is provided (from _pairing_ctx), pay/TAFB/nights show relative labels.
    """
    legs_text = "\n".join(
        f"      Leg {i+1}: {l.dep} ({l.dep_city}) → {l.arr} ({l.arr_city})\n"
        f"               Departs Day {l.dep_day} at {l.dep_time}, "
        f"arrives Day {l.arr_day} at {l.arr_time}\n"
        f"               Block time: {l.block_hours}h | Distance: {l.distance_mi:,} miles"
        for i, l in enumerate(pair.legs)
    )

    overnight_lines = []
    for i in range(len(pair.legs) - 1):
        if pair.legs[i + 1].dep_day > pair.legs[i].arr_day:
            overnight_lines.append(
                f"      Overnight {len(overnight_lines)+1}: hotel in {pair.legs[i].arr_city} (Standard)"
            )
    overnights = "\n".join(overnight_lines) if overnight_lines else "      (Same-day return, no overnight stays)"

    pilot_pay   = pair.pay_for_pilot(pilot.base_pay)
    total_value = pair.total_trip_value(pilot.base_pay)
    qualified   = "yes" if pair.is_qualified(pilot) else "NO — pilot not qualified for this aircraft"

    dow_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    start_dow = pair.start_dow if pair.start_dow in dow_list else "—"

    if ctx:
        pay_rel    = _pay_label(pilot_pay, ctx["mean_pay"])
        tafb_str   = _tafb_label(pair.tafb, ctx["all_tafbs"])
        nights_str = _nights_label(pair.nights_away, ctx["all_nights"])
        pay_line   = (
            f"    Your pay: ${pilot_pay:,} ({pair.credit_hours}h × ${pilot.base_pay}/hr — {pay_rel}) "
            f"| Per diem: ${pair.per_diem} | Total trip value: ${total_value:,}"
        )
    else:
        tafb_str   = f"{pair.tafb}h"
        nights_str = f"{pair.nights_away}"
        pay_line   = (
            f"    Your pay: ${pilot_pay:,} ({pair.credit_hours}h × ${pilot.base_pay}/hr) "
            f"| Per diem: ${pair.per_diem} | Total trip value: ${total_value:,}"
        )

    return (
        f"  Pairing {pair.id} — {pair.num_legs} legs, {pair.nights_away} night(s) away\n"
        f"    Aircraft: {pair.aircraft_label} ({pair.aircraft}) | Qualified: {qualified}\n"
        f"    Route: {'→'.join(l.dep for l in pair.legs)}→{pair.legs[-1].arr}\n"
        f"    Cities: {', '.join(pair.cities)}\n"
        f"{legs_text}\n"
        f"{overnights}\n"
        f"    TAFB: {tafb_str} | Block: {pair.block_hours}h | Credit: {pair.credit_hours}h "
        f"| Nights away: {nights_str}\n"
        f"    Report time Day 1: {pair.report_time_str()}\n"
        f"    Starts: {start_dow}\n"
        f"{pay_line}"
    )


def _pilot_profile(pilot: Pilot) -> str:
    return (
        f"Name: Capt. {pilot.name}\n"
        f"Age: {pilot.age}\n"
        f"Family status: {pilot.family_status}\n"
        f"Home base: {pilot.home_base}\n"
        f"Aircraft qualifications: {', '.join(pilot.qualified_types)}\n"
        f"Preferred aircraft: {pilot.preferred_type}\n"
        f"Minimum rest required between duties: {pilot.min_rest}h\n"
        f"Base pay rate: ${pilot.base_pay}/hr (credit hours are paid, not just block)"
    )


def _priority_list(pilot: Pilot) -> str:
    if pilot.has_kids:
        return (
            "This pilot has family obligations. They strongly prefer:\n"
            "  1. Shorter TAFB (less time away from home)\n"
            "  2. Fewer hotel nights away\n"
            "  3. Reasonable report times — early reports are disruptive\n"
            "  4. Qualified aircraft type they prefer\n"
            "  5. Higher credit pay and per diem\n"
            "  6. Convenient start day of week"
        )
    elif pilot.is_mid_career:
        return (
            "Mid-career pilot. They balance home-time and compensation:\n"
            "  1. Reasonable TAFB (not too long)\n"
            "  2. Hotel nights away — this pilot prefers trips with overnight stays over quick turns; more nights means more per diem income and richer flying experience\n"
            "  3. Report times — avoids very early reports where possible\n"
            "  4. Preferred aircraft type\n"
            "  5. Credit pay and per diem\n"
            "  6. Start day of week"
        )
    else:
        return (
            "Early-career pilot. They prioritise flying and compensation:\n"
            "  1. Qualified aircraft type (ideally preferred)\n"
            "  2. Higher credit pay and per diem\n"
            "  3. Interesting destinations and routes\n"
            "  4. Report times (some tolerance for early reports)\n"
            "  5. Hotel nights — actively preferred; overnight layovers mean more per diem pay and more hours building experience, which is valuable at this stage of their career"
        )


# ---------------------------------------------------------------------------
# Oracle mode prompt (rank all N pairings)
# ---------------------------------------------------------------------------

def oracle_prompt(pilot: Pilot, pairings: List[Pairing]) -> str:
    """
    Prompt asking the LLM to rank all pairings for a single pilot.
    Returns a string ready to send to any LLM.
    """
    ctx           = _pairing_ctx(pilot, pairings)
    pairing_block = "\n\n".join(_pairing_summary(pilot, p, ctx) for p in pairings)
    n = len(pairings)

    return (
        "You are simulating a commercial airline pilot making monthly schedule bids.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        "== AIRCRAFT QUALIFICATION — HARD CONSTRAINT ==\n"
        f"This pilot is qualified for: {', '.join(pilot.qualified_types)}.\n"
        "Any pairing with an aircraft type NOT in this list CANNOT be bid — mark eligible: false and rank last.\n"
        "Seniority determines bid ORDER only — all otherwise-qualified pairings can be bid.\n\n"
        "== AVAILABLE PAIRINGS ==\n"
        f"{pairing_block}\n\n"
        "== TASK ==\n"
        f"Rank all {n} pairings from best (#1) to worst (#{n}) for THIS pilot only.\n"
        "Reply ONLY with a raw JSON array — no explanation, no markdown, no code fences.\n\n"
        "[\n"
        "  {\n"
        '    "pairingId": <number>,\n'
        '    "rank": <1=best>,\n'
        '    "eligible": <true if pilot is qualified for this aircraft type>,\n'
        '    "shortReason": "<max 25 words explaining this rank>",\n'
        '    "pros": ["<pro 1>", "<pro 2>"],\n'
        '    "cons": ["<con 1>", "<con 2>"]\n'
        "  }\n"
        "]"
    )


# ---------------------------------------------------------------------------
# Pairwise prompt (compare two pairings head-to-head)
# ---------------------------------------------------------------------------

def pairwise_prompt(pilot: Pilot, pairing_a: Pairing, pairing_b: Pairing) -> str:
    """
    Prompt asking the LLM to pick the better of two pairings for a pilot.
    Returns a string ready to send to any LLM.
    """
    ctx = _pairing_ctx(pilot, [pairing_a, pairing_b])
    return (
        "You are evaluating which of two flight pairings is better for a specific airline pilot.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        "IMPORTANT: Both pairings listed below are eligible for this pilot to bid.\n"
        "Choose purely based on which pairing better fits this pilot's preference profile.\n\n"
        "== OPTION 1 ==\n"
        f"{_pairing_summary(pilot, pairing_a, ctx)}\n\n"
        "== OPTION 2 ==\n"
        f"{_pairing_summary(pilot, pairing_b, ctx)}\n\n"
        "== TASK ==\n"
        "Which pairing is better for this pilot? Reply ONLY with JSON (no markdown):\n"
        '{"winner": <1 or 2>, "confidence": "<high|medium|low>", '
        '"reason": "<max 40 words explaining why>"}'
    )


# ---------------------------------------------------------------------------
# Scoring prompt (independent per-pairing scoring)
# ---------------------------------------------------------------------------

def generate_anchor(pilot: Pilot) -> str:
    """
    Generate a calibration anchor string for the independent scoring prompt.

    Values are derived from the pilot's oracle weights and profile so the LLM
    understands what 90-100, 45-55, and 0-15 mean for THIS specific pilot.
    The anchor is embedded in the scoring prompt to reduce inter-call variance.
    """
    w   = pilot.weights
    ac  = pilot.preferred_type
    high_pay = round(pilot.base_pay * 7)
    mid_pay  = round(pilot.base_pay * 5)
    low_pay  = round(pilot.base_pay * 3)

    if pilot.has_kids:
        top_desc = (
            f"0 hotel nights (home each day), TAFB under 30h, "
            f"{ac} aircraft, report after 8am, "
            f"pay above ${high_pay:,}/trip"
        )
        mid_desc = (
            f"1–2 hotel nights, TAFB 40–55h, acceptable aircraft, "
            f"report 5–7am, pay ~${mid_pay:,}/trip"
        )
        low_desc = (
            f"wrong aircraft type (not qualified), TAFB over 70h, "
            f"2+ nights away, report before 5am, pay under ${low_pay:,}/trip"
        )
    elif pilot.is_mid_career:
        top_desc = (
            f"preferred {ac} aircraft, TAFB under 35h, 1 night away, "
            f"report after 7am, total value over ${high_pay:,}/trip"
        )
        mid_desc = (
            f"acceptable aircraft, TAFB 45–60h, 2 nights, "
            f"report 5:30–7am, pay ~${mid_pay:,}/trip"
        )
        low_desc = (
            f"wrong aircraft type, TAFB over 75h, 3+ nights, "
            f"report before 4:30am, low pay"
        )
    else:
        top_desc = (
            f"preferred {ac} aircraft, interesting route, "
            f"high credit pay over ${high_pay:,}/trip, good per diem"
        )
        mid_desc = (
            f"acceptable aircraft, moderate pay ~${mid_pay:,}/trip, "
            f"mixed schedule"
        )
        low_desc = (
            f"not qualified for aircraft, very low pay, "
            f"tedious short hops, report before 4am"
        )

    return (
        f"SCORING ANCHOR — calibrated for Capt. {pilot.name}:\n"
        f"  90–100: {top_desc}\n"
        f"  45–55:  {mid_desc}\n"
        f"  0–15:   {low_desc}\n"
        f"  Weight priorities: TAFB ({w.tafb}%), hotel nights ({w.hotel_nights}%), "
        f"report time ({w.report_time}%), aircraft ({w.aircraft}%), "
        f"credit pay ({w.credit_pay}%)"
    )


def scoring_prompt(pilot: Pilot, pairing: Pairing, anchor: str = "") -> str:
    """
    Prompt asking the LLM to score a single pairing for a pilot (0–100).

    Each call is independent — the LLM does NOT see other pairings.
    Scores are sorted externally to produce a ranking.

    Args:
        pilot:   The pilot being evaluated.
        pairing: The single pairing to score.
        anchor:  Calibration string from generate_anchor(). If empty,
                 a generic 0/50/100 description is used instead.

    Note: validate score consistency with validate_scoring_consistency()
    before relying on rankings derived from these scores at scale.
    """
    anchor_block = anchor if anchor else (
        "  100 = perfectly matches all preferences\n"
        "  50  = neutral / mixed tradeoffs\n"
        "  0   = completely unsuitable (wrong aircraft, extreme TAFB, etc.)"
    )

    return (
        "You are evaluating how well a flight pairing fits a specific airline pilot's preferences.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        "== SCORING CALIBRATION ==\n"
        f"{anchor_block}\n\n"
        "== PAIRING TO EVALUATE ==\n"
        f"{_pairing_summary(pilot, pairing)}\n\n"
        "== TASK ==\n"
        "Score this pairing for this pilot on a scale of 0–100 using the calibration above.\n"
        "If the pilot is not qualified for the aircraft type, score must be 0.\n\n"
        "Reply ONLY with JSON (no markdown):\n"
        '{"score": <0-100>, "eligible": <true/false>, '
        '"shortReason": "<max 20 words>", "pros": ["...", "..."], "cons": ["...", "..."]}'
    )


# ---------------------------------------------------------------------------
# Line-level helpers
# ---------------------------------------------------------------------------

def _line_summary(pilot: Pilot, line: Line, ctx: Optional[Dict] = None) -> str:
    """
    Full description of a monthly line for a prompt.
    Shows aggregate stats then each constituent pairing.
    When ctx is provided (from _line_ctx), pay/TAFB/nights show relative labels.
    Individual pairings within the line always show relative context within the line.
    """
    qualified    = line.is_qualified(pilot)
    qual_label   = "yes — all pairings within qualification" if qualified else "NO — pilot not qualified for one or more pairings"
    conflict_lbl = "⚠ scheduling conflicts detected" if line.has_conflicts() else "no scheduling conflicts"
    ac_str       = " + ".join(line.aircraft_types)
    total_pay    = line.total_pay_for_pilot(pilot)
    total_val    = line.total_trip_value_for_pilot(pilot)

    # Compact per-pairing schedule: one row per pairing showing each leg's
    # departure time and route, with overnight breaks marked.
    sched_rows = []
    for p in line.pairings:
        leg_parts = []
        for i, l in enumerate(p.legs):
            leg_parts.append(f"Day{l.dep_day} {l.dep_time} {l.dep}→{l.arr} arr {l.arr_time}")
            if i < len(p.legs) - 1 and p.legs[i + 1].dep_day > l.arr_day:
                leg_parts.append("(hotel)")
        sched_rows.append(f"    P{p.id} ({p.start_dow}): " + " | ".join(leg_parts))
    schedule_block = "\n".join(sched_rows)

    if ctx:
        pay_rel    = _pay_label(total_pay, ctx["mean_pay"])
        tafb_str   = _tafb_label(line.total_tafb, ctx["all_tafbs"])
        nights_str = _nights_label(line.total_nights_away, ctx["all_nights"])
        pay_detail = f"${total_pay:,} ({line.total_credit_hours}h × ${pilot.base_pay}/hr — {pay_rel})"
    else:
        tafb_str   = f"{line.total_tafb}h"
        nights_str = f"{line.total_nights_away} nights"
        pay_detail = f"${total_pay:,} ({line.total_credit_hours}h × ${pilot.base_pay}/hr)"

    header = (
        f"LINE {line.id} — monthly schedule summary\n"
        f"  Aircraft types used:    {ac_str}\n"
        f"  Qualified for all:      {qual_label}\n"
        f"  Scheduling conflicts:   {conflict_lbl}\n"
        f"  Total TAFB:             {tafb_str} across {len(line.pairings)} pairings\n"
        f"  Total nights away:      {nights_str} this month\n"
        f"  Total block hours:      {line.total_block_hours}h\n"
        f"  Total credit hours:     {line.total_credit_hours}h\n"
        f"  Total block pay:        {pay_detail}\n"
        f"  Total per diem:         ${line.total_per_diem:,}\n"
        f"  Total monthly value:    ${total_val:,}\n"
        f"  Start days of week:     {', '.join(line.start_days)}\n"
        f"  Cities visited:         {', '.join(line.cities)}\n"
        f"  Flight schedule:\n{schedule_block}"
    )

    # Pairings within a line always show relative context against each other
    pairing_ctx = _pairing_ctx(pilot, line.pairings)
    pairing_blocks = "\n\n".join(
        f"  — Pairing {p.id} of Line {line.id} —\n"
        + "\n".join("  " + ln for ln in _pairing_summary(pilot, p, pairing_ctx).splitlines())
        for p in line.pairings
    )

    return header + "\n\n" + pairing_blocks


# ---------------------------------------------------------------------------
# Line-level anchor
# ---------------------------------------------------------------------------

def generate_line_anchor(pilot: Pilot) -> str:
    """
    Generate a calibration anchor string for independent line scoring.

    Describes what a 0, 50, and 100 monthly schedule looks like for this
    specific pilot.  Values are derived from oracle weights and profile.
    """
    w        = pilot.weights
    ac       = pilot.preferred_type
    high_pay = round(pilot.base_pay * 85 * 1.15)   # 15% above expected month
    mid_pay  = round(pilot.base_pay * 85)            # baseline monthly
    low_pay  = round(pilot.base_pay * 85 * 0.80)

    if pilot.has_kids:
        top_desc = (
            f"all {ac} aircraft, total TAFB < 130h, 8–9 nights away, "
            f"all reports after 8am, total monthly value > ${high_pay:,}"
        )
        mid_desc = (
            f"mixed aircraft (some {ac}), total TAFB ~200h, 11–13 nights, "
            f"some early reports, pay ~${mid_pay:,}/month"
        )
        low_desc = (
            f"wrong aircraft throughout, total TAFB > 300h, 15+ nights away, "
            f"multiple 4am reports, pay under ${low_pay:,}/month"
        )
    elif pilot.is_mid_career:
        top_desc = (
            f"mostly {ac} aircraft, TAFB < 150h, 10–12 nights, "
            f"reports after 7am, monthly value > ${high_pay:,}"
        )
        mid_desc = (
            f"mixed aircraft, TAFB ~200h, 12–14 nights, "
            f"some early reports, pay ~${mid_pay:,}/month"
        )
        low_desc = (
            f"wrong aircraft, TAFB > 300h, 15+ nights, "
            f"many early-morning reports, low pay"
        )
    else:
        top_desc = (
            f"preferred {ac} aircraft, high monthly credit pay > ${high_pay:,}, "
            f"varied international routes, reasonable report times"
        )
        mid_desc = (
            f"mixed aircraft, moderate pay ~${mid_pay:,}/month, "
            f"standard domestic schedule"
        )
        low_desc = (
            f"disqualified aircraft, low pay, all very short hops, "
            f"multiple early-morning reports"
        )

    return (
        f"LINE SCORING ANCHOR — calibrated for Capt. {pilot.name}:\n"
        f"  90–100 (ideal month):   {top_desc}\n"
        f"  45–55  (acceptable):    {mid_desc}\n"
        f"  0–15   (unacceptable):  {low_desc}\n"
        f"  Weight priorities: TAFB ({w.tafb}%), hotel nights ({w.hotel_nights}%), "
        f"report time ({w.report_time}%), aircraft ({w.aircraft}%), "
        f"credit pay ({w.credit_pay}%)"
    )


# ---------------------------------------------------------------------------
# Oracle line prompt (rank all N lines)
# ---------------------------------------------------------------------------

def oracle_line_prompt(pilot: Pilot, lines: List[Line]) -> str:
    """
    Prompt asking the LLM to rank all lines for a single pilot.

    Each line is shown with its aggregate monthly statistics and a full
    breakdown of the 5 constituent pairings.  The pilot is choosing their
    entire month's flying, not a single trip.

    Returns a string ready to send to any LLM.
    """
    ctx        = _line_ctx(pilot, lines)
    sep        = "\n\n" + ("=" * 60) + "\n\n"
    line_blocks = sep.join(_line_summary(pilot, ln, ctx) for ln in lines)
    n = len(lines)

    return (
        "You are simulating a commercial airline pilot choosing their monthly schedule.\n\n"
        "Each option below is a COMPLETE MONTHLY LINE OF FLYING — a pre-assembled schedule\n"
        "of multiple pairings representing the pilot's entire month.\n"
        "The pilot bids on lines, NOT individual pairings.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        "== AIRCRAFT QUALIFICATION — HARD CONSTRAINT ==\n"
        f"This pilot is qualified for: {', '.join(pilot.qualified_types)}.\n"
        "Any line containing a pairing with an unqualified aircraft type "
        "CANNOT be bid — mark eligible: false and rank last.\n\n"
        "== AVAILABLE MONTHLY LINES ==\n"
        f"{line_blocks}\n\n"
        "== TASK ==\n"
        f"Rank all {n} monthly lines from best (#1) to worst (#{n}) for THIS pilot only.\n"
        "Consider the line as a complete monthly schedule — tradeoffs across all pairings matter.\n"
        "Reply ONLY with a raw JSON array — no explanation, no markdown, no code fences.\n\n"
        "[\n"
        "  {\n"
        '    "lineId": <number>,\n'
        '    "rank": <1=best>,\n'
        '    "eligible": <true if pilot qualifies for ALL pairings in this line>,\n'
        '    "shortReason": "<max 30 words explaining this rank>",\n'
        '    "pros": ["<pro 1>", "<pro 2>"],\n'
        '    "cons": ["<con 1>", "<con 2>"]\n'
        "  }\n"
        "]"
    )


# ---------------------------------------------------------------------------
# Independent scoring line prompt
# ---------------------------------------------------------------------------

def independent_scoring_line_prompt(
    pilot: Pilot,
    line: Line,
    anchor: str = "",
) -> str:
    """
    Prompt asking the LLM to score a single monthly line (0–100) for a pilot.

    Each call is independent — the LLM does NOT see other lines.
    Scores are sorted externally to produce a ranking.

    Args:
        pilot:  The pilot being evaluated.
        line:   The single monthly line to score.
        anchor: Calibration string from generate_line_anchor(). If empty,
                a generic 0/50/100 description is used instead.
    """
    anchor_block = anchor if anchor else (
        "  100 = perfect month for this pilot\n"
        "  50  = acceptable but mixed tradeoffs\n"
        "  0   = unacceptable (unqualified aircraft, extreme TAFB, etc.)"
    )

    return (
        "You are evaluating how well a monthly flying line fits a specific airline pilot's preferences.\n\n"
        "A LINE is a complete monthly schedule of multiple pairings — "
        "the pilot would fly ALL of these pairings this month.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        "== SCORING CALIBRATION ==\n"
        f"{anchor_block}\n\n"
        "== LINE TO EVALUATE ==\n"
        f"{_line_summary(pilot, line)}\n\n"
        "== TASK ==\n"
        "Score this monthly line for this pilot on a scale of 0–100.\n"
        "If the pilot is not qualified for any pairing in the line, score must be 0.\n\n"
        "Reply ONLY with JSON (no markdown):\n"
        '{"score": <0-100>, "eligible": <true/false>, '
        '"shortReason": "<max 25 words>", "pros": ["...", "..."], "cons": ["...", "..."]}'
    )


# ---------------------------------------------------------------------------
# Pairwise line prompt
# ---------------------------------------------------------------------------

def pairwise_line_prompt(pilot: Pilot, line_a: Line, line_b: Line) -> str:
    """
    Prompt asking the LLM to pick the better of two monthly lines for a pilot.

    Returns a string ready to send to any LLM.
    Output JSON: {"winner": <1 or 2>, "confidence": "...", "reason": "..."}
    """
    ctx = _line_ctx(pilot, [line_a, line_b])
    return (
        "You are evaluating which of two monthly flying lines is better for a specific airline pilot.\n\n"
        "Each option is a COMPLETE MONTHLY SCHEDULE — the pilot would fly ALL pairings in their chosen line.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        "IMPORTANT: Both lines listed below contain only pairings the pilot is qualified for.\n"
        "Choose purely based on which line better fits this pilot's monthly preference profile.\n\n"
        "== OPTION 1 — Monthly Line ==\n"
        f"{_line_summary(pilot, line_a, ctx)}\n\n"
        "== OPTION 2 — Monthly Line ==\n"
        f"{_line_summary(pilot, line_b, ctx)}\n\n"
        "== TASK ==\n"
        "Which monthly line is better for this pilot? Reply ONLY with JSON (no markdown):\n"
        '{"winner": <1 or 2>, "confidence": "<high|medium|low>", '
        '"reason": "<max 50 words explaining the choice>"}'
    )


# ---------------------------------------------------------------------------
# Adaptive pairwise prompt
# ---------------------------------------------------------------------------

def adaptive_pairwise_prompt(
    pilot: Pilot,
    item_a: Union[Line, Pairing],
    item_b: Union[Line, Pairing],
    round_num: int,
    context: str = "",
) -> str:
    """
    Prompt for a single adaptive pairwise comparison.

    Differs from pairwise_prompt / pairwise_line_prompt in two ways:
    1. Includes round_num context ("This is comparison N of approximately M")
    2. Accepts an optional context string giving provisional ranking hints
       for round 2 comparisons — helps the LLM be consistent.

    Output JSON:
    {"winner": <1 or 2>, "confidence": "<high|medium|low>",
     "reason": "<max 40 words>"}
    """
    is_line = isinstance(item_a, Line)

    if is_line:
        ctx = _line_ctx(pilot, [item_a, item_b])
        option1 = _line_summary(pilot, item_a, ctx)
        option2 = _line_summary(pilot, item_b, ctx)
        item_label = "monthly flying line"
        option_label = "MONTHLY LINE"
        task_line = "Which monthly line is better for this pilot?"
    else:
        ctx = _pairing_ctx(pilot, [item_a, item_b])
        option1 = _pairing_summary(pilot, item_a, ctx)
        option2 = _pairing_summary(pilot, item_b, ctx)
        item_label = "flight pairing"
        option_label = "PAIRING"
        task_line = "Which pairing is better for this pilot?"

    round_note = f"[Round {round_num} comparison]"
    context_block = f"\n== CONTEXT FROM EARLIER COMPARISONS ==\n{context}\n" if context else ""

    return (
        f"You are evaluating which of two {item_label}s is better for a specific airline pilot.\n"
        f"{round_note}\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n"
        f"{context_block}\n"
        f"== OPTION 1 — {option_label} ==\n"
        f"{option1}\n\n"
        f"== OPTION 2 — {option_label} ==\n"
        f"{option2}\n\n"
        "== TASK ==\n"
        f"{task_line} Reply ONLY with JSON (no markdown):\n"
        '{"winner": <1 or 2>, "confidence": "<high|medium|low>", '
        '"reason": "<max 40 words>"}'
    )


# ---------------------------------------------------------------------------
# MaxDiff prompt
# ---------------------------------------------------------------------------

def maxdiff_prompt(
    pilot: Pilot,
    items: List[Union[Line, Pairing]],
    item_numbers: List[int],
) -> str:
    """
    MaxDiff (Maximum Difference Scaling) prompt.
    Shows 4-5 items simultaneously and asks for best AND worst.

    Each call gives 2 data points (best and worst) for Bradley-Terry fitting.

    Output JSON:
    {"best": <1-5>, "worst": <1-5>,
     "reason_best": "<20 words>", "reason_worst": "<20 words>"}
    """
    if not items:
        raise ValueError("items must be non-empty")

    is_line = isinstance(items[0], Line)
    n = len(items)

    if is_line:
        ctx = _line_ctx(pilot, items)
        summaries = [_line_summary(pilot, item, ctx) for item in items]
        item_type = "monthly line"
    else:
        ctx = _pairing_ctx(pilot, items)
        summaries = [_pairing_summary(pilot, item, ctx) for item in items]
        item_type = "flight pairing"

    options_block = "\n\n".join(
        f"== OPTION {item_numbers[i]} ==\n{summaries[i]}"
        for i in range(n)
    )
    num_range = f"1–{n}"

    return (
        f"You are evaluating {n} {item_type}s for a specific airline pilot.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        f"{options_block}\n\n"
        "== TASK ==\n"
        f"From the {n} options above, identify the BEST and WORST for this pilot.\n"
        f"Reply ONLY with JSON (no markdown):\n"
        f'{{"best": <{num_range}>, "worst": <{num_range}>, '
        f'"reason_best": "<20 words>", "reason_worst": "<20 words>"}}'
    )
