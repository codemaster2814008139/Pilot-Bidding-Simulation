"""
prompt_builder.py
-----------------
Builds prompts for the two evaluation modes:
  - oracle_prompt(pilot, pairings)   → LLM ranks all pairings for this pilot
  - pairwise_prompt(pilot, pA, pB)   → LLM picks better of two pairings

Mirrors the JavaScript buildPromptFromPairings() / buildPwPrompt() in the POC.
"""

from typing import List
from models import Pairing, Pilot


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pairing_summary(pilot: Pilot, pair: Pairing) -> str:
    """
    Full pairing description for a prompt.
    Includes per-pilot personalised pay figure.
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

    return (
        f"  Pairing {pair.id} — {pair.num_legs} legs, {pair.nights_away} night(s) away\n"
        f"    Aircraft: {pair.aircraft_label} ({pair.aircraft}) | Qualified: {qualified}\n"
        f"    Route: {'→'.join(l.dep for l in pair.legs)}→{pair.legs[-1].arr}\n"
        f"    Cities: {', '.join(pair.cities)}\n"
        f"{legs_text}\n"
        f"{overnights}\n"
        f"    TAFB: {pair.tafb}h | Block: {pair.block_hours}h | Credit: {pair.credit_hours}h "
        f"| Nights away: {pair.nights_away}\n"
        f"    Report time Day 1: {pair.report_time_str()}\n"
        f"    Starts: {start_dow}\n"
        f"    Your pay: ${pilot_pay:,} ({pair.credit_hours}h × ${pilot.base_pay}/hr) "
        f"| Per diem: ${pair.per_diem} | Total trip value: ${total_value:,}"
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
            "  2. Hotel nights (moderate preference for fewer)\n"
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
            "  5. TAFB and nights away (lower priority at this career stage)"
        )


# ---------------------------------------------------------------------------
# Oracle mode prompt (rank all N pairings)
# ---------------------------------------------------------------------------

def oracle_prompt(pilot: Pilot, pairings: List[Pairing]) -> str:
    """
    Prompt asking the LLM to rank all pairings for a single pilot.
    Returns a string ready to send to any LLM.
    """
    pairing_block = "\n\n".join(_pairing_summary(pilot, p) for p in pairings)
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
    return (
        "You are evaluating which of two flight pairings is better for a specific airline pilot.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        "IMPORTANT: Both pairings listed below are eligible for this pilot to bid.\n"
        "Choose purely based on which pairing better fits this pilot's preference profile.\n\n"
        "== OPTION 1 ==\n"
        f"{_pairing_summary(pilot, pairing_a)}\n\n"
        "== OPTION 2 ==\n"
        f"{_pairing_summary(pilot, pairing_b)}\n\n"
        "== TASK ==\n"
        "Which pairing is better for this pilot? Reply ONLY with JSON (no markdown):\n"
        '{"winner": <1 or 2>, "confidence": "<high|medium|low>", '
        '"reason": "<max 40 words explaining why>"}'
    )


# ---------------------------------------------------------------------------
# Scoring prompt (Scenario 5+ — score a single pairing independently)
# ---------------------------------------------------------------------------

def scoring_prompt(pilot: Pilot, pairing: Pairing) -> str:
    """
    Prompt asking the LLM to score a single pairing for a pilot (0–100).
    Used for the scaled scoring approach (Scenarios 5-7 in the roadmap).

    Each call is independent — the LLM does NOT see other pairings.
    Scores are then sorted externally to produce a ranking.

    Note: validate score consistency before relying on this at scale.
    """
    return (
        "You are evaluating how well a flight pairing fits a specific airline pilot's preferences.\n\n"
        "== PILOT PROFILE ==\n"
        f"{_pilot_profile(pilot)}\n\n"
        "== HOW THIS PILOT PRIORITISES ==\n"
        f"{_priority_list(pilot)}\n\n"
        "== PAIRING TO EVALUATE ==\n"
        f"{_pairing_summary(pilot, pairing)}\n\n"
        "== TASK ==\n"
        "Score this pairing for this pilot on a scale of 0–100, where:\n"
        "  100 = perfectly matches all preferences\n"
        "  50  = neutral / mixed tradeoffs\n"
        "  0   = completely unsuitable (wrong aircraft, extreme TAFB, etc.)\n\n"
        "If the pilot is not qualified for the aircraft type, score must be 0.\n\n"
        "Reply ONLY with JSON (no markdown):\n"
        '{"score": <0-100>, "eligible": <true/false>, '
        '"shortReason": "<max 20 words>", "pros": ["...", "..."], "cons": ["...", "..."]}'
    )
