"""
allocator.py
------------
Seniority-based pairing allocation.
Mirrors the JavaScript runAlloc() logic in the POC.

Two allocation modes:
  - oracle_allocation(pilots, pairings):     uses oracle rankings as preferences
  - llm_allocation(pilots, pairings, llm_rankings): uses LLM rankings as preferences

Rules:
  1. Pilots bid in seniority order (rank #1 = most senior, bids first)
  2. Each pilot is awarded their highest-ranked AVAILABLE pairing they are QUALIFIED for
  3. If a pilot's top choice is taken by a more senior pilot, they get their next choice
  4. Seniority is the tiebreaker — no simultaneous bidding, it's sequential

Note: There are no true "ties" in sequential seniority bidding.
      Seniority resolves all conflicts deterministically.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from models import AllocationResult, Pairing, Pilot, RankedPairing
from oracle import oracle_rank_pilot


# ---------------------------------------------------------------------------
# Core allocation engine
# ---------------------------------------------------------------------------

def _run_allocation(
    pilots: List[Pilot],
    ranked_by_pilot: Dict[int, List],  # pilot.id → ordered list of pairings
) -> List[AllocationResult]:
    """
    Sequential seniority-order allocation.
    ranked_by_pilot values must be ordered lists of pairings (best first).
    Each entry needs .id and .is_qualified(pilot) or a .qualified flag.
    """
    # Sort pilots by seniority (1 = most senior, bids first)
    ordered = sorted(pilots, key=lambda p: p.seniority)
    taken:  Set[int]              = set()
    awarded: Dict[int, Pairing]   = {}   # pilot.id → pairing

    results: List[AllocationResult] = []

    for pilot in ordered:
        ranked = ranked_by_pilot.get(pilot.id, [])
        bumped_by: List[str] = []
        awarded_pairing: Optional[Pairing] = None
        rank_awarded: int = 0

        for rank_idx, pair in enumerate(ranked):
            # Get the actual Pairing object (handle both RankedPairing and Pairing)
            if isinstance(pair, RankedPairing):
                pairing   = pair.pairing
                qualified = pair.qualified
            else:
                pairing   = pair
                qualified = pilot.can_fly(pairing)

            if not qualified:
                continue

            if pairing.id not in taken:
                awarded_pairing = pairing
                awarded[pilot.id] = pairing
                taken.add(pairing.id)
                rank_awarded = rank_idx + 1
                break
            else:
                # Track who took this pairing
                taker = next(
                    (f"Capt. {p.name} (#{p.seniority})"
                     for p in ordered if awarded.get(p.id) and awarded[p.id].id == pairing.id),
                    "another pilot"
                )
                bumped_by.append(f"P{pairing.id} taken by {taker}")

        results.append(AllocationResult(
            pilot=pilot,
            pairing=awarded_pairing,
            rank_awarded=rank_awarded,
            bumped_by=bumped_by,
        ))

    return results


# ---------------------------------------------------------------------------
# Oracle-based allocation
# ---------------------------------------------------------------------------

def oracle_allocation(
    pilots: List[Pilot],
    pairings: List[Pairing],
) -> List[AllocationResult]:
    """
    Allocate pairings using oracle rankings as pilot preferences.
    Always available — does not require LLM responses.
    """
    ranked_by_pilot = {
        pilot.id: oracle_rank_pilot(pilot, pairings)
        for pilot in pilots
    }
    return _run_allocation(pilots, ranked_by_pilot)


# ---------------------------------------------------------------------------
# LLM-based allocation
# ---------------------------------------------------------------------------

def llm_allocation(
    pilots: List[Pilot],
    pairings: List[Pairing],
    llm_rankings: Dict[int, List],  # pilot.id → List[LLMPairingRank]
) -> List[AllocationResult]:
    """
    Allocate pairings using LLM rankings as pilot preferences.
    Requires LLM responses for all pilots.

    Raises ValueError if any pilot is missing LLM rankings.
    """
    missing = [p for p in pilots if p.id not in llm_rankings or not llm_rankings[p.id]]
    if missing:
        names = ", ".join(f"Capt. {p.name}" for p in missing)
        raise ValueError(f"Missing LLM rankings for: {names}")

    pairing_map = {p.id: p for p in pairings}

    # Convert LLMPairingRank lists to ordered Pairing lists
    ranked_by_pilot: Dict[int, List[Pairing]] = {}
    for pilot in pilots:
        llm_ranks = sorted(llm_rankings[pilot.id], key=lambda r: r.rank)
        ordered_pairings = []
        for r in llm_ranks:
            pair = pairing_map.get(r.pairing_id)
            if pair:
                ordered_pairings.append(pair)
        ranked_by_pilot[pilot.id] = ordered_pairings

    return _run_allocation(pilots, ranked_by_pilot)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class AllocationComparison:
    pilot:             Pilot
    oracle_pairing:    Optional[Pairing]
    llm_pairing:       Optional[Pairing]
    same_outcome:      bool
    oracle_rank:       int    # which rank choice oracle got
    llm_rank:          int    # which rank choice LLM got


def compare_allocations(
    oracle_results: List[AllocationResult],
    llm_results:    List[AllocationResult],
) -> List[AllocationComparison]:
    """
    Compare oracle and LLM allocation outcomes side by side.
    """
    oracle_map = {r.pilot.id: r for r in oracle_results}
    llm_map    = {r.pilot.id: r for r in llm_results}

    comparisons = []
    all_pilot_ids = set(oracle_map) | set(llm_map)

    for pid in sorted(all_pilot_ids):
        ora = oracle_map.get(pid)
        llm = llm_map.get(pid)
        if not ora or not llm:
            continue

        ora_pair  = ora.pairing
        llm_pair  = llm.pairing
        same      = (ora_pair and llm_pair and ora_pair.id == llm_pair.id)

        comparisons.append(AllocationComparison(
            pilot=ora.pilot,
            oracle_pairing=ora_pair,
            llm_pairing=llm_pair,
            same_outcome=same,
            oracle_rank=ora.rank_awarded,
            llm_rank=llm.rank_awarded,
        ))

    return comparisons


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_allocation_report(
    results: List[AllocationResult],
    mode: str = "Oracle",
    comparison: Optional[List[AllocationComparison]] = None,
) -> str:
    """
    Human-readable allocation report.
    """
    lines = [f"\n{'='*60}", f"ALLOCATION RESULT — {mode.upper()} MODE", f"{'='*60}"]

    tiebreaks = [r for r in results if r.rank_awarded > 1]

    for r in results:
        pilot = r.pilot
        pair  = r.pairing
        if pair:
            route = "→".join(l.dep for l in pair.legs) + f"→{pair.legs[-1].arr}"
            lines.append(
                f"#{pilot.seniority}  Capt. {pilot.name:<18} "
                f"P{pair.id}: {route:<25} "
                f"({pilot.qualified_types}) | "
                f"rank #{r.rank_awarded} choice | "
                f"{pair.nights_away} nights"
            )
        else:
            lines.append(
                f"#{pilot.seniority}  Capt. {pilot.name:<18} "
                f"{'NO QUALIFIED PAIRING AVAILABLE'}"
            )

    if tiebreaks:
        lines.append(f"\n--- Preference conflicts resolved by seniority ({len(tiebreaks)}) ---")
        for r in tiebreaks:
            lines.append(f"  Capt. {r.pilot.name}: settled for rank #{r.rank_awarded} choice")
            for b in r.bumped_by:
                lines.append(f"    {b}")
    else:
        lines.append("\n✓ No conflicts — every pilot received their top qualified choice.")

    if comparison:
        matches = sum(1 for c in comparison if c.same_outcome)
        total   = len(comparison)
        lines.append(f"\n--- Comparison with Oracle ---")
        lines.append(f"Agreement: {matches}/{total} pilots awarded the same pairing")
        for c in comparison:
            if not c.same_outcome:
                op = f"P{c.oracle_pairing.id}" if c.oracle_pairing else "none"
                lp = f"P{c.llm_pairing.id}"    if c.llm_pairing    else "none"
                lines.append(
                    f"  Capt. {c.pilot.name}: Oracle→{op}, LLM→{lp}"
                )

    lines.append("=" * 60)
    return "\n".join(lines)
