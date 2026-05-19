"""
main.py
-------
End-to-end demonstration of the pilot bidding POC.

Runs a full experiment:
  1. Generate pilots and pairings
  2. Compute oracle rankings
  3. Build and print LLM prompts
  4. (Optional) Call LLM API automatically if key is provided
  5. Parse LLM responses and compute evaluation metrics
  6. Run oracle and LLM-based allocation
  7. Export results to JSON

Usage (manual mode — paste prompts to any LLM):
    python main.py

Usage (automated mode — requires ANTHROPIC_API_KEY or OPENAI_API_KEY):
    ANTHROPIC_API_KEY=sk-... python main.py --auto
    OPENAI_API_KEY=sk-...    python main.py --auto --model gpt-4o
"""

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

from models import LLMLineRank, LLMPairingRank, LineScoringResult, ScoredPairing, Pilot, Pairing, OracleWeights
from generator import ScenarioGenerator
from oracle import oracle_rank_all, oracle_rank_lines, oracle_rank_pilot
from prompt_builder import (
    generate_anchor, generate_line_anchor,
    independent_scoring_line_prompt, oracle_line_prompt, oracle_prompt,
    pairwise_line_prompt, pairwise_prompt, scoring_prompt,
)
from evaluator import (
    evaluate_line_pilot, evaluate_pilot,
    pairwise_agreement, pairwise_summary, evaluate_scoring_stability,
)
from allocator import (
    llm_line_allocation, oracle_allocation, oracle_line_allocation,
    llm_allocation, compare_allocations, format_allocation_report,
)
from llm_api import call_llm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "n_pilots":          3,
    "n_pairings":        100,  # 20 lines × 5 pairings (line mode default)
    "n_lines":           20,
    "pairings_per_line": 5,
    "base":              "BOS",
    "pilot_seed":        1234567,
    "pairing_seed":      42,
    "llm_name":          "manual",
}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_oracle_response(raw: str, n_pairings: int) -> Optional[List[LLMPairingRank]]:
    """Parse LLM JSON response for compare-all-at-once pairing ranking (single prompt, all pairings)."""
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
        return [
            LLMPairingRank(
                pairing_id=int(r["pairingId"]),
                rank=int(r["rank"]),
                eligible=bool(r.get("eligible", True)),
                short_reason=r.get("shortReason", ""),
                pros=r.get("pros", []),
                cons=r.get("cons", []),
            )
            for r in data
        ]
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  ⚠ Parse error: {e}")
        return None


def parse_line_oracle_response(raw: str, n_lines: int) -> Optional[List[LLMLineRank]]:
    """Parse LLM JSON response for compare-all-at-once line ranking (one prompt, all lines)."""
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
        return [
            LLMLineRank(
                line_id=int(r["lineId"]),
                rank=int(r["rank"]),
                eligible=bool(r.get("eligible", True)),
                short_reason=r.get("shortReason", ""),
                pros=r.get("pros", []),
                cons=r.get("cons", []),
            )
            for r in data
        ]
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  ⚠ Parse error (line response): {e}")
        return None


def parse_pairwise_response(raw: str) -> Optional[dict]:
    """Parse LLM JSON response for pairwise comparison."""
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)
        return {
            "winner":     int(data["winner"]),
            "confidence": data.get("confidence", "?"),
            "reason":     data.get("reason", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  ⚠ Parse error: {e}")
        return None


# ---------------------------------------------------------------------------
# Scoring mode helpers
# ---------------------------------------------------------------------------

def parse_line_score_response(raw: str, call_idx: int) -> Optional[dict]:
    """Parse a single independent line scoring LLM response."""
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)
        return {
            "score":       max(0, min(100, int(data["score"]))),
            "eligible":    bool(data.get("eligible", True)),
            "short_reason": data.get("shortReason", ""),
            "pros":        data.get("pros", []),
            "cons":        data.get("cons", []),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  ⚠ Parse error (call {call_idx}): {e}")
        return None


def parse_scoring_response(
    raw: str,
    call_idx: int,
    pairing: Pairing,
) -> Optional[ScoredPairing]:
    """Parse a single independent scoring LLM response into a ScoredPairing."""
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)
        return ScoredPairing(
            pairing=pairing,
            score=max(0, min(100, int(data["score"]))),
            eligible=bool(data.get("eligible", True)),
            short_reason=data.get("shortReason", ""),
            pros=data.get("pros", []),
            cons=data.get("cons", []),
            call_idx=call_idx,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  ⚠ Parse error (call {call_idx}, P{pairing.id}): {e}")
        return None


async def _score_one(
    pilot: Pilot,
    pairing: Pairing,
    call_idx: int,
    provider: str,
    model: str,
    sem: asyncio.Semaphore,
) -> Optional[ScoredPairing]:
    """
    Single async scoring call protected by a semaphore.
    Uses the Anthropic or OpenAI async client depending on provider.
    """
    prompt = scoring_prompt(pilot, pairing, anchor=generate_anchor(pilot))
    async with sem:
        try:
            if provider == "anthropic":
                import anthropic as _anthropic
                client = _anthropic.AsyncAnthropic()
                msg = await client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text
            else:
                import openai as _openai
                client = _openai.AsyncOpenAI()
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content
            return parse_scoring_response(raw, call_idx, pairing)
        except Exception as e:
            print(f"  ⚠ API error (call {call_idx}, P{pairing.id} for {pilot.name}): {e}")
            return None


async def run_independent_scoring(
    pilots: List[Pilot],
    pairings: List[Pairing],
    provider: str,
    model: str,
    batch_size: int = 20,
) -> Dict[int, List[ScoredPairing]]:
    """
    Score all pilot × pairing combinations in parallel batches.

    Uses asyncio + the provider's async client.  At most `batch_size`
    concurrent calls are active at any moment (semaphore-controlled).

    Returns:
        dict: pilot.id → List[ScoredPairing] sorted by score descending.
    """
    sem   = asyncio.Semaphore(batch_size)
    tasks = []
    owner = []   # parallel list: which pilot.id owns each task

    call_idx = 0
    for pilot in pilots:
        for pairing in pairings:
            tasks.append(_score_one(pilot, pairing, call_idx, provider, model, sem))
            owner.append(pilot.id)
            call_idx += 1

    total = len(tasks)
    print(f"  Dispatching {total} scoring calls (≤{batch_size} concurrent)…")
    results_flat = await asyncio.gather(*tasks)

    by_pilot: Dict[int, List[ScoredPairing]] = {p.id: [] for p in pilots}
    for pilot_id, scored in zip(owner, results_flat):
        if scored is not None:
            by_pilot[pilot_id].append(scored)

    for pid in by_pilot:
        by_pilot[pid].sort(key=lambda s: -s.score)

    return by_pilot


async def validate_scoring_consistency(
    pilot: Pilot,
    pairing: Pairing,
    provider: str,
    model: str,
    n_runs: int = 3,
) -> dict:
    """
    Call the scoring prompt n_runs times for the same pilot+pairing.

    Returns stability metrics from evaluate_scoring_stability().
    Prints a warning when std > 10 (scores are unreliable for ranking).
    """
    sem   = asyncio.Semaphore(n_runs)
    tasks = [_score_one(pilot, pairing, i, provider, model, sem) for i in range(n_runs)]
    raw_results = await asyncio.gather(*tasks)
    scores = [r.score for r in raw_results if r is not None]
    stats  = evaluate_scoring_stability(scores)
    if not stats["stable"]:
        print(
            f"  ⚠ UNSTABLE — P{pairing.id} × {pilot.name}: "
            f"std={stats['std']}  mean={stats['mean']}  scores={scores}"
        )
    return stats


async def run_consistency_checks(
    pilots: List[Pilot],
    pairings: List[Pairing],
    provider: str,
    model: str,
    n_samples: int = 3,
) -> None:
    """
    Randomly select n_samples qualified pilot+pairing pairs and validate
    score consistency.  Prints a warning for any unstable pair.
    """
    candidates = [(p, pair) for p in pilots for pair in pairings if p.can_fly(pair)]
    sample     = random.sample(candidates, min(n_samples, len(candidates)))
    print(f"  Consistency check on {len(sample)} pilot+pairing sample(s)…")
    for pilot, pairing in sample:
        stats = await validate_scoring_consistency(pilot, pairing, provider, model)
        marker = "✓ stable" if stats["stable"] else "⚠ UNSTABLE"
        print(
            f"    P{pairing.id} × {pilot.name}: "
            f"mean={stats['mean']}  std={stats['std']}  [{marker}]"
        )


def scored_pairings_to_llm_ranking(
    scored: List[ScoredPairing],
) -> List[LLMPairingRank]:
    """
    Convert a score-sorted List[ScoredPairing] to List[LLMPairingRank].

    Enables the existing evaluate_pilot() / spearman_rho() functions to work
    unchanged on scoring-mode output.  The scored list must already be sorted
    best → worst (highest score = rank 1).
    """
    return [
        LLMPairingRank(
            pairing_id=sp.pairing.id,
            rank=rank,
            eligible=sp.eligible,
            short_reason=sp.short_reason,
            pros=sp.pros,
            cons=sp.cons,
        )
        for rank, sp in enumerate(scored, start=1)
    ]


# ---------------------------------------------------------------------------
# Consistency-checked line scoring (--mode scoring --lines)
# ---------------------------------------------------------------------------

async def _score_one_line(
    pilot: Pilot,
    line,
    call_idx: int,
    provider: str,
    model: str,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    """Single async scoring call for one pilot×line pair."""
    anchor = generate_line_anchor(pilot)
    prompt = independent_scoring_line_prompt(pilot, line, anchor=anchor)
    async with sem:
        try:
            if provider == "anthropic":
                import anthropic as _anthropic
                client = _anthropic.AsyncAnthropic()
                msg = await client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text
            else:
                import openai as _openai
                client = _openai.AsyncOpenAI()
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content
            return parse_line_score_response(raw, call_idx)
        except Exception as e:
            print(f"  ⚠ API error (call {call_idx}, L{line.id} for {pilot.name}): {e}")
            return None


async def _pairwise_one_line(
    pilot: Pilot,
    line_a,
    line_b,
    provider: str,
    model: str,
    sem: asyncio.Semaphore,
) -> Optional[int]:
    """
    Single async pairwise comparison between two lines for one pilot.
    Returns the winning line's id, or None on failure.
    """
    prompt = pairwise_line_prompt(pilot, line_a, line_b)
    async with sem:
        try:
            if provider == "anthropic":
                import anthropic as _anthropic
                client = _anthropic.AsyncAnthropic()
                msg = await client.messages.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text
            else:
                import openai as _openai
                client = _openai.AsyncOpenAI()
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content
            clean  = raw.strip().replace("```json", "").replace("```", "").strip()
            data   = json.loads(clean)
            winner = int(data["winner"])
            return line_a.id if winner == 1 else line_b.id
        except Exception as e:
            print(f"  ⚠ Pairwise error (L{line_a.id} vs L{line_b.id} for {pilot.name}): {e}")
            return None


async def run_scored_lines(
    pilots: List[Pilot],
    lines: List,
    provider: str,
    model: str,
    n_runs: int = 2,
    batch_size: int = 20,
) -> Dict[int, List[LineScoringResult]]:
    """
    Score all pilot × line combinations with n_runs independent calls each.

    For each pilot:
      1. Score each line n_runs times; compute mean/std/stability.
      2. Preliminary sort: ineligible last, then by mean score descending.
      3. For each consecutive eligible pair where either member is UNSTABLE
         and their score intervals overlap, run a pairwise comparison to
         resolve the ordering.
      4. Set ranking_method = "pairwise_tiebreak" on involved lines.
      5. Print per-pilot stability summary and a warning if > 30% unstable.

    Returns:
        dict: pilot.id → List[LineScoringResult] in final ranked order.
    """
    sem = asyncio.Semaphore(batch_size)

    # Dispatch all n_runs scoring calls for every pilot × line
    tasks:     List = []
    task_keys: List = []   # (pilot.id, line.id)
    call_idx = 0
    for pilot in pilots:
        for line in lines:
            for _ in range(n_runs):
                tasks.append(_score_one_line(pilot, line, call_idx, provider, model, sem))
                task_keys.append((pilot.id, line.id))
                call_idx += 1

    total = len(tasks)
    print(
        f"  Dispatching {total} scoring call(s) "
        f"({n_runs} run(s) × {len(pilots)} pilot(s) × {len(lines)} line(s), "
        f"≤{batch_size} concurrent)…"
    )
    raw_results = await asyncio.gather(*tasks)

    # Group raw results by (pilot.id, line.id)
    run_buckets: Dict = {}
    for key, result in zip(task_keys, raw_results):
        if result is not None:
            run_buckets.setdefault(key, []).append(result)

    by_pilot: Dict[int, List[LineScoringResult]] = {p.id: [] for p in pilots}
    global_stable = global_marginal = global_unstable = global_pairwise = 0

    for pilot in pilots:
        pilot_results: List[LineScoringResult] = []

        for line in lines:
            runs = run_buckets.get((pilot.id, line.id), [])
            if not runs:
                continue
            scores = [r["score"] for r in runs]
            stats  = evaluate_scoring_stability(scores)
            last   = runs[-1]
            pilot_results.append(LineScoringResult(
                line=line,
                score_runs=scores,
                score_mean=stats["mean"],
                score_std=stats["std"],
                stability=stats["stability"],
                eligible=last["eligible"],
                short_reason=last["short_reason"],
                pros=last["pros"],
                cons=last["cons"],
            ))

        # Preliminary sort: ineligible last, then by mean score desc
        pilot_results.sort(
            key=lambda r: (0 if r.eligible else 1, -r.score_mean, r.line.id)
        )

        # Find consecutive eligible pairs where either is UNSTABLE and
        # their score intervals [mean-std, mean+std] overlap
        pairwise_tasks: List = []
        pairwise_indices: List = []
        for i in range(len(pilot_results) - 1):
            a, b = pilot_results[i], pilot_results[i + 1]
            if not a.eligible or not b.eligible:
                continue
            if a.stability == "unstable" or b.stability == "unstable":
                if abs(a.score_mean - b.score_mean) < a.score_std + b.score_std:
                    pairwise_tasks.append(
                        _pairwise_one_line(pilot, a.line, b.line, provider, model, sem)
                    )
                    pairwise_indices.append((i, i + 1))

        if pairwise_tasks:
            print(
                f"  Capt. {pilot.name}: running {len(pairwise_tasks)} pairwise "
                f"tiebreak(s) for overlapping unstable scores…"
            )
            pw_results = await asyncio.gather(*pairwise_tasks)
            n_resolved = 0
            for (i, j), winner_id in zip(pairwise_indices, pw_results):
                if winner_id is None:
                    continue
                a, b = pilot_results[i], pilot_results[j]
                a.ranking_method = "pairwise_tiebreak"
                b.ranking_method = "pairwise_tiebreak"
                if winner_id == b.line.id:   # pairwise disagrees with score order — swap
                    pilot_results[i], pilot_results[j] = pilot_results[j], pilot_results[i]
                n_resolved += 1
            global_pairwise += n_resolved

        # Tally per-pilot stability
        n_stable   = sum(1 for r in pilot_results if r.stability == "stable")
        n_marginal = sum(1 for r in pilot_results if r.stability == "marginal")
        n_unstable = sum(1 for r in pilot_results if r.stability == "unstable")
        n_total    = len(pilot_results)

        global_stable   += n_stable
        global_marginal += n_marginal
        global_unstable += n_unstable

        if n_total > 0 and n_unstable / n_total > 0.30:
            print(
                f"  ⚠ WARNING: scoring is unreliable for Capt. {pilot.name}. "
                f"Consider switching to Swiss tournament mode for this pilot."
            )

        by_pilot[pilot.id] = pilot_results

    # Aggregate summary
    global_total = global_stable + global_marginal + global_unstable
    print(
        f"\n  Scoring stability: {global_stable}/{global_total} stable, "
        f"{global_marginal} marginal, {global_unstable} unstable"
    )
    if global_pairwise > 0:
        print(f"  {global_pairwise} unstable pair(s) resolved via pairwise tiebreak")

    return by_pilot


def scored_lines_to_llm_line_rankings(
    scored_by_pilot: Dict[int, List[LineScoringResult]],
) -> Dict[int, List[LLMLineRank]]:
    """
    Convert sorted LineScoringResult lists to LLMLineRank lists.
    Enables evaluate_line_pilot() and llm_line_allocation() to work unchanged.
    """
    return {
        pid: [
            LLMLineRank(
                line_id=r.line.id,
                rank=rank,
                eligible=r.eligible,
                short_reason=r.short_reason,
                pros=r.pros,
                cons=r.cons,
            )
            for rank, r in enumerate(results, start=1)
        ]
        for pid, results in scored_by_pilot.items()
    }


# ---------------------------------------------------------------------------
# Manual mode helpers
# ---------------------------------------------------------------------------

def manual_oracle_run(
    pilots: List[Pilot],
    pairings: List[Pairing],
) -> Dict[int, List[LLMPairingRank]]:
    """
    Print rank-all prompts and collect LLM responses manually (paste from any LLM).
    Returns dict: pilot.id → list of LLMPairingRank
    """
    results: Dict[int, List[LLMPairingRank]] = {}

    for pilot in pilots:
        prompt = oracle_prompt(pilot, pairings)
        print(f"\n{'='*60}")
        print(f"PROMPT FOR: Capt. {pilot.name} (Seniority #{pilot.seniority})")
        print(f"{'='*60}")
        print(prompt)
        print(f"\n{'='*60}")
        print("Paste LLM response below (JSON array), then press Enter twice:")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        raw = "\n".join(lines).strip()
        parsed = parse_oracle_response(raw, len(pairings))
        if parsed:
            results[pilot.id] = parsed
            print(f"  ✓ Parsed {len(parsed)} rankings for Capt. {pilot.name}")
        else:
            print(f"  ✗ Failed to parse response for Capt. {pilot.name}")

    return results


# ---------------------------------------------------------------------------
# Automated mode
# ---------------------------------------------------------------------------

def auto_oracle_run(
    pilots: List[Pilot],
    pairings: List[Pairing],
    model: str,
    provider: str,
) -> Dict[int, List[LLMPairingRank]]:
    """
    Automatically call LLM API for all pilots (compare-all-at-once rank-all prompt).
    """
    results: Dict[int, List[LLMPairingRank]] = {}
    for pilot in pilots:
        prompt = oracle_prompt(pilot, pairings)
        print(f"  Calling {provider}/{model} for Capt. {pilot.name}...", end=" ")
        try:
            raw    = call_llm(prompt, model, provider)
            parsed = parse_oracle_response(raw, len(pairings))
            if parsed:
                results[pilot.id] = parsed
                print("✓")
            else:
                print("✗ parse error")
        except Exception as e:
            print(f"✗ API error: {e}")
    return results


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_results(
    pilots: List[Pilot],
    pairings: List[Pairing],
    oracle_rankings: dict,
    llm_rankings: Dict[int, List[LLMPairingRank]],
    eval_metrics: dict,
    oracle_alloc: list,
    llm_alloc: Optional[list],
    llm_name: str,
    config: dict,
    scored_lines_by_pilot: Optional[dict] = None,
) -> dict:
    """Build full results dict for JSON export."""

    def pairing_to_dict(p: Pairing) -> dict:
        return {
            "id": p.id,
            "aircraft": p.aircraft,
            "aircraft_label": p.aircraft_label,
            "base": p.base,
            "num_legs": p.num_legs,
            "nights_away": p.nights_away,
            "start_dow": p.start_dow,
            "tafb_h": p.tafb,
            "block_hours": p.block_hours,
            "credit_hours": p.credit_hours,
            "per_diem": p.per_diem,
            "report_time": p.report_time_str(),
            "legs": [
                {
                    "dep": l.dep, "arr": l.arr,
                    "dep_city": l.dep_city, "arr_city": l.arr_city,
                    "distance_mi": l.distance_mi,
                    "block_mins": l.block_mins,
                    "dep_day": l.dep_day, "dep_time": l.dep_time,
                    "arr_day": l.arr_day, "arr_time": l.arr_time,
                }
                for l in p.legs
            ],
        }

    def pilot_to_dict(p: Pilot) -> dict:
        m = eval_metrics.get(p.id)
        llm = llm_rankings.get(p.id)
        return {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "family_status": p.family_status,
            "seniority": p.seniority,
            "home_base": p.home_base,
            "qualified_types": p.qualified_types,
            "preferred_type": p.preferred_type,
            "min_rest": p.min_rest,
            "base_pay": p.base_pay,
            "oracle_weights": {
                "tafb": p.weights.tafb,
                "hotel_nights": p.weights.hotel_nights,
                "report_time": p.weights.report_time,
                "aircraft": p.weights.aircraft,
                "credit_pay": p.weights.credit_pay,
            },
            "eval_metrics": {
                "spearman": m.spearman if m else None,
                "top1_match": m.top1_match if m else None,
                "elig_accuracy": m.elig_accuracy if m else None,
            } if m else None,
            "llm_ranking": [
                {
                    "pairingId": r.pairing_id,
                    "rank": r.rank,
                    "eligible": r.eligible,
                    "shortReason": r.short_reason,
                    "pros": r.pros,
                    "cons": r.cons,
                }
                for r in llm
            ] if llm else None,
        }

    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "config": config,
        "llm": llm_name,
        "scenario": {
            "pilots":   [pilot_to_dict(p) for p in pilots],
            "pairings": [pairing_to_dict(p) for p in pairings],
            "oracle_rankings": {
                pilot.name: [
                    {
                        "pairing_id": r.pairing.id,
                        "oracle_rank": r.oracle_rank,
                        "oracle_score": r.oracle_score,
                        "qualified": r.qualified,
                    }
                    for r in ranked
                ]
                for pilot, ranked in zip(pilots, oracle_rankings.values())
            },
        },
        "allocation": {
            "oracle": [
                {
                    "pilot": r.pilot.name,
                    "seniority": r.pilot.seniority,
                    "pairing": r.pairing.id if r.pairing else None,
                    "rank_awarded": r.rank_awarded,
                    "bumped_by": r.bumped_by,
                }
                for r in oracle_alloc
            ],
            "llm": [
                {
                    "pilot": r.pilot.name,
                    "seniority": r.pilot.seniority,
                    "pairing": r.pairing.id if r.pairing else None,
                    "rank_awarded": r.rank_awarded,
                    "bumped_by": r.bumped_by,
                }
                for r in llm_alloc
            ] if llm_alloc else None,
        },
        "meta": {
            "tool": "Pilot Bidding POC — Python",
            "version": "1.0",
            "base_airport": config["base"],
            "pay_method": "credit_hours × pilot_base_pay + per_diem",
        },
        "line_scoring": (
            {
                pilot.name: [
                    {
                        "line_id":        r.line.id,
                        "score_mean":     r.score_mean,
                        "score_runs":     r.score_runs,
                        "score_std":      r.score_std,
                        "stability":      r.stability,
                        "ranking_method": r.ranking_method,
                        "eligible":       r.eligible,
                        "short_reason":   r.short_reason,
                    }
                    for r in scored_lines_by_pilot.get(pilot.id, [])
                ]
                for pilot in pilots
            }
            if scored_lines_by_pilot
            else None
        ),
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_results_html(
    pilots: List[Pilot],
    pairings: List[Pairing],
    oracle_rankings: dict,
    llm_rankings: dict,
    llm_line_rankings: dict,
    oracle_line_rankings: dict,
    eval_metrics: dict,
    oracle_alloc: list,
    llm_alloc: Optional[list],
    llm_name: str,
    config: dict,
    schedule_lines: Optional[list] = None,
    scored_lines_by_pilot: Optional[dict] = None,
) -> str:
    line_mode = bool(schedule_lines)
    line_map  = {ln.id: ln for ln in (schedule_lines or [])}
    ts        = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    oracle_alloc_map = {r.pilot.id: r for r in oracle_alloc}
    llm_alloc_map    = {r.pilot.id: r for r in llm_alloc} if llm_alloc else {}

    # oracle_line_rankings is keyed by pilot.name; convert to pilot.id
    pilot_by_name = {p.name: p for p in pilots}
    ora_line_by_pid: Dict[int, list] = {
        pilot_by_name[name].id: ranked
        for name, ranked in oracle_line_rankings.items()
        if name in pilot_by_name
    }

    def esc(s) -> str:
        return (str(s)
                .replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    # ── CSS ──────────────────────────────────────────────────────────────────
    css = """<style>
body{font-family:system-ui,-apple-system,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1.5rem;color:#222;line-height:1.5}
h1{border-bottom:3px solid #2c3e50;padding-bottom:.4rem;margin-bottom:.3rem}
h2{color:#2c3e50;margin-top:2.5rem;margin-bottom:.5rem}
.meta{color:#555;font-size:.9rem;margin-bottom:2rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.87rem}
th{background:#2c3e50;color:#fff;padding:.5rem .8rem;text-align:left;white-space:nowrap}
td{padding:.38rem .8rem;border-bottom:1px solid #e5e5e5;vertical-align:middle}
tbody tr:hover td{background:#f4f8fb}
.r-both td{background:#bbf7d0!important;font-weight:600}
.r-llm td{background:#dcfce7!important;font-weight:600}
.r-ora td{background:#dbeafe!important}
.first{color:#16a34a;font-weight:700}
.nth{color:#d97706}
.inelig{color:#bbb;text-decoration:line-through}
.m-yes{color:#16a34a}
.m-no{color:#dc2626}
details{border:1px solid #d1d5db;border-radius:6px;margin:.35rem 0}
summary{padding:.65rem 1.1rem;cursor:pointer;background:#f9fafb;font-weight:600;
        border-radius:6px;list-style:none;display:flex;align-items:center;gap:.5rem}
summary::-webkit-details-marker{display:none}
summary::before{content:"▶";font-size:.65rem;transition:transform .15s;flex-shrink:0}
details[open]>summary::before{transform:rotate(90deg)}
details[open]>summary{border-radius:6px 6px 0 0;border-bottom:1px solid #d1d5db}
.inner{padding:.9rem 1.1rem;overflow-x:auto}
.leg{font-size:.8rem;color:#555;margin:.15rem 0 .15rem 1rem}
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:3px;vertical-align:middle}
</style>"""

    # ── Allocation summary table ──────────────────────────────────────────────

    def rank_badge(rank: int) -> str:
        if rank == 1:
            return '<span class="first">★ #1</span>'
        return f'<span class="nth">#{rank}</span>' if rank else "—"

    def summary_rows() -> str:
        rows = []
        for p in sorted(pilots, key=lambda x: x.seniority):
            ora_r = oracle_alloc_map.get(p.id)
            llm_r = llm_alloc_map.get(p.id)
            ora_item = (ora_r.line if (ora_r and ora_r.line) else
                        (ora_r.pairing if ora_r else None))
            llm_item = (llm_r.line if (llm_r and llm_r.line) else
                        (llm_r.pairing if llm_r else None))

            ora_id   = f"L{ora_item.id}" if ora_item else "—"
            llm_id   = f"L{llm_item.id}" if llm_item else ("—" if llm_alloc else "<em>N/A</em>")
            ora_rank = rank_badge(ora_r.rank_awarded if ora_r else 0)
            llm_rank = rank_badge(llm_r.rank_awarded if llm_r else 0) if llm_alloc else "<em>N/A</em>"

            same = ora_item and llm_item and ora_item.id == llm_item.id
            match = ('<span class="m-yes">✓ same</span>' if same else
                     '<span class="m-no">✗ diff</span>' if (ora_item and llm_item) else "—")

            if line_mode and ora_item:
                details_str = (f"{ora_item.total_credit_hours}h credit &nbsp;·&nbsp; "
                               f"{ora_item.total_nights_away} nights &nbsp;·&nbsp; "
                               f"{' / '.join(ora_item.aircraft_types)}")
            elif not line_mode and ora_item:
                route = "→".join(l.dep for l in ora_item.legs) + f"→{ora_item.legs[-1].arr}"
                details_str = f"{ora_item.credit_hours}h &nbsp;·&nbsp; {ora_item.nights_away} nights &nbsp;·&nbsp; {route}"
            else:
                details_str = "—"

            # bumped info
            bumped = ""
            if ora_r and ora_r.bumped_by:
                bumped = f'<br><small style="color:#888">bumped from: {esc(", ".join(ora_r.bumped_by))}</small>'

            rows.append(
                f"<tr>"
                f"<td>#{p.seniority}</td>"
                f"<td><strong>{esc(p.name)}</strong><br>"
                f"<small>{esc('/'.join(p.qualified_types))}, pref: {esc(p.preferred_type)}</small></td>"
                f"<td>{ora_id}{bumped}</td>"
                f"<td>{ora_rank}</td>"
                f"<td>{llm_id}</td>"
                f"<td>{llm_rank}</td>"
                f"<td>{match}</td>"
                f"<td>{details_str}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    summary_html = f"""
<h2>Allocation Summary</h2>
<table>
  <thead><tr>
    <th>Snr</th><th>Pilot</th>
    <th>Oracle — Assigned</th><th>Oracle Rank</th>
    <th>LLM — Assigned</th><th>LLM Rank</th>
    <th>Match?</th><th>Details</th>
  </tr></thead>
  <tbody>{summary_rows()}</tbody>
</table>"""

    # ── Per-pilot ranking toggles (line mode only) ────────────────────────────

    def pilot_toggle(pilot: Pilot) -> str:
        ora_r        = oracle_alloc_map.get(pilot.id)
        llm_r        = llm_alloc_map.get(pilot.id)
        ora_assigned = (ora_r.line.id if (ora_r and ora_r.line) else
                        (ora_r.pairing.id if (ora_r and ora_r.pairing) else None))
        llm_assigned = (llm_r.line.id if (llm_r and llm_r.line) else
                        (llm_r.pairing.id if (llm_r and llm_r.pairing) else None))

        # Build per-line lookup dicts
        ora_ranked   = ora_line_by_pid.get(pilot.id, [])
        ora_rank_map  = {r.line.id: r.oracle_rank  for r in ora_ranked}
        ora_score_map = {r.line.id: r.oracle_score for r in ora_ranked}

        llm_ranked    = llm_line_rankings.get(pilot.id, [])
        llm_rank_map  = {r.line_id: r.rank         for r in llm_ranked}
        llm_elig_map  = {r.line_id: r.eligible     for r in llm_ranked}
        llm_reason_map = {r.line_id: r.short_reason for r in llm_ranked}

        scored_map: dict = {}
        if scored_lines_by_pilot:
            for sr in scored_lines_by_pilot.get(pilot.id, []):
                scored_map[sr.line.id] = sr

        # Order rows by LLM rank, then oracle rank, then line id
        if llm_ranked:
            ordered_ids = [r.line_id for r in sorted(llm_ranked, key=lambda x: x.rank)]
        elif ora_ranked:
            ordered_ids = [r.line.id for r in sorted(ora_ranked, key=lambda x: x.oracle_rank)]
        else:
            ordered_ids = sorted(line_map.keys())

        rows = []
        for lid in ordered_ids:
            ln = line_map.get(lid)
            if not ln:
                continue

            llm_rank  = llm_rank_map.get(lid)
            ora_rank  = ora_rank_map.get(lid)
            eligible  = llm_elig_map.get(lid, True)
            reason    = esc(llm_reason_map.get(lid, ""))

            is_ora = (lid == ora_assigned)
            is_llm = (lid == llm_assigned)
            row_cls = ("r-both" if (is_ora and is_llm) else
                       "r-llm"  if is_llm else
                       "r-ora"  if is_ora else "")

            llm_cell = (rank_badge(llm_rank) if llm_rank is not None else "—")
            ora_cell = (rank_badge(ora_rank) if ora_rank is not None else "—")

            tags = []
            if is_llm and is_ora:
                tags.append("← LLM &amp; Oracle assigned")
            elif is_llm:
                tags.append("← LLM assigned")
            elif is_ora:
                tags.append("← Oracle assigned")
            tag_str = f' &nbsp;<small style="color:#555">{" ".join(tags)}</small>' if tags else ""

            # Score column (scoring mode)
            score_cell = ""
            if scored_map and lid in scored_map:
                sr = scored_map[lid]
                icon = {"stable": "✓", "marginal": "~", "unstable": "⚠"}.get(sr.stability, "")
                score_cell = f"<td>{sr.score_mean:.0f} ± {sr.score_std:.1f} {icon}</td>"
            elif scored_lines_by_pilot:
                score_cell = "<td>—</td>"

            inelig_style = ' style="color:#bbb;text-decoration:line-through"' if not eligible else ""

            # Compact pairing list for the line
            pairing_ids = "  ".join(f"P{p.id}({p.start_dow})" for p in ln.pairings)

            rows.append(
                f'<tr class="{row_cls}">'
                f"<td>{llm_cell}</td>"
                f"<td>{ora_cell}</td>"
                f'<td{inelig_style}><strong>L{lid}</strong>{tag_str}</td>'
                f"<td>{ln.total_credit_hours}h</td>"
                f"<td>{ln.total_nights_away}</td>"
                f"<td>{esc(' / '.join(ln.aircraft_types))}</td>"
                f"<td>{reason}</td>"
                f"<td><small>{esc(pairing_ids)}</small></td>"
                f"{score_cell}"
                f"</tr>"
            )

        m = eval_metrics.get(pilot.id)
        eval_str = ""
        if m:
            top1 = "✓" if m.top1_match else "✗"
            eval_str = (f"&nbsp;·&nbsp; ρ={m.spearman} &nbsp; top-1={top1} &nbsp;"
                        f" elig={m.elig_accuracy:.0%}")

        score_th = "<th>Score (mean±std)</th>" if scored_lines_by_pilot else ""
        score_col_html = score_th

        return f"""<details>
<summary>
  <span>#{pilot.seniority} &nbsp; Capt. {esc(pilot.name)}</span>
  <span style="font-weight:400;color:#555">&nbsp;·&nbsp; {esc('/'.join(pilot.qualified_types))}, pref: {esc(pilot.preferred_type)}{eval_str}</span>
</summary>
<div class="inner">
<p style="font-size:.82rem;color:#555;margin:.2rem 0 .6rem">
  <span class="swatch" style="background:#bbf7d0"></span>Both oracle &amp; LLM assigned &nbsp;
  <span class="swatch" style="background:#dcfce7"></span>LLM assigned &nbsp;
  <span class="swatch" style="background:#dbeafe"></span>Oracle assigned
</p>
<table>
  <thead><tr>
    <th>LLM Rank</th><th>Oracle Rank</th><th>Line</th>
    <th>Credit Hrs</th><th>Nights Away</th><th>Aircraft</th>
    <th>LLM Reason</th><th>Pairings</th>{score_col_html}
  </tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>
</details>"""

    pilot_toggles = ""
    if line_mode:
        pilot_toggles = (
            "<h2>Per-Pilot Line Rankings</h2>"
            "<p style='color:#555;font-size:.9rem;margin-bottom:.8rem'>"
            "Click a pilot to expand their full ranked list of lines. "
            "Rows are ordered by LLM preference rank (best first).</p>"
            + "\n".join(
                pilot_toggle(p)
                for p in sorted(pilots, key=lambda x: x.seniority)
            )
        )

    # ── Scenario header ───────────────────────────────────────────────────────
    if line_mode:
        scenario = (
            f"{config['n_pilots']} pilots &nbsp;·&nbsp; "
            f"{config.get('n_lines','?')} lines × {config.get('pairings_per_line','?')} pairings/line"
            f" &nbsp;·&nbsp; Base: {config['base']} &nbsp;·&nbsp; Mode: line"
        )
    else:
        scenario = (
            f"{config['n_pilots']} pilots &nbsp;·&nbsp; "
            f"{config['n_pairings']} pairings"
            f" &nbsp;·&nbsp; Base: {config['base']} &nbsp;·&nbsp; Mode: pairing"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pilot Bidding Results</title>
{css}
</head>
<body>
<h1>Pilot Bidding Results</h1>
<div class="meta">
  <strong>LLM:</strong> {esc(llm_name)} &nbsp;&nbsp;
  <strong>Exported:</strong> {ts} &nbsp;&nbsp;
  <strong>Scenario:</strong> {scenario}
</div>
{summary_html}
{pilot_toggles}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pilot Bidding POC")
    parser.add_argument("--auto",     action="store_true",   help="Auto-call LLM API")
    parser.add_argument("--provider", default="openai",      help="anthropic or openai")
    parser.add_argument("--model",    default="gpt-4o",      help="Model name")
    parser.add_argument(
        "--mode",
        choices=["oracle", "scoring"],
        default="oracle",
        help=(
            'LLM testing approach: \"oracle\" = compare-all-at-once (single rank-all prompt per pilot); '
            '"scoring" = independent 0–100 per pairing (oracle ground truth from weights is always computed).'
        ),
    )
    parser.set_defaults(lines=True)
    parser.add_argument(
        "--no-lines",
        dest="lines",
        action="store_false",
        help="Use individual pairing bidding instead of monthly lines",
    )
    parser.add_argument("--n-lines",           type=int, default=20, help="Number of lines")
    parser.add_argument("--pairings-per-line", type=int, default=5,  help="Pairings per line")
    parser.add_argument("--pilots",   type=int, default=3,   help="Number of pilots")
    parser.add_argument("--pairings", type=int, default=5,   help="Number of pairings")
    parser.add_argument("--pilot-seed",   type=int, default=1234567)
    parser.add_argument("--pairing-seed", type=int, default=42)
    parser.add_argument(
        "--consistency-runs", type=int, default=2,
        help="Independent scoring runs per pilot×line in scoring+lines mode (default: 2)",
    )
    parser.add_argument("--output",   default="results.html", help="Output file")
    args = parser.parse_args()

    config = {
        **DEFAULT_CONFIG,
        "n_pilots":     args.pilots,
        "n_pairings":   args.pairings,
        "pilot_seed":   args.pilot_seed,
        "pairing_seed": args.pairing_seed,
        "llm_name":     args.model if args.auto else "manual",
    }

    print("\n" + "="*60)
    print("PILOT BIDDING POC")
    print(f"  Pilots:   {config['n_pilots']}")
    print(f"  Pairings: {config['n_pairings']}")
    print(f"  Base:     {config['base']}")
    mode_label = args.mode.upper() + (" — automated (" + args.model + ")" if args.auto else " — manual")
    print(f"  Mode:     {mode_label}")
    print("="*60)

    # 1. Generate scenario
    print("\n[1/5] Generating scenario...")
    gen      = ScenarioGenerator(args.pilot_seed, args.pairing_seed)
    pilots   = gen.build_pilots(args.pilots, config["base"])

    if args.lines:
        n_pair   = args.n_lines * args.pairings_per_line
        max_b767 = round(n_pair * 0.32)   # ~32% B767; rest B737 for qualification coverage
        pairings = gen.build_pairings(n_pair, pilots, config["base"], max_b767=max_b767)
        lines    = gen.build_lines(pairings, args.n_lines, args.pairings_per_line)
        print(f"  {len(pilots)} pilots, {len(pairings)} pairings → {len(lines)} lines "
              f"({args.pairings_per_line} pairings/line)")
    else:
        pairings = gen.build_pairings(args.pairings, pilots, config["base"])
        lines    = []
        print(f"  {len(pilots)} pilots, {len(pairings)} pairings generated")

    # 2. Oracle rankings
    print("\n[2/5] Computing oracle rankings...")
    if args.lines:
        oracle_line_rankings = {
            pilot.name: oracle_rank_lines(pilot, lines)
            for pilot in pilots
        }
        oracle_rankings = {}   # unused in line mode
        for pilot in pilots:
            ranked = oracle_line_rankings[pilot.name]
            top    = next(r for r in ranked if r.oracle_rank == 1)
            print(f"  Capt. {pilot.name}: best line = L{top.line.id} ({top.oracle_score}pts) "
                  f"— {top.line.total_credit_hours}h credit, {top.line.total_nights_away} nights")
    else:
        oracle_rankings = oracle_rank_all(pilots, pairings)
        oracle_line_rankings = {}
        for pilot in pilots:
            ranked = oracle_rankings[pilot.name]
            top    = next(r for r in ranked if r.oracle_rank == 1)
            print(f"  Capt. {pilot.name}: best = P{top.pairing.id} ({top.oracle_score}pts)")

    # 3. Get LLM rankings
    scored_lines_by_pilot: dict = {}
    llm_line_rankings:     dict = {}   # populated only in line mode
    if args.mode == "scoring":
        if not args.auto:
            print("\nScoring mode requires --auto (API key needed for parallel calls).")
            print('Use --mode oracle for compare-all-at-once manual copy-paste (CLI flag name unchanged).')
            sys.exit(1)

        if args.lines:
            print(
                f"\n[3/5] Independent line scoring — "
                f"{len(pilots)} pilot(s) × {len(lines)} line(s) × "
                f"{args.consistency_runs} run(s)…"
            )
            scored_lines_by_pilot = asyncio.run(
                run_scored_lines(
                    pilots, lines, args.provider, args.model,
                    n_runs=args.consistency_runs,
                )
            )
            llm_line_rankings = scored_lines_to_llm_line_rankings(scored_lines_by_pilot)
            llm_rankings = {}
            for pilot in pilots:
                results = scored_lines_by_pilot.get(pilot.id, [])
                top = results[0] if results else None
                if top:
                    print(f"  Capt. {pilot.name}: best = L{top.line.id} (mean {top.score_mean})")
        else:
            print("\n[2.5/5] Pre-flight consistency check…")
            asyncio.run(run_consistency_checks(pilots, pairings, args.provider, args.model))

            print(f"\n[3/5] Independent scoring — {len(pilots)} pilots × {len(pairings)} pairings…")
            scored_by_pilot = asyncio.run(
                run_independent_scoring(pilots, pairings, args.provider, args.model)
            )
            llm_rankings = {
                pid: scored_pairings_to_llm_ranking(scored)
                for pid, scored in scored_by_pilot.items()
            }
            for pilot in pilots:
                if pilot.id in scored_by_pilot:
                    top = scored_by_pilot[pilot.id][0] if scored_by_pilot[pilot.id] else None
                    score_str = f"best = P{top.pairing.id} (score {top.score})" if top else "no results"
                    print(f"  Capt. {pilot.name}: {score_str}")
    else:
        print(f"\n[3/5] {'Calling LLM API' if args.auto else 'Collecting LLM responses (manual)'}...")
        if args.auto:
            llm_rankings = auto_oracle_run(pilots, pairings, args.model, args.provider)
        else:
            llm_rankings = manual_oracle_run(pilots, pairings)

    if not llm_rankings and not args.lines:
        print("No LLM rankings collected. Exiting.")
        sys.exit(0)

    # 4. Evaluate
    print("\n[4/5] Evaluating...")
    eval_metrics = {}

    if args.lines:
        if args.mode != "scoring":
            # Collect line rankings via rank-all oracle prompt (manual or automated)
            llm_line_rankings = {}
            if not args.auto:
                for pilot in pilots:
                    prompt = oracle_line_prompt(pilot, lines)
                    print(f"\n{'='*60}")
                    print(f"LINE PROMPT FOR: Capt. {pilot.name}")
                    print("=" * 60)
                    print(prompt)
                    print("\nPaste LLM response (JSON array), then press Enter twice:")
                    input_lines = []
                    while True:
                        ln = input()
                        if ln == "" and input_lines and input_lines[-1] == "":
                            break
                        input_lines.append(ln)
                    raw    = "\n".join(input_lines).strip()
                    parsed = parse_line_oracle_response(raw, len(lines))
                    if parsed:
                        llm_line_rankings[pilot.id] = parsed
                        print(f"  ✓ Parsed {len(parsed)} line rankings for Capt. {pilot.name}")
            else:
                for pilot in pilots:
                    prompt = oracle_line_prompt(pilot, lines)
                    print(f"  Calling {args.provider}/{args.model} for Capt. {pilot.name} (lines)…", end=" ")
                    try:
                        raw    = call_llm(prompt, args.model, args.provider)
                        parsed = parse_line_oracle_response(raw, len(lines))
                        if parsed:
                            llm_line_rankings[pilot.id] = parsed
                            print("✓")
                        else:
                            print("✗ parse error")
                    except Exception as e:
                        print(f"✗ API error: {e}")
        # else: llm_line_rankings was already set by run_scored_lines in step 3

        for pilot in pilots:
            if pilot.id not in llm_line_rankings:
                continue
            metrics = evaluate_line_pilot(
                pilot, llm_line_rankings[pilot.id], oracle_line_rankings[pilot.name]
            )
            eval_metrics[pilot.id] = metrics
            print(
                f"  Capt. {pilot.name}: "
                f"ρ={metrics.spearman}  "
                f"top-1={'✓' if metrics.top1_match else '✗'}  "
                f"elig={metrics.elig_accuracy:.0%}"
            )
    else:
        llm_line_rankings = {}
        for pilot in pilots:
            if pilot.id not in llm_rankings:
                continue
            metrics = evaluate_pilot(
                pilot, llm_rankings[pilot.id], oracle_rankings[pilot.name]
            )
            eval_metrics[pilot.id] = metrics
            print(
                f"  Capt. {pilot.name}: "
                f"ρ={metrics.spearman}  "
                f"top-1={'✓' if metrics.top1_match else '✗'}  "
                f"elig={metrics.elig_accuracy:.0%}"
            )

    # 5. Allocation
    print("\n[5/5] Running allocation...")
    if args.lines:
        oracle_alloc = oracle_line_allocation(pilots, lines)
        print("\n--- Oracle line allocation ---")
        for r in sorted(oracle_alloc, key=lambda x: x.pilot.seniority):
            ln = r.line
            if ln:
                print(f"  #{r.pilot.seniority} Capt. {r.pilot.name}: "
                      f"Line {ln.id} ({ln.total_credit_hours}h credit, "
                      f"{ln.total_nights_away} nights) — rank #{r.rank_awarded} choice")
            else:
                print(f"  #{r.pilot.seniority} Capt. {r.pilot.name}: no qualified line available")

        llm_alloc = None
        if len(llm_line_rankings) == len(pilots):
            try:
                llm_alloc = llm_line_allocation(pilots, lines, llm_line_rankings)
                print("\n--- LLM line allocation ---")
                for r in sorted(llm_alloc, key=lambda x: x.pilot.seniority):
                    ln = r.line
                    ora_ln = next(
                        (x.line for x in oracle_alloc if x.pilot.id == r.pilot.id), None
                    )
                    match = ora_ln and ln and ora_ln.id == ln.id
                    marker = "✓" if match else "✗"
                    if ln:
                        print(f"  #{r.pilot.seniority} Capt. {r.pilot.name}: "
                              f"Line {ln.id} — rank #{r.rank_awarded} choice  {marker}")
                    else:
                        print(f"  #{r.pilot.seniority} Capt. {r.pilot.name}: none")
            except ValueError as e:
                print(f"  ⚠ LLM line allocation skipped: {e}")
    else:
        oracle_alloc = oracle_allocation(pilots, pairings)
        print(format_allocation_report(oracle_alloc, "Oracle"))

        llm_alloc = None
        if len(llm_rankings) == len(pilots):
            try:
                llm_alloc = llm_allocation(pilots, pairings, llm_rankings)
                comparison = compare_allocations(oracle_alloc, llm_alloc)
                print(format_allocation_report(llm_alloc, "LLM", comparison))
            except ValueError as e:
                print(f"  ⚠ LLM allocation skipped: {e}")

    # Export
    report = format_results_html(
        pilots, pairings, oracle_rankings,
        llm_rankings, llm_line_rankings, oracle_line_rankings,
        eval_metrics, oracle_alloc, llm_alloc,
        config["llm_name"], config,
        schedule_lines=lines if args.lines else None,
        scored_lines_by_pilot=scored_lines_by_pilot,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✓ Results exported to {args.output}")


if __name__ == "__main__":
    main()
