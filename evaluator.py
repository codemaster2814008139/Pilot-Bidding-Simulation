"""
evaluator.py
------------
Computes evaluation metrics comparing LLM rankings to oracle rankings.
Mirrors the JavaScript scoreCurrentPilot() logic in the POC.

Metrics:
  - Spearman ρ:       rank correlation (-1 to 1)
  - Top-1 match:      did LLM pick oracle's #1?
  - Eligibility accuracy: correct qualification flags

Note on interpretation:
  These metrics measure agreement with the oracle, not absolute correctness.
  The oracle is based on designer-chosen weights — agreement reflects
  alignment with those weights, not ground truth pilot behaviour.

Note on ties:
  Oracle comparisons with gap=0 (tied scores) are excluded from pairwise
  agreement calculations — the oracle has no preference in these cases.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from models import (
    AdaptivePairwiseResult, EvalMetrics, LLMLineRank, LLMPairingRank,
    Pilot, PairwiseComparison, RankedLine, RankedPairing,
)


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------

def spearman_rho(
    llm_ranking: List[LLMPairingRank],
    oracle_ranking: List[RankedPairing],
) -> float:
    """
    Spearman ρ between LLM ranking and oracle ranking.

    Returns value in [-1, 1]:
      1.0  = perfect agreement
      0.0  = no correlation
     -1.0  = perfectly reversed

    Only includes pairings present in both lists.
    """
    # Build oracle rank lookup: pairing_id → oracle_rank
    oracle_map = {r.pairing.id: r.oracle_rank for r in oracle_ranking}

    # Align LLM ranks with oracle ranks
    pairs: List[Tuple[int, int]] = []  # (llm_rank, oracle_rank)
    for r in llm_ranking:
        if r.pairing_id in oracle_map:
            pairs.append((r.rank, oracle_map[r.pairing_id]))

    n = len(pairs)
    if n < 2:
        return 0.0

    llm_ranks  = [p[0] for p in pairs]
    ora_ranks  = [p[1] for p in pairs]
    llm_mean   = sum(llm_ranks) / n
    ora_mean   = sum(ora_ranks) / n

    num    = sum((l - llm_mean) * (o - ora_mean) for l, o in zip(llm_ranks, ora_ranks))
    den_l  = math.sqrt(sum((l - llm_mean) ** 2 for l in llm_ranks))
    den_o  = math.sqrt(sum((o - ora_mean) ** 2 for o in ora_ranks))

    if den_l == 0 or den_o == 0:
        return 0.0

    return round(num / (den_l * den_o), 2)


# ---------------------------------------------------------------------------
# Top-1 match
# ---------------------------------------------------------------------------

def top1_match(
    llm_ranking: List[LLMPairingRank],
    oracle_ranking: List[RankedPairing],
) -> bool:
    """
    Returns True if the LLM's top-ranked pairing matches the oracle's top-ranked.
    """
    llm_top    = next((r for r in llm_ranking if r.rank == 1), None)
    oracle_top = next((r for r in oracle_ranking if r.oracle_rank == 1), None)
    if not llm_top or not oracle_top:
        return False
    return llm_top.pairing_id == oracle_top.pairing.id


# ---------------------------------------------------------------------------
# Eligibility accuracy
# ---------------------------------------------------------------------------

def eligibility_accuracy(
    llm_ranking: List[LLMPairingRank],
    oracle_ranking: List[RankedPairing],
    pilot: Pilot,
) -> float:
    """
    Fraction of pairings where LLM correctly flagged eligibility.
    Eligibility = pilot is qualified for the aircraft type.
    Returns 0.0–1.0.
    """
    oracle_map = {r.pairing.id: r for r in oracle_ranking}
    correct = 0
    total   = 0
    for r in llm_ranking:
        if r.pairing_id not in oracle_map:
            continue
        actual_eligible = oracle_map[r.pairing_id].qualified
        if r.eligible == actual_eligible:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_pilot(
    pilot: Pilot,
    llm_ranking: List[LLMPairingRank],
    oracle_ranking: List[RankedPairing],
) -> EvalMetrics:
    """
    Compute all evaluation metrics for a single pilot.
    """
    sp    = spearman_rho(llm_ranking, oracle_ranking)
    top1  = top1_match(llm_ranking, oracle_ranking)
    elig  = eligibility_accuracy(llm_ranking, oracle_ranking, pilot)

    # Composite score (for logging — not a primary metric)
    overall = round(((sp + 1) / 2 * 60) + (25 if top1 else 0) + (elig * 15))

    return EvalMetrics(
        spearman=sp,
        top1_match=top1,
        elig_accuracy=elig,
        overall=overall,
    )


# ---------------------------------------------------------------------------
# Pairwise evaluation helpers
# ---------------------------------------------------------------------------

def pairwise_agreement(
    llm_winner_id: int,
    oracle_winner_id: int,
    oracle_score_gap: int,
) -> dict:
    """
    Evaluate a single pairwise comparison.

    Returns dict with:
      correct:      bool (LLM matched oracle winner)
      tied:         bool (oracle gap = 0 — comparison is meaningless)
      meaningful:   bool (gap > 0 — should count toward accuracy)
    """
    tied       = oracle_score_gap == 0
    correct    = llm_winner_id == oracle_winner_id
    return {
        "correct":    correct,
        "tied":       tied,
        "meaningful": not tied,
    }


def pairwise_summary(comparisons: List[dict]) -> dict:
    """
    Aggregate pairwise results across all comparisons for a pilot.

    comparisons: list of dicts with keys: correct, tied, meaningful.
    Returns: total, clear (non-tied), correct_clear, agreement_pct.
    """
    total         = len(comparisons)
    clear         = [c for c in comparisons if c["meaningful"]]
    correct_clear = [c for c in clear if c["correct"]]
    tied          = total - len(clear)
    pct           = round(len(correct_clear) / len(clear) * 100) if clear else 0

    return {
        "total":          total,
        "clear":          len(clear),
        "correct_clear":  len(correct_clear),
        "tied":           tied,
        "agreement_pct":  pct,
    }


# ---------------------------------------------------------------------------
# Scoring stability (independent scoring mode)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Line evaluation
# ---------------------------------------------------------------------------

def _line_spearman(
    llm_ranking: List[LLMLineRank],
    oracle_ranking: List[RankedLine],
) -> float:
    """Spearman ρ between LLM line ranking and oracle line ranking."""
    oracle_map = {r.line.id: r.oracle_rank for r in oracle_ranking}
    pairs: List[Tuple[int, int]] = []
    for r in llm_ranking:
        if r.line_id in oracle_map:
            pairs.append((r.rank, oracle_map[r.line_id]))
    n = len(pairs)
    if n < 2:
        return 0.0
    llm_ranks = [p[0] for p in pairs]
    ora_ranks  = [p[1] for p in pairs]
    llm_mean  = sum(llm_ranks) / n
    ora_mean  = sum(ora_ranks) / n
    num   = sum((l - llm_mean) * (o - ora_mean) for l, o in zip(llm_ranks, ora_ranks))
    den_l = math.sqrt(sum((l - llm_mean) ** 2 for l in llm_ranks))
    den_o = math.sqrt(sum((o - ora_mean) ** 2 for o in ora_ranks))
    if den_l == 0 or den_o == 0:
        return 0.0
    return round(num / (den_l * den_o), 2)


def _line_top1(
    llm_ranking: List[LLMLineRank],
    oracle_ranking: List[RankedLine],
) -> bool:
    """True if LLM's top-ranked line matches oracle's top-ranked line."""
    llm_top    = next((r for r in llm_ranking if r.rank == 1), None)
    oracle_top = next((r for r in oracle_ranking if r.oracle_rank == 1), None)
    if not llm_top or not oracle_top:
        return False
    return llm_top.line_id == oracle_top.line.id


def _line_elig_accuracy(
    llm_ranking: List[LLMLineRank],
    oracle_ranking: List[RankedLine],
) -> float:
    """Fraction of lines where LLM correctly flagged eligibility."""
    oracle_map = {r.line.id: r for r in oracle_ranking}
    correct = 0
    total   = 0
    for r in llm_ranking:
        if r.line_id not in oracle_map:
            continue
        if r.eligible == oracle_map[r.line_id].qualified:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def evaluate_line_pilot(
    pilot: Pilot,
    llm_ranking: List[LLMLineRank],
    oracle_ranking: List[RankedLine],
) -> EvalMetrics:
    """
    Compute all evaluation metrics for a single pilot in line-bidding mode.

    Uses the same metric definitions as evaluate_pilot() but operates on
    line IDs instead of pairing IDs.  Returns EvalMetrics so all downstream
    reporting code works unchanged.
    """
    sp   = _line_spearman(llm_ranking, oracle_ranking)
    top1 = _line_top1(llm_ranking, oracle_ranking)
    elig = _line_elig_accuracy(llm_ranking, oracle_ranking)
    overall = round(((sp + 1) / 2 * 60) + (25 if top1 else 0) + (elig * 15))
    return EvalMetrics(spearman=sp, top1_match=top1, elig_accuracy=elig, overall=overall)


def evaluate_scoring_stability(scores: List[float]) -> dict:
    """
    Assess reliability of repeated independent scores for the same pilot+pairing.

    Used before a main scoring run to check whether the LLM produces consistent
    scores when called multiple times on the same input. High variance (std > 10)
    means the ranking derived from a single pass may not be trustworthy.

    Args:
        scores: List of 0–100 scores from n repeated calls for the same pair.

    Returns:
        dict with keys:
          mean, std, min, max, coefficient_of_variation (%), stable (bool).
        stable = True when std <= 10.
    """
    if not scores:
        return {
            "mean": 0.0, "std": 0.0, "min": 0, "max": 0,
            "coefficient_of_variation": 0.0, "stable": True,
        }

    n        = len(scores)
    mean     = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std      = math.sqrt(variance)
    cv       = (std / mean * 100) if mean > 0 else 0.0

    stability = "stable" if std < 5 else ("marginal" if std <= 10 else "unstable")

    return {
        "mean":                     round(mean, 1),
        "std":                      round(std, 1),
        "min":                      min(scores),
        "max":                      max(scores),
        "coefficient_of_variation": round(cv, 1),
        "stable":                   std <= 10,   # True for stable + marginal (backwards-compat)
        "stability":                stability,   # "stable" | "marginal" | "unstable"
    }


# ---------------------------------------------------------------------------
# Round-robin derived ranking
# ---------------------------------------------------------------------------

def derive_ranking_from_wins(
    pairing_ids: List[int],
    comparisons: List[dict],  # each must have llm_winner and pairing_a/b ids
) -> List[Tuple[int, int]]:
    """
    Derive a full ranking from round-robin pairwise results by win count.
    Returns list of (pairing_id, wins) sorted best → worst.

    Note: if there's a cycle (A>B, B>C, C>A), rankings are not fully
    transitive. Cycles are flagged separately — see detect_cycle().
    """
    wins = {pid: 0 for pid in pairing_ids}
    for comp in comparisons:
        winner = comp.get("llm_winner")
        if winner in wins:
            wins[winner] += 1
    return sorted(wins.items(), key=lambda x: -x[1])


def detect_cycle(
    pairing_ids: List[int],
    comparisons: List[dict],
) -> Optional[List[int]]:
    """
    Detect if LLM preferences form a cycle (non-transitive).
    Only meaningful with exactly 3 pairings (6 possible cycles to check).
    Returns the cycle as a list of pairing ids, or None if consistent.
    """
    if len(pairing_ids) != 3:
        return None  # only check for 3 pairings

    wins = {}  # wins[a][b] = True means a beats b
    for comp in comparisons:
        a = comp.get("pairing_a")
        b = comp.get("pairing_b")
        w = comp.get("llm_winner")
        if a and b and w:
            wins.setdefault(w, set()).add(b if w == a else a)

    p1, p2, p3 = pairing_ids
    # Check both cycle directions
    for cycle in [(p1, p2, p3), (p1, p3, p2)]:
        a, b, c = cycle
        if (b in wins.get(a, set()) and
                c in wins.get(b, set()) and
                a in wins.get(c, set())):
            return list(cycle) + [a]  # e.g. [1, 2, 3, 1]

    return None


# ---------------------------------------------------------------------------
# Bradley-Terry model
# ---------------------------------------------------------------------------

class BradleyTerryModel:
    """
    Fits a Bradley-Terry model from pairwise comparison outcomes.

    The probability that item i beats item j is:
      P(i beats j) = strength_i / (strength_i + strength_j)

    Strengths are fitted via maximum likelihood estimation.
    Items with higher fitted strength rank higher.

    Handles cycles and inconsistencies gracefully — finds the
    ranking that best explains all observed comparisons.
    """

    def __init__(self, item_ids: List[int]):
        self._ids: List[int] = list(item_ids)
        self._idx: Dict[int, int] = {iid: i for i, iid in enumerate(self._ids)}
        self._comparisons: List[Tuple[int, int]] = []   # (winner_idx, loser_idx)
        self._theta: Optional[np.ndarray] = None        # log-strengths after fit
        self._fitted = False

    # ------------------------------------------------------------------
    def add_comparison(self, winner_id: int, loser_id: int) -> None:
        """Record one pairwise comparison outcome."""
        self._comparisons.append((self._idx[winner_id], self._idx[loser_id]))
        self._fitted = False

    def add_maxdiff(self, best_id: int, worst_id: int, other_ids: List[int]) -> None:
        """
        Record a MaxDiff observation:
          best beats every other item in the set
          every other item beats worst
        """
        all_others = [iid for iid in other_ids if iid != best_id and iid != worst_id]
        for oid in all_others:
            self.add_comparison(best_id, oid)
            self.add_comparison(oid, worst_id)
        self.add_comparison(best_id, worst_id)

    # ------------------------------------------------------------------
    def fit(self) -> bool:
        """
        Fit via MLE using log-parameterisation for numerical stability.
        theta[0] fixed at 0 for identifiability.
        Returns True if convergence achieved.
        """
        n = len(self._ids)
        if n < 2 or not self._comparisons:
            self._theta = np.zeros(n)
            self._fitted = True
            return True

        def neg_log_likelihood(theta_free: np.ndarray) -> float:
            theta = np.concatenate([[0.0], theta_free])
            total = 0.0
            for wi, li in self._comparisons:
                # log P(wi beats li) = theta_wi - log(exp(theta_wi) + exp(theta_li))
                tw, tl = theta[wi], theta[li]
                # numerically stable via log-sum-exp
                total += tw - np.logaddexp(tw, tl)
            return -total

        x0 = np.zeros(n - 1)
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                          options={"maxiter": 1000, "ftol": 1e-10})
        self._theta = np.concatenate([[0.0], result.x])
        self._fitted = True
        return bool(result.success)

    # ------------------------------------------------------------------
    def _ensure_fitted(self) -> None:
        if not self._fitted:
            self.fit()

    def ranking(self) -> List[Tuple[int, float]]:
        """Returns list of (item_id, strength) sorted best to worst."""
        self._ensure_fitted()
        strengths = [(iid, float(np.exp(self._theta[i])))
                     for i, iid in enumerate(self._ids)]
        return sorted(strengths, key=lambda x: -x[1])

    def rank_positions(self) -> Dict[int, int]:
        """Returns dict: item_id → rank (1=best)."""
        return {iid: rank for rank, (iid, _) in enumerate(self.ranking(), start=1)}

    # ------------------------------------------------------------------
    def confidence_intervals(
        self, n_bootstrap: int = 100
    ) -> Dict[int, Tuple[float, float]]:
        """
        Bootstrap confidence intervals on rank positions.
        Resample observed comparisons with replacement n_bootstrap times,
        refit model each time, record rank position distribution.
        Returns dict: item_id → (lower_rank, upper_rank) at 90% CI.
        A wide interval means this item's rank position is uncertain.
        """
        self._ensure_fitted()
        n_comp = len(self._comparisons)
        rank_samples: Dict[int, List[int]] = {iid: [] for iid in self._ids}

        for _ in range(n_bootstrap):
            boot_model = BradleyTerryModel(self._ids)
            if n_comp > 0:
                indices = [random.randrange(n_comp) for _ in range(n_comp)]
                for idx in indices:
                    wi, li = self._comparisons[idx]
                    boot_model.add_comparison(self._ids[wi], self._ids[li])
            boot_model.fit()
            for iid, rank in boot_model.rank_positions().items():
                rank_samples[iid].append(rank)

        ci: Dict[int, Tuple[float, float]] = {}
        for iid, samples in rank_samples.items():
            if samples:
                lower = float(np.percentile(samples, 5))
                upper = float(np.percentile(samples, 95))
            else:
                r = self.rank_positions().get(iid, 1)
                lower, upper = float(r), float(r)
            ci[iid] = (lower, upper)
        return ci

    # ------------------------------------------------------------------
    def fit_quality(self) -> float:
        """
        Returns proportion of observed comparisons correctly predicted
        by the fitted model (analogous to accuracy).
        Compare to human baseline of ~70-85% (inconsistency rate 15-30%).
        """
        self._ensure_fitted()
        if not self._comparisons:
            return 1.0
        correct = 0
        for wi, li in self._comparisons:
            if self._theta[wi] >= self._theta[li]:
                correct += 1
        return correct / len(self._comparisons)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable summary of ranking with confidence intervals."""
        self._ensure_fitted()
        ranked = self.ranking()
        ci = self.confidence_intervals(n_bootstrap=100)
        fq = self.fit_quality()
        lines = [f"Bradley-Terry ranking ({len(self._ids)} items, fit quality {fq:.0%}):"]
        for rank, (iid, strength) in enumerate(ranked, start=1):
            lo, hi = ci[iid]
            lines.append(f"  #{rank}  item {iid}  strength={strength:.3f}  90%CI=[{lo:.0f},{hi:.0f}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adaptive comparison design
# ---------------------------------------------------------------------------

def design_adaptive_comparisons(
    item_ids: List[int],
    provisional_ranking: Optional[List[int]] = None,
    n_uncertain_threshold: int = 2,
) -> List[Tuple[int, int]]:
    """
    Design a minimal set of pairwise comparisons.

    Round 1 (always):
      Compare adjacent pairs in randomised order.
      For N items: ceil(N/2) comparisons.

    Round 2 (adaptive):
      After round 1, fit a provisional BT model.
      Identify uncertain pairs: items where |rank_i - rank_j| <= n_uncertain_threshold
      AND the pair was not compared in round 1.
      Add these pairs to round 2.

    Returns list of (item_a_id, item_b_id) tuples for all rounds.
    """
    ids = list(item_ids)
    random.shuffle(ids)

    # Round 1: adjacent pairs
    round1: List[Tuple[int, int]] = []
    for i in range(0, len(ids) - 1, 2):
        round1.append((ids[i], ids[i + 1]))

    if not provisional_ranking:
        return round1

    # Round 2: uncertain pairs from provisional ranking
    round1_set = {(min(a, b), max(a, b)) for a, b in round1}
    rank_pos = {iid: r for r, iid in enumerate(provisional_ranking)}

    round2: List[Tuple[int, int]] = []
    for i in range(len(provisional_ranking)):
        for j in range(i + 1, len(provisional_ranking)):
            a, b = provisional_ranking[i], provisional_ranking[j]
            if abs(rank_pos[a] - rank_pos[b]) <= n_uncertain_threshold:
                key = (min(a, b), max(a, b))
                if key not in round1_set:
                    round2.append((a, b))

    return round1 + round2


# ---------------------------------------------------------------------------
# Run adaptive pairwise — fit BT model from completed comparisons
# ---------------------------------------------------------------------------

def run_adaptive_pairwise(
    pilot: Pilot,
    items: List,
    comparisons: List[PairwiseComparison],
    n_uncertain_threshold: int = 2,
) -> AdaptivePairwiseResult:
    """
    Given completed pairwise comparisons, fit Bradley-Terry model
    and return full ranking with confidence intervals.
    """
    item_ids = [item.id for item in items]
    model = BradleyTerryModel(item_ids)
    for comp in comparisons:
        model.add_comparison(comp.winner_id,
                             comp.item_b_id if comp.winner_id == comp.item_a_id
                             else comp.item_a_id)

    model.fit()
    final_ranking = model.ranking()
    rank_pos = model.rank_positions()
    ci = model.confidence_intervals(n_bootstrap=100)
    fq = model.fit_quality()

    n_rounds = max((c.round_num for c in comparisons), default=1)

    if fq >= 0.85:
        note = f"High consistency (fit quality {fq:.0%}) — above human baseline"
    elif fq >= 0.70:
        note = f"Moderate consistency ({fq:.0%}) — within human baseline range"
    else:
        note = (
            f"Low consistency ({fq:.0%}) — below human baseline "
            f"(15-30% human error rate implies ~70-85% fit quality)"
        )

    return AdaptivePairwiseResult(
        pilot=pilot,
        comparisons=comparisons,
        bt_model=model,
        final_ranking=final_ranking,
        rank_positions=rank_pos,
        confidence_intervals=ci,
        fit_quality=fq,
        n_comparisons=len(comparisons),
        n_rounds=n_rounds,
        consistency_note=note,
    )
