# Pilot Bidding POC — Python

Python implementation of the pilot flight pairing bidding simulation.
Mirrors the logic in the HTML/JavaScript POC, structured for extensibility and scaling.

## Structure

```
pilot_bidding/
├── models.py          # Dataclasses: Pilot, Pairing, Leg, OracleWeights, etc.
├── generator.py       # Deterministic scenario generation (pilots + pairings)
├── oracle.py          # Oracle scoring and ranking logic
├── prompt_builder.py  # LLM prompt generation (oracle, pairwise, scoring modes)
├── evaluator.py       # Evaluation metrics: Spearman ρ, top-1, eligibility accuracy
├── allocator.py       # Seniority-order allocation with tiebreak tracking
├── main.py            # End-to-end runner (manual and automated modes)
└── README.md
```

## Quick start

### Manual mode (paste prompts to any LLM, no API key needed)
```bash
python main.py --pilots 3 --pairings 5
```

### Automated mode (requires API key)
```bash
# OpenAI
OPENAI_API_KEY=sk-... python main.py --auto --model gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-... python main.py --auto --provider anthropic --model claude-haiku-4-5-20251001

# More pilots and pairings
OPENAI_API_KEY=sk-... python main.py --auto --pilots 10 --pairings 20 --model gpt-4o-mini
```

## Scaling roadmap

| Scenario | Pilots | Pairings | Method | Est. cost |
|---|---|---|---|---|
| 1 (current) | 3 | 5 | Oracle ranking, manual | $0 |
| 2 | 5 | 10 | Oracle ranking, automated | ~$0.10 |
| 3 | 10 | 20 | Oracle ranking, automated | ~$1 |
| 4* | 10 | 50 (10 lines) | Pairwise, automated | ~$2 |
| 5* | 10 | 100 (20 lines) | Independent scoring | ~$5 |
| 6* | 50 | 500 (100 lines) | Independent scoring | ~$15 |
| 7 (production) | 100 | 1000 (200 lines) | Independent scoring | ~$48 |

*Scenarios 4+ group pairings into monthly lines. Pilots bid on lines.

## Key design decisions

### Seniority
Seniority determines **bid order only** — not eligibility. More senior pilots
choose first from the available pool. All pilots can bid any pairing they are
qualified for.

### Aircraft qualification
Hard constraint. A B737-only pilot cannot bid a B767 pairing.
Unqualified pairings score 0 and are ranked last by the oracle.

### Oracle weights
Derived from pilot profile (family status, age). Two of five factors
are linked to profile; three are fixed ratios. Oracle is a reference
point, not ground truth.

### Pay
Per-pilot: `credit_hours × pilot.base_pay`. Each pilot sees their own
personalised pay figure in the prompt. Pairing cards show credit hours
only (not dollars) since pay varies by pilot.

### Hotel quality
Fixed at Standard for all hotels (removed as a variable factor).

## Methods

### A · Rank-all

The LLM receives a single prompt containing the pilot's full profile and all
pairings/lines. It returns a JSON-ranked list in one call.

- No statistical model — purely prompt engineering.
- Evaluated directly with Spearman ρ against the oracle ranking.
- Fastest (1 call per pilot) but degrades on long lists as context grows.

---

### B · Pairwise — Bradley-Terry model

Each LLM call presents exactly two options: *"Which is better for this pilot?"*
Votes are aggregated into a global ranking via the **Bradley-Terry (BT) model**.

#### Model

$$P(i \text{ beats } j) = \frac{s_i}{s_i + s_j}$$

where $s_i > 0$ is the latent strength of item $i$.

#### Fitting — MM algorithm

Strengths are estimated by maximising the log-likelihood via the
Minorization-Maximization (MM) iterative update:

$$s_i^{\text{new}} = \frac{W_i}{\displaystyle\sum_{(i,j)\in\text{comparisons}} \frac{1}{s_i + s_j}}$$

where $W_i$ is the total number of wins for item $i$.

- Initialised at $s_i = 1$ for all items.
- Normalised after each iteration so $\max_i s_i = 1$.
- Converges when $\max_i |s_i^{\text{new}} - s_i| < 10^{-9}$ (≤ 500 iterations).
- Guaranteed convergence; no external solver required.

**Zero-win floor**: items with $W_i = 0$ receive $s_i = 0.01$ instead of 0.
This prevents rank collapse and ensures all items appear in the final ranking.

#### Adaptive pair design (~N comparisons vs N(N−1)/2 brute force)

**Round 1** — seeded shuffle → adjacent pairs:
1. Fisher-Yates shuffle of all item IDs (seeded by `Date.now()`).
2. Consecutive pairs from the shuffled list: $(L_1, L_2),\,(L_3, L_4),\ldots$
3. For odd $N$, the last item wraps back to pair with the first → exactly $\lceil N/2 \rceil$ comparisons.

**Round 2** — targeted uncertain pairs:
1. Fit a provisional BT model on Round-1 results.
2. Add pairs $(i, j)$ where $|\text{rank}_i - \text{rank}_j| \leq 2$ that have not yet been compared.
3. Fit the final BT model on all comparisons combined.

Total comparisons: $\approx \lceil N/2 \rceil + \text{a few}$, vs $\binom{N}{2}$ for full round-robin.

---

### C · Scoring — independent scores with CI-overlap tie detection

Each item is scored independently on a 0–100 scale. One LLM call per item.

#### Personalised calibration anchor

Every prompt includes a pilot-specific rubric block with concrete examples:

| Score range | Meaning for this pilot |
|---|---|
| 90–100 | Ideal: home same day, preferred aircraft, report after 07:00 |
| 45–55 | Acceptable: one overnight, mixed fleet, moderate TAFB |
| 0–15 | Unacceptable: 2+ nights, wrong aircraft, pre-05:00 report |

This anchors the LLM's scale to the individual pilot's preferences rather than a
generic rubric.

#### Multi-run aggregation

- 2 runs → final score = mean.
- 3 runs → final score = median (triggered when consistency is unstable after run 2).

#### Tie detection (CI-overlap)

Items $A$ and $B$ are marked **tied** if their score intervals overlap:

$$(\mu_A - \sigma_A \leq \mu_B + \sigma_B) \;\text{AND}\; (\mu_B - \sigma_B \leq \mu_A + \sigma_A)$$

where $\mu$ is the mean score and $\sigma$ is the standard deviation across runs.
Overlapping intervals indicate the LLM cannot reliably distinguish the two items.

**Oracle tie parameters** (ground-truth grouping):
- Threshold: ±3 points between adjacent items triggers a tie.
- Max group size: 3 items; groups larger than 3 are split using item ID as a tiebreaker.

---

## Evaluation metrics

- **Spearman ρ**: rank correlation between LLM and oracle (−1 to 1).
  Converts both lists to rank positions (1st, 2nd, …) and measures order similarity.
  Tied items receive average ranks.
- **Top-1 match**: did the LLM's #1 choice match the oracle's #1 choice?
- **Eligibility accuracy**: fraction of items correctly flagged as qualified/unqualified.
- **Overall score** (0–100): `0.60 × ρ_normalised + 0.25 × top1 + 0.15 × eligibility`
  where ρ is normalised from [−1, 1] to [0, 1].

Note: these measure agreement with the oracle, not absolute pilot truth.

## Extending to lines

For Scenarios 4+, group pairings into monthly lines before building prompts:

```python
from models import Pairing
from typing import List

def group_into_lines(pairings: List[Pairing], pairings_per_line: int = 5):
    lines = []
    for i in range(0, len(pairings), pairings_per_line):
        lines.append(pairings[i:i+pairings_per_line])
    return lines
```

Then build one prompt per line (or per pilot×line for scoring mode).

## Dependencies

```
pip install anthropic   # for Anthropic API
pip install openai      # for OpenAI API
```

No other dependencies — standard library only.
