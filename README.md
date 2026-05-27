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
python main.py --pilots 5 --pairings 5
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
| 1 (current) | 5 | 5 | Oracle ranking, manual | $0 |
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
Derived from pilot profile (family status, age). **Four factors** sum to 100:
`tafb`, `hotel_nights`, `report_time`, and `credit_pay`. Two factors (`tafb`
and `hotel_nights`) are profile-linked; `report_time` is fixed at 12 for all
pilots; `credit_pay` fills the remainder. The oracle is a reference point, not
ground truth.

### Pay
Per-pilot: `credit_hours × pilot.base_pay`. Each pilot sees their own
personalised pay figure in the prompt. Pairing cards show credit hours
only (not dollars) since pay varies by pilot.

### Hotel quality
Fixed at Standard for all hotels (removed as a variable factor).

---

## Scenario generation

### Pilot generation philosophy

Pilots are generated deterministically from a seeded PRNG (`pilot_seed=1234567`).
The same seed always produces the same pilot pool, making results reproducible
across runs and machines.

**Profile attributes drawn randomly:**

| Attribute | Range / options |
|---|---|
| Age | 26–58 (uniform) |
| Family status | Single · Married no kids · Married 1 child · Married 2+ kids · Single parent |
| Aircraft qualification | 65% dual (B737 + B767) · 35% B737-only |
| Min rest | 10–14 h |

**Oracle weights are derived automatically from age + family status** — the
analyst never sets weights by hand. The derivation encodes real-world priority
differences across three pilot archetypes:

| Archetype | `has_kids` | `is_mid_career` (age ≥ 40) | hotel_nights w | tafb w | credit_pay w |
|---|---|---|---|---|---|
| Family pilot | ✓ | either | 26 | 22 | 40 |
| Mid-career, no kids | ✗ | ✓ | 16 | 15 | 57 |
| Early-career, no kids | ✗ | ✗ | 9 | 9 | 70 |

`report_time` is fixed at 12 for all pilots. `credit_pay` fills to 100.

The direction of hotel_nights scoring also flips by archetype:
- **Family pilots** — fewer nights away = better (less time from home).
- **Non-family pilots** — more nights away = better (more per-diem income and
  flying experience). The LLM prompt reflects this explicitly via the priority
  list in `prompt_builder._priority_list()`.

**Pay rates** follow the Delta 2023 contract longevity table (6 steps mapped
from age). Older pilots earn more per credit hour, so their per-pilot pay
figures in prompts are higher even for identical pairings.

---

### Pairing generation philosophy

All pairings are **circular** — every trip starts and ends at the home base
(default `BOS`). This mirrors real airline contract pairings.

**Structural choices:**

- **Legs**: sampled from `[2, 3, 3, 4]` (weighted toward 3-leg trips).
  `nights_away = num_legs − 1`, giving a natural 1–3 overnight spread.
- **Schedules are chained realistically**:
  - First departure: morning bank (06:00–09:30, 55%) or afternoon bank
    (14:00–18:00, 45%), rounded to 5-min slots.
  - Hotel departures (day 2+): always 06:00–08:30.
  - Ground turns: B767 needs 75–120 min; B737 needs 50–90 min.
- **Aircraft mix**: roughly 50/50 B737/B767 unless `max_b767` is set (e.g.
  when building 25 pairings for line mode, a 17/8 ratio is enforced so
  B737-only pilots always have biddable options).

**Pay computation** (per Delta 2023 contract):

```
credit_hours  = max(block_hours, tafb / 3.5)   # 1-for-3.5 rig
block_pay     = credit_hours × pilot.base_pay
per_diem      = tafb × $2.85
total_value   = block_pay + per_diem
```

---

### Line generation philosophy

Lines group pairings into monthly schedules — the atomic unit pilots actually
bid on in Scenarios 4+.

**Key invariant — aircraft qualification**: B737 pairings are placed into lines
before B767 pairings. This ensures that the first lines in the pool are
all-B737, giving B737-only pilots at least some fully-qualified options. A line
requires qualification on **every pairing it contains**.

**Conflict detection**: the `Line.has_conflicts()` method checks whether any
two pairings overlap on the calendar (using a 6-day spacing convention). The
generator does not enforce conflict-free lines automatically — callers can
filter or regenerate if needed.

Use `ScenarioGenerator.build_lines()` directly:

```python
from generator import ScenarioGenerator
gen      = ScenarioGenerator()
pilots   = gen.build_pilots(n=5)
pairings = gen.build_pairings(n=25, pilots=pilots, max_b767=8)
lines    = gen.build_lines(pairings, n_lines=5, pairings_per_line=5)
```

---

## Oracle sub-scores

Each pairing/line is scored on four sub-dimensions (each 0–100), then combined
as a weighted sum using the pilot's weights.

### Pairing-level bounds

| Sub-score | Formula / bounds |
|---|---|
| TAFB | Linear: 18h → 100, 80h → 0 |
| Hotel nights (family) | 100 − 35 × nights (floor 0); 0 nights = 100, 3 nights = 0 |
| Hotel nights (non-family) | 20 + 40 × nights (cap 100); prefers more overnights |
| Report time | ≥ 07:00 → 100, ≤ 04:00 → 0, linear between |
| Credit pay | Normalised: best-paying pairing = 100, worst = 0; B767 carries a 4% pay premium |

### Line-level bounds

| Sub-score | Formula / bounds |
|---|---|
| Total TAFB | Linear: 90h → 100, 400h → 0 (5× per-pairing bounds) |
| Total nights (family) | Ideal = 9 nights; −15 per night deviation (floor 0) |
| Total nights (non-family) | 7 × nights up to 14 nights → 98, then −20 per night above 14 |
| Report time | Average of per-pairing report-time sub-scores |
| Credit pay | Normalised across lines; B767 pairings carry 4% premium within line totals |

---

## Methods

> **Implementation note**: The HTML/JavaScript POC and the Python backend
> implement the same three methods but with some differences in the fitting
> algorithms. Where they differ this is called out explicitly below.

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

#### Fitting — HTML/JavaScript implementation (MM algorithm)

Strengths are estimated via the Minorization-Maximization (MM) iterative update:

$$s_i^{\text{new}} = \frac{W_i}{\displaystyle\sum_{(i,j)\in\text{comparisons}} \frac{1}{s_i + s_j}}$$

where $W_i$ is the total number of wins for item $i$.

- Initialised at $s_i = 1$ for all items.
- Normalised after each iteration so $\max_i s_i = 1$.
- Converges when $\max_i |s_i^{\text{new}} - s_i| < 10^{-9}$ (≤ 500 iterations).
- Guaranteed convergence; no external solver required.

**Zero-win floor**: items with $W_i = 0$ receive $s_i = 0.01$ instead of 0.
This prevents rank collapse and ensures all items appear in the final ranking.

> **Python backend note**: `evaluator.BradleyTerryModel` uses a different
> fitting approach — log-parameterisation MLE solved via **scipy L-BFGS-B**
> (`theta[0]` fixed at 0 for identifiability). The end result is equivalent
> but the Python version leverages scipy for numerical stability. The MM
> algorithm above is the primary implementation used in the HTML POC.

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

#### Tie detection (mean gap + CI-overlap)

Items $A$ and $B$ are marked **tied** only when **both** conditions hold:

1. **Mean gap** is small: $|\mu_A - \mu_B| < 5$ points
2. **Confidence intervals overlap**: $(\mu_A - \sigma_A \leq \mu_B + \sigma_B)$ AND $(\mu_B - \sigma_B \leq \mu_A + \sigma_A)$

where $\mu$ is the mean score and $\sigma$ is the standard deviation across runs.

The mean-gap guard (condition 1) prevents two items whose CIs happen to be wide
from being declared tied even when their means are clearly separated. Both
conditions must be satisfied together for a tie to be declared.

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

---

## Dependencies

### HTML/JavaScript POC
No dependencies — runs entirely in the browser with no build step.

### Python backend
```
pip install anthropic      # Anthropic API (Claude models)
pip install openai         # OpenAI API (GPT models)
pip install numpy scipy    # Required by evaluator.py (BradleyTerry MLE fitting)
```
