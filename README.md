# Pilot Bidding — AI Evaluation Proof of Concept

## What this is

Airlines assign monthly flight schedules to pilots through a **bidding process**: pilots rank the trips they want, and assignments are made in seniority order — the most senior pilot picks first, then the next, and so on down the roster.

This tool is a **proof-of-concept simulation** that tests whether an AI (a large language model, or LLM) can accurately predict which flight schedules a pilot would prefer — and how well that matches a reference formula built from the pilot's known profile.

**The central question: Can AI understand a pilot's personal preferences well enough to rank their flight options the way they themselves would?**

---

## Why it matters

If AI can reliably infer pilot preferences from a profile, it opens the door to:

- Assisted or automated bidding that reduces manual effort for pilots
- Better schedule design by understanding preference patterns at scale
- Tools that help pilots with complex preferences navigate large option sets

---

## How the simulation works

The simulation has three components that run in sequence.

### 1. Scenario — the pilots and trips

A set of pilots is generated, each with a profile: seniority, family status, age, aircraft qualification, and pay rate. A set of flight trips is also generated. Each pilot can only bid on trips they are qualified to fly.

The simulation supports two modes that mirror how real airline bidding works:

- **Pairing mode** — pilots bid on individual 3–5 day trips
- **Line mode** — pilots bid on a full monthly schedule made up of several trips (this is how real airline bidding is structured)

### 2. Oracle — the reference answer

Before the AI is involved, the simulation calculates a "correct" ranking for each pilot using a rule-based formula called the **oracle**. The oracle scores each trip based on what is known about the pilot: how much they value pay, time at home, favorable report times, and so on.

This gives a benchmark to measure the AI against. The oracle is a research reference point, not a claim about what any pilot truly wants.

### 3. LLM Evaluation — the AI's turn

The AI is asked to rank the same trips for each pilot, given only the pilot's profile and the trip details. The AI has no access to the oracle formula. Three different approaches are tested:

| Method | How it works | Best for |
|---|---|---|
| **Rank-all** | One prompt — AI ranks all trips at once | Fast; works well on short lists |
| **Scoring** | AI scores each trip 0–100 independently, then sorted by score | Longer lists; easier to check consistency |
| **Pairwise** | AI compares two trips at a time; ~10 comparisons instead of hundreds | Most rigorous; handles ties well |

Results from each method are compared against the oracle ranking to measure how well the AI did.

---

## What you can learn from a run

- **Which AI method** (rank-all, scoring, pairwise) best matches the oracle
- **How consistent** the AI is — does it give the same ranking when asked the same question twice?
- **Whether the AI correctly identifies** which trips a pilot cannot fly (eligibility)
- **How AI-based allocation compares to oracle allocation** in final trip assignments
- **How accuracy and cost scale** with the number of pilots and trips

---

## Using the tool

The easiest way to explore the simulation is the **browser-based UI** — no programming experience needed to run experiments.

### Setup (one time, requires Python)

1. In the `pilot_bidding` folder, run: `pip install -r requirements-backend.txt`
2. Copy `env.example` to `.env` and add your AI provider key (OpenAI or Anthropic/Claude)
3. Start the local server: `python proxy_server.py`
4. Open your browser to: `http://127.0.0.1:8765/pilot_bidding_poc.html`

> **No API key?** Open `pilot_bidding_poc.html` directly in your browser. You can still generate prompts manually, paste them into any AI chat (ChatGPT, Claude, etc.), and paste the response back. All metrics are computed automatically.

---

## The UI — tab by tab

### Backend status (top of every page)

Before doing anything, confirm the green **Connected** status. If it shows an error, the local server is not running — return to step 3 above. The card also shows which AI provider and model are active.

---

### Tab 1 — Scenario

Configure and generate the simulation.

Click **Generate scenario** to create a random set of pilots and trips. The page shows each pilot's profile (name, seniority, aircraft qualification) and each trip's key details: hours away from home, pay, route, and aircraft type.

**Sensitivity test** — lets you adjust one pilot's preference weights and see how the oracle ranking changes in real time. Useful for checking that the scoring formula behaves as expected before running AI evaluations.

---

### Tab 2 — Oracle

Calculates the rule-based reference ranking for each pilot. No AI is involved here — this is the benchmark everything else is measured against. Click **Compute oracle rankings** to see the ranked list for every pilot.

---

### Tab 3 — LLM Evaluation

This is where the AI ranks the trips. Run any combination of the three methods, then compare them on the Results tab.

**Rank-all** — sends one prompt with all trips and asks the AI to rank them from best to worst for each pilot. Fast (one AI call per pilot) but quality can drop on long lists.

**Scoring** — sends one trip at a time and asks the AI to score it 0–100. Scores are sorted to produce a ranking. Each pilot is run twice by default to check consistency; a third run is triggered automatically if the first two disagree significantly.

**Pairwise** — shows the AI two options at a time: *"Which does this pilot prefer?"* Uses an adaptive design — typically around 10 comparisons instead of hundreds — then builds a full ranking from the results. The results card shows how self-consistent the AI's choices were (high = 85%+, moderate = 70–85%, low = below 70%).

**Manual copy-paste (no backend)** — click **Build prompt** on any card to see the raw text. Copy it to any AI chat, paste the response back, and click **Submit**. The UI parses the response and computes all metrics automatically.

**LLM Evaluation Summary table** — below the evaluation cards, a side-by-side table shows the oracle ranking and the AI ranking for each pilot, so you can see where they agree and disagree at a glance.

---

### Tab 4 — Allocation

Simulates the actual bidding: pilots choose in seniority order, each claiming their highest-ranked available trip. Run oracle allocation and AI allocation side by side to see who received what and where the two approaches diverge.

---

### Tab 5 — Prompts

Shows the exact text sent to the AI. Useful for understanding what information the model received, or for diagnosing unexpected rankings.

---

### Tab 6 — Results

After running at least one evaluation, click **Refresh charts** to see the full analysis.

**Per-pilot cards** — each pilot gets a card with three sections:

1. **Within-method consistency** — how similar are the AI's rankings when the same method is run twice? A score near 1.0 means the AI gives the same answer every time (stable); below 0.7 means results are unreliable for that pilot and method.

2. **Between-method agreement** — how much do the three AI methods agree with each other and with the oracle? Shown as a grid: green cells = strong agreement, red = disagreement.

3. **Ranking comparison** — a table showing which trip each method places at each rank. Green checkmarks where a method matches the oracle; red marks where it differs.

**Aggregate summary** — averages all metrics across pilots and generates a plain-English finding, such as: *"Rank-all is most consistent (avg similarity = 0.93) and most aligned with oracle (0.88). Scoring shows the most variance across runs (0.71)."*

**Export options** — results can be downloaded as CSV or JSON for use in spreadsheets or further analysis.

---

## Key concepts

**Pairing**
A 2–5 day sequence of flights starting and ending at the pilot's home base. Each pairing has a route, aircraft type, hours in the air, total time away from home (TAFB), and pay.

**Line**
A full month's schedule made up of ~5 non-overlapping pairings. In real airline bidding, pilots bid on lines — not individual trips.

**Oracle**
A mathematical formula (not AI) that scores each trip according to a pilot's preference profile — pay, time away from home, report time. Serves as the reference "correct answer" in this simulation. It is a research baseline, not ground truth about what any pilot actually wants.

**Seniority**
A pilot's rank by years of service. Lower number = more senior = picks first. Seniority determines bid order only — any pilot can bid on any trip they are qualified to fly.

**Allocation**
The process of assigning exactly one trip or line to each pilot in seniority order. Each pilot claims their top-ranked available option; no two pilots receive the same item.

**TAFB (Time Away From Base)**
Total elapsed time from when a pilot reports for their first flight to when they return home after the last. Includes all flying, layovers, and ground time. A key factor in pilot preferences.

**Spearman ρ (rho)**
A number from −1 to +1 measuring how similar two ranked lists are. 1.0 means identical order; 0 means no relationship; −1 means completely reversed. Values above 0.8 are considered strong agreement in this context.

**BT fit quality (pairwise method)**
After the AI makes a series of head-to-head comparisons, this score measures how internally consistent those choices were — analogous to asking "did the AI contradict itself?" Human judges typically score 70–85%; values below 70% suggest the AI's choices were inconsistent.

---

## Scaling

The simulation can run at different scales. AI costs below assume automated mode (API calls); manual copy-paste is always free.

| Scenario | Pilots | Pairings (→ Lines) | Method | Estimated AI cost |
|---|---|---|---|---|
| Proof-of-concept | 5 | 5 pairings | Any (manual) | $0 |
| Small automated | 5 | 10 pairings | Rank-all or scoring | ~$0.10 |
| Medium | 10 | 20 pairings | Rank-all or scoring | ~$1 |
| Realistic (line mode) | 10 | 50 pairings → 10 lines | Pairwise | ~$2 |
| Large (line mode) | 50 | 500 pairings → 100 lines | Scoring | ~$15 |
| Production scale | 100 | 1,000 pairings → 200 lines | Scoring | ~$48 |

In line mode (scenarios at realistic scale and above), trips are grouped into monthly lines before pilots bid. This matches how real airline bidding works and is the recommended mode for larger experiments.

---

## Key simulation rules

**Seniority determines bid order only.** Every pilot can bid on any trip they are qualified to fly — seniority only controls who picks first.

**Aircraft qualification is a hard constraint.** A pilot certified on one aircraft type cannot be assigned a trip that requires another. Unqualified trips are excluded from their ranking.

**Pay is personalised.** Each pilot sees their own pay figure for each trip, based on their seniority-linked pay rate. More senior pilots earn more per hour, so the same trip looks different to different pilots.

**The oracle is a reference, not ground truth.** It encodes a reasonable model of pilot preferences based on profile attributes, but it does not represent what any individual pilot would actually choose.

---

# Appendix — Technical Reference

The sections below are intended for developers and researchers working with the codebase directly. They cover implementation details, mathematical models, and configuration options that are not needed to understand or use the simulation at a business level.

---

## Code structure

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

---

## Running from the command line

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

### Full backend + UI setup
```bash
cd pilot_bidding
pip install -r requirements-backend.txt
# Copy env.example to .env and fill in one API key and set LLM_PROVIDER
python proxy_server.py
# Then open http://127.0.0.1:8765/pilot_bidding_poc.html
```

### Line mode via Python API
```python
from generator import ScenarioGenerator
gen      = ScenarioGenerator()
pilots   = gen.build_pilots(n=5)
pairings = gen.build_pairings(n=25, pilots=pilots, max_b767=8)
lines    = gen.build_lines(pairings, n_lines=5, pairings_per_line=5)
```

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

---

## Method implementation details

> **Note**: The HTML/JavaScript POC and the Python backend implement the same three methods but with some differences in fitting algorithms. Where they differ this is called out explicitly below.

### A · Rank-all

The LLM receives a single prompt containing the pilot's full profile and all pairings/lines. It returns a JSON-ranked list in one call.

- No statistical model — purely prompt engineering.
- Evaluated directly with Spearman ρ against the oracle ranking.
- Fastest (1 call per pilot) but degrades on long lists as context grows.

---

### B · Pairwise — Bradley-Terry model

Each LLM call presents exactly two options: *"Which is better for this pilot?"* Votes are aggregated into a global ranking via the **Bradley-Terry (BT) model**.

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

**Zero-win floor**: items with $W_i = 0$ receive $s_i = 0.01$ instead of 0. This prevents rank collapse and ensures all items appear in the final ranking.

> **Python backend note**: `evaluator.BradleyTerryModel` uses a different fitting approach — log-parameterisation MLE solved via **scipy L-BFGS-B** (`theta[0]` fixed at 0 for identifiability). The end result is equivalent but the Python version leverages scipy for numerical stability. The MM algorithm above is the primary implementation used in the HTML POC.

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

This anchors the LLM's scale to the individual pilot's preferences rather than a generic rubric.

#### Multi-run aggregation

- 2 runs → final score = mean.
- 3 runs → final score = median (triggered when consistency is unstable after run 2).

#### Tie detection (mean gap + CI-overlap)

Items $A$ and $B$ are marked **tied** only when **both** conditions hold:

1. **Mean gap** is small: $|\mu_A - \mu_B| < 5$ points
2. **Confidence intervals overlap**: $(\mu_A - \sigma_A \leq \mu_B + \sigma_B)$ AND $(\mu_B - \sigma_B \leq \mu_A + \sigma_A)$

where $\mu$ is the mean score and $\sigma$ is the standard deviation across runs.

The mean-gap guard (condition 1) prevents two items whose CIs happen to be wide from being declared tied even when their means are clearly separated.

**Oracle tie parameters** (ground-truth grouping):
- Threshold: ±3 points between adjacent items triggers a tie.
- Max group size: 3 items; groups larger than 3 are split using item ID as a tiebreaker.

---

## Evaluation metrics

- **Spearman ρ**: rank correlation between LLM and oracle (−1 to 1). Converts both lists to rank positions (1st, 2nd, …) and measures order similarity. Tied items receive average ranks.
- **Top-1 match**: did the LLM's #1 choice match the oracle's #1 choice?
- **Eligibility accuracy**: fraction of items correctly flagged as qualified/unqualified.
- **Overall score** (0–100): `0.60 × ρ_normalised + 0.25 × top1 + 0.15 × eligibility` where ρ is normalised from [−1, 1] to [0, 1].

Note: these measure agreement with the oracle, not absolute pilot truth.

---

## Oracle sub-scores

Each pairing/line is scored on four sub-dimensions (each 0–100), then combined as a weighted sum using the pilot's weights.

### Pairing-level bounds

| Sub-score | Formula / bounds |
|---|---|
| TAFB | Linear: 18h → 100, 80h → 0 |
| Hotel nights (family pilot) | 100 − 35 × nights (floor 0); 0 nights = 100, 3 nights = 0 |
| Hotel nights (non-family pilot) | 20 + 40 × nights (cap 100); prefers more overnights |
| Report time | ≥ 07:00 → 100, ≤ 04:00 → 0, linear between |
| Credit pay | Normalised: best-paying pairing = 100, worst = 0; B767 carries a 4% pay premium |

### Line-level bounds

| Sub-score | Formula / bounds |
|---|---|
| Total TAFB | Linear: 90h → 100, 400h → 0 (5× per-pairing bounds) |
| Total nights (family pilot) | Ideal = 9 nights; −15 per night deviation (floor 0) |
| Total nights (non-family pilot) | 7 × nights up to 14 nights → 98, then −20 per night above 14 |
| Report time | Average of per-pairing report-time sub-scores |
| Credit pay | Normalised across lines; B767 pairings carry 4% premium within line totals |

### Oracle weights by pilot archetype

Oracle weights are derived automatically from age + family status — they are never set by hand.

| Archetype | `has_kids` | `is_mid_career` (age ≥ 40) | hotel_nights w | tafb w | credit_pay w |
|---|---|---|---|---|---|
| Family pilot | ✓ | either | 26 | 22 | 40 |
| Mid-career, no kids | ✗ | ✓ | 16 | 15 | 57 |
| Early-career, no kids | ✗ | ✗ | 9 | 9 | 70 |

`report_time` is fixed at 12 for all pilots. `credit_pay` fills to 100.

The direction of hotel_nights scoring also flips by archetype:
- **Family pilots** — fewer nights away = better (less time from home).
- **Non-family pilots** — more nights away = better (more per-diem income and flying experience).

---

## Scenario generation

### Pilot generation

Pilots are generated deterministically from a seeded PRNG (`pilot_seed=1234567`). The same seed always produces the same pilot pool, making results reproducible across runs and machines.

**Profile attributes drawn randomly:**

| Attribute | Range / options |
|---|---|
| Age | 26–58 (uniform) |
| Family status | Single · Married no kids · Married 1 child · Married 2+ kids · Single parent |
| Aircraft qualification | 65% dual (B737 + B767) · 35% B737-only |
| Min rest | 10–14 h |

**Pay rates** follow the Delta 2023 contract longevity table (6 steps mapped from age). Older pilots earn more per credit hour.

### Pairing generation

All pairings are **circular** — every trip starts and ends at the home base (default `BOS`). This mirrors real airline contract pairings.

**Structural choices:**

- **Legs**: sampled from `[2, 3, 3, 4]` (weighted toward 3-leg trips). `nights_away = num_legs − 1`, giving a natural 1–3 overnight spread.
- **Schedules are chained realistically**:
  - First departure: morning bank (06:00–09:30, 55%) or afternoon bank (14:00–18:00, 45%), rounded to 5-min slots.
  - Hotel departures (day 2+): always 06:00–08:30.
  - Ground turns: B767 needs 75–120 min; B737 needs 50–90 min.
- **Aircraft mix**: roughly 50/50 B737/B767 unless `max_b767` is set (e.g. when building 25 pairings for line mode, a 17/8 ratio is enforced so B737-only pilots always have biddable options).

**Pay computation** (per Delta 2023 contract):

```
credit_hours  = max(block_hours, tafb / 3.5)   # 1-for-3.5 rig
block_pay     = credit_hours × pilot.base_pay
per_diem      = tafb × $2.85
total_value   = block_pay + per_diem
```

### Line generation

Lines group pairings into monthly schedules — the unit pilots bid on in line mode.

**Key invariant — aircraft qualification**: B737 pairings are placed into lines before B767 pairings, ensuring the first lines in the pool are all-B737 so B737-only pilots always have fully-qualified options. A line requires qualification on every pairing it contains.

**Conflict detection**: `Line.has_conflicts()` checks whether any two pairings overlap on the calendar (using a 6-day spacing convention). The generator does not enforce conflict-free lines automatically — callers can filter or regenerate if needed.
