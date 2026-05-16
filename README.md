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

## Evaluation metrics

- **Spearman ρ**: rank correlation between LLM and oracle (-1 to 1)
- **Top-1 match**: did LLM pick oracle's #1 choice?
- **Eligibility accuracy**: correct aircraft qualification flags

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
