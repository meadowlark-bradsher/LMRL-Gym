# Pre-registration: are uninformed seller counters less acceptable than uninformed buyer counters?

**Registered:** 2026-10-25, before any probe decision was collected.
**Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` · **Regime:** hard (r = 0.55) · **Aid:** prior-only

## Origin

`HARD_RESULTS.md` §Exploratory located the hard-regime role asymmetry in the **prior-only
baseline**: with identical, mirror-symmetric information, the uninformed seller-as-responder's
first counter was acceptable to the buyer 0 % of the time versus 47 % for the uninformed
buyer-as-responder. That rested on few counters (responders countered in only 10–16 % of
episodes) and may be **selection** — countering only in the tough situations. This test
removes the selection problem and fixes the hypothesis before data.

## Hypothesis

**H:** Under identical mirror-symmetric information (prior-only aid), a seller-as-responder's
counter is less often acceptable to the starter than a buyer-as-responder's.

## Design — a single-decision probe with a forced counter

- **Contexts:** every feasible episode's opening in `results/jj_hard_20261025/prior_only/runs.jsonl`
  (480 contexts: 240 buyer-start → seller responds, 240 seller-start → buyer responds). The opening
  is the starter's first offer, made under the uniform prior with identical information in both
  roles (`HARD_RESULTS.md` A5 diagnosis), so it is a fair sample of Jev's openings.
- **Decision:** the responder is shown the opening and a menu of **offers only** — no ACCEPT,
  REJECT or WALK — so it must counter. Prior-only aid (uniform prior → mirror-identical
  P(accept) for the two roles), own-utility revealed, no salience, same instruction text.
- **Repeats:** K = 3 fresh decisions per context → ~1440 decisions, ~720 per cell. One API call each.
- **Unit of inference:** the feasible type pair (30 clusters; each contributes both cells).
- `--seed 4` is provenance only (`jev-latest` is stochastic server-side). Bootstrap seed 4.

```bash
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_discrete.jev.run_counter_probe \
  results/jj_hard_20261025/prior_only --repeats 3 --aid-mode prior_only --seed 4 \
  --out results/jj_counter_20261025
```

**What this measures, exactly:** the counter Jev produces *given that it must counter* — not its
natural decision whether to counter. That is the hypothesis as stated.

## Outcomes and decision rules (fixed in advance)

All Δ = seller_responds − buyer_responds, paired type-pair-clustered bootstrap, B = 5000, seed 4.

| | Prediction / quantity | Rule |
|---|---|---|
| **C1 (primary)** | Δ P(counter acceptable to starter) < 0 | CI excludes 0, negative → **confirmed**; CI excludes 0, positive → opposite sign; CI includes 0 → **not supported** |
| C2 | Δ utility of the counter to the starter < 0 | reported with CI |
| C3 | Δ own utility of the counter (is the seller's counter more self-serving?) | reported |
| C4 | Δ shown P(accept) of the chosen counter (does Jev-as-seller pick lower-rated options?) | reported |
| **C5 (selection check)** | acceptability by cell × whether the opening was acceptable to the responder | reported; if forced counters are acceptable at similar rates regardless of opening acceptability, the natural 0 % was selection |
| descriptive | chosen-deal distributions by cell; total-variation distance between the seller cell's choices mapped through the buyer/seller mirror and the buyer cell's choices | reported, no rule |

**Power.** ~720 decisions per cell over 30 clusters is adequate for an effect anywhere near the
observed 47-point gap. **A null here is informative**, unlike the six-cluster infeasible tests: it
would mean the natural asymmetry was selection, and H is not supported in its strong form.

## Sanity gates

- The information shown is mirror-symmetric by construction: the recomputed prior-only P(accept)
  dict for a buyer equals the seller's under the deal mirror (checked by the analyzer).
- Every recorded decision is an OFFER (forced menu respected); failures reported and excluded.
- 480 contexts, 30 feasible pairs, both cells populated.

## Analysis (fixed in advance)

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_counter_probe \
  results/jj_counter_20261025 --seed 4
```

Verdict applied by code and frozen before reading. No rule changes after data.

## Prediction and the honest alternative

From the exploratory data: Δ acceptable strongly negative. Honest alternative: the natural
0 %-vs-47 % was selection into tough openings, forced counters are symmetric, C1 is null, and the
hard-regime asymmetry is attributed to *when* roles counter rather than *what* they counter with.

## Limits accepted in advance

Forced menu (no opt-out); contexts reused from one logged run; prior-only aid only; hard regime
only; one prompt; `jev-latest` pinned by name.
