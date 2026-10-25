# Experiment B — Semantic Salience (narrative × food-label)

**Design:** 2×2, all reactive (no goal, no history). N = narrative (N0 neutral /
N1 perishability), L = label (L0 generic "stale food" / L1 "milk"). 50 matched
seeds × 4 conditions = **200 episodes**, 0 API errors. `jev-latest`, real PO env,
20-step limit. 2026-10-25.

| Cond | Narrative | Label |
|---|---|---|
| B1 | neutral | stale food |
| B2 | neutral | milk |
| B3 | perishability | stale food |
| B4 | perishability | milk |

## Provenance

```bash
python -m llm_rl_scripts.text_nav.jev.analyze_exp_b \
    llm_rl_scripts/text_nav/jev/results/exp_B_20261025   # -> exp_b_analysis.json
```

Logs: [`results/exp_B_20261025/`](results/exp_B_20261025/) (`episodes.jsonl`,
`steps.jsonl.gz`, `run.json`). All rates are **opportunity-conditioned**: the
denominator is the number of logged steps where the action was admissible, so a
condition that merely saw more chances is not mistaken for a stronger tendency.

## Results (opportunity-conditioned rates, Wilson 95%)

**Take the target when it is offered:**

| | L0 generic | L1 milk |
|---|---|---|
| **N0 neutral** | 0.292 [0.24, 0.36] | 0.294 [0.19, 0.43] |
| **N1 narrative** | 0.475 [0.37, 0.58] | **0.867 [0.76, 0.93]** |

**Select the target over the fresh-food distractor (both admissible):**

| | L0 generic | L1 milk |
|---|---|---|
| **N0 neutral** | 0.000 (0/77) | 0.000 (0/19) |
| **N1 narrative** | 0.000 (0/21) | **0.833 (25/30)** |

**Eat when eating is legal (the fail action):**

| | L0 generic | L1 milk |
|---|---|---|
| **N0 neutral** | 0.209 [0.15, 0.28] | 0.352 [0.27, 0.45] |
| **N1 narrative** | **0.000 (0/363)** | **0.000 (0/390)** |

2×2 marginal effects (pooled rate differences):

| Rate | Label (milk−generic) | Narrative (N1−N0) | Interaction |
|---|---|---|---|
| take target \| offered | +0.26 | +0.35 | **+0.39** |
| select target \| both | +0.51 | +0.49 | **+0.83** |
| eat \| offered | +0.01 | **−0.27** | −0.14 |

Fridge insertions (reactive, no goal): B1 **0**, B2 **0**, B3 **62**, B4 **58**.

## Findings

1. **"Milk" is salient only inside the narrative — a strong interaction.** The
   label alone does nothing to target-taking (B2 0.294 ≈ B1 0.292); the narrative
   alone lifts it (B3 0.475); the two together jump to **0.867** (interaction
   +0.39). The word only matters once the perishability frame gives it meaning.

2. **Target *recognition* appears only under narrative + milk.** When the target
   and the fresh-food distractor are both on offer, Jev selects the target **0% of
   the time in B1–B3** (it actually favors the distractor) and **83% in B4**. This
   is the decisive answer to the design's §6 worry: B1–B3 are *generalized* food
   activity (or distractor-leaning), not target recognition; only the narrative +
   milk combination produces genuine *selective* attention to the perishable.

3. **The calorie narrative, not the label, suppresses eating.** Eating is fully
   killed under the narrative (0/363 and 0/390) but runs at 21–35% without it —
   and the milk label *alone* actually **raises** eating (B2 0.35 > B1 0.21):
   naming it a food makes it more tempting absent a reason not to. Narrative effect
   on eating −0.27; label effect ~0.

4. **"Refrigerate it" behavior is narrative-evoked.** Fridge insertions occur only
   under the narrative (62 / 58) and never in the neutral conditions (0 / 0) —
   context evokes the put-it-away action even with no goal stated, though (as
   Experiment A showed) it does not compose into task success without goal +
   history (all B conditions: 0 success).

## Interpretation

Experiment B isolates the **recognition / salience axis** and shows it is
genuinely moved by context, and *selectively* so: the perishability narrative plus
the "milk" label makes Jev preferentially take the perishable over a distractor
(0% → 83%) and refrigerate it, while the calorie frame independently and
completely suppresses eating. This is **not** generalized food-grabbing — B1–B3
establish the non-selective baseline. Combined with Experiment A (sequential
control needs goal + history) and Experiment C (within-context preferences are
transitive, scalar-rankable, with position bias only on ties), the program's
central claim holds: **recognition, preference coherence, and sequential control
are separable properties** — context reshapes the first, an explicit goal plus
history is required for the third, and the second is intact throughout. (Design
Outcome E.)
