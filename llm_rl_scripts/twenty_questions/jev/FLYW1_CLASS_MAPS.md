# Remediation-class maps for the 20q pool (FLYW-3)

Jira: FLYW-3, under epic FLYW-1 ("aim at a decision, not an identity"). Generator: `class_maps.py`.
Files: `results/flyw1_class_maps_20261118/` (`maps/<map_id>.json`, one per map, and `index.json`).

## What a class map is

A class map puts each of the 158 real objects in one class. A class stands in for a remediation: two
secrets in the same class need the same action, so a policy that aims at the action does not need to tell
them apart. Experiment 1 (`FLYW1_DECISION_PREREG.md`) runs its arms on these maps.

## The family: 57 maps

| what varies | values |
|---|---|
| number of classes K | 2, 4, 8, 16 |
| size profile | `bal`: equal sizes. `skew`: each class half the size of the one before (smallest kept at 2), so the largest holds about half the pool |
| alignment | `align`, `mix05`, `mix15`, `rand` (below) |
| seeds | 1 and 2 for `mix05`, `mix15` and `rand`; `align` is unique |

That gives 7 maps for each (K, profile) pair, 56 in all, plus **`cat17`**: the benchmark's own 17 categories.
`cat17` is a special case because the question bank has a category question for each class ("Is it a kind
of fruit?"), and those answers are exact even when attribute answers are noisy.

**How the alignment levels are built (generator version 3).** Alignment is measured against the questions
P0a actually asks.

1. **P0a's question tree.** P0a is deterministic at flip 0, so the control arm's disclosed run
   (`results/flyw1_a_calibration_20261118/a_games.json`, 158 games) gives every object's sequence of
   (question, answer) pairs. With the by-name guesses dropped, these sequences form a tree. Sorting them, yes
   before no and a branch before its extensions, gives a leaf order that keeps every branch together.
2. **`align`** cuts that leaf order into consecutive blocks of the profile's sizes, so each class is close to a
   set of whole branches of P0a's tree.
3. **`mix05` / `mix15`** start from `align` and swap the classes of random pairs of objects until 5% / 15% of
   objects have moved. Sizes stay the same.
4. **`rand`** deals a random shuffle into the same sizes, so class boundaries ignore the tree. Its
   assignments are unchanged from version 1.

**History.**

* **Versions 1 and 2** cut `align` from a balanced tree on the binarized answers (the most even question
  first). P0a does not follow that tree. Its graded EIG prefers the exact category questions and asks them
  one at a time: over A's 158 games it asked 602 category and 363 attribute questions. A trial run of arm B
  showed the mismatch (`FLYW1_DECISION_PREREG.md`, "Build amendments"), and version 3 rebuilt `align` and
  `mix` on P0a's own tree. The balanced-tree index is still reported as `alignment_bintree`.
* **Swap levels.** The first build used 25% and 50% swaps. Those maps were already as unaligned as `rand`, so
  the levels were moved to 5% and 15%. No arm had been run.

## How alignment is measured

All measures are noise-free and averaged over objects, each object equally likely.

| measure | meaning |
|---|---|
| `p0a_id_depth` | Questions P0a asks before it starts guessing by name: **6.11** on average |
| `p0a_pure_depth` | Questions until the object's branch of P0a's tree holds only one class |
| **`alignment`** | **1 − p0a_pure_depth / p0a_id_depth: the share of P0a's questions that stopping at the class saves.** 0 means the classes only separate where P0a starts guessing |
| `p0a_pure_cost` | Arm B's probes to decide: `p0a_pure_depth` plus its actions. Where a branch never becomes class-pure before A starts guessing, the objects sharing that branch are taken in the plurality order (biggest class first) |
| `p0a_unresolvable` | Objects in such a mixed guess-only group |
| `class_cost`, `ec2_cost` | Arm C's and D's probes to decide, from greedy trees on the binarized table (class information; EC² with uniform weights), with actions |
| `alignment_bintree`, `pure_depth`, `pure_cost`, `id_depth`, `unresolvable` | The same measures on the balanced binarized tree (version 1–2 basis) |
| `best_q_nmi` | The best single question's information about the class, as a share of the class entropy |

Arm A's cost is measured, not modelled: `flyw1_a_calibration.py` scores the control arm's games on every
map. All of these are design-time checks on the answer table and on the control arm, never on arms B–D, and
they are the basis for the pre-registered predictions.

## What the maps look like

Mean per construction, in probes to decide (`index.json` has each map):

| construction | maps | alignment (min–max) | B | C | D | `alignment_bintree` |
|---|---:|---|---:|---:|---:|---:|
| `align` | 8 | 0.04–0.30 | 6.11 | 6.12 | 5.95 | 0.19 |
| `mix05` | 16 | 0.01–0.15 | 6.70 | 7.25 | 6.53 | 0.15 |
| `mix15` | 16 | 0.005–0.04 | 7.26 | 8.93 | 7.02 | 0.11 |
| `rand` | 16 | 0.00–0.005 | 7.98 | 10.85 | 7.81 | 0.04 |
| `cat17` | 1 | 0.09 | 6.55 | 6.01 | 6.49 | 0.16 |

* **P0a's tree leaves little room for class stopping.** It peels off one category at a time, so a branch
  rarely holds a single class until a few objects are left. Even aligned maps save at most 30% of P0a's
  questions (`k2_bal_align`), and 16-class aligned maps save 4–9%.
* **Greedy class information (C) needs more probes than B on random maps (10.9 against 8.0),** even without
  noise. This is the myopia EC² was built to fix: a question can tell you nothing about the class on its own
  and still be needed to get there.
* **EC² (D) is slightly ahead of B on every synthetic construction (0.16–0.24 probes on average) and needs
  no more probes than C on every `mix15` and `rand` map.**
* **`cat17` is where aiming at the class (C) beats B** (6.01 against 6.55), because category questions test
  the class directly.

## Reproducing

```bash
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.class_maps --self-test
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.class_maps --out llm_rl_scripts/twenty_questions/jev/results/flyw1_class_maps_20261118
```

The build is deterministic, takes a few seconds, and needs the calibration run's `a_games.json`. Each map file records its `assignment_sha256_12`, the
matrix hash (`matrix_sha256_12`), the hash of the A run the tree was read from (`p0a_tree_sha256_12`) and
`generator_version` (3). An arm loads a map with `class_maps.load_map`
and should refuse a map whose version or hash differs from the one registered.
