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

**How the alignment levels are built.** Everything is built on the noise-free answer table: the 44 attribute
cells of `matrix_jev.json` cut at 0.5 (the oracle's threshold and the hard kill rule's), plus the 17
category questions.

1. The *identity tree* splits the pool on the most even question, then splits each half the same way, until no
   question separates what is left. It stands for the questions that are naturally informative about *which
   object*. Reading its leaves in order keeps every branch together.
2. **`align`** cuts that leaf order into consecutive blocks of the profile's sizes, so each class is close to a
   set of whole branches.
3. **`mix05` / `mix15`** start from `align` and swap the classes of random pairs of objects until 5% / 15% of
   objects have moved. Sizes stay the same.
4. **`rand`** deals a random shuffle into the same sizes, so class boundaries ignore the natural splits.

The first build used 25% and 50% swaps. Those maps were already as unaligned as `rand` (alignment about 0.1
and 0.05), so the levels were moved to 5% and 15% to fill the gap. No arm had been run.

## How alignment is measured

All measures are noise-free, averaged over objects with each object equally likely, and in questions. The
final guess or action is not counted, because every arm spends exactly one.

| measure | meaning |
|---|---|
| `id_depth` | Questions the identity tree needs to isolate an object: **7.23** for every map |
| `pure_depth` | Questions until the object's branch of the identity tree holds only one class. This is arm B's cost noise-free: identity questions, stopping at the class |
| **`alignment`** | **1 − pure_depth / id_depth: the share of identity questions that stopping at the class saves.** 0 means classes separate only at the leaves; near 1 means the first questions already separate them |
| `class_depth` | Questions for a greedy tree that maximises information about the class (arm C, noise-free) |
| `ec2_depth` | Questions for the greedy EC² tree, uniform prior (arm D, noise-free) |
| `best_q_nmi` | The best single question's information about the class, as a share of the class entropy |
| `unresolvable` | Objects that share every answer with an object of a different class. No question separates them; only a by-name guess or an action can. This is the 20q version of fault-graph's kill/pause pair |

These trees are a design-time check on the answer table, not the arms themselves: the arms run P0a with
the graded likelihood. They are used as the basis for the pre-registered predictions.

## What the maps look like

Mean per alignment level (`index.json` has each map):

| construction | maps | alignment (min–max) | pure (B) | class (C) | EC² (D) | unresolvable |
|---|---:|---|---:|---:|---:|---:|
| `align` | 8 | 0.25–0.74 | 3.61 | 3.51 | 3.70 | 0.8 |
| `mix05` | 16 | 0.16–0.46 | 5.01 | 4.84 | 4.53 | 3.4 |
| `mix15` | 16 | 0.12–0.30 | 5.88 | 6.24 | 5.50 | 7.3 |
| `rand` | 16 | 0.00–0.08 | 6.96 | 9.78 | 6.74 | 23.0 |
| `cat17` | 1 | 0.16 | 6.09 | 5.01 | 5.49 | 0 |

Against the identity tree's 7.23 questions:

* **Stopping at the class (B) saves about half the questions on aligned maps and almost none on random ones.**
  Alignment falls as K grows: an aligned map with 16 classes saves 25%, one with 2 classes saves 70–74%.
* **Greedy class information (C) needs more questions than B on random maps (9.8 against 7.0).** It does so
  even without noise. This is the myopia that EC² was built to fix: a question can tell you nothing about the
  class on its own and still be needed to get there.
* **EC² (D) needs fewer questions than B on 47 of the 48 non-aligned maps, and no more than C on every `mix15`
  and `rand` map.** On aligned maps it is about even with B, and worse on the skewed ones
  (`k8_skew_align` 4.83 against B's 3.99).
* **`cat17` is the case where aiming at the class pays even though alignment is low (0.16).** Category questions
  split one category off at a time, so the identity tree never picks them, but they test the class directly.
  C needs 5.0 questions against B's 6.1.
* **Random maps have about 23 unresolvable objects.** Cutting at random across identical-answer pairs leaves
  objects that only an action can separate. Arms B–D need a rule for this (`FLYW1_DECISION_PREREG.md`).

## Reproducing

```bash
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.class_maps --self-test
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.class_maps --out llm_rl_scripts/twenty_questions/jev/results/flyw1_class_maps_20261118
```

The build is deterministic and takes a few seconds. Each map file records its `assignment_sha256_12`, the
matrix hash (`matrix_sha256_12`) and `generator_version` (1). An arm loads a map with `class_maps.load_map`
and should refuse a map whose version or hash differs from the one registered.
