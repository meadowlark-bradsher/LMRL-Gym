# Binned Car Dealer, Condition B — Jev results (pre-registered)

_Rendered by `render_jev_note.py` from frozen JSON; do not edit numbers by hand._

Hypotheses, rules, seeds and gates were fixed in [`PREREGISTRATION_B.md`](PREREGISTRATION_B.md) before any live call; verdicts were applied by `verdicts_b.py` and frozen before being read. Scripted baselines: [`SCRIPTED_RESULTS.md`](SCRIPTED_RESULTS.md). Design: [`DESIGN_B.md`](DESIGN_B.md). One deviation from the registered command lines: the deterministic arms (`exact`, `greedy_eig`, `myopic`) were included in each run so every pair is in one file; they are identical to the frozen scripted arms by construction (zero noise, one play per cell). `jev-latest` is stochastic server-side; `--seed` fixes only the option shuffle and cell order. All Jev claims are **stochastic**; the deterministic arms are exact-on-logs.

**Provenance note on S4.** The first analysis pass assigned Q-gap terciles by inclusive value bounds, which double-counted decisions whose gap tied at a boundary (most gaps are exactly 0). It was replaced by disjoint rank-based terciles before this note was written. The S4 verdict under the flawed binning was "not supported" for both arms at the primary horizon; the rule itself is unchanged.

## Headline (computed from the verdict files)

- **P1** (Δ total decision regret, jev_plain − random): -0.0656 [-0.0775, -0.0535] → **confirmed**
- **S1** (Δ per-decision EIG regret, jev_eig − jev_plain): -0.0055 [-0.0137, +0.0024] → **not supported**
- **S2** (Δ total decision regret, jev_eig − jev_plain): -0.0056 [-0.0108, -0.0004] → **not supported (EIG shown lowered decision regret)**
- **S3** (Δ questions asked, jev_eig − jev_plain): +0.1306 [+0.0167, +0.2583] → **confirmed**
- **S5_jev_plain** (Δ learner utility, jev_plain − exact): -0.0620 [-0.0836, -0.0422] → **confirmed**
- **S5_jev_eig** (Δ learner utility, jev_eig − exact): -0.0509 [-0.0700, -0.0334] → **confirmed**
- **S4** (jev_plain): large-gap ρ +0.449 [+0.258, +0.614] (n=280), small-gap ρ -0.434 [-0.535, -0.337] (n=279) → **not supported**
  - Reading: the entropy–regret relation *reverses sign* with the gap for `jev_plain`. When one option is clearly best, a diffuse choice goes with error; when the top options tie (gap = 0, i.e. several equally good actions), a diffuse choice goes with *lower* regret and the confident picks are the mistakes. The registered pattern asked for ρ ≈ 0 in the small-gap tercile, so it is "not supported" as written, but the document's underlying claim — that the relation changes with the gap — holds in a stronger form.
- **S4** (jev_eig): large-gap ρ +0.525 [+0.382, +0.643] (n=283), small-gap ρ -0.507 [-0.629, -0.394] (n=282) → **not supported**
  - Reading: the entropy–regret relation *reverses sign* with the gap for `jev_eig`. When one option is clearly best, a diffuse choice goes with error; when the top options tie (gap = 0, i.e. several equally good actions), a diffuse choice goes with *lower* regret and the confident picks are the mistakes. The registered pattern asked for ρ ≈ 0 in the small-gap tercile, so it is "not supported" as written, but the document's underlying claim — that the relation changes with the gap — holds in a stronger form.
- **S6** (Δ total decision regret, jev_plain − random (6 rounds)): -0.1037 [-0.1203, -0.0876] → **confirmed**
- 6 rounds, `jev_plain`: utility 0.7821 vs exact 0.7821; zero-regret decisions 1.000 of 699 (random: 0.845).
- 6 rounds, `jev_eig`: utility 0.7821 vs exact 0.7821; zero-regret decisions 1.000 of 630 (random: 0.845).

## Primary horizon — 2 rounds + final offer, easy

Source: `results/binned_b_jev_r2_easy_20261026` — `analysis.json` (`analyze_condition_b.py`, B = 5000, seed 11), `verdicts.json` (`verdicts_b.py`). Config: rounds = 2, regime easy (r = 2/5), K = 5 for stochastic arms, 1296 episodes, 0 error stubs, option order shuffled = True, model `jev-latest` via `http`.

### Verdicts (rules fixed in `PREREGISTRATION_B.md`)

| rule | quantity | prediction | estimate [95 % CI] | verdict |
|---|---|---|---|---|
| **P1** | Δ total decision regret, jev_plain − random | < 0 | -0.0656 [-0.0775, -0.0535] | **confirmed** |
| **S1** | Δ per-decision EIG regret, jev_eig − jev_plain | < 0 | -0.0055 [-0.0137, +0.0024] | **not supported** |
| **S2** | Δ total decision regret, jev_eig − jev_plain | not < 0 | -0.0056 [-0.0108, -0.0004] | **not supported (EIG shown lowered decision regret)** |
| **S3** | Δ questions asked, jev_eig − jev_plain | > 0 | +0.1306 [+0.0167, +0.2583] | **confirmed** |
| **S4** (jev_plain) | Spearman(choice entropy, decision regret) by Q-gap tercile | large-gap CI > 0 and small-gap CI includes 0 | large-gap ρ +0.449 [+0.258, +0.614] (n=280); small-gap ρ -0.434 [-0.535, -0.337] (n=279); overall ρ +0.014 [-0.102, +0.134] (n=838) | **not supported** |
| **S4** (jev_eig) | Spearman(choice entropy, decision regret) by Q-gap tercile | large-gap CI > 0 and small-gap CI includes 0 | large-gap ρ +0.525 [+0.382, +0.643] (n=283); small-gap ρ -0.507 [-0.629, -0.394] (n=282); overall ρ -0.109 [-0.211, -0.003] (n=847) | **not supported** |
| **S5_jev_plain** | Δ learner utility, jev_plain − exact | < 0 | -0.0620 [-0.0836, -0.0422] | **confirmed** |
| **S5_jev_eig** | Δ learner utility, jev_eig − exact | < 0 | -0.0509 [-0.0700, -0.0334] | **confirmed** |

### Arms

| arm | episodes | U | gain G | agreement | V* − U | Σ decision regret | zero-regret share | EIG regret / decision (bits) | questions | offers | final H (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `exact` | 72 | 0.7756 | 0.9779 | 1.000 | 0.0000 | 0.0000 | 1.000 | 0.1290 | 1.25 | 1.33 | 0.667 |
| `greedy_eig` | 72 | 0.7171 | 0.7778 | 0.778 | 0.0585 | 0.0585 | 0.741 | 0.0000 | 2.00 | 1.00 | 0.222 |
| `myopic` | 72 | 0.7167 | 0.6667 | 0.667 | 0.0590 | 0.0590 | 0.231 | 0.0958 | 0.00 | 2.17 | 0.667 |
| `random` | 360 | 0.6480 | 0.6416 | 0.917 | 0.1277 | 0.1220 | 0.504 | 0.2617 | 0.82 | 1.28 | 1.222 |
| `jev_plain` | 360 | 0.7136 | 0.6778 | 0.678 | 0.0620 | 0.0564 | 0.384 | 0.0685 | 0.37 | 1.96 | 0.569 |
| `jev_eig` | 360 | 0.7248 | 0.7154 | 0.717 | 0.0509 | 0.0508 | 0.442 | 0.0630 | 0.50 | 1.86 | 0.536 |

### Jev-specific

- `jev_plain`: well-formed 1.000; mean choice entropy 1.393 bits; mean confidence 0.401; chosen position counts {'0': 271, '1': 205, '2': 188, '3': 174}; chosen kind by turn {'1': {'ASK': 80, 'OFFER': 280}, '2': {'ASK': 52, 'OFFER': 219}, '3': {'OFFER': 207}}; offer tags {'AMBITIOUS': 671, 'SAFE': 35}.
  - small_gap: n = 279, gap ∈ [0.0000, 0.0000], ρ = -0.434 [-0.535, -0.337] (n=279), mean entropy 1.548, mean regret 0.0375
  - mid_gap: n = 279, gap ∈ [0.0000, 0.0064], ρ = -0.403 [-0.534, -0.273] (n=279), mean entropy 1.591, mean regret 0.0191
  - large_gap: n = 280, gap ∈ [0.0064, 0.1846], ρ = +0.449 [+0.258, +0.614] (n=280), mean entropy 1.042, mean regret 0.0161
- `jev_eig`: well-formed 1.000; mean choice entropy 1.436 bits; mean confidence 0.346; chosen position counts {'0': 267, '1': 220, '2': 177, '3': 183}; chosen kind by turn {'1': {'ASK': 122, 'OFFER': 238}, '2': {'ASK': 57, 'OFFER': 219}, '3': {'OFFER': 211}}; offer tags {'AMBITIOUS': 621, 'SAFE': 47}.
  - small_gap: n = 282, gap ∈ [0.0000, 0.0000], ρ = -0.507 [-0.629, -0.394] (n=282), mean entropy 1.566, mean regret 0.0357
  - mid_gap: n = 282, gap ∈ [0.0000, 0.0064], ρ = -0.476 [-0.613, -0.344] (n=282), mean entropy 1.591, mean regret 0.0111
  - large_gap: n = 283, gap ∈ [0.0064, 0.1846], ρ = +0.525 [+0.382, +0.643] (n=283), mean entropy 1.152, mean regret 0.0181

### Pairwise (cell-paired cluster bootstrap)

| pair | Δ utility | Δ Σ decision regret | Δ EIG regret / decision | Δ questions | Δ agreement |
|---|---|---|---|---|---|
| jev_plain - random | +0.0656 [+0.0323, +0.0982] | -0.0656 [-0.0775, -0.0535] | -0.3078 [-0.3483, -0.2671] | -0.4500 [-0.5750, -0.3250] | -0.2389 [-0.3361, -0.1472] |
| jev_eig - random | +0.0768 [+0.0419, +0.1097] | -0.0712 [-0.0845, -0.0575] | -0.3133 [-0.3540, -0.2715] | -0.3194 [-0.4583, -0.1778] | -0.2000 [-0.3000, -0.1083] |
| jev_eig - jev_plain | +0.0112 [-0.0013, +0.0248] | -0.0056 [-0.0108, -0.0004] | -0.0055 [-0.0137, +0.0024] | +0.1306 [+0.0167, +0.2583] | +0.0389 [-0.0028, +0.0861] |
| jev_plain - exact | -0.0620 [-0.0836, -0.0422] | +0.0564 [+0.0471, +0.0660] | -0.0387 [-0.0745, -0.0036] | -0.8833 [-1.0306, -0.7333] | -0.3222 [-0.4250, -0.2250] |
| jev_eig - exact | -0.0509 [-0.0700, -0.0334] | +0.0508 [+0.0414, +0.0608] | -0.0443 [-0.0823, -0.0078] | -0.7528 [-0.9028, -0.5917] | -0.2833 [-0.3833, -0.1944] |
| jev_plain - greedy_eig | -0.0035 [-0.0344, +0.0285] | -0.0021 [-0.0164, +0.0115] | +0.0724 [+0.0648, +0.0800] | -1.6333 [-1.7444, -1.5111] | -0.1000 [-0.2139, +0.0083] |
| jev_plain - myopic | -0.0031 [-0.0172, +0.0098] | -0.0026 [-0.0073, +0.0021] | -0.0225 [-0.0294, -0.0158] | +0.3667 [+0.2556, +0.4889] | +0.0111 [-0.0333, +0.0583] |

### Sanity gates

- Well-formed answers: jev_plain 1.0, jev_eig 1.0 → PASS.
- In-run `random` mean utility 0.6480 vs frozen scripted `random` 0.6556: difference -0.0076, scripted cell-bootstrap half-width 0.0151 → PASS.
- Error stubs: 0. Deterministic arms in this run reproduce the frozen scripted values by construction (same code, zero noise).

## Secondary horizon — 6 rounds + final offer, easy

Source: `results/binned_b_jev_r6_easy_20261026` — `analysis.json` (`analyze_condition_b.py`, B = 5000, seed 12), `verdicts.json` (`verdicts_b.py`). Config: rounds = 6, regime easy (r = 2/5), K = 3 for stochastic arms, 864 episodes, 0 error stubs, option order shuffled = True, model `jev-latest` via `http`.

### Verdicts (rules fixed in `PREREGISTRATION_B.md`)

| rule | quantity | prediction | estimate [95 % CI] | verdict |
|---|---|---|---|---|
| **S6** | Δ total decision regret, jev_plain − random (6 rounds) | < 0 | -0.1037 [-0.1203, -0.0876] | **confirmed** |
| **S5_jev_plain_6r** | Δ learner utility, jev_plain − exact (6 rounds) | < 0 | -0.0000 [-0.0000, +0.0000] | **not supported** |
| **S5_jev_eig_6r** | Δ learner utility, jev_eig − exact (6 rounds) | < 0 | -0.0000 [-0.0000, +0.0000] | **not supported** |

### Arms

| arm | episodes | U | gain G | agreement | V* − U | Σ decision regret | zero-regret share | EIG regret / decision (bits) | questions | offers | final H (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `exact` | 72 | 0.7821 | 1.0000 | 1.000 | 0.0000 | 0.0000 | 1.000 | 0.0778 | 0.00 | 2.67 | 0.333 |
| `greedy_eig` | 72 | 0.7821 | 1.0000 | 1.000 | 0.0000 | 0.0000 | 1.000 | 0.0000 | 6.00 | 1.00 | 0.000 |
| `myopic` | 72 | 0.7756 | 0.9702 | 1.000 | 0.0064 | 0.0064 | 0.867 | 0.2163 | 0.00 | 2.50 | 0.667 |
| `random` | 216 | 0.6702 | 0.7283 | 1.000 | 0.1118 | 0.1037 | 0.845 | 0.1719 | 1.66 | 1.36 | 1.108 |
| `jev_plain` | 216 | 0.7821 | 1.0000 | 1.000 | 0.0000 | 0.0000 | 1.000 | 0.0538 | 0.89 | 2.34 | 0.306 |
| `jev_eig` | 216 | 0.7821 | 1.0000 | 1.000 | 0.0000 | 0.0000 | 1.000 | 0.0507 | 0.77 | 2.14 | 0.282 |

### Jev-specific

- `jev_plain`: well-formed 1.000; mean choice entropy 1.586 bits; mean confidence 0.306; chosen position counts {'0': 193, '1': 153, '2': 224, '3': 129}; chosen kind by turn {'1': {'ASK': 33, 'OFFER': 183}, '2': {'ASK': 23, 'OFFER': 133}, '3': {'ASK': 34, 'OFFER': 84}, '4': {'ASK': 48, 'OFFER': 40}, '5': {'ASK': 41, 'OFFER': 25}, '6': {'ASK': 14, 'OFFER': 27}, '7': {'OFFER': 14}}; offer tags {'AMBITIOUS': 426, 'SAFE': 80}.
  - small_gap: n = 233, gap ∈ [0.0000, 0.0000], ρ = —, mean entropy 1.592, mean regret 0.0000
  - mid_gap: n = 233, gap ∈ [0.0000, 0.0000], ρ = —, mean entropy 1.584, mean regret 0.0000
  - large_gap: n = 233, gap ∈ [0.0000, 0.0000], ρ = —, mean entropy 1.582, mean regret 0.0000
- `jev_eig`: well-formed 1.000; mean choice entropy 1.639 bits; mean confidence 0.298; chosen position counts {'0': 139, '1': 147, '2': 171, '3': 173}; chosen kind by turn {'1': {'ASK': 65, 'OFFER': 151}, '2': {'ASK': 24, 'OFFER': 145}, '3': {'ASK': 34, 'OFFER': 83}, '4': {'ASK': 41, 'OFFER': 38}, '5': {'ASK': 3, 'OFFER': 43}, '6': {'OFFER': 3}}; offer tags {'AMBITIOUS': 378, 'SAFE': 85}.
  - small_gap: n = 210, gap ∈ [0.0000, 0.0000], ρ = —, mean entropy 1.604, mean regret 0.0000
  - mid_gap: n = 210, gap ∈ [0.0000, 0.0000], ρ = —, mean entropy 1.655, mean regret 0.0000
  - large_gap: n = 210, gap ∈ [0.0000, 0.0000], ρ = —, mean entropy 1.657, mean regret 0.0000

### Pairwise (cell-paired cluster bootstrap)

| pair | Δ utility | Δ Σ decision regret | Δ EIG regret / decision | Δ questions | Δ agreement |
|---|---|---|---|---|---|
| jev_plain - random | +0.1118 [+0.0903, +0.1341] | -0.1037 [-0.1203, -0.0876] | -0.3018 [-0.3535, -0.2510] | -0.7639 [-1.0556, -0.4583] | +0.0000 [+0.0000, +0.0000] |
| jev_eig - random | +0.1118 [+0.0903, +0.1341] | -0.1037 [-0.1203, -0.0876] | -0.3096 [-0.3619, -0.2587] | -0.8843 [-1.1574, -0.6065] | +0.0000 [+0.0000, +0.0000] |
| jev_eig - jev_plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0078 [-0.0131, -0.0023] | -0.1204 [-0.3056, +0.0602] | +0.0000 [+0.0000, +0.0000] |
| jev_plain - exact | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0181 [-0.0238, -0.0126] | +0.8935 [+0.6435, +1.1667] | +0.0000 [+0.0000, +0.0000] |
| jev_eig - exact | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0259 [-0.0323, -0.0198] | +0.7731 [+0.6065, +0.9491] | +0.0000 [+0.0000, +0.0000] |
| jev_plain - greedy_eig | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0670 [+0.0595, +0.0746] | -5.1065 [-5.3565, -4.8333] | +0.0000 [+0.0000, +0.0000] |
| jev_plain - myopic | +0.0064 [+0.0032, +0.0101] | -0.0064 [-0.0085, -0.0045] | -0.1037 [-0.1326, -0.0771] | +0.8935 [+0.6435, +1.1667] | +0.0000 [+0.0000, +0.0000] |

### Sanity gates

- Well-formed answers: jev_plain 1.0, jev_eig 1.0 → PASS.
- In-run `random` mean utility 0.6702 vs frozen scripted `random` 0.6767: difference -0.0065, scripted cell-bootstrap half-width 0.0133 → PASS.
- `jev_plain` − `exact` utility is exactly 0 in every cell: the arm matched the exact policy's realised utility everywhere (zero-regret share 1.000); the registered S5 rule (CI < 0) therefore reads "not supported", which here means *equal to optimal*, not worse.
- `jev_eig` − `exact` utility is exactly 0 in every cell: the arm matched the exact policy's realised utility everywhere (zero-regret share 1.000); the registered S5 rule (CI < 0) therefore reads "not supported", which here means *equal to optimal*, not worse.
- Error stubs: 0. Deterministic arms in this run reproduce the frozen scripted values by construction (same code, zero noise).

## Regenerate the analysis (the live logs are not regenerable)

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.analyze_condition_b results/binned_b_jev_r2_easy_20261026 --seed 11 --B 5000 --json results/binned_b_jev_r2_easy_20261026/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.verdicts_b results/binned_b_jev_r2_easy_20261026/analysis.json --horizon primary
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.analyze_condition_b results/binned_b_jev_r6_easy_20261026 --seed 12 --B 5000 --json results/binned_b_jev_r6_easy_20261026/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.verdicts_b results/binned_b_jev_r6_easy_20261026/analysis.json --horizon secondary
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.render_jev_note --primary results/binned_b_jev_r2_easy_20261026 --secondary results/binned_b_jev_r6_easy_20261026 --out llm_rl_scripts/car_dealer_binned/JEV_RESULTS.md
```
