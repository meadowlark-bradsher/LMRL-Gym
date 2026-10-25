# PROVENANCE (generated from evidence.json -- do not edit by hand)

Source: `position_scores.jsonl` (600 records: 200 positions x 3 arms).

All claims are exact functions of that frozen file (abs tol 1e-09).

| claim | asserts | value |
|-------|---------|-------|
| `blunder.heuristic_legal` | blunder (>=300cp loss) rate for heuristic_legal | 0.4050 |
| `blunder.jev` | blunder (>=300cp loss) rate for jev | 0.5550 |
| `blunder.uniform_legal` | blunder (>=300cp loss) rate for uniform_legal | 0.5100 |
| `conf.high.match1` | match@1 in high-confidence tertile | 0.4091 |
| `conf.high.medcpl` | median non-mate cpL in high-confidence tertile | 22.0000 |
| `conf.low.match1` | match@1 in low-confidence tertile | 0.1212 |
| `conf.low.medcpl` | median non-mate cpL in low-confidence tertile | 522.0000 |
| `illegal.heuristic_legal` | illegal-move rate for heuristic_legal | 0.0000 |
| `illegal.jev` | illegal-move rate for jev | 0.0000 |
| `illegal.uniform_legal` | illegal-move rate for uniform_legal | 0.0000 |
| `match1.heuristic_legal` | best-move match rate for heuristic_legal | 0.2650 |
| `match1.jev` | best-move match rate for jev | 0.2050 |
| `match1.uniform_legal` | best-move match rate for uniform_legal | 0.0300 |
| `medcpl.heuristic_legal` | median non-mate centipawn loss for heuristic_legal | 105.5000 |
| `medcpl.jev` | median non-mate centipawn loss for jev | 425.5000 |
| `medcpl.uniform_legal` | median non-mate centipawn loss for uniform_legal | 266.0000 |
| `missedmate.heuristic_legal` | missed-forced-mate rate for heuristic_legal | 0.4091 |
| `missedmate.jev` | missed-forced-mate rate for jev | 0.5455 |
| `missedmate.uniform_legal` | missed-forced-mate rate for uniform_legal | 0.7273 |
| `paired.jev_vs_heuristic.blunder` | jev-heuristic paired blunder diff + 95% CI | +0.1500 [+0.0966, +0.2976] (n=200) |
| `paired.jev_vs_heuristic.match1` | jev-heuristic paired match@1 diff + 95% CI | -0.0600 [-0.2059, +0.0000] (n=200) |
| `paired.jev_vs_uniform.match1` | jev-uniform paired match@1 diff + 95% CI | +0.1750 [+0.1136, +0.2595] (n=200) |
| `tac.blunder.heuristic_legal` | tactical blunder rate for heuristic_legal | 0.2667 |
| `tac.blunder.jev` | tactical blunder rate for jev | 0.6333 |
| `tac.blunder.uniform_legal` | tactical blunder rate for uniform_legal | 0.7833 |
| `tac.match1.heuristic_legal` | tactical best-move match rate for heuristic_legal | 0.5333 |
| `tac.match1.jev` | tactical best-move match rate for jev | 0.2500 |
| `tac.match1.uniform_legal` | tactical best-move match rate for uniform_legal | 0.0333 |
