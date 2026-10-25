# P1 fixed-position move quality

Records: 600  |  arms: heuristic_legal, jev, uniform_legal

## Per-arm (all positions)

cpL columns exclude forced-mate positions; mate handled via 'missed mate'.

| arm | n | illegal | mean cpL* | median cpL* | match@1 | match@3 | blunder | missed mate |
|-----|---|---------|-----------|-------------|---------|---------|---------|-------------|
| heuristic_legal | 200 | 0.0% | 1989.6 | 105.5 | 26.5% | 40.0% | 40.5% | 40.9% (n=22) |
| jev | 200 | 0.0% | 1067.1 | 425.5 | 20.5% | 33.0% | 55.5% | 54.5% (n=22) |
| uniform_legal | 200 | 0.0% | 2134.6 | 266.0 | 3.0% | 13.0% | 51.0% | 72.7% (n=22) |

*cpL = centipawn loss vs best legal move, non-mate positions only.

## Median centipawn loss by phase (non-mate positions)

| arm | opening | middlegame | tactical | endgame |
|-----|---|---|---|---|
| heuristic_legal | 77.5 (n=40) | 526.0 (n=60) | 0.0 (n=60) | 4.5 (n=40) |
| jev | 50.5 (n=40) | 577.0 (n=60) | 560.5 (n=60) | 2.0 (n=40) |
| uniform_legal | 113.5 (n=40) | 129.5 (n=60) | 660.0 (n=60) | 28.5 (n=40) |

## Paired comparison: jev - baseline (same positions)

Negative mean cpL difference = Jev loses *fewer* centipawns than the baseline (better).

| vs | metric | mean diff | 95% CI (cluster bootstrap) |
|----|--------|-----------|----------------------------|
| heuristic_legal | cp loss (non-mate) | -922.506 | [-3424.449, +235.031]  (n=178)|
| heuristic_legal | match@1 | -0.060 | [-0.206, +0.000]  (n=200)|
| heuristic_legal | blunder | +0.150 | [+0.097, +0.298]  (n=200)|
| uniform_legal | cp loss (non-mate) | -1067.539 | [-3769.346, +94.906]  (n=178)|
| uniform_legal | match@1 | +0.175 | [+0.114, +0.259]  (n=200)|
| uniform_legal | blunder | +0.045 | [-0.200, +0.196]  (n=200)|

