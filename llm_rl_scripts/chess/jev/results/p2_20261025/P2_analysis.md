# P1 fixed-position move quality

Records: 1400  |  arms: claude_constrained, claude_free, gpt4o_constrained, gpt4o_free, heuristic_legal, jev, uniform_legal

## Per-arm (all positions)

cpL columns exclude forced-mate positions; mate handled via 'missed mate'.

| arm | n | illegal | mean cpL* | median cpL* | match@1 | match@3 | blunder | missed mate |
|-----|---|---------|-----------|-------------|---------|---------|---------|-------------|
| claude_constrained | 200 | 0.5% | 131.0 | 2.5 | 45.2% | 63.3% | 15.6% | 23.8% (n=21) |
| claude_free | 200 | 3.5% | 81.9 | 3.0 | 43.5% | 66.3% | 10.9% | 28.6% (n=21) |
| gpt4o_constrained | 200 | 1.0% | 1621.1 | 422.5 | 17.2% | 28.3% | 54.5% | 59.1% (n=22) |
| gpt4o_free | 200 | 18.0% | 1649.5 | 39.0 | 31.7% | 45.1% | 32.9% | 31.6% (n=19) |
| heuristic_legal | 200 | 0.0% | 1989.6 | 105.5 | 26.5% | 40.0% | 40.5% | 40.9% (n=22) |
| jev | 200 | 0.0% | 1067.1 | 425.5 | 20.5% | 33.0% | 55.5% | 54.5% (n=22) |
| uniform_legal | 200 | 0.0% | 2134.6 | 266.0 | 3.0% | 13.0% | 51.0% | 72.7% (n=22) |

*cpL = centipawn loss vs best legal move, non-mate positions only.

## Median centipawn loss by phase (non-mate positions)

| arm | opening | middlegame | tactical | endgame |
|-----|---|---|---|---|
| claude_constrained | 1.0 (n=40) | 28.0 (n=60) | 0.0 (n=60) | 0.5 (n=39) |
| claude_free | 2.5 (n=38) | 19.0 (n=59) | 0.0 (n=58) | 0.0 (n=38) |
| gpt4o_constrained | 57.0 (n=40) | 528.0 (n=60) | 517.0 (n=58) | 12.5 (n=40) |
| gpt4o_free | 9.0 (n=37) | 203.0 (n=43) | 228.0 (n=46) | 11.5 (n=38) |
| heuristic_legal | 77.5 (n=40) | 526.0 (n=60) | 0.0 (n=60) | 4.5 (n=40) |
| jev | 50.5 (n=40) | 577.0 (n=60) | 560.5 (n=60) | 2.0 (n=40) |
| uniform_legal | 113.5 (n=40) | 129.5 (n=60) | 660.0 (n=60) | 28.5 (n=40) |

## Paired comparison: jev - baseline (same positions)

Negative mean cpL difference = Jev loses *fewer* centipawns than the baseline (better).

| vs | metric | mean diff | 95% CI (cluster bootstrap) |
|----|--------|-----------|----------------------------|
| claude_constrained | cp loss (non-mate) | +936.056 | [+311.086, +3122.573]  (n=178)|
| claude_constrained | match@1 | -0.246 | [-0.389, -0.178]  (n=199)|
| claude_constrained | blunder | +0.400 | [+0.329, +0.469]  (n=200)|
| claude_free | cp loss (non-mate) | +1016.488 | [+380.089, +3212.343]  (n=172)|
| claude_free | match@1 | -0.233 | [-0.364, -0.182]  (n=193)|
| claude_free | blunder | +0.450 | [+0.305, +0.578]  (n=200)|
| gpt4o_constrained | cp loss (non-mate) | -552.062 | [-2864.836, +56.746]  (n=176)|
| gpt4o_constrained | match@1 | +0.035 | [-0.053, +0.089]  (n=198)|
| gpt4o_constrained | blunder | +0.015 | [-0.082, +0.078]  (n=200)|
| gpt4o_free | cp loss (non-mate) | -479.434 | [-2124.523, +248.190]  (n=145)|
| gpt4o_free | match@1 | -0.116 | [-0.235, -0.058]  (n=164)|
| gpt4o_free | blunder | +0.285 | [+0.200, +0.361]  (n=200)|
| heuristic_legal | cp loss (non-mate) | -922.506 | [-3424.449, +235.031]  (n=178)|
| heuristic_legal | match@1 | -0.060 | [-0.206, +0.000]  (n=200)|
| heuristic_legal | blunder | +0.150 | [+0.097, +0.298]  (n=200)|
| uniform_legal | cp loss (non-mate) | -1067.539 | [-3769.346, +94.906]  (n=178)|
| uniform_legal | match@1 | +0.175 | [+0.114, +0.259]  (n=200)|
| uniform_legal | blunder | +0.045 | [-0.200, +0.196]  (n=200)|

