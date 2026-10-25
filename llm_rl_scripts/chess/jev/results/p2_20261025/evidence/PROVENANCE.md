# PROVENANCE (generated from evidence.json -- do not edit by hand)

Source: `combined_scores.jsonl` (P1 arms + 4 P2 LLM arms).

Exact functions of the frozen file (abs tol 1e-09).

| claim | asserts | value |
|-------|---------|-------|
| `blunder.claude_constrained` | blunder rate for claude_constrained | 0.1550 |
| `blunder.claude_free` | blunder rate for claude_free | 0.1050 |
| `blunder.gpt4o_constrained` | blunder rate for gpt4o_constrained | 0.5400 |
| `blunder.gpt4o_free` | blunder rate for gpt4o_free | 0.2700 |
| `blunder.heuristic_legal` | blunder rate for heuristic_legal | 0.4050 |
| `blunder.jev` | blunder rate for jev | 0.5550 |
| `blunder.uniform_legal` | blunder rate for uniform_legal | 0.5100 |
| `illegal.claude_constrained` | illegal rate for claude_constrained | 0.0050 |
| `illegal.claude_free` | illegal rate for claude_free | 0.0350 |
| `illegal.gpt4o_constrained` | illegal rate for gpt4o_constrained | 0.0100 |
| `illegal.gpt4o_free` | illegal rate for gpt4o_free | 0.1800 |
| `illegal.heuristic_legal` | illegal rate for heuristic_legal | 0.0000 |
| `illegal.jev` | illegal rate for jev | 0.0000 |
| `illegal.uniform_legal` | illegal rate for uniform_legal | 0.0000 |
| `match1.claude_constrained` | best-move match for claude_constrained | 0.4523 |
| `match1.claude_free` | best-move match for claude_free | 0.4352 |
| `match1.gpt4o_constrained` | best-move match for gpt4o_constrained | 0.1717 |
| `match1.gpt4o_free` | best-move match for gpt4o_free | 0.3171 |
| `match1.heuristic_legal` | best-move match for heuristic_legal | 0.2650 |
| `match1.jev` | best-move match for jev | 0.2050 |
| `match1.uniform_legal` | best-move match for uniform_legal | 0.0300 |
| `medcpl.claude_constrained` | median non-mate cpL for claude_constrained | 2.5000 |
| `medcpl.claude_free` | median non-mate cpL for claude_free | 3.0000 |
| `medcpl.gpt4o_constrained` | median non-mate cpL for gpt4o_constrained | 422.5000 |
| `medcpl.gpt4o_free` | median non-mate cpL for gpt4o_free | 39 |
| `medcpl.heuristic_legal` | median non-mate cpL for heuristic_legal | 105.5000 |
| `medcpl.jev` | median non-mate cpL for jev | 425.5000 |
| `medcpl.uniform_legal` | median non-mate cpL for uniform_legal | 266.0000 |
| `paired.jev_vs_claude_constrained.blunder` | jev-claude_constrained paired blunder diff + CI | +0.4000 [+0.3294, +0.4691] (n=200) |
| `paired.jev_vs_claude_constrained.match_best` | jev-claude_constrained paired match_best diff + CI | -0.2462 [-0.3889, -0.1782] (n=199) |
| `paired.jev_vs_gpt4o_constrained.blunder` | jev-gpt4o_constrained paired blunder diff + CI | +0.0150 [-0.0823, +0.0780] (n=200) |
| `paired.jev_vs_gpt4o_constrained.match_best` | jev-gpt4o_constrained paired match_best diff + CI | +0.0354 [-0.0526, +0.0886] (n=198) |
