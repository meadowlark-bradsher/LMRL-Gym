#!/usr/bin/env bash
# Regenerates this CD-SEL-1 result dir as registered in llm_rl_scripts/car_dealer_live/PREREGISTRATION_SEL.md
# (tag prereg-sel-20261030). Needs TYPESAFE_API_KEY; jev-latest is stochastic server-side, so a re-run
# reproduces the protocol and the analysis, not the exact decisions. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
D=results/live_sel_jev_h4_cq0_20261030
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 4 --ask-cost 0 --n-offers 3 \
    --arms exact_ask,exact_noask,random,jev_ask_plain,jev_ask_belief,jev_ask_pacc,jev_ask_prior,jev_ask_placebo,jev_ask_both,jev_noask_both \
    --repeats 5 --seed 51 --jev-backend http --out $D
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir $D --B 5000 --seed 51
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_sel --analysis $D/analysis.json \
    --scripted results/live_phase1_scripted3_20261029/h4_cq0/analysis.json --phase1 results/live_phase1_jev_h4_cq0_20261029/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_sel_note --dir $D --out llm_rl_scripts/car_dealer_live/SEL_RESULTS.md
