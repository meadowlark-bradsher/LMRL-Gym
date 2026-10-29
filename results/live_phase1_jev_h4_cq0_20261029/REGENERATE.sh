#!/usr/bin/env bash
# Regenerates this Jev result dir as registered in llm_rl_scripts/car_dealer_live/PREREGISTRATION_LIVE.md
# (tag prereg-live-20261029). Needs TYPESAFE_API_KEY; jev-latest is stochastic server-side, so a re-run
# reproduces the protocol and the analysis, not the exact decisions. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 4 --ask-cost 0 --n-offers 3 \
    --arms exact_ask,exact_noask,greedy_eig,random,jev_ask_plain,jev_ask_eig,jev_noask_plain,jev_noask_eig --repeats 5 --seed 21 --jev-backend http --out results/live_phase1_jev_h4_cq0_20261029
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_jev_h4_cq0_20261029 --B 5000 --seed 21
# verdicts and the rendered note need all six dirs analysed
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_live \
    --primary results/live_phase1_jev_h4_cq0_20261029/analysis.json --h2 results/live_phase1_jev_h2_cq0_20261029/analysis.json \
    --h3 results/live_phase1_jev_h3_cq0_20261029/analysis.json --cq3 results/live_phase1_jev_h4_cq3_20261029/analysis.json \
    --full results/live_phase1_jev_h4_cq0_full_20261029/analysis.json --scripted results/live_phase1_scripted3_20261029/h4_cq0/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase1_note --dirs results/live_phase1_jev_h4_cq0_20261029,results/live_phase1_jev_h2_cq0_20261029,results/live_phase1_jev_h3_cq0_20261029,results/live_phase1_jev_h4_cq3_20261029,results/live_phase1_jev_h4_cq0_full_20261029,results/live_phase1_jev_h4_cq0_replicate_20261029 \
    --verdicts results/live_phase1_jev_h4_cq0_20261029/verdicts.json --out llm_rl_scripts/car_dealer_live/JEV_RESULTS_LIVE.md
