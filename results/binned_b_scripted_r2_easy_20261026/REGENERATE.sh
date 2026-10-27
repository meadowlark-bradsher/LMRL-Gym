#!/bin/sh
# Regenerate this directory from scratch (deterministic given the seeds). ~15 s.
set -e
cd "$(dirname "$0")/../.."
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.run_condition_b --arms exact,greedy_eig,myopic,random --rounds 2 --regime easy --repeats 20 --seed 0 --out results/binned_b_scripted_r2_easy_20261026
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.analyze_condition_b results/binned_b_scripted_r2_easy_20261026 --seed 0 --B 5000 --json results/binned_b_scripted_r2_easy_20261026/analysis.json
gzip -f results/binned_b_scripted_r2_easy_20261026/runs.jsonl
