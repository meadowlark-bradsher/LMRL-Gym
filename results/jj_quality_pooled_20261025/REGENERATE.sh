#!/usr/bin/env sh
# Rebuild the pooled two-run logs (labelled-secondary analysis in RESPONDER_RESULTS.md).
# They are pure concatenations of two committed logs and are not tracked themselves.
set -eu
cd "$(dirname "$0")/../.."
cat results/jj_quality_20261025/bayes/runs.jsonl      results/jj_quality_rep1_20261025/bayes/runs.jsonl      > results/jj_quality_pooled_20261025/bayes/runs.jsonl
cat results/jj_quality_20261025/prior_only/runs.jsonl results/jj_quality_rep1_20261025/prior_only/runs.jsonl > results/jj_quality_pooled_20261025/prior_only/runs.jsonl
echo "pooled logs rebuilt: $(wc -l < results/jj_quality_pooled_20261025/bayes/runs.jsonl) + $(wc -l < results/jj_quality_pooled_20261025/prior_only/runs.jsonl) episodes"
