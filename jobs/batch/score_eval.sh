#!/bin/bash

#SBATCH --job-name=eval_score
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

# Merges the per-task .partN.jsonl files and scores them against the leaderboard targets.
# CPU only; no GPU, no partition pinned here (the submitter chooses). Cheap enough to run
# on the login node instead — submit/eval.sh prints that form — but wrapped so a scoring
# step chained after the array gets the same configuration the array ran with.
#
# Required env vars (set via --export):
#   RESULTS_DIR  — directory holding the manifests; must match the array's
#   NUM_SHARDS   — array task count, so a missing final part is caught before the merge
#                  destroys the parts
#
# MODEL comes from jobs/pinned_model.sh via _env.sh and is checked before anything runs;
# it selects which manifests are merged and scored.
#
# Optional:
#   EXPECT       — space-separated dataset:split=wer overriding the built-in targets
#   TOLERANCE    — absolute WER tolerance (default 0.1)
#   NO_ASSERT    — non-empty to report without ever failing
#
# Exit status is the verdict: non-zero means a split fell outside tolerance, so this is
# safe to chain with --dependency=afterok.

set -eo pipefail

source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

# MODEL is resolved by _env.sh. It selects which manifests are scored, so a wrong value
# here does not error — it silently scores a different run, or none.
require_pinned_model

NO_ASSERT_FLAG=()
[[ -n "${NO_ASSERT}" ]] && NO_ASSERT_FLAG=(--no-assert)

python -m scripts.evaluation.score_eval \
    --results-dir "${RESULTS_DIR:-${REPO_HOME}/results}" \
    --model "${MODEL}" \
    ${NUM_SHARDS:+--num-shards "${NUM_SHARDS}"} \
    ${TOLERANCE:+--tolerance "${TOLERANCE}"} \
    ${EXPECT:+--expect ${EXPECT}} \
    "${NO_ASSERT_FLAG[@]}"
