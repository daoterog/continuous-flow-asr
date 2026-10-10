#!/bin/bash

# Where the SBU budget stands: spent since BUDGET_SINCE, what queued jobs will still cost,
# what is left under the cap, and the last jobs the ledger recorded. Run on the login node.
#
#   ./jobs/budget.sh
#   ./jobs/budget.sh 25            # show the last 25 ledger rows

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common/budget.sh
source "${SCRIPT_DIR}/common/budget.sh"

if ! spent="$(budget_spent)"; then
    echo "cannot read spend: no accuse on this machine (run this on Snellius)" >&2
    exit 1
fi
queued="$(budget_queued)"
left=$((BUDGET_CAP - spent - queued))

printf 'since %s, cap %s SBU\n' "${BUDGET_SINCE}" "${BUDGET_CAP}"
printf '  spent   %8s SBU\n' "${spent}"
printf '  queued  %8s SBU  (running + pending, expected)\n' "${queued}"
printf '  left    %8s SBU  = %s full 300-epoch runs, or %s packed screen jobs\n' "${left}" \
    "$((left / $(seconds_to_sbu "$(est_job_seconds 300 1)")))" \
    "$((left / $(seconds_to_sbu "$(est_job_seconds 100 2)")))"

if [[ -f "${LEDGER}" ]]; then
    echo
    echo "last ledger rows (${LEDGER#"${SCRIPT_DIR}/../"}):"
    { head -1 "${LEDGER}"; tail -n +2 "${LEDGER}" | tail -n "${1:-10}"; } | column -t -s $'\t'
fi
