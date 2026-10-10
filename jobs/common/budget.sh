# SBU accounting for every Snellius submission. Source it, do not execute it.
#
# The budget is 100k SBU counted from 25-08-2026, and every training job bills 256 SBU per
# hour (one A100 plus its quarter of the node), so a full-data 300-epoch run costs ~1,730.
# What is billed is the time a job actually runs, not its walltime: asking for a generous
# walltime is free, and a run cut off before it finishes is the expensive failure.
#
# Three numbers decide whether a submission goes ahead:
#   spent    what `accuse` reports since BUDGET_SINCE
#   queued   what the jobs already running or pending are still expected to cost
#   this     what the submission itself is expected to cost
# `budget_gate` refuses when spent + queued + this exceeds BUDGET_CAP, with no override,
# and also refuses a single sweep over SWEEP_CAP unless BUDGET_OK=1 is set.

BUDGET_SINCE="${BUDGET_SINCE:-25-08-2026}"
BUDGET_CAP="${BUDGET_CAP:-100000}"
SWEEP_CAP="${SWEEP_CAP:-3000}"
# From sacct: every flow_train job is allocated billing=256 for 1 GPU, 18 CPUs, 160G.
SBU_PER_GPU_HOUR="${SBU_PER_GPU_HOUR:-256}"
# Full train.100 at batch 128, warm page cache: the flm-parity runs took 6.7 h for 300.
SEC_PER_EPOCH="${SEC_PER_EPOCH:-80}"
# How much slower each run gets when two share a GPU. 1.5 is a placeholder until the
# calibration job measures it; packing pays as long as this stays below the pack size.
PACK_SLOWDOWN="${PACK_SLOWDOWN:-1.5}"
# Startup, the tau build, the cold first epochs off GPFS.
JOB_OVERHEAD_S="${JOB_OVERHEAD_S:-600}"

_BUDGET_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEDGER="${LEDGER:-${_BUDGET_REPO}/jobs/ledger.tsv}"

# Seconds in a SLURM duration: D-HH:MM:SS, HH:MM:SS or MM:SS. Anything else counts as 0.
_slurm_seconds() {
    awk -v t="$1" 'BEGIN {
        d = 0
        if (t !~ /^[0-9:-]+$/) { print 0; exit }
        if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
        n = split(t, p, ":"); s = 0
        for (i = 1; i <= n; i++) s = s * 60 + p[i]
        print d * 86400 + s
    }'
}

# Expected wall seconds of one job running EPOCHS epochs, PACK runs side by side.
est_job_seconds() {
    local epochs="$1" pack="${2:-1}"
    awk -v e="${epochs}" -v p="${pack}" -v s="${SEC_PER_EPOCH}" -v f="${PACK_SLOWDOWN}" \
        -v o="${JOB_OVERHEAD_S}" 'BEGIN { printf "%d", e * s * (p > 1 ? f : 1) + o }'
}

# SBU for a given number of wall seconds on one GPU.
seconds_to_sbu() {
    awk -v s="$1" -v r="${SBU_PER_GPU_HOUR}" 'BEGIN { printf "%.0f", s / 3600 * r }'
}

# SBU spent since BUDGET_SINCE, net of restitutions. BUDGET_SPENT overrides it, which is
# how a dry run on a machine without `accuse` still shows the arithmetic.
budget_spent() {
    if [[ -n "${BUDGET_SPENT}" ]]; then
        echo "${BUDGET_SPENT}"
        return 0
    fi
    command -v accuse >/dev/null 2>&1 || return 1
    accuse -u "${USER}" -s "${BUDGET_SINCE}" -e "$(date +%d-%m-%Y)" --sbu 2>/dev/null \
        | awk '/^Totals/ { printf "%.0f", $(NF-1) - $NF; found = 1 } END { exit !found }'
}

# What the user's running and pending jobs are still expected to cost. A job the ledger
# knows is charged its estimate minus what it has used; one it does not know is charged
# its whole remaining walltime, which is the worst case.
budget_queued() {
    command -v squeue >/dev/null 2>&1 || { echo 0; return 0; }
    local total=0 id left used est rem
    while IFS='|' read -r id left used; do
        [[ -z "${id}" ]] && continue
        est=""
        [[ -f "${LEDGER}" ]] && est="$(awk -F'\t' -v j="${id}" '$2 == j { e = $7 } END { print e }' "${LEDGER}")"
        if [[ -n "${est}" ]]; then
            rem="$(awk -v e="${est}" -v u="$(_slurm_seconds "${used}")" -v r="${SBU_PER_GPU_HOUR}" \
                'BEGIN { x = e - u / 3600 * r; printf "%.0f", (x > 0 ? x : 0) }')"
        else
            rem="$(seconds_to_sbu "$(_slurm_seconds "${left}")")"
        fi
        total=$((total + rem))
    done < <(squeue -u "${USER}" -h -t RUNNING,PENDING -o "%i|%L|%M" 2>/dev/null)
    echo "${total}"
}

# Refuse a submission expected to cost THIS SBU if it would break the cap.
budget_gate() {
    local this="$1" spent queued after
    if ! spent="$(budget_spent)"; then
        echo "BUDGET: cannot read spend (no accuse here). Set BUDGET_SPENT=<SBU> to dry-run" >&2
        echo "        the arithmetic; on Snellius this is a hard stop." >&2
        return 1
    fi
    queued="$(budget_queued)"
    after=$((spent + queued + this))
    printf 'budget    : spent %s + queued %s + this %s = %s of %s SBU (since %s)\n' \
        "${spent}" "${queued}" "${this}" "${after}" "${BUDGET_CAP}" "${BUDGET_SINCE}"
    if ((after > BUDGET_CAP)); then
        echo "BUDGET: refused — only $((BUDGET_CAP - spent - queued)) SBU remain under the cap and" >&2
        echo "        this needs ${this}. There is no override for the cap; shrink the sweep." >&2
        return 1
    fi
    if ((this > SWEEP_CAP)) && [[ -z "${BUDGET_OK}" ]]; then
        echo "BUDGET: refused — ${this} SBU is over the per-sweep cap of ${SWEEP_CAP}." >&2
        echo "        Re-run with BUDGET_OK=1 if this is deliberate." >&2
        return 1
    fi
    return 0
}

# One row per submitted job: when, the job id, what it was, and what it was expected to cost.
ledger_append() {
    local jobid="$1" project="$2" tier="$3" epochs="$4" pack="$5" est="$6" names="$7"
    if [[ ! -f "${LEDGER}" ]]; then
        printf 'date\tjobid\tproject\ttier\tepochs\tpack\test_sbu\tnames\n' > "${LEDGER}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%H:%M)" "${jobid}" "${project}" \
        "${tier}" "${epochs}" "${pack}" "${est}" "${names}" >> "${LEDGER}"
}
