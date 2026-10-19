# SBU accounting for every Snellius submission. Source it, do not execute it.
#
# The budget is 100k SBU counted from 25-08-2026. An A100 hour is 128 SBU, but a job is
# charged for the largest of its shares of the node — 4 GPUs, 72 cores, 480 GiB, so one
# share is 1 GPU, 18 cores or 120 GiB. A training job needs 160G (the 102 GiB feature cache
# plus its processes), which is 2 shares, so even a one-run job is billed as 2 GPUs
# (sacct's billing=256). jobs/submit/train.sh therefore pairs runs, one per GPU, which is
# the same price as one run alone.
#
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
# PARTITION picks the node type. gpu_h100 (`scontrol show partition gpu_h100`, 2026-10-17):
# TRESBillingWeights=gres/gpu=192,cpu=12 with no memory weight, on 4-GPU, 64-core, 720 GiB
# nodes. So a share is 1 GPU or 16 cores, memory is free, and a run with 160G pays for one
# GPU — 192 SBU/h against the A100's 256. There are no H200s on Snellius.
PARTITION="${PARTITION:-gpu_a100}"
case "${PARTITION}" in
    gpu_a100) _rate=128 _cores=18 _mem_share=120 ;;
    gpu_h100) _rate=192 _cores=16 _mem_share="" ;;
    *) echo "budget.sh: unknown PARTITION ${PARTITION}" >&2; return 1 ;;
esac
SBU_PER_GPU_HOUR="${SBU_PER_GPU_HOUR:-${_rate}}"
# Per run, per GPU: one share of cores and at least 160G for the whole job (see
# jobs/batch/train.sh). MEM_G overrides the job's memory: on gpu_h100 it costs nothing, and
# whatever the job holds is page cache for train.960's 1.1 TB of latents.
CPUS_PER_RUN="${_cores}"
MEM_SHARE_G="${_mem_share}"
MEM_PER_RUN_G=120
MEM_MIN_G=160
# Measured on the stage-2 screen (flow-asr-flm-parity, 2026-10-12): the fulldata recipe under
# the snellius profile ran 49-58 s per epoch, evaluation included, depending on the node.
SEC_PER_EPOCH="${SEC_PER_EPOCH:-55}"
# Startup, the tau build, compile's warmup, the cold first epochs off GPFS.
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

# What a job of N runs asks for, one GPU and 18 cores per run: "GPUS CPUS MEM_G".
job_resources() {
    local n="${1:-1}" mem
    mem=$((MEM_PER_RUN_G * n))
    ((mem < MEM_MIN_G)) && mem=${MEM_MIN_G}
    echo "${n} $((CPUS_PER_RUN * n)) ${MEM_G:-${mem}}"
}

# GPUs a job of N runs is charged for: the largest of its GPU, CPU and memory shares.
charged_gpus() {
    local gpus cpus mem
    read -r gpus cpus mem <<< "$(job_resources "${1:-1}")"
    awk -v g="${gpus}" -v c="${cpus}" -v m="${mem}" -v cc="${CPUS_PER_RUN}" -v mc="${MEM_SHARE_G}" 'BEGIN {
        x = g; cs = c / cc; ms = (mc == "" ? 0 : m / mc)
        if (cs > x) x = cs; if (ms > x) x = ms
        printf "%d", (x == int(x) ? x : int(x) + 1)
    }'
}

# Expected wall seconds of one job running EPOCHS epochs. Its runs each have their own
# GPU, so the number of runs does not change it.
est_job_seconds() {
    awk -v e="$1" -v s="${SEC_PER_EPOCH}" -v o="${JOB_OVERHEAD_S}" \
        'BEGIN { printf "%d", e * s + o }'
}

# SBU for SECONDS of wall time on a job of N runs.
job_sbu() {
    awk -v s="$1" -v g="$(charged_gpus "${2:-1}")" -v r="${SBU_PER_GPU_HOUR}" \
        'BEGIN { printf "%.0f", s / 3600 * r * g }'
}

# SBU spent since BUDGET_SINCE, net of restitutions. BUDGET_SPENT overrides it, which is
# how a dry run on a machine without `accuse` still shows the arithmetic.
budget_spent() {
    if [[ -n "${BUDGET_SPENT}" ]]; then
        echo "${BUDGET_SPENT}"
        return 0
    fi
    command -v accuse >/dev/null 2>&1 || return 1
    # Bounded: accuse queries the accounting database and can stall, which would leave a
    # submission hanging before sbatch. A timeout fails the gate loudly instead.
    timeout 60 accuse -u "${USER}" -s "${BUDGET_SINCE}" -e "$(date +%d-%m-%Y)" --sbu 2>/dev/null \
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
            rem="$(job_sbu "$(_slurm_seconds "${left}")" 1)"
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
