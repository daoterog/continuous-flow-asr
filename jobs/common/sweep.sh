# The standard shape of a sweep: tiers, packing, and one budget gate for the whole sweep.
# Source it from a sweep script, do not execute it:
#
#   source "${REPO_DIR}/jobs/common/sweep.sh"
#   tier_defaults                      # EPOCHS, SEEDS, PACK from TIER
#   queue_run "<name>" <overrides...>  # once per run
#   flush_runs                         # pack, price, gate, submit (or print, without SUBMIT=1)
#
# Tiers. Each picks a run-kind profile (configs/profiles/run/), which owns the epoch
# count and the eval cadence; explicit EPOCHS / SEEDS / PACK in the environment still win.
#
#   screen   --run sweep   100 epochs, 1 seed, 2 runs per job    the default: ranks arms
#   confirm  --run full    300 epochs, 2 seeds, 2 runs per job   only the screen's winner(s)
#
# A job of PACK runs gives each its own GPU. Two runs cost what one does, since a job is
# billed for 2 GPUs either way (jobs/common/budget.sh), so an odd run out wastes a GPU.
#
# EPOCHS here only prices the sweep and names its runs. Set it explicitly to change the
# length: tier_defaults then sets EPOCH_OVERRIDE, which a sweep adds to its runs.
#
# Screening at 100 is backed by the flm-parity runs: their sampled-WER ranking at epoch 100
# matched the final one exactly, and the gaps to the control correlated at 0.97. That was
# epoch 100 inside a 300-epoch schedule, so a 100-epoch run is a close proxy, not the same
# thing. Its numbers are only comparable to other runs at the same EPOCHS, which is why a
# screen always carries its own control.

# shellcheck source=budget.sh
source "$(dirname "${BASH_SOURCE[0]}")/budget.sh"

tier_defaults() {
    TIER="${TIER:-screen}"
    EPOCH_OVERRIDE=""
    [[ -n "${EPOCHS}" ]] && EPOCH_OVERRIDE="train.num_epochs=${EPOCHS}"
    case "${TIER}" in
        screen)  TRAIN_RUN=sweep; : "${EPOCHS:=100}" "${SEEDS:=1}" "${PACK:=2}" ;;
        confirm) TRAIN_RUN=full;  : "${EPOCHS:=300}" "${SEEDS:=0 1}" "${PACK:=2}" ;;
        *) echo "unknown TIER=${TIER}; expected screen or confirm" >&2; exit 1 ;;
    esac
    export TIER TRAIN_RUN
}

_RUN_NAMES=()
_RUN_OVERRIDES=()

# queue_run NAME OVERRIDE...  — one run; NAME is only for the printout.
queue_run() {
    _RUN_NAMES+=("$1")
    shift
    _RUN_OVERRIDES+=("$*")
}

# Pack the queued runs PACK to a job, price the sweep, gate it once, then submit or print.
flush_runs() {
    local n=${#_RUN_NAMES[@]} jobs=0 total=0 i j size job_s job_sbu wall
    ((n == 0)) && { echo "no runs queued" >&2; return 1; }
    local groups=()
    for ((i = 0; i < n; i += PACK)); do groups+=("${i}"); done

    echo "tier      : ${TIER} (--run ${TRAIN_RUN})  epochs ${EPOCHS}  seeds '${SEEDS}'  ${PACK} run(s) per job, one GPU each"
    for i in "${groups[@]}"; do
        size=$((n - i < PACK ? n - i : PACK))
        job_s="$(est_job_seconds "${EPOCHS}")"
        total=$((total + $(job_sbu "${job_s}" "${size}")))
        jobs=$((jobs + 1))
    done
    echo "plan      : ${n} runs in ${jobs} GPU jobs, expected ${total} SBU"
    echo "            (${SEC_PER_EPOCH} s/epoch + ${JOB_OVERHEAD_S} s, one GPU per run, ${SBU_PER_GPU_HOUR} SBU/GPU-h, a job billed >=2 GPUs)"

    if [[ -n "${SUBMIT}" ]]; then
        budget_gate "${total}" || exit 1
    fi

    for i in "${groups[@]}"; do
        size=$((n - i < PACK ? n - i : PACK))
        job_s="$(est_job_seconds "${EPOCHS}")"
        # Walltime is free — billing is on elapsed time — so ask for twice the estimate
        # plus half an hour. A run cut off before it finishes is the expensive failure.
        wall="$(awk -v s="${job_s}" 'BEGIN { w = 2 * s + 1800; if (w > 432000) w = 432000;
            printf "%02d:%02d:%02d", w / 3600, (w % 3600) / 60, w % 60 }')"
        local args=() label="" words=()
        for ((j = i; j < i + size; j++)); do
            ((j > i)) && args+=("::")
            # Split on spaces without globbing: `[a,b]` must not match files in the repo.
            read -ra words <<< "${_RUN_OVERRIDES[$j]}"
            args+=("${words[@]}")
            label="${label}${label:+ + }${_RUN_NAMES[$j]}"
        done
        echo "=== ${label}"
        if [[ -n "${SUBMIT}" ]]; then
            WALLTIME="${wall}" EXPECTED_S="${job_s}" EPOCHS="${EPOCHS}" BUDGET_GATED=1 \
                ./jobs/submit/train.sh "${CONFIG}" "${args[@]}"
        else
            echo "    TRAIN_RUN=${TRAIN_RUN} WALLTIME=${wall} ./jobs/submit/train.sh ${CONFIG} ${args[*]}"
        fi
    done

    echo
    if [[ -n "${SUBMIT}" ]]; then
        echo "submitted ${jobs} GPU jobs (${n} runs), expected ${total} SBU"
    else
        echo "${jobs} GPU jobs NOT submitted (dry run). SUBMIT=1 to send them; expected ${total} SBU."
    fi
}
