# guarded_sbatch: sbatch, refusing any GPU job billed for more than it asks for. Source it.
#
# A GPU job is billed for the largest of its shares of the node (jobs/common/budget.sh,
# `gpu_share`): on gpu_a100 one GPU comes with 18 cores and 120 GiB, so 1 GPU with 160G is
# billed as 2 GPUs, and 16 cores on a MIG slice as 2 slices. Nothing in sbatch says so; the
# bill does, later. This reads what the job will actually ask for — the command line's
# --partition/--gpus/--cpus-per-task/--mem, falling back to the script's #SBATCH header —
# and when the charge exceeds the GPUs requested it submits nothing and prints the request
# both ways: cut down to fit the GPUs asked for, and widened to the GPUs paid for.
#
# GUARD_HINT, when set, is printed after a refusal: how the submitter that built the flags
# lets you change them.
#
# Otherwise it is sbatch with the same arguments and the same output, so `--parsable` and
# "Submitted batch job <id>" parsing work as before. CPU partitions pass through unchecked.
#
# Direct submissions go through jobs/sbatch.sh, which is this function as a command.

_GUARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=budget.sh
source "${_GUARD_DIR}/budget.sh"

# A --mem value in GiB, rounded up: 160G, 16000M, 1T; a bare number is MB, as in sbatch.
_mem_gib() {
    awk -v m="$1" 'BEGIN {
        u = toupper(substr(m, length(m)))
        n = (u ~ /[KMGT]/) ? substr(m, 1, length(m) - 1) : m
        g = (u == "T") ? n * 1024 : (u == "G") ? n : (u == "K") ? n / 1048576 : n / 1024
        printf "%d", (g == int(g) ? g : int(g) + 1)
    }'
}

guarded_sbatch() {
    local -A req=()
    local arg key val script="" i
    local -a args=("$@")
    # The script is the first argument that is a file; options before it are sbatch's.
    for ((i = 0; i < ${#args[@]}; i++)); do
        arg="${args[$i]}"
        if [[ "${arg}" != -* && -f "${arg}" ]]; then
            script="${arg}"
            break
        fi
        case "${arg}" in
            --partition=*|--gpus=*|--cpus-per-task=*|--mem=*|--gres=*)
                key="${arg%%=*}"; req["${key#--}"]="${arg#*=}" ;;
            -p|--partition|-G|--gpus|-c|--cpus-per-task|--mem|--gres)
                val="${args[$((i + 1))]}"
                case "${arg}" in -p) key=partition ;; -G) key=gpus ;; -c) key=cpus-per-task ;;
                                 *) key="${arg#--}" ;; esac
                req["${key}"]="${val}"; i=$((i + 1)) ;;
        esac
    done
    if [[ -n "${script}" ]]; then
        while read -r arg; do
            key="${arg%%=*}"; key="${key#--}"
            [[ -z "${req[${key}]+set}" ]] && req["${key}"]="${arg#*=}"
        done < <(sed -n 's/^#SBATCH[[:space:]]\{1,\}\(--[a-z-]*=[^[:space:]]*\).*/\1/p' "${script}")
    fi

    local part="${req[partition]:-}" gpus="${req[gpus]:-}" cpus="${req[cpus-per-task]:-1}"
    if [[ -z "${gpus}" && "${req[gres]:-}" == gpu* ]]; then
        gpus="${req[gres]##*:}"
    fi
    if gpu_share "${part}" >/dev/null; then
        gpus="${gpus:-1}"
        local mem charged cs ms
        # No --mem means Slurm's per-core default, which never outgrows the core share.
        mem="$([[ -n "${req[mem]:-}" ]] && _mem_gib "${req[mem]}" || echo 0)"
        charged="$(slice_charge "${part}" "${gpus}" "${cpus}" "${mem}")"
        read -r _ cs ms _ <<< "$(gpu_share "${part}")"
        # N whole shares as flags: what N GPUs pay for anyway, cores and memory included.
        # Memory is left as asked where it is not weighted (gpu_h100).
        slice_flags() {
            local n="$1" m
            m=$([[ "${ms}" == "-" ]] && { ((mem > 0)) && echo "${mem}"; } || echo $((n * ms)))
            echo "--gpus=${n} --cpus-per-task=$((n * cs))${m:+ --mem=${m}G}"
        }
        if ((charged > gpus)); then
            local why=()
            ((cpus > gpus * cs)) && why+=("${cpus} cores > ${cs} per GPU share")
            [[ "${ms}" != "-" ]] && ((mem > gpus * ms)) && why+=("mem ${mem}G > ${ms}G per GPU share")
            echo "REFUSED: ${gpus} GPU asked, billed as ${charged} on ${part}" \
                 "($(IFS=';'; echo "${why[*]}" | sed 's/;/; /g'))" >&2
            echo "  ${script:-the job} was not submitted. Ask for whole slices, one of:" >&2
            echo "    fit what you ask for:  $(slice_flags "${gpus}")" >&2
            echo "    use what you pay for:  $(slice_flags "${charged}")" >&2
            # A submitter that builds these flags itself says how to change them there.
            [[ -n "${GUARD_HINT:-}" ]] && echo "  ${GUARD_HINT}" >&2
            return 1
        fi
        # Under the share is not overspending, but it leaves paid-for cores or memory idle.
        if ((cpus < gpus * cs)) || { [[ "${ms}" != "-" ]] && ((mem < gpus * ms)); }; then
            echo "note: billed for ${gpus} whole GPU share(s) on ${part}; the same price buys" \
                 "$(slice_flags "${gpus}")" >&2
        fi
    fi
    sbatch "$@"
}
