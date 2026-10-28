#!/usr/bin/env bash
# Run every RCMIP3 experiment (rcmip3_all_scenarios, 135) as an ensemble with the Python
# model, one experiment after the other, on a desktop. Each experiment is one call of
#   python3 -m scm4opt.ensemble --rname=<experiment> ...
# which runs the 500 ensemble members in parallel with GNU parallel and adds the
# experiment's standard options (--load_conc=all / nco2; scm4opt/scenarios/rcmip3_data.py).
#
# Usage (from anywhere):
#   bash scm4opt/run_rcmip3_all_py.sh                     # all experiments, all members
#   bash scm4opt/run_rcmip3_all_py.sh -j 8 --cases 0-49   # 50 members, 8 jobs
#   bash scm4opt/run_rcmip3_all_py.sh --only "ssp126 ssp585" --ocean-model hector
#   bash scm4opt/run_rcmip3_all_py.sh --start-from ssp245   # resume at an experiment
#   bash scm4opt/run_rcmip3_all_py.sh --skip-done           # skip finished experiments
#   bash scm4opt/run_rcmip3_all_py.sh --only esm-allGHG-ssp370 \
#        --out-dir /path/to/output/   # output folder
#   bash scm4opt/run_rcmip3_all_py.sh --dry-run -- --cm6mdl=MIROC6   # extra model flags
#
# Options
#   -j, --jobs N          parallel members per experiment (default: CPUs - 1)
#   --cases SPEC          member subset, 0-based (e.g. 0-49, 0,5,7; default all)
#   --ocean-model M       osc (default) or hector (results in rcmip_<rname>_hector/)
#   --out-dir DIR         output folder: results in DIR/rcmip_<rname>/, logs in DIR/exe/
#                         (default: the repository root, or $mdl_dir if set)
#   --only "A B ..."      run only these experiments (in rcmip3_all_scenarios order)
#   --start-from NAME     skip the experiments before NAME
#   --skip-done           skip an experiment whose results folder already holds one
#                         RCMIP3 CSV per selected member
#   --full-report         also write the per-step report <rname>_<case>_all.csv of every
#                         member (~27 MB each); default: only the RCMIP3 CSV (~2 MB)
#   --python PATH         Python interpreter (default: $PYTHON, else python3)
#   --dry-run             print the commands only (ensemble.py --dry-run)
#   --                    everything after is passed to every model run
#
# Results: <out>/rcmip_<rname>/, member logs in <out>/exe/<rname>/, this script's log
# <out>/exe/rcmip3_all_<date-time>.log; <out> = --out-dir, else $mdl_dir, else <repo>.
# Needs: Python 3.10+ with numpy, pandas, scipy; GNU parallel. If the Python cannot import
# them, the run stops before the first experiment (ensemble.py exit code 4).
# Runs with bash 3.2 (macOS) and newer (Linux); on Windows use WSL.

set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trunk="$(cd "${script_dir}/.." && pwd)"           # holds the scm4opt package
repo="${trunk}"                                   # the repository root
base="${mdl_dir:-${repo}}"; base="${base%/}"      # ensemble.py's base folder ($mdl_dir)
py="${PYTHON:-python3}"

jobs=""
out_dir=""
cases="all"
ocean=""
only=""
start_from=""
skip_done=0
full_report=0
dry_run=0
extra=()

while [ $# -gt 0 ]; do
    case "$1" in
        -j|--jobs)       jobs="$2"; shift 2 ;;
        --cases)         cases="$2"; shift 2 ;;
        --ocean-model)   ocean="$2"; shift 2 ;;
        --out-dir)       out_dir="$2"; shift 2 ;;
        --only)          only="$2"; shift 2 ;;
        --start-from)    start_from="$2"; shift 2 ;;
        --skip-done)     skip_done=1; shift ;;
        --full-report)   full_report=1; shift ;;
        --python)        py="$2"; shift 2 ;;
        --dry-run)       dry_run=1; shift ;;
        --)              shift; extra=("$@"); break ;;
        -h|--help)       sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

command -v "${py}" >/dev/null 2>&1 || { echo "Python not found: ${py}" >&2; exit 2; }
if [ -n "${out_dir}" ]; then                      # output folder for ensemble.py ($mdl_dir)
    mkdir -p "${out_dir}" || exit 2
    base="$(cd "${out_dir}" && pwd)"
    export mdl_dir="${base}/"
fi
if [ "${dry_run}" -eq 0 ]; then
    command -v parallel >/dev/null 2>&1 || {
        echo "GNU parallel not found (macOS: brew install parallel; Linux: apt/dnf install parallel)" >&2
        exit 2; }
fi

cd "${trunk}" || exit 2
scenarios="$("${py}" -m scm4opt.ensemble --list-rcmip3)" || exit 2
n_all=$(printf '%s\n' "${scenarios}" | wc -l | tr -d ' ')

mkdir -p "${base}/exe"
log="${base}/exe/rcmip3_all_$(date +%Y%m%d-%H%M%S).log"
suffix=""
[ "${ocean}" = "hector" ] && suffix="_hector"

# number of members selected by --cases (for --skip-done)
n_sel=$("${py}" - "${cases}" <<'PYEOF'
import sys
from scm4opt.ensemble import parse_cases
print(len(parse_cases(sys.argv[1], 500)))
PYEOF
) || n_sel=500

ok=0; skipped=0; failed=0; done_before=0; started=0
[ -z "${start_from}" ] && started=1
failed_list=""; skipped_list=""

{
    echo "SCM4OPT Python: all RCMIP3 experiments (${n_all}), $(date)"
    echo "  trunk ${trunk}"
    echo "  output ${base}/rcmip_<rname>${suffix}/"
    echo "  jobs=${jobs:-default} cases=${cases} ocean=${ocean:-osc} only=${only:-all}" \
         "start_from=${start_from:--} skip_done=${skip_done} full_report=${full_report}" \
         "extra=${extra[*]:-}"
} | tee "${log}"

i=0
while IFS= read -r rname; do
    [ -z "${rname}" ] && continue
    i=$((i + 1))
    if [ "${started}" -eq 0 ]; then
        [ "${rname}" = "${start_from}" ] && started=1 || continue
    fi
    if [ -n "${only}" ]; then
        case " ${only} " in *" ${rname} "*) ;; *) continue ;; esac
    fi
    if [ "${skip_done}" -eq 1 ]; then
        out="${base}/rcmip_${rname}${suffix}"
        n_done=$(ls "${out}"/SCM4OPTv40_rcmip_phase3_*_[0-9][0-9][0-9][0-9][0-9][0-9].csv 2>/dev/null | wc -l | tr -d ' ')
        if [ "${n_done}" -ge "${n_sel}" ]; then
            echo "[${i}/${n_all}] ${rname}: already done (${n_done} members), skipped" | tee -a "${log}"
            done_before=$((done_before + 1)); continue
        fi
    fi

    args=(--rname="${rname}" --cases="${cases}")
    [ -n "${jobs}" ] && args+=(-j "${jobs}")
    [ -n "${ocean}" ] && args+=(--ocean_model="${ocean}")
    [ "${dry_run}" -eq 1 ] && args+=(--dry-run)
    model=()
    [ "${full_report}" -eq 0 ] && model+=(--report_csv=0)
    [ ${#extra[@]} -gt 0 ] && model+=("${extra[@]}")
    [ ${#model[@]} -gt 0 ] && args+=(-- "${model[@]}")

    echo "[${i}/${n_all}] ${rname}  $(date +%H:%M:%S)" | tee -a "${log}"
    t0=$(date +%s)
    "${py}" -m scm4opt.ensemble "${args[@]}" < /dev/null 2>&1 | tee -a "${log}"
    rc=${PIPESTATUS[0]}
    dt=$(( $(date +%s) - t0 ))
    case ${rc} in
        0) ok=$((ok + 1));           echo "    ok (${dt} s)" | tee -a "${log}" ;;
        3) skipped=$((skipped + 1)); skipped_list="${skipped_list} ${rname}"
           echo "    skipped: missing inputs" | tee -a "${log}" ;;
        4) failed=$((failed + 1));   failed_list="${failed_list} ${rname}"
           echo "    STOPPED: the Python cannot run the model (see above)" | tee -a "${log}"
           break ;;
        *) failed=$((failed + 1));   failed_list="${failed_list} ${rname}"
           echo "    FAILED (exit ${rc}, ${dt} s)" | tee -a "${log}" ;;
    esac
done <<EOF
${scenarios}
EOF

{
    echo "summary: ${ok} ok, ${failed} failed, ${skipped} skipped (missing inputs)," \
         "${done_before} already done"
    [ -n "${failed_list}" ] && echo "  failed:${failed_list}"
    [ -n "${skipped_list}" ] && echo "  skipped:${skipped_list}"
    echo "log: ${log}"
} | tee -a "${log}"
[ "${failed}" -eq 0 ]
