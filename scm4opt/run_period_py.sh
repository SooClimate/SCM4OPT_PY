#!/bin/bash
#PBS -q ve
#PBS -T intmpi
#PBS -b 1
#PBS -r n
#PBS -l elapstim_req=10:00:00
#PBS --custom vesetnum-lhost=2
#PBS -v OMP_NUM_THREADS=2
#PBS -v I_MPI_PIN_DOMAIN=omp
#PBS -o stdout.%s.%j
#PBS -e stderr.%s.%j

# Contribution runs of one year (default 2020) or of the whole period 1850-2100 (tAll)
# for the first nmem (default 1000) members of the contribution ensemble
# (scm4opt/data/ensemble/contrib_unc_h.txt), SSP1_19 with Carbon Monitor CO2. Each run
# reduces one emission species by 0.1 % in that period (or holds the land cover /
# removes a natural forcing); the reference run is --ctag=all --ems=co2i.
# ptype=periodb: the reduction covers all steps up to the end of that year (--byear,
# t < year+1) instead of the year alone (--tyear); tAll is the same run as with
# ptype=period.
# --reduce_after_cm=1: the reduction acts on the emissions after the Carbon Monitor step,
# so co2i/co2n reductions in 2019-2025 count and a reduction in 2018-2022 stays in its
# period (the reference order, --reduce_after_cm=0, reduces before it, which cancels
# the former and leaks the latter into 2023-2025).
#
# Usage:
#   bash scm4opt/run_period_py.sh                 # year 2020
#   bash scm4opt/run_period_py.sh 2010            # another year
#   bash scm4opt/run_period_py.sh tAll            # 1850-2100
#   bash scm4opt/run_period_py.sh 2010 periodb    # 1850-2010
#   qsub -v tyear=2010 scm4opt/run_period_py.sh   # PBS (tyear=tAll for 1850-2100), submitted
#                                                 # from the repository root
#   # all periods, one job each:
#   for y in tAll 1940 1950 1960 1970 1980 1990 2000 2010 2020 2030 2040 2050; do
#       qsub -N p${y} -v tyear=${y} ./scm4opt/run_period_py.sh
#   done
#   # periodb (up to the end of each year):
#   for y in 1940 1950 1960 1970 1980 1990 2000 2010 2020 2030 2040 2050; do
#       qsub -N pb${y} -v tyear=${y},ptype=periodb ./scm4opt/run_period_py.sh
#   done
# Optional environment: tyear (year or tAll), ptype (period or periodb), njobs (parallel
# jobs), nmem (members, default 1000), PYTHON (interpreter, default python3),
# mdl_dir (output folder).
#
# Output in rsl_dir (default csv_<ptype>_<year>_py/ under $mdl_dir or the current
# folder): ${rsl_dir}WORLD/COMB_IAMC_SSP1_19_<ems>_EMISS_IND_<case>_<ctag>.csv, one per
# run and member (the --cout5 report of scm4opt); joblog.txt in ${tmp_dir}. A rerun
# with the same year skips the jobs that already finished (parallel --resume).

year="${1:-${tyear:-2020}}"
ptype="${2:-${ptype:-period}}"
# tAll: no --tyear/--byear, so the reduction covers the whole period 1850-2100
if [ "${year}" = "tAll" ]; then
    period=""
elif [ "${ptype}" = "period" ]; then
    period="--tyear=${year}"
elif [ "${ptype}" = "periodb" ]; then
    period="--byear=${year}"
else
    echo "ptype must be period or periodb, not ${ptype}"; exit 2
fi

nmem="${nmem:-1000}"
py="${PYTHON:-python3}"

# output under $mdl_dir (default: the folder the script is started from; on the
# remote machine of user d0997 its data folder); qsub -v mdl_dir=... overrides it
[ -n "${PBS_O_WORKDIR:-}" ] && cd "${PBS_O_WORKDIR}"
if [ -z "${mdl_dir:-}" ] && [[ $USER == "d0997" ]]; then
    mdl_dir="/S/data00/G6008/d0997/contrib/"
fi
base="${mdl_dir:-${PWD}}"; base="${base%/}"
export rsl_dir="${rsl_dir:-${base}/csv_${ptype}_${year}_py/}"
export tmp_dir="${tmp_dir:-${base}/exe/}"
mkdir -p "${rsl_dir}" "${tmp_dir}"
njobs="${njobs:-$(( $(getconf _NPROCESSORS_ONLN) - 1 ))}"
# the repository root (this script is scm4opt/run_period_py.sh), else an installed package
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${script_dir}/__init__.py" ]; then pkg_root="$(dirname "${script_dir}")"; else pkg_root="${PWD}"; fi
model="scm4opt run"

# ensemble members: one line of --name=value flags per member
members_file="$(cd "${pkg_root}" && "${py}" -c 'from scm4opt.core.config import PACKAGE_DATA_DIR as d; print(d / "ensemble" / "contrib_unc_h.txt")')" || exit 2
member_lines() {
    head -n "${nmem}" "${members_file}"
}
mkdir -p "${rsl_dir}/WORLD/"
export tmp_dir="${tmp_dir}${ptype}_${year}_py/"
mkdir -p "${tmp_dir}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
member_lines > "${tmp_dir}members.txt"

# runs: the options of each run
cat > "${tmp_dir}runs.txt" <<'EOF'
--minus_ems=1 --rdc_ems=bc --ctag=minus_ems --ems=bc --rdc_mix=1
--minus_ems=1 --rdc_ems=ch4 --ctag=minus_ems --ems=ch4 --rdc_mix=1
--minus_ems=1 --rdc_ems=co --ctag=minus_ems --ems=co --rdc_mix=1
--ctag=all --ems=co2i
--minus_ems=1 --rdc_ems=co2i --ctag=minus_ems --ems=co2i
--minus_ems=1 --rdc_ems=co2n --ctag=minus_ems --ems=co2n
--minus_ems=1 --rdc_ems=co2l --ctag=minus_ems --ems=co2l
--minus_ems=1 --rdc_ems=n2o --ctag=minus_ems --ems=n2o --rdc_mix=1
--minus_ems=1 --rdc_ems=nh3 --ctag=minus_ems --ems=nh3 --rdc_mix=1
--minus_ems=1 --rdc_ems=nox --ctag=minus_ems --ems=nox --rdc_mix=1
--minus_ems=1 --rdc_ems=oc --ctag=minus_ems --ems=oc --rdc_mix=1
--minus_ems=1 --rdc_ems=so2 --ctag=minus_ems --ems=so2 --rdc_mix=1
--minus_ems=1 --rdc_ems=voc --ctag=minus_ems --ems=voc --rdc_mix=1
--minus_ems=1 --rdc_ems=c2f6 --ctag=minus_ems --ems=c2f6
--minus_ems=1 --rdc_ems=c3f8 --ctag=minus_ems --ems=c3f8
--minus_ems=1 --rdc_ems=c4f10 --ctag=minus_ems --ems=c4f10
--minus_ems=1 --rdc_ems=c5f12 --ctag=minus_ems --ems=c5f12
--minus_ems=1 --rdc_ems=c6f14 --ctag=minus_ems --ems=c6f14
--minus_ems=1 --rdc_ems=c7f16 --ctag=minus_ems --ems=c7f16
--minus_ems=1 --rdc_ems=cc4f8 --ctag=minus_ems --ems=cc4f8
--minus_ems=1 --rdc_ems=ccl4 --ctag=minus_ems --ems=ccl4
--minus_ems=1 --rdc_ems=cf4 --ctag=minus_ems --ems=cf4
--minus_ems=1 --rdc_ems=cfc11 --ctag=minus_ems --ems=cfc11
--minus_ems=1 --rdc_ems=cfc113 --ctag=minus_ems --ems=cfc113
--minus_ems=1 --rdc_ems=cfc114 --ctag=minus_ems --ems=cfc114
--minus_ems=1 --rdc_ems=cfc115 --ctag=minus_ems --ems=cfc115
--minus_ems=1 --rdc_ems=cfc13 --ctag=minus_ems --ems=cfc13
--minus_ems=1 --rdc_ems=cfc12 --ctag=minus_ems --ems=cfc12
--minus_ems=1 --rdc_ems=ch3br --ctag=minus_ems --ems=ch3br
--minus_ems=1 --rdc_ems=ch3ccl3 --ctag=minus_ems --ems=ch3ccl3
--minus_ems=1 --rdc_ems=ch3cl --ctag=minus_ems --ems=ch3cl
--minus_ems=1 --rdc_ems=halon1202 --ctag=minus_ems --ems=halon1202
--minus_ems=1 --rdc_ems=halon1211 --ctag=minus_ems --ems=halon1211
--minus_ems=1 --rdc_ems=halon1301 --ctag=minus_ems --ems=halon1301
--minus_ems=1 --rdc_ems=halon2402 --ctag=minus_ems --ems=halon2402
--minus_ems=1 --rdc_ems=hcfc141b --ctag=minus_ems --ems=hcfc141b
--minus_ems=1 --rdc_ems=hcfc142b --ctag=minus_ems --ems=hcfc142b
--minus_ems=1 --rdc_ems=hcfc22 --ctag=minus_ems --ems=hcfc22
--minus_ems=1 --rdc_ems=hcfc124 --ctag=minus_ems --ems=hcfc124
--minus_ems=1 --rdc_ems=hcfc132b --ctag=minus_ems --ems=hcfc132b
--minus_ems=1 --rdc_ems=hcfc133a --ctag=minus_ems --ems=hcfc133a
--minus_ems=1 --rdc_ems=hfc125 --ctag=minus_ems --ems=hfc125
--minus_ems=1 --rdc_ems=hfc134a --ctag=minus_ems --ems=hfc134a
--minus_ems=1 --rdc_ems=hfc143a --ctag=minus_ems --ems=hfc143a
--minus_ems=1 --rdc_ems=hfc152a --ctag=minus_ems --ems=hfc152a
--minus_ems=1 --rdc_ems=hfc227ea --ctag=minus_ems --ems=hfc227ea
--minus_ems=1 --rdc_ems=hfc23 --ctag=minus_ems --ems=hfc23
--minus_ems=1 --rdc_ems=hfc236fa --ctag=minus_ems --ems=hfc236fa
--minus_ems=1 --rdc_ems=hfc245fa --ctag=minus_ems --ems=hfc245fa
--minus_ems=1 --rdc_ems=hfc32 --ctag=minus_ems --ems=hfc32
--minus_ems=1 --rdc_ems=hfc365mfc --ctag=minus_ems --ems=hfc365mfc
--minus_ems=1 --rdc_ems=hfc4310mee --ctag=minus_ems --ems=hfc4310mee
--minus_ems=1 --rdc_ems=hfc134 --ctag=minus_ems --ems=hfc134
--minus_ems=1 --rdc_ems=hfc143 --ctag=minus_ems --ems=hfc143
--minus_ems=1 --rdc_ems=hfc41 --ctag=minus_ems --ems=hfc41
--minus_ems=1 --rdc_ems=nf3 --ctag=minus_ems --ems=nf3
--minus_ems=1 --rdc_ems=sf6 --ctag=minus_ems --ems=sf6
--minus_ems=1 --rdc_ems=so2f2 --ctag=minus_ems --ems=so2f2
--minus_ems=1 --rdc_ems=c8f18 --ctag=minus_ems --ems=c8f18
--minus_ems=1 --rdc_ems=ch2cl2 --ctag=minus_ems --ems=ch2cl2
--minus_ems=1 --rdc_ems=chcl3 --ctag=minus_ems --ems=chcl3
--rdc_lcc=WORLD --ctag=WORLD --ems=CO2
--rdc_rfc=rfc_volc --ctag=rfc_volc --ems=CO2
--rdc_rfc=rfc_solar --ctag=rfc_solar --ems=CO2
--rdc_rfc=rfc_DUST --ctag=rfc_DUST --ems=CO2
EOF

echo "${ptype} ${year}: $(wc -l < "${tmp_dir}runs.txt") runs x ${nmem} members, ${njobs} jobs -> ${rsl_dir}"
cd "${pkg_root}"
# {1}: run options, {2}: member flags (last, so the member's values win)
parallel --no-run-if-empty -j "${njobs}" --resume --joblog "${tmp_dir}joblog.txt" \
    -a "${tmp_dir}runs.txt" -a "${tmp_dir}members.txt" \
    eval "${py}" -m ${model} --scen=contrib --cout5="${rsl_dir}" --psrc=IAMC \
    --psce=SSP1_19 --use_cm=1 --reduce_after_cm=1 ${period} --rgn=WORLD {1} {2} '>' /dev/null
rc=$?
echo "done: parallel exit ${rc} (number of failed jobs; see ${tmp_dir}joblog.txt)"
exit ${rc}
