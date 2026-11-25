#!/usr/bin/env bash
# All reproduction checks of docs/OFFICIAL_REPRO.md, in order. Each step skips work whose output exists.
#
#   bash repro/run_all.sh DATA_DIR [OFFICIAL_PYTHON] [FICL_PYTHON]
#
# DATA_DIR         where torchvision may download CIFAR-100 (FedCOF, AFL, Fed3R runs; STSA uses ~/data/cifar100)
# OFFICIAL_PYTHON  interpreter of the environment that runs the official code (default: python)
# FICL_PYTHON      interpreter of the FICL environment, for run_ours.py and summarize.py (default: python)
# Steps can be run one by one; see the guide. A failed step stops the script.
set -euo pipefail
DATA=${1:?give the data folder}
OPY=${2:-python}
FPY=${3:-python}
cd "$(dirname "$0")/.."

$FPY repro/fetch_official.py
$OPY repro/run_fedcof_official.py --data "$DATA" --seeds 0 1 2 3 4 --python "$OPY"
$OPY repro/run_afl_official.py --data "$DATA" --seeds 1 2 3 --python "$OPY"
$OPY repro/run_fed3r_cifar.py --data "$DATA" --seed 0
$OPY repro/run_stsa_official.py --python "$OPY"

status=0
for f in repro/work/dumps/*.npz; do
  $FPY repro/run_ours.py "$f" || status=1
done
$FPY repro/summarize.py --md repro/work/REPRO_SUMMARY.md
exit $status
