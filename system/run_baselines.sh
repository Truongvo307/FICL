#!/usr/bin/env bash
# Run the closed-form FCIL baselines (Fed3R, FedCOF, STSA, STSA-E, AFL) on one shared setting, then print
# one table. Every method reads the same scenario files and the same feature cache (docs/BASELINES.md).
#
#   cd system
#   bash run_baselines.sh                      # the CIFAR-100 configs of configs/baselines/ (3 seeds each)
#   bash run_baselines.sh smoke                # the synthetic smoke configs
#   bash run_baselines.sh cifar100 -bb dinov2_vitb14 -alpha 0.1    # any extra flag goes to every run
#
# A dry run of every config comes first: nothing trains unless all settings and data check out.
set -euo pipefail
cd "$(dirname "$0")"
setting="${1:-cifar100}"
shift || true
case "$setting" in
  smoke)    configs=(smoke_fed3r smoke_fedcof smoke_stsa smoke_stsae smoke_afl) ;;
  cifar100) configs=(fed3r_cifar100_vit_in21k_b0.5 fedcof_cifar100_vit_in21k_b0.5 stsa_cifar100_vit_in21k_b0.5
                     stsae_cifar100_vit_in21k_b0.5 afl_cifar100_vit_in21k_b0.5) ;;
  *) echo "usage: bash run_baselines.sh [cifar100|smoke] [extra main.py flags]" >&2; exit 2 ;;
esac
for c in "${configs[@]}"; do
  python main.py -cfg "configs/baselines/$c.yaml" --dry_run "$@" > /dev/null
  echo "dry run ok: $c"
done
dirs=()
for c in "${configs[@]}"; do
  echo "=== $c"
  out=$(python main.py -cfg "configs/baselines/$c.yaml" "$@" 2>&1 | tee /dev/stderr | grep '^EXPERIMENT_DIR=' | tail -1)
  dirs+=("${out#EXPERIMENT_DIR=}")
done
python summarize_runs.py "${dirs[@]}"
