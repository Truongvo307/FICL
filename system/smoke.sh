#!/usr/bin/env bash
# End-to-end smoke test of both regimes on the synthetic dataset (CPU, a few minutes):
#   single-round (closed-form) with Dirichlet clients, single-round with natural clients,
#   multi-round with rounds per task, client sampling off and curve points every 5 rounds,
#   and the image path (FedAvg training the tiny CNN from scratch on the synthetic images; needs torch).
# Every run checks its own result files; the script stops on the first failure.
#
#   cd system && ./smoke.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
(cd "$here/../dataset" && python make_synthetic.py)
cd "$here"
for cfg in configs/smoke_single.yaml configs/smoke_single_natural.yaml configs/smoke_multi.yaml \
           configs/smoke_multi_images.yaml; do
  echo
  echo "######## $cfg"
  bash ./run.sh "$cfg"          # bash: works even if a copy lost the executable bit
done
echo
echo "smoke test passed: both regimes and the image path ran, and every result check passed"
