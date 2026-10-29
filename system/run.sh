#!/usr/bin/env bash
# Run one FICL experiment config: first a dry run (settings, regime and the data of every seed), then
# the run itself. main.py checks its own result files at the end (check_run.py) and exits
# non-zero if a run or a check fails, so this script stops on the first problem.
#
#   ./run.sh configs/smoke_single.yaml
#   ./run.sh configs/smoke_multi.yaml -rpt 30 -t 1      # extra flags override the YAML values
#
# For long runs, keep the console output in logs/ (ignored by git):
#   mkdir -p logs && nohup ./run.sh configs/<config>.yaml > logs/<name>.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
if [ $# -lt 1 ]; then
  echo "usage: ./run.sh <config.yaml> [main.py flags]" >&2
  exit 2
fi
cfg="$1"
shift
python main.py -cfg "$cfg" "$@" --dry_run
python main.py -cfg "$cfg" "$@"
