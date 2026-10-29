#!/usr/bin/env bash
# Run the FICL tests. Each side runs from its own folder: with both dataset/ and
# system/ on Python's path, dataset/utils (a regular package) would shadow system/utils.
# Default: CPU tests only. Pass -m gpu or -m data to run those groups instead.
# Both sides always run; the exit code is non-zero if either side failed.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
args=("$@")
if [ ${#args[@]} -eq 0 ]; then args=(-m "not gpu and not data"); fi

python - <<'EOF'
import os
import sys

import numpy

prefix = os.path.realpath(sys.prefix)


def where(mod):
    path = os.path.dirname(os.path.realpath(mod.__file__))
    inside = path.startswith(prefix + os.sep)
    return path + ("" if inside else f"   <- outside this environment ({prefix}): see README, Known problems")


line = f"python {sys.version.split()[0]} ({sys.executable}), numpy {numpy.__version__}"
try:
    import torch
    line += f", torch {torch.__version__}"
    if int(numpy.__version__.split(".")[0]) >= 2 and torch.__version__.startswith("2.0."):
        line += "  <- PyTorch 2.0.x needs NumPy < 2: see README, Known problems"
except ImportError:
    torch = None
    line += ", torch not installed (torch tests are skipped)"
print(line)
print(f"  numpy from {where(numpy)}")
if torch is not None:
    print(f"  torch from {where(torch)}, CUDA available: {torch.cuda.is_available()}")
    from importlib import metadata

    others = []
    for dist in ("torchvision", "timm", "open_clip_torch"):        # pip names (open_clip has no __version__)
        try:
            others.append(f"{dist} {metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            others.append(f"{dist} not installed (its tests are skipped)")
    print("  " + ", ".join(others))
sys.stdout.flush()
EOF

failed=()
run_side() {
  echo "== $1 side"
  (cd "$here/$1" && python -m pytest tests -q "${args[@]}")
  local rc=$?
  if [ $rc -eq 5 ]; then
    echo "(no tests selected on this side)"
  elif [ $rc -ne 0 ]; then
    failed+=("$1")
  fi
}

run_side dataset
run_side system
if [ ${#failed[@]} -gt 0 ]; then
  echo "== failed on: ${failed[*]}"
  exit 1
fi
echo "== both sides passed"
