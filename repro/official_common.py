"""Helpers for the scripts that run the official repositories (the torch side of the reproduction checks).

The official code is never edited. Scripts either run an official entry point as a subprocess with its own
command line (FedCOF), or start it in-process with runpy after wrapping a few of its functions to record
what it computes (AFL, STSA); the wrappers call the original functions unchanged.

Compatibility shims, applied only when the installed library lacks the old name, and reported in the dump:
    numpy.mat -> numpy.asmatrix      (removed in numpy 2.0; AFL's local_update uses it; same function)
    wandb      -> repro/stubs/wandb   (imported by FedCOF's server_aggregate.py and STSA's main.py at module level;
                                      never called with the default --wandb 0)
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STUBS = os.path.join(HERE, "stubs")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def commit_of(folder: str) -> str:
    p = os.path.join(folder, ".ficl_commit")
    if os.path.isfile(p):
        return _read(p)
    head = os.path.join(folder, ".git", "HEAD")
    if os.path.isfile(head):                                 # a git clone: resolve HEAD
        ref = _read(head)
        if ref.startswith("ref: "):
            rp = os.path.join(folder, ".git", ref[5:])
            if os.path.isfile(rp):
                return _read(rp)
        return ref
    return "unknown"


def require_commit(folder: str, expected: str, name: str) -> str:
    got = commit_of(folder)
    if got != expected:
        print(f"WARNING  {name}: {folder} is at commit {got[:7]}, the checks were written for {expected[:7]} "
              f"(python repro/fetch_official.py fetches that one)")
    return got


def module_available(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_modules(names, optional=()) -> list:
    missing = [n for n in names if not module_available(n)]
    for n in optional:
        if not module_available(n):
            print(f"note     optional module {n} is not installed")
    if missing:
        raise SystemExit(f"missing Python packages for the official code: {missing} "
                         f"(pip install them in this environment, see docs/OFFICIAL_REPRO.md section 4)")
    return missing


def shims_applied() -> list:
    """Install the compatibility shims that are needed here; return their names (recorded in the dump)."""
    out = []
    import numpy as np
    if not hasattr(np, "mat"):
        np.mat = np.asmatrix
        out.append("numpy.mat = numpy.asmatrix (numpy >= 2)")
    if not module_available("wandb"):
        if STUBS not in sys.path:
            sys.path.append(STUBS)
        out.append("wandb stub (not installed; never called)")
    return out


def subprocess_env() -> dict:
    env = dict(os.environ)
    if not module_available("wandb"):
        env["PYTHONPATH"] = os.pathsep.join([p for p in [env.get("PYTHONPATH", ""), STUBS] if p])
    return env


def run_logged(cmd: list, cwd: str, log_path: str) -> str:
    """Run a command, echo and store its output; raise on a non-zero exit. Returns the output."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print("run      " + " ".join(cmd) + f"   (cwd {cwd})")
    t0 = time.time()
    lines = []
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=subprocess_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            lines.append(line)
            if not line.startswith(("\r", " ")) and "it/s]" not in line and "s/it]" not in line:
                sys.stdout.write("  | " + line)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"command failed with exit code {code}; full output in {log_path}")
    print(f"done     {time.time() - t0:.0f} s, log {log_path}")
    return "".join(lines)


def extract_functions(path: str, names) -> dict:
    """Source of the named functions of a file, without importing the file (whose imports may be unavailable)."""
    with open(path) as f:
        source = f.read()
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in names and node.name not in found:
            found[node.name] = ast.get_source_segment(source, node)
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"{path}: functions {missing} not found (other commit?)")
    return found


def official_fed3r_functions(fed3r_folder: str, torch) -> dict:
    """fed3r_step, calc_ridge_params, normalize_W of Fed3R src/fed3r.py, executed from the file's own source."""
    names = ("fed3r_step", "calc_ridge_params", "normalize_W")
    src = extract_functions(os.path.join(fed3r_folder, "src", "fed3r.py"), names)
    ns = {"torch": torch}
    for n in names:
        exec(src[n], ns)
    return {n: ns[n] for n in names}


class FeatureBackbone:
    """Stand-in backbone for fed3r_step when the inputs already are features: model(x) -> x."""

    def __call__(self, x, return_feats=False):
        if return_feats:
            raise TypeError("plain feature extractor")       # fed3r_step then calls model(data)
        return x


def batches(torch, X, y, size: int = 1024):
    """(features, labels) mini-batches, as a DataLoader would give them, float32 like the official runs."""
    import numpy as np
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
    return [(Xt[i:i + size], yt[i:i + size]) for i in range(0, len(yt), size)]
