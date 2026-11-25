#!/usr/bin/env python
"""Download the four official repositories at the commits that were checked, without git.

    python repro/fetch_official.py                 # into repro/official_src/<Name>
    python repro/fetch_official.py --dest /data/official

Each repository is fetched as the GitHub archive of the exact commit, unpacked, and stamped with
.ficl_commit. Existing folders with the right stamp are kept. Nothing of these repositories is copied into
FICL: they stay in their own folders, under their own licenses (Fed3R GPL-3.0; FedCOF, STSA and AFL have no
license file at these commits, so they are read-only references), and are never redistributed.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import urllib.request
import zipfile

REPOS = {
    "Fed3R": ("Erosinho13/Fed3R", "c2a8e5de59e4b86efaa351c241a4a3357bcdab34"),
    "FedCOF": ("dipamgoswami/FedCOF", "960014f1535b9a63f902b2f97a151b3dc3602294"),
    "STSA": ("Yuqin-G/STSA", "aaaa006e0b63ebc637b16118925b0e19730c7d25"),
    "AFL": ("RyougiSh1k1/AFL", "2a5f7eaa2fe23af6e1bde3f7b4225f82462e05b9"),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def stamp_ok(folder: str, sha: str) -> bool:
    p = os.path.join(folder, ".ficl_commit")
    if not os.path.isfile(p):
        return False
    with open(p) as f:
        return f.read().strip() == sha


def fetch(name: str, dest: str) -> str:
    repo, sha = REPOS[name]
    folder = os.path.join(dest, name)
    if stamp_ok(folder, sha):
        print(f"ok       {name} already at {sha[:7]} in {folder}")
        return folder
    if os.path.exists(folder):
        raise SystemExit(f"{folder} exists without the stamp of commit {sha[:7]}; move it away first")
    url = f"https://github.com/{repo}/archive/{sha}.zip"
    print(f"fetch    {url}")
    data = urllib.request.urlopen(url, timeout=600).read()
    tmp = os.path.join(dest, f".{name}.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
    inner = os.path.join(tmp, f"{repo.split('/')[1]}-{sha}")
    if not os.path.isdir(inner):
        raise SystemExit(f"unexpected archive layout for {name}: {os.listdir(tmp)}")
    os.rename(inner, folder)
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(folder, ".ficl_commit"), "w") as f:
        f.write(sha + "\n")
    print(f"ok       {name} {sha[:7]} -> {folder}")
    return folder


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dest", default=os.path.join(HERE, "official_src"))
    p.add_argument("names", nargs="*", help=f"any of {list(REPOS)} (default: all)")
    a = p.parse_args(argv)
    unknown = sorted(set(a.names) - set(REPOS))
    if unknown:
        raise SystemExit(f"unknown repositories {unknown}; known: {list(REPOS)}")
    os.makedirs(a.dest, exist_ok=True)
    for n in a.names or list(REPOS):
        fetch(n, a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
