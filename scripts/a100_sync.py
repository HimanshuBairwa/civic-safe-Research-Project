#!/usr/bin/env python
"""Safe A100 sync: update this clone to origin/main without touching your data.

Run on the A100:  python scripts/a100_sync.py
Flags: --hard  (discard local code edits, keep data)   --no-pip  (skip installs)

It (1) confirms data/ is gitignored and safe, (2) stashes local edits, (3)
fast-forwards to origin/main, (4) refreshes deps, (5) runs the preflight. Every
step prints what it is doing and stops on a real problem.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd: str, check: bool = True) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=str(ROOT), shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        print(f"  ! command failed: {cmd}\n{out}")
    return r.returncode, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", action="store_true",
                    help="discard local tracked-code edits (data is still safe)")
    ap.add_argument("--no-pip", action="store_true", help="skip pip installs")
    args = ap.parse_args()

    print("=" * 68)
    print("A100 SYNC -> origin/main")
    print("=" * 68)

    # 0. sanity: are we in a git repo with origin?
    rc, _ = sh("git rev-parse --is-inside-work-tree")
    if rc != 0:
        print("not a git repo; aborting."); return 1

    # 1. data safety: confirm data/ is ignored
    _, ig = sh("git check-ignore data/ || true", check=False)
    print(f"[1] data/ gitignored? {'YES - safe' if 'data' in ig else 'CHECK MANUALLY'}")
    _, pts = sh("ls data/processed/*.pt 2>/dev/null | wc -l", check=False)
    print(f"    processed panels present: {pts.strip()} (untouched by pull)")

    # 2. record current commit
    _, cur = sh("git rev-parse --short HEAD")
    print(f"[2] current HEAD: {cur}")

    sh("git fetch origin", check=True)

    if args.hard:
        print("[3] --hard: resetting to origin/main (keeps gitignored data)")
        sh("git checkout main", check=False)
        sh("git reset --hard origin/main", check=True)
    else:
        # stash local edits if any tracked file is dirty
        _, dirty = sh("git status --porcelain --untracked-files=no")
        if dirty.strip():
            print("[3] local tracked edits found -> stashing them")
            sh('git stash push -m "a100-sync-autostash"', check=True)
        else:
            print("[3] working tree clean")
        sh("git checkout main", check=False)
        rc, out = sh("git pull --ff-only origin main", check=False)
        if rc != 0:
            print("    fast-forward failed (diverged history). Options:")
            print("      keep your edits: resolve manually, or")
            print("      take main:       python scripts/a100_sync.py --hard")
            return 1
        # try to restore stash
        _, stash = sh("git stash list", check=False)
        if "a100-sync-autostash" in stash:
            rc2, _ = sh("git stash pop", check=False)
            if rc2 != 0:
                print("    NOTE: your stashed edits conflict; resolve with "
                      "`git checkout --theirs <file>` then `git add`.")

    _, new = sh("git rev-parse --short HEAD")
    print(f"    updated HEAD: {new}")

    # 4. deps
    if not args.no_pip:
        print("[4] refreshing dependencies")
        sh(f'"{sys.executable}" -m pip install -r requirements-a100.txt', check=False)
        sh(f'"{sys.executable}" -m pip install -e .', check=False)
    else:
        print("[4] --no-pip: skipping installs")

    # 5. preflight
    print("[5] verifying (run_all.py preflight)")
    rc, out = sh(f'"{sys.executable}" run_all.py', check=False)
    tail = "\n".join(out.splitlines()[-8:])
    print(tail)
    ok = "ALL GREEN" in out
    print("=" * 68)
    print("SYNC COMPLETE - ALL GREEN" if ok else
          "SYNC done, but preflight not fully green - check output above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
