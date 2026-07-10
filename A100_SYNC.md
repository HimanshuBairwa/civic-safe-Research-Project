# A100 Sync Guide — updating your GPU Jupyter/Docker to the latest code

*Your A100 already has an OLDER version of this repo, plus the datasets. This is
the safe, no-surprise way to pull the new code (OICC + A100 hardening + figures)
without touching your data. TL;DR: your data is gitignored, so it is never
overwritten; only code updates.*

---

## The 30-second version

```bash
cd /path/to/civic-safe-Research-Project        # your existing clone on the A100
python scripts/a100_sync.py                     # checks + safely updates from main
```

That script does everything below with safety checks. If you prefer to do it by
hand, read on.

---

## What actually changes (so there are no surprises)

Since the version your A100 has (`0a0e47e`), `main` added **86 files** and
**modified only 5**, with **zero deletions**:

| Modified file | Change | Risk |
|---|---|---|
| `scripts/baselines.py` | `xgboost` import moved inside the function | none |
| `scripts/proxy_audit.py` | `sklearn` import moved inside the function | none |
| `scripts/visualize.py` | `seaborn` lazy + Agg backend | none |
| `src/civicsafe/theory/feedback_law.py` | floored a log to kill a divide-by-zero warning | none |
| `.gitignore` | added cache/artifact ignores | none |

Everything else is **new** (`src/oicc/`, `tests_oicc/`, `experiments/oicc_runs/`,
`paper/`, `docs/data_access/`, `Dockerfile`, `run_all.py`, roadmap). **Your
`data/` is gitignored and is NOT touched by the pull.**

---

## Manual steps (what the script automates)

```bash
cd /path/to/civic-safe-Research-Project

# 1. confirm your data is safe (these are gitignored -> pull can't overwrite them)
ls data/processed/*.pt              # your panels should be here, untouched after pull

# 2. stash any local edits you made on the A100 (usually none), then update
git fetch origin
git stash push -u -m "a100-local" || true      # only if you edited tracked files
git checkout main
git pull origin main                            # fast-forward; brings all new code

# 3. (only if you had local edits) re-apply them
git stash pop || true                           # resolve if it reports a conflict

# 4. refresh the environment (new optional deps; numpy<2.1 pin for the stack)
pip install -r requirements-a100.txt
pip install -e .                                # reinstall the package (new modules)

# 5. verify the whole thing runs on THIS box
python run_all.py                               # env + GPU smoke + oicc tests + reproduce
```

If `run_all.py` prints **ALL GREEN**, you are fully synced and everything works.

---

## Running on your REAL data (it's already on the A100)

Your datasets are already there from the old version. Point the code at them:

```bash
# US panels: already at data/processed/*.pt -> the loaders find them automatically.
# India NCRB: if it's in a sibling folder, tell the resolver where:
export OICC_INDIA_DATA=/path/to/crime-detection-ai/data

# then run the real experiments (they auto-skip cleanly if a dataset is missing):
python experiments/oicc_runs/run_ncrb_experiment.py       # India
python experiments/oicc_runs/run_us_experiment.py         # US contrast
python experiments/oicc_runs/make_pub_figures.py          # publication figures (uses real geo)
```

The original GNN training still works exactly as before:
```bash
python scripts/train.py                                    # unchanged entry point
```

---

## If something looks off

- **`git pull` reports a conflict** on one of the 5 modified files → you edited it
  locally on the A100. Keep your version or take `main`'s:
  `git checkout --theirs <file>` (take main) or `--ours` (keep yours), then
  `git add <file>`. The 5 files are code fixes; taking `main` is safe.
- **`pip install -e .` fails** → your torch/PyG is fine (civicsafe used them
  before); just ensure `numpy<2.1` (`pip install "numpy<2.1"`). `oicc` itself
  needs only numpy+scipy.
- **A real-data experiment says "data not found"** → set the env var above or copy
  the dataset; it never crashes, it skips.
- **Nuclear option (discard ALL local A100 code changes, keep data):**
  `git fetch origin && git reset --hard origin/main` — safe for `data/` (gitignored)
  but discards any code edits you made on the A100.

---

## Why this is safe

- Your data lives under `data/` which is in `.gitignore` — git will never
  overwrite or delete it on a pull or even a hard reset.
- Only 5 tracked code files changed, all low-risk fixes; 86 are pure additions.
- `oicc` is pure numpy/scipy — no new heavy dependency to fight with.
- `run_all.py` verifies the box end-to-end before you trust it.
