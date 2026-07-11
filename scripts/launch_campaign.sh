#!/usr/bin/env bash
# ============================================================================
# ONE-COMMAND launcher for the full A100 campaign. Paste-safe: no multi-line
# fragility, no manual mkdir/mv, no fragile `\` continuations.
#
# Usage (from the repo root, on the A100):
#     bash scripts/launch_campaign.sh                 # OICC + 15-seed training (days)
#     bash scripts/launch_campaign.sh --oicc-only     # just the contribution (~5 min)
#     bash scripts/launch_campaign.sh --skip-train    # OICC + figures, no GPU training
#     bash scripts/launch_campaign.sh --bg            # run in the background (nohup)
#
# It archives old outputs, auto-finds the India data, runs the campaign, and
# writes everything into results_campaign_<timestamp>/ + campaign.log.
# ============================================================================
set -euo pipefail

# --- locate repo root (works no matter where you call it from) --------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src"
export MPLBACKEND=Agg
export WANDB_MODE=disabled

# --- parse our own flags; pass the rest through to the python campaign -------
BG=0
PASSTHROUGH=()
for a in "$@"; do
  if [ "$a" = "--bg" ]; then BG=1; else PASSTHROUGH+=("$a"); fi
done

# --- archive old outputs (safe: no data loss; gitignored) -------------------
if [ -d outputs ] && [ -n "$(ls -A outputs 2>/dev/null || true)" ]; then
  mkdir -p archive
  STAMP="$(date +%Y%m%d_%H%M%S)"
  mv outputs "archive/outputs_OLD_${STAMP}"
  echo "[launcher] archived old outputs -> archive/outputs_OLD_${STAMP}"
fi
mkdir -p outputs

# --- auto-find India NCRB data (so you don't have to export it) -------------
if [ -z "${OICC_INDIA_DATA:-}" ]; then
  for cand in \
      "$ROOT/../crime-detection-ai/data" \
      "$ROOT/data/ncrb" \
      "/workspace/crime-detection-ai/data" ; do
    if [ -f "$cand/crime/01_District_wise_crimes_committed_IPC_2001_2012.csv" ]; then
      export OICC_INDIA_DATA="$cand"
      echo "[launcher] found India NCRB data -> $OICC_INDIA_DATA"
      break
    fi
  done
fi
[ -z "${OICC_INDIA_DATA:-}" ] && echo "[launcher] India NCRB not found (US + synthetic will still run). Set OICC_INDIA_DATA to include it."

# --- launch -----------------------------------------------------------------
CMD=(python scripts/run_full_campaign.py "${PASSTHROUGH[@]:-}")
echo "[launcher] running: ${CMD[*]}"
if [ "$BG" = "1" ]; then
  nohup "${CMD[@]}" > campaign.log 2>&1 &
  echo "[launcher] started in background (PID $!). Watch it with:"
  echo "           tail -f campaign.log"
else
  "${CMD[@]}" 2>&1 | tee campaign.log
fi
