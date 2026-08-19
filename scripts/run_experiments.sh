#!/usr/bin/env bash
# Full CE-NBI experiment matrix for a rented H100/H200 session.
# Run from dissertation_project/ with the venv activated.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS=("resnet50" "dinov2_vits14" "endo_fm")
FRACS=(0.05 0.10 0.25 0.50 1.0)
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
# Override via e.g. SEEDS_OVERRIDE="0" for a quick single-seed pass.

if [[ ! -f data/processed/ce_nbi_metadata.csv ]]; then
  echo "Missing metadata. Run data prep first (see README)."
  exit 1
fi

ENDO_FM_WEIGHTS="models/external_weights/endo_fm.pth"
ENDO_FM_MIN_BYTES=2000000000
endo_fm_size=0
if [[ -f "$ENDO_FM_WEIGHTS" ]]; then
  endo_fm_size=$(stat -f%z "$ENDO_FM_WEIGHTS" 2>/dev/null || stat -c%s "$ENDO_FM_WEIGHTS")
fi
if [[ ! -f "$ENDO_FM_WEIGHTS" || "$endo_fm_size" -lt "$ENDO_FM_MIN_BYTES" ]]; then
  if [[ "${ALLOW_MISSING_ENDO_FM:-0}" == "1" ]]; then
    echo "WARNING: Endo-FM weights missing or truncated at $ENDO_FM_WEIGHTS (size=$endo_fm_size)"
    echo "         ALLOW_MISSING_ENDO_FM=1 set — endo_fm runs will be SKIPPED (a third of the matrix)."
    MODELS=("resnet50" "dinov2_vits14")
  else
    echo "ERROR: Endo-FM weights missing or truncated at $ENDO_FM_WEIGHTS (size=$endo_fm_size, expected >= $ENDO_FM_MIN_BYTES bytes)."
    echo "       See scripts/download_endo_fm_weights.md, or set ALLOW_MISSING_ENDO_FM=1 to run without endo_fm."
    exit 1
  fi
fi

mkdir -p reports/logs
ENV_CAPTURE="reports/logs/env_capture_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "=== python --version ==="
  python --version 2>&1
  echo "=== pip freeze ==="
  python -m pip freeze
  echo "=== nvidia-smi ==="
  nvidia-smi || true
  echo "=== torch ==="
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
} > "$ENV_CAPTURE" 2>&1
echo "Environment captured to $ENV_CAPTURE"

echo "Models: ${MODELS[*]}"
echo "Label fractions: ${FRACS[*]}"
echo "Seeds: ${SEEDS[*]}"

FAILURES=()
RUNS_ATTEMPTED=0

# Resume-safe: train.py skips finished runs (metrics.json present) and
# resumes mid-run from checkpoint_last.pt. Re-running this script after a
# disconnect continues where it left off.
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for frac in "${FRACS[@]}"; do
      echo "============================================================"
      echo "TRAIN model=$model label_frac=$frac seed=$seed  $(date)"
      echo "============================================================"
      RUNS_ATTEMPTED=$((RUNS_ATTEMPTED + 1))
      if ! python -m src.training.train --model "$model" --label-frac "$frac" --seed "$seed" --resume; then
        echo "ERROR: train failed for model=$model frac=$frac seed=$seed — skipping rest of this run"
        FAILURES+=("model=$model frac=$frac seed=$seed stage=train")
        continue
      fi
      if ! python -m src.training.evaluate --model "$model" --label-frac "$frac" --seed "$seed" --bootstrap; then
        echo "ERROR: evaluate failed for model=$model frac=$frac seed=$seed — skipping rest of this run"
        FAILURES+=("model=$model frac=$frac seed=$seed stage=evaluate")
        continue
      fi
      if ! python -m src.training.calibrate --model "$model" --label-frac "$frac" --seed "$seed"; then
        echo "ERROR: calibrate failed for model=$model frac=$frac seed=$seed — continuing"
        FAILURES+=("model=$model frac=$frac seed=$seed stage=calibrate")
      fi
    done
  done
done

if ! python -m src.analysis.plots --aggregate; then
  echo "ERROR: aggregation failed"
  FAILURES+=("model=- frac=- seed=- stage=aggregate")
fi

echo "============================================================"
echo "SUMMARY: $RUNS_ATTEMPTED runs attempted"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "All succeeded. Aggregated tables/figures under reports/tables and reports/figures."
  exit 0
else
  echo "${#FAILURES[@]} failure(s):"
  for f in "${FAILURES[@]}"; do
    echo "  FAILED: $f"
  done
  exit 1
fi
