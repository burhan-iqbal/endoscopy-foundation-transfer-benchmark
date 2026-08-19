#!/usr/bin/env bash
# Local Mac runner: CUDA > MPS > CPU (selected inside train.py).
# Resume-safe. Prefer starting with ResNet-50, then DINOv2, then Endo-FM.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

echo "=== Device probe ==="
python - <<'PY2'
from src.utils.device import get_device
import torch
d = get_device()
print("selected", d)
print("cuda", torch.cuda.is_available())
print("mps", torch.backends.mps.is_available())
PY2

MODELS=("resnet50" "dinov2_vits14" "endo_fm")
FRACS=(0.05 0.10 0.25 0.50 1.0)
SEED=0

for model in "${MODELS[@]}"; do
  if [[ "$model" == "endo_fm" && ! -f models/external_weights/endo_fm.pth ]]; then
    echo "Skipping endo_fm — weights missing"
    continue
  fi
  for frac in "${FRACS[@]}"; do
    echo "============================================================"
    echo "TRAIN model=$model label_frac=$frac seed=$SEED  $(date)"
    echo "============================================================"
    if ! python -m src.training.train --model "$model" --label-frac "$frac" --seed "$SEED" --resume; then
      echo "ERROR: train failed for $model $frac — continuing"
      continue
    fi
    if ! python -m src.training.evaluate --model "$model" --label-frac "$frac" --seed "$SEED" --bootstrap; then
      echo "ERROR: evaluate failed for $model $frac — continuing"
    fi
    if ! python -m src.training.calibrate --model "$model" --label-frac "$frac" --seed "$SEED"; then
      echo "ERROR: calibrate failed for $model $frac — continuing"
    fi
  done
done

python -m src.analysis.plots --aggregate || echo "plot aggregate failed"
echo "Local matrix finished at $(date)."
