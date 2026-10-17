#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "[1/7] Installing training dependencies..."
python -m pip install -q \
  transformers==4.57.3 \
  datasets \
  accelerate \
  evaluate \
  safetensors \
  scikit-learn==1.7.2 \
  scipy==1.15.3

echo "[2/7] Checking CUDA and A100..."
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Select an A100 GPU runtime and retry.")

name = torch.cuda.get_device_name(0)
memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {name} ({memory_gb:.1f} GB)")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

if "A100" not in name.upper():
    print("WARNING: This script is tuned for A100, but a different GPU was detected.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected GPU does not support the BF16 training configuration.")
PY

echo "[3/7] Backing up existing outputs, if present..."
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -d MODEL ]]; then
  mv MODEL "MODEL_backup_${RUN_STAMP}"
fi
if [[ -d NLIMODEL ]]; then
  mv NLIMODEL "NLIMODEL_backup_${RUN_STAMP}"
fi

echo "[4/7] Training MiniLM on MNLI..."
python train.py 2>&1 | tee train.log

echo "[5/7] Running the supplied test.py..."
WANDB_DISABLED=true python test.py 2>&1 | tee test.log

echo "[6/7] Validating the offline submission artifacts..."
python - <<'PY'
from transformers import AutoTokenizer
from model import NLI

model = NLI.from_pretrained("./MODEL", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("./MODEL", local_files_only=True)
parameters = sum(parameter.numel() for parameter in model.parameters())

if parameters > 40_000_000:
    raise SystemExit(f"Parameter limit exceeded: {parameters:,}")
if len(tokenizer) != model.config.vocab_size:
    raise SystemExit(
        f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} != "
        f"{model.config.vocab_size}"
    )

print(f"Parameters: {parameters:,}")
print(f"Tokenizer size: {len(tokenizer):,}")
print("Offline model/tokenizer load: OK")
PY

echo "[7/7] Creating the CodaBench archive..."
SUBMISSION="submission_${RUN_STAMP}.zip"
zip -qr "$SUBMISSION" model.py MODEL

echo
echo "Done. CodaBench submission: $SCRIPT_DIR/$SUBMISSION"
echo "Moodle submission: $SCRIPT_DIR/model.py"
echo "Training log: $SCRIPT_DIR/train.log"
echo "Test log: $SCRIPT_DIR/test.log"
