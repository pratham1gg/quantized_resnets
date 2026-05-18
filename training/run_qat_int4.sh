#!/bin/bash
set -e

SCRIPT="training/qat_training experiment.py"

for bits in 4 2 1; do
    echo "===== INT4 QAT — input ${bits}-bit ====="
    python "$SCRIPT" --precision int4 --input-quant-bits "$bits" --epochs 15
    echo ""
done

echo "All INT4 QAT runs complete."
