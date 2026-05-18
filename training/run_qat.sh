#!/usr/bin/env bash
set -o pipefail

INPUT_BITS=(8 4 2 1)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/qat_training.py"
LOG_DIR="$REPO_ROOT/logs"
CKPT_ROOT="$REPO_ROOT/checkpoints/qat"

mkdir -p "$LOG_DIR"

succeeded=()
failed=()

echo "========================================"
echo "  QAT INT8 training"
echo "  Input bits: ${INPUT_BITS[*]}"
echo "  Started: $(date)"
echo "========================================"

for bits in "${INPUT_BITS[@]}"; do
    run_name="int8_in${bits}b"
    ckpt_dir="$CKPT_ROOT/$run_name"
    log_file="$LOG_DIR/qat_${run_name}.log"

    echo ""
    echo "----------------------------------------"
    echo "  QAT INT8 / input ${bits}-bit — $(date)"
    echo "  FP32 base: $REPO_ROOT/checkpoints/fp32_${bits}bit/seed_42/best.pth"
    echo "  Checkpoints: $ckpt_dir"
    echo "  Log: $log_file"
    echo "----------------------------------------"

    python "$TRAIN_SCRIPT" \
        --input-quant-bits "$bits" \
        --checkpoint-dir "$CKPT_ROOT" \
        2>&1 | tee "$log_file"

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[$run_name] SUCCESS"
        succeeded+=("$run_name")
    else
        echo "[$run_name] FAILED (exit code $exit_code)"
        failed+=("$run_name")
    fi
done

echo ""
echo "========================================"
echo "  Summary — $(date)"
echo "========================================"
echo "  Succeeded: ${succeeded[*]:-none}"
echo "  Failed:    ${failed[*]:-none}"
echo "========================================"
