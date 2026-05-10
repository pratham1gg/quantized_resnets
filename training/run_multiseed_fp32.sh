#!/usr/bin/env bash
set -o pipefail

SEEDS=(42 1 2)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_fp32.py"
LOG_DIR="$REPO_ROOT/logs"
CKPT_ROOT="$REPO_ROOT/checkpoints/fp32"

mkdir -p "$LOG_DIR"

succeeded=()
failed=()

echo "========================================"
echo "  Multi-seed FP32 training"
echo "  Seeds: ${SEEDS[*]}"
echo "  Started: $(date)"
echo "========================================"

for seed in "${SEEDS[@]}"; do
    ckpt_dir="$CKPT_ROOT/seed_${seed}"
    best_path="$ckpt_dir/best.pth"
    log_file="$LOG_DIR/fp32_seed_${seed}.log"

    echo ""
    echo "----------------------------------------"
    echo "  Seed $seed — $(date)"
    echo "  Checkpoints: $ckpt_dir"
    echo "  Log: $log_file"
    echo "----------------------------------------"

    python "$TRAIN_SCRIPT" \
        --seed "$seed" \
        --checkpoint-dir "$ckpt_dir" \
        --best-path "$best_path" \
        2>&1 | tee "$log_file"

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[seed $seed] SUCCESS"
        succeeded+=("$seed")
    else
        echo "[seed $seed] FAILED (exit code $exit_code)"
        failed+=("$seed")
    fi
done

echo ""
echo "========================================"
echo "  Summary — $(date)"
echo "========================================"
echo "  Succeeded: ${succeeded[*]:-none}"
echo "  Failed:    ${failed[*]:-none}"
echo "========================================"
