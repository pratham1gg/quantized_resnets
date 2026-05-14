#!/usr/bin/env bash
set -o pipefail

SEEDS=(42)
INPUT_BITS=(2 1)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_lowbit.py"
LOG_DIR="$REPO_ROOT/logs"
CKPT_ROOT="$REPO_ROOT/checkpoints"

mkdir -p "$LOG_DIR"

succeeded=()
failed=()

echo "========================================"
echo "  Multi-seed low-bit training"
echo "  Seeds: ${SEEDS[*]}"
echo "  Input bits: ${INPUT_BITS[*]}"
echo "  Started: $(date)"
echo "========================================"

for bits in "${INPUT_BITS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt_dir="$CKPT_ROOT/fp32_${bits}bit/seed_${seed}"
        best_path="$ckpt_dir/best.pth"
        log_file="$LOG_DIR/fp32_${bits}bit_seed_${seed}.log"

        echo ""
        echo "----------------------------------------"
        echo "  ${bits}-bit / seed $seed — $(date)"
        echo "  Checkpoints: $ckpt_dir"
        echo "  Log: $log_file"
        echo "----------------------------------------"

        python "$TRAIN_SCRIPT" \
            --input-bits "$bits" \
            --seed "$seed" \
            --checkpoint-dir "$ckpt_dir" \
            --best-path "$best_path" \
            2>&1 | tee "$log_file"

        exit_code=${PIPESTATUS[0]}
        run_label="${bits}bit/seed_${seed}"

        if [ $exit_code -eq 0 ]; then
            echo "[$run_label] SUCCESS"
            succeeded+=("$run_label")
        else
            echo "[$run_label] FAILED (exit code $exit_code)"
            failed+=("$run_label")
        fi
    done
done

echo ""
echo "========================================"
echo "  Summary — $(date)"
echo "========================================"
echo "  Succeeded: ${succeeded[*]:-none}"
echo "  Failed:    ${failed[*]:-none}"
echo "========================================"
