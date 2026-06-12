#!/usr/bin/env bash
set -o pipefail

SEEDS=(1 2)
INPUT_BITS=(8 4 2 1)
PRECISIONS=(int8 int4)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_qat.py"
LOG_DIR="$REPO_ROOT/logs"
CKPT_ROOT="$REPO_ROOT/.checkpoints/qat"

mkdir -p "$LOG_DIR"

succeeded=()
failed=()

echo "========================================"
echo "  QAT Multi-Seed Training"
echo "  Seeds: ${SEEDS[*]}"
echo "  Precisions: ${PRECISIONS[*]}"
echo "  Input bits: ${INPUT_BITS[*]}"
echo "  Started: $(date)"
echo "========================================"

for seed in "${SEEDS[@]}"; do
    for prec in "${PRECISIONS[@]}"; do
        for bits in "${INPUT_BITS[@]}"; do
            run_name="${prec}_in${bits}b_seed${seed}"
            log_file="$LOG_DIR/qat_${run_name}.log"
            fp32_ckpt="$REPO_ROOT/.checkpoints/fp32_${bits}bit/seed_${seed}/best.pth"

            echo ""
            echo "----------------------------------------"
            echo "  QAT ${prec^^} / input ${bits}-bit / seed ${seed} — $(date)"
            echo "  FP32 base: $fp32_ckpt"
            echo "  Log: $log_file"
            echo "----------------------------------------"

            if [ ! -f "$fp32_ckpt" ]; then
                echo "[$run_name] SKIPPED — FP32 checkpoint not found"
                failed+=("$run_name (no fp32)")
                continue
            fi

            python "$TRAIN_SCRIPT" \
                --precision "$prec" \
                --input-quant-bits "$bits" \
                --seed "$seed" \
                --checkpoint "$fp32_ckpt" \
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
    done
done

echo ""
echo "========================================"
echo "  Summary — $(date)"
echo "========================================"
echo "  Succeeded: ${succeeded[*]:-none}"
echo "  Failed:    ${failed[*]:-none}"
echo "========================================"
