#!/usr/bin/env bash
# scripts/profile_all_2080.sh
# Profile all kernels on 2080 GPU using auto-detected profiler
# Usage: scripts/profile_all_2080.sh [output_dir]
set -euo pipefail

OUTPUT_DIR="${1:-data/profiling_2080}"

mkdir -p "$OUTPUT_DIR"

echo "==================================================================="
echo "GPU Kernel Profiling - RTX 2080"
echo "==================================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if we have any profiler before starting
if ! command -v nvprof &> /dev/null && ! command -v ncu &> /dev/null && ! command -v nsys &> /dev/null; then
    echo "ERROR: No NVIDIA profiling tools found!"
    echo ""
    echo "Please install NVIDIA Nsight Compute (ncu) or see PROFILING_SETUP.md"
    echo ""
    echo "Quick install (Ubuntu/Debian):"
    echo "  sudo apt install nvidia-nsight-compute"
    echo ""
    exit 1
fi

# Define kernel configurations (same as run_2080ti_full.sh but with lighter profiling args)
declare -a KERNELS=(
    "vector_add:--rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10"
    "saxpy:--rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10"
    "strided_copy_8:--rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10"
    "naive_transpose:--rows 2048 --cols 2048 --warmup 5 --reps 10"
    "shared_transpose:--rows 2048 --cols 2048 --warmup 5 --reps 10"
    "reduce_sum:--rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10"
    "dot_product:--rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10"
    "histogram:--rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10"
    "conv2d_3x3:--rows 1024 --cols 1024 --warmup 5 --reps 10"
    "conv2d_7x7:--rows 1024 --cols 1024 --warmup 5 --reps 10"
    "random_access:--rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10"
    "vector_add_divergent:--rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10"
    "shared_bank_conflict:--warmup 5 --reps 10"
    "atomic_hotspot:--rows 1048576 --cols 1 --block 256 --iters 100 --warmup 5 --reps 10"
    "matmul_naive:--rows 512 --cols 512 --warmup 5 --reps 10"
    "matmul_tiled:--rows 512 --cols 512 --warmup 5 --reps 10"
)

TOTAL=${#KERNELS[@]}
CURRENT=0
FAILED=0
SUCCEEDED=0

START_TIME=$(date +%s)

for kernel_config in "${KERNELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Split kernel name and args
    KERNEL="${kernel_config%%:*}"
    ARGS="${kernel_config#*:}"

    echo ""
    echo "==================================================================="
    echo "[$CURRENT/$TOTAL] Profiling: $KERNEL"
    echo "==================================================================="
    echo "Arguments: $ARGS"
    echo ""

    # Use the auto-detect profiling script
    if scripts/profile_kernel_2080.sh "$KERNEL" "$ARGS" "$OUTPUT_DIR"; then
        echo ""
        echo "✓ SUCCESS: $KERNEL profiled"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo ""
        echo "✗ FAILED: $KERNEL profiling failed"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "==================================================================="
echo "Profiling Complete"
echo "==================================================================="
echo "Total kernels:    $TOTAL"
echo "Succeeded:        $SUCCEEDED"
echo "Failed:           $FAILED"
echo "Duration:         ${DURATION}s"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Try to generate summary
if [ -f scripts/parse_profiling_results.py ]; then
    echo "Generating summary..."
    if python3 scripts/parse_profiling_results.py "$OUTPUT_DIR"; then
        echo ""
        echo "Summary generated successfully"

        # Show summary files
        echo ""
        echo "Summary files:"
        ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null | tail -n 5 || true
    else
        echo "Note: Could not generate summary (this is optional)"
    fi
fi

echo ""
echo "View results:"
echo "  ls -lh $OUTPUT_DIR"
echo ""
echo "Parse results:"
echo "  python3 scripts/parse_profiling_results.py $OUTPUT_DIR"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Warning: $FAILED kernel(s) failed to profile"
    exit 1
fi
