#!/usr/bin/env bash
# scripts/profile_all_kernels_2080.sh
# Profile all kernels on 2080 GPU
# Usage: scripts/profile_all_kernels_2080.sh [nvprof|ncu] [output_dir]
set -euo pipefail

TOOL="${1:-nvprof}"  # nvprof or ncu
OUTPUT_DIR="${2:-data/profiling_2080}"

mkdir -p "$OUTPUT_DIR"

echo "=== Profiling all kernels on 2080 using $TOOL ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Define kernel configurations (same as run_2080ti_full.sh)
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

for kernel_config in "${KERNELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Split kernel name and args
    KERNEL="${kernel_config%%:*}"
    ARGS="${kernel_config#*:}"

    echo ""
    echo "[$CURRENT/$TOTAL] Profiling: $KERNEL"
    echo "Arguments: $ARGS"
    echo "---"

    if [ "$TOOL" = "nvprof" ]; then
        scripts/profile_nvprof_2080.sh "$KERNEL" "$ARGS" "$OUTPUT_DIR" || {
            echo "WARNING: Failed to profile $KERNEL with nvprof"
            continue
        }
    elif [ "$TOOL" = "ncu" ]; then
        scripts/profile_ncu_2080.sh "$KERNEL" "$ARGS" "$OUTPUT_DIR" || {
            echo "WARNING: Failed to profile $KERNEL with ncu"
            continue
        }
    else
        echo "ERROR: Unknown tool '$TOOL'. Use 'nvprof' or 'ncu'"
        exit 1
    fi

    echo "✓ Completed $KERNEL"
done

echo ""
echo "=== All Profiling Complete ==="
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR" | tail -n +2
