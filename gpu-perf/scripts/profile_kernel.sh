#!/bin/bash
set -euo pipefail

# Quick profiling script for individual kernels
# Usage: ./scripts/profile_kernel.sh <kernel_name> [size]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <kernel_name> [size]"
    echo ""
    echo "Examples:"
    echo "  $0 vector_add 1048576"
    echo "  $0 matmul_tiled 1024"
    echo "  $0 conv2d_3x3 2048"
    echo ""
    echo "Available kernels:"
    echo "  1D: vector_add, saxpy, strided_copy_8, reduce_sum, dot_product,"
    echo "      histogram, random_access, vector_add_divergent"
    echo "  2D: naive_transpose, shared_transpose, conv2d_3x3, conv2d_7x7"
    echo "  Matrix: matmul_naive, matmul_tiled"
    echo "  Special: shared_bank_conflict, atomic_hotspot"
    exit 1
fi

KERNEL=$1
SIZE=${2:-1048576}  # Default size
PROFILE_DIR="profiles"
mkdir -p $PROFILE_DIR

echo "=========================================="
echo "Profiling: $KERNEL"
echo "=========================================="

# Determine kernel type and build appropriate command
case $KERNEL in
    vector_add|saxpy|strided_copy_8|reduce_sum|dot_product|histogram|random_access|vector_add_divergent)
        ARGS="--N $SIZE --block 256 --warmup 5 --reps 10"
        ;;
    naive_transpose|shared_transpose)
        ARGS="--rows $SIZE --cols $SIZE --warmup 5 --reps 10"
        ;;
    conv2d_3x3|conv2d_7x7)
        ARGS="--H $SIZE --W $SIZE --warmup 5 --reps 10"
        ;;
    matmul_naive|matmul_tiled)
        ARGS="--matN $SIZE --warmup 5 --reps 10"
        ;;
    shared_bank_conflict)
        ARGS="--warmup 5 --reps 10"
        ;;
    atomic_hotspot)
        ARGS="--N $SIZE --block 256 --iters 100 --warmup 5 --reps 10"
        ;;
    *)
        echo "Error: Unknown kernel '$KERNEL'"
        exit 1
        ;;
esac

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "Error: ncu (Nsight Compute) not found"
    echo "Try: module load cuda"
    exit 1
fi

# Profile with full metrics
echo ""
echo "Running: ncu --set full ./bin/runner --kernel $KERNEL $ARGS"
echo "Output: ${PROFILE_DIR}/${KERNEL}.ncu-rep"
echo ""

ncu --set full \
    --target-processes all \
    -o ${PROFILE_DIR}/${KERNEL} \
    ./bin/runner --kernel $KERNEL $ARGS

# Generate text summary
echo ""
echo "=== Generating summary ==="
ncu --import ${PROFILE_DIR}/${KERNEL}.ncu-rep \
    --print-summary stdout \
    > ${PROFILE_DIR}/${KERNEL}_summary.txt

# Show key metrics
echo ""
echo "=== Key Metrics ==="
ncu --import ${PROFILE_DIR}/${KERNEL}.ncu-rep \
    --print-summary stdout | \
    grep -E "SM Efficiency|Achieved Occupancy|Memory Throughput|Compute Throughput" | \
    head -10

echo ""
echo "=========================================="
echo "✓ Profiling complete!"
echo "=========================================="
echo ""
echo "Files generated:"
echo "  - ${PROFILE_DIR}/${KERNEL}.ncu-rep (detailed profile)"
echo "  - ${PROFILE_DIR}/${KERNEL}_summary.txt (text summary)"
echo ""
echo "View full report:"
echo "  ncu --import ${PROFILE_DIR}/${KERNEL}.ncu-rep"
echo ""
echo "Export to CSV:"
echo "  ncu --import ${PROFILE_DIR}/${KERNEL}.ncu-rep --csv --page raw > ${KERNEL}_metrics.csv"
