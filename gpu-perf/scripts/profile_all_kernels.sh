#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "Batch Profiling All Kernels"
echo "=========================================="
echo ""
echo "This will profile all 16 kernels with Nsight Compute"
echo "Estimated time: 20-30 minutes"
echo ""

PROFILE_DIR="profiles"
mkdir -p $PROFILE_DIR

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "Error: ncu (Nsight Compute) not found"
    echo "Try: module load cuda"
    exit 1
fi

# Counter for progress
TOTAL=16
CURRENT=0

profile_kernel() {
    local kernel=$1
    local args=$2
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "[$CURRENT/$TOTAL] Profiling: $kernel"
    echo "Args: $args"

    ncu --set full \
        --target-processes all \
        -o ${PROFILE_DIR}/${kernel} \
        ./bin/runner --kernel $kernel $args \
        > ${PROFILE_DIR}/${kernel}.log 2>&1

    echo "  ✓ Saved to ${PROFILE_DIR}/${kernel}.ncu-rep"
}

echo "=== Profiling 1D Kernels ==="
profile_kernel "vector_add" "--N 1048576 --block 256 --warmup 5 --reps 10"
profile_kernel "saxpy" "--N 1048576 --block 256 --warmup 5 --reps 10"
profile_kernel "strided_copy_8" "--N 1048576 --block 256 --warmup 5 --reps 10"
profile_kernel "reduce_sum" "--N 1048576 --block 256 --warmup 5 --reps 10"
profile_kernel "dot_product" "--N 1048576 --block 256 --warmup 5 --reps 10"
profile_kernel "histogram" "--N 1048576 --block 256 --warmup 5 --reps 10"
profile_kernel "random_access" "--N 1048576 --block 256 --warmup 5 --reps 10"
profile_kernel "vector_add_divergent" "--N 1048576 --block 256 --warmup 5 --reps 10"

echo ""
echo "=== Profiling 2D Kernels ==="
profile_kernel "naive_transpose" "--rows 2048 --cols 2048 --warmup 5 --reps 10"
profile_kernel "shared_transpose" "--rows 2048 --cols 2048 --warmup 5 --reps 10"
profile_kernel "conv2d_3x3" "--H 1024 --W 1024 --warmup 5 --reps 10"
profile_kernel "conv2d_7x7" "--H 1024 --W 1024 --warmup 5 --reps 10"

echo ""
echo "=== Profiling Matrix Multiply ==="
profile_kernel "matmul_naive" "--matN 512 --warmup 5 --reps 10"
profile_kernel "matmul_tiled" "--matN 512 --warmup 5 --reps 10"

echo ""
echo "=== Profiling Special Kernels ==="
profile_kernel "shared_bank_conflict" "--warmup 5 --reps 10"
profile_kernel "atomic_hotspot" "--N 1048576 --block 256 --iters 100 --warmup 5 --reps 10"

echo ""
echo "=========================================="
echo "✓ All kernels profiled!"
echo "=========================================="
echo ""
echo "Profiles saved to: $PROFILE_DIR/"
echo ""
ls -lh ${PROFILE_DIR}/*.ncu-rep | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Next steps:"
echo "  1. View individual profile:"
echo "     ncu --import ${PROFILE_DIR}/vector_add.ncu-rep"
echo ""
echo "  2. Generate summary report:"
echo "     for f in ${PROFILE_DIR}/*.ncu-rep; do"
echo "       echo \$(basename \$f .ncu-rep)"
echo "       ncu --import \$f --print-summary stdout | grep -E 'SM Efficiency|Achieved Occupancy'"
echo "     done"
echo ""
echo "  3. Export all to CSV:"
echo "     for f in ${PROFILE_DIR}/*.ncu-rep; do"
echo "       ncu --import \$f --csv --page raw > \${f%.ncu-rep}.csv"
echo "     done"
