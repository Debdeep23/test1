#!/bin/bash
# Profile all kernels on RTX 4070 using ncu
# Run this on cuda5

SCRIPT_DIR=$(dirname "$0")

# List of all kernels
KERNELS=(
    "vector_add"
    "saxpy"
    "strided_copy_8"
    "reduce_sum"
    "dot_product"
    "histogram"
    "random_access"
    "naive_transpose"
    "shared_transpose"
    "matmul_naive"
    "matmul_tiled"
    "matmul_tiled_coarse"
    "shared_bank_conflict"
    "atomic_hotspot"
    "vector_add_divergent"
    "conv2d_3x3"
    "conv2d_7x7"
)

echo "============================================"
echo "Profiling all kernels on RTX 4070"
echo "Total kernels: ${#KERNELS[@]}"
echo "============================================"
echo ""

# Profile each kernel
SUCCESS_COUNT=0
FAIL_COUNT=0

for kernel in "${KERNELS[@]}"; do
    echo ""
    echo ">>> Profiling: $kernel"
    echo ""

    # Use appropriate args for each kernel type
    # Continue on errors - don't let one failure stop all profiling
    set +e
    case $kernel in
        atomic_hotspot)
            "$SCRIPT_DIR/profile_4070.sh" "$kernel" --N 1048576 --iters 100 --warmup 1 --reps 1
            RESULT=$?
            ;;
        shared_bank_conflict)
            "$SCRIPT_DIR/profile_4070.sh" "$kernel" --warmup 1 --reps 1
            RESULT=$?
            ;;
        matmul_* | *transpose | conv2d_*)
            # Reduce size for profiling to save time
            "$SCRIPT_DIR/profile_4070.sh" "$kernel" --rows 512 --cols 512 --warmup 1 --reps 1
            RESULT=$?
            ;;
        *)
            "$SCRIPT_DIR/profile_4070.sh" "$kernel" --N 1048576 --warmup 1 --reps 1
            RESULT=$?
            ;;
    esac
    set -e

    if [ $RESULT -eq 0 ]; then
        echo ">>> SUCCESS: $kernel"
        ((SUCCESS_COUNT++))
    else
        echo ">>> FAILED: $kernel" >&2
        ((FAIL_COUNT++))
    fi

    echo "---"
done

echo ""
echo "============================================"
echo "Profiling Complete!"
echo "============================================"
echo "Success: $SUCCESS_COUNT"
echo "Failed:  $FAIL_COUNT"
echo ""
echo "View reports:"
echo "  ls data/profiling_4070/"
echo ""
echo "Open in ncu-ui:"
echo "  ncu-ui data/profiling_4070/ncu_<kernel>.ncu-rep"
echo "============================================"
