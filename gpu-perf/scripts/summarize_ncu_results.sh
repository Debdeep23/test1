#!/bin/bash
# Print a human-readable summary of all profiled kernels
# Usage: ./summarize_ncu_results.sh

PROFILE_DIR="data/profiling_4070"

echo "============================================"
echo "NCU Profiling Summary - All Kernels"
echo "============================================"
echo ""

cd "$PROFILE_DIR"

for ncu_file in ncu_*.ncu-rep; do
    if [ ! -f "$ncu_file" ]; then
        continue
    fi

    kernel=$(basename "$ncu_file" .ncu-rep | sed 's/ncu_//')

    echo "----------------------------------------"
    echo "Kernel: $kernel"
    echo "----------------------------------------"

    # Print summary
    ncu --import "$ncu_file" --print-summary per-kernel 2>/dev/null | grep -v "^==" | head -30

    echo ""
done

echo "============================================"
echo "To view detailed metrics for a specific kernel:"
echo "  ncu --import data/profiling_4070/ncu_<kernel>.ncu-rep --page details"
echo ""
echo "To export to CSV:"
echo "  scripts/export_ncu_metrics.sh"
echo "============================================"
