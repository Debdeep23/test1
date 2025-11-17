#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Process Existing Trials → Final Dataset"
echo "=========================================="
echo ""
echo "This script processes existing trial CSVs WITHOUT regenerating them."
echo "Use this if you already ran gen_trials_2080ti.sh"
echo ""

# Check if trial files exist
TRIAL_COUNT=$(ls -1 data/trials_*__2080ti.csv 2>/dev/null | wc -l)

if [ "$TRIAL_COUNT" -eq 0 ]; then
    echo "ERROR: No trial files found in data/"
    echo ""
    echo "Please run gen_trials_2080ti.sh first to generate trials:"
    echo "  ./scripts/gen_trials_2080ti.sh"
    echo ""
    exit 1
fi

echo "Found $TRIAL_COUNT trial CSV files"
echo ""

# Check for required calibration files
echo "Checking calibration files..."
REQUIRED_FILES=(
    "data/props_2080ti.out"
    "data/stream_like_2080ti.out"
    "data/gemm_cublas_2080ti.out"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Missing required file: $file"
        echo ""
        echo "Required calibration files:"
        echo "  - data/props_2080ti.out (GPU properties)"
        echo "  - data/stream_like_2080ti.out (memory bandwidth calibration)"
        echo "  - data/gemm_cublas_2080ti.out (compute calibration)"
        echo ""
        exit 1
    fi
    echo "  ✓ $file"
done

echo ""
echo "=== Processing trials → final dataset ==="
echo "This will:"
echo "  - Aggregate $TRIAL_COUNT trial files"
echo "  - Calculate all kernel metrics (11/11)"
echo "  - Add all GPU metrics (13/13)"
echo "  - Generate runs_2080ti_final.csv"
echo ""

python3 scripts/build_final_dataset.py \
  "data/trials_*__2080ti.csv" \
  data/props_2080ti.out \
  data/stream_like_2080ti.out \
  data/gemm_cublas_2080ti.out \
  data/runs_2080ti_final.csv

echo ""
echo "=== Verification ==="
TOTAL_ROWS=$(wc -l < data/runs_2080ti_final.csv)
TOTAL_COLS=$(head -1 data/runs_2080ti_final.csv | tr ',' '\n' | wc -l)

echo "Final dataset: data/runs_2080ti_final.csv"
echo "  Rows: $TOTAL_ROWS (including header)"
echo "  Data rows: $((TOTAL_ROWS - 1))"
echo "  Columns: $TOTAL_COLS"
echo ""

echo "=== Preview (first 3 rows) ==="
head -3 data/runs_2080ti_final.csv

echo ""
echo "=========================================="
echo "✓ Processing complete!"
echo "=========================================="
echo ""
echo "Dataset includes:"
echo "  ✓ 11/11 kernel metrics (100%)"
echo "  ✓ 13/13 GPU metrics (100%)"
echo "  ✓ $((TOTAL_ROWS - 1)) configurations"
echo "  ✓ $TOTAL_COLS organized columns"
echo ""
echo "Output: data/runs_2080ti_final.csv"
