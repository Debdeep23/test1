#!/usr/bin/env bash
# scripts/profile_nvprof_2080.sh
# Profile kernels on 2080 using nvprof
# Usage: scripts/profile_nvprof_2080.sh <kernel> "<argstr>" [output_dir]
set -euo pipefail

KERNEL="${1:-vector_add}"
ARGSTR="${2:---rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10}"
OUTPUT_DIR="${3:-data/profiling}"

mkdir -p "$OUTPUT_DIR"

echo "=== Profiling $KERNEL on 2080 with nvprof ==="

# Output files
SUMMARY_FILE="${OUTPUT_DIR}/nvprof_${KERNEL}_summary.txt"
METRICS_FILE="${OUTPUT_DIR}/nvprof_${KERNEL}_metrics.csv"
EVENTS_FILE="${OUTPUT_DIR}/nvprof_${KERNEL}_events.csv"

# Basic profiling with summary
echo "Running basic summary profiling..."
nvprof --print-gpu-trace \
       --print-summary \
       bin/runner --kernel "$KERNEL" $ARGSTR \
       > "$SUMMARY_FILE" 2>&1

echo "Summary saved to: $SUMMARY_FILE"

# Comprehensive metrics collection
echo "Collecting comprehensive metrics..."
nvprof --csv \
       --metrics \
flop_count_sp,\
flop_count_sp_add,\
flop_count_sp_mul,\
flop_count_sp_fma,\
flop_sp_efficiency,\
dram_read_throughput,\
dram_write_throughput,\
dram_utilization,\
gld_throughput,\
gst_throughput,\
gld_efficiency,\
gst_efficiency,\
l2_read_throughput,\
l2_write_throughput,\
shared_load_throughput,\
shared_store_throughput,\
achieved_occupancy,\
sm_efficiency,\
warp_execution_efficiency,\
branch_efficiency,\
l1_cache_global_hit_rate,\
l2_l1_read_hit_rate,\
shared_efficiency,\
inst_per_warp,\
inst_executed,\
inst_issued \
       bin/runner --kernel "$KERNEL" $ARGSTR \
       > "$METRICS_FILE" 2>&1

echo "Metrics saved to: $METRICS_FILE"

# Hardware events
echo "Collecting hardware events..."
nvprof --csv \
       --events \
active_warps,\
active_cycles,\
elapsed_cycles_sm,\
sm_cta_launched,\
shared_load,\
shared_store,\
global_load,\
global_store,\
local_load,\
local_store,\
branch,\
divergent_branch \
       bin/runner --kernel "$KERNEL" $ARGSTR \
       > "$EVENTS_FILE" 2>&1

echo "Events saved to: $EVENTS_FILE"

echo ""
echo "=== Profiling Complete ==="
echo "Files generated:"
echo "  - $SUMMARY_FILE"
echo "  - $METRICS_FILE"
echo "  - $EVENTS_FILE"
echo ""
echo "Quick view of summary:"
tail -n 30 "$SUMMARY_FILE"
