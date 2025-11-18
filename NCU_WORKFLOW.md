# NCU Profiling Workflow - RTX 2080

## Your Current Status
✅ ncu is installed and working
✅ Successfully profiled vector_add

## Step-by-Step: Profile All Kernels

### Option 1: Automated (Recommended)

```bash
cd /home/user/test1/gpu-perf

# Profile all 16 kernels automatically (~10-15 minutes)
scripts/profile_all_2080.sh

# Parse all results
python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt
```

### Option 2: Manual (Profile Specific Kernels)

```bash
cd /home/user/test1/gpu-perf

# Profile individual kernels
scripts/profile_kernel_2080.sh saxpy
scripts/profile_kernel_2080.sh matmul_tiled
scripts/profile_kernel_2080.sh matmul_naive
scripts/profile_kernel_2080.sh shared_transpose
scripts/profile_kernel_2080.sh strided_copy_8
scripts/profile_kernel_2080.sh reduce_sum
scripts/profile_kernel_2080.sh dot_product
scripts/profile_kernel_2080.sh histogram
scripts/profile_kernel_2080.sh conv2d_3x3
scripts/profile_kernel_2080.sh conv2d_7x7
scripts/profile_kernel_2080.sh random_access
scripts/profile_kernel_2080.sh vector_add_divergent
scripts/profile_kernel_2080.sh shared_bank_conflict
scripts/profile_kernel_2080.sh atomic_hotspot
scripts/profile_kernel_2080.sh naive_transpose

# Parse all results at once
python3 scripts/parse_ncu_text.py data/profiling/ncu_*_details.txt
```

### Option 3: Direct ncu Commands

```bash
cd /home/user/test1/gpu-perf

# Basic profiling with ncu (creates .ncu-rep file)
ncu --set full --export data/profiling/ncu_saxpy_report \
    bin/runner --kernel saxpy --rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10

# Get text output for parsing
ncu --set detailed --page details \
    bin/runner --kernel saxpy --rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10 \
    > data/profiling/ncu_saxpy_details.txt 2>&1
```

## Parsing the Results

### Parse All ncu Text Files

```bash
# Parse all ncu text files in profiling directory
python3 scripts/parse_ncu_text.py data/profiling/ncu_*_details.txt

# Or specify output location
python3 scripts/parse_ncu_text.py data/profiling/ncu_*_details.txt data/ncu_summary.csv
```

### Parse Single File

```bash
python3 scripts/parse_ncu_text.py data/profiling/ncu_vector_add_details.txt
```

## Understanding Your vector_add Results

From your ncu output, here's what we see:

```
Kernel: vector_add
------------------
Achieved Occupancy:       87.15%    ✅ EXCELLENT (>70% is good)
Theoretical Occupancy:   100.00%
Occupancy Gap:            12.85%    ⚠️  Some room for improvement

Branch Efficiency:         0.00%    ⚠️  All branches diverge (expected for this kernel)
Avg Divergent Branches:       0    ✅ No divergence

DRAM Active vs Elapsed:
  Average DRAM Active:   243,552 cycles
  Total DRAM Elapsed:  1,672,192 cycles
  DRAM Utilization:        14.6%    ⚠️  Memory bound, low utilization
```

**Key Findings for vector_add:**
1. **High occupancy (87%)** - GPU well utilized
2. **Low DRAM utilization (14.6%)** - Memory bound, bandwidth limited
3. **No divergence** - Good control flow
4. **Optimization opportunity**: 12.85% speedup possible by improving occupancy

## What to Look For in Other Kernels

### Memory-Bound Kernels (vector_add, saxpy, strided_copy_8)
- **DRAM utilization**: Should be high (>70%)
- **SM utilization**: May be lower
- **Occupancy**: High is good (>70%)

### Compute-Bound Kernels (matmul_tiled, conv2d_*)
- **SM utilization**: Should be high (>70%)
- **DRAM utilization**: May be lower
- **Occupancy**: Critical (>50%)

### Problem Kernels to Watch
- **vector_add_divergent**: Expect low branch efficiency
- **shared_bank_conflict**: Expect low shared memory efficiency
- **atomic_hotspot**: Expect serialization, low throughput

## Key Metrics Extracted by Parser

The parser extracts these metrics from ncu text output:

### Occupancy
- `achieved_occupancy` - Actual GPU utilization (%)
- `theoretical_occupancy` - Max possible (%)
- `achieved_warps_per_sm` - Active warps per SM

### Utilization
- `sm_utilization_pct` - Compute unit usage (%)
- `dram_utilization_pct` - Memory bandwidth usage (%)

### Branch Behavior
- `branch_efficiency` - Non-divergent branches (%)
- `avg_divergent_branches` - Divergence count

### Memory Cycles
- `avg_dram_active_cycles` - Time DRAM is active
- `total_dram_elapsed_cycles` - Total time
- Same for L1, L2, SM

## Example Analysis Workflow

```bash
# 1. Profile all kernels
scripts/profile_all_2080.sh

# 2. Parse results
python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt

# 3. View summary
cat data/profiling_2080/ncu_parsed_summary.csv

# 4. Find lowest occupancy kernels
cat data/profiling_2080/ncu_parsed_summary.csv | sort -t',' -k3 -n | head -5

# 5. Find most divergent kernels
cat data/profiling_2080/ncu_parsed_summary.csv | sort -t',' -k7 -n | head -5

# 6. Deep dive on problem kernels
ncu-ui data/profiling_2080/ncu_{kernel}_report.ncu-rep
```

## Expected Results for Each Kernel

| Kernel | Expected Occupancy | Expected Bottleneck |
|--------|-------------------|---------------------|
| vector_add | >80% | Memory bandwidth |
| saxpy | >80% | Memory bandwidth |
| strided_copy_8 | >70% | Memory (strided access) |
| naive_transpose | >70% | Memory (uncoalesced) |
| shared_transpose | >70% | Compute (good efficiency) |
| reduce_sum | 50-70% | Shared memory + compute |
| dot_product | 50-70% | Shared memory + compute |
| histogram | <50% | Atomic operations |
| conv2d_3x3 | >70% | Compute + cache |
| conv2d_7x7 | >60% | Compute + registers |
| random_access | >70% | Cache misses |
| vector_add_divergent | >80% | Branch divergence |
| shared_bank_conflict | >70% | Shared memory conflicts |
| atomic_hotspot | <30% | Atomic serialization |
| matmul_naive | >70% | Cache misses |
| matmul_tiled | >80% | Compute (optimized) |

## Comparing Kernels

```bash
# Compare naive vs optimized transpose
grep transpose data/profiling_2080/ncu_parsed_summary.csv

# Compare naive vs tiled matmul
grep matmul data/profiling_2080/ncu_parsed_summary.csv

# See optimization impact
python3 -c "
import csv
with open('data/profiling_2080/ncu_parsed_summary.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if 'transpose' in row['kernel']:
            print(f\"{row['kernel']:20s} Occupancy: {row.get('achieved_occupancy', 'N/A'):>6s}%\")
"
```

## Next Steps

1. **Profile all kernels**: `scripts/profile_all_2080.sh`
2. **Parse results**: `python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt`
3. **Identify bottlenecks**: Look at the CSV summary
4. **Deep dive**: Use `ncu-ui` on interesting kernels
5. **Optimize**: Make targeted improvements
6. **Validate**: Re-profile to confirm gains

## Tips

- Profiling is slow (5-15 min for all kernels) - ncu runs kernels multiple times
- Save the .ncu-rep files - you can open them later with `ncu-ui`
- Text parsing is faster than GUI for batch analysis
- Compare before/after optimization using the CSV summaries
- Focus on kernels with <70% occupancy or high divergence first
