# GPU Kernel Profiling Workflow

Complete workflow for profiling all kernels on RTX 4070 (cuda5) using NVIDIA Nsight Compute.

## Quick Start

```bash
# On cuda5
cd /home/dn2491/gpu-perf

# 1. Profile all kernels
scripts/profile_all_4070.sh

# 2. Export metrics to CSV and combine
scripts/export_ncu_metrics.sh

# 3. View the combined results
cat data/profiling_4070/all_kernels_summary.csv
```

## Output Files

After running the workflow, you'll have:

### Binary Reports (for ncu-ui)
- `data/profiling_4070/ncu_<kernel>.ncu-rep` - Binary reports for each kernel
- `data/profiling_4070/ncu_<kernel>_details.txt` - Text logs

### Individual CSV Files
- `data/profiling_4070/csv_exports/<kernel>_summary.csv` - Summary metrics per kernel
- `data/profiling_4070/csv_exports/<kernel>_details.csv` - Detailed metrics per kernel

### Combined CSV Files (BEST FOR ANALYSIS)
- `data/profiling_4070/all_kernels_summary.csv` - **All kernels with summary metrics**
- `data/profiling_4070/all_kernels_details.csv` - **All kernels with detailed metrics**

## Analysis

### View in Terminal

```bash
# Quick view (requires column command)
column -t -s, < data/profiling_4070/all_kernels_summary.csv | less -S

# Or just cat
cat data/profiling_4070/all_kernels_summary.csv
```

### Analyze with Python

```python
import pandas as pd

# Load the combined CSV
df = pd.read_csv('data/profiling_4070/all_kernels_summary.csv')

# Set kernel as index
df = df.set_index('kernel')

# View all metrics for one kernel
print(df.loc['vector_add'])

# Compare metrics across kernels
print(df[['Duration', 'Occupancy', 'SM Throughput']])

# Find kernels with low occupancy
low_occ = df[df['Occupancy'] < 50]
print(low_occ)
```

### Open in Excel/Google Sheets

Just open `all_kernels_summary.csv` in Excel or upload to Google Sheets.

## Metrics Included

The profiling captures comprehensive metrics including:

**Execution**
- Kernel duration
- Grid/block configuration
- Registers per thread
- Shared memory usage

**Occupancy & Utilization**
- Theoretical occupancy
- Achieved occupancy
- SM utilization
- Memory bandwidth utilization

**Compute**
- Instructions executed
- Warp execution efficiency
- Branch efficiency

**Memory**
- DRAM throughput
- L1/L2 cache hit rates
- Global memory load/store efficiency

## Re-running Profiling

To profile again (e.g., after code changes):

```bash
# Clean old results
rm -rf data/profiling_4070/*

# Profile again
scripts/profile_all_4070.sh

# Export again
scripts/export_ncu_metrics.sh
```

## Manual Operations

### Profile a Single Kernel

```bash
scripts/profile_4070.sh vector_add --N 1048576
```

### Export Single Kernel to CSV

```bash
ncu --import data/profiling_4070/ncu_vector_add.ncu-rep --csv > vector_add.csv
```

### View Metrics in Terminal

```bash
# Summary
ncu --import data/profiling_4070/ncu_vector_add.ncu-rep --print-summary per-kernel

# Detailed
ncu --import data/profiling_4070/ncu_vector_add.ncu-rep --page details
```

### Open in GUI

```bash
ncu-ui data/profiling_4070/ncu_vector_add.ncu-rep
```

## Summary of Scripts

| Script | Purpose |
|--------|---------|
| `profile_4070.sh` | Profile a single kernel |
| `profile_all_4070.sh` | Profile all 17 kernels |
| `export_ncu_metrics.sh` | Export all .ncu-rep files to CSV |
| `combine_ncu_results.py` | Combine all CSVs into one file |
| `summarize_ncu_results.sh` | Print summary to terminal |

## Tips

1. **All profiling must be on cuda5** - ncu doesn't work on cuda2
2. **Use the combined CSV** - Much easier to analyze than individual files
3. **Profiling is slow** - Each kernel takes 30-60 seconds with `--set full`
4. **Binary reports are large** - .ncu-rep files can be 10-50 MB each
5. **CSV has all the data** - You don't need to keep .ncu-rep files after export

## Complete Example Session

```bash
# SSH to cuda5
ssh cuda5
cd /home/dn2491/gpu-perf

# Pull latest code
git pull origin claude/profile-kernels-nvpro-019i8c3K8ZYbpYF91j6WaY9B

# Build if needed
make

# Profile everything (takes ~10-15 minutes)
scripts/profile_all_4070.sh

# Export to CSV and combine
scripts/export_ncu_metrics.sh

# Download the combined CSV to your local machine
# (from your local terminal)
scp cuda5:/home/dn2491/gpu-perf/data/profiling_4070/all_kernels_summary.csv .

# Analyze locally with Excel, pandas, etc.
```

That's it! You now have comprehensive profiling data for all 17 kernels in one easy-to-analyze CSV file.
