# Profiling Results Parser Guide

## Problem: "No key metrics found"

If you see this error when parsing ncu results:
```
Warning: Could not parse data/profiling_2080/ncu_vector_add_metrics.csv: 'NoneType' object has no attribute 'strip'
No key metrics found
```

This has been fixed! The parser now handles None values and empty fields properly.

## Which Parser to Use?

### For NCU (NVIDIA Nsight Compute) Results

**Option 1: Text Parser (RECOMMENDED)**
```bash
python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt
```

**Parses:** `ncu_{kernel}_details.txt` files
**Output:** More detailed metrics including occupancy, utilization, branch efficiency

**Option 2: CSV Parser (Basic)**
```bash
python3 scripts/parse_profiling_results.py data/profiling_2080
```

**Parses:** `ncu_{kernel}_metrics.csv` files
**Output:** Basic throughput and timing metrics

### For nvprof Results

```bash
python3 scripts/parse_profiling_results.py data/profiling
```

**Parses:** `nvprof_{kernel}_metrics.csv` files
**Output:** Occupancy, efficiency, throughput, cache metrics

## NCU File Types

When you run ncu profiling, you get 3 types of files per kernel:

### 1. `ncu_{kernel}_report.ncu-rep` (Binary)
- **What**: Complete profiling data in binary format
- **Use**: Open with `ncu-ui` GUI for interactive analysis
- **Command**: `ncu-ui ncu_vector_add_report.ncu-rep`

### 2. `ncu_{kernel}_details.txt` (Text)
- **What**: Human-readable detailed metrics
- **Use**: Parse with `parse_ncu_text.py` for comprehensive analysis
- **Metrics**: Occupancy, utilization, branch efficiency, memory cycles
- **Command**: `python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt`

### 3. `ncu_{kernel}_metrics.csv` (CSV)
- **What**: Selected metrics in CSV format
- **Use**: Parse with `parse_profiling_results.py` for quick overview
- **Metrics**: Throughput percentages, timing
- **Command**: `python3 scripts/parse_profiling_results.py data/profiling_2080`

## Recommended Workflow

### Quick Analysis
```bash
# Parse all ncu results quickly
cd /home/user/test1/gpu-perf
python3 scripts/parse_profiling_results.py data/profiling_2080

# View summary
cat data/profiling_2080/profiling_summary_ncu.csv
```

### Detailed Analysis
```bash
# Parse ncu text files for more metrics
python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt

# View comprehensive summary
cat data/profiling_2080/ncu_parsed_summary.csv
```

### Deep Dive (Single Kernel)
```bash
# Open in GUI
ncu-ui data/profiling_2080/ncu_matmul_tiled_report.ncu-rep

# Or view text report
less data/profiling_2080/ncu_matmul_tiled_details.txt
```

## Metrics Comparison

### parse_profiling_results.py (CSV Parser)
Extracts from ncu CSV:
- ✅ Duration (ns)
- ✅ SM throughput (%)
- ✅ DRAM throughput (%)
- ✅ L1/L2 throughput (%)
- ✅ Warp activity (%)
- ✅ FP operations counts
- ✅ Memory operations
- ✅ Instruction rate

### parse_ncu_text.py (Text Parser)
Extracts from ncu text:
- ✅ Achieved occupancy (%)
- ✅ Theoretical occupancy (%)
- ✅ Occupancy gap
- ✅ SM/DRAM utilization (%)
- ✅ Branch efficiency (%)
- ✅ Divergent branches
- ✅ Active/Elapsed cycles (DRAM, L1, L2, SM)
- ✅ Block limits (registers, shared mem, warps)
- ✅ All metrics from CSV parser above

**Recommendation**: Use `parse_ncu_text.py` for comprehensive analysis.

## Error Handling Improvements

The CSV parser now:
- ✅ Handles None values gracefully (no more `'NoneType' object has no attribute 'strip'`)
- ✅ Continues parsing even if one file fails
- ✅ Shows debug info (number of metrics found per kernel)
- ✅ Provides helpful suggestions for better parsers

## Example Output

### Using parse_profiling_results.py (CSV)
```
Found 16 profiling result files
  Parsing vector_add...
    Found 150 total metrics, 8 key metrics
  Parsing matmul_tiled...
    Found 148 total metrics, 10 key metrics

kernel,dram_throughput_pct,sm_throughput_pct,warp_activity_pct,...
vector_add,45.2,12.3,87.1,...
matmul_tiled,35.6,78.9,92.3,...
```

### Using parse_ncu_text.py (Text)
```
Found 16 ncu text file(s)...
  Parsing: data/profiling_2080/ncu_vector_add_details.txt
  Parsing: data/profiling_2080/ncu_matmul_tiled_details.txt

vector_add:
----------------------------------------
  Achieved Occupancy (%):          87.15
  Theoretical Occupancy (%):      100.00
  SM Utilization (%):               2.00
  DRAM Utilization (%):            14.56
  Branch Efficiency (%):            0.00

matmul_tiled:
----------------------------------------
  Achieved Occupancy (%):          92.34
  Theoretical Occupancy (%):      100.00
  SM Utilization (%):              78.45
  DRAM Utilization (%):            35.67
  Branch Efficiency (%):          100.00
```

## Troubleshooting

### "No key metrics found" for all kernels
**Fix**: Make sure ncu actually collected the metrics in the CSV
**Solution**: Use `parse_ncu_text.py` instead - it's more reliable

### "Warning: Could not parse ... 'NoneType' object"
**Fix**: This is now fixed! Update to latest parse_profiling_results.py
**Or**: Use `parse_ncu_text.py` which is more robust

### CSV summary is empty (219 bytes)
**Cause**: Parser couldn't extract any metrics from CSV files
**Solution**: Use `parse_ncu_text.py` on the *_details.txt files instead

### Files not found
**Check**: Make sure you're in the right directory
```bash
ls -lh data/profiling_2080/ncu_*
```

## Quick Reference

| File Type | Parser | Command |
|-----------|--------|---------|
| ncu_*_details.txt | parse_ncu_text.py | `python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt` |
| ncu_*_metrics.csv | parse_profiling_results.py | `python3 scripts/parse_profiling_results.py data/profiling_2080` |
| ncu_*_report.ncu-rep | ncu-ui (GUI) | `ncu-ui data/profiling_2080/ncu_vector_add_report.ncu-rep` |
| nvprof_*_metrics.csv | parse_profiling_results.py | `python3 scripts/parse_profiling_results.py data/profiling` |

**Bottom line**: For ncu results, use `parse_ncu_text.py` for best results!
