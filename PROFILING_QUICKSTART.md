# Kernel Profiling Quick Start - RTX 2080

## TL;DR - Get Started in 30 Seconds

```bash
cd /home/user/test1/gpu-perf

# Profile one kernel with nvprof (fast, basic metrics)
scripts/profile_nvprof_2080.sh vector_add

# Profile one kernel with ncu (detailed, slower)
scripts/profile_ncu_2080.sh matmul_tiled

# Profile ALL kernels with nvprof (~5-10 minutes)
scripts/profile_all_kernels_2080.sh nvprof

# Parse and summarize results
python3 scripts/parse_profiling_results.py data/profiling_2080
```

## What You Get

### nvprof outputs (per kernel):
- `nvprof_{kernel}_summary.txt` - Human-readable timeline and summary
- `nvprof_{kernel}_metrics.csv` - 20+ performance metrics
- `nvprof_{kernel}_events.csv` - Hardware event counts

### ncu outputs (per kernel):
- `ncu_{kernel}_report.ncu-rep` - GUI report (open with `ncu-ui`)
- `ncu_{kernel}_details.txt` - Detailed text analysis
- `ncu_{kernel}_metrics.csv` - Selected performance metrics

### Parsed summary:
- `profiling_summary_nvprof.csv` - Key metrics for all kernels in one CSV

## Key Metrics to Look At

| Metric | Good Value | What It Means |
|--------|------------|---------------|
| achieved_occupancy | >0.5 | GPU utilization |
| sm_efficiency | >60% | Compute unit usage |
| warp_execution_efficiency | >80% | Thread divergence |
| branch_efficiency | >90% | Control flow efficiency |
| dram_utilization | varies | Memory bandwidth usage |
| gld_efficiency / gst_efficiency | >80% | Memory coalescing |
| l1_cache_global_hit_rate | >50% | Cache effectiveness |
| flop_sp_efficiency | varies | Compute throughput |

## Common Workflows

### 1. Quick Check - Is My Kernel Fast?
```bash
scripts/profile_nvprof_2080.sh my_kernel
cat data/profiling/nvprof_my_kernel_summary.txt | grep "GPU activities"
```

### 2. Find Memory Issues
```bash
scripts/profile_nvprof_2080.sh my_kernel
grep -E "(gld_efficiency|gst_efficiency|dram)" data/profiling/nvprof_my_kernel_metrics.csv
```

### 3. Find Divergence Issues
```bash
scripts/profile_nvprof_2080.sh my_kernel
grep -E "(warp_execution|branch_efficiency)" data/profiling/nvprof_my_kernel_metrics.csv
```

### 4. Deep Analysis
```bash
scripts/profile_ncu_2080.sh my_kernel
ncu-ui data/profiling/ncu_my_kernel_report.ncu-rep
```

### 5. Compare Two Versions
```bash
# Profile original
scripts/profile_nvprof_2080.sh matmul_naive

# Profile optimized
scripts/profile_nvprof_2080.sh matmul_tiled

# Compare
python3 scripts/parse_profiling_results.py data/profiling
cat data/profiling/profiling_summary_nvprof.csv | grep matmul
```

### 6. Batch Profile Everything
```bash
# Profile all 16 kernels
scripts/profile_all_kernels_2080.sh nvprof data/batch_2080

# Generate summary
python3 scripts/parse_profiling_results.py data/batch_2080

# View results
cat data/batch_2080/profiling_summary_nvprof.csv
```

## Interpreting Results

### Memory Bound Kernel
```
dram_utilization: 80-100%
flop_sp_efficiency: <20%
→ Limited by memory bandwidth, not compute
→ Optimize: coalescing, caching, reuse
```

### Compute Bound Kernel
```
flop_sp_efficiency: 60-100%
dram_utilization: <50%
→ Limited by compute throughput
→ Optimize: occupancy, instruction mix
```

### Divergence Issues
```
warp_execution_efficiency: <70%
branch_efficiency: <80%
→ Threads within warps take different paths
→ Optimize: reduce conditionals, restructure
```

### Cache Thrashing
```
l1_cache_global_hit_rate: <30%
gld_efficiency: <60%
→ Poor cache reuse or uncoalesced access
→ Optimize: access patterns, blocking
```

## Tool Comparison

### Use nvprof when:
- ✅ Quick initial profiling
- ✅ Comparing many kernels
- ✅ Older CUDA toolkit (10.x, early 11.x)
- ✅ Scripting and automation

### Use ncu when:
- ✅ Deep analysis needed
- ✅ Want optimization suggestions
- ✅ GUI analysis preferred
- ✅ Modern CUDA toolkit (11+, 12+)

## Next Steps

1. **Profile your kernels** - Start with the batch script
2. **Identify bottlenecks** - Look at key metrics
3. **Read full guide** - See [PROFILING_GUIDE.md](PROFILING_GUIDE.md)
4. **Optimize** - Make targeted improvements
5. **Validate** - Re-profile to confirm gains

## Available Kernels

- `vector_add` - Memory bandwidth baseline
- `saxpy` - FMA operations
- `strided_copy_8` - Strided access patterns
- `naive_transpose` - Uncoalesced memory
- `shared_transpose` - Optimized with shared memory
- `reduce_sum` - Parallel reduction
- `dot_product` - Reduction + compute
- `histogram` - Atomic operations
- `conv2d_3x3`, `conv2d_7x7` - Convolutions
- `random_access` - Random memory patterns
- `vector_add_divergent` - Branch divergence
- `shared_bank_conflict` - Bank conflicts
- `atomic_hotspot` - Atomic contention
- `matmul_naive`, `matmul_tiled` - Matrix multiplication

## Example Output

```
=== Profiling matmul_tiled on 2080 with nvprof ===
Running basic summary profiling...
Summary saved to: data/profiling/nvprof_matmul_tiled_summary.txt
Collecting comprehensive metrics...
Metrics saved to: data/profiling/nvprof_matmul_tiled_metrics.csv
Collecting hardware events...
Events saved to: data/profiling/nvprof_matmul_tiled_events.csv

=== Profiling Complete ===
```

Then view results:
```bash
cat data/profiling/nvprof_matmul_tiled_metrics.csv | grep -E "(occupancy|efficiency)"
```

## Troubleshooting

**Error: nvprof not found**
- Install CUDA Toolkit 10.x or 11.x
- Add to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

**Error: ncu not found**
- Install NVIDIA Nsight Compute from NVIDIA website

**Error: Permission denied**
- Some metrics need root: `sudo scripts/profile_nvprof_2080.sh kernel_name`

**Error: Kernel not found**
- Check `bin/runner` exists
- Use exact kernel name from list above

## Full Documentation

See [PROFILING_GUIDE.md](PROFILING_GUIDE.md) for:
- Detailed metric explanations
- Advanced usage examples
- Integration with dataset generation
- RTX 2080 architecture details
- Complete troubleshooting guide
