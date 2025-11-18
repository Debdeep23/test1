# GPU Kernel Profiling Guide - RTX 2080

This guide explains how to profile CUDA kernels on the RTX 2080 GPU using nvprof and NVIDIA Nsight Compute (ncu).

## Overview

We provide three profiling scripts:
1. **profile_nvprof_2080.sh** - Profile individual kernels using nvprof (legacy tool, good for basic metrics)
2. **profile_ncu_2080.sh** - Profile individual kernels using ncu (modern tool, more detailed analysis)
3. **profile_all_kernels_2080.sh** - Batch profile all kernels using either tool

## Prerequisites

### For nvprof (Legacy, works on all GPUs)
- CUDA Toolkit 10.x or 11.x (nvprof deprecated in 11.x)
- Works on RTX 2080, 2080 Ti

### For ncu (Modern, recommended)
- CUDA Toolkit 11.0 or newer
- NVIDIA Nsight Compute installed
- Works on all recent GPUs (Turing, Ampere, etc.)

## Quick Start

### Profile a Single Kernel with nvprof

```bash
# Basic usage
cd /home/user/test1/gpu-perf
scripts/profile_nvprof_2080.sh vector_add

# With custom arguments
scripts/profile_nvprof_2080.sh vector_add "--rows 2097152 --cols 1 --block 512 --warmup 10 --reps 20"

# Specify output directory
scripts/profile_nvprof_2080.sh vector_add "--rows 1048576 --cols 1 --block 256" data/my_profiling
```

### Profile a Single Kernel with ncu

```bash
# Basic usage
scripts/profile_ncu_2080.sh matmul_tiled

# With custom arguments
scripts/profile_ncu_2080.sh matmul_tiled "--rows 1024 --cols 1024 --warmup 10 --reps 20"

# Specify output directory
scripts/profile_ncu_2080.sh matmul_tiled "--rows 512 --cols 512" data/my_profiling
```

### Profile All Kernels

```bash
# Profile all kernels with nvprof
scripts/profile_all_kernels_2080.sh nvprof

# Profile all kernels with ncu
scripts/profile_all_kernels_2080.sh ncu

# Specify output directory
scripts/profile_all_kernels_2080.sh nvprof data/batch_profiling_2080
```

## Output Files

### nvprof Output
Each kernel profiling generates three files:

1. **nvprof_{kernel}_summary.txt** - Human-readable summary with:
   - GPU trace timeline
   - API calls summary
   - Kernel execution times

2. **nvprof_{kernel}_metrics.csv** - Detailed metrics in CSV format:
   - FLOP counts and efficiency
   - Memory throughput (DRAM, L2, L1)
   - Memory efficiency (load/store)
   - Occupancy and SM efficiency
   - Warp execution efficiency
   - Branch efficiency
   - Cache hit rates
   - Instruction counts

3. **nvprof_{kernel}_events.csv** - Hardware events in CSV format:
   - Active warps and cycles
   - Memory operations (shared, global, local)
   - Branch and divergent branch counts

### ncu Output
Each kernel profiling generates three files:

1. **ncu_{kernel}_report.ncu-rep** - Binary report file
   - Open with `ncu-ui` for interactive GUI analysis
   - Contains complete profiling data

2. **ncu_{kernel}_details.txt** - Detailed text report:
   - Kernel configuration
   - Resource usage
   - Performance limiters
   - Optimization suggestions

3. **ncu_{kernel}_metrics.csv** - Selected metrics in CSV:
   - Kernel duration
   - SM, DRAM, L1, L2 throughput
   - Memory throughput utilization
   - Warp activity
   - Issue stalls (scoreboard, drain)
   - Floating-point operations
   - Shared memory operations

## Key Metrics Explained

### Memory Metrics
- **dram_throughput** - Memory bandwidth to/from global memory
- **dram_utilization** - % of peak DRAM bandwidth used
- **l1/l2_cache_hit_rate** - % of memory accesses served by cache
- **gld/gst_efficiency** - Ratio of requested to actual global memory transactions

### Compute Metrics
- **flop_count_sp** - Single-precision floating-point operations
- **flop_sp_efficiency** - % of peak FLOPS achieved
- **achieved_occupancy** - Average active warps / max possible warps
- **sm_efficiency** - % of time at least one warp is active

### Execution Metrics
- **warp_execution_efficiency** - % of threads in warps that execute (low = divergence)
- **branch_efficiency** - % of branches that don't diverge
- **inst_per_warp** - Average instructions executed per warp

## Example Workflow

### 1. Start with Quick Profiling
```bash
# Quick profile of a problematic kernel
scripts/profile_nvprof_2080.sh vector_add_divergent
```

### 2. Analyze Summary
```bash
# View the summary
cat data/profiling/nvprof_vector_add_divergent_summary.txt
```

Look for:
- High execution time
- Low occupancy
- Poor memory efficiency
- High divergence

### 3. Deep Dive with ncu
```bash
# Get detailed analysis
scripts/profile_ncu_2080.sh vector_add_divergent

# Open in GUI for interactive analysis
ncu-ui data/profiling/ncu_vector_add_divergent_report.ncu-rep
```

### 4. Compare Kernels
```bash
# Profile similar kernels
scripts/profile_nvprof_2080.sh vector_add
scripts/profile_nvprof_2080.sh vector_add_divergent

# Compare metrics
grep "achieved_occupancy" data/profiling/nvprof_vector_add_metrics.csv
grep "achieved_occupancy" data/profiling/nvprof_vector_add_divergent_metrics.csv
```

## Common Performance Issues and Metrics

### Low Memory Bandwidth
**Symptoms:**
- Low `dram_throughput`
- Low `dram_utilization`

**Check:**
- Global memory access patterns (coalescing)
- Use `gld_efficiency` and `gst_efficiency`

### Low Compute Utilization
**Symptoms:**
- Low `flop_sp_efficiency`
- Low `sm_efficiency`

**Check:**
- Occupancy (`achieved_occupancy`)
- Register usage
- Shared memory usage

### Warp Divergence
**Symptoms:**
- Low `warp_execution_efficiency`
- Low `branch_efficiency`

**Check:**
- `divergent_branch` events
- Conditional code in kernels

### Memory Bound
**Symptoms:**
- High `dram_throughput` but low compute
- Low arithmetic intensity

**Check:**
- Ratio of FLOPS to memory ops
- Cache hit rates

### Bank Conflicts
**Symptoms:**
- Low `shared_efficiency`
- High shared memory stalls

**Check:**
- Shared memory access patterns
- Benchmark: `shared_bank_conflict` kernel

## Available Kernels

All kernels from the benchmark suite can be profiled:

| Kernel | Purpose | Key Metrics |
|--------|---------|-------------|
| vector_add | Memory bandwidth baseline | dram_throughput, memory_efficiency |
| saxpy | FMA operations | flop_sp_efficiency, arithmetic_intensity |
| strided_copy_8 | Strided memory access | gld_efficiency, cache_hit_rate |
| naive_transpose | Uncoalesced memory | gld/gst_efficiency |
| shared_transpose | Optimized transpose | shared_efficiency, occupancy |
| reduce_sum | Reduction pattern | shared_efficiency, warp_execution |
| dot_product | Reduction + compute | flop_efficiency, memory_efficiency |
| histogram | Atomic operations | atomic_throughput, occupancy |
| conv2d_3x3 | Small convolution | cache_hit_rate, arithmetic_intensity |
| conv2d_7x7 | Large convolution | register_usage, occupancy |
| random_access | Random memory pattern | cache_miss_rate, dram_utilization |
| vector_add_divergent | Branch divergence | warp_execution_efficiency, branch_efficiency |
| shared_bank_conflict | Bank conflicts | shared_efficiency, memory_conflicts |
| atomic_hotspot | Atomic contention | atomic_throughput, serialization |
| matmul_naive | Naive matrix multiply | arithmetic_intensity, cache_hit_rate |
| matmul_tiled | Tiled matrix multiply | shared_efficiency, occupancy, flop_efficiency |

## Tips for RTX 2080

### GPU Architecture (Turing TU104)
- **SM count**: 46 (for RTX 2080)
- **CUDA cores**: 2944
- **Memory bandwidth**: 448 GB/s
- **L2 cache**: 4 MB
- **Shared memory**: 64 KB per SM (configurable)
- **Max threads per SM**: 1024
- **Max blocks per SM**: 16

### Profiling Best Practices
1. **Warm up the GPU**: Use `--warmup 5` or more to stabilize clocks
2. **Multiple runs**: Use `--reps 10` or more for statistical significance
3. **Start with nvprof**: Faster, good for initial insights
4. **Deep dive with ncu**: When you need detailed analysis
5. **Profile incrementally**: Test one optimization at a time

### Known Issues
- nvprof is deprecated in CUDA 11+, may not work on newer toolkits
- ncu can be slower for profiling (runs kernel multiple times internally)
- Some metrics may require elevated privileges (`sudo`)

## Advanced Usage

### Custom Metric Sets with nvprof
Edit `profile_nvprof_2080.sh` to add/remove metrics:

```bash
nvprof --csv \
       --metrics metric1,metric2,metric3 \
       bin/runner --kernel "$KERNEL" $ARGSTR
```

List all available metrics:
```bash
nvprof --query-metrics
```

### Custom Metric Sets with ncu
Edit `profile_ncu_2080.sh` to change metric sets:

```bash
# Use predefined sets
ncu --set full    # Most comprehensive
ncu --set detailed  # Balanced
ncu --set basic   # Quick overview

# Custom metrics
ncu --metrics metric1,metric2 \
    bin/runner --kernel "$KERNEL" $ARGSTR
```

List all available metrics:
```bash
ncu --query-metrics
```

### Comparing Across GPUs
Profile the same kernel on different GPUs:

```bash
# On RTX 2080
scripts/profile_nvprof_2080.sh matmul_tiled > profile_2080.log

# On RTX 4070 (modify script or create new one)
scripts/profile_nvprof_4070.sh matmul_tiled > profile_4070.log

# Compare
diff profile_2080.log profile_4070.log
```

## Integrating with Dataset Generation

You can incorporate profiling metrics into your dataset:

1. Run profiling for all kernels:
```bash
scripts/profile_all_kernels_2080.sh nvprof data/profiling_2080
```

2. Extract key metrics from CSV files:
```bash
# Example: Extract achieved_occupancy for all kernels
for csv in data/profiling_2080/nvprof_*_metrics.csv; do
    kernel=$(basename "$csv" | sed 's/nvprof_//;s/_metrics.csv//')
    occupancy=$(grep "achieved_occupancy" "$csv" | cut -d',' -f2)
    echo "$kernel,$occupancy"
done
```

3. Add to your existing dataset or create a profiling dataset

## Troubleshooting

### nvprof not found
- Install CUDA Toolkit with nvprof (10.x or early 11.x)
- Add to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

### ncu not found
- Install NVIDIA Nsight Compute separately
- Available at: https://developer.nvidia.com/nsight-compute

### Permission denied errors
- Some metrics require root: `sudo scripts/profile_nvprof_2080.sh kernel_name`
- Or disable secure boot in BIOS

### Kernel not found
- Ensure runner binary is built: check `bin/runner` exists
- Kernel name must match those in `runner/main.cu`

### Out of memory
- Reduce problem size in arguments (--rows, --cols, --N)
- Use smaller --reps values

## Next Steps

1. **Profile your kernels** - Start with `scripts/profile_all_kernels_2080.sh nvprof`
2. **Analyze results** - Look for bottlenecks (memory, compute, divergence)
3. **Optimize** - Make targeted improvements based on metrics
4. **Validate** - Re-profile to confirm improvements
5. **Document** - Add findings to your dataset or reports

## References

- [NVIDIA nvprof Documentation](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
