# GPU Kernel Profiling for RTX 2080 - README

## Quick Start (When on RTX 2080 Machine)

### 1. First Time Setup
```bash
# Check if you have profiling tools
scripts/profile_kernel_2080.sh vector_add

# If you get "nvprof: command not found" or "ncu: command not found"
# See PROFILING_SETUP.md for installation instructions
```

### 2. Install Profiling Tools (If Needed)

**Recommended: Install NVIDIA Nsight Compute**
```bash
sudo apt install nvidia-nsight-compute

# Verify
ncu --version
```

**Alternative: See PROFILING_SETUP.md for other options**

### 3. Profile Your Kernels

**Single kernel:**
```bash
cd /home/user/test1/gpu-perf
scripts/profile_kernel_2080.sh vector_add
```

**All kernels (~5-10 minutes):**
```bash
scripts/profile_all_2080.sh
```

**Custom arguments:**
```bash
scripts/profile_kernel_2080.sh matmul_tiled "--rows 1024 --cols 1024 --warmup 10 --reps 20"
```

## What We've Built

### Smart Profiling Scripts

1. **profile_kernel_2080.sh** (RECOMMENDED)
   - Auto-detects which profiler you have (ncu/nvprof/nsys)
   - Uses the best available tool automatically
   - Provides helpful installation instructions if tools missing
   - Works with CUDA 10.x through 12.x

2. **profile_all_2080.sh** (RECOMMENDED)
   - Profiles all 16 kernels automatically
   - Uses the smart auto-detect script
   - Generates summary at the end

3. **profile_nvprof_2080.sh** (Legacy, if you specifically want nvprof)
   - Only works if nvprof is installed (CUDA ≤ 11.x)
   - Direct nvprof usage, no auto-detection

4. **profile_ncu_2080.sh** (Direct ncu, if you specifically want it)
   - Only works if ncu is installed
   - Direct ncu usage, no auto-detection

### Parser and Analysis

5. **parse_profiling_results.py**
   - Parses CSV output from nvprof or ncu
   - Generates summary CSV with key metrics
   - Prints human-readable report

## Available Kernels

All 16 kernels from your benchmark suite:
- vector_add, saxpy, strided_copy_8
- naive_transpose, shared_transpose
- reduce_sum, dot_product
- histogram
- conv2d_3x3, conv2d_7x7
- random_access, vector_add_divergent
- shared_bank_conflict, atomic_hotspot
- matmul_naive, matmul_tiled

## Output Directory Structure

After running `scripts/profile_all_2080.sh`:

```
data/profiling_2080/
├── ncu_vector_add_report.ncu-rep         # GUI report (if using ncu)
├── ncu_vector_add_details.txt            # Text details
├── ncu_vector_add_metrics.csv            # Metrics
├── ncu_saxpy_report.ncu-rep
├── ncu_saxpy_details.txt
├── ncu_saxpy_metrics.csv
... (for all 16 kernels)
└── profiling_summary_ncu.csv             # Combined summary
```

Or with nvprof:
```
data/profiling_2080/
├── nvprof_vector_add_summary.txt
├── nvprof_vector_add_metrics.csv
├── nvprof_vector_add_events.csv
... (for all 16 kernels)
└── profiling_summary_nvprof.csv
```

## Common Workflows

### Workflow 1: Quick Check on One Kernel
```bash
# Profile
scripts/profile_kernel_2080.sh vector_add

# View summary (if ncu)
cat data/profiling/ncu_vector_add_details.txt | head -50

# View summary (if nvprof)
cat data/profiling/nvprof_vector_add_summary.txt
```

### Workflow 2: Find the Slowest Kernels
```bash
# Profile all
scripts/profile_all_2080.sh

# Parse and view summary
python3 scripts/parse_profiling_results.py data/profiling_2080
cat data/profiling_2080/profiling_summary_*.csv
```

### Workflow 3: Deep Dive on Problem Kernel
```bash
# Profile with ncu for detailed analysis
scripts/profile_kernel_2080.sh matmul_naive

# Open in GUI (if you have display)
ncu-ui data/profiling/ncu_matmul_naive_report.ncu-rep

# Or view text report
cat data/profiling/ncu_matmul_naive_details.txt
```

### Workflow 4: Compare Before/After Optimization
```bash
# Profile before
scripts/profile_kernel_2080.sh matmul_naive data/before_opt

# Make your optimizations...

# Profile after
scripts/profile_kernel_2080.sh matmul_tiled data/after_opt

# Compare
diff data/before_opt/ncu_matmul_naive_metrics.csv data/after_opt/ncu_matmul_tiled_metrics.csv
```

## Key Metrics to Watch

### Memory-Bound Kernels
Look for:
- **gld_efficiency / gst_efficiency** (should be >80%)
- **dram_utilization** (how much bandwidth you're using)
- **l1_cache_hit_rate** (should be >50% if reuse expected)

Examples: vector_add, saxpy, naive_transpose

### Compute-Bound Kernels
Look for:
- **sm_efficiency** (should be >60%)
- **achieved_occupancy** (should be >0.5)
- **flop_sp_efficiency** (varies by kernel)

Examples: matmul_tiled, conv2d_*

### Divergence Problems
Look for:
- **warp_execution_efficiency** (should be >80%)
- **branch_efficiency** (should be >90%)

Examples: vector_add_divergent (intentionally bad)

### Shared Memory Issues
Look for:
- **shared_efficiency** (should be high)
- Bank conflicts (visible in ncu detailed view)

Examples: shared_bank_conflict (intentionally bad), shared_transpose (good)

## Documentation

- **PROFILING_SETUP.md** - How to install profiling tools (START HERE if tools not found)
- **PROFILING_GUIDE.md** - Comprehensive guide with all details
- **PROFILING_QUICKSTART.md** - Quick reference for common tasks
- **README_PROFILING.md** - This file

## Troubleshooting

### Error: "nvprof: command not found"
→ Install ncu instead (see PROFILING_SETUP.md)
→ Or use the auto-detect script: `scripts/profile_kernel_2080.sh`

### Error: "ncu: command not found"
→ Install it: `sudo apt install nvidia-nsight-compute`
→ See PROFILING_SETUP.md for details

### Error: "Permission denied"
→ Some metrics need sudo: `sudo scripts/profile_kernel_2080.sh vector_add`
→ Or: `sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`

### Error: "CUDA driver version is insufficient"
→ Update your NVIDIA driver: `sudo apt install nvidia-driver-535`

### Profile runs but no output
→ Check the output directory: `ls -la data/profiling/`
→ Check for errors in the summary files

## Running Without Profiling Tools

If you can't install profiling tools, you can still get basic timing:

```bash
# Direct timing with runner
bin/runner --kernel vector_add --rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100

# Using trials script for statistics
scripts/run_trials.sh vector_add "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100" 12 0 10
```

This gives you execution time, from which you can calculate:
- Throughput (GB/s)
- GFLOPS
- Compare relative performance

## Integration with Your Workflow

### Add Profiling to Dataset Generation

You can incorporate profiling metrics into your dataset:

```bash
# 1. Run standard dataset generation
scripts/run_2080ti_full.sh

# 2. Profile all kernels
scripts/profile_all_2080.sh

# 3. Parse profiling results
python3 scripts/parse_profiling_results.py data/profiling_2080

# 4. Merge with your existing CSV
# (You'll need to write a merge script or do this manually)
```

### Continuous Profiling

Add to your workflow:

```bash
# Before each major change
scripts/profile_all_2080.sh data/profiling_baseline

# After optimization
scripts/profile_all_2080.sh data/profiling_optimized

# Compare
python3 scripts/parse_profiling_results.py data/profiling_baseline
python3 scripts/parse_profiling_results.py data/profiling_optimized
```

## Next Steps

1. **Install tools** (if on RTX 2080 machine)
   ```bash
   sudo apt install nvidia-nsight-compute
   ```

2. **Test installation**
   ```bash
   scripts/profile_kernel_2080.sh vector_add
   ```

3. **Profile all kernels**
   ```bash
   scripts/profile_all_2080.sh
   ```

4. **Analyze results**
   ```bash
   python3 scripts/parse_profiling_results.py data/profiling_2080
   ```

5. **Identify bottlenecks** and optimize!

## Getting Help

1. Read the docs:
   - PROFILING_SETUP.md (installation)
   - PROFILING_GUIDE.md (detailed metrics)
   - PROFILING_QUICKSTART.md (quick commands)

2. Check your setup:
   ```bash
   nvidia-smi           # Check GPU
   nvcc --version       # Check CUDA
   ncu --version        # Check profiler
   which ncu            # Check PATH
   ```

3. Try the auto-detect script:
   ```bash
   scripts/profile_kernel_2080.sh vector_add
   ```

4. Use basic timing as fallback:
   ```bash
   bin/runner --kernel vector_add --warmup 10 --reps 100
   ```
