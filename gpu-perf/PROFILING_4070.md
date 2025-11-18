# Profiling Kernels on RTX 4070

This guide shows how to profile GPU kernels using NVIDIA Nsight Compute (ncu) on cuda5 (RTX 4070).

## Prerequisites

1. SSH to cuda5:
   ```bash
   ssh cuda5
   ```

2. Load CUDA module (if needed):
   ```bash
   module load cuda
   ```

3. Build the kernels:
   ```bash
   cd /home/user/test1/gpu-perf
   make
   ```

## Quick Start

### Profile a Single Kernel

```bash
cd /home/user/test1/gpu-perf

# Profile vector_add
scripts/profile_4070.sh vector_add --N 1048576

# Profile matmul_tiled
scripts/profile_4070.sh matmul_tiled --rows 1024 --cols 1024
```

### Profile All Kernels

```bash
cd /home/user/test1/gpu-perf
scripts/profile_all_4070.sh
```

This will profile all 17 kernels and save results to `data/profiling_4070/`.

## Viewing Results

### Command Line

View the text log with detailed metrics:
```bash
cat data/profiling_4070/ncu_<kernel>_details.txt
```

### GUI (ncu-ui)

Open the interactive GUI to explore metrics:
```bash
ncu-ui data/profiling_4070/ncu_<kernel>.ncu-rep
```

## Output Files

For each kernel, profiling generates:
- `ncu_<kernel>.ncu-rep` - Binary report for ncu-ui
- `ncu_<kernel>_details.txt` - Text log with all metrics

## What Metrics Are Collected

NCU with `--set full` collects comprehensive metrics including:

**Kernel Launch**
- Grid size, block size
- Registers per thread
- Shared memory usage

**Execution**
- Kernel duration
- Achieved occupancy
- Theoretical occupancy

**Compute**
- SM utilization
- Pipeline utilization
- Instructions executed

**Memory**
- DRAM throughput
- L1/L2 cache hit rates
- Memory bandwidth utilization

**Efficiency**
- Branch efficiency
- Warp execution efficiency
- Instruction efficiency

## Example Workflow

1. Profile all kernels:
   ```bash
   scripts/profile_all_4070.sh
   ```

2. View summary of what was profiled:
   ```bash
   ls -lh data/profiling_4070/
   ```

3. Check a specific kernel's metrics:
   ```bash
   cat data/profiling_4070/ncu_vector_add_details.txt | less
   ```

4. Open in GUI for interactive analysis:
   ```bash
   ncu-ui data/profiling_4070/ncu_vector_add.ncu-rep
   ```

## Notes

- Profiling uses `--warmup 1 --reps 1` to minimize profiling time
- Matrix sizes are reduced (512x512) during profiling to save time
- All profiling should be done on cuda5 where ncu works correctly
- The RTX 4070 has 46 SMs and 12GB VRAM

## Troubleshooting

**ncu not found:**
```bash
module load cuda
ncu --version
```

**bin/runner not found:**
```bash
make clean
make
```

**Out of memory:**
Reduce the problem size in the profiling command:
```bash
scripts/profile_4070.sh matmul_tiled --rows 256 --cols 256
```
