# GPU Kernel Profiling Guide for CUDA Cluster

## Overview

This guide shows how to profile your GPU kernels using NVIDIA profiling tools on a CUDA cluster.

---

## Available Profiling Tools

| Tool | Use Case | Best For |
|------|----------|----------|
| **Nsight Compute (ncu)** | Detailed kernel metrics | Bottleneck analysis, optimization |
| **Nsight Systems (nsys)** | Timeline view | Multi-kernel analysis, CPU-GPU interaction |
| **nvprof** | Legacy profiler | Older CUDA versions (deprecated) |

---

## 1. Nsight Compute (ncu) - Recommended

### Basic Kernel Profiling

```bash
# Profile a single kernel run
ncu --set full ./bin/runner --kernel vector_add --N 1048576 --block 256

# Save detailed report
ncu --set full -o vector_add_profile ./bin/runner --kernel vector_add --N 1048576 --block 256

# View the report (opens GUI or generates text)
ncu --import vector_add_profile.ncu-rep
```

### Specific Metrics Collection

```bash
# Memory throughput metrics
ncu --metrics dram_throughput,l2_throughput,shared_throughput \
    ./bin/runner --kernel matmul_tiled --matN 1024

# Compute metrics
ncu --metrics sm_efficiency,achieved_occupancy,ipc \
    ./bin/runner --kernel saxpy --N 1048576 --block 256

# Warp divergence metrics
ncu --metrics branch_efficiency,warp_execution_efficiency \
    ./bin/runner --kernel vector_add_divergent --N 1048576 --block 256

# Atomic contention metrics
ncu --metrics atomic_throughput,atomic_transactions \
    ./bin/runner --kernel atomic_hotspot --N 1048576 --block 256 --iters 100
```

### Full Metric Set for All Kernels

```bash
# Comprehensive profiling (takes longer)
ncu --set full \
    --target-processes all \
    --kernel-name-base function \
    -o full_profile \
    ./bin/runner --kernel matmul_tiled --matN 512

# Export to CSV for analysis
ncu --csv --import full_profile.ncu-rep > full_profile.csv
```

---

## 2. Nsight Systems (nsys) - Timeline Analysis

### Profile All Kernels in Sequence

```bash
# Generate timeline
nsys profile -o timeline \
    --stats=true \
    ./bin/runner --kernel vector_add --N 1048576 --block 256

# View report
nsys stats timeline.nsys-rep

# Export to SQLite for custom analysis
nsys export -o timeline.sqlite timeline.nsys-rep
```

### Profile Script Running Multiple Kernels

```bash
# Create a profiling script
cat > profile_all.sh << 'EOF'
#!/bin/bash
for kernel in vector_add saxpy matmul_tiled reduce_sum; do
    ./bin/runner --kernel $kernel --N 1048576 --block 256
done
EOF

chmod +x profile_all.sh

# Profile the entire script
nsys profile -o all_kernels_timeline --stats=true ./profile_all.sh
```

---

## 3. Legacy nvprof (Older CUDA Versions)

```bash
# Basic profiling
nvprof ./bin/runner --kernel vector_add --N 1048576 --block 256

# Detailed metrics
nvprof --metrics all ./bin/runner --kernel matmul_tiled --matN 512

# Save to file
nvprof --analysis-metrics -o vector_add.nvvp \
    ./bin/runner --kernel vector_add --N 1048576 --block 256
```

---

## 4. Batch Profiling All Kernels

### Create Profiling Script

```bash
cd ~/gpu-perf

# Create profiling directory
mkdir -p profiles

cat > scripts/profile_all_kernels.sh << 'EOF'
#!/bin/bash
set -euo pipefail

PROFILE_DIR="profiles"
mkdir -p $PROFILE_DIR

echo "=== Profiling all kernels with Nsight Compute ==="

# 1D kernels
for kernel in vector_add saxpy strided_copy_8 reduce_sum dot_product \
              histogram random_access vector_add_divergent; do
    echo "Profiling $kernel..."
    ncu --set full -o ${PROFILE_DIR}/${kernel} \
        ./bin/runner --kernel $kernel --N 1048576 --block 256 \
        > ${PROFILE_DIR}/${kernel}.log 2>&1
done

# 2D kernels
for kernel in naive_transpose shared_transpose conv2d_3x3 conv2d_7x7; do
    echo "Profiling $kernel..."
    ncu --set full -o ${PROFILE_DIR}/${kernel} \
        ./bin/runner --kernel $kernel --rows 1024 --cols 1024 \
        > ${PROFILE_DIR}/${kernel}.log 2>&1
done

# Matrix multiply
for kernel in matmul_naive matmul_tiled; do
    echo "Profiling $kernel..."
    ncu --set full -o ${PROFILE_DIR}/${kernel} \
        ./bin/runner --kernel $kernel --matN 512 \
        > ${PROFILE_DIR}/${kernel}.log 2>&1
done

# Special cases
echo "Profiling shared_bank_conflict..."
ncu --set full -o ${PROFILE_DIR}/shared_bank_conflict \
    ./bin/runner --kernel shared_bank_conflict \
    > ${PROFILE_DIR}/shared_bank_conflict.log 2>&1

echo "Profiling atomic_hotspot..."
ncu --set full -o ${PROFILE_DIR}/atomic_hotspot \
    ./bin/runner --kernel atomic_hotspot --N 1048576 --block 256 --iters 100 \
    > ${PROFILE_DIR}/atomic_hotspot.log 2>&1

echo ""
echo "=== Profiling complete! ==="
echo "Reports saved to: $PROFILE_DIR/"
ls -lh $PROFILE_DIR/*.ncu-rep
EOF

chmod +x scripts/profile_all_kernels.sh

# Run profiling
./scripts/profile_all_kernels.sh
```

---

## 5. Extract Key Metrics to CSV

```bash
# Extract specific metrics from all profiles
cat > scripts/extract_profile_metrics.sh << 'EOF'
#!/bin/bash

PROFILE_DIR="profiles"
OUTPUT="profile_metrics.csv"

echo "kernel,sm_efficiency,achieved_occupancy,dram_throughput,l2_hit_rate,warp_execution_efficiency" > $OUTPUT

for profile in ${PROFILE_DIR}/*.ncu-rep; do
    kernel=$(basename $profile .ncu-rep)

    # Extract metrics (adjust metric names as needed)
    ncu --import $profile --csv --page raw | \
        grep -E "sm__throughput.avg.pct_of_peak_sustained_elapsed|achieved_occupancy|dram_throughput|l2_tex_hit_rate|warp_execution_efficiency" | \
        awk -v kernel=$kernel '{print kernel "," $0}'
done >> $OUTPUT

echo "Metrics extracted to: $OUTPUT"
EOF

chmod +x scripts/extract_profile_metrics.sh
./scripts/extract_profile_metrics.sh
```

---

## 6. Cluster-Specific Commands

### SLURM Cluster

```bash
# Interactive profiling session
srun --gres=gpu:1 --pty bash
cd ~/gpu-perf
ncu --set full ./bin/runner --kernel vector_add --N 1048576 --block 256

# Batch job profiling
cat > profile_job.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=gpu_profile
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=8G

module load cuda/11.8  # Adjust version as needed

cd ~/gpu-perf
./scripts/profile_all_kernels.sh
EOF

sbatch profile_job.slurm
```

### PBS/Torque Cluster

```bash
cat > profile_job.pbs << 'EOF'
#!/bin/bash
#PBS -N gpu_profile
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=01:00:00
#PBS -o profile.out
#PBS -e profile.err

module load cuda/11.8

cd $PBS_O_WORKDIR
cd gpu-perf
./scripts/profile_all_kernels.sh
EOF

qsub profile_job.pbs
```

---

## 7. Useful Metric Sets

### Memory-Bound Kernels
```bash
ncu --metrics dram_throughput,l2_throughput,global_hit_rate,shared_efficiency \
    ./bin/runner --kernel vector_add --N 1048576 --block 256
```

### Compute-Bound Kernels
```bash
ncu --metrics sm_efficiency,achieved_occupancy,ipc,flop_count_sp \
    ./bin/runner --kernel matmul_tiled --matN 1024
```

### Divergence Analysis
```bash
ncu --metrics branch_efficiency,warp_execution_efficiency,warp_nonpred_execution_efficiency \
    ./bin/runner --kernel vector_add_divergent --N 1048576 --block 256
```

### Atomic Contention
```bash
ncu --metrics atomic_throughput,atomic_transactions,atomic_transactions_per_request \
    ./bin/runner --kernel atomic_hotspot --N 1048576 --block 256 --iters 100
```

---

## 8. Quick Reference Commands

```bash
# Check which profilers are available
which ncu nsys nvprof

# Get CUDA version
nvcc --version

# List available metrics
ncu --query-metrics

# List available sections
ncu --list-sections

# Profile with specific section
ncu --section MemoryWorkloadAnalysis \
    ./bin/runner --kernel vector_add --N 1048576 --block 256

# Compare two runs
ncu --set full -o baseline ./bin/runner --kernel matmul_naive --matN 512
ncu --set full -o optimized ./bin/runner --kernel matmul_tiled --matN 512
ncu --import baseline.ncu-rep optimized.ncu-rep  # Opens comparison view
```

---

## 9. Profiling Best Practices

### On Shared Clusters

1. **Use compute nodes, not login nodes:**
   ```bash
   srun --gres=gpu:1 --pty bash  # Get interactive GPU session
   ```

2. **Limit profiling overhead:**
   ```bash
   # Use targeted metrics instead of --set full
   ncu --metrics sm_efficiency,achieved_occupancy,dram_throughput \
       ./bin/runner --kernel vector_add --N 1048576 --block 256
   ```

3. **Profile representative sizes:**
   ```bash
   # Use realistic problem sizes
   ncu ./bin/runner --kernel matmul_tiled --matN 2048  # Not too small
   ```

4. **Reduce repetitions for profiling:**
   ```bash
   # Override warmup/reps for faster profiling
   ncu ./bin/runner --kernel vector_add --N 1048576 --block 256 --warmup 1 --reps 1
   ```

---

## 10. Output Analysis

### View Text Report
```bash
# Generate text summary
ncu --import profile.ncu-rep --print-summary stdout

# Export full report to text
ncu --import profile.ncu-rep --page raw > profile_detailed.txt
```

### Generate HTML Report
```bash
# Export to HTML (can view in browser)
ncu --import profile.ncu-rep --export profile --format html
```

### Compare Before/After Optimization
```bash
# Profile unoptimized
ncu --set full -o naive ./bin/runner --kernel matmul_naive --matN 1024

# Profile optimized
ncu --set full -o tiled ./bin/runner --kernel matmul_tiled --matN 1024

# Generate comparison report
ncu --import naive.ncu-rep tiled.ncu-rep --print-summary stdout
```

---

## 11. Automated Profiling Pipeline

```bash
# Complete profiling + analysis pipeline
cat > scripts/profile_and_analyze.sh << 'EOF'
#!/bin/bash
set -euo pipefail

KERNEL=$1
PROFILE_DIR="profiles"
mkdir -p $PROFILE_DIR

echo "=== Profiling $KERNEL ==="

# Profile with full metrics
ncu --set full -o ${PROFILE_DIR}/${KERNEL} \
    ./bin/runner --kernel $KERNEL --N 1048576 --block 256

# Generate text summary
ncu --import ${PROFILE_DIR}/${KERNEL}.ncu-rep --print-summary stdout \
    > ${PROFILE_DIR}/${KERNEL}_summary.txt

# Export key metrics to CSV
ncu --import ${PROFILE_DIR}/${KERNEL}.ncu-rep --csv --page raw \
    > ${PROFILE_DIR}/${KERNEL}_metrics.csv

echo "✓ Profile saved to: ${PROFILE_DIR}/${KERNEL}.ncu-rep"
echo "✓ Summary saved to: ${PROFILE_DIR}/${KERNEL}_summary.txt"
echo "✓ Metrics saved to: ${PROFILE_DIR}/${KERNEL}_metrics.csv"
EOF

chmod +x scripts/profile_and_analyze.sh

# Usage
./scripts/profile_and_analyze.sh vector_add
```

---

## Troubleshooting

### "ncu: command not found"
```bash
# Load CUDA module (cluster-specific)
module load cuda/11.8

# Or add to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### "Permission denied" on profiling
```bash
# May need to run with elevated permissions or configure system
# Check with cluster admin about profiling permissions
```

### "Profiling overhead too high"
```bash
# Use lighter metric sets
ncu --metrics sm_efficiency,achieved_occupancy \
    ./bin/runner --kernel vector_add --N 1048576 --block 256

# Or profile smaller problem sizes
ncu --set full ./bin/runner --kernel vector_add --N 65536 --block 256
```

---

## Next Steps

1. **Profile all kernels:** `./scripts/profile_all_kernels.sh`
2. **Extract metrics:** `./scripts/extract_profile_metrics.sh`
3. **Analyze bottlenecks:** Compare achieved vs peak metrics
4. **Optimize:** Use profiling insights to improve kernels
5. **Re-profile:** Validate optimizations with new profiles
