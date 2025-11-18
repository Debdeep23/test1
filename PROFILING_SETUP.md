# Setting Up Profiling Tools for RTX 2080

## Error: nvprof: command not found

This is expected if you have CUDA 12.x or newer, as nvprof was removed. Here's how to set up profiling tools.

## Quick Solution: Use the Auto-Detect Script

We've created a smart script that detects which profiler you have:

```bash
scripts/profile_kernel_2080.sh vector_add
```

This will:
1. Check which profiling tools are available
2. Use the best available tool (ncu > nvprof > nsys)
3. Give you installation instructions if none are found

## Installation Options

### Option 1: Install NVIDIA Nsight Compute (ncu) - RECOMMENDED

**For Ubuntu/Debian:**
```bash
# If you have CUDA Toolkit 11.x or 12.x
sudo apt update
sudo apt install nvidia-nsight-compute

# Or download standalone from NVIDIA
wget https://developer.nvidia.com/downloads/nsight-compute-latest
# Follow installation instructions
```

**Verify installation:**
```bash
which ncu
ncu --version
```

**Add to PATH if needed:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export PATH=/opt/nvidia/nsight-compute/*/bin:$PATH
```

### Option 2: Install nvprof (Legacy, for CUDA 11.x or older)

**Download CUDA Toolkit 11.x:**
```bash
# Go to https://developer.nvidia.com/cuda-11-8-0-download-archive
# Select your platform and download

# Example for Ubuntu 22.04:
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

**Verify:**
```bash
which nvprof
nvprof --version
```

### Option 3: Install NVIDIA Nsight Systems (nsys)

Good for timeline profiling, less detailed metrics:

```bash
# Ubuntu/Debian
sudo apt install nvidia-nsight-systems

# Or download from
# https://developer.nvidia.com/nsight-systems
```

**Verify:**
```bash
which nsys
nsys --version
```

## Checking Your Current CUDA Version

```bash
nvcc --version
# or
cat /usr/local/cuda/version.txt
# or
nvidia-smi
```

**CUDA Version Guide:**
- **CUDA 12.x**: Use `ncu` (nvprof removed)
- **CUDA 11.x**: Use `ncu` (preferred) or `nvprof` (deprecated)
- **CUDA 10.x or older**: Use `nvprof`

## Quick Test

Once installed, test with a simple kernel:

```bash
# Test basic timing (no profiler needed)
cd /home/user/test1/gpu-perf
bin/runner --kernel vector_add --rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10

# Test with profiler
scripts/profile_kernel_2080.sh vector_add
```

## What Each Tool Provides

### ncu (NVIDIA Nsight Compute)
**Pros:**
- ✅ Most detailed metrics
- ✅ Optimization suggestions
- ✅ Modern GUI (ncu-ui)
- ✅ Works with CUDA 12+
- ✅ Active development

**Cons:**
- ❌ Slower (runs kernel multiple times)
- ❌ Larger download (~2GB)

**Best for:** Deep analysis, finding bottlenecks

### nvprof (Legacy)
**Pros:**
- ✅ Fast profiling
- ✅ Simple CSV output
- ✅ Good for batch scripting
- ✅ Smaller footprint

**Cons:**
- ❌ Deprecated (removed in CUDA 12+)
- ❌ No new features
- ❌ Limited on newer GPUs

**Best for:** Quick checks, older systems

### nsys (NVIDIA Nsight Systems)
**Pros:**
- ✅ Great timeline visualization
- ✅ CPU + GPU profiling
- ✅ Multi-GPU support

**Cons:**
- ❌ Less detailed GPU metrics
- ❌ Not ideal for kernel optimization

**Best for:** Understanding overall application flow

## Common Installation Issues

### Issue: "Permission denied" when profiling

**Solution:**
```bash
# Some metrics require elevated privileges
sudo scripts/profile_kernel_2080.sh vector_add

# Or disable secure boot in BIOS
# Or use modprobe to load nvidia driver with admin capabilities
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

### Issue: ncu installed but not in PATH

**Solution:**
```bash
# Find ncu
find /usr -name ncu 2>/dev/null
find /opt -name ncu 2>/dev/null

# Add to PATH (add to ~/.bashrc to make permanent)
export PATH=/opt/nvidia/nsight-compute/2024.1.1/bin:$PATH
```

### Issue: Multiple CUDA versions installed

**Solution:**
```bash
# Check current version
ls -la /usr/local/cuda

# Switch version (example)
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.0 /usr/local/cuda

# Update PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### Issue: "CUDA driver version is insufficient"

**Solution:**
```bash
# Check driver version
nvidia-smi

# Update driver (Ubuntu/Debian)
sudo apt update
sudo apt install nvidia-driver-535  # or latest version

# Reboot
sudo reboot
```

## Alternative: Basic Profiling Without External Tools

If you can't install profiling tools, you can still get basic metrics using CUDA events (already built into the runner):

```bash
# Basic timing is always available
bin/runner --kernel vector_add --rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100

# Output: KERNEL=vector_add N=1048576 block=256 grid=4096 time_ms=0.123456
```

This gives you kernel execution time, which you can use to:
- Calculate throughput (GB/s, GFLOPS)
- Compare before/after optimization
- Identify slowest kernels

We also have a script for this (no profiler needed):

```bash
scripts/run_trials.sh vector_add "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100" 12 0 10
```

## Recommended Setup for RTX 2080

**Minimum (no external tools):**
- ✅ CUDA Runtime (already have via driver)
- ✅ `bin/runner` (already built)
- ✅ `scripts/run_trials.sh` (basic timing)

**Recommended (better insights):**
- ✅ CUDA Toolkit 12.x
- ✅ NVIDIA Nsight Compute (ncu)
- ✅ `scripts/profile_kernel_2080.sh` (auto-detect profiler)

**Advanced (complete toolkit):**
- ✅ CUDA Toolkit 12.x
- ✅ NVIDIA Nsight Compute
- ✅ NVIDIA Nsight Systems
- ✅ Visual Profiler (ncu-ui, nsys-ui)

## Next Steps

1. **Check what you have:**
   ```bash
   scripts/profile_kernel_2080.sh vector_add
   ```

2. **Install recommended tools** (if missing):
   ```bash
   sudo apt install nvidia-nsight-compute
   ```

3. **Test installation:**
   ```bash
   ncu --version
   scripts/profile_kernel_2080.sh vector_add
   ```

4. **Profile all kernels:**
   ```bash
   # Update the batch script to use new auto-detect script
   # Or manually run for each kernel
   for kernel in vector_add saxpy matmul_tiled; do
       scripts/profile_kernel_2080.sh "$kernel"
   done
   ```

## Getting Help

If you continue to have issues:

1. Check your setup:
   ```bash
   nvidia-smi
   nvcc --version
   which ncu
   which nvprof
   echo $PATH
   ```

2. See the NVIDIA documentation:
   - [Nsight Compute Docs](https://docs.nvidia.com/nsight-compute/)
   - [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

3. Or run without profiling tools using basic timing:
   ```bash
   bin/runner --kernel vector_add --warmup 10 --reps 100
   ```
