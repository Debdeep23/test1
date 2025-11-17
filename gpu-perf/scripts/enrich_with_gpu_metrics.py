#!/usr/bin/env python3
import sys, csv, re

RUNS_IN, PROPS_OUT, STREAM_OUT, GEMM_OUT, RUNS_OUT = sys.argv[1:6]

def kv_from_file(path):
    kv = {}
    pat = re.compile(r'^([A-Za-z0-9_]+)=(.*)$')
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            m = pat.match(ln)
            if m: kv[m.group(1)] = m.group(2)
    return kv

def read_props(path):
    kv = kv_from_file(path)
    out = {}
    out["device_name"] = kv.get("name","")
    out["cc_major"] = int(re.sub(r'[^0-9]','', kv.get("major","0")) or 0)
    out["cc_minor"] = int(re.sub(r'[^0-9]','', kv.get("minor","0")) or 0)
    ints = ["multiProcessorCount","maxThreadsPerMultiProcessor","maxBlocksPerMultiProcessor",
            "regsPerMultiprocessor","sharedMemPerMultiprocessor","sharedMemPerBlockOptin",
            "maxThreadsPerBlock","warpSize","l2CacheSizeBytes"]
    for k in ints:
        out[k] = int(re.sub(r'[^0-9\-]','', kv.get(k,"0")) or 0)
    return out

def val(path, key):
    with open(path) as f:
        for ln in f:
            if ln.startswith(key+"="):
                return float(ln.split("=",1)[1])
    return 0.0

P  = read_props(PROPS_OUT)
BW = val(STREAM_OUT, "SUSTAINED_MEM_BW_GBPS")
FL = val(GEMM_OUT,   "SUSTAINED_COMPUTE_GFLOPS")

rd = csv.DictReader(open(RUNS_IN))
fields = rd.fieldnames + [
    "gpu_device_name","gpu_cc_major","gpu_cc_minor","gpu_sms",
    "gpu_max_threads_per_sm","gpu_max_blocks_per_sm","gpu_regs_per_sm",
    "gpu_shared_mem_per_sm","gpu_l2_bytes","gpu_warp_size",
    "calibrated_mem_bandwidth_gbps","calibrated_compute_gflops",
    "achieved_bandwidth_gbps","achieved_compute_gflops"
]
w = csv.DictWriter(open(RUNS_OUT,"w",newline=""), fieldnames=fields)
w.writeheader()

for r in rd:
    r.update({
        "gpu_device_name": P["device_name"],
        "gpu_cc_major": P["cc_major"],
        "gpu_cc_minor": P["cc_minor"],
        "gpu_sms": P["multiProcessorCount"],
        "gpu_max_threads_per_sm": P["maxThreadsPerMultiProcessor"],
        "gpu_max_blocks_per_sm": P["maxBlocksPerMultiProcessor"],
        "gpu_regs_per_sm": P["regsPerMultiprocessor"],
        "gpu_shared_mem_per_sm": P["sharedMemPerMultiprocessor"],
        "gpu_l2_bytes": P["l2CacheSizeBytes"],
        "gpu_warp_size": P["warpSize"],
        "calibrated_mem_bandwidth_gbps": f"{BW:.2f}",
        "calibrated_compute_gflops":     f"{FL:.2f}",
    })
    ms = float(r.get("mean_ms") or r.get("time_ms") or 0.0)
    fl = float(r.get("FLOPs") or 0.0)
    by = float(r.get("BYTES") or 0.0)
    if ms > 0:
        r["achieved_compute_gflops"] = f"{(fl/(ms/1000.0)/1e9):.2f}"
        r["achieved_bandwidth_gbps"] = f"{(by/(ms/1000.0)/1e9):.2f}"
    else:
        r["achieved_compute_gflops"] = ""
        r["achieved_bandwidth_gbps"] = ""
    w.writerow(r)

