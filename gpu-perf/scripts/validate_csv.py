#!/usr/bin/env python3
import sys, csv

path = sys.argv[1]
bad = 0

def I(x):
    try: return int(float(x))
    except: return 0

def F(x):
    try: return float(x)
    except: return 0.0

with open(path) as f:
    rd = csv.DictReader(f)
    fn = set(rd.fieldnames or [])
    # minimal expectations for trials files
    req = {"kernel","rows","cols","block_x","block_y","block_z","grid_x","grid_y","grid_z","time_ms"}
    miss = sorted(list(req - fn))
    if miss:
        print(f"{path}: missing fields: {', '.join(miss)}")
        bad += 1

    for i, r in enumerate(rd, start=2):
        k  = r.get("kernel","")
        bx,by,bz = I(r.get("block_x")),I(r.get("block_y")),I(r.get("block_z"))
        gx,gy,gz = I(r.get("grid_x")), I(r.get("grid_y")), I(r.get("grid_z"))
        flat = bx*by*bz
        if flat <= 0:
            print(f"[{path}:{i}] {k}: block mismatch flat={flat} vs >0")
            bad += 1
        if gx<=0 or gy<=0 or gz<=0:
            print(f"[{path}:{i}] {k}: suspicious grid ({gx},{gy},{gz})")
            bad += 1
        if F(r.get("time_ms",0.0)) <= 0:
            print(f"[{path}:{i}] {k}: non-positive time_ms")
            bad += 1

if bad==0:
    print(f"[OK] {path} passed basic validation")
else:
    print(f"[FAIL] {path} had {bad} issues")

