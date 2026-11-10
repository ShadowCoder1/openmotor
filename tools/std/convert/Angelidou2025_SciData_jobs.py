"""Angelidou2025_SciData: resumable batch job (run on the PC with pc.py py). For each participant file SUBxx.mat
(Figshare 27180288, 2.7-3.4 GB, MATLAB v7.3 = HDF5): download it with parallel range requests, read it lazily with h5py,
keep per-gait-cycle event times and the time-normalised mean joint-angle and vertical-force curves per condition, write
a small json to tmp_samples/Angelidou2025_SciData/, and delete the raw file. One file on disk at a time."""
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

DS = "Angelidou2025_SciData"
ROOT = Path.cwd()
RAW = ROOT / "raw_extra" / DS
TMP = ROOT / "tmp_samples" / DS
IDS = {1: 51346451, 2: 51346475, 3: 51346481, 4: 51346565, 5: 51346631, 6: 51346634, 7: 51346646, 8: 51346652,
       9: 51346700, 10: 51346703, 11: 51346724, 16: 51346727, 12: 51347084, 17: 51347087, 13: 51347456, 18: 51347459,
       14: 51347558, 19: 51347561, 15: 51347627, 20: 51347651}
SPEEDS = [("speed0p6", 0.6), ("speed0p8", 0.8), ("speed1p0", 1.0)]
STIFFS = [("stiff1000", 1000), ("stiff25", 25), ("stiff80", 80), ("stiff40", 40)]  # protocol order (README)
JOINTS = ["LHip", "LKnee", "LAnkle", "RHip", "RKnee", "RAnkle"]
NPT = 100


def download(url, dest, size, nthreads=8, chunk=32 << 20):
    """Parallel range download into a preallocated file."""
    with open(dest, "wb") as fh:
        fh.truncate(size)
    parts = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    lock = threading.Lock()
    err = []

    def worker():
        while True:
            with lock:
                if not parts or err:
                    return
                a, b = parts.pop(0)
            for attempt in range(6):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    assert len(data) == b - a + 1, (len(data), a, b)
                    with open(dest, "r+b") as fh:
                        fh.seek(a)
                        fh.write(data)
                    break
                except Exception as e:  # retry
                    last = e
                    time.sleep(3 * (attempt + 1))
            else:
                err.append(last)
                return

    ts = [threading.Thread(target=worker) for _ in range(nthreads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if err:
        raise RuntimeError(f"download failed: {err[0]}")


def curve(arr, n=NPT):
    arr = np.asarray(arr, float)
    x = np.arange(arr.shape[-1])
    xn = np.linspace(0, arr.shape[-1] - 1, n)
    return np.interp(xn, x, arr)


def reduce_file(path, sub):
    f = h5py.File(path, "r")
    S = f[f"SUB{sub:02d}"]
    out = {"sub": sub, "speeds": {}}
    for sp, spv in SPEEDS:
        g = S[sp]
        ge = g["gaitEvents"]
        L = ge["leftHeelStrikes"][0]
        LT = ge["leftToeOffs"][0]
        R = ge["rightHeelStrikes"][0]
        RT = ge["rightToeOffs"][0]
        rec = {"lhs": L.tolist(), "lto": LT.tolist(), "right_equals_left": bool(np.array_equal(L, R) and np.array_equal(LT, RT)),
               "stiff": {}}
        j0 = 0
        for st, stv in STIFFS:
            sg = g[st]
            n = sg["time"]["time"].shape[1]
            t0 = [float(f[sg["time"]["time"][0, i]][0, 0]) for i in (0, n - 1)]
            keep = range(150, n) if st == "stiff1000" else range(n)  # first 150 stiff1000 cycles = force-mat warm-up
            acc = {k: [] for k in JOINTS + ["GRF_L", "GRF_R"]}
            for i in keep:
                for jn in JOINTS:
                    a = f[sg["jointAngles"][jn][0, i]][0, :]
                    acc[jn].append(curve(a))
                gr = f[sg["GRF"]["GRF"][0, i]][:]
                acc["GRF_L"].append(curve(gr[0]))
                acc["GRF_R"].append(curve(gr[1]))
            rec["stiff"][st] = {"n": int(n), "j0": j0, "t_first": t0[0], "t_last": t0[1],
                                "mean": {k: np.nanmean(np.array(v), 0).round(5).tolist() for k, v in acc.items()},
                                "sd": {k: np.nanstd(np.array(v), 0).round(5).tolist() for k, v in acc.items()},
                                "n_curves": len(keep)}
            j0 += n
        rec["n_events"] = len(L)
        out["speeds"][sp] = rec
    f.close()
    return out


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 6000.0
    only = [int(x) for x in sys.argv[2:]]
    t_start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for sub in (only or sorted(IDS)):
        outp = TMP / f"SUB{sub:02d}.json"
        if outp.exists():
            continue
        if time.time() - t_start > budget:
            print("time budget used; rerun to continue")
            return
        path = RAW / f"SUB{sub:02d}.mat"
        free = shutil.disk_usage(ROOT).free / 1e9
        url = f"https://ndownloader.figshare.com/files/{IDS[sub]}"
        if not path.exists():
            if free < 9:
                print(f"free {free:.1f} GB < 9, stopping")
                return
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=120) as r:
                size = int(r.headers["Content-Length"])
            t = time.time()
            part = path.with_suffix(".part")
            download(url, part, size)  # every range request is a GET on the original URL (its redirect is signed for GET)
            os.replace(part, path)
            print(f"SUB{sub:02d} downloaded {size/1e9:.2f} GB in {time.time()-t:.0f}s", flush=True)
        t = time.time()
        res = reduce_file(path, sub)
        json.dump(res, open(outp, "w"))
        path.unlink()
        print(f"SUB{sub:02d} reduced in {time.time()-t:.0f}s, raw deleted", flush=True)


if __name__ == "__main__":
    main()
