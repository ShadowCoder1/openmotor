"""Read-only helper for Codol2023_eLife: reads one participant .mat file of the authors' deposit (pure data, scipy only) and
writes the per-trial scalars (no EMG or kinematic traces) to raw_extra/Codol2023_eLife/scalars/<stem>.csv; then removes the
big .mat to save disk (the PC has little room for the 8.5 GB deposit). usage: Codol2023_eLife_extract.py <stem>"""
import os
import sys

import numpy as np
import scipy.io as sio

stem = sys.argv[1]
RAW = "raw_extra/Codol2023_eLife"
D = sio.loadmat(f"{RAW}/{stem}.mat", squeeze_me=True, struct_as_record=False)["D"]
n = len(D.trial)
nan = np.full(n, np.nan)


def get(name):
    return np.asarray(getattr(D, name), dtype=float) if hasattr(D, name) else nan


def targ(which, field):
    out = np.full(n, np.nan)
    for i, t in enumerate(D.targ):
        s = getattr(t, which, None)
        if s is not None and hasattr(s, field):
            out[i] = float(getattr(s, field))
    return out


cols = {"trial": get("trial"), "block": get("b"), "control": get("control"), "displ": get("displ"), "reward": get("reward"),
        "tau": get("tau"), "n_targets": get("n_targets"), "MT": get("MT"), "pt_field": get("pt"),
        "start_x": targ("start", "x"), "start_y": targ("start", "y"),
        "end_x": targ("end", "x"), "end_y": targ("end", "y"), "end_angle": targ("end", "angle"),
        "jump_x": targ("jump", "x"), "jump_y": targ("jump", "y"), "jump_angle": targ("jump", "angle"),
        "end_vis_rad": targ("end", "vis_rad")}
perf = sorted({str(p) for p in D.performance_measure})
os.makedirs(f"{RAW}/scalars", exist_ok=True)
with open(f"{RAW}/scalars/{stem}.csv", "w") as fh:
    fh.write(",".join(cols) + ",performance_measure\n")
    for i in range(n):
        fh.write(",".join("" if np.isnan(v[i]) else repr(float(v[i])) for v in cols.values()) + f",{str(D.performance_measure[i])}\n")
print(stem, n, perf)
os.remove(f"{RAW}/{stem}.mat")
