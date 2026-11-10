"""Probe/extractor run on the PC: dataVals + expt of one participant (-> one small CSV of per-trial values).
usage: extract.py <exp 1|2> <label>   reads raw_extra/Krakauer2024_JSLHR/<label>_dataVals.mat and <label>_expt.mat,
writes raw_extra/Krakauer2024_JSLHR/trials_<label>.csv"""
import sys, csv
import numpy as np, scipy.io as sio
D = "raw_extra/Krakauer2024_JSLHR/"
exp, label = sys.argv[1], sys.argv[2]

def mel(h): return 2595 * np.log10(1 + h / 700.0)
def S(x):
    x = np.asarray(x)
    return str(x.ravel()[0]) if x.size else ""
def scal(x):
    x = np.asarray(x, float).ravel()
    return float(x[0]) if x.size else float("nan")
def nm(a):
    a = np.asarray(a, float)
    return float(np.mean(a)) if a.size and np.isfinite(a).any() and np.isfinite(a).sum() else float("nan")
def nmn(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")

dv = sio.loadmat(D + label + "_dataVals.mat", squeeze_me=False)["dataVals"][0]
ex = sio.loadmat(D + label + "_expt.mat", squeeze_me=True, struct_as_record=False)["expt"]
n = len(dv)
listWords = [str(x) for x in np.atleast_1d(ex.listWords)]
shiftMags = np.atleast_1d(ex.shiftMags).astype(float)
shiftAng = np.atleast_1d(ex.shiftAngles).astype(float)
if exp == "1":
    listConds = [str(x) for x in np.atleast_1d(ex.listConds)]
    listDots = [str(x) for x in np.atleast_1d(ex.listDots)]
    dots_of = lambda i: listDots[i]
    block_of = lambda i: str(i // int(ex.ntrials_per_block) + 1)
else:
    listConds = [str(x) for x in np.atleast_1d(ex.listConds)]
    dots_of = lambda i: "dots" if int(ex.bDots) == 1 else "noDots"
    block_of = lambda i: listConds[i]
gender = S(getattr(ex, 'gender', ''))
coh = scal(getattr(ex, 'coherence', np.nan)) if exp == '1' else float('nan')
rows = []
for i in range(n):
    r = dv[i]
    tok = int(scal(r["token"]))
    ft = np.asarray(r["ftrack_taxis"], float).ravel()
    f1 = np.asarray(r["f1"], float).ravel()
    f2 = np.asarray(r["f2"], float).ravel()
    row = dict(label=label, exp=exp, gender=gender, coherence=coh, ntrials=int(ex.ntrials), idx=i + 1, token=tok, excl=int(scal(r["bExcl"])), word=listWords[tok - 1],
               cond=listConds[tok - 1], dots=dots_of(tok - 1), block=block_of(tok - 1), dur_s=scal(r["dur"]),
               shift_mag=shiftMags[tok - 1], shift_ang=shiftAng[tok - 1],
               n_samp=len(ft), tstep=float(np.median(np.diff(ft))) if len(ft) > 1 else float("nan"))
    row["shift_f1"] = float(np.cos(shiftAng[tok - 1]) * shiftMags[tok - 1])
    row["shift_f2"] = float(np.sin(shiftAng[tok - 1]) * shiftMags[tok - 1])
    if len(ft) > 2:
        tt = ft - ft[0]
        T = tt[-1]
        win = {"first50": (0, 0.05), "mid": (0.25 * T, 0.75 * T + 1e-9), "w200_300": (0.2, 0.3), "w300_400": (0.3, 0.4)}
        for k, (a, b) in win.items():
            m = (tt >= a) & (tt < b) if k != "first50" else (tt < 0.05)
            row[f"f1_hz_{k}"] = nmn(f1[m]); row[f"f2_hz_{k}"] = nmn(f2[m])
            row[f"f1_mel_{k}"] = nmn(mel(f1[m])); row[f"f2_mel_{k}"] = nmn(mel(f2[m]))
        row["track_s"] = float(T)
        pt = np.asarray(r["pitch_taxis"], float).ravel(); f0 = np.asarray(r["f0"], float).ravel()
        it = np.asarray(r["int"], float).ravel(); at = np.asarray(r["ampl_taxis"], float).ravel()
        a, b = ft[0] + 0.25 * T, ft[0] + 0.75 * T
        mf = (pt >= a) & (pt <= b) & (f0 > 0)
        row["f0_hz_mid"] = nmn(f0[mf]) if len(pt) == len(f0) else float("nan")
        mi = (at >= a) & (at <= b)
        row["int_mid"] = nmn(it[mi]) if len(at) == len(it) else float("nan")
    rows.append(row)
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(D + f"trials_{label}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
meta = dict(label=label, exp=exp, n=n, gender=S(getattr(ex, "gender", "")), coherence=scal(getattr(ex, "coherence", np.nan)) if exp == "1" else float("nan"),
            ntrials=int(ex.ntrials), bDots=int(getattr(ex, "bDots", -1)) if exp == "2" else -1, sessionNum=int(getattr(ex, "sessionNum", 0)) if exp == "2" else 0)
print(meta, "excl", sum(r["excl"] for r in rows), "nonempty", sum(1 for r in rows if "f1_hz_mid" in r))
print("angles/mags sample", sorted(set((round(r["shift_f1"], 1), round(r["shift_f2"], 1), r["cond"]) for r in rows))[:8])
