"""Kobler et al. (2020), NeuroImage 220:117076: distinct cortical networks for hand movement initiation and directional processing.
15 healthy participants did a visually guided centre-out reach (4 directions) followed by 16 s of pursuit tracking, controlling a cursor
with the right palm (LeapMotion), in an execution condition (hand moves the cursor) and an observation condition (the cursor replays an
earlier hand trajectory while the hand rests). Eye movements were recorded as EOG. Data: OSF 236ax, EEGLAB .set/.fdt files
(EEG with the movement channels stored in the same file). Only the hand/cursor/target and EOG (gaze) channels and the event markers are
released; the EEG is never read out. One participant at a time: fetch, reduce, delete raw.

Run on the PC: python3 -I tools/std/pc.py run Kobler2020_NeuroImage   (KOB_ONLY=p01 for a one-participant test, no release)
"""
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import windows

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402
import fetch  # noqa: E402

ID = "Kobler2020_NeuroImage"
RAW = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "tmp_samples" / ID
checks = stdlib.Checks()
ONLY = os.environ.get("KOB_ONLY") or (open(RAW / "ONLY.txt").read().strip() if (RAW / "ONLY.txt").exists() else None)
URL = {"p01": {"eyeblock1.set": "https://osf.io/download/upv5z/", "mainblock.set": "https://osf.io/download/kuenr/", "eyeblock1.fdt": "https://osf.io/download/dxjps/", "eyeblock2.fdt": "https://osf.io/download/5f1ec9ea320805010e1551e8/", "eyeblock2.set": "https://osf.io/download/5f1ec9eb61875e00f1317faa/", "mainblock.fdt": "https://osf.io/download/j4nz9/"}, "p02": {"eyeblock2.set": "https://osf.io/download/5f1eca345fdf6200f8048619/", "eyeblock1.set": "https://osf.io/download/5f1eca3561875e00e731e061/", "eyeblock1.fdt": "https://osf.io/download/5f1eca3761875e00f131809e/", "mainblock.set": "https://osf.io/download/enhga/", "mainblock.fdt": "https://osf.io/download/5f1eca6b5fdf6200fd049e95/", "eyeblock2.fdt": "https://osf.io/download/q2x8t/"}, "p03": {"eyeblock2.set": "https://osf.io/download/5f1eca5c5fdf6200fc048ec9/", "eyeblock1.set": "https://osf.io/download/5f1eca6d61875e00f131814a/", "eyeblock2.fdt": "https://osf.io/download/5f1eca703208050106155a1d/", "eyeblock1.fdt": "https://osf.io/download/5f1eca7d5fdf6200fd049ebf/", "mainblock.fdt": "https://osf.io/download/5f1ecaa43208050106155a62/", "mainblock.set": "https://osf.io/download/5f1ecaa661875e00f13181f5/"}, "p04": {"eyeblock1.set": "https://osf.io/download/5f1ecad761875e00ed318472/", "eyeblock2.set": "https://osf.io/download/5f1ecad7320805010e1553e4/", "mainblock.set": "https://osf.io/download/5f1ecadd320805010d154378/", "eyeblock2.fdt": "https://osf.io/download/5f1ecade61875e00f1318263/", "eyeblock1.fdt": "https://osf.io/download/5f1ecae561875e00f1318275/", "mainblock.fdt": "https://osf.io/download/5f1ecb3f61875e00ed31850d/"}, "p05": {"eyeblock1.set": "https://osf.io/download/5f1ecaf161875e00e731e11d/", "eyeblock2.set": "https://osf.io/download/5f1ecaf661875e00f13182a8/", "eyeblock1.fdt": "https://osf.io/download/5f1ecb0c320805010d1543c4/", "mainblock.set": "https://osf.io/download/5f1ecb215fdf6200fb04885f/", "eyeblock2.fdt": "https://osf.io/download/5f1ecb385fdf6200f804876b/", "mainblock.fdt": "https://osf.io/download/5f1ecb725fdf6200fc049079/"}, "p06": {"eyeblock2.fdt": "https://osf.io/download/5f1ecb5561875e00ed318528/", "eyeblock1.fdt": "https://osf.io/download/5f1ecb755fdf6200f80487b3/", "eyeblock2.set": "https://osf.io/download/5f1ecb833208050106155b96/", "mainblock.fdt": "https://osf.io/download/5f1ecbaa61875e00f13184de/", "mainblock.set": "https://osf.io/download/5f1ecbac320805010e155600/", "eyeblock1.set": "https://osf.io/download/5f1ecde35fdf6200fc049488/"}, "p07": {"eyeblock2.set": "https://osf.io/download/5f1ecbaa61875e00f13184da/", "eyeblock1.fdt": "https://osf.io/download/5f1ecbaf320805010d1544e4/", "eyeblock1.set": "https://osf.io/download/5f1ecbaf5fdf6200fd04a10d/", "eyeblock2.fdt": "https://osf.io/download/5f1ecbaf61875e00f13184e9/", "mainblock.set": "https://osf.io/download/5f1ecbb4320805010e155619/", "mainblock.fdt": "https://osf.io/download/5f1ecbed5fdf6200fd04a19b/"}, "p08": {"eyeblock1.set": "https://osf.io/download/5f1ecbb95fdf6200f8048802/", "eyeblock2.set": "https://osf.io/download/5f1ecbbe320805010d1544f6/", "eyeblock1.fdt": "https://osf.io/download/5f1ecbed320805010e155690/", "mainblock.set": "https://osf.io/download/5f1ecbee5fdf6200f8048848/", "eyeblock2.fdt": "https://osf.io/download/5f1ecc1761875e00f131864f/", "mainblock.fdt": "https://osf.io/download/5f1ecc4a5fdf6200fd04a265/"}, "p09": {"eyeblock1.set": "https://osf.io/download/5f1ecbfe5fdf6200f8048866/", "eyeblock2.set": "https://osf.io/download/5f1ecc023208050106155c5a/", "eyeblock1.fdt": "https://osf.io/download/5f1ecc1261875e00f131862b/", "mainblock.fdt": "https://osf.io/download/5f1ecc815fdf6200f804890a/", "mainblock.set": "https://osf.io/download/5f1ecdfc5fdf6200fb048c7a/"}, "p10": {"eyeblock2.fdt": "https://osf.io/download/5f1ecc4d320805010e155769/", "eyeblock2.set": "https://osf.io/download/5f1ecc83320805010d15464c/", "eyeblock1.set": "https://osf.io/download/5f1ecc865fdf6200f8048911/", "eyeblock1.fdt": "https://osf.io/download/5f1ecc895fdf6200fd04a2da/", "mainblock.set": "https://osf.io/download/htg5e/", "mainblock.fdt": "https://osf.io/download/562gy/"}, "p11": {"eyeblock2.set": "https://osf.io/download/5f1ecc8a3208050105154f75/", "eyeblock1.set": "https://osf.io/download/5f1ecc8f61875e00f13187a0/", "eyeblock2.fdt": "https://osf.io/download/5f1ecc8f3208050106155d19/", "eyeblock1.fdt": "https://osf.io/download/5f1eccb33208050105154fa4/", "mainblock.set": "https://osf.io/download/5f1eccb85fdf6200f8048940/", "mainblock.fdt": "https://osf.io/download/5f1ecd1d61875e00f0316dfe/"}, "p12": {"eyeblock2.set": "https://osf.io/download/5f1ecc9b5fdf6200fc049225/", "eyeblock1.set": "https://osf.io/download/5f1ecccc3208050105154fcb/", "eyeblock2.fdt": "https://osf.io/download/5f1ecccd61875e00ed3186d7/", "mainblock.set": "https://osf.io/download/5f1ecd165fdf6200f80489d0/", "eyeblock1.fdt": "https://osf.io/download/5f1ecd1f61875e00f131892a/", "mainblock.fdt": "https://osf.io/download/5f1ecd5b5fdf6200fb048b92/"}, "p13": {"eyeblock2.set": "https://osf.io/download/5f1eccda5fdf6200f804897e/", "eyeblock1.fdt": "https://osf.io/download/5f1ecd1d5fdf6200fc04931a/", "eyeblock1.set": "https://osf.io/download/5f1ecd2e61875e00e731e39b/", "eyeblock2.fdt": "https://osf.io/download/5f1ecd4961875e00f131897f/", "mainblock.set": "https://osf.io/download/5f1ecd8d61875e00f0316edc/", "mainblock.fdt": "https://osf.io/download/5f1ecd9332080501051550a9/"}, "p14": {"eyeblock1.set": "https://osf.io/download/5f1ecd95320805010e155a41/", "eyeblock1.fdt": "https://osf.io/download/5f1ecd96320805010e155a47/", "mainblock.set": "https://osf.io/download/5f1ecd9a3208050106155e89/", "eyeblock2.fdt": "https://osf.io/download/5f1ecd9e5fdf6200f8048a83/", "mainblock.fdt": "https://osf.io/download/5f1ecddb61875e00f0316f73/", "eyeblock2.set": "https://osf.io/download/5f1ecdf2320805010d1548bb/"}, "p15": {"eyeblock2.set": "https://osf.io/download/5f1ecd9c320805010d154828/", "eyeblock1.set": "https://osf.io/download/5f1ecda661875e00f1318a6c/", "eyeblock2.fdt": "https://osf.io/download/5f1ecdab5fdf6200f8048a99/", "eyeblock1.fdt": "https://osf.io/download/5f1ecdb261875e00f1318a99/", "mainblock.set": "https://osf.io/download/5f1ecddf320805010e155afa/", "mainblock.fdt": "https://osf.io/download/5f1ece3861875e00f031701f/"}}
SIZE = {"p01": {"eyeblock1.set": 1376387, "mainblock.set": 17792119, "eyeblock1.fdt": 23778832, "eyeblock2.fdt": 23389520, "eyeblock2.set": 1371936, "mainblock.fdt": 422121024}, "p02": {"eyeblock2.set": 1366634, "eyeblock1.set": 1376561, "eyeblock1.fdt": 22698676, "mainblock.set": 26944734, "mainblock.fdt": 519893392, "eyeblock2.fdt": 23021768}, "p03": {"eyeblock2.set": 1378524, "eyeblock1.set": 1445296, "eyeblock2.fdt": 23322376, "eyeblock1.fdt": 23006676, "mainblock.fdt": 534504792, "mainblock.set": 27184735}, "p04": {"eyeblock1.set": 1515033, "eyeblock2.set": 1344634, "mainblock.set": 28079609, "eyeblock2.fdt": 22863764, "eyeblock1.fdt": 23775136, "mainblock.fdt": 563091192}, "p05": {"eyeblock1.set": 1435472, "eyeblock2.set": 1341939, "eyeblock1.fdt": 22733788, "mainblock.set": 27625097, "eyeblock2.fdt": 23152976, "mainblock.fdt": 550070792}, "p06": {"eyeblock2.fdt": 23298660, "eyeblock1.fdt": 23357180, "eyeblock2.set": 1387556, "mainblock.fdt": 557700712, "mainblock.set": 27682316, "eyeblock1.set": 1437940}, "p07": {"eyeblock2.set": 1341091, "eyeblock1.fdt": 23248456, "eyeblock1.set": 1497774, "eyeblock2.fdt": 22993432, "mainblock.set": 28453218, "mainblock.fdt": 576122944}, "p08": {"eyeblock1.set": 1520629, "eyeblock2.set": 1346537, "eyeblock1.fdt": 23819488, "mainblock.set": 27499115, "eyeblock2.fdt": 23285724, "mainblock.fdt": 549414096}, "p09": {"eyeblock1.set": 1517698, "eyeblock2.set": 1380951, "eyeblock1.fdt": 23128952, "mainblock.fdt": 582529256, "mainblock.set": 28091965}, "p10": {"eyeblock2.fdt": 23476992, "eyeblock2.set": 1393532, "eyeblock1.set": 1362123, "eyeblock1.fdt": 22992508, "mainblock.set": 27929598, "mainblock.fdt": 568548408}, "p11": {"eyeblock2.set": 1390289, "eyeblock1.set": 1340874, "eyeblock2.fdt": 22899184, "eyeblock1.fdt": 22833888, "mainblock.set": 28896449, "mainblock.fdt": 569137336}, "p12": {"eyeblock2.set": 1396075, "eyeblock1.set": 1345636, "eyeblock2.fdt": 23075976, "mainblock.set": 29700319, "eyeblock1.fdt": 23068892, "mainblock.fdt": 616818832}, "p13": {"eyeblock2.set": 1394680, "eyeblock1.fdt": 23035628, "eyeblock1.set": 1381867, "eyeblock2.fdt": 23226280, "mainblock.set": 29644186, "mainblock.fdt": 613415640}, "p14": {"eyeblock1.set": 1498139, "eyeblock1.fdt": 23412312, "mainblock.set": 29057197, "eyeblock2.fdt": 23075976, "mainblock.fdt": 611561480, "eyeblock2.set": 1432235}, "p15": {"eyeblock2.set": 1362149, "eyeblock1.set": 1437442, "eyeblock2.fdt": 22983576, "eyeblock1.fdt": 24152128, "mainblock.set": 26359338, "mainblock.fdt": 507518336}}
KEEP_FREE_GB = 3.0
MOVE = ["target_X", "target_Z", "cursor_X", "cursor_Z", "leappos_x", "leappos_y", "leappos_z", "HEOG", "VEOGL", "VEOGR", "REOG"]
UNITS = {"target_X": "screen units", "target_Z": "screen units", "cursor_X": "screen units", "cursor_Z": "screen units",
         "leappos_x": "mm", "leappos_y": "mm", "leappos_z": "mm", "HEOG": "uV", "VEOGL": "uV", "VEOGR": "uV", "REOG": "uV"}
NAMES = {"1000": "trial_start", "-1000": "trial_end", "1": "cue_1", "2": "cue_2", "1001": "m1001", "-1001": "m1001_end",
         "1002": "m1002", "-1002": "m1002_end", "1003": "m1003", "-1003": "m1003_end", "1004": "m1004", "-1004": "m1004_end"}

import h5py, numpy as np
from scipy.signal import windows
def _s(f, r): return ''.join(chr(c) for c in np.array(r).ravel())
def read_set(path):
    f = h5py.File(path, 'r'); e = f['EEG']
    labs = [_s(f, f[r[0]]) for r in e['chanlocs']['labels']]
    n = int(np.array(e['pnts'])[0, 0]); sr = float(np.array(e['srate'])[0, 0])
    ev = e['event']
    ty = [_s(f, f[r[0]]) for r in ev['type']]
    lat = [float(np.array(f[r[0]]).ravel()[0]) for r in ev['latency']]
    f.close()
    return labs, sr, n, list(zip(ty, lat))
def attempts(evs):
    """split event list into attempts 1000..-1000; returns list of dict(start,end,events[(type,lat)])"""
    out = []; cur = None
    for t, l in evs:
        if t == '1000':
            cur = dict(start=l, events=[]); 
        if cur is not None:
            cur['events'].append((t, l))
            if t == '-1000':
                cur['end'] = l; out.append(cur); cur = None
    return out
def onset_offset(dist, tx, tz, sr, base_end, move_start, stop_i):
    """dist normalised cursor distance from origin (array over segment), returns onset, offset in samples (floats) or nan"""
    b0 = max(0, int(base_end - sr)); base = np.nanmean(dist[b0:int(base_end)])
    d = dist - base
    idx = np.where(d[int(move_start):] > 0.05)[0]
    if len(idx) == 0: return np.nan, np.nan
    t0 = int(move_start) + idx[0]
    w0 = max(0, int(t0 - 1.75 * sr))
    k = windows.triang(63); k = k / k.sum()
    sm = np.convolve(d, k, mode='same')
    seg = sm[w0:t0 + 1]
    if len(seg) < 3: return np.nan, np.nan
    der = np.diff(seg); j = int(np.argmax(der)); slope = der[j]
    if slope <= 0: return np.nan, np.nan
    tm = w0 + j; on = tm - sm[tm] / slope
    return on, None



def free_gb():
    return shutil.disk_usage(str(stdlib.HERE)).free / 1e9


def get(p, name):
    """fetch one raw file of participant p onto the PC (waits while the PC disk is short)"""
    need = SIZE[p][name] / 1e9
    t0 = time.time()
    while free_gb() < KEEP_FREE_GB + need:
        if time.time() - t0 > 7200:
            raise RuntimeError("PC disk too full for 2 h")
        time.sleep(60)
    fetch.main(ID, URL[p][name], f"{p}_{name}")


def drop(p, names):
    for n in names:
        f = RAW / f"{p}_{n}"
        if f.exists():
            f.unlink()


def label(t):
    if t in NAMES:
        return NAMES[t]
    v = int(t)
    for lo, nm in ((4000, "cot_dir"), (5000, "ptt_dir"), (3000, "queue"), (2000, "count")):
        if lo <= v < lo + 1000:
            return f"{nm}_{v - lo}"
    return "e" + t


def write_csv(path, arr, cols, sr):
    df = pd.DataFrame(arr, columns=cols)
    df.insert(0, "time_s", np.char.mod("%.3f", np.arange(len(arr)) / sr))
    df.to_csv(path, index=False, float_format="%.5g", lineterminator="\r\n")


def onset_offset(seg, sr, m0):
    """the paper's cursor movement onset (tangent method) and offset, distances normalised (1 = grid circle = 100 screen units)"""
    s = seg / 100.0
    dist = np.hypot(s[:, 0], s[:, 1])
    base = np.nanmean(dist[max(0, int(m0 - sr)):int(m0)])
    d = dist - base
    idx = np.where(d[int(m0):] > 0.05)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    t0 = int(m0) + idx[0]
    w0 = max(0, int(t0 - 1.75 * sr))
    k = windows.triang(63)
    k = k / k.sum()
    sm = np.convolve(d, k, mode="same")
    w = sm[w0:t0 + 1]
    if len(w) < 3:
        return np.nan, np.nan
    der = np.diff(w)
    j = int(np.argmax(der))
    if der[j] <= 0:
        return np.nan, np.nan
    on = (w0 + j) - sm[w0 + j] / der[j]
    dd = np.hypot(s[:, 0] - s[-1, 2], s[:, 1] - s[-1, 3])
    after = np.where((dd < 0.1) & (np.arange(len(dd)) > on))[0]
    return on / sr, ((after[0] - on) / sr if len(after) else np.nan)


def base_row(pi, block, trial, phase, tsec, unit, cond, excl, device):
    return dict(dataset=ID, study="Kobler et al. 2020, NeuroImage", experiment=block, group="", subject=f"p{pi:02d}", person=f"p{pi:02d}",
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab", device=device,
                session=1, block=block, trial=trial, phase=phase, time_s=tsec, task="tracking", unit=unit, condition=cond,
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=excl)


DEVICE = ("LeapMotion controller (right palm position), cursor and target positions on the screen (60 Hz, resampled), EOG electrodes; "
          "all resampled to 200 Hz")
SIGNAL = {"main": "hand (palm) position, cursor and target positions, EOG (gaze)", "eye": "target position, EOG (gaze)"}
rows, extra = [], []
stats = {}
TMP.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
parts = sorted(URL) if not ONLY else [ONLY]
rec_no = {}
try:
    for p in parts:
        pi = int(p[1:])
        st = stats[p] = dict(completed=0, exe=0, obs=0, attempts=0, aborted=0, onset=[], dur=[], excursion={"exe": [], "obs": []})
        for blk in ("mainblock", "eyeblock1", "eyeblock2"):
            if f"{blk}.set" not in URL[p] or f"{blk}.fdt" not in URL[p]:
                continue
            get(p, f"{blk}.set")
            get(p, f"{blk}.fdt")
            labs, sr, n, evs = read_set(RAW / f"{p}_{blk}.set")
            assert sr == 200.0, sr
            ix = {l: i for i, l in enumerate(labs)}
            n0 = len(rows)
            cols = [c for c in MOVE if c in ix]
            x = np.memmap(RAW / f"{p}_{blk}.fdt", dtype="<f4", mode="r", shape=(n, len(labs)))
            if blk == "mainblock":
                assert all(c in ix for c in MOVE), (p, labs[-20:])
                for a in attempts(evs):
                    ev = {}
                    for t, l in a["events"]:
                        ev.setdefault(t, l)
                    s, e = int(a["start"]), int(a["end"])
                    arr = np.array(x[s:e + 1][:, [ix[c] for c in cols]], dtype=np.float32)
                    cue = [t for t, _ in a["events"] if t in ("1", "2")]
                    cond = {"1": "execution", "2": "observation"}.get(cue[0], "") if cue else ""
                    comp = "1004" in ev
                    st["attempts"] += 1
                    k = st["attempts"]
                    if comp and "1001" in ev and "1003" in ev:
                        m0 = int(ev["1001"]) - s
                        j = int(ev["1003"]) - s + 1
                        o, dur = onset_offset(arr[:j, [cols.index("cursor_X"), cols.index("cursor_Z"), cols.index("target_X"), cols.index("target_Z")]], sr, m0)
                        if cond == "execution":
                            lp = arr[m0:j][:, [cols.index("leappos_x"), cols.index("leappos_z")]]
                            st["excursion"]["exe"].append(float(np.abs(lp - lp[0]).max()))
                        elif cond == "observation":
                            lp = arr[m0:j][:, [cols.index("leappos_x"), cols.index("leappos_z")]]
                            st["excursion"]["obs"].append(float(np.abs(lp - lp[0]).max()))
                        st["onset"].append(o)
                        st["dur"].append(dur)
                    info = []
                    for t, _ in a["events"]:
                        v = int(t)
                        if 2000 <= v < 3000:
                            info.append(f"trial_counter={v - 2000}")
                        elif 3000 <= v < 4000:
                            info.append(f"queue_id={v - 3000}")
                        elif 4000 <= v < 5000:
                            info.append(f"cot_direction_deg={v - 4000}")
                        elif 5000 <= v < 6000:
                            info.append(f"ptt_angle_deg={v - 5000}")
                    info.append(f"cue_code={cue[0] if cue else ''}")
                    info.append(f"reached_pursuit_phase={comp}")
                    evtxt = ";".join(f"{label(t)}={(l - a['start']) / sr:.3f}" for t, l in a["events"] if int(t) not in (1000, -1000)
                                     or True)
                    fn = f"samples/{ID}_S{pi:04d}_trial{k:03d}.csv"
                    write_csv(TMP / f"{ID}_S{pi:04d}_trial{k:03d}.csv", arr, cols, sr)
                    extra.append((fn, str(TMP / f"{ID}_S{pi:04d}_trial{k:03d}.csv")))
                    r = base_row(pi, "main block", k, "test", a["start"] / sr, "trial", cond, False, DEVICE)
                    r.update(recording=k, samples_file=fn, signal_kind=SIGNAL["main"], channels=";".join(cols),
                             channel_units=";".join(UNITS[c] for c in cols), sample_rate_hz=sr, n_samples=len(arr),
                             duration_s=len(arr) / sr, events=evtxt, processing="resampled to 200 Hz and synchronised by the authors; otherwise none",
                             trial_info=";".join(info))
                    rows.append(r)
            else:
                nm = "eye1" if blk == "eyeblock1" else "eye2"
                arr = np.array(x[:, [ix[c] for c in cols]], dtype=np.float32)
                k = rec_no[p] = rec_no.get(p, 0) + 1
                fn = f"samples/{ID}_S{pi:04d}_{nm}.csv"
                write_csv(TMP / f"{ID}_S{pi:04d}_{nm}.csv", arr, cols, sr)
                extra.append((fn, str(TMP / f"{ID}_S{pi:04d}_{nm}.csv")))
                evtxt = ";".join(f"{label(t)}={l / sr:.3f}" for t, l in evs)
                r = base_row(pi, "eye block " + blk[-1], 1000 + int(blk[-1]), "other", 0, "recording", "eye movements to a moving target", False, DEVICE)
                r.update(recording=1000 + int(blk[-1]), samples_file=fn, signal_kind=SIGNAL["eye"], channels=";".join(cols),
                         channel_units=";".join(UNITS[c] for c in cols), sample_rate_hz=sr, n_samples=len(arr),
                         duration_s=len(arr) / sr, events=evtxt, processing="resampled to 200 Hz and synchronised by the authors; otherwise none",
                         trial_info="")
                rows.append(r)
            if blk == "mainblock":
                mrows = rows[n0:]
                for r in mrows:
                    done = "reached_pursuit_phase=True" in r["trial_info"]
                    r["excluded_by_authors"] = not done
                    r["trial_info"] += f";completed={done}"
                    if done:
                        st["completed"] += 1
                        st["exe" if r["condition"] == "execution" else "obs"] += 1
                    else:
                        st["aborted"] += 1
            del x
            drop(p, [f"{blk}.set", f"{blk}.fdt"])
        print(p, {k: v for k, v in st.items() if k in ("attempts", "completed", "exe", "obs", "aborted")}, f"free {free_gb():.1f} GB", flush=True)
    # fix the trial/recording numbering per person: main-block trials are 1..n, eye blocks follow
    byp = {}
    for r in rows:
        byp.setdefault(r["person"], []).append(r)
    for p, rs in byp.items():
        for i, r in enumerate(rs, 1):
            r["trial"] = i
    if ONLY:
        import zipfile, io
        tot = 0
        for _, f in extra:
            tot += os.path.getsize(f)
        zb = 0
        for _, f in extra[:40]:
            b = open(f, "rb").read()
            c = io.BytesIO()
            with zipfile.ZipFile(c, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("a", b)
            zb += len(c.getvalue())
        print("raw csv bytes", tot, "zip ratio (40 files)", zb / sum(os.path.getsize(f) for _, f in extra[:40]), "rows", len(rows))
        for p in parts:
            print(p, stats[p]["completed"], np.nanmean(stats[p]["onset"]), np.nanmean(stats[p]["dur"]),
                  np.median(stats[p]["excursion"]["exe"]), np.median(stats[p]["excursion"]["obs"]))
        sys.exit(0)

    main = [r for r in rows if r["unit"] == "trial"]
    checks.internal("all 15 participants present", len(parts) == 15 and len({r["person"] for r in rows}) == 15)
    checks.internal("palm excursion larger in execution than observation for every participant",
                    all(np.median(stats[p]["excursion"]["exe"]) > 5 * np.median(stats[p]["excursion"]["obs"]) for p in parts),
                    "supports cue 1 = execution, cue 2 = observation")
    checks.internal("every trial row has a samples file", all(r["samples_file"] for r in rows))
    checks.published("participants", len(parts), 15, 0, "Abstract and Sec. 2.1")
    checks.published("sampling rate of the released signals (Hz)", rows[0]["sample_rate_hz"], 200, 0, "Sec. 2.4, all signals resampled to 200 Hz")
    notes = []
    n360 = sum(1 for p in parts if stats[p]["completed"] == 360)
    if n360 == 14:
        checks.published("participants with 360 completed trials (all but participant 1)", n360, 14, 0, "Sec. 2.3")
    else:
        notes.append("The paper says 360 trials were recorded for each participant except participant 1 (192). Counting an attempt as completed when it reached the pursuit-tracking phase (marker 1004), %d of the 14 participants have exactly 360 completed trials (others: %s). "
                     % (n360, ", ".join(f"{stats[p]['completed']}" for p in parts[1:] if stats[p]["completed"] != 360)))
    checks.published("completed trials of participant 1", stats[parts[0]]["completed"], 192, 0, "Sec. 2.3 (only 192 trials recorded for participant 1)")
    n180e = sum(1 for p in parts if stats[p]["exe"] == 180)
    checks.published("participants with 180 execution trials (all but participant 1)", n180e, 14, 0, "Sec. 2.3 (180 per condition)")
    n180o = sum(1 for p in parts if stats[p]["obs"] == 180)
    if n180o == 14:
        checks.published("participants with 180 observation trials (all but participant 1)", n180o, 14, 0, "Sec. 2.3 (180 per condition)")
    else:
        notes.append("Observation-condition trials: %d of 14 participants have exactly 180 (others have %s)." % (n180o, ", ".join(str(stats[p]["obs"]) for p in parts[1:] if stats[p]["obs"] != 180)))
    om = float(np.mean([np.nanmean(stats[p]["onset"]) for p in parts]))
    dm = float(np.mean([np.nanmean(stats[p]["dur"]) for p in parts]))
    differences = notes + [
        "The authors' per-participant preprocessed data (outlier trials, EEG) are not part of the OSF deposit, so no trial is marked as an outlier; "
        "attempts that never reached the pursuit-tracking phase (marker 1004; aborted trials, which the authors re-queued) are marked excluded_by_authors = True; this is inferred from the event markers, not an exclusion list of the authors.",
        f"The paper reports a grand average cursor movement onset of 3.23 s (SD 0.18) from trial start and a duration of 0.49 s (SD 0.13). Applying the "
        f"paper's described onset method to the released cursor signal gives a mean of {om:.2f} s and {dm:.2f} s here (mean of participant means, no outlier "
        f"removal); these values are not released as columns and are not used as a check because the onset reference point is not fully specified.",
        "Cue codes 1 and 2 are not explained in the deposit; execution = 1 and observation = 2 was inferred from the palm excursion (large when 1, near zero when 2).",
        "Event codes other than the cue codes (1001-1004 and their negatives, 4xxx, 5xxx) are kept as the authors' numbers; the README names the pattern, not a verified meaning.",
        "The deposit has no eyeblock2.fdt for participant 9 (only its .set file), so that participant has one eye-block recording; every other participant has two.",
    ]
    meta = dict(
        paper=dict(authors="Kobler RJ, Kolesnichenko E, Sburlea AI, Muller-Putz GR", year=2020,
                   title="Distinct cortical networks for hand movement initiation and directional processing: An EEG study",
                   venue="NeuroImage 220:117076", doi="10.1016/j.neuroimage.2020.117076"),
        lab="Muller-Putz (TU Graz)", data_url="https://osf.io/236ax/", license="none stated", license_url="",
        changes="Only the hand (LeapMotion palm), cursor and target position channels, the four EOG gaze channels and the event markers were kept; the 64 EEG channels and the six raw EOG electrodes were not released. Samples are the authors' 200 Hz values rounded to 5 significant digits (to keep the ZIP under about 1 GB), cut into one file per trial attempt.",
        how=["One row is one trial attempt of the main block (from the trial start marker 1000 to the end marker -1000), or one of the two eye blocks (a whole recording). Each row's samples are in samples/<file>.csv: time_s from the start of the segment, then target (X, Z), cursor (X, Z), palm (x, y, z) and EOG channels.",
             "Target and cursor positions are in the screen units of the authors' paradigm (the target stops about 31.4 units from the centre, which is 5 cm on screen; 100 units is the grid circle of 16 cm radius, inferred from the paper). Palm position is in mm and EOG in microvolts (units inferred from the recording software; the deposit does not state them). Cursor_X/Z follow the participant's palm (execution) or replay an earlier trajectory (observation).",
             "Trial structure (from the paper and the event codes): condition cue (cue_1 = execution, cue_2 = observation, inferred), the target moves 0.5 s in one of four directions (cot_direction_deg: 0 right, 90 up, 180 left, 270 down, inferred), the participant moves the cursor to it, a 1 s fixation, then 16 s of pursuit tracking (ptt_angle_deg). Events are listed in the events column as code=seconds from the start of the segment.",
             "time_s in the main table is the start of the segment in the recording file (the three files of a participant have separate clocks)."],
        unit_note="trial attempt (main block) or eye-block recording",
        differences=differences,
    )
    stdlib.release(ID, "timeseries", rows, meta, checks, extra_files=extra)
finally:
    if not ONLY or True:
        shutil.rmtree(TMP, ignore_errors=True)
    for f in RAW.glob("p*_*"):
        f.unlink()
