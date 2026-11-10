"""Ofner et al. (2017), PLoS ONE 12(8):e0182578: upper limb movements (elbow flexion/extension, forearm supination/pronation, hand
close/open, rest) executed by 15 healthy people while 61-channel EEG was recorded (BNCI Horizon 2020 data set 001-2017, files
S01_ME..S15_ME, 10 runs of 42 trials each). The GDF files hold EEG, 3 EOG and 32 movement channels (exoskeleton joint angles, hand and
elbow positions, 5DT data glove sensors) at 64 Hz. Only the movement channels and the event markers are released; EEG and EOG are never
decoded. The GDF 2 files are read with the minimal reader below (no deposit code is run). One participant at a time: fetch, reduce, delete.

Run on the PC: python3 -I tools/std/pc.py run Ofner2017_PLoSOne   (raw_extra/<id>/ONLY.txt containing a subject number: test, no release)
"""
import os
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402
import fetch  # noqa: E402

ID = "Ofner2017_PLoSOne"
RAW = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "tmp_samples" / ID
checks = stdlib.Checks()
ONLY = (open(RAW / "ONLY.txt").read().strip() if (RAW / "ONLY.txt").exists() else None)
KEEP_FREE_GB = 3.0
URL = "https://lampx.tugraz.at/~bci/database/001-2017/S%02d_ME.zip"
MOVE = ["thumb_near", "thumb_far", "thumb_index", "index_near", "index_far", "index_middle", "middle_near", "middle_far", "middle_ring",
        "ring_near", "ring_far", "ring_little", "litte_near", "litte_far", "thumb_palm", "wrist_bend", "roll", "pitch", "gesture",
        "handPosX", "handPosY", "handPosZ", "elbowPosX", "elbowPosY", "elbowPosZ", "ShoulderAdductio", "ShoulderFlexionE",
        "ShoulderRotation", "Elbow", "ProSupination", "Wrist", "GripPressure"]
CLASSES = {1536: "elbow flexion", 1537: "elbow extension", 1538: "forearm supination", 1539: "forearm pronation", 1540: "hand close",
           1541: "hand open", 1542: "rest"}
EVN = {768: "trial_start", 785: "beep", 786: "cross_on", 33536: "trial_end", 33554: "cross_off"}
for c, n in CLASSES.items():
    EVN[c] = "cue_" + n.replace(" ", "_")
    EVN[c + 32768] = "cue_end_" + n.replace(" ", "_")

import struct, numpy as np
GT = {1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<i8',8:'<u8',16:'<f4',17:'<f8'}
def read_gdf(buf, want=None):
    """minimal GDF 2.x reader. returns dict(labels, fs, data{label: array}, events (pos,typ,dur,chn), header info). want: set of labels to decode"""
    assert buf[:3] == b'GDF', buf[:8]
    ver = float(buf[3:8].decode().strip().replace('GDF','') or 0) if False else buf[4:8].decode()
    hl = struct.unpack('<H', buf[184:186])[0]; nrec = struct.unpack('<q', buf[236:244])[0]
    num, den = struct.unpack('<II', buf[244:252]); ns = struct.unpack('<H', buf[252:254])[0]
    dur = num / den
    o = 256
    def arr(sz, dt=None, n=ns):
        nonlocal o
        b = buf[o:o + sz * n]; o += sz * n
        return b
    labels = [arr(16)[i*16:(i+1)*16].decode('latin1').strip('\x00 ') if False else None for i in range(0)]
    b = arr(16); labels = [b[i*16:(i+1)*16].decode('latin1').strip('\x00 ') for i in range(ns)]
    arr(80)
    b = arr(6); pdim = [b[i*6:(i+1)*6].decode('latin1').strip('\x00 ') for i in range(ns)]
    arr(2)
    physmin = np.frombuffer(arr(8), '<f8'); physmax = np.frombuffer(arr(8), '<f8')
    digmin = np.frombuffer(arr(8), '<f8'); digmax = np.frombuffer(arr(8), '<f8')
    b = arr(68); pre = [b[i*68:(i+1)*68].decode('latin1').strip('\x00 ') for i in range(ns)]
    lp = np.frombuffer(arr(4), '<f4'); hp = np.frombuffer(arr(4), '<f4'); nt = np.frombuffer(arr(4), '<f4')
    spr = np.frombuffer(arr(4), '<u4'); gt = np.frombuffer(arr(4), '<u4')
    info = dict(version=buf[:8].decode(), hl=hl, nrec=nrec, dur=dur, ns=ns, labels=labels, pdim=pdim, physmin=physmin, physmax=physmax,
                digmin=digmin, digmax=digmax, pre=pre, lp=lp, hp=hp, nt=nt, spr=spr, gt=gt)
    sizes = [np.dtype(GT[int(g)]).itemsize * int(s) for g, s in zip(gt, spr)]
    recb = sum(sizes); offs = np.concatenate([[0], np.cumsum(sizes)])
    d0 = hl * 256
    raw = np.frombuffer(buf, dtype=np.uint8, count=recb * nrec, offset=d0).reshape(nrec, recb)
    data = {}
    for i, l in enumerate(labels):
        if want is not None and l not in want: continue
        blk = np.ascontiguousarray(raw[:, offs[i]:offs[i + 1]]).view(GT[int(gt[i])]).reshape(-1)
        x = blk.astype(np.float64)
        if physmax[i] != physmin[i] and digmax[i] != digmin[i] and int(gt[i]) < 16:
            x = (x - digmin[i]) / (digmax[i] - digmin[i]) * (physmax[i] - physmin[i]) + physmin[i]
        data[l] = x
    # events
    e0 = d0 + recb * nrec
    ev = None
    if len(buf) > e0 + 8:
        mode = buf[e0]; nev = int.from_bytes(buf[e0+1:e0+4], 'little'); fs_ev = struct.unpack('<f', buf[e0+4:e0+8])[0]
        p = e0 + 8
        pos = np.frombuffer(buf, '<u4', nev, p); p += 4 * nev
        typ = np.frombuffer(buf, '<u2', nev, p); p += 2 * nev
        chn = dur_ = None
        if mode == 3:
            chn = np.frombuffer(buf, '<u2', nev, p); p += 2 * nev
            dur_ = np.frombuffer(buf, '<u4', nev, p); p += 4 * nev
        ev = dict(mode=mode, n=nev, fs=fs_ev, pos=pos, typ=typ, chn=chn, dur=dur_)
    info['rate'] = [float(s) / dur for s in spr]
    return info, data, ev



def free_gb():
    return shutil.disk_usage(str(stdlib.HERE)).free / 1e9


def wait_disk(need):
    t0 = time.time()
    while free_gb() < KEEP_FREE_GB + need:
        if time.time() - t0 > 7200:
            raise RuntimeError("PC disk too full for 2 h")
        time.sleep(60)


rows, extra = [], []
stat = {}
TMP.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
subjects = [int(ONLY)] if ONLY else list(range(1, 16))
try:
    for s in subjects:
        zname = f"S{s:02d}_ME.zip"
        if not (RAW / zname).exists():
            wait_disk(0.4)
            fetch.main(ID, URL % s, zname)
        z = zipfile.ZipFile(RAW / zname)
        names = sorted(z.namelist(), key=lambda n: int(n.split("run")[-1].split(".")[0]))
        st = stat[s] = dict(runs=0, trials=0, per_class={}, cue_delay=[], delta={})
        k = 0
        for nm in names:
            run = int(nm.split("run")[-1].split(".")[0])
            buf = z.read(nm)
            info, data, ev = read_gdf(buf, set(MOVE))
            del buf
            st["runs"] += 1
            fs = info["rate"][info["labels"].index("Elbow")]
            assert fs == 64.0, fs
            fs_ev = ev["fs"]
            pos = ev["pos"].astype(float) / fs_ev
            typ = ev["typ"].astype(int)
            arr = np.column_stack([data[c] for c in MOVE]).astype(np.float32)
            starts = [i for i, t in enumerate(typ) if t == 768]
            for j, i0 in enumerate(starts):
                i1 = starts[j + 1] if j + 1 < len(starts) else len(typ)
                t0 = pos[i0]
                t1 = pos[starts[j + 1]] if j + 1 < len(starts) else len(arr) / fs
                a, b = int(round(t0 * fs)), int(round(t1 * fs))
                seg = arr[a:b]
                evs = [(int(typ[i]), pos[i]) for i in range(i0, i1)]
                cue = [t for t, _ in evs if 1536 <= t <= 1542]
                assert len(cue) == 1, (s, run, j, cue)
                cls = CLASSES[cue[0]]
                k += 1
                st["trials"] += 1
                st["per_class"][cls] = st["per_class"].get(cls, 0) + 1
                cross = [p for t, p in evs if t == 786]
                cuet = [p for t, p in evs if t == cue[0]]
                if cross:
                    st["cue_delay"].append(cuet[0] - cross[0])
                base = seg[: max(1, int(0.8 * fs))].mean(0)
                late = seg[int((cuet[0] - t0 + 2.0) * fs): int((cuet[0] - t0 + 3.0) * fs)]
                if len(late):
                    st["delta"].setdefault(cls, []).append(late.mean(0) - base)
                fn = f"samples/{ID}_S{s:04d}_trial{k:03d}.csv"
                df = pd.DataFrame(seg, columns=MOVE)
                df.insert(0, "time_s", np.char.mod("%.4f", np.arange(len(seg)) / fs))
                df.to_csv(TMP / f"t{s}_{k}.csv", index=False, float_format="%.6g", lineterminator="\r\n")
                extra.append((fn, str(TMP / f"t{s}_{k}.csv")))
                evtxt = ";".join(f"{EVN.get(t, 'e%d' % t)}={p - t0:.4f}" for t, p in evs)
                rows.append(dict(
                    dataset=ID, study="Ofner et al. 2017, PLoS ONE", experiment="movement execution", group="", subject=f"S{s:02d}",
                    person=f"S{s:02d}", subject_excluded_by_authors=False, age="", sex="", handedness=("" if s == 1 else "right"),
                    population="healthy", setting="lab",
                    device="arm exoskeleton joint-angle sensors (Hocoma), 5DT data glove, hand and elbow position, 64 Hz",
                    session=1, block=f"run {run}", trial=k, phase="test", time_s=t0, task="single-joint arm and hand movements",
                    unit="trial", condition=cls, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                    excluded_by_authors="", recording=k, samples_file=fn,
                    signal_kind="exoskeleton joint angles, hand and elbow positions, data glove finger sensors",
                    channels=";".join(MOVE), channel_units=";".join(["unknown"] * len(MOVE)), sample_rate_hz=fs, n_samples=len(seg),
                    duration_s=len(seg) / fs, events=evtxt, processing="none stated for the movement channels",
                    trial_info=f"cue_code={cue[0]};run={run};trial_in_run={j + 1}"))
        z.close()
        if not ONLY:
            (RAW / zname).unlink()
        print(s, {k2: v for k2, v in st.items() if k2 in ("runs", "trials", "per_class")}, f"free {free_gb():.1f} GB", flush=True)
    if ONLY:
        tot = sum(os.path.getsize(f) for _, f in extra)
        print("raw csv bytes", tot, "rows", len(rows))
        for cls in CLASSES.values():
            print(cls, np.round(np.mean(stat[subjects[0]]["delta"][cls], 0), 2)[[28, 29, 30, 0, 2, 12]])
        print("cue delay", np.mean(stat[subjects[0]]["cue_delay"]), np.std(stat[subjects[0]]["cue_delay"]))
        sys.exit(0)
    S_ = list(stat)
    checks.published("participants", len(S_), 15, 0, "Methods, Subjects")
    checks.published("runs per participant (min over participants)", min(stat[s]["runs"] for s in S_), 10, 0, "Methods, Paradigm (10 runs per session)")
    checks.published("runs per participant (max over participants)", max(stat[s]["runs"] for s in S_), 10, 0, "Methods, Paradigm (10 runs per session)")
    checks.published("trials per participant (min)", min(stat[s]["trials"] for s in S_), 420, 0, "Methods, Paradigm (10 runs x 42 trials)")
    checks.published("trials per participant (max)", max(stat[s]["trials"] for s in S_), 420, 0, "Methods, Paradigm (10 runs x 42 trials)")
    for cls in CLASSES.values():
        pc = [stat[s]["per_class"].get(cls, 0) for s in S_]
        checks.published(f"trials per participant of class {cls} (min over participants)", min(pc), 60, 0, "Methods, Paradigm (60 trials per class)")
        checks.published(f"trials per participant of class {cls} (max over participants)", max(pc), 60, 0, "Methods, Paradigm (60 trials per class)")
    cd = float(np.mean([np.mean(stat[s]["cue_delay"]) for s in S_]))
    checks.published("cue after the cross (s)", cd, 2.0, 0.05, "Fig 2 caption: cross at second 0, cue at second 2")
    dm = {c: np.mean([np.mean(stat[s]["delta"][c], 0) for s in S_], 0) for c in CLASSES.values()}
    ie, ip = MOVE.index("Elbow"), MOVE.index("ProSupination")
    checks.internal("elbow angle changes in opposite directions for flexion and extension", dm["elbow flexion"][ie] * dm["elbow extension"][ie] < 0,
                    f"{dm['elbow flexion'][ie]:.1f} vs {dm['elbow extension'][ie]:.1f}")
    checks.internal("forearm angle changes in opposite directions for supination and pronation",
                    dm["forearm supination"][ip] * dm["forearm pronation"][ip] < 0, f"{dm['forearm supination'][ip]:.1f} vs {dm['forearm pronation'][ip]:.1f}")
    checks.internal("every trial has one cue and a samples file", all(r["samples_file"] for r in rows))
    differences = [
        "The files do not state units for the movement channels. Glove sensors lie between 0 and 1 (normalised), angle channels are presumably degrees and positions presumably mm; the channel_units column therefore says unknown.",
        "Cue codes (1536 elbow flexion, 1537 elbow extension, 1538 forearm supination, 1539 forearm pronation, 1540 hand close, 1541 hand open, 1542 rest) follow the BNCI Horizon 2020 data set description; the direction of the elbow and forearm sensors was checked against the cue (see the checks).",
        "The authors' detected movement onsets and artefact marks were not part of the files, so excluded_by_authors is empty. Participant 1 was not right-handed (handedness left empty); the paper gives no age or sex per participant.",
        "Channel labels are as in the files; some are truncated or misspelt there (ShoulderAdductio = shoulder adduction, ShoulderFlexionE = shoulder flexion/extension, litte_* = little finger).",
    ]
    meta = dict(
        paper=dict(authors="Ofner P, Schwarz A, Pereira J, Muller-Putz GR", year=2017,
                   title="Upper limb movements can be decoded from the time-domain of low-frequency EEG", venue="PLoS ONE 12(8):e0182578",
                   doi="10.1371/journal.pone.0182578"),
        lab="Muller-Putz (TU Graz)", data_url="https://bnci-horizon-2020.eu/database/data-sets (001-2017)", license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Only the 32 movement channels (exoskeleton, hand/elbow position, data glove) and the event markers of the movement-execution files were kept; EEG, EOG and the motor-imagery files are not included. Samples are cut into one file per trial (from one trial start marker to the next, so the return to the starting position is included) and rounded to 6 significant digits.",
        how=["One row is one trial (cross on, cue 2 s later, sustained movement, return to the start position) of a movement-execution run; 10 runs of 42 trials per participant, 6 movement classes plus rest. Its samples are in samples/<file>.csv at 64 Hz: time_s from the trial start marker, then the 32 movement channels.",
             "Events are listed as name=seconds from the start of the segment (the authors' marker codes, named with the BNCI description: trial_start 768, beep 785, cross_on 786, cue_* 1536-1542, and the matching end markers).",
             "time_s in the main table is the segment start within its run file. condition is the cued movement."],
        unit_note="trial (movement-execution run segment)", differences=differences)
    stdlib.release(ID, "timeseries", rows, meta, checks, extra_files=extra)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
    if not ONLY:
        for f in RAW.glob("S*_ME.zip"):
            f.unlink()
