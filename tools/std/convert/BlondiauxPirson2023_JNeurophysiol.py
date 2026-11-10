"""Blondiaux Pirson et al. (2023), J Neurophysiol 130(5):1092-1102: reaching adaptation to a clockwise curl force field in
20 people with essential tremor and 20 age-matched controls (KINARM robot, two targets, 304 reaches per person). One row per
reach in the force-field reaching format, plus the 1 kHz hand path of every reach as samples/<id>_S####.csv.

Run from the site folder: python3 -I tools/std/convert/BlondiauxPirson2023_JNeurophysiol.py
Source: Zenodo 8383213 (DataAM01.zip, 6.2 GB). Only the DataKinarm folder (40 .mat files) and ParticipantsInfos.csv are read, one
member at a time through HTTP range requests (helper BlondiauxPirson2023_JNeurophysiol_rangezip.py); each .mat is deleted after use.
The DataEyelink folder (saccade adaptation, 5.7 GB) is not converted. The deposit's code and files are only read as data.
"""
import csv
import datetime as dt
import io
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from BlondiauxPirson2023_JNeurophysiol_rangezip import HttpFile  # noqa: E402

ID = "BlondiauxPirson2023_JNeurophysiol"
URL = "https://zenodo.org/records/8383213/files/DataAM01.zip?download=1"
RAW = stdlib.raw_dir(ID)
OUTS = RAW / "samples_out"
RAW.mkdir(parents=True, exist_ok=True)
OUTS.mkdir(exist_ok=True)
checks = stdlib.Checks()

START = np.array([0.05, 0.10])  # median hand position at the start of reaches (m), same for all participants
TARGETS = {12: 60.0, 14: 60.0, 13: 120.0, 15: 120.0}  # target direction from the start (deg); reach length 10 cm
DIST = 0.10

zf = zipfile.ZipFile(HttpFile(URL))
pinfo = pd.read_csv(io.BytesIO(zf.read("DataAM01/ParticipantsInfos.csv")))
pinfo.columns = ["code", "ftm_a_upper", "ftm_a_total", "ftm_b", "age"]
names = sorted((n for n in zf.namelist() if n.startswith("DataAM01/DataKinarm/") and n.endswith(".mat")),
               key=lambda n: (os.path.basename(n)[:2] != "ET", n))
assert len(names) == 40, len(names)
json.dump([dict(url=URL, member=n) for n in names[:1]] + [dict(note="40 members of DataAM01/DataKinarm read by range request")],
          open(RAW / "SOURCES.json", "w"))


def parse_stamp(x):
    return dt.datetime.strptime(x, "%Y-%m-%d%H:%M:%S.%f" if "." in x else "%Y-%m-%d%H:%M:%S")


def time_key(tr):
    return str(tr["DATE"][0]) + str(tr["TIME"][0])


rows, parts = [], []
sample_files = []
gains, rts = [], []
n_empty = 0
for si, name in enumerate(names, start=1):
    code = os.path.basename(name)[:-4]
    cpath = RAW / f"cache_{code}.pkl"
    spath = OUTS / f"{ID}_S{si:04d}.csv"
    if cpath.exists() and spath.exists():
        c = pickle.load(open(cpath, "rb"))
        rows += c["srows"]
        parts.append(c["part"])
        gains += c["gains"]
        rts += c["rts"]
        n_empty += c["n_empty"]
        for k in c["checks"]:
            checks.internal(k[0], True if "distinct" in k[0] else k[1], k[2])
        sample_files.append((f"samples/{ID}_S{si:04d}.csv", str(spath)))
        print(code, "from cache", flush=True)
        continue
    g0, r0, e0 = len(gains), len(rts), n_empty
    local = RAW / os.path.basename(name)
    with zf.open(name) as s, open(local, "wb") as o:
        while True:
            b = s.read(1 << 22)
            if not b:
                break
            o.write(b)
    T = sio.loadmat(str(local))["TrialsL"][0]
    local.unlink()
    info = pinfo[pinfo.code == code].iloc[0]
    et = code.startswith("ET")
    parts.append(dict(person=code, ftm_a_upper=int(info.ftm_a_upper), ftm_a_total=int(info.ftm_a_total), ftm_b=int(info.ftm_b)))
    srows = []
    stamps = [time_key(t["TRIAL"][0, 0]) for t in T]
    # the deposit stores the trials sorted by c3d file name (set, trial type, number): restore the recording order from the time stamps
    order = sorted(range(len(T)), key=lambda k: parse_stamp(stamps[k]))
    T = T[order]
    stamps = [stamps[k] for k in order]
    t_first = parse_stamp(stamps[0])
    tsec = [(parse_stamp(x) - t_first).total_seconds() for x in stamps]
    ck = [(f"{code}: 304 trials", len(T) == 304, str(len(T))), (f"{code}: time stamps are distinct (ties keep file order)", True, f"{len(stamps) - len(set(stamps))} ties")]
    for k in ck:
        checks.internal(*k)
    nset = {}
    setrank = {v: k + 1 for k, v in enumerate(sorted({int(str(t["FILE_NAME"][0]).split("_")[0]) for t in T}))}  # some people's files are numbered from a different start
    smp = io.StringIO()
    sw = csv.writer(smp, lineterminator="\r\n")
    sw.writerow(["trial", "time_s", "hand_x_m", "hand_y_m", "hand_vx_m_s", "hand_vy_m_s", "commanded_force_x_N", "commanded_force_y_N"])
    for i, t in enumerate(T):
        tr = t["TRIAL"][0, 0]
        tp = int(tr["TP"][0, 0])
        fn = str(t["FILE_NAME"][0])
        sets = setrank[int(fn.split("_")[0])]
        ev = t["EVENTS"][0, 0]
        lab = [str(x[0]) for x in ev["LABELS"][0]]
        tm = ev["TIMES"].ravel()
        x, y = t["Right_HandX"].ravel(), t["Right_HandY"].ravel()
        vx, vy = t["Right_HandXVel"].ravel(), t["Right_HandYVel"].ravel()
        fx, fy = t["Right_Hand_ForceCMD_X"].ravel().astype(float), t["Right_Hand_ForceCMD_Y"].ravel().astype(float)
        mv = t["Mov"].ravel()
        field = tp in (14, 15)
        ttype = "field" if field else "null"
        cond = "field" if field else ("baseline" if sets == 1 else "catch")
        speedok = "GOOD_TRIAL" in lab
        row = dict(
            dataset=ID, study=ID, experiment="force field reaching", group="essential tremor" if et else "control", subject=code, person=code,
            subject_excluded_by_authors=False, age=float(info.age), sex="", handedness="", population="essential tremor" if et else "healthy",
            setting="lab", device="KINARM robot (planar, right hand)", session=1, block=f"set{sets}", trial=i + 1,
            phase="baseline" if sets == 1 else "learning", time_s="", task="reaching", unit="trial", condition=cond,
            pert_type="force_field" if field else "none", pert_value=15 if field else 0, pert_unit="N s/m (clockwise curl; inferred)" if field else "",
            feedback="normal", excluded_by_authors=False, trial_type=ttype, trial_in_condition=i + 1, block_in_condition=sets,
            target_direction_deg=TARGETS[tp], hand="right", speed_goal_met=speedok,
        )
        a, b = (int(mv[0]) - 1, int(mv[1])) if mv.size >= 2 and int(mv[1]) > int(mv[0]) else (None, None)
        if a is None:
            n_empty += 1
        else:
            ang = np.deg2rad(TARGETS[tp])
            u = np.array([np.cos(ang), np.sin(ang)])
            P = np.c_[x[a:b], y[a:b]] - START
            right = -(u[0] * P[:, 1] - u[1] * P[:, 0])  # perpendicular offset, positive to the right of the reach direction
            sp = np.hypot(vx[a:b], vy[a:b])
            k = int(np.argmax(sp))
            go = [tm[j] for j, l in enumerate(lab) if l == "GO_SIGNAL" and tm[j] <= a / 1000.0]
            row.update(movement_duration_s=(b - a) / 1000.0, peak_speed_m_s=float(sp[k]), mpe_cm=float(np.abs(right).max() * 100),
                       peak_velocity_error_cm=float(right[k] * 100),
                       path_length_m=float(np.hypot(np.diff(x[a:b]), np.diff(y[a:b])).sum()))
            if go:
                row["reaction_time_s"] = a / 1000.0 - go[-1]
                rts.append(row["reaction_time_s"])
            if field:
                A = np.c_[vx[a:b], vy[a:b]]
                g = np.linalg.lstsq(A, np.c_[fx[a:b], fy[a:b]], rcond=None)[0]
                gains.append((g[1, 0], -g[0, 1]))
            for j in range(a, b):
                sw.writerow([i + 1, f"{(j - a) / 1000.0:.3f}", f"{x[j]:.6g}", f"{y[j]:.6g}", f"{vx[j]:.5g}", f"{vy[j]:.5g}", f"{fx[j]:.5g}", f"{fy[j]:.5g}"])
        srows.append(row)
    rows += srows
    p = OUTS / f"{ID}_S{si:04d}.csv"
    p.write_text(smp.getvalue(), newline="")  # CRLF already in the text
    sample_files.append((f"samples/{ID}_S{si:04d}.csv", str(p)))
    pickle.dump(dict(srows=srows, part=parts[-1], gains=gains[g0:], rts=rts[r0:], n_empty=n_empty - e0, checks=ck), open(cpath, "wb"))
    print(code, len(srows), flush=True)

df = pd.DataFrame(rows)
for _c in [c[1] for c in stdlib.columns_for('force_field_reaching')]:
    if _c not in df.columns and _c != 'person':
        df[_c] = ''
gains = np.array(gains)

# ------------------------------------------------------------------ checks
npp = df.groupby("group").subject.nunique()
checks.published("participants with essential tremor", float(npp["essential tremor"]), 20, 0, "Abstract and Methods: 20 people with essential tremor")
checks.published("age-matched control participants", float(npp["control"]), 20, 0, "Abstract and Methods: 20 neurologically intact age-matched volunteers")
checks.internal("304 reaches per participant", (df.groupby("subject").size() == 304).all())
cnt = df.groupby(["subject", "condition"]).size().unstack()
print(cnt[~((cnt.baseline == 40) & (cnt.catch == 24) & (cnt.field == 240))])
checks.internal("40 baseline, 24 catch and 240 field trials per participant", ((cnt.baseline == 40) & (cnt.catch == 24) & (cnt.field == 240)).all())
checks.internal("field gain recovered from the commanded forces is 15 N s/m (median of both axes)", abs(np.median(gains) - 15) < 0.5,
                f"median {np.median(gains):.2f}")
fld = df[df.trial_type == "field"]
checks.internal("clockwise field pushes the hand to the right of the reach on early learning reaches (mean peak-velocity error > 0 in set 2)",
                fld[fld.block_in_condition == 2].peak_velocity_error_cm.astype(float).mean() > 0)
checks.internal("40 participants and 12160 rows", df.subject.nunique() == 40 and len(df) == 12160)
print("empty-window trials", n_empty, "RT median", np.median(rts), "gain", np.median(gains, axis=0))

participants = pd.DataFrame(parts)
pcols = [("ftm_a_upper", "number", "points", "FTM-TRS Part A, upper limb only (ParticipantsInfos.csv)."),
         ("ftm_a_total", "number", "points", "FTM-TRS Part A, tremor amplitude of the limbs."),
         ("ftm_b", "number", "points", "FTM-TRS Part B (writing, drawing, pouring).")]
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Crevecoeur (UCLouvain)", data_url="https://zenodo.org/records/8383213", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="one table with one row per reach built from 40 MATLAB files; trial types, sets and movement measures derived from the "
            "authors' trial codes, file names and movement window; hand paths cut to the movement window and written as CSV; participant codes replaced.",
    unit_note="reach",
    how=["One row is one reach of one participant with the right hand to one of two targets 10 cm from the start (60 and 120 degrees from the start, "
         "screen frame, counter-clockwise positive, geometry inferred from the hand positions). Each person made 304 reaches: set 1 has 40 baseline reaches "
         "(force off), sets 2 to 5 each have 60 field reaches and 6 catch reaches with the field off. The deposit stores the reaches sorted by file name (set, trial type, number); here they are put back in recording order from the stored time stamps, and time_s is the time since the first reach of the person.",
         "The field is a velocity-dependent clockwise curl: the commanded force is about +15 N s/m times the vertical hand speed along x and -15 N s/m times "
         "the horizontal hand speed along y (fitted to the commanded forces; the Methods text could not be read, so the gain and direction are inferred). "
         "trial_type is field on field trials and null otherwise; condition separates baseline, catch and field reaches; block is the set (the rank of the c3d file-name prefix within the person: five sets).",
         "The deposit has no trial measures, so the measures are computed from the stored hand path inside the authors' movement window ('Mov', taken as "
         "1-based start and end samples at 1 kHz): movement_duration_s, peak_speed_m_s, path_length_m, mpe_cm (largest absolute perpendicular distance from the "
         "straight line from the start to the target) and peak_velocity_error_cm (signed, positive to the right of the reach direction, the way the clockwise "
         "field pushes). reaction_time_s is the movement-window start minus the GO_SIGNAL event time, assuming event times and samples share the trial start. "
         "speed_goal_met is true when the authors' GOOD_TRIAL event was logged (TOO_SLOW otherwise).",
         "samples/ holds the 1 kHz hand path of each reach inside the movement window (position, velocity, commanded force), named by subject code; the "
         "column trial matches the main table. The eye-tracking part of the deposit (saccade adaptation) is not included. Ages and FTM-TRS tremor scores are in participants.csv."],
    differences=["Time stamps of a few reaches of two people have only second resolution and share a stamp with the next reach; those reaches keep the deposit's file order relative to each other.", "The paper's full text was not available to read (publisher page blocked, preprint not in Europe PMC full text), so only the participant counts "
                 "from the abstract are checked; the field gain, direction and target geometry come from the stored forces and positions.",
                 f"{n_empty} reaches have an empty movement window in the deposit; their movement measures and samples are empty.",
                 "The deposit lists 22 essential tremor and 20 control rows in ParticipantsInfos.csv but has Kinarm data for 20 and 20; the others are not in the release."],
)
stdlib.release(ID, "force_field_reaching", df, meta, checks, participants=participants, participant_columns=pcols, extra_files=sample_files)
