"""Moore, Hnat & van den Bogert (2015), PeerJ 3:e918: perturbed treadmill walking, 15 participants (Zenodo 13030, CC0).

Raw files, raw_extra/<id>/ (read directly from the two tar.gz files, nothing is extracted):
  T<nnn>/meta-<nnn>.yml    study / subject / trial metadata (read as text with a small line parser, never as YAML objects)
  T<nnn>/mocap-<nnn>.txt   tab separated, ~100 Hz: TimeStamp, FrameNumber, 47 markers (.PosX/Y/Z, m), FP1/FP2 CoP (m), force (N),
                           moment (N m), 12 analog force-plate voltages and 12 accelerometer channels
  T<nnn>/record-<nnn>.txt  tab separated, ~300 Hz: Time, LeftBeltSpeed, RightBeltSpeed (m/s), ReferenceFY_FP1, and comment lines
                           '# EVENT A - COUNT 1' just before the row at which each protocol event occurred
Released per trial: samples/<code>_trial<nnn>.csv = time_s, FrameNumber, force plate CoP/force/moment for both plates, and the markers
SACR, LHEE, LTOE, RHEE, RTOE; samples/<code>_trial<nnn>_belts.csv = time_s, LeftBeltSpeed, RightBeltSpeed, ReferenceFY_FP1.
Left out for size: the other 42 markers, the analog channels. Trials with subject id 0 (no person on the treadmill) are not released.
"""
import re
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Moore2015_PeerJ"
RAW = stdlib.raw_dir(ID)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
MARK = ["SACR", "LHEE", "LTOE", "RHEE", "RTOE"]
FPC = [f"FP{p}.{k}{a}" for p in (1, 2) for k in ("Cop", "For", "Mom") for a in "XYZ"]
MCOLS = FPC + [f"{m}.Pos{a}" for m in MARK for a in "XYZ"]
EVN = {"A": "force_plate_zeroing", "B": "calibration_pose", "C": "first_normal_walking", "D": "longitudinal_perturbation",
       "E": "second_normal_walking", "F": "unloaded_end"}


def meta_parse(txt):
    """Flat read of the indented key: value lines (no YAML objects are built)."""
    sec, d = None, {"subject": {}, "trial": {}, "study": {}}
    for l in txt.splitlines():
        if re.match(r"^[a-z]+:\s*$", l):
            sec = l.strip()[:-1]
        elif sec in d and (m := re.match(r"^\s+([a-z-]+):\s*(\S.*)?$", l)) and m[2] and m[2] != ">":
            d[sec].setdefault(m[1], m[2].strip())
    return d


def marker_map(txt):
    out, on = {}, False
    for l in txt.splitlines():
        if re.match(r"^\s+marker-map:\s*$", l):
            on = True
            ind = len(l) - len(l.lstrip())
            continue
        if on:
            if len(l) - len(l.lstrip()) <= ind and l.strip():
                break
            if (m := re.match(r"^\s+(\S+):\s*(\S+)\s*$", l)):
                out[m[1]] = m[2]
    return out


def process(tid, meta, mocap, record):
    md = meta_parse(meta.decode())
    sid = int(md["subject"]["id"])
    if sid == 0:
        return None
    import io
    hdr = pd.read_csv(io.BytesIO(mocap), sep="\t", nrows=0).columns
    mm = marker_map(meta.decode())
    ren = {}
    for h in hdr:
        base, _, ax = h.strip().rpartition(".")
        if ax in ("PosX", "PosY", "PosZ") and base in mm:
            ren[h] = f"{mm[base]}.{ax}"
        elif h != h.strip():
            ren[h] = h.strip()
    df = pd.read_csv(io.BytesIO(mocap), sep="\t").rename(columns=ren)
    miss = [c for c in MCOLS if c not in df.columns]
    for c in miss:
        df[c] = np.nan
    t0 = float(df.TimeStamp.iloc[0])
    out = pd.DataFrame({"time_s": df.TimeStamp - t0, "FrameNumber": df.FrameNumber})
    for c in MCOLS:
        out[c] = df[c]
    p = TMP / f"{tid}.csv"
    out.to_csv(p, index=False, float_format="%.6f", lineterminator="\n")
    # record file: belts + events
    ev, rows, pend = [], [], []
    for l in record.decode().splitlines():
        if l.startswith("# EVENT") and "COUNT" in l:
            pend.append(re.match(r"# EVENT ([A-F]) - COUNT (\d+)", l)[1])
        elif l.startswith("#") or l.startswith("Time"):
            continue
        else:
            p4 = l.split("\t")
            if len(p4) == 4:
                if pend:
                    ev += [f"{EVN[e]}={float(p4[0]) - t0:.6f}" for e in pend]
                    pend = []
                rows.append((float(p4[0]) - t0, p4[1], p4[2], p4[3]))
    b = TMP / f"{tid}_belts.csv"
    if rows:
      with open(b, "w") as fh:
        fh.write("time_s,LeftBeltSpeed,RightBeltSpeed,ReferenceFY_FP1\n")
        for r in rows:
            fh.write(f"{r[0]:.6f},{r[1]},{r[2]},{r[3]}\n")
    nfr = int(df.FrameNumber.iloc[-1] - df.FrameNumber.iloc[0])
    return dict(tid=int(tid[1:]), sid=sid, md=md, n=len(df), dur=float(out.time_s.iloc[-1]), events=";".join(ev),
                rate=nfr / (float(out.time_s.iloc[-1]) or 1), nbelt=len(rows), missing_markers=[c for c in miss if "Pos" in c])


recs, buf = [], {}
for tp in sorted(RAW.glob("perturbed-walking-data-0*.tar.gz")):
    with tarfile.open(tp, "r|gz") as t:
        for m in t:
            if not m.isfile() or not m.name.startswith("T"):
                continue
            tid, fn = m.name.split("/")
            kind = fn.split("-")[0]
            buf.setdefault(tid, {})[kind] = t.extractfile(m).read()
            if len(buf[tid]) == 3:
                b = buf.pop(tid)
                r = process(tid, b["meta"], b["mocap"], b["record"])
                if r:
                    recs.append(r)
                print(tid, r and (r["n"], round(r["dur"]), r["sid"]), flush=True)
for tid_, b in list(buf.items()):
    assert set(b) == {"meta", "mocap"}, (tid_, list(b))  # trials 3-5 have no record file in the deposit
    r = process(tid_, b["meta"], b["mocap"], b"")
    if r:
        recs.append(r)
    print(tid_, r and (r["n"], round(r["dur"]), r["sid"]), "(no record file)", flush=True)
recs.sort(key=lambda r: (r["sid"], r["tid"]))
sids = sorted({r["sid"] for r in recs})
code = {s: f"{ID}_S{i + 1:04d}" for i, s in enumerate(sids)}
# the release assigns codes in order of first appearance: rows are written in this same order
rows_out, extra, tr_of = [], [], {}
for r in recs:
    s, md = r["sid"], r["md"]
    sub, tri = md["subject"], md["trial"]
    tr_of[s] = tr_of.get(s, 0) + 1
    tn = tr_of[s]
    fn = f"samples/{code[s]}_trial{r['tid']:03d}.csv"
    fb = f"samples/{code[s]}_trial{r['tid']:03d}_belts.csv"
    extra += [(fn, str(TMP / f"T{r['tid']:03d}.csv"))]
    if r["nbelt"]:
        extra += [(fb, str(TMP / f"T{r['tid']:03d}_belts.csv"))]
    tid = r["tid"]
    force_ok = not (6 <= tid <= 15)
    pilot = 3 <= tid <= 15
    speed = float(tri["nominal-speed"])
    info = (f"trial_id={tid};nominal_speed_mps={speed:g};belts_file={fb if r['nbelt'] else ''};marker_set={tri.get('marker-set', '')};"
            f"stationary_platform={tri.get('stationary-platform', '')};pitch={tri.get('pitch', '')};sway={tri.get('sway', '')};"
            f"markers_not_found={len(r['missing_markers'])};dflow_version={tri.get('dflow-version', '')};force_plate_data_valid={force_ok};pilot_protocol={pilot}")
    rows_out.append(dict(
        dataset=ID, study="Moore 2015 PeerJ", experiment="perturbed treadmill walking", group="", subject=str(s), person=str(s),
        subject_excluded_by_authors=False, age=int(sub["age"]), sex={"male": "M", "female": "F"}.get(sub.get("gender", ""), ""), handedness="",
        population="healthy", setting="lab", device="instrumented treadmill with 2 force plates and independent belts; Motion Analysis Osprey cameras",
        session=1, block=f"trial {tid}", trial=tn, phase="test", time_s="", task="walking", unit="recording",
        condition=f"nominal belt speed {speed:g} m/s, 1 min normal, 8 min perturbed, 1 min normal", pert_type="longitudinal_belt_speed_perturbation",
        pert_value="", pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=tn, samples_file=fn, signal_kind="force plate loads (centre of pressure, force, moment) and sacrum, heel and toe marker positions",
        channels=";".join(["FrameNumber"] + FPC + [f"{m}.Pos{a}" for m in MARK for a in "XYZ"]),
        channel_units=";".join(["frame"] + ["m" if "Cop" in c else "N" if "For" in c else "N m" for c in FPC] + ["m"] * 15),
        sample_rate_hz=100, n_samples=r["n"], duration_s=r["dur"], events=r["events"],
        processing="none; marker coordinates are 0 where a marker was lost (D-Flow 3.16.2rc4 or later) or hold the last visible value (earlier versions); time_s is TimeStamp minus its first value",
        trial_info=info))
D = pd.DataFrame(rows_out)
heights = {}
masses = {}
ages = {}
sex = {}
for r in recs:
    sb = r["md"]["subject"]
    heights[r["sid"]] = float(sb["height"])
    masses[r["sid"]] = float(sb["mass"])
    ages[r["sid"]] = int(sb["age"])
    sex[r["sid"]] = sb.get("gender")
part = pd.DataFrame([dict(person=str(s), height_m=heights[s], mass_kg=masses[s]) for s in sids])
pcols = [("height_m", "number", "m", "Height measured by the experimenter (meta file)."),
         ("mass_kg", "number", "kg", "Mass from the meta file: measured from the force plates where possible, otherwise self-reported (Table 1 of the paper).")]

c = stdlib.Checks()
a = np.array([ages[s] for s in sids], float)
h = np.array([heights[s] for s in sids])
ms = np.array([masses[s] for s in sids])
c.published("participants", len(sids), 15, 0, "Participants")
c.published("females", sum(1 for s in sids if sex[s] == "female"), 4, 0, "Participants")
c.published("males", sum(1 for s in sids if sex[s] == "male"), 11, 0, "Participants")
c.published("age mean (years)", a.mean(), 24, 0.5, "Participants (printed as 24 +/- 4)")
c.published("age SD (years)", a.std(ddof=1), 4, 0.5, "Participants (printed as 24 +/- 4)")
c.published("height mean (m)", h.mean(), 1.75, 0.005, "Participants")
c.published("height SD (m)", h.std(ddof=1), 0.09, 0.005, "Participants")
c.published("mass mean (kg)", ms.mean(), 74, 0.5, "Participants (printed as 74 +/- 13)")
c.published("mass SD (kg)", ms.std(ddof=1), 13, 0.5, "Participants (printed as 74 +/- 13)")
c.published("trials assigned to a participant in Table 1", len(D), 46, 0, "Table 1 (trial numbers per subject and speed)")
for sp, n in (("0.8", 15), ("1.2", 17), ("1.6", 14)):
    c.published(f"trials at {sp} m/s in Table 1", D.trial_info.str.contains(f"nominal_speed_mps={sp};").sum(), n, 0, "Table 1")
c.internal("every trial has the 6 protocol events or the pilot variant", sum(1 for r in recs if r["events"]) >= len(recs) - 8, str([r["tid"] for r in recs if not r["events"]]))
print("trials with missing markers:", [(r["tid"], len(r["missing_markers"])) for r in recs if r["missing_markers"]])
c.internal("belt file read for every trial that has a record file", sum(1 for r in recs if r["nbelt"] > 1000) >= len(recs) - 6, str([(r["tid"], r["nbelt"]) for r in recs if r["nbelt"] <= 1000]))
c.internal("frame rate near 100 Hz", sum(1 for r in recs if abs(r["rate"] - 100) < 2) >= len(recs) - 3, str([(r["tid"], round(r["rate"], 2)) for r in recs if abs(r["rate"] - 100) >= 2]))
print("trials per speed", D.trial_info.str.extract(r"nominal_speed_mps=([0-9.]+)")[0].value_counts().to_dict())

meta = dict(
    paper=dict(authors="Moore JK, Hnat SK, van den Bogert AJ", year=2015, title="An elaborate data set on human gait and the effect of mechanical perturbations",
               venue="PeerJ 3:e918", doi="10.7717/peerj.918"),
    lab="van den Bogert (Cleveland State University)", data_url="https://doi.org/10.5281/zenodo.13030",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Read from the two tar.gz files without extracting. For every trial with a participant, the force-plate loads and five markers (SACR, LHEE, LTOE, RHEE, RTOE) "
             "of the mocap file became one samples CSV and the belt-speed columns of the record file another; the 42 other markers and the analog channels were left out "
             "to keep the release under 1 GB. Times are shifted so that each trial starts at 0 s. Participant identifiers are replaced by codes; the metadata notes and trial dates are not copied."),
    how=["One row is one 10-minute trial of one participant on the instrumented treadmill: 1 min of normal walking, 8 min with pseudo-random longitudinal belt-speed "
         "perturbations, 1 min of normal walking (pilot trials 3-15 use variants of this protocol; see trial_info). Samples are at about 100 Hz (the motion capture frame rate): "
         "time_s, FrameNumber, the centre of pressure (m), force (N) and moment (N m) of force plate 1 (left belt) and 2 (right belt), and the sacrum, heel and toe markers (m). "
         "Axes are X right, Y up, Z backwards (the walking direction is -Z).",
         "The second file of each trial (samples/..._belts.csv, about 300 Hz) holds the measured belt speeds (m/s) and the first force plate's reference signal, on the same time origin. "
         "events lists the protocol events the operator or software recorded, in seconds on that origin (the time of the first row after the event comment).",
         "Per the authors' notes, the force plate data of trials 6-15 are unreliable (treadmill vibration, wooden blocks; trial_info says force_plate_data_valid=False) and the ankle torques of participant 9's trials are abnormal. "
         "Trials 1-2 were not part of the study, trials 37-39 do not exist, and trials with no participant on the treadmill (compensation trials) are not released. "
         "pert_value is empty because the perturbation signal (Zenodo 16064) is not part of this deposit."],
    differences=["Some trials have no protocol events in the record file (listed by an empty events cell), trials 3-5 have no record file (so no belts file), and a few trials have a frame rate below 100 Hz (frames dropped; time_s is the system TimeStamp, not frame number / 100). In trials 49-51 three of the five markers were not found in the file (trial_info markers_not_found); their columns are empty."],
    unit_note="10-minute treadmill trial of one participant",
)
stdlib.release(ID, "timeseries", D, meta, c, participants=part, participant_columns=pcols, extra_files=extra)
shutil.rmtree(TMP)
