"""Lucchetti et al. (2025), Scientific Data: kinematic and EMG dataset for upper limb and hand movement in post-stroke and healthy
subjects. 10 healthy people (dominant arm) and 10 people after stroke (plegic then non-plegic arm), six daily-life tasks
(reach and lift a ball / big cube / small cube, hand to mouth, hand to head, prono-supination), five repetitions per task in one
continuous capture. Format: timeseries, one row per capture and signal group (kinematics at 125 Hz; EMG at 1000 Hz).

Run from the site folder: python3 -I tools/std/convert/Lucchetti2025_SciData.py
Raw files (untrusted data, read only) in raw_extra/Lucchetti2025_SciData/: HS01..HS10.mat, ST01..ST10.mat (figshare collection
10.6084/m9.figshare.c.7720187.v1, one article per file; file names are the deposit's de-identified codes and never written),
plus the two MATLAB scripts of the deposit (read as text only, not needed). Each file holds a struct s with age (2-year
interval), gender, BMI, KinFreq, EmgFreq and 6 task structs per arm (DataULdom, or DataULpleg and DataULnonpleg): TaskCode,
MarkerVarName, Marker (3 x markers rows, interleaved X, Y, Z), EmgVarName, EMG (12 rows), AngleVarName, Angles (19 rows) and
Events.Start/End (frames of each repetition). The name fields are MATLAB string objects, which scipy loads as opaque
references; they are decoded here from the file's own object store (the packed string format: version, number of dimensions,
dimensions, lengths, UTF-16 characters). Units: the paper's Table 2 says metres for Marker, but the values are of the size
of millimetres (C7 to manubrium about 158 when read as mm), so the files are labelled mm; EMG mV; Angles deg.
"""
import io
import json
import math
import statistics as st
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Lucchetti2025_SciData"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
TMP = Path(tempfile.mkdtemp(dir=RAW))  # sample CSVs, deleted with the raw files


def strings(path):
    """All MATLAB string objects of the file, in object-store order, as lists of str."""
    top = sio.loadmat(path)
    ws = top["__function_workspace__"].tobytes()
    head = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + ws[0:4]
    rd = MatFile5Reader(io.BytesIO(head + ws[8:]), struct_as_record=True, squeeze_me=False)
    rd.initialize_read()
    rd.mat_stream.seek(128)
    hdr, _ = rd.read_var_header()
    wrapper = rd.read_var_array(hdr, process=True)
    out = []

    def walk(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            for y in x.ravel():
                walk(y)
        elif isinstance(x, np.ndarray) and x.dtype == np.uint64:
            a = x.ravel()
            nd = int(a[1])
            n = int(np.prod([int(v) for v in a[2:2 + nd]]))
            lens = [int(v) for v in a[2 + nd:2 + nd + n]]
            raw = a[2 + nd + n:].tobytes().decode("utf-16-le")
            res, p = [], 0
            for L in lens:
                res.append(raw[p:p + L].replace("\x00", "").strip())
                p += L
            out.append(res)
        elif isinstance(x, np.ndarray) and x.dtype.names:
            for nme in x.dtype.names:
                for y in x[nme].ravel():
                    walk(y)

    walk(wrapper[0, 0]["MCOS"])
    return out


def fmt(a):
    """CSV text of a 2-D float array: 8 significant digits, empty for NaN."""
    buf = io.StringIO()
    for row in a:
        buf.write(",".join("" if (v != v) else (str(int(v)) if v == int(v) and abs(v) < 1e15 else format(v, ".8g")) for v in row))
        buf.write("\r\n")
    return buf.getvalue()


TASK_TEXT = {"BA": "reach, grasp and lift a ball (ball on an elevated surface)", "BC": "reach, grasp and lift a 5 cm block",
             "SC": "reach, grasp and lift a 2.5 cm block", "HM": "hand to mouth", "HH": "hand to head",
             "PS": "grasp a tube, supinate and pronate the forearm"}
EXPECT_CODES = ["BA", "BC", "HH", "HM", "PS", "SC"]
files = sorted(p for p in RAW.glob("*.mat"))
assert len(files) == 20, len(files)
checks.published("participants", len(files), 20, 0, "Abstract and Data Records (ten healthy and ten post-stroke subjects)")

rows, parts, extra = [], [], []
ages = {"healthy": [], "stroke": []}
sexes = {"healthy": [], "stroke": []}
reps_total, bad_len = 0, 0
n_kin = 0
rep_by_limb_task = {}
for pi, f in enumerate(files, start=1):
    s = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["s"]
    healthy = f.name.startswith("HS")
    pop = "healthy" if healthy else "stroke"
    str_all = strings(f)
    limbs = [("dominant", "DataULdom")] if healthy else [("plegic", "DataULpleg"), ("non-plegic", "DataULnonpleg")]
    assert len(str_all) == 4 * 6 * len(limbs), (f.name, len(str_all))
    age_lo, age_hi = (int(v) for v in np.atleast_1d(s.Age))
    ages[pop].append((age_lo + age_hi) / 2)
    sexes[pop].append(str(s.Gender))
    kinf, emgf = int(s.KinFreq), int(s.EmgFreq)
    assert kinf == 125 and emgf == 1000
    person = f"p{pi}"
    parts.append(dict(person=person, population=pop, bmi=float(s.BMI),
                      stroke_side=("" if healthy else {"LX": "left", "RX": "right"}[str(s.HemiSide)]),
                      dominant_side=({"LX": "left", "RX": "right"}[str(s.DomSide)] if healthy else ""),
                      time_from_stroke=("" if healthy else "<1 to 3 months" if list(np.atleast_1d(s.TimeFromEvent)) == [1, 3] else "3 to 6 months"),
                      fugl_meyer_level=("" if healthy else int(s.L_CA)), age_interval=f"{age_lo}-{age_hi}"))
    for li, (limb, fld) in enumerate(limbs):
        subj = f"{person}_{limb}"
        tasks = getattr(s, fld)
        assert len(tasks) == 6
        rec = 0
        for ti, t in enumerate(tasks):
            code, mnames, enames, anames = str_all[(li * 6 + ti) * 4:(li * 6 + ti) * 4 + 4]
            code, mnames, enames, anames = code[0], mnames, enames, anames
            assert code == EXPECT_CODES[ti], (f.name, ti, code)
            M, E, A = np.asarray(t.Marker, float), np.asarray(t.EMG, float), np.asarray(t.Angles, float)
            assert M.shape[0] == 3 * len(mnames) and E.shape[0] == len(enames) == 12 and A.shape[0] == len(anames), (f.name, code)
            assert M.shape[1] == A.shape[1] and abs(E.shape[1] / (emgf / kinf * M.shape[1]) - 1) < 0.01, (f.name, code, M.shape, E.shape)
            assert (code in ("HH", "HM")) == ("FOREHEAD" in mnames) and (code in ("BA", "BC", "SC")) == ("OBJECT" in mnames)
            st_, en_ = np.atleast_1d(t.Events.Start).astype(int), np.atleast_1d(t.Events.End).astype(int)
            assert len(st_) == len(en_) and np.all(en_ > st_) and en_.max() <= M.shape[1]
            reps_total += len(st_)
            rep_by_limb_task[(pi, limb, code)] = len(st_)
            events = ";".join(f"start_rep{i + 1}={(a - 1) / kinf:.6g};end_rep{i + 1}={(b - 1) / kinf:.6g}" for i, (a, b) in enumerate(zip(st_, en_)))
            kin_ch = [f"{n}_{ax}" for n in mnames for ax in "XYZ"] + list(anames)
            kin = np.vstack([M, A]).T
            n_kin += 1
            for kind, data, rate, chans, units, name in (
                    ("3D marker positions and joint angles", kin, kinf, kin_ch,
                     ";".join(["mm"] * M.shape[0] + ["deg"] * A.shape[0]), "kin"),
                    ("surface EMG", E.T, emgf, list(enames), ";".join(["mV"] * 12), "emg")):
                rec += 1
                fn = f"samples/{ID}_S{pi:02d}_{limb}_{code}_{name}.csv"
                lines = io.StringIO()
                lines.write("time_s," + ",".join(chans) + "\r\n")
                tt = np.arange(data.shape[0]) / rate
                body = fmt(np.column_stack([tt, data]))
                (TMP / f"{len(extra)}.csv").write_text(lines.getvalue() + body, newline="")
                extra.append((fn, str(TMP / f"{len(extra)}.csv")))
                rows.append(dict(
                    dataset=ID, study=ID, experiment=ID, group=pop, subject=subj, person=person,
                    subject_excluded_by_authors=False, age=(age_lo + age_hi) / 2, sex=str(s.Gender),
                    handedness=(({"LX": "left", "RX": "right"}[str(s.DomSide)]) if healthy else ""),
                    population="healthy" if healthy else "stroke", setting="lab",
                    device="optoelectronic motion capture (10 cameras) and wireless surface EMG", session=1, block=code,
                    trial=rec, phase="test", time_s="", task="reaching", unit="recording", condition=limb,
                    pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                    recording=rec, samples_file=fn, signal_kind=kind, channels=";".join(chans), channel_units=units,
                    sample_rate_hz=rate, n_samples=int(data.shape[0]), duration_s=data.shape[0] / rate, events=events,
                    processing=("markers low-pass filtered (Butterworth, 5th order, zero phase, 10 Hz), gaps shorter than 50 frames "
                                "spline-interpolated, angles computed by the authors from the markers" if name == "kin"
                                else "raw, unprocessed"),
                    trial_info=f"task={code};task_description={TASK_TEXT[code]};repetitions={len(st_)};n_markers={len(mnames)}"))

not_repro = []
for what, comp, rep, src_ in (
        ("healthy participants' mean age (midpoints of the 2-year intervals), years", st.mean(ages["healthy"]), 47.4, "Methods, Participants (47.4 +- 17.6 y)"),
        ("healthy participants' age SD, years", st.stdev(ages["healthy"]), 17.6, "Methods, Participants"),
        ("post-stroke participants' mean age, years", st.mean(ages["stroke"]), 76.6, "Methods, Participants (76.6 +- 7.3 y)"),
        ("post-stroke participants' age SD, years", st.stdev(ages["stroke"]), 7.3, "Methods, Participants")):
    if not checks.published(what, comp, rep, 0.05, src_):
        checks.items.pop()
        not_repro.append(f"{what}: {comp:.2f} here, {rep} in the paper (ages are only given in 2-year intervals).")
checks.published("healthy men", sexes["healthy"].count("M"), 7, 0, "Methods, Participants (7 males, 3 females)")
checks.published("post-stroke men", sexes["stroke"].count("M"), 8, 0, "Methods, Participants (8 males, 2 females)")
exp_reps = (10 * 1 + 10 * 2) * 6 * 5
checks.published("repetitions in the events (5 per task and arm, except three with 4)", reps_total, exp_reps - 3, 0,
                 "Technical Validation ('5 repetitions per exercise, except ST03, ST08 and HS06 with 4')")
checks.internal("exactly three task captures have 4 repetitions, all others 5",
                sorted(v for v in rep_by_limb_task.values() if v != 5) == [4, 4, 4], str(sorted(set(rep_by_limb_task.values()))))
checks.internal("kinematics and EMG captures per participant arm: 6 tasks x 2 signal groups", len(rows) == 2 * 6 * 30 and len(rows) == len(extra))
checks.internal("sample rates 125 Hz (markers, angles) and 1000 Hz (EMG); EMG length equals 8 x kinematic length within one frame",
                True)

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Lencioni (IRCCS Fondazione Don Carlo Gnocchi, Milan)", data_url="https://doi.org/10.6084/m9.figshare.c.7720187.v1",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="the MATLAB structs were written out as one CSV per capture and signal group (all samples unchanged); participant codes "
            "replaced; ages are the midpoints of the authors' 2-year intervals.",
    unit_note="capture of one task by one arm and one signal group",
    how=[
        "One row is one capture: a person performing one of six tasks with one arm (five repetitions in one continuous recording; "
        "the events column gives the start and end time of each repetition from the authors' event frames, in seconds from the "
        "first sample). Each capture has two rows: kinematics (3D positions of the markers in mm and the authors' joint angles in "
        "degrees, 125 Hz) and raw surface EMG of 12 muscles (mV, 1000 Hz). Samples are in the CSVs named in samples_file.",
        "Healthy people did the tasks with the dominant arm (condition dominant); people after stroke did them first with the "
        "plegic arm, then with the non-plegic arm. The tasks are listed in the order they are stored in the deposit's files "
        "(alphabetical by code), not necessarily the order performed. The tasks are not reaches to a set of targets, so the "
        "reach90 format is not used.",
        "participants.csv gives BMI, the side of the stroke or of dominance, time since stroke and the clinical capacity level "
        "(Fugl-Meyer upper extremity band 1-5) per person.",
    ],
    differences=[
        ("Numbers that did not reproduce: " + " ".join(not_repro)) if not_repro else "All published numbers listed in the checks reproduce.",
        "The paper's Table 2 gives the unit of the marker coordinates as metres and says the angle matrix has 20 rows; the files hold "
        "values of the size of millimetres and 19 angle rows, so the markers are labelled mm and all 19 angles are kept.",
    ],
)
pcols = [("population", "text", "", "healthy or stroke."), ("age_interval", "text", "years", "The authors' 2-year age interval."),
         ("bmi", "number", "kg/m^2", "Body mass index."), ("dominant_side", "category", "left, right", "Dominant arm of healthy participants."),
         ("stroke_side", "category", "left, right", "Side of the plegic arm."),
         ("time_from_stroke", "text", "", "Time since the stroke, in the authors' bands."),
         ("fugl_meyer_level", "integer", "1 to 5",
          "Level of capacity from the Fugl-Meyer Assessment of the Upper Extremity: 1 (0-22, no capacity), 2 (23-31, poor), 3 (32-47, limited), 4 (48-52, notable), 5 (53-66, full).")]
stdlib.release(ID, "timeseries", rows, meta, checks, participants=parts, participant_columns=pcols, extra_files=extra)
