"""Voudouris, Smeets, Fiehler & Brenner (2018), Journal of Vision: gaze and digit positions while reaching to grasp a glass.
Experiment 1: 10 people, glass of water or milk at eye height or hip height (2 blocks of 44 trials), grasp and pour into a
glass on the left. Experiment 2: 8 people, glass of water at hip height, then place on a coaster or pour into a glass on the
left or right (88 trials). One row per trial recording in the timeseries module; the samples are the authors' trial files.

Run from the site folder: python3 -I tools/std/convert/Voudouris2018_JVis.py
Raw (untrusted, read only): raw_extra/Voudouris2018_JVis/Data.zip from https://doi.org/10.5281/zenodo.1401845 (read in place;
ReadMe.rtf was read as text). Nothing in the archive is executed.
"""
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Voudouris2018_JVis"
MODULE = "timeseries"
RAW = stdlib.raw_dir(ID)
ck = stdlib.Checks()
z = zipfile.ZipFile(RAW / "Data.zip")
names = [n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith(".DS_Store")]

CH = ["gaze_h_deg", "gaze_v_deg", "fixed_marker_h_deg", "fixed_marker_v_deg", "index_h_deg", "index_v_deg", "thumb_h_deg", "thumb_v_deg",
      "index_x_mm", "index_y_mm", "index_z_mm", "thumb_x_mm", "thumb_y_mm", "thumb_z_mm", "fixed_x_mm", "fixed_y_mm", "fixed_z_mm",
      "left_eye_h_px", "left_eye_v_px", "right_eye_h_px", "right_eye_v_px"]
UNITS = ["deg"] * 8 + ["mm"] * 9 + ["px"] * 4
N_EXP = {"Experiment 1": 21, "Experiment 2": 17}   # Experiment 2 files have 37 columns, only the first 17 channels are documented


def load(name):
    rows = []
    for ln in z.read(name).decode("latin1").strip().split("\n"):
        p = ln.rstrip("\t").split("\t")
        rows.append([float(v) if v.strip() != "" else np.nan for v in p])
    return np.array(rows)


def stim(name):
    return [tuple(int(x) for x in ln.split()) for ln in z.read(name).decode().strip().split("\n")]


EXP2 = {1: ("place", "left"), 2: ("pour", "left"), 3: ("place", "right"), 4: ("pour", "right")}
tmp = Path(tempfile.mkdtemp(prefix="vou_"))
rows, extra = [], []
ndur, dts, widths = [], [], set()
cnt = {}
for exp in ("Experiment 1", "Experiment 2"):
    dirs = sorted({n.split("/")[1] for n in names if n.startswith(exp + "/") and n.count("/") >= 2})
    for dname in dirs:
        base = f"{exp}/{dname}"
        sl = dict(stim(f"{base}/stimlist"))
        trials = sorted((int(n.rsplit("_", 1)[1]), n) for n in names if n.startswith(base + "/trial_"))
        assert [t for t, _ in trials] == list(range(1, len(trials) + 1)) and set(sl) == {t for t, _ in trials}
        pid = f"E{exp[-1]}_{dname[:3]}"
        height = ({"High": "eye height", "Low": "hip height"}[dname.split("_")[1]] if exp == "Experiment 1" else "hip height")
        for t, n in trials:
            a = load(n)
            widths.add(a.shape[1])
            k = N_EXP[exp]
            data = a[:, 1:1 + k]
            tm = a[:, 0] / 1000.0
            dts.append(np.diff(tm))
            df = pd.DataFrame(data, columns=CH[:k])
            df.insert(0, "time_s", np.round(tm, 3))
            arc = f"samples/{exp.replace(' ', '')}_{dname}_trial{t:02d}.csv"
            path = tmp / Path(arc).name
            df.to_csv(path, index=False, float_format="%.8g", lineterminator="\r\n")
            extra.append((arc, str(path)))
            c = sl[t]
            if exp == "Experiment 1":
                content = {1: "water", 2: "milk"}[c]
                cond, info = f"{height}; {content}", f"height={height};content={content};stimlist_code={c}"
            else:
                task, side = EXP2[c]
                cond, info = f"{task} on the {side}", f"height={height};content=water;action={task};side={side};stimlist_code={c}"
            ndur.append(len(df))
            cnt[(exp, pid, dname)] = cnt.get((exp, pid, dname), 0) + 1
            rows.append(dict(
                dataset=ID, study=ID, experiment=exp, group="", subject=f"{exp[-1]}_{dname}", person=pid,
                subject_excluded_by_authors=False, age=np.nan, sex="", handedness="right", population="healthy adults",
                setting="lab", device="Optotrak 3020 (100 Hz) with Eyelink II eye tracker (500 Hz, merged to 100 Hz), bite-board head tracking",
                session=1, block=(f"glass at {height}" if exp == "Experiment 1" else "all trials"), trial=t, phase="test", time_s=np.nan,
                task="grasping and lifting", unit="recording", condition=cond, pert_type="none", pert_value=0, pert_unit="",
                feedback="normal", excluded_by_authors=False, recording=t, samples_file=arc,
                signal_kind="gaze direction and index finger and thumb positions (head-centred angles and Cartesian marker positions), fixed reference marker, eye-tracker pixel values",
                channels=";".join(CH[:k]), channel_units=";".join(UNITS[:k]), sample_rate_hz=100, n_samples=len(df),
                duration_s=round(float(tm[-1] - tm[0]) + 0.01, 3), events="",
                processing="eye-tracker values synchronized to the 100 Hz Optotrak samples and expressed in a head-centred frame by the authors; otherwise none stated",
                trial_info=info))

out = pd.DataFrame(rows)
for c in [c[1] for c in stdlib.columns_for(MODULE)]:
    if c not in out.columns:
        out[c] = np.nan
e1, e2 = out[out.experiment == "Experiment 1"], out[out.experiment == "Experiment 2"]
allt = np.concatenate(dts)
ck.published("Experiment 1 participants", e1.person.nunique(), 10, 0, "Experiment 1, Methods, Participants (P1-P10)")
ck.published("Experiment 2 participants", e2.person.nunique(), 8, 0, "Experiment 2 (eight participants)")
ck.published("Experiment 1 trials per block (22 water + 22 milk)", e1.groupby("subject").size().iloc[0], 44, 0,
             "Experiment 1, Experimental procedure (44 trials per height block)")
ck.published("Experiment 1 water trials per block", (e1[e1.subject == e1.subject.iloc[0]].condition.str.endswith("water")).sum(), 22, 0,
             "Experiment 1, Experimental procedure (each content presented 22 times per block)")
ck.published("Experiment 1 milk trials per block", (e1[e1.subject == e1.subject.iloc[0]].condition.str.endswith("milk")).sum(), 22, 0,
             "Experiment 1, Experimental procedure (each content presented 22 times per block)")
ck.published("Experiment 1 height blocks per participant", e1.groupby("person").subject.nunique().iloc[0], 2, 0,
             "Experiment 1, Experimental procedure (eye height and hip height blocks)")
ck.published("sampling interval of the combined signal (ms)", round(float(np.median(allt)) * 1000, 6), 10, 0, "Data analysis (combined 100 Hz signal)")
ck.published("sample rate written (Hz)", out.sample_rate_hz.median(), 100, 0, "Apparatus (Optotrak at 100 Hz)")
ck.internal("Experiment 1: 10 people x 2 blocks x 44 trials = 880 recordings; Experiment 2: 8 people x 88 = 704",
            len(e1) == 880 and len(e2) == 704)
ck.internal("Experiment 2: each of the 4 stimlist codes occurs 22 times per participant",
            bool((e2.groupby(["person", "condition"]).size() == 22).all()))
ck.internal("sample times are 10 ms apart in every recording", bool(np.allclose(allt, 0.01, atol=1e-6)))
ck.internal("trial files have 22 columns (Experiment 1) or 37 (Experiment 2)", widths == {22, 37} or widths == {22, 37, 38}, str(widths))
ck.internal("recordings are 1 s or longer", min(ndur) > 100, str(min(ndur)))

meta = dict(
    study=ID, paper=json.load(open(stdlib.HERE / "source/extra_studies.json")) and next(
        e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0],
    lab="Fiehler (Giessen) / Brenner & Smeets (VU Amsterdam)", data_url="https://doi.org/10.5281/zenodo.1401845",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor timeseries format: each trial file of the deposit is a samples CSV with named channels and time in "
            "seconds (the deposit's milliseconds divided by 1000), empty cells where the deposit has blanks; stimulus lists became the "
            "condition; participant folders replaced by codes; Experiment 2's undocumented columns 19-37 are not included.",
    unit_note="trial recording",
    how=["One row is one trial: a person (stood in front of a table, head held on a bite-board) looks at a start LED, reaches to grasp a "
         "glass with thumb and index finger of the right hand, and pours it into, or places it next to, another glass. Its samples file "
         "has the authors' 100 Hz signal: gaze direction (horizontal and vertical, degrees in a head-centred frame), the horizontal and "
         "vertical angle of the index finger and thumb markers, their Cartesian positions (mm, Optotrak frame) and a fixed reference marker.",
         "Experiment 1 (10 people): water or milk in the glass, at eye height or hip height, in two blocks of 44 trials. The folders are "
         "named High and Low; they are read as eye height and hip height. The order of the two blocks (counterbalanced) is not in the "
         "deposit, so each block is its own subject sequence of 44 trials (the same person). Experiment 2 (8 people, always water at hip "
         "height): 88 trials of placing the glass on a coaster or pouring into a glass, at 50 cm left or right.",
         "Channel names were assigned from the deposit's ReadMe (22 numbered columns). Its list skips two labels: columns 4 and 5 are "
         "read as the horizontal and vertical angle of the fixed reference marker (which the paper puts on the line of sight with the "
         "start LED; it varies little in the data) and columns 6 and 7 as the index finger's horizontal and vertical angle (these vary "
         "with the reach, and the index and thumb pairs sit close together at the start). Columns 19-22 are called pixels in the ReadMe "
         "but their values are small (tens); they are kept under that name. Sign conventions of the angles are not stated.",
         "Experiment 2's files have 37 columns, but the ReadMe describes the Experiment 1 layout of 22. Its first 17 data columns have the "
         "same pattern (angles, then digit and reference-marker coordinates, with a constant reference marker) and are released; the "
         "remaining columns are undocumented and omitted. The ReadMe says condition 4 in the Experiment 2 stimulus list is 'pour left', "
         "which repeats condition 2; it is read as pour right (each code occurs 22 times, as four conditions of 22 would). Experiment 2 "
         "participants P1-P4 also took part in Experiment 1 according to the paper, but the folders do not say which, so the two "
         "experiments' participants are coded separately. Age and sex per participant are not shared."],
    differences=["The paper (J Vis 18(8):16, 2018) says Experiment 2 had one block of 44 trials; the deposit has 88 trials per participant "
                 "(four stimulus codes of 22 each), so the 44 is not used as a check for Experiment 2.",
                 "The paper's gaze-bias results (critical fixation at first contact, contact asynchrony) are derived measures the deposit does "
                 "not contain, so they are not reproduced; the checks are the design numbers (participants, trials, sampling rate).",
                 "The paper text was read from the VU Amsterdam repository copy (publisher version, CC BY-NC-ND); the data are CC BY 4.0."],
)
stdlib.release(ID, MODULE, out, meta, ck, extra_files=extra)
shutil.rmtree(tmp, ignore_errors=True)
for c in ck.items:
    if c["kind"] == "published":
        print(f"  P {c['what'][:70]} {c['computed']:.4f} vs {c['reported']}")
