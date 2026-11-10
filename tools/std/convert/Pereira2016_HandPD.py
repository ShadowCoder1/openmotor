"""Pereira et al. (2016), NewHandPD: handwritten dynamics from the BiSP smart pen (six channels, header says 1000 Hz) of 66 exam sessions
(35 healthy, 31 patients with Parkinson's disease), twelve exams each: 4 spirals, 4 meanders, a circle on paper, a circle in the air and
diadochokinesis with each hand. Published with the HandPD / NewHandPD home page (https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/).

Raw files (untrusted data, read only): raw_extra/Pereira2016_HandPD/HealthySignal.zip, PatientSignal.zip (Signal/*.txt: a '#<meta>' header then
tab-separated rows of 6 numbers), NewSpiral.csv (the authors' per-image table, used only to check ages and the codes of gender and hand).
NOT carried: the header fields Person_ID_Number (used only to group the files of a person), Surename, Forename, Date, Time, Weight, Height,
Notice, Comment; the images (spirals, meanders, circles) are not part of this conversion.
Run from the site folder: python3 -I tools/std/convert/Pereira2016_HandPD.py
"""
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Pereira2016_HandPD"
RD = stdlib.raw_dir(ID)
TMP = RD / "tmp_samples"
TMP.mkdir(exist_ok=True)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RD / "SOURCES.json"))}
for f in ("HealthySignal.zip", "PatientSignal.zip", "NewSpiral.csv"):
    checks.internal(f"{f} matches the downloaded hash", stdlib.sha256(RD / f) == src[f]["sha256"])

EXAM = {"c_spira": "spiral", "c_spiral": "spiral", "d_mea": "meander", "a_circ_p": "circle on paper", "b_circ_a": "circle in the air",
        "e_diador": "diadochokinesis, right hand", "f_diadol": "diadochokinesis, left hand"}
ORDER = {"spiral": 0, "meander": 1, "circle on paper": 2, "circle in the air": 3, "diadochokinesis, right hand": 4, "diadochokinesis, left hand": 5}
CH = "microphone;finger_grip;axial_pressure_of_ink_refill;tilt_acceleration_x;tilt_acceleration_y;tilt_acceleration_z"
UNITS = ";".join(["recorded value, unit not documented"] * 6)

recs = []
for fz, grp in (("HealthySignal.zip", "healthy"), ("PatientSignal.zip", "Parkinson's disease")):
    z = zipfile.ZipFile(RD / fz)
    for info in sorted(z.infolist(), key=lambda i: i.filename):
        fn = info.filename
        if not (fn.startswith("Signal/") and "/._" not in fn and fn.endswith(".txt")):
            continue
        txt = z.read(info).decode("latin1")
        lines = txt.splitlines()
        n = 0
        meta = {}
        while n < len(lines) and lines[n].startswith("#"):
            mm = re.match(r"#<(\w+)>(.*)</\w+>", lines[n])
            if mm:
                meta[mm.group(1)] = mm.group(2)
            n += 1
        d = pd.read_csv(io.StringIO("\n".join(lines[n:])), sep="\t", header=None)
        assert d.shape[1] == 6 and d.notna().all().all(), fn
        lab = re.search(r"-([HP]\d+)\.txt$", fn)
        base = fn.split("/")[-1]
        recs.append(dict(grp=grp, file=base, label=lab.group(1) if lab else None, pid=meta["Person_ID_Number"], age=int(meta["Age"]),
                         gender=int(meta["Gender"]), hand=int(meta["Writing_Hand"]), obj=meta["Object"], oi=int(meta["Object_Index"]),
                         fs=int(meta["Samplerate"]), pen=meta["Pen"], data=d))
R = pd.DataFrame([{k: v for k, v in r.items() if k != "data"} for r in recs])
# file with no index in its name (sigMea1-28): take the index of the other files of the same person
for i, r in R[R.label.isna()].iterrows():
    labs = R[(R.pid == r.pid) & R.label.notna()].label.unique()
    assert len(labs) == 1
    R.loc[i, "label"] = labs[0]
checks.internal("one pen, one sample rate (1000 Hz in every header), six columns in every file", R.pen.nunique() == 1 and (R.fs == 1000).all())
R["kind"] = R.obj.map(EXAM)
assert R.kind.notna().all()
R["ord"] = R.kind.map(ORDER)
R["labnum"] = R.label.str[1:].astype(int)
R["gsort"] = (R.grp != "healthy").astype(int)

# codes of Gender and Writing_Hand are not documented: infer them against the authors' per-image table
sp = pd.read_csv(RD / "NewSpiral.csv")
sp["label"] = sp.IMAGE_NAME.str.extract(r"-([HP]\d+)\.jpg")[0]
c = sp.groupby("label").agg(age_c=("AGE", "first"), sex_c=("GENDER", "first"), hand_c=("RIGH/LEFT-HANDED", "first")).reset_index()
per_label = R.groupby("label").agg(age=("age", "first"), gender=("gender", "first"), hand=("hand", "first")).reset_index().merge(c, on="label")
agree = per_label[per_label.age == per_label.age_c]
tab_g = pd.crosstab(agree.gender, agree.sex_c)
tab_h = pd.crosstab(agree.hand, agree.hand_c)
print(tab_g, tab_h, len(agree), len(per_label))
checks.internal("codes of Gender and Writing_Hand compared with the authors' table (not decoded: they disagree in some persons)", True,
                f"gender {tab_g.to_dict()}, hand {tab_h.to_dict()}")
SEX = {1: "M", 2: "F"}
HAND = {1: "right", 2: "left"}

# ------------------------------------------------------------------ table and sample files
sub_order = R.sort_values(["gsort", "labnum", "ord", "oi", "file"]).index
rows, extra = [], []
cnt = {}
total = 0
for i in sub_order:
    r = R.loc[i]
    code = f"{'H' if r.gsort == 0 else 'P'}{r.labnum:02d}"
    cnt[code] = cnt.get(code, 0) + 1
    k = cnt[code]
    d = recs[i]["data"]
    t = np.arange(len(d)) / float(r.fs)
    out = pd.DataFrame({"time_s": t})
    for j, nm in enumerate(CH.split(";")):
        out[nm] = d[j].to_numpy()
    fp = TMP / f"{code}_{k:02d}.csv"
    out.to_csv(fp, index=False, float_format="%.7g", lineterminator="\r\n")
    total += len(d)
    rows.append(dict(
        dataset=ID, study=ID, experiment="NewHandPD", group=r.grp, subject=f"{code}|{r.pid}", person=r.pid, subject_excluded_by_authors=False,
        age=r.age, sex="", handedness="", population=r.grp, setting="clinic",
        device="BiSP smart pen (Pentrics Alberich1) with six sensors", session=1, block=r.kind, trial=k, phase="test",
        time_s="", task="handwriting and drawing", unit="recording", condition=r.kind, pert_type="none", pert_value=0, pert_unit="",
        feedback="normal", excluded_by_authors=False, recording=k, samples_file=f"samples/{code}_{k:02d}.csv",
        signal_kind="smart pen signals: microphone, finger grip, axial pressure of the ink refill, tilt and acceleration in x, y, z",
        channels=CH, channel_units=UNITS, sample_rate_hz=int(r.fs), n_samples=len(d), duration_s=round(len(d) / float(r.fs), 3), events="",
        processing="none stated by the authors; time_s is the row number divided by the sample rate in the header (1000 Hz)",
        trial_info=f"source_file_index={r.label}; exam_object={r.obj}; exam_object_index={r.oi}; gender_code={r.gender}; writing_hand_code={r.hand}"))
    extra.append((f"samples/{code}_{k:02d}.csv", fp))
df = pd.DataFrame(rows)

# ------------------------------------------------------------------ checks against the dataset description
H, P = "healthy", "Parkinson's disease"
PG = "HandPD / NewHandPD home page"
checks.published("signal files of healthy individuals", (R.grp == H).sum(), 420, 0, f"{PG}, NewHandPD (420 signals of healthy individuals)")
checks.published("signal files of patients", (R.grp == P).sum(), 372, 0, f"{PG}, NewHandPD (372 signals from patients)")
lab = R.groupby(["grp", "label"]).agg(age=("age", "first"), gender=("gender", "first"), hand=("hand", "first")).reset_index()
lab = lab.merge(per_label[["label", "age_c"]], on="label", how="left")
checks.published("healthy individuals", (lab.grp == H).sum(), 35, 0, f"{PG}, NewHandPD")
checks.published("patients", (lab.grp == P).sum(), 31, 0, f"{PG}, NewHandPD")
# ages of the persons with an unambiguous header (the page gives the group means)
for g, m, s in ((H, 44.05, 14.88), (P, 57.83, 7.85)):
    a = lab[lab.grp == g].age
    checks.published(f"mean age, {g}", a.mean(), m, 1.0, f"{PG}, NewHandPD group description (see differences)")
    checks.published(f"SD of age, {g}", a.std(), s, 1.0, f"{PG}, NewHandPD group description (see differences)")
checks.internal("every sample line kept", total == int(sum(len(r["data"]) for r in recs)), f"{total} lines in {len(df)} recordings")
checks.internal("792 recordings = 420 + 372", len(df) == 792)
pers = df.groupby("person").size()
print(len(pers), "persons by header id;", (R.groupby("label").pid.nunique() > 1).sum(), "labels with two header ids")
print(lab.groupby("grp").size())

s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
p = s["papers"][1]
meta = dict(
    study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
    license="none stated", license_url="",
    changes=("Converted the tab-separated pen signals into one CSV of samples per recording (samples/...csv, values re-encoded to 7 significant "
             "digits) plus the OpenMotor table; removed the header fields that identify people (ID number, names, date and time of the exam, "
             "weight, height) and kept age and the raw gender and writing-hand codes; replaced file names with codes. The images of spirals, meanders and circles and the "
             "authors' image-feature tables were not converted."),
    how=["One row is one exam of one session: a spiral (4 per session), a meander (4), a circle drawn on paper, a circle drawn in the air, or "
         "diadochokinesis with the right or the left hand. Samples are in samples/<session>_<n>.csv: time_s then the six channels of the BiSP smart pen "
         "in the order the dataset description lists them (microphone, finger grip, axial pressure of the ink refill, tilt and acceleration in x, y and z). "
         "Units of the channels are not documented. time_s is the row number divided by the header's sample rate (1000 Hz).",
         "A person is one header ID: in 5 patients the same person has two or three sessions (same header ID, a different index in the file name, the "
         "authors' P1/P4, P9/P10, P15/P17, P2/P25/P26...). Sex and writing hand are coded 1/2 in the headers without a codebook; they are NOT decoded "
         "(sex and handedness are empty) and the raw codes are in trial_info as gender_code and writing_hand_code. Matched against the authors' NewSpiral.csv, 1 mostly means male / right "
         "and 2 female / left, but not for every person, so the codes cannot be relied on.",
         "Group is healthy or Parkinson's disease; severity and medication are not in the deposit."],
    unit_note="exam (smart pen signals)",
    differences=["Nothing in the deposit documents the channel order or units; the order is the one given in the dataset description (secondary source "
                 "quoting it: Sequence-based Dynamic Handwriting Analysis for PD detection, arXiv 2101.09461, section 3.2). The home page numbers (35 healthy, "
                 "31 patients, 420 and 372 signals) reproduce. The file labels are not all reliable: one healthy session holds a diadochokinesis file that "
                 "belongs to a different person (header age 50, a different ID), and one meander file is named sigMea1-28 without a group letter. The "
                 "mean ages reproduce within 1 year; the deposit's per-session ages may differ from the page by session counting.",
                 "License: the home page states none. The signals belong to the SIBGRAPI 2016 paper (the DOI listed in extra_studies, 10.1109/sibgrapi.2016.054, "
                 "is not the one the UNESP repository gives, 10.1109/SIBGRAPI.2016.51); the first listed paper (CMPB 2016) uses the image dataset."],
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
