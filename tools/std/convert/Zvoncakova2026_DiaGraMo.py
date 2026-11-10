"""Zvoncakova et al. (2026), DiaGraMo: Czech online handwriting of children with and without handwriting disabilities (Zenodo 21236910).

Run from the site folder: python3 -I tools/std/convert/Zvoncakova2026_DiaGraMo.py
Raw file (untrusted data, read only): DiaGraMo-project.zip. Used: data/raw_data/handwriting_data/data_SVC/<id>/<id>_TSKnn(TA)-name_1.svc
(7 columns per sample) and data/raw_data/cognitive_test_data/cognitiveTestCSV2data.csv (only id, group, age, sexFemale are read).
The cognitive and school scores are not included (restricted instruments), nor the processed/clean JSON copies.
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zvoncakova2026_DiaGraMo"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
assert stdlib.sha256(RAW / "DiaGraMo-project.zip") == src["DiaGraMo-project.zip"]["sha256"]
z = zipfile.ZipFile(RAW / "DiaGraMo-project.zip")
P = "DiaGraMo-project/data/raw_data/"

cog = pd.read_csv(io.BytesIO(z.read(P + "cognitive_test_data/cognitiveTestCSV2data.csv")), sep=";", usecols=None, low_memory=False)
bad = [c for c in cog.columns if re.search(r"retain|consent|retention", c, re.I)]
assert not bad, bad     # a consent-limit column would have to be honoured
meta_by = {}
for _, r in cog.iterrows():
    age = r["age"]
    age = float(str(age).replace(",", ".")) if pd.notna(age) and str(age).strip() != "" else np.nan
    g = r["group"]
    meta_by.setdefault(str(r["id_participant"]), dict(age=age, group=g, female=r["sexFemale"]))

pat = re.compile(re.escape(P) + r"handwriting_data/data_SVC/([^/]+)/\1_TSK(\d+)\(TA\)-(.+)_(\d+)\.svc")
items = []
for n in z.namelist():
    m = pat.fullmatch(n)
    if m:
        items.append((m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), n))
    elif n.endswith(".svc"):
        raise AssertionError(n)
items.sort()
subs = []
for it in items:
    if it[0] not in subs:
        subs.append(it[0])
code = {s: f"{ID}_S{i + 1:04d}" for i, s in enumerate(subs)}
n_nometa = sum(s not in meta_by for s in subs)
print("participants with SVC but no row in the cognitive table:", n_nometa)

rows, files, nrec, rates = [], [], {}, []
nonmono, badpen, nonmono_ex, badpen_vals = 0, 0, [], set()
for sid, tsk, tname, ver, name in items:
    d = pd.read_csv(io.BytesIO(z.read(name)), sep=r"\s+", header=None, skiprows=1)
    n_decl = int(z.read(name).split(b"\n", 1)[0])
    assert d.shape[1] == 7 and len(d) == n_decl, (name, d.shape, n_decl)
    d.columns = ["x_mm", "y_mm", "time_s", "pen_status", "azimuth_deg", "altitude_deg", "pressure"]
    out = d[["time_s", "x_mm", "y_mm", "pen_status", "azimuth_deg", "altitude_deg", "pressure"]]
    dt = np.diff(d["time_s"].to_numpy())
    if (dt < 0).any():
        nonmono += 1
        nonmono_ex.append((name, int((dt < 0).sum()), int((dt == 0).sum()), float(dt.min())))
    if not d["pen_status"].isin([0, 1]).all():
        badpen += 1
        badpen_vals |= set(d["pen_status"].unique().tolist()) - {0, 1}
    nrec[sid] = nrec.get(sid, 0) + 1
    rec = nrec[sid]
    t = d["time_s"].to_numpy()
    rates.append((len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else np.nan)
    fn = f"samples/{code[sid]}_rec{rec:02d}_TSK{tsk:02d}_{tname}.csv"
    files.append((fn, out.to_csv(index=False, lineterminator="\r\n", float_format="%.8g")))
    m = meta_by.get(sid, dict(age=np.nan, group=np.nan, female=np.nan))
    grp = {1.0: "handwriting disabilities", 0.0: "typical handwriting development"}.get(m["group"], "") if pd.notna(m["group"]) else ""
    rows.append(dict(
        dataset=ID, study="Zvoncakova et al. 2026 DiaGraMo", experiment="DiaGraMo", group=grp, subject=sid, person=sid,
        subject_excluded_by_authors=False, age=m["age"], sex={0: "M", 1: "F"}.get(int(m["female"]), "") if pd.notna(m["female"]) else "",
        handedness="", population=grp, setting="lab", device="Wacom Cintiq 16 display tablet with digital pen", session=1,
        block=f"TSK{tsk:02d}", trial=rec, phase="test", time_s="", task="handwriting and drawing", unit="recording",
        condition=tname, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=rec, samples_file=fn, signal_kind="pen position, pen state, azimuth, altitude and pressure on a tablet",
        channels="x;y;pen_status;azimuth;altitude;pressure", channel_units="mm;mm;0/1;deg;deg;device units",
        sample_rate_hz=round(float(rates[-1]), 2), n_samples=len(d), duration_s=float(t[-1] - t[0]), events="",
        processing="none (raw SVC file)", trial_info=f"task_number={tsk};task_name={tname}"))
df = pd.DataFrame(rows)
rates = np.array(rates)
persons = df.drop_duplicates("subject")
checks.internal("every SVC file kept, none lost", len(df) == len(items) == 4000, str(len(df)))
print("NONMONO", nonmono, nonmono_ex[:5]); print("BADPEN", badpen, sorted(badpen_vals)[:10])
checks.internal("time axis never runs backward in any recording", nonmono == 0, f"{nonmono} recordings with a backward step")
checks.internal("pen_status holds only 0 and 1", badpen == 0, f"{badpen} recordings with other values")
checks.published("number of graphomotor and writing tasks", df["condition"].nunique(), 16, 0, "Zenodo record abstract: 16 graphomotor and writing tasks")
checks.published("average sampling frequency in Hz (abstract: 167 Hz; here mean of samples per second over recordings)", np.nanmean(rates), 167, 5, "Zenodo record abstract / README (average frequency of 167 Hz)")
n_p = persons.shape[0]
n_dys = int((persons["group"] == "handwriting disabilities").sum())
print("participants with SVC:", n_p, "with disabilities:", n_dys, "mean rate", np.nanmean(rates))

diffs = [
    f"The record states 276 children with 161 diagnosed with handwriting disabilities; the raw SVC files here come from {n_p} children, {n_dys} coded group 1 (handwriting disabilities) in the cognitive table, so the participant counts of the abstract were not used as checks (the deposit's version 2 may count after exclusions).",
    "Cognitive, school-mark and questionnaire scores (WJ-IV, RCFT, BACH, DIB, HPSQ-C) are not included: the instrument items are restricted by copyright and the table has 760 columns; only age, sex and the diagnostic group are kept. Age is the authors' months-since-birth divided by 12 (the deposit coarsened test dates to a three-month window).",
    "Units are inferred, not stated in the deposit: x and y are in the SVC file's first two columns (values of tens to hundreds, taken as millimetres on the 16-inch tablet), time in seconds, azimuth and altitude in degrees, pressure in the tablet's raw units; pen_status 1 is pen on the surface, 0 in air (standard SVC convention).",
    f"{n_nometa} children have raw handwriting files but no row in the cognitive table, so their age, sex and group are empty. The deposit's own test-version-2 dates and the participant folder names were not carried over.",
]
meta = dict(
    study="Zvoncakova et al. 2026",
    paper=dict(authors="Zvoncakova K, Mekyska J, Klocek A, Mucha J, Galaz Z",
               year=2026, title="Multimodal Czech online handwriting and cognitive data from children with and without handwriting disabilities",
               venue="Zenodo dataset (DiaGraMo), version 2; see also Mekyska et al. 2023, Lecture Notes in Computer Science 14319:21-35", doi="10.5281/zenodo.21236910"),
    lab="Czech Academy of Sciences Institute of Psychology / Brno University of Technology (Mekyska)", data_url="https://doi.org/10.5281/zenodo.21236910",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Raw SVC files re-encoded as CSV recordings (columns reordered so time comes first; values unchanged); participant folder codes replaced; only age, sex and group kept from the cognitive table.",
    how=["One row is one task recording (one of 16 graphomotor and writing tasks, named in condition and task_name) of one child; its samples are in samples/<subject code>_rec<NN>_TSK<NN>_<task>.csv with columns time_s, x_mm, y_mm, pen_status, azimuth_deg, altitude_deg, pressure.",
         "group and population are the authors' diagnostic criterion (1 = handwriting disabilities, 0 = typical handwriting development); sex is from the authors' sexFemale (1 = female). Not every child did every task: recordings exist only for tasks the child completed (tasks 3, 4, 15, 16 are the dictation and copy tasks with fewer children).",
         "Samples are the raw tablet output at an average of about 167 Hz (irregular time steps); in-air samples (pen_status 0) are kept. Nothing was resampled or filtered."],
    unit_note="recording (one task)", differences=diffs)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=files)
