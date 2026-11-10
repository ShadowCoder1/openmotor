"""Sigcha et al. (2023), Front Neurol 14:1326640 (Monipar): smartwatch accelerometer recordings of 21 people with Parkinson's disease
and 7 healthy controls, one row per weekly recording (the eight Monipar exercises in one recording), samples in samples/*.csv.

Run from the site folder: python3 -I tools/std/convert/Sigcha2023_FrontNeurol_Monipar.py
Raw files (untrusted data, read only) in raw_extra/Sigcha2023_FrontNeurol_Monipar/, from https://zenodo.org/records/8104853:
  MONIPAR_PD_SUPERVISED.mat, MONIPAR_PD_REMOTE.mat, MONIPAR_HEALTHYCONTROL.mat   cell arrays (rows subjects, columns weeks) of
      N x 5 matrices: time stamp (ms), accelerometer x, y, z (m/s^2), exercise label (Monipar_README.txt)
  MONIPAR_PD_*_TREMOR_LABEL.mat   per-sample tremor label (-1 not labelled, 0, 1, 2), same cells and lengths
  MONIPAR_SUBJECTS_DATA.xlsx      participant table (sex, age as a formula year-of-assessment minus birth year, H&Y, ...)
"""
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Sigcha2023_FrontNeurol_Monipar"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

# ---- participant table
wb = openpyxl.load_workbook(RAW / "MONIPAR_SUBJECTS_DATA.xlsx")
ws = wb["Subject data"]
subj = {}
for r in list(ws.iter_rows(values_only=True))[1:]:
    sid, sex, age, onset, diag, hy, weekly, tremor = r[:8]
    if isinstance(age, str):
        m = re.fullmatch(r"=(\d{4})-(\d{4})", age)
        age = int(m.group(1)) - int(m.group(2))
    num = lambda v: v if isinstance(v, (int, float)) else None  # noqa: E731
    subj[sid] = dict(sex=sex, age=age, onset=num(onset), diag=num(diag), hy=num(hy), weekly=weekly,
                     tremor={"Yes": "yes", "No": "no"}.get(tremor, ""))

GROUPS = [("PD_SUPERVISED", "SUPERVISED_RAWDATA", "SUPERVISED", "Supervised", "SUPERVISED_TREMOR_LABEL", "Parkinson's disease"),
          ("PD_REMOTE", "REMOTE_RAWDATA", "REMOTE", "Remote", "REMOTE_TREMOR_LABEL", "Parkinson's disease"),
          ("HEALTHYCONTROL", "HEALTHYCONTROL_RAWDATA", "HEALTHY_CTRL", "Healthy control", None, "healthy")]
EXN = {0: "postural transition", 1: "rest", 2: "postural tremor of the hands", 3: "hands to chest", 4: "finger tapping, thumb and index",
       5: "rapid hand open-close", 6: "pronation-supination", 7: "arising from a chair", 8: "gait"}
CH = ["acc_x", "acc_y", "acc_z", "exercise_label", "tremor_label"]

rows, extra, parts = [], [], []
seen = {}
hours = {}
nrec = {}
for fname, key, prefix, gname, tkey, pop in GROUPS:
    raw = sio.loadmat(RAW / f"MONIPAR_{fname}.mat")[key]
    lab = None
    if tkey:
        lab = sio.loadmat(RAW / f"MONIPAR_{fname}_TREMOR_LABEL.mat")[tkey]
        checks.internal(f"{gname}: tremor label array has the shape of the raw array", lab.shape == raw.shape)
    for i in range(raw.shape[0]):
        sid = f"{prefix}{'_' if prefix != 'HEALTHY_CTRL' else ''}{i + 1}"
        info = subj[sid]
        n_here = 0
        for j in range(raw.shape[1]):
            c = raw[i, j]
            if not c.size:
                continue
            n_here += 1
            assert c.shape[1] == 5
            tl = np.full(len(c), np.nan)
            if lab is not None and lab[i, j].size:
                l = np.asarray(lab[i, j]).reshape(-1)
                assert len(l) == len(c), (sid, j, len(l), len(c))
                tl = l.astype(float)
            t = c[:, 0] / 1000.0
            df = pd.DataFrame({"time_s": t, "acc_x": c[:, 1], "acc_y": c[:, 2], "acc_z": c[:, 3],
                               "exercise_label": c[:, 4].astype(int), "tremor_label": pd.array(np.where(np.isnan(tl), -99, tl).astype(int), dtype="Int64")})
            if np.isnan(tl).all():
                df["tremor_label"] = pd.array([pd.NA] * len(df), dtype="Int64")
            fn = f"samples/{sid}_week{j + 1:02d}.csv"
            extra.append((fn, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
            hours[gname] = hours.get(gname, 0) + (t[-1] - t[0]) / 3600
            rows.append(dict(
                dataset=ID, study=ID, experiment=ID, group=gname, subject=sid, person=sid, subject_excluded_by_authors=False,
                age=info["age"], sex=info["sex"], handedness="", population=pop,
                setting="lab" if gname == "Healthy control" else "clinic",
                device="Mobvoi Ticwatch S2 smartwatch (triaxial accelerometer, about 50 Hz) with the Monipar app",
                session=j + 1, block=f"week {j + 1}", trial=n_here, phase="test", time_s="", task="finger tapping and timing",
                unit="recording", condition="eight Monipar exercises in sequence", pert_type="none", pert_value=0, pert_unit="",
                feedback="normal", excluded_by_authors=False, recording=n_here, samples_file=fn,
                signal_kind="wrist-worn smartwatch accelerometer", channels=";".join(CH),
                channel_units="m/s^2;m/s^2;m/s^2;label 0-8 (see README);tremor label -1/0/1/2 (see README)",
                sample_rate_hz=50, n_samples=len(c), duration_s=t[-1] - t[0], events="",
                processing="none (raw samples as shared by the authors; time stamps are irregular, about 19-20 ms apart)"))
        nrec[gname] = nrec.get(gname, 0) + n_here
        checks.internal(f"{sid}: recordings equal the table's weekly tests", n_here == info["weekly"], f"{n_here} vs {info['weekly']}")
        parts.append(dict(person=sid, hoehn_yahr=info["hy"], symptom_onset_year=info["onset"], first_diagnosis_year=info["diag"],
                          weekly_tests=info["weekly"], tremor_reported=info["tremor"]))

df = pd.DataFrame(rows)
# ---- published numbers (Methods and Table 5 of the paper)
checks.published("recordings, Remote", nrec["Remote"], 72, 0, "Table 5, collected trials")
checks.published("recordings, Supervised", nrec["Supervised"], 46, 0, "Table 5, collected trials")
checks.published("recordings, Control", nrec["Healthy control"], 56, 0, "Table 5, collected trials")
checks.published("recordings, all", len(df), 174, 0, "Results, collected trials")
checks.published("PD participants", df[df.population != "healthy"].person.nunique(), 21, 0, "Methods, Participants")
checks.published("healthy controls", df[df.population == "healthy"].person.nunique(), 7, 0, "Methods, Participants")
for g, hh in (("Remote", 9.1), ("Supervised", 5.8), ("Healthy control", 7.1)):
    checks.internal(f"hours recorded, {g}: {hours[g]:.2f} h from the time stamps (paper Table 5: {hh} h)", abs(hours[g] - hh) < 1.0)
pp = df.drop_duplicates("person")
for g, m_ in (("Remote", 63.6), ("Supervised", 64.2), ("Healthy control", 64.0)):
    checks.published(f"mean age, {g}", pp[pp.group == g].age.mean(), m_, 0.15, "Methods, Participants")
checks.published("men in Remote", (pp[pp.group == "Remote"].sex == "M").sum(), 8, 0, "Methods, Participants")
checks.published("men in Healthy control", (pp[pp.group == "Healthy control"].sex == "M").sum(), 3, 0, "Methods, Participants")
lens = df.n_samples
checks.internal("every recording has about 7 minutes of samples", lens.min() > 15000 and lens.max() < 30000, f"{lens.min()}-{lens.max()}")

meta = dict(
    paper=dict(authors="Sigcha L, Polvorinos-Fernandez C, Costa N, Costa S, Arezes P, Gago M, Lee C, Lopez JM, de Arcas G, Pavon I", year=2023,
               title="Monipar: movement data collection tool to monitor motor symptoms in Parkinson's disease using smartwatches and smartphones",
               venue="Front Neurol 14:1326640", doi="10.3389/fneur.2023.1326640"),
    lab="de Arcas / Lopez (Universidad Politecnica de Madrid) with Arezes (University of Minho)",
    data_url="https://zenodo.org/records/8104853 (Zenodo title: Monipar Database, doi 10.5281/zenodo.8104853)",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted from MATLAB to one CSV of samples per recording plus the OpenMotor table; nothing resampled or filtered. "
             "The tremor labels (Supervised and Remote groups) are joined to the samples as a column. Participant codes replace the authors' "
             "subject names; the age formula in the authors' table (year of assessment minus birth year) is stored as age only, without birth years. "
             "The MDS-UPDRS clinical score sheet and the code and figure files are not included."),
    how=["One row is one weekly recording of one person wearing a smartwatch on the wrist, while doing the eight Monipar exercises in order "
         "(rest, postural tremor, hands to chest, thumb-index finger tapping, rapid hand open-close, pronation-supination, rising from a chair, gait), "
         "about 7 minutes. Each recording's samples are in samples/<participant>_week<NN>.csv: time_s (the authors' time stamp in ms, divided by 1000), "
         "acc_x, acc_y, acc_z in m/s^2, exercise_label and tremor_label.",
         "exercise_label (authors' coding): 0 postural transition, 1 rest, 2 postural tremor of the hands, 3 hands to chest, 4 finger tapping of thumb and index, "
         "5 rapid hand open and close, 6 pronation and supination, 7 arising from a chair, 8 gait. tremor_label: -1 no labelled signal, 0 no tremor "
         "(MDS-UPDRS 0), 1 slight (under 1 cm), 2 mild (1 to under 3 cm); empty where the authors gave no tremor label file (healthy controls and some PD participants).",
         "Groups: Supervised (PD, tested at one site under similar supervised conditions), Remote (PD, tested at PD associations), Healthy control. The task column "
         "is the closest listed task; the recordings cover several movements. participants.csv holds Hoehn and Yahr stage, symptom onset and diagnosis years and the authors' tremor flag."],
    differences=["Table 5 gives 9.1 h (Remote), 5.8 h (Supervised) and 7.1 h (Control), about 7.6 minutes per recording; the shared recordings last "
                 "6.8 minutes on average by their time stamps (8.2, 5.2 and 6.8 h), so the hour totals do not reproduce (about 10 percent less). "
                 "The numbers of recordings and participants, mean ages and sex counts do reproduce."],
    unit_note="weekly smartwatch recording (all eight exercises)",
)
stdlib.release(ID, "timeseries", df, meta, checks, participants=pd.DataFrame(parts),
               participant_columns=[("hoehn_yahr", "number", "stage", "Hoehn and Yahr stage, as in the authors' table. Empty for healthy controls."),
                                    ("symptom_onset_year", "integer", "year", "Year of symptom onset, as in the authors' table where given."),
                                    ("first_diagnosis_year", "integer", "year", "Year of first diagnosis, where given."),
                                    ("weekly_tests", "integer", "", "Number of weekly recordings in the authors' table."),
                                    ("tremor_reported", "text", "yes, no", "Authors' tremor flag for the participant. Empty where the table gives none.")],
               extra_files=extra)
