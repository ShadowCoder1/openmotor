"""Warmerdam, Hansen, Romijnders, Hobert, Welzel & Maetzler (2022) Data 7(10):136: the figshare deposit holds full-body IMU and optical motion capture
recordings of 21 healthy participants in seven RAR parts of about 2 GB each (about 15 GB), and one small Excel file with their demographics and clinical test scores.
Only the Excel file (raw_extra/Warmerdam2022_Data/demographics_scores_extern.xlsx, sheet Mobility_Dataset_Healthy and its Labels sheet) is converted, in the 'walking'
module: one row per timed 5 m over-ground walk (five per participant). The RAR parts exceed the 5 GB limit and are not converted.
Run from the site folder: python3 -I tools/std/convert/Warmerdam2022_Data.py"""
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Warmerdam2022_Data"
MODULE = "walking"
RAW = stdlib.raw_dir(ID)
wb = openpyxl.load_workbook(RAW / "demographics_scores_extern.xlsx", read_only=True, data_only=True)
assert wb.sheetnames == ["Mobility_Dataset_Healthy", "Labels"]
rows_x = list(wb["Mobility_Dataset_Healthy"].iter_rows(values_only=True))
hdr = list(rows_x[0])
D = pd.DataFrame(rows_x[1:], columns=hdr)
labels = {r[0]: r[1] for r in wb["Labels"].iter_rows(values_only=True)}
assert "time in seconds" in labels["speed_5m_hallway"] and "five" in labels["speed_5m_hallway"]
T = [f"speed_5m_hallway_{i}" for i in range(1, 6)]
for c in T + ["speed_5m_hallway_mean", "age", "vital_weight", "bmi", "foot_length"]:
    D[c] = pd.to_numeric(D[c], errors="coerce")
D["vital_height"] = pd.to_numeric(D["vital_height"], errors="coerce")  # one height was typed as text ('182.5')
assert len(D) == 21 and D.id.is_unique
rows = []
for _, p in D.iterrows():
    pid = f"P{int(p.id)}"
    for k, c in enumerate(T, 1):
        rows.append(dict(
            dataset=ID, study="Warmerdam et al. 2022 Data", experiment="", group="", subject=pid, person=pid, subject_excluded_by_authors=False,
            age=p.age, sex="F" if p.gender == 1 else "M", handedness="" if pd.isna(p.handedness) else ("left" if p.handedness == 1 else "right"),
            population="healthy", setting="lab", device="hallway (5 m timed walk)", session=1, block="5 m over-ground walking test", trial=k,
            phase="test", time_s=np.nan, task="walking", unit="trial", condition="5 m walk, hallway", pert_type="none", pert_value=0, pert_unit="",
            feedback="normal", excluded_by_authors=False, walking_context="overground", walk_time_s=p[c]))
df = pd.DataFrame(rows)
ck = stdlib.Checks()
ck.published("healthy participants whose data are shared online", len(D), 21, 0, "Abstract (the data of 21 healthy subjects are shared online)")
ck.published("timed 5 m walks per participant", df.groupby("subject").size().unique().tolist()[0] if df.groupby("subject").size().nunique() == 1 else -1, 5, 0,
             "Labels sheet of the deposit (speed_5m_hallway: the five 5 m over ground walking trials)")
ck.published("timed walks in all", len(df), 105, 0, "Abstract and Labels sheet: 21 participants x five 5 m walks")
ck.internal("the deposit's mean column equals the mean of the five times", bool(np.allclose(D[T].mean(axis=1), D.speed_5m_hallway_mean, atol=1e-6)))
ck.internal("all walk times are between 2 and 6 s (5 m, so 0.8-2.5 m/s)", bool(df.walk_time_s.between(2, 6).all()), f"{df.walk_time_s.min():.2f} to {df.walk_time_s.max():.2f}")
ck.internal("no missing walk times", bool(df.walk_time_s.notna().all()))
ck.internal("21 participants in the rows", df.subject.nunique() == 21, f"{(D.gender == 0).sum()} men, {(D.gender == 1).sum()} women")
parts = [dict(person=f"P{int(p.id)}", height_cm=p.vital_height, body_mass_kg=p.vital_weight, bmi=p.bmi, foot_length_cm=p.foot_length, moca_total=p.moca_total,
              sppb_total=p.sppb_total, updrs_part3_total=p.updrs_3_total, updrs_part3_gait_item=p.updrs_3_10_gait) for _, p in D.iterrows()]
pcols = [("height_cm", "number", "cm", "Height (vital_height); one value in the deposit is typed as text and read as a number."),
         ("body_mass_kg", "number", "kg", "Body weight (vital_weight)."), ("bmi", "number", "kg/m2", "Body mass index (bmi)."),
         ("foot_length_cm", "number", "cm", "Foot length (foot_length)."), ("moca_total", "number", "points", "Montreal Cognitive Assessment total (moca_total)."),
         ("sppb_total", "number", "points", "Short Physical Performance Battery total (sppb_total)."),
         ("updrs_part3_total", "number", "points", "MDS-UPDRS part 3 total (updrs_3_total), as scored in this healthy group."),
         ("updrs_part3_gait_item", "number", "points", "MDS-UPDRS part 3 item 10, gait (updrs_3_10_gait).")]
dif = ["Only 21 healthy participants are in the deposit (the paper's 167 participants include patients whose data are available on request under a data sharing agreement).",
       "The deposit's seven RAR parts (full-body IMU and optical motion capture, about 15 GB) are not converted; only the demographics and clinical test file, whose five timed 5 m walks per participant "
       "are the only per-trial values outside the RAR parts, is.",
       "The columns are called speed_5m_hallway_1 to _5 in the Excel file but the Labels sheet defines them as times in seconds of five 5 m over-ground walks; values of %.2f to %.2f are consistent with seconds "
       "(%.1f to %.1f m/s over 5 m), so they are carried as times (walk_time_s), not as speeds." % (df.walk_time_s.min(), df.walk_time_s.max(), 5 / df.walk_time_s.max(), 5 / df.walk_time_s.min()),
       "The paper itself could not be read (the publisher blocks automated access and it is not in Europe PMC), so only numbers in its abstract (via Crossref) and the deposit's Labels sheet are used as checks.",
       "The Excel file's other columns (questionnaires and cognitive tests, mostly zero or empty for these healthy participants) are not carried; the participants table keeps a few."]
meta = dict(
    paper=dict(authors="Warmerdam E, Hansen C, Romijnders R, Hobert MA, Welzel J, Maetzler W", year=2022,
               title="Full-Body Mobility Data to Validate Inertial Measurement Unit Algorithms in Healthy and Neurological Cohorts", venue="Data 7(10):136",
               doi="10.3390/data7100136"),
    lab="Maetzler / Hansen (Kiel University)", data_url="https://doi.org/10.6084/m9.figshare.20238006.v2", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("The five timed 5 m walks of each of the 21 participants (columns speed_5m_hallway_1 to _5) became five rows each, in the authors' order, with the time in seconds unchanged. "
             "Age, gender and handedness are the deposit's (gender 0 = male, 1 = female; handedness 0 = right, 1 = left, empty when not given). Participant numbers replaced by codes."),
    how=["One row is one timed 5 m over-ground walk in a hallway (five per participant, as the deposit's Labels sheet defines the columns); walk_time_s is the time in seconds the authors "
         "recorded. Walking speed would be 5 m divided by this time; it is not written into the table.",
         "The participants table has height, weight, body mass index, foot length and a few clinical totals from the same Excel file. The recordings of the full-body inertial and motion capture "
         "measurements that the paper describes are in the deposit's large RAR parts and are not part of this conversion."],
    unit_note="timed 5 m walk", differences=dif)
stdlib.release(ID, MODULE, df.to_dict("records"), meta, ck, participants=pd.DataFrame(parts), participant_columns=pcols)
