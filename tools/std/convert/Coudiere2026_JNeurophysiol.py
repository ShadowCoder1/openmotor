"""Coudiere & Danion (2026), J Neurophysiol 135(3):674-685: manual tracking with the eyes on, off or away from the target.
26 participants, ten hand/eye tracking measures in up to five conditions (SINGLE, EYE, FIX, DOUBLE, DIFF).

Run from the site folder: python3 -I tools/std/convert/Coudiere2026_JNeurophysiol.py
Raw file (untrusted data, read only): raw_extra/Coudiere2026_JNeurophysiol/data.xlsx from https://osf.io/v6k7h/ (DATA.xlsx).
The full text was not reachable (publisher 403, HAL bot check); the abstract (Crossref and HAL record) defines SINGLE, DOUBLE, FIX and n = 26.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Coudiere2026_JNeurophysiol"
v = pd.read_excel(stdlib.raw_dir(ID) / "data.xlsx", header=None)
conds = ["SINGLE", "EYE", "FIX", "DOUBLE", "DIFF"]
MEAS = [  # (first column of the id column, name, authors' header, unit)
    (5, "hand_target_positional_error", "Hand-Target Positional Error", "cm"),
    (12, "hand_target_lag", "Hand-Target Lag", "ms"),
    (19, "hand_target_directional_error", "Hand-Target Directional Error", "deg"),
    (26, "hand_target_velocity_error", "Hand-Target Velocity Error", "cm/s"),
    (33, "eye_target_error", "Eye-Target Error", "cm"),
    (40, "eye_target_lag", "Eye-Target Lag", "ms"),
    (47, "saccade_rate", "Saccade Rate", "saccades/s"),
    (54, "time_in_smooth_pursuit", "Time in SP", "%"),
    (61, "distance_covered_in_smooth_pursuit", "Distance Covered in SP", "%"),
    (68, "velocity_error_in_smooth_pursuit", "Velocity Error in SP", "cm/s"),
]
COND_NOTE = {"SINGLE": "eyes and hand track one target", "EYE": "eye-only tracking (no hand measures in the sheet)",
             "FIX": "hand tracks one target while the eyes fixate away from it",
             "DOUBLE": "eyes and hand track separate targets with the same trajectory in separate hemifields",
             "DIFF": "column labelled DIFF by the authors (not defined in the deposit; read as the dissimilar eye/hand task of the abstract)"}
body = v.iloc[3:].reset_index(drop=True)
rows, dropped = [], 0
for i in range(len(body)):
    sid = body.iloc[i, 0]
    for c0, m, hdr, unit in MEAS:
        assert body.iloc[i, c0] == sid
        for k, cond in enumerate(conds):
            val = body.iloc[i, c0 + 1 + k]
            if pd.isna(val):
                dropped += 1
                continue
            rows.append(dict(
                dataset=ID, study="Coudiere & Danion 2026, J Neurophysiol", experiment="manual tracking, eyes free/fixating/separate target",
                group="", subject=sid, person=sid, subject_excluded_by_authors=False, age=float(body.iloc[i, 1]),
                sex=body.iloc[i, 2], handedness=np.nan, population="healthy", setting="lab",
                device="joystick-controlled cursor and eye tracker", session=1, block="", trial=np.nan, phase="test", time_s=np.nan,
                task="tracking", unit="participant summary", condition=f"{cond}: {COND_NOTE[cond]}", pert_type="none", pert_value=0,
                pert_unit="", feedback="normal", excluded_by_authors=False, measure=m,
                measure_description=f"{hdr} ({unit}); one value per participant and condition, as in the authors' sheet",
                statistic="value", value=float(val), value_unit=unit, n_trials=np.nan, session_context=""))
df = pd.DataFrame(rows)
people = pd.DataFrame(dict(person=body[0], laterality_score=body[3].astype(float)))

chk = stdlib.Checks()
chk.published("participants", df.person.nunique(), 26, 0, "Abstract (n=26)")
chk.internal("26 participants x 10 measures; conditions present per measure are consistent across participants",
             bool((df.groupby(["person", "measure"]).size().groupby("measure").nunique() == 1).all()))
chk.internal("rows + empty cells = 26 x 10 x 5", len(df) + dropped == 26 * 10 * 5, f"{len(df)} rows, {dropped} empty")
chk.internal("ages 19-31, sexes F/M only", bool(df.age.between(18, 35).all() and set(df.sex) <= {"F", "M"}))
chk.internal("hand_target_lag values are all negative in SINGLE", bool(df[(df.measure == "hand_target_lag") & df.condition.str.startswith("SINGLE")].value.lt(0).all()))
print([c for c in chk.items if not c["ok"]])
meta = dict(
    paper=dict(authors="Coudiere A, Danion FR", year=2026, title="Extraretinal information provided by smooth pursuit is crucial for accurate hand tracking",
               venue="J Neurophysiol 135(3):674-685", doi="10.1152/jn.00573.2025"),
    lab="Danion (CNRS, Universite de Poitiers)", data_url="https://osf.io/v6k7h/", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="The authors' single wide Excel sheet (ten blocks of five condition columns) was reshaped to long form, one row per participant, "
            "measure and condition. Empty cells were not written. Participant letters were replaced by codes.",
    how=["One row is one measure of one participant in one condition (statistic = value), as shared by the authors; the deposit has one number "
         "per participant and condition, no trials. Units are from the authors' headers: cm, ms, degrees (the header's degree sign was damaged "
         "in the file), cm/s, saccades/s, percent.",
         "Conditions: SINGLE, DOUBLE and FIX are defined in the abstract. EYE has no hand values and is read as eye-only tracking, and DIFF is "
         "undefined in the deposit (the abstract mentions eye and hand tasks that are 'dissimilar'); both readings are inferred, the condition text says so. "
         "The sign of the lag is as in the sheet (negative in SINGLE); the deposit does not define it.",
         "Age and sex are from the sheet. The sheet's 'Laterality' value (57-100) is in participants.csv as laterality_score; the deposit does not "
         "define it (probably a handedness quotient), so handedness is left empty."],
    unit_note="measure of one participant in one condition (long form)",
    differences=["The full text could not be read (publisher and HAL refuse automated access), so only the participant count of the abstract is checked; "
                 "condition definitions beyond the abstract and the trial structure are not documented here."])
stdlib.release(ID, "summary", df, meta, chk, participants=people,
               participant_columns=[("laterality_score", "number", "57-100", "Value of the authors' 'Laterality' column; definition not given in the deposit.")])
