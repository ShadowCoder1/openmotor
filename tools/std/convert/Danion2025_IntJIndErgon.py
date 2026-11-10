"""Danion et al. (2025), Int J Ind Ergon 106:103710: stick and rudder tracking (single, dual and triple tasking) and a mental arithmetic
task in 1578 candidates for the French Air and Space Force flight school; one row per candidate, task, condition and trial.

Run from the site folder: python3 -I tools/std/convert/Danion2025_IntJIndErgon.py
Raw file (untrusted data, read only): raw_extra/Danion2025_IntJIndErgon/data.xlsx = DataSECPIL-OSF.xlsx from https://osf.io/96rtv/.
The paper's full text was not reachable (Elsevier; HAL bot check); the abstract (HAL record) gives the design and N = 1578.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Danion2025_IntJIndErgon"
X = pd.read_excel(stdlib.raw_dir(ID) / "data.xlsx", sheet_name=None, header=None)
rud, stk, lat = X["RUDDER data"], X["STICK data"], X["Laterality & Arithmetic Tests"]
isSub = lambda d: d[0].astype(str).str.fullmatch(r"Subject \d+")
rud, stk, lat = [d[isSub(d)].reset_index(drop=True) for d in (rud, stk, lat)]
N = len(rud)
assert len(stk) == N == len(lat)
assert (rud[0] == stk[0]).all() and (rud[0] == lat[0]).all()
assert (rud[1] == lat[4]).all() and (rud[2].astype(float) == lat[5].astype(float)).all() and (rud[1] == stk[1]).all()

CONDS = [("UNI", "single tasking: this device alone"), ("DUAL", "dual tasking: stick and rudder tracked together"),
         ("TRI", "triple tasking: stick and rudder tracked together while doing mental calculation")]
rows = []


def base(i, d, exp, cond, measure, desc, unit, stat, val, trial, ntr):
    sex = d.iloc[i, 1] if exp != "arithmetic" else d.iloc[i, 4]
    age = d.iloc[i, 2] if exp != "arithmetic" else d.iloc[i, 5]
    hand = lat.iloc[i, 1]
    return dict(dataset=ID, study="Danion et al. 2025, Int J Ind Ergon", experiment=exp, group="", subject=d.iloc[i, 0], person=d.iloc[i, 0],
                subject_excluded_by_authors=False, age=float(age), sex=sex,
                handedness={"R": "right", "L": "left"}.get(hand, np.nan), population="healthy", setting="clinic",
                device="flight evaluation system with control stick, rudder pedals and screen", session=1, block=cond, trial=trial,
                phase="test", time_s=np.nan, task="tracking", unit="trial" if trial is not None else "participant summary", condition=cond,
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False, measure=measure,
                measure_description=desc, statistic=stat, value=float(val), value_unit=unit, n_trials=ntr, session_context="")


for exp, d in (("rudder tracking", rud), ("stick tracking", stk)):
    dev = exp.split()[0]
    for i in range(N):
        for m, c0, unit, mdesc in (("tracking_error", 3, "pixels", "Mean distance between cursor and target"),
                                   ("tracking_lag", 19, "ms", "Lag between cursor and target")):
            for k, (cond, cdesc) in enumerate(CONDS):
                for t in range(4):
                    v = d.iloc[i, c0 + 4 * k + t]
                    if pd.notna(v):
                        rows.append(base(i, d, exp, cond, f"{dev}_{m}", f"{mdesc} for the {dev} ({cdesc})", unit, "value", v, t + 1, 1))
            for k, (cond, cdesc) in enumerate(CONDS):
                v = d.iloc[i, c0 + 12 + k]
                if pd.notna(v):
                    rows.append(base(i, d, exp, cond, f"{dev}_{m}_mean4", f"{mdesc} for the {dev}, averaged over the 4 trials by the authors ({cdesc})",
                                     unit, "mean", v, None, 4))
for i in range(N):
    for t in range(4):
        rows.append(base(i, lat, "arithmetic", "TRI", "arithmetic_correct", "Mental calculation response in trial t: 1 = good response, 0 = wrong response (sheet legend)",
                         "score", "value", lat.iloc[i, 6 + t], t + 1, 1))
    rows.append(base(i, lat, "arithmetic", "TRI", "arithmetic_correct_mean4", "Mean of the 4 arithmetic responses (proportion good)", "proportion", "mean",
                     lat.iloc[i, 10], None, 4))
df = pd.DataFrame(rows)
df["trial"] = df["trial"].astype("Int64")
people = pd.DataFrame(dict(person=lat[0], laterality_hand=lat[1], laterality_foot=lat[2], laterality_eye=lat[3]))

chk = stdlib.Checks()
chk.published("candidates", df.person.nunique(), 1578, 0, "Title and abstract (1578 candidates)")
chk.internal("sex is M or F for every candidate", set(df.sex) <= {"M", "F"})
chk.internal("both sheets list the same candidates, sexes and ages, and the arithmetic sheet too", True)
chk.internal("per-trial rows: 4 trials per candidate, condition and measure",
             bool((df[df.unit == "trial"].groupby(["person", "measure", "condition"]).size() == 4).all()))
for dev in ("rudder", "stick"):
    for m in ("tracking_error", "tracking_lag"):
        a = df[(df.measure == f"{dev}_{m}") & (df.unit == "trial")].groupby(["person", "condition"]).value.mean()
        b = df[df.measure == f"{dev}_{m}_mean4"].set_index(["person", "condition"]).value
        j = a.align(b, join="inner")
        chk.internal(f"{dev} {m}: authors' 4-trial averages equal the mean of the trial values (max abs diff < 1e-3)",
                     bool((j[0] - j[1]).abs().max() < 1e-3), f"{(j[0] - j[1]).abs().max():.3g} over {len(j[0])}")
chk.internal("arithmetic sheet: authors' mean equals mean of 4 responses",
             bool(np.abs(lat[[6, 7, 8, 9]].astype(float).mean(axis=1) - lat[10].astype(float)).max() < 1e-6))
print([c for c in chk.items if not c["ok"]])
meta = dict(
    paper=dict(authors="Danion FR, Scotto CR, Choisay F, Debray L, Rey A, Ferrandez V", year=2025,
               title="Investigating the influence of multitasking and gender in visuomotor skills of 1578 pilot candidates",
               venue="International Journal of Industrial Ergonomics 106:103710", doi="10.1016/j.ergon.2025.103710"),
    lab="Danion (CNRS, Universite de Poitiers)", data_url="https://osf.io/96rtv/", license="none stated", license_url="",
    changes="The authors' Excel sheets (stick, rudder and arithmetic; one row per candidate) were reshaped to long form, one row per candidate, "
            "measure, condition and trial. The deposit's rows of group means and SDs at the bottom of each sheet were left out. Candidate numbers were replaced by codes.",
    how=["Candidates tracked a moving target in a flight simulator with a control stick or rudder pedals, 4 trials per condition: UNI (that device alone), "
         "DUAL (stick and rudder together) or TRI (both while doing mental calculation), as described in the abstract. Error is the mean "
         "cursor-target distance in pixels, lag in ms; the sign of the lag is as in the sheet (negative values occur, mostly for the stick) and is not defined in the deposit. "
         "The 4-trial averages the authors computed are kept as rows with statistic = mean and n_trials = 4.",
         "The arithmetic sheet holds a 0/1 score for each of 4 trials (the sheet's legend: 1 = good response). It is labelled TRI here because "
         "the abstract names mental calculation as the third task; the sheet itself does not state this.",
         "Sex and age are from the sheets (age in whole years). Handedness is the sheet's hand-laterality letter (R/L), empty where missing; foot and eye "
         "laterality are in participants.csv. The abstract also mentions oscillating and non-oscillating simulator runs, but the sheets have no column for it.",
         "Setting is entered as clinic (selection testing at the flight school)."],
    unit_note="measure of one trial or candidate (long form)",
    differences=["Only the number of candidates (1578) could be checked against the paper: its full text could not be read (publisher and HAL refuse automated access)."])
stdlib.release(ID, "summary", df, meta, chk, participants=people,
               participant_columns=[("laterality_hand", "category", "R, L, empty", "Hand laterality from the sheet (empty = not recorded)."),
                                    ("laterality_foot", "category", "R, L, none, empty", "Foot laterality from the sheet."),
                                    ("laterality_eye", "category", "R, L, none, empty", "Eye laterality from the sheet.")])
