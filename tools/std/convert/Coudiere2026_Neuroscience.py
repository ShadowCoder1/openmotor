"""Coudiere, Michon & Danion (2026), Neuroscience 611:306-316: joystick tracking of a target whose position is refreshed at
1.5-240 Hz, gaze fixed or free, 28 young and 30 older adults.

Raw file raw_extra/Coudiere2026_Neuroscience/AGING.xlsx (OSF mv3zn, no license stated; original name
AGING-MANUALTRACKING-OSF-MARCH2026.xlsx). Sheets: SUBJECTS (laterality, sex, age, group, MMSE), HAND TRACKING,
ECCENTRICITY, EYE FIXATION+TRACKING. Each measure sheet has blocks of columns, each starting with a GROUP column followed by
8 refresh rates (x gaze condition) and, for some, one authors' average column. Rows below the 58 participants (MEAN/SEM) are left out.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Coudiere2026_Neuroscience"
SRC = "Coudiere et al. 2026, Neuroscience"
N = 58
DESC = {
    "Raw CURSOR-TARGET dist (cm)": "Mean distance between cursor and target over the tracking trial, in cm.",
    "CURSOR-TARGET lag (ms)": "Temporal lag between cursor and target signals, in ms.",
    "Residual CURSOR-TARGET dist (cm)": "Mean cursor-target distance after temporally realigning the hand and target signals (lag removed), in cm.",
    "CORR  target EXC & cursor-target DIST": "Correlation between target eccentricity and cursor-target distance.",
    "REGRESSION SLOPE (cm/deg)": "Slope of the regression of cursor-target distance on target eccentricity (only when the correlation exceeded 0.2, per the authors' sheet), cm per deg.",
    "REGRESSION INTERCEPT (cm)": "Intercept of the regression of cursor-target distance on target eccentricity (only when the correlation exceeded 0.2), cm.",
    "SD XY EYE (cm)": "Standard deviation of gaze position (x and y) during the trial, in cm (gaze-fixed condition).",
    "EYE-TARGET dist (cm)": "Mean distance between gaze and target, in cm (gaze-free condition).",
    "EYE-TARGET lag (ms)": "Temporal lag between gaze and target signals, in ms (gaze-free condition).",
}


def unit_of(title):
    m = re.search(r"\(([^)]*)\)", title)
    return m.group(1) if m else "correlation coefficient"


def main():
    f = stdlib.raw_dir(ID) / "AGING.xlsx"
    x = pd.read_excel(f, sheet_name=None, header=None)
    sub = x["SUBJECTS"].iloc[1:1 + N].reset_index(drop=True)
    sub.columns = ["laterality", "sex", "age", "group", "mmse", "x"]
    hand = x["HAND TRACKING"]
    assert (hand.iloc[3:3 + N, 0].values == sub.sex.values).all() and (hand.iloc[3:3 + N, 1].astype(float).values == sub.age.astype(float).values).all()
    rec = []  # (person, title, gaze, label, rate, value)
    for sh, d in x.items():
        if sh == "SUBJECTS":
            continue
        gr = [r for r in range(d.shape[0]) if (d.iloc[r] == "GROUP").any()][0]
        gcols = [c for c in range(d.shape[1]) if d.iloc[gr, c] == "GROUP"]
        for g0, g1 in zip(gcols, gcols[1:] + [d.shape[1]]):
            strs = [str(d.iloc[r, c]) for r in range(gr) for c in range(g0, g1) if isinstance(d.iloc[r, c], str)]
            title = next(s for s in strs if "GAZE" not in s and not s.startswith("only"))
            gazes = [s for s in strs if "GAZE" in s]
            seg, prev, gi = -1, None, -1
            for c in range(g0 + 1, g1):
                h = d.iloc[gr, c]
                if isinstance(h, str):
                    label, rate, gaze = h, None, None
                elif pd.isna(h):
                    continue
                else:
                    if prev is None or float(h) <= prev:
                        gi += 1
                    prev = float(h)
                    label, rate, gaze = None, float(h), gazes[gi].title()
                for p in range(N):
                    assert d.iloc[gr + 1 + p, g0] == sub.group[p]
                    v = d.iloc[gr + 1 + p, c]
                    rec.append((p, title, gaze, label, rate, None if pd.isna(v) else float(v), sh))
    rows = []
    cnt = {}
    for p, title, gaze, label, rate, v, sh in rec:
        cnt[p] = cnt.get(p, 0) + 1
        if rate is not None:
            cond, stat = f"target refresh rate {rate:g} Hz; {gaze}", "mean"
            extra = ""
        else:
            cond, stat = f"{label} (authors' column after the refresh-rate columns)", "mean"
        rows.append({"dataset": ID, "study": SRC, "experiment": "", "group": "young" if sub.group[p] == "YOUNG" else "older",
                     "subject": f"P{p + 1}", "subject_excluded_by_authors": False, "age": float(sub.age[p]),
                     "sex": sub.sex[p], "handedness": "", "population": "healthy", "setting": "lab",
                     "device": "joystick-controlled cursor and eye tracker", "session": 1, "block": sh, "trial": cnt[p],
                     "phase": "test", "time_s": "", "task": "tracking", "unit": "participant summary", "condition": cond,
                     "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
                     "measure": title, "measure_description": DESC[title] + (" Value of the authors' extra column, presumably the average over refresh rates." if rate is None else ""),
                     "session_context": "", "statistic": stat, "value": v, "value_unit": unit_of(title), "n_trials": ""})
    df = pd.DataFrame(rows)
    parts = [{"person": f"P{p + 1}", "laterality": sub.laterality[p], "mmse": None if pd.isna(sub.mmse[p]) else sub.mmse[p]} for p in range(N)]
    ck = stdlib.Checks()
    ab = "Abstract (Europe PMC record of the paper)"
    ck.published("older participants", (sub.group == "OLDER").sum(), 30, 0, ab)
    ck.published("young participants", (sub.group == "YOUNG").sum(), 28, 0, ab)
    ck.published("mean age of older participants (years, printed as 71)", sub.age[sub.group == "OLDER"].astype(float).mean(), 71, 0.5, ab)
    ck.published("mean age of young participants (years, printed as 21)", sub.age[sub.group == "YOUNG"].astype(float).mean(), 21, 0.5, ab)
    ck.published("females among older participants", ((sub.group == "OLDER") & (sub.sex == "F")).sum(), 17, 0, ab)
    ck.published("females among young participants", ((sub.group == "YOUNG") & (sub.sex == "F")).sum(), 17, 0, ab)
    ck.internal("every participant has the same number of rows", len(set(cnt.values())) == 1, str(set(cnt.values())))
    ck.internal("no row from MEAN/SEM lines: 58 persons", df.subject.nunique() == N, str(df.subject.nunique()))
    for c in ck.items:
        print(("ok " if c["ok"] else "FAILED "), c["what"], c.get("computed", ""))
    s = next(z for z in json.load(open(stdlib.HERE / "source/extra_studies.json")) if z["id"] == ID)
    pp = s["papers"][0]
    meta = dict(study=ID, paper={k: pp[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
                license="none stated", license_url="",
                changes="Stacked the measure sheets of the Excel file into long form (one row per participant, measure and condition), left out MEAN and SEM rows, added sex, age and group from the SUBJECTS sheet, replaced row positions with codes. Values unchanged.",
                how=["One row is one measure of one participant in one condition: target refresh rate (1.5, 2, 2.5, 3, 3.5, 4, 8 or 240 Hz) and gaze condition (fixed or free), as in the authors' columns. The deposit has one value per participant and condition, not trials.",
                     "Participants tracked a target with a joystick-controlled cursor. Distances are in cm, lags in ms. Extra columns the authors placed after the refresh rates (labelled FIXED, FREE or HAND ONLY) are kept with their label; they are presumably averages over refresh rates but the sheet does not say. The eccentricity regression columns are blank when the authors' correlation criterion (r > 0.2) was not met. participants.csv holds the laterality score and MMSE (older adults) from the SUBJECTS sheet; laterality units are not documented."],
                unit_note="measure of one participant in one condition (long form)",
                differences=["The paper's statistical results were not re-computed; checks use the participant counts, mean ages and sex counts in the abstract."])
    return stdlib.release(ID, "summary", df, meta, ck, participants=parts,
                          participant_columns=[("laterality", "number", "", "Handedness score from the SUBJECTS sheet (scale not documented)."),
                                               ("mmse", "number", "", "MMSE score, older adults only.")])


if __name__ == "__main__":
    main()
