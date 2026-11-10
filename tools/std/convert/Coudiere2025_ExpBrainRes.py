"""Coudiere & Danion (2025), Exp Brain Res 243(6):143: eye-hand coordination while moving a joystick-controlled cursor
through 13 successive targets (Fitts' task with unveiled next target; amplitude A = 5, 10, 20 cm x width W = 1.2, 0.6, 0.3 cm).

Raw file, raw_extra/Coudiere2025_ExpBrainRes/ (OSF d5bc4, fetched with tools/std/fetch.py; no license stated):
  DataUncertainTargetOSF.xlsx, three sheets, each with blocks of 9 columns (the 9 A x W conditions, in the order
  W=1.2 A=5,10,20; W=0.6 A=5,10,20; W=0.3 A=5,10,20) and 24 participant rows (Subject 1-24) followed by MEAN and SEM rows:
    HAND only : movement time (ms), percentage of time after peak velocity, percentage of distance after peak velocity
                (stored as proportions 0-1: the group means are 0.5-0.8 where the paper says "close to 50%" and "almost 80%"),
                number of submovements per reach
    HAND-EYE  : eye-hand lag (ms, positive = eye leads), peak cross-correlation (R)
    EYE only  : amplitude of the primary saccade (cm), number of saccades per reach
Run from the site folder: python3 -I tools/std/convert/Coudiere2025_ExpBrainRes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Coudiere2025_ExpBrainRes"
SRC = "Coudiere & Danion 2025, Exp Brain Res 243:143"
FILE = "DataUncertainTargetOSF.xlsx"

# (sheet, first column of block, measure, description, statistic, unit)
MEAS = [
    ("HAND only", 1, "movement_time", "Mean duration of the hand (cursor) movement between successive targets, averaged over the 12 reaches of a trial and the 10 trials of the condition.", "mean", "ms"),
    ("HAND only", 12, "time_after_peak_velocity", "Share of the movement duration spent after the hand's peak velocity (deceleration phase), averaged over reaches and trials; the file stores it as a proportion.", "proportion", "proportion"),
    ("HAND only", 23, "distance_after_peak_velocity", "Share of the distance travelled after the hand's peak velocity, averaged over reaches and trials; stored as a proportion.", "proportion", "proportion"),
    ("HAND only", 34, "submovements_per_reach", "Number of submovements per reach (a submovement is counted whenever hand acceleration changes sign twice, after Chen et al. 2015), averaged over reaches and trials.", "mean", "count per reach"),
    ("HAND-EYE", 1, "eye_hand_lag", "Lag of the cross-correlation between gaze and cursor over each trial, averaged over the 10 trials; positive means the eye leads the hand.", "mean", "ms"),
    ("HAND-EYE", 12, "eye_hand_cross_correlation", "Peak coefficient of the gaze-cursor cross-correlation (spatial coupling after removing the lag), averaged over the 10 trials.", "mean", "correlation coefficient"),
    ("EYE only", 1, "primary_saccade_amplitude", "Amplitude of the primary saccade of each pointing movement, averaged over reaches and trials.", "mean", "cm"),
    ("EYE only", 12, "saccades_per_reach", "Total number of saccades triggered during each reach, averaged over reaches and trials.", "mean", "count per reach"),
]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def load():
    f = stdlib.raw_dir(ID) / FILE
    out = {}
    for sh in ("HAND only", "HAND-EYE", "EYE only"):
        out[sh] = pd.read_excel(f, sh, header=None)
    return out


def build(sheets):
    rows, tab = [], {}
    n = 24
    for p in range(n):
        subj = f"Subject {p + 1}"
        k = 0
        for sh, c0, name, desc, stat, unit in MEAS:
            d = sheets[sh]
            assert d.iloc[4 + p, 0] == subj
            for j in range(9):
                col = c0 + j
                W, A, idv = d.iloc[2, col], d.iloc[3, col], float(d.iloc[1, col])
                v = d.iloc[4 + p, col]
                v = float(v) if not pd.isna(v) else None
                k += 1
                tab.setdefault((name, p), []).append((A, W, idv, v))
                rows.append({"dataset": ID, "study": SRC, "experiment": "", "group": "", "subject": subj,
                             "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "right", "population": "healthy",
                             "setting": "lab", "device": "joystick-controlled cursor (1000 Hz) and Eyelink eye tracker (1000 Hz)",
                             "session": 1, "block": "", "trial": k, "phase": "test", "time_s": "", "task": "reaching",
                             "unit": "participant summary", "condition": f"{A}; {W}; ID={idv:g}", "pert_type": "none",
                             "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
                             "measure": name, "measure_description": desc, "statistic": stat, "value": v, "value_unit": unit,
                             "n_trials": 10})
    return rows, tab


def main():
    sheets = load()
    rows, tab = build(sheets)
    df = pd.DataFrame(rows)
    checks = stdlib.Checks()
    R = f"{SRC}, Results"
    mt = df[df.measure == "movement_time"]
    mean = lambda m, cond: df[(df.measure == m) & (df.condition == cond)].value.mean()
    checks.published("participants analysed (25 recruited, 1 excluded for a noisy eye signal)", df.subject.nunique(), 24, 0,
                     f"{SRC}, Methods (Participants); df of the ANOVAs F(4,92)")
    checks.published("effect of W on movement time at A = 20 cm (W=0.3 minus W=1.2, ms)",
                     mean("movement_time", "A=20 cm; W=0.3 cm; ID=6.08")
                     - mean("movement_time", "A=20 cm; W=1.2 cm; ID=4.14"), 381, 0.5, f"{R}, Hand related variables (Fig. 3B)")
    checks.published("... at A = 5 cm (ms)", mean("movement_time", "A=5 cm; W=0.3 cm; ID=4.14") - mean("movement_time", "A=5 cm; W=1.2 cm; ID=2.36"),
                     304, 0.5, f"{R}, Hand related variables (Fig. 3B)")
    checks.published("movement time at ID 4.14, small scale (A=5, W=0.3 cm, ms)", mean("movement_time", "A=5 cm; W=0.3 cm; ID=4.14"), 753, 0.5,
                     f"{R}, Fig. 3C")
    checks.published("eye-hand lag at ID 4.14, small scale (ms)", mean("eye_hand_lag", "A=5 cm; W=0.3 cm; ID=4.14"), 120, 0.5, f"{R}, Time lag (Fig. 4C)")
    checks.published("eye-hand lag at ID 4.14, large scale (ms)", mean("eye_hand_lag", "A=20 cm; W=1.2 cm; ID=4.14"), 154, 0.5, f"{R}, Time lag (Fig. 4C)")
    checks.internal("24 participants x 9 conditions x 8 measures rows", len(df) == 24 * 9 * 8, str(len(df)))
    checks.internal("only one value missing (primary saccade amplitude, subject 1, lowest ID)", int(df.value.isna().sum()) == 1, str(int(df.value.isna().sum())))
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Stacked the three sheets of the Excel file into long form (one row per participant, measure and condition, "
                 "in the order of the authors' columns), left out the MEAN and SEM rows, wrote the amplitude and width of the condition "
                 "into condition, and replaced 'Subject n' with codes. Values are unchanged."),
        how=["One row is one measure of one participant in one of the 9 amplitude x width conditions (A = 5, 10, 20 cm; W = 1.2, 0.6, 0.3 cm; "
             "index of difficulty ID = log2(A/W + 1)). The deposit has one value per participant and condition, each the authors' average over "
             "the 12 pointing movements of a trial and the 10 trials of the condition; no per-trial data.",
             "Participants moved a joystick (+-25 degrees = 15 cm of cursor) with the right hand to hit 13 successive targets on a screen, "
             "the next target appearing only when the current one was hit. Movement time is in ms, saccade amplitude in cm, the two "
             "'after peak velocity' shares are proportions (the file stores 0-1), the eye-hand lag is in ms and is positive when the eye leads. "
             "The deposit gives no age or sex per participant (the paper reports 19.5 +- 0.9 years, 17 females of 25 recruited)."],
        unit_note="measure of one participant in one condition (long form)",
        differences=[
            "The paper reports a linear fit MT(ms) = 11.2 + 176.7 x ID; a fit to the 24 x 9 values here gives a slope of about 176.9 and an intercept of about 4.3, so the fit was not used as a check.",
            "The paper gives the movement times at ID 4.14 for the intermediate and large scales as 707 and 723 ms; the rows give 707.6 and 723.7 ms (the paper appears to truncate there), so those two values are not checks.",
            "One primary-saccade amplitude (the lowest-difficulty condition of one participant, whose signal the paper calls too noisy) is empty in the deposit and stays empty here; the paper replaced it with the group mean for its statistics.",
        ],
    )
    return stdlib.release(ID, "summary", rows, meta, checks)


if __name__ == "__main__":
    main()
