"""Mathew, Masson & Danion (2020), Sci Rep: sex differences in visuomotor tracking. Two cohorts (50 and 77 participants),
one 10-trial baseline block each, hand and eye tracking measures per trial.

Run from the site folder: python3 -I tools/std/convert/Mathew2020_SciRep.py
Raw file (untrusted data, read only): raw_extra/Mathew2020_SciRep/data.xlsx from https://doi.org/10.5281/zenodo.3903369.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mathew2020_SciRep"
checks = stdlib.Checks()
sheets = pd.read_excel(stdlib.raw_dir(ID) / "data.xlsx", sheet_name=None, header=None)

# (sheet, first column, measure, description, unit)
HAND = ("HAND-TARGET DISTANCE", "Mean distance between cursor (hand) and target over the 10 s trial", "cm")
LAG = ("HAND-TARGET LAG", "Time shift that maximizes the cross-correlation between cursor and target motion (x and y interleaved)", "ms")
RES = ("RESIDUAL HAND-TARGET DISTANCE", "Mean cursor-target distance after realigning cursor and target for the lag", "cm")
BLOCKS = {
    "DATA HAND FIG4": [(2, "hand_target_distance", *HAND[1:]), (13, "hand_target_lag", *LAG[1:]), (24, "hand_target_residual_distance", *RES[1:])],
    "TAN VEL + SUBMOV": [
        (2, "cursor_tangential_velocity_mean", "Mean tangential velocity of the cursor over the trial", "cm/s"),
        (13, "cursor_tangential_velocity_sd", "SD of the tangential velocity of the cursor over the trial", "cm/s"),
        (24, "n_submovements", "Number of cursor submovements in the trial: zero crossings of tangential acceleration divided by 2", "count")],
    "ENTROPY": [
        (2, "apen_cursor_horizontal", "Approximate entropy of horizontal cursor motion (larger = less predictable)", "a.u."),
        (13, "apen_cursor_vertical", "Approximate entropy of vertical cursor motion (larger = less predictable)", "a.u.")],
    "DATA EYE FIG6": [
        (2, "eye_target_distance", "Mean Euclidean distance between gaze and target over the trial", "cm"),
        (13, "eye_target_lag", "Time shift that maximizes the cross-correlation between gaze and target motion", "ms"),
        (24, "saccade_rate", "Saccades per second during the trial", "Hz"),
        (35, "smooth_pursuit_gain", "Gain of smooth pursuit during the trial", "ratio")],
}
BLOCKS2 = BLOCKS["DATA HAND FIG4"]
EYEHAND = [(2, "eye_target_distance_mean10", "Eye-target distance averaged over the 10 trials", "cm"),
           (3, "cursor_target_distance_mean10", "Cursor-target distance averaged over the 10 trials", "cm"),
           (5, "eye_target_lag_mean10", "Eye-target lag averaged over the 10 trials", "ms"),
           (6, "cursor_target_lag_mean10", "Cursor-target lag averaged over the 10 trials", "ms")]
FIRST = {"HAND-TARGET": None}

rows = []


def add(cohort, sid, sex, measure, desc, unit, stat, val, trial, ntr, device, exp):
    rows.append(dict(
        dataset=ID, study="Mathew et al. 2020, Sci Rep", experiment=exp, group="", subject=f"{cohort}_{sid}", person=f"{cohort}_{sid}",
        subject_excluded_by_authors=False, age=np.nan, sex=sex, handedness="right", population="healthy", setting="lab", device=device,
        session=1, block="baseline block of 10 trials", trial=trial, phase="baseline", time_s=np.nan, task="tracking",
        unit="trial" if trial is not None else "participant summary", condition="", pert_type="none", pert_value=0, pert_unit="",
        feedback="normal", excluded_by_authors=False, measure=measure, measure_description=desc, statistic=stat,
        value=val, value_unit=unit, n_trials=ntr, session_context=""))


DEV1 = "joystick moving a cursor on a 27-inch 144 Hz screen, chin rest, infrared video eye tracker"
DEV2 = "joystick moving a cursor on a 27-inch 144 Hz screen, chin rest"
FREQ_DESC = "Power of {} cursor motion at {:.3f} Hz (frequency bin of the authors' spectral analysis; power unit not stated)"
n_cells = 0


def participants(df, ids_from=3):
    d = df.iloc[3:].reset_index(drop=True)
    return d


def build(cohort, exp, blocksets, device, sheet_names, n_expected):
    global n_cells
    base = sheets[sheet_names[0]].iloc[3:, 0].astype(str).tolist()
    sexes = sheets[sheet_names[0]].iloc[3:, 1].astype(str).tolist()
    assert len(base) == n_expected
    for i, (sid, sex) in enumerate(zip(base, sexes)):
        for sh in sheet_names:
            d = sheets[sh]
            assert d.iloc[3 + i, 0] == sid and d.iloc[3 + i, 1] == sex, (sh, sid)
            for c0, m, desc, unit in blocksets.get(sh, []):
                for t in range(10):
                    v = d.iloc[3 + i, c0 + t]
                    add(cohort, sid, sex, m, desc, unit, "value", float(v) if pd.notna(v) else np.nan, t + 1, 1, device, exp)
                    n_cells += 1
            if sh == "DATA FFT FIG5":
                for off, axis in ((2, "horizontal"), (21, "vertical")):
                    for k in range(18):
                        f = float(d.iloc[2, off + k])
                        add(cohort, sid, sex, f"cursor_power_{axis}_{f:.3f}Hz", FREQ_DESC.format(axis, f), "power", "value",
                            float(d.iloc[3 + i, off + k]), None, 10, device, exp)
                        n_cells += 1
            if sh == "DATA EYE-HAND FIG7":
                for c0, m, desc, unit in EYEHAND:
                    add(cohort, sid, sex, m, desc, unit, "mean", float(d.iloc[3 + i, c0]), None, 10, device, exp)
                    n_cells += 1


build("C1", "first cohort (N=50, gaze recorded)", BLOCKS | {"DATA FFT FIG5": []}, DEV1,
      ["DATA HAND FIG4", "DATA FFT FIG5", "TAN VEL + SUBMOV", "ENTROPY", "DATA EYE FIG6", "DATA EYE-HAND FIG7"], 50)
build("C2", "second cohort (N=77, hand only)", {"DATA second cohort": BLOCKS2}, DEV2, ["DATA second cohort"], 77)

df = pd.DataFrame(rows)
df["trial"] = df["trial"].astype("Int64")

# ---------------------------------------------------------------- checks
chk = stdlib.Checks()
p1 = df[df.experiment.str.startswith("first")].drop_duplicates("person")
p2 = df[df.experiment.str.startswith("second")].drop_duplicates("person")
chk.published("first cohort participants", len(p1), 50, 0, "Methods, Participants")
chk.published("first cohort females", (p1.sex == "F").sum(), 25, 0, "Methods, Participants")
chk.published("first cohort males", (p1.sex == "M").sum(), 25, 0, "Methods, Participants")
chk.published("second cohort participants", len(p2), 77, 0, "Methods, Participants")
chk.published("second cohort females", (p2.sex == "F").sum(), 51, 0, "Methods, Participants")
chk.published("second cohort males", (p2.sex == "M").sum(), 26, 0, "Methods, Participants")
hand = df[df.measure == "hand_target_distance"]
chk.published("hand-tracking trials (both cohorts)", len(hand), 1270, 0, "Methods, Participants/Materials (1,270 trials for hand movements)")
chk.published("eye-tracking trials (first cohort)", df[df.measure == "eye_target_distance"].shape[0], 500, 0,
              "Methods (500 trials for eye movements)")

per = df[(df.unit == "trial")].groupby(["experiment", "measure", "person", "sex"]).value.mean().reset_index()


def stat(measure, sex, cohort="first", fn="mean"):
    s = per[(per.measure == measure) & (per.sex == sex) & per.experiment.str.startswith(cohort)].value
    return s.mean() if fn == "mean" else s.std(ddof=1)


def tt(measure):
    a = per[(per.measure == measure) & per.experiment.str.startswith("first")]
    from scipy import stats
    r = stats.ttest_ind(a[a.sex == "F"].value, a[a.sex == "M"].value)
    return abs(r.statistic)


R4 = "Results, Manual tracking is more accurate in male"
chk.published("hand-target distance, women, mean (cm)", stat("hand_target_distance", "F"), 1.87, 0.005, R4)
chk.published("hand-target distance, women, SD (cm)", stat("hand_target_distance", "F", fn="sd"), 0.42, 0.005, R4)
chk.published("hand-target distance, men, mean (cm)", stat("hand_target_distance", "M"), 1.62, 0.005, R4)
chk.published("hand-target distance, men, SD (cm)", stat("hand_target_distance", "M", fn="sd"), 0.30, 0.005, R4)
chk.published("hand-target distance sex difference, |t|(48)", tt("hand_target_distance"), 2.43, 0.005, R4)
chk.published("hand-target lag, women, mean (ms)", stat("hand_target_lag", "F"), 71, 0.5, R4)
chk.published("hand-target lag, women, SD (ms)", stat("hand_target_lag", "F", fn="sd"), 25, 0.5, R4)
chk.published("hand-target lag, men, mean (ms)", stat("hand_target_lag", "M"), 49, 0.5, R4)
chk.published("hand-target lag, men, SD (ms)", stat("hand_target_lag", "M", fn="sd"), 21, 0.5, R4)
chk.published("hand-target lag sex difference, |t|(48)", tt("hand_target_lag"), 3.46, 0.005, R4)
chk.published("residual distance, men (cm)", stat("hand_target_residual_distance", "M"), 1.40, 0.005, R4)
chk.published("residual distance, women (cm)", stat("hand_target_residual_distance", "F"), 1.50, 0.005, R4)
R5 = "Results, Similar hand movement kinematics"
chk.published("tangential velocity mean, men (cm/s)", stat("cursor_tangential_velocity_mean", "M"), 18.5, 0.05, R5)
chk.published("tangential velocity mean, women (cm/s)", stat("cursor_tangential_velocity_mean", "F"), 18.6, 0.05, R5)
chk.published("tangential velocity SD, men (cm/s)", stat("cursor_tangential_velocity_sd", "M"), 9.67, 0.005, R5)
chk.published("tangential velocity SD, women (cm/s)", stat("cursor_tangential_velocity_sd", "F"), 9.66, 0.005, R5)
chk.published("ApEn vertical, men", stat("apen_cursor_vertical", "M"), 0.51, 0.005, R5)
chk.published("ApEn vertical, women", stat("apen_cursor_vertical", "F"), 0.50, 0.005, R5)
chk.published("submovements, men", stat("n_submovements", "M"), 57.9, 0.05, R5)
chk.published("submovements, women", stat("n_submovements", "F"), 59.0, 0.05, R5)
R6 = "Results, Similar gaze behavior"
chk.published("eye-target distance, men (cm)", stat("eye_target_distance", "M"), 1.39, 0.005, R6)
chk.published("eye-target distance, women (cm)", stat("eye_target_distance", "F"), 1.44, 0.005, R6)
chk.published("eye-target lag, men (ms)", stat("eye_target_lag", "M"), 58, 0.5, R6)
chk.published("eye-target lag, women (ms)", stat("eye_target_lag", "F"), 60, 0.5, R6)
chk.published("saccade rate, men (Hz)", stat("saccade_rate", "M"), 2.73, 0.005, R6)
chk.published("saccade rate, women (Hz)", stat("saccade_rate", "F"), 2.59, 0.005, R6)
chk.published("smooth pursuit gain, men", stat("smooth_pursuit_gain", "M"), 0.80, 0.005, R6)
chk.published("smooth pursuit gain, women", stat("smooth_pursuit_gain", "F"), 0.82, 0.005, R6)

h2 = df[(df.measure == "hand_target_distance") & df.experiment.str.startswith("second")]
chk.internal("every participant has 10 trials of each per-trial measure",
             bool((df[df.unit == "trial"].groupby(["person", "measure"]).size() == 10).all()))
chk.internal("no missing values", bool(df.value.notna().all()), str(int(df.value.isna().sum())))
e = sheets["DATA EYE-HAND FIG7"].iloc[3:, 2].astype(float).to_numpy()
m10 = df[df.measure == "eye_target_distance"].groupby("person", sort=False).value.mean().to_numpy()
chk.internal("eye-hand sheet's eye-target distance equals the mean of the 10 trial values (max abs diff < 1e-3)",
             bool(np.abs(e - m10).max() < 1e-3), f"{np.abs(e - m10).max():.2g}")
print([c for c in chk.items if not c["ok"]])

meta = dict(
    paper=dict(authors="Mathew J, Masson GS, Danion FR", year=2020, title="Sex differences in visuomotor tracking", venue="Sci Rep 10:11863",
               doi="10.1038/s41598-020-68069-0"),
    lab="Danion / Masson (CNRS, Aix-Marseille Universite)", data_url="https://doi.org/10.5281/zenodo.3903369", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="The authors' Excel sheets (one per figure, participants in rows, panels side by side) were reshaped to long form, one row per "
            "participant, measure and trial. Subject labels S1-S50 and S1-S77 were replaced by codes. Nothing else was recomputed.",
    how=["Participants tracked a target moving on a pseudo-random path with a joystick-controlled cursor for 10 s per trial; the data are the "
         "10 baseline trials of each participant. Cohort 1 (50 participants, 25 women and 25 men chosen by the authors) also had gaze recorded; "
         "cohort 2 (77 participants) had hand movements only.",
         "One row is one measure of one trial (statistic = value, trial 1-10), or, for the spectral power bins and for the eye-hand sheet, one "
         "per-participant value (trial empty, n_trials = 10 for the eye-hand averages). Distances are in cm, lags in ms, velocity in cm/s, "
         "saccade rate in Hz. Power spectra values are given by the authors without a unit.",
         "Sex is as in the authors' sheets (M/F, self-report). All participants were right-handed (Methods); age is given only as group means "
         "in the paper, so age is empty. Trial numbers are the order in the authors' sheets; the paper says pattern order was randomized.",
         "Whether any cohort 2 participant also appears in cohort 1 cannot be told from the sheets; the paper calls cohort 2 'new' "
         "participants, so they are coded as different people."],
    unit_note="measure of one trial or participant (long form)")
meta["differences"] = [d for d in []]
stdlib.release(ID, "summary", df, meta, chk)
