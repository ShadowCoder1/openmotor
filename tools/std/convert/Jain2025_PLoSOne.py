"""Jain, Schweighofer & Finley (2025) PLoS ONE: perceived severity of forward (trip) versus backward (slip) treadmill belt
perturbations in 20 young and 20 older adults, on two days.

Raw files (OSF n6qcy / 8wjku, no licence stated): Data_Table.csv (40 participants x 2 days x 2 sides x 7 forward magnitudes: the backward
magnitude judged equally severe), All_Subs_Assessments.csv (age, sex, clinical scales, falls, activity), State_Anxiety.csv (state
anxiety after trips and after slips) and Peak_VCOM_Means.csv (peak centre-of-mass velocity deviations; 37 participants).
Ethnicity and race columns of All_Subs_Assessments.csv are not released.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Jain2025_PLoSOne"
MODULE = "summary"
RAW = stdlib.raw_dir(DS)
ASSESS = {
    "PA_Hours": ("Self-reported exercise time per week (the paper tabulates minutes per week; the deposit's unit is hours, rounded)", "value", "hours/week"),
    "ABC": ("Activities-specific Balance Confidence scale score (0-100)", "value", "score"),
    "Falls": ("Number of falls in the 12 months before the visit (History of Falls questionnaire)", "count", "falls"),
    "MiniBest_Anticipatory": ("Mini-BESTest anticipatory postural control subscore", "value", "score"),
    "MiniBest_Gait": ("Mini-BESTest stability in gait subscore", "value", "score"),
    "MiniBest_Reactive": ("Mini-BESTest reactive postural control subscore", "value", "score"),
    "MiniBest_Sensory": ("Mini-BESTest sensory orientation subscore", "value", "score"),
    "MiniBest": ("Mini-BESTest total score (of 28)", "value", "score"),
}
ANX = {"Worry": "worry (2 items)", "Somatic": "somatic anxiety (5 items)", "Concentration": "concentration disruption (2 items)", "Total": "total (9 items)"}
PERT = {"Trips": ("forward belt perturbation (trip)", "trip"), "Slips": ("backward belt perturbation (slip)", "slip")}


def main():
    a = pd.read_csv(RAW / "All_Subs_Assessments.csv")
    anx = pd.read_csv(RAW / "State_Anxiety.csv")
    vc = pd.read_csv(RAW / "Peak_VCOM_Means.csv")
    dt = pd.read_csv(RAW / "Data_Table.csv")
    ck = stdlib.Checks()
    ck.internal("ID sets agree", set(a.ID) == set(anx.ID) == set(dt.ID) and set(vc.ID) <= set(a.ID) and len(a) == 40)
    info = a.set_index("ID")

    def base(pid, **kw):
        r = info.loc[pid]
        young = pid.startswith("HY")
        b = dict(dataset=DS, study="Jain 2025 PLoS ONE", experiment="perceived severity of trips and slips",
                 group="young adults" if young else "older adults", subject=pid, person=pid, subject_excluded_by_authors=False,
                 age=float(r["Age"]), sex={"Male": "M", "Female": "F"}[r["Gender"]], handedness=None,
                 population="healthy" if young else "healthy older adults", setting="lab",
                 device="instrumented split-belt treadmill, motion capture", session=1, block=None, trial=None, phase="test",
                 time_s=None, task="walking", unit="participant summary", condition=None, pert_type="none", pert_value=None,
                 pert_unit=None, feedback="normal", excluded_by_authors=False)
        b.update(kw)
        return b
    rows = []
    for pid in a.ID:
        r = info.loc[pid]
        for k, (desc, stat, unit) in ASSESS.items():
            if pd.notna(r[k]):
                rows.append({**base(pid, condition="clinical assessment"), "measure": k, "measure_description": desc,
                             "statistic": stat, "value": r[k], "value_unit": unit})
    for r in anx.itertuples(index=False):
        for d, (pd_, tag) in PERT.items():
            for k, desc in ANX.items():
                rows.append({**base(r.ID, condition=tag, pert_type="belt_perturbation_" + tag), "measure": f"state_anxiety_{k.lower()}",
                             "measure_description": f"State anxiety after the {tag} perturbations, {desc}; items scored 1 (not at all) to 9 (extremely) and summed",
                             "statistic": "value", "value": getattr(r, f"{d}_{k}"), "value_unit": "score"})
    for r in vc.itertuples(index=False):
        for d, (pd_, tag) in PERT.items():
            for day in (1, 2):
                rows.append({**base(r.ID, condition=tag, session=day, pert_type="belt_perturbation_" + tag),
                             "measure": "peak_com_velocity_deviation",
                             "measure_description": f"Mean over the {tag} perturbations of the deviation of peak pelvis centre-of-mass velocity from unperturbed walking (file column {d}_Delta_VCOM_D{day})",
                             "statistic": "mean", "value": getattr(r, f"{d}_Delta_VCOM_D{day}"), "value_unit": "m/s (inferred)"})
    for r in dt.itertuples(index=False):
        rows.append({**base(r.ID, condition=f"side {r.Side}", session=int(r.Day[1]), pert_type="belt_perturbation_pair",
                            pert_value=r.Fwd, pert_unit="m/s (forward magnitude)"),
                     "measure": "equally_severe_backward_magnitude",
                     "measure_description": "Magnitude of the backward (slip) belt perturbation judged as severe as a forward (trip) perturbation of the magnitude in pert_value, from a staircase of two-alternative forced choices; Side is the side of the staircase (left or right) as named in the deposit",
                     "statistic": "value", "value": r.Bwd, "value_unit": "m/s"})
    n_expected = int(a[list(ASSESS)].notna().sum().sum()) + 40 * 8 + 37 * 4 + 1120
    ck.internal("rows = values", len(rows) == n_expected, f"{len(rows)} vs {n_expected}")
    ck.internal("28 staircase points per participant", dt.groupby("ID").size().eq(28).all())
    ck.internal("7 forward magnitudes 0.30-0.60", sorted(dt.Fwd.unique().round(2)) == [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    y, o = a[a.ID.str.startswith("HY")], a[a.ID.str.startswith("HO")]
    ck.published("young adults", len(y), 20, 0, "Table 1")
    ck.published("older adults", len(o), 20, 0, "Table 1")
    ck.published("young adults SD of age", y.Age.std(), 5, 0.5, "Table 1")
    ck.published("older adults mean age", o.Age.mean(), 71, 0.5, "Table 1")
    ck.published("older adults SD of age", o.Age.std(), 4, 0.5, "Table 1")
    ck.published("young adults female", (y.Gender == "Female").sum(), 11, 0, "Table 1")
    ck.published("older adults female", (o.Gender == "Female").sum(), 12, 0, "Table 1")
    ck.published("young adults fallers", (y.Falls > 0).sum(), 7, 0, "Table 1")
    ck.published("older adults fallers", (o.Falls > 0).sum(), 10, 0, "Table 1")
    ck.published("young adults median MiniBESTest", y.MiniBest.median(), 28, 0.5, "Table 1")
    ck.published("older adults median MiniBESTest", o.MiniBest.median(), 26, 0.5, "Table 1")
    ck.published("young adults median ABC", y.ABC.median(), 96, 0.5, "Table 1")
    ck.published("older adults median ABC", o.ABC.median(), 95, 0.5, "Table 1")
    ck.published("equivalence points per participant (7 magnitudes x 2 sides x 2 days)", dt.groupby("ID").size().iloc[0], 28, 0,
                 "Methods, Two-alternative forced-choice task (14 per visit, two visits)")
    meta = dict(
        study=DS,
        paper=dict(authors="Jain S, Schweighofer N, Finley JM", year=2025,
                   title="Perceived severity of forward versus backward balance disturbances during walking in young and older adults",
                   venue="PLoS ONE 20(11):e0317943", doi="10.1371/journal.pone.0317943"),
        lab="Finley (University of Southern California)", data_url="https://osf.io/n6qcy/",
        license="none stated", license_url="https://osf.io/n6qcy/",
        changes=("Combined four per-participant tables into one long table with one row per participant, measure and (where relevant) "
                 "session, perturbation direction and forward magnitude; sex was coded F/M and age moved to the core columns; ethnicity "
                 "and race were not released; participant IDs were re-coded."),
        how=["One row is one measure of one participant. The main measure is equally_severe_backward_magnitude: for each forward "
             "(trip-like) belt perturbation magnitude in pert_value (0.30 to 0.60 m/s in 0.05 steps, shown in pert_unit) a staircase "
             "found the backward (slip-like) magnitude, in m/s, judged equally severe, on the left and right side (condition) on "
             "day 1 and day 2 (session): 28 rows per participant. The same table holds clinical scales, falls, state anxiety after "
             "trips and after slips, and peak centre-of-mass velocity deviations (37 participants), each in its own measure rows.",
             "Units are as the paper states them (m/s for perturbation magnitudes). The unit of the centre-of-mass velocity deviation "
             "is inferred (m/s) because the deposit does not state it, and activity is in hours per week as in the deposit. 'Side' in "
             "the staircase table is kept as the left or right of the deposit; the deposit does not say more.",
             "Peak_VCOM_Means.csv lacks 3 participants; no rows exist for them."],
        unit_note="measure of one participant",
        differences=["The paper prints a mean age of 27 for the young adults; the deposit gives %.2f (the paper seems to truncate), so that number is not used as a check." % y.Age.mean(), "Paper reports weekly activity in minutes (284 +- 216 and 427 +- 296); the deposit's hours column is rounded, so it was not used as a check."])
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
