"""Killeen et al. (2017) R Soc open sci: cognitive load and arm swing in treadmill walking.

Raw files: raw_extra/Killeen2017_RSOS/final_for_dryad.xls (Sheet1: 363 rows = 121 participants x 3 conditions, 117 columns) and
README.txt (Zenodo copy 4977170 of the Dryad deposit 10.5061/dryad.2kd0b). Nothing is executed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Killeen2017_RSOS"
COND = {0: "normal walking", 1: "congruent Stroop", 2: "incongruent Stroop"}
SKIP = {"Variable Name", "Subject", "Condition", "Age Group", "STARTS>-/+20", "Age Group 2", "Gender", "Handedness", "Age"}
UNITS = {"LWR_Traj": "mm", "RWR_Traj": "mm", "TRAJ ASI": "percent", "ABS_TRAJ_ASI": "percent", "Weight": "kg", "Height": "cm"}
DESC = {
    "LWR_Traj": "Left wrist trajectory travelled in 3D (mm)", "RWR_Traj": "Right wrist trajectory travelled in 3D (mm)",
    "TRAJ ASI": "Arm swing asymmetry index from the left and right wrist trajectories (main outcome; positive = smaller right arm movement)",
    "ABS_TRAJ_ASI": "Absolute value of the arm swing asymmetry index", "ARM_SYMM": "Arm length symmetry index",
    "25FWT1": "25-foot walk test, first attempt", "25FWT2": "25-foot walk test, second attempt",
    "10MWT1": "10 m walk test, first attempt", "10MWT2": "10 m walk test, second attempt", "6MWT@6": "6 minute walk test",
    "L_MAX_PROTRACTION": "Maximal protraction of the left upper arm in the sagittal plane",
    "R_MAX_PROTRACTION": "Maximal protraction of the right upper arm in the sagittal plane",
    "L_MAX_RETRACTION": "Maximal retraction of the left upper arm in the sagittal plane",
    "R_MAX_RETRACTION": "Maximal retraction of the right upper arm in the sagittal plane",
    "Weight": "Body weight", "Height": "Body height",
}


def main():
    d = pd.read_excel(stdlib.raw_dir(DS) / "final_for_dryad.xls", sheet_name=0)
    num = [c for c in d.columns if c not in SKIP and not str(c).startswith("Unnamed") and pd.api.types.is_numeric_dtype(d[c])
           and d[c].notna().any()]
    ck = stdlib.Checks()
    ck.internal("363 rows, 121 participants x 3 conditions", len(d) == 363 and d.Subject.nunique() == 121
                and (d.groupby("Subject").size() == 3).all())
    first = d[d.Condition == 0].set_index("Subject")
    incl = first["STARTS>-/+20"] == 1
    ck.published("volunteers after medical screening (145 - 24)", d.Subject.nunique(), 121, 0, "Results, first paragraph")
    ck.published("excluded for baseline arm swing asymmetry (8 + 28)", int((~incl).sum()), 36, 0, "Results, first paragraph")
    ag = first.loc[incl].groupby("Age Group")
    ck.published("included aged 18-39", int(ag.size()[0]), 31, 0, "Results / Table 1")
    ck.published("included aged 60-80", int(ag.size()[2]), 29, 0, "Results / Table 1")
    ck.published("mean age 18-39 (years)", ag.Age.mean()[0], 28.6, 0.05, "Table 1")
    ck.published("mean age 60-80 (years)", ag.Age.mean()[2], 67.8, 0.05, "Table 1")
    ck.published("SD age 18-39 (years)", ag.Age.std()[0], 4.9, 0.05, "Table 1")
    ck.published("SD age 60-80 (years)", ag.Age.std()[2], 4.5, 0.05, "Table 1")
    rows = []
    for t, r in enumerate(d.itertuples(index=False), 1):
        r = r._asdict() if hasattr(r, "_asdict") else r
    for _, r in d.iterrows():
        ex = r["STARTS>-/+20"] != 1
        base = dict(dataset=DS, study="Killeen 2017 R Soc open sci", experiment="treadmill walking with Stroop task", group=None,
                    subject=r.Subject, person=r.Subject, subject_excluded_by_authors=bool(ex), age=r.Age,
                    sex=r.Gender, handedness={"R": "right", "L": "left"}.get(r.Handedness), population="healthy", setting="lab",
                    device="treadmill with 3D motion capture", session=1, block=None, trial=int(r.Condition) + 1, phase="test",
                    time_s=None, task="walking", unit="participant summary", condition=COND[int(r.Condition)],
                    pert_type="none", pert_value=0, pert_unit=None, feedback="normal", excluded_by_authors=bool(ex))
        for c in num:
            v = r[c]
            if pd.isna(v):
                continue
            rows.append({**base, "measure": str(c), "measure_description": DESC.get(c, f"Column '{c}' of the authors' spreadsheet (see their README)"),
                         "statistic": "value", "value": v, "value_unit": UNITS.get(c, "")})
    ck.internal("every non-empty numeric cell kept", len(rows) == int(d[num].notna().sum().sum()), str(len(rows)))
    meta = dict(
        study=DS,
        paper=dict(authors="Killeen T, Easthope CS, Filli L, Lorincz L, Schrafl-Altermatt M, Brugger P, Linnebank M, Curt A, Zorner B, Bolliger M",
                   year=2017, title="Increasing cognitive load attenuates right arm swing in healthy human walking",
                   venue="R Soc open sci 4(1):160993", doi="10.1098/rsos.160993"),
        lab="Curt / Bolliger (Balgrist, Zurich)", data_url="https://doi.org/10.5061/dryad.2kd0b (copy: https://doi.org/10.5281/zenodo.4977170)",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes="Turned the spreadsheet into a long table with one row per participant, condition and measure, replaced participant ids and file names with codes, dropped empty header columns and the audio-free 'Presentations' archive.",
        how=["One row is one measure of one participant in one walking condition (normal walking, congruent Stroop, incongruent Stroop) on a treadmill. Each participant has three recordings; values are the authors' per-recording summaries of the motion-capture data (wrist trajectories in mm, angles, ranges of motion, spatiotemporal and phase measures), as in their spreadsheet.",
             "The README says units only for some columns (wrist trajectory mm); other columns keep the authors' names and the unit is left empty rather than guessed. Weight and height are listed as measures (kg and cm are assumed from the values). Age, sex and handedness are in the standard columns.",
             "The authors' main analysis included 83 participants whose baseline arm swing asymmetry index was between -20 and +20. The 36 others are flagged as excluded (STARTS>-/+20 not 1). Two further participants whose data were unusable cannot be told apart in the spreadsheet, so 85 participants are unflagged."],
        unit_note="measure of one participant in one condition",
        differences=["The paper includes 83 participants (23 aged 40-59); the spreadsheet has 85 unflagged participants (25 aged 40-59) because the 2 participants with unusable data (Results) are not marked.",
                     "Mean baseline ASI of the whole cohort is 6.16 in the paper; the spreadsheet gives 6.09 for the 121 participants (not used as a check)."],
    )
    stdlib.release(DS, "summary", rows, meta, ck)


if __name__ == "__main__":
    main()
