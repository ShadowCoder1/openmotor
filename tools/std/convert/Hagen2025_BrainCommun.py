"""Hagen, Whittier, Stephens & Fling (2025) Brain Commun 7(4):fcaf255: TENS and split-belt locomotor adaptation savings in
28 people with multiple sclerosis and 20 healthy controls, two visits each (TENS on / off). The deposit
(OSF pgv37, RepositoryData.xlsx, CC BY 4.0 per the paper) shares per participant and visit the step length asymmetry (SLA)
averaged over seven stride windows; there are no stride-level data. One row is one SLA value (long form).
The fNIRS beta sheets and the brain-behaviour correlation sheet are not motor behaviour data and are not converted.
Run from the site folder: python3 -I tools/std/convert/Hagen2025_BrainCommun.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Hagen2025_BrainCommun"
MODULE = "summary"
F = "RepositoryData.xlsx"
# column -> (block, phase, strides in the window, window text); windows from the paper (Methods, SLA analysis)
WIN = [("Baseline_SLA", "Baseline", "baseline", 30, "last 30 strides of the 2-min tied-belt baseline"),
       ("InitialAdapt_SLA", "Initial Adapt", "learning", 5, "strides 1-5 of split-belt adaptation"),
       ("EarlyAdapt_SLA", "Early Adapt", "learning", 25, "strides 6-30 of split-belt adaptation"),
       ("LateAdapt_SLA", "Late Adapt", "learning", 30, "last 30 strides of split-belt adaptation"),
       ("InitialDeadapt_SLA", "Initial Deadapt", "washout", 5, "strides 1-5 of tied-belt deadaptation"),
       ("EarlyDeadapt_SLA", "Early Deadapt", "washout", 25, "strides 6-30 of tied-belt deadaptation"),
       ("LateDeadapt_SLA", "Late Deadapt", "washout", 30, "last 30 strides of tied-belt deadaptation")]


def main():
    raw = stdlib.raw_dir(DS)
    ck = stdlib.Checks()
    dm = pd.read_excel(raw / F, sheet_name="Demographics")
    sla = pd.read_excel(raw / F, sheet_name="Step length asymmetry (SLA)")
    dm = dm.set_index("Participant")
    rows = []
    for r in sla.itertuples(index=False):
        d = dm.loc[r.Participant]
        grp = {"HC": "Healthy control", "MS": "Multiple sclerosis"}[r.Group]
        visit = int(str(r.Visit).split()[-1])
        for col, block, phase, n, win in WIN:
            split = phase == "learning"
            rows.append(dict(
                dataset=DS, study=DS, experiment="", group=grp, subject=r.Participant, subject_excluded_by_authors=False,
                age=float(d["Age"]), sex={"Female": "F", "Male": "M"}[d["Sex"]], handedness="",
                population="healthy adults" if r.Group == "HC" else "people with multiple sclerosis", setting="lab",
                device="Instrumented split-belt treadmill", session=visit, block=block, trial=len(rows) % 7 + 1, phase=phase,
                time_s=np.nan, task="walking", unit="participant summary", condition=r.Condition,
                pert_type="split_belt" if split else "none", pert_value=2.0 if split else 0.0, pert_unit="ratio" if split else "",
                feedback="normal", excluded_by_authors=False, measure="step_length_asymmetry",
                measure_description=f"Step length asymmetry (fast minus slow step length over their sum, dimensionless) over the {win}; "
                                    f"fast leg {d['Fast Leg']}.",
                statistic="mean", value=getattr(r, col), value_unit="proportion", n_trials=n))
    df = pd.DataFrame(rows)
    np_ = df.groupby("group").subject.nunique().to_dict()
    ck.internal("96 participant-visits x 7 windows = 672 rows", len(df) == 672, str(len(df)))
    ck.internal("every value present", df.value.notna().all())
    ck.published("people with MS analysed", np_["Multiple sclerosis"], 28, 0, "Results, Participants")
    ck.published("healthy controls analysed", np_["Healthy control"], 20, 0, "Results, Participants")
    per = df.drop_duplicates("subject").groupby("group")
    ck.published("MS mean age", per.age.mean()["Multiple sclerosis"], 53.5, 0.05, "Results, Participants (53.5 +- 10.7)")
    ck.published("MS age SD", per.age.std()["Multiple sclerosis"], 10.7, 0.05, "Results, Participants")
    ck.published("HC mean age", per.age.mean()["Healthy control"], 53.8, 0.05, "Results, Participants (53.8 +- 14.0)")
    ck.published("HC age SD", per.age.std()["Healthy control"], 14.0, 0.05, "Results, Participants")
    sx = df.drop_duplicates("subject").groupby(["group", "sex"]).size()
    ck.published("MS women", sx[("Multiple sclerosis", "F")], 19, 0, "Results, Participants (19/9)")
    ck.published("HC women", sx[("Healthy control", "F")], 12, 0, "Results, Participants (12/8)")
    base = sla.groupby("Group").Baseline_SLA
    ck.internal("each participant has one TENS on and one TENS off visit", (sla.groupby("Participant").Condition.nunique() == 2).all())
    df["trial"] = df.groupby("subject", sort=False).cumcount() + 1
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in df.columns:
            df[c] = np.nan
    meta = dict(
        study=DS,
        paper=dict(authors="Hagen AC, Whittier TT, Stephens JA, Fling BW", year=2025,
                   title="Transcutaneous electrical nerve stimulation enhances locomotor adaptation savings in people with multiple sclerosis",
                   venue="Brain Communications 7(4):fcaf255", doi="10.1093/braincomms/fcaf255"),
        lab="Fling (Colorado State University)", data_url="https://osf.io/pgv37/",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Read the Demographics and Step length asymmetry sheets of RepositoryData.xlsx, made one row per participant, "
                 "visit and stride window (long form), added age and sex from the demographics sheet, and left out the "
                 "visit dates, the clinical scores, the fNIRS sheets and the correlation sheet."),
        how=["One row is one participant's step length asymmetry in one stride window of one visit: (fast step length minus slow "
             "step length) over their sum, dimensionless, averaged over the window (the authors' Baseline, Initial/Early/Late "
             "Adapt and Initial/Early/Late Deadapt columns; n_trials is the window length in strides).",
             "Each participant came twice, 4 weeks apart: TENS on in one visit and off in the other (condition); the order is "
             "given by session. Adaptation is 10 min of split-belt walking at a 2:1 speed ratio, baseline and deadaptation are "
             "tied-belt walking at the preferred speed.",
             "The deposit shares no stride-level data, so these are the authors' window averages; the clinical scores and the "
             "fNIRS and correlation sheets are not included."],
        unit_note="participant summary (one stride window of one visit)",
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for c in ck.items:
        print(("  P " if c["kind"] == "published" else "  I ") + c["what"][:80], c.get("computed", ""), c.get("reported", ""), c.get("detail", "")[:150])


if __name__ == "__main__":
    main()
