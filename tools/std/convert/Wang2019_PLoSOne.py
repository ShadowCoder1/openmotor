"""Wang, Bao & Tays (2019) PLoS ONE: lack of generalization between explicit and implicit visuomotor learning.
Raw (OSF j5zt9, SPSS files read with pyreadstat; nothing executed): five files that the authors prepared for their analyses:
'Cycle data for grps 1to3' (cycle means, 27 participants), 'Cycle data for G2 vs 4_bsln adj_abs val_implicit only' (cycle-level direction
errors), 'Cycle data for grps 2n4_implicit only' (checked to repeat the previous file's direction-error columns),
'Trial data for grps 2n3_implicit only' (single trials) and 'Cycle data for G5 vs 6_no fb only' (no-feedback sessions)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wang2019_PLoSOne"
RAW = stdlib.raw_dir(DS)
PHASE = {"BS": "baseline", "EX": "learning", "IM": "learning", "NF": "test"}
SESS = {"BS": "baseline", "EX": "explicit session", "IM": "implicit session", "NF": "no visual feedback session"}
DEG = "deg"


def rd(name):
    df, _ = pyreadstat.read_sav(str(RAW / name))
    return df


def main():
    ck = stdlib.Checks()
    rows = []

    def add(exp, grp, subj, sess, trial, unit, name, desc, v, vu=DEG, n=None):
        if pd.isna(v):
            return
        rows.append(dict(dataset=DS, study="Wang 2019 PLoS ONE", experiment=exp, group=grp, subject=f"{exp}_{int(subj)}",
                         person=f"{exp}_{int(subj)}", subject_excluded_by_authors=False, age=None, sex=None, handedness="right",
                         population="healthy", setting="lab", device="tablet", session=None, block=SESS[sess], trial=int(trial),
                         phase=PHASE[sess], time_s=None, task="reaching", unit=unit, condition=SESS[sess],
                         pert_type="none" if sess in ("BS", "NF") else "rotation", pert_value=0 if sess in ("BS", "NF") else 30,
                         pert_unit=DEG, feedback="none" if sess == "NF" else ("normal" if sess == "BS" else "altered"),
                         excluded_by_authors=False, measure=name, measure_description=desc, statistic="mean" if unit != "trial" else "value",
                         value=float(v), value_unit=vu, n_trials=n))

    # cycle means, groups 1-3
    c = rd("Cycle data for grps 1to3.sav")
    for r in c.itertuples():
        g = f"group {int(r.Grp)}"
        add("Exp1_cycles", g, r.Subject, r.sess, r.cyc, "participant summary", "direction_at_peak_velocity",
            "Mean over the 6-trial cycle of the hand direction at peak velocity relative to the target (positive = right of target)", r.Dir_Peak_vel_mean, n=6)
        add("Exp1_cycles", g, r.Subject, r.sess, r.cyc, "participant summary", "abs_direction_at_peak_velocity",
            "Cycle mean of the absolute hand direction at peak velocity", r.Abs_DirVel_mean, n=6)
    ck.internal("1to3: 3 groups of 9 participants", c.groupby("Grp").Subject.nunique().tolist() == [9, 9, 9])
    ck.published("Exp 1 participants", c.Subject.nunique() if False else sum(c.groupby("Grp").Subject.nunique()), 27, 0, "Experiment 1, Participants")
    ck.published("baseline cycles", c[c.sess == "BS"].cyc.max(), 8, 0, "Experimental design (baseline 48 trials, 6-trial cycles)")
    ck.published("training cycles", c[c.sess == "EX"].cyc.max(), 20, 0, "Experimental design (120 trials, 6-trial cycles)")
    # cycle direction errors, groups 2 and 4
    a = rd("Cycle data for G2 vs 4_bsln adj_abs val_implicit only.sav")
    b = rd("Cycle data for grps 2n4_implicit only.sav")
    same = all(np.allclose(a[k].to_numpy(float), b[k].to_numpy(float), equal_nan=True) for k in ("grp", "Subject", "cyc", "DE", "bsln", "DE_adj", "Abs_DE_adj"))
    ck.internal("2n4 file repeats the direction-error columns of the G2 vs 4 file (not released twice)", bool(same))
    for r in a.itertuples():
        g = f"group {int(r.grp)}"
        for nm, d, v in (("DE", "Cycle direction error (hand direction at peak velocity minus target direction; positive = right of target)", r.DE),
                         ("baseline_DE", "Baseline direction error subtracted for this participant", r.bsln),
                         ("DE_adj", "Baseline-adjusted direction error", r.DE_adj),
                         ("abs_DE_adj", "Absolute baseline-adjusted direction error", r.Abs_DE_adj)):
            add("ImplicitSession_G2vsG4", g, r.Subject, r.sess, r.cyc, "participant summary", nm, d, v, n=6)
    # trials
    t = rd("Trial data for grps 2n3_implicit only.sav")
    for r in t.itertuples():
        g = f"group {int(r.Group)}"
        add("ImplicitSession_trials", g, r.Subject, "IM", r.Trialnumber, "trial", "direction_at_peak_velocity",
            "Hand direction at peak velocity relative to the target on this implicit-session trial (positive = right of target)", r.Dir_Peak_vel)
        add("ImplicitSession_trials", g, r.Subject, "IM", r.Trialnumber, "trial", "abs_direction_at_peak_velocity",
            "Absolute hand direction at peak velocity", r.Abs_DirVel)
    t1 = t[t.Trialnumber == 1].groupby("Group").Dir_Peak_vel.mean()
    ck.published("implicit session trial 1 mean direction error, group 2 (Imp-to-Exp)", t1[2.0], -27.7, 0.05, "Results, Experiment 1")
    ck.published("implicit session trial 1 mean direction error, group 4 (Exp-to-Imp/Inst)", t1[4.0], -21.7, 0.05, "Results, Experiment 1")
    # no-feedback
    n = rd("Cycle data for G5 vs 6_no fb only.sav")
    for r in n.itertuples():
        add("NoFeedback_G5vsG6", f"group {int(r.grp)}", r.Subject, "NF", r.Cyc, "participant summary", "direction_at_peak_velocity",
            "Cycle mean hand direction at peak velocity relative to the target without visual feedback", r.Dir_Peak_vel, n=6)
    ck.internal("no-feedback file: 6 participants in each of two groups", n.groupby("grp").Subject.nunique().tolist() == [6, 6])
    ck.published("Exp 2 participants in the Imp-to-NoVFB and Exp-to-NoVFB groups", n.Subject.nunique(), 12, 0, "Experiment 2, Experimental design (2 groups of 6)")
    ck.internal("rows written", len(rows) == int(len(rows)) and len(rows) > 2500, str(len(rows)))
    meta = dict(
        study=DS,
        paper=dict(authors="Wang J, Bao S, Tays GD", year=2019, title="Lack of generalization between explicit and implicit visuomotor learning",
                   venue="PLoS ONE 14(10):e0224099", doi="10.1371/journal.pone.0224099"),
        lab="Wang (University of Wisconsin-Milwaukee)", data_url="https://osf.io/j5zt9/", license="none stated", license_url="",
        changes=("Read the authors' SPSS tables, turned each into a long table (a row per participant, cycle or trial, and measure) in one file, "
                 "and replaced the subject numbers with codes. Columns without documented units (RT, MT, peak velocity and its time) were not carried over."),
        how=["Participants reached to six targets 10 cm away with a 30 deg cursor rotation (implicit session) or a veridical cursor and an instruction "
             "to aim 30 deg away (explicit session); a baseline session of 8 cycles precedes. One cycle is 6 trials (one per target). "
             "Directions are in deg at peak hand velocity, positive when to the right of the target (paper, Data analysis).",
             "The deposit is five analysis files, not one table. Group is the authors' number in each file and the numbers are not consistent "
             "between files: in the trial file groups 2 and 4 are the paper's Imp-to-Exp and Exp-to-Imp/Inst (trial-1 means match the paper), "
             "and the same subjects appear as groups 2 and 3 of the cycle file. Subject numbers were not merged across files, so one participant can "
             "appear under more than one experiment code. The authors do not document the group names; the mapping is our reading of the paper.",
             "Age and sex were not shared per participant; the paper reports 27 participants in Experiment 1 and 18 in Experiment 2."],
        unit_note="measure of one participant in one cycle (6 trials) or one trial",
        differences=["The 'grps 2n3' trial file is labelled groups 2 and 4 inside, and holds 11 and 9 participants (8 with a first trial) in them, "
                     "not the 9 per group of the paper; released as is.",
                     "The file 'Cycle data for grps 2n4' repeats the direction-error columns of 'G2 vs 4' and was not released twice."])
    stdlib.release(DS, "summary", rows, meta, ck)
    for c_ in ck.items:
        if c_["kind"] == "published":
            print(f"  {c_['what']}: {c_['computed']:.4f} vs {c_['reported']} ok={c_['ok']}")


if __name__ == "__main__":
    main()
