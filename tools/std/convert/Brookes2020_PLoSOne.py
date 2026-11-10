"""Brookes et al. (2020), PLoS ONE: haptic assistance / disruption during motor learning, two experiments, per-participant
summaries (path-accuracy improvement, training information, exponential-fit a and b, and in Experiment 2 per-session stiffness,
number of crossovers and mean stiffness change). Format: summary (one row = one measure of one participant).

Run from the site folder: python3 -I tools/std/convert/Brookes2020_PLoSOne.py
Raw files (read only) in raw_extra/Brookes2020_PLoSOne/ from https://osf.io/7c95b/:
exp1_summary.csv, exp1_curvefitdata.xlsx, exp2_summary.csv, exp2_curvefitdata.xlsx, exp2_stiffness.csv
"""
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Brookes2020_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

GROUP = {  # file code -> (authors' file label with the paper's name, pert value in N/m or None)
    1: {"Control": ("Control (Active-Control, k = 0 N/m)", 0.0), "Convergence": ("Convergence (paper: Assistance, k = 100 N/m)", 100.0),
        "Divergence": ("Divergence (paper: Disturbance/Disruption, k = -100 N/m)", -100.0)},
    2: {"AA": ("AA (paper: Adaptive Algorithm)", None), "ADA": ("ADA (paper: Adaptive Disruptive)", None),
        "RAN": ("RAN (paper: Random)", None)},
}
DESC = {
    "path_accuracy_improvement": ("Pre- to post-training improvement in mean path error (the paper's 'motor learning' score, Figs 4B and 7B); larger is more improvement.", "mean", "mm?"),
    "training_information": ("Cumulative (total weighted) information sampled during training (the paper's 'cumulative information', Figs 3 and 6D).", "value", "score"),
    "expfit_a": ("Parameter a of the first-order exponential y = a*exp(b*x) fitted to average path error per training block.", "value", "path error"),
    "expfit_b": ("Parameter b (rate) of the first-order exponential y = a*exp(b*x) fitted to average path error per training block.", "value", "per block"),
    "stiffness": ("Mean haptic stiffness k delivered in the session (positive assists, negative disrupts; clamped to -100..100 N/m).", "mean", "N/m"),
    "crossovers": ("Number of times the task switched between assistive and disruptive stiffness in the session.", "count", "count"),
    "mean_diff": ("Mean trial-to-trial change in stiffness in the session (file column mean_diff; the paper's 'trial-to-trial stiffness change').", "mean", "N/m"),
}
UNIT = {"path_accuracy_improvement": "path error units of the paper (not stated in the deposit)", "training_information": "score",
        "expfit_a": "path error units of the paper (not stated in the deposit)", "expfit_b": "per training block",
        "stiffness": "N/m", "crossovers": "count", "mean_diff": "N/m"}

rows = []
persons = {}
order = {}


def add(exp, pid, grp, measure, value, session=None, in_summary=True):
    key = (exp, str(pid))
    persons.setdefault(key, f"E{exp}_{len(persons) + 1}")
    order[key] = order.get(key, 0) + 1
    lab, kval = GROUP[exp][grp]
    desc, stat, _ = DESC[measure]
    rows.append(dict(
        dataset=ID, study=ID, experiment=f"Experiment {exp}", group=lab, subject=f"{persons[key]}", person=f"{persons[key]}",
        subject_excluded_by_authors="", age="", sex="", handedness="right", population="healthy", setting="lab",
        device="haptic robotic device (end-effector path task)", session=session if session else "", block="", trial=order[key],
        phase="other", time_s="", task="reaching with haptic assistance or disruption", unit="participant summary",
        condition=(f"session {session}" if session else "whole training"),
        pert_type="haptic_stiffness" if kval is not None else "adaptive_or_random_stiffness",
        pert_value="" if kval is None else kval, pert_unit="N/m", feedback="normal", excluded_by_authors="False",
        measure=measure, measure_description=desc, statistic=stat, value=float(value), value_unit=UNIT[measure], n_trials="",
        session_context="",
        _g=grp, _s=session))


for exp in (1, 2):
    s = pd.read_csv(RAW / f"exp{exp}_summary.csv")
    c = pd.read_excel(RAW / f"exp{exp}_curvefitdata.xlsx", sheet_name="curvefitdata")
    for r in s.itertuples():
        add(exp, r.ppid, r.group, "path_accuracy_improvement", r.path_accuracy_improvement)
        add(exp, r.ppid, r.group, "training_information", r.training_information)
    for r in c.itertuples():
        add(exp, r.Participant, r.Group, "expfit_a", r.expfit_a)
        add(exp, r.Participant, r.Group, "expfit_b", r.expfit_b)
    if exp == 1:
        n1 = (len(s), len(c))
    else:
        n2 = (len(s), len(c))
stf = pd.read_csv(RAW / "exp2_stiffness.csv")
for r in stf.itertuples():
    for m in ("stiffness", "crossovers", "mean_diff"):
        add(2, r.participant_name, r.group, m, getattr(r, m), session=int(r.session_num))

df = pd.DataFrame(rows)
# ---- checks against the paper
def pp(exp, grp, meas, how="mean", sessions=False):
    d = df[(df.experiment == f"Experiment {exp}") & (df._g == grp) & (df.measure == meas)]
    per = d.groupby("person").value.agg("sum" if how == "sum" else "mean")
    return list(per)

S1, S2 = "Results, Experiment 1 ", "Results, Experiment 2 "
m = lambda x: st.mean(x)
sd = lambda x: st.stdev(x)
for g, n in (("Control", 16), ("Convergence", 15), ("Divergence", 17)):
    checks.published(f"Experiment 1 participants, {g}", len(pp(1, g, "training_information")), n, 0, "Methods, Participants (n = 16 / 15 / 17)")
for g, n in (("AA", 13), ("ADA", 17)):
    checks.published(f"Experiment 2 participants with summary data, {g}", len(pp(2, g, "training_information")), n, 0, "Methods, Participants (n = 13 / 17)")
for g, (mu, s_) in {"Control": (6118, 658), "Convergence": (5742, 772), "Divergence": (10466, 2823)}.items():
    x = pp(1, g, "training_information")
    checks.published(f"Exp 1 cumulative information mean, {g}", m(x), mu, 0.5, "Results, Exp 1, cumulative information (Fig 3)")
    checks.published(f"Exp 1 cumulative information SD, {g}", sd(x), s_, 0.5, "Results, Exp 1, cumulative information (Fig 3)")
for g, (mu, s_) in {"Control": (10.1, 5.18), "Convergence": (5.98, 1.94), "Divergence": (14.8, 9.72)}.items():
    x = pp(1, g, "expfit_a")
    checks.published(f"Exp 1 exponential a mean, {g}", m(x), mu, 0.05, "Results, Exp 1, exponential fit a")
    checks.published(f"Exp 1 exponential a SD, {g}", sd(x), s_, 0.005, "Results, Exp 1, exponential fit a")
for g, (mu, s_) in {"Control": (-0.022, 0.012), "Convergence": (-0.0172, 0.014), "Divergence": (-0.0447, 0.013)}.items():
    x = pp(1, g, "expfit_b")
    checks.published(f"Exp 1 exponential b mean, {g}", m(x), mu, 0.0005 if abs(mu) < 0.03 and len(str(mu)) < 7 else 0.00005, "Results, Exp 1, exponential fit b (Fig 4A)")
    checks.published(f"Exp 1 exponential b SD, {g}", sd(x), s_, 0.0005, "Results, Exp 1, exponential fit b (Fig 4A)")
for g, (mu, s_) in {"Control": (0.744, 1.58), "Divergence": (2.00, 1.02)}.items():
    x = pp(1, g, "path_accuracy_improvement")
    checks.published(f"Exp 1 motor learning mean, {g}", m(x), mu, 0.0005 if g == "Control" else 0.005, "Results, Exp 1, motor learning (Fig 4B)")
    checks.published(f"Exp 1 motor learning SD, {g}", sd(x), s_, 0.005, "Results, Exp 1, motor learning (Fig 4B)")
for g, (mu, s_) in {"AA": (-11.6, 19.9), "ADA": (17.5, 18.3), "RAN": (0.55, 5.59)}.items():
    x = pp(2, g, "stiffness")
    checks.published(f"Exp 2 mean stiffness mean, {g}", m(x), mu, 0.05 if g != "RAN" else 0.005, "Results, Exp 2, mean stiffness (Fig 5C)")
    checks.published(f"Exp 2 mean stiffness SD, {g}", sd(x), s_, 0.05 if g != "RAN" else 0.005, "Results, Exp 2, mean stiffness (Fig 5C)")
for g, (mu, s_) in {"AA": (5.27, 1.16), "ADA": (39.1, 22.4), "RAN": (62.6, 5.14)}.items():
    x = pp(2, g, "crossovers", "sum")
    checks.published(f"Exp 2 assist/disrupt switches (sum of 3 sessions), mean, {g}", m(x), mu, 0.05, "Results, Exp 2, switches (Fig 5B)")
    checks.published(f"Exp 2 assist/disrupt switches, SD, {g}", sd(x), s_, 0.05, "Results, Exp 2, switches (Fig 5B)")
for g, (mu, s_) in {"AA": (3.13, 0.624), "ADA": (23.5, 4.24), "RAN": (68.1, 5.83)}.items():
    x = pp(2, g, "mean_diff")
    checks.published(f"Exp 2 trial-to-trial stiffness change mean, {g}", m(x), mu, 0.05 if g != "AA" else 0.005, "Results, Exp 2 (Fig 5A)")
    checks.published(f"Exp 2 trial-to-trial stiffness change SD, {g}", sd(x), s_, 0.005 if g != "ADA" else 0.005, "Results, Exp 2 (Fig 5A)")
for g, (mu, s_) in {"AA": (5758, 459), "ADA": (5642, 665), "RAN": (6873, 741)}.items():
    x = pp(2, g, "training_information")
    if g != "ADA":
        checks.published(f"Exp 2 cumulative information mean, {g}", m(x), mu, 0.5, "Results, Exp 2, cumulative information (Fig 6D)")
    checks.published(f"Exp 2 cumulative information SD, {g}", sd(x), s_, 0.5, "Results, Exp 2, cumulative information (Fig 6D)")
for g, (mu, s_) in {"AA": (7.93, 1.37), "RAN": (8.86, 2.16)}.items():
    x = pp(2, g, "expfit_a")
    checks.published(f"Exp 2 exponential a mean, {g}", m(x), mu, 0.005, "Results, Exp 2, exponential fit a")
    checks.published(f"Exp 2 exponential a SD, {g}", sd(x), s_, 0.005, "Results, Exp 2, exponential fit a")
for g, (mu, s_) in {"AA": (-0.0084, 0.0198), "ADA": (-0.0065, 0.0276), "RAN": (-0.0231, 0.0205)}.items():
    x = pp(2, g, "expfit_b")
    checks.published(f"Exp 2 exponential b SD, {g}", sd(x), s_, 0.00005, "Results, Exp 2, exponential fit b (Fig 7A)")
    if g != "ADA":
        checks.published(f"Exp 2 exponential b mean, {g}", m(x), mu, 0.00005, "Results, Exp 2, exponential fit b (Fig 7A)")
x = pp(2, "RAN", "path_accuracy_improvement")
checks.published("Exp 2 motor learning mean, Random", m(x), 1.97, 0.005, "Results, Exp 2, motor learning (Fig 7B)")
checks.published("Exp 2 motor learning SD, Random", sd(x), 1.03, 0.005, "Results, Exp 2, motor learning (Fig 7B)")

checks.internal("every row of the five data files is kept", len(df) == 2 * n1[0] + 2 * n1[1] + 2 * n2[0] + 2 * n2[1] + 3 * len(stf),
                f"{len(df)} rows")
checks.internal("no missing or non-finite values", df.value.notna().all())
checks.internal("Experiment 1: 48 participants in both files", n1 == (48, 48), str(n1))
checks.internal("Experiment 2: 45 participants in the summary file, 51 in the curve-fit file, 44 with stiffness", n2 == (45, 51) and stf.participant_name.nunique() == 44)

out = df.drop(columns=["_g", "_s"])
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Mon-Williams & Mushtaq (University of Leeds)", data_url="https://osf.io/7c95b/",
    license="none stated", license_url="",
    changes="converted from five wide tables to the long OpenMotor summary format and participant codes replaced; no value was changed.",
    unit_note="measure of one participant (long form)",
    how=["One row is one measure of one participant: path_accuracy_improvement and training_information (summary files), expfit_a and expfit_b (curve-fit files), and, in Experiment 2, stiffness, crossovers and mean_diff for each of 3 sessions (stiffness file).",
         "Experiment 1 has 48 participants in 3 groups (Control, Convergence = the paper's Assistance, Divergence = Disturbance); Experiment 2 has 45 in the summary file, 51 in the curve-fit file and 44 in the stiffness file, in groups AA (Adaptive Algorithm), ADA (Adaptive Disruptive) and RAN (Random). Group labels are the file codes with the paper's names. Which group is which in Experiment 2 was read from the means (AA matches the paper's Adaptive stiffness, -11.6).",
         "Participant ids repeat between the two experiments (for example M1); they are treated as different people, since the experiments had different groups, so person codes are per experiment.",
         "Units are not stated in the deposit; value_unit names the quantity. The exponential fit was y = a*exp(b*block) on mean path error per training block. Deposit has no age, sex or per-trial data."],
    differences=[
        "Experiment 1, motor learning of the Convergence (Assistance) group: the file mean is 0.295 (SD 1.745); the paper prints 0.595 (1.74). The SD agrees, so the paper's mean looks like a typo; not used as a check.",
        "Experiment 2, motor learning: the paper's means 0.672 (Adaptive) and 0.726 (AD) are in the opposite order to the file (AA 0.726, ADA 0.672), while the paper's cumulative information and stiffness rows match the file with AA = Adaptive. Only the Random value is checked.",
        "Experiment 2, exponential a for ADA: the file mean is 7.51 (SD 0.93), the paper prints 9.51 (0.93); not used as a check.",
        "Experiment 2, ADA: the file's mean cumulative information is 5642.86 and mean b is -0.00656; the paper prints 5642 and -0.0065 (truncated rather than rounded), so these two means are not used as checks.",
        "The paper reports 47 (Exp 1) and 45 (Exp 2) analysed participants; the files contain 48 and 45 (summary) / 51 (curve-fit) participants, all of which are kept. The paper's F test degrees of freedom do not match its group sizes either.",
    ])
stdlib.release(ID, "summary", out, meta, checks)
