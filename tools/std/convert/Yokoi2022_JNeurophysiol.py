"""Yokoi & Weiler (2022), J Neurophysiol 128(5):1224-1243: pupil diameter tracked during force-field adaptation. Four
experiments (28, 29, 30 and 34 participants) of straight reaches with a Phantom robot in a velocity-dependent curl field with
20-28% channel trials; one row per reach in the force-field reaching format, with the authors' trial measures (perpendicular
error at peak velocity, endpoint error, force over velocity, pupil baseline and dilation, RT, MT, peak velocity).

Run from the site folder: python3 -I tools/std/convert/Yokoi2022_JNeurophysiol.py
Raw files (untrusted data, read only) in raw_extra/Yokoi2022_JNeurophysiol/, from https://github.com/ayokoi/pupilMotor01
(branch master, folder data): Exp1_trial.mat ... Exp4_trial.mat (plain MATLAB 5 files with numeric columns, read with scipy).
Not converted: the pupil time-course files (Exp1/Exp2 *_pupilTs_*.mat: alignment and averaging are not documented in the
deposit) and the Exp3/Exp4 interview files (one recall rating per participant and block).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Yokoi2022_JNeurophysiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
B_FULL = {1: 0.15, 2: 0.12, 3: 0.12, 4: 0.12}  # field coefficient B at full strength (Methods)
COND = {1: "abrupt_step", 2: "abrupt_reversals", 3: "gradual_ramp", 4: "gradual_ramp"}


def load(e):
    m = sio.loadmat(str(RAW / f"Exp{e}_trial.mat"))
    return pd.DataFrame({k: v.ravel() for k, v in m.items() if not k.startswith("__")})


def num(x):
    return "" if x is None or x != x else float(x)


rows = []
frames = {}
for e in (1, 2, 3, 4):
    d = load(e)
    d = d.assign(exp=e)
    frames[e] = d
    for s, g in d.groupby("subj", sort=True):
        g = g.sort_values("trial")
        assert (g.trial.values == np.arange(1, len(g) + 1)).all()
        ff = g.ffCoeff.values
        ch = g.channel.values == 1
        on = np.flatnonzero((~ch) & (ff != 0))
        first, last = on.min(), on.max()
        ccw = (ff.min() < 0) if e != 2 else None
        if e == 1:
            grp = "CW" if g.group.iloc[0] == 1 else "CCW"
        elif e == 2:
            grp = "ABC"[int(g.group.iloc[0]) - 1]
        else:
            grp = "CCW" if ccw else "CW"
        for i, r in enumerate(g.itertuples(index=False)):
            ttype = "channel" if r.channel == 1 else ("field" if r.ffCoeff != 0 else "null")
            ph = "baseline" if i < first else ("learning" if i <= last else "aftereffect")
            row = dict(
                dataset=ID, study=ID, experiment=f"Exp{e}", group=grp, subject=f"E{e}_s{int(s)}", person=f"E{e}_s{int(s)}",
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device="robotic manipulandum (Phantom Premium) with eye tracker", session=1, block=f"block{int(r.BN)}",
                trial=i + 1, phase=ph, time_s="", task="reaching", unit="trial", condition=COND[e],
                pert_type={"field": "force_field", "channel": "channel", "null": "none"}[ttype],
                pert_value=(float(r.ffCoeff) * B_FULL[e]) if ttype == "field" else 0,
                pert_unit="B of the curl field (positive clockwise; unit not given in the paper)" if ttype == "field" else "",
                feedback="masked", excluded_by_authors=False,
                trial_type=ttype, trial_in_condition=i + 1, block_in_condition=int(r.BN),
                movement_duration_s=num(r.MT / 1000.0), reaction_time_s=num(r.RT / 1000.0),
                peak_velocity_error_cm=num(r.PVE), endpoint_error_cm=num(r.EPE), force_over_velocity=num(r.FoverV),
                pupil_baseline_norm=num(r.pupwait), pupil_dilation_rate_norm=num(r.pupdilation),
                peak_velocity_y_cm_s=num(r.peakVy), field_coefficient_signed=float(r.ffCoeff),
                field_ramp_level=float(r.ptb) if e in (3, 4) else "",
            )
            rows.append(row)
df = pd.DataFrame(rows)

# ------------------------------------------------------------------ published checks (numbers from the paper's Methods)
per = df.groupby("experiment").subject.nunique()
for e, n, src in ((1, 28, "Table of experiments / Methods, Exp 1 (28 participants)"), (2, 29, "Methods, Exp 2 (29 analysed of 30)"),
                  (3, 30, "Methods, Exp 3 (30 participants)"), (4, 34, "Methods, Exp 4 (34 analysed of 35)")):
    checks.published(f"Exp{e} participants", float(per[f"Exp{e}"]), n, 0, src)
g1 = df[df.experiment == "Exp1"].groupby("subject").group.first().value_counts()
checks.published("Exp1 participants with a CW field", float(g1["CW"]), 20, 0, "Methods, Exp 1 (20 CW, 8 CCW)")
checks.published("Exp1 participants with a CCW field", float(g1["CCW"]), 8, 0, "Methods, Exp 1 (20 CW, 8 CCW)")
g4 = df[df.experiment == "Exp4"].groupby("subject").group.first().value_counts()
checks.published("Exp4 participants with a CW field", float(g4["CW"]), 19, 0, "Methods, Exp 4 (19 CW, 15 CCW)")
checks.published("Exp4 participants with a CCW field", float(g4["CCW"]), 15, 0, "Methods, Exp 4 (19 CW, 15 CCW)")
g3 = df[df.experiment == "Exp3"].groupby("subject").group.first().value_counts()
checks.published("Exp3 participants with a CCW field (half of 30)", float(g3["CCW"]), 15, 0, "Methods, Exp 3 (half CW, half CCW)")
g2 = df[df.experiment == "Exp2"].groupby("subject").group.first().value_counts()
for k, n in (("A", 10), ("B", 10), ("C", 9)):
    checks.published(f"Exp2 subgroup {k} participants", float(g2[k]), n, 0, "Methods, Exp 2 (subgroups A 10, B 10, C 9)")
d1 = df[df.experiment == "Exp1"]
sizes = d1[d1.subject == d1.subject.iloc[0]].groupby("block_in_condition").size().tolist()
for b, (got, want) in enumerate(zip(sizes, (55, 55, 40, 55, 55)), start=1):
    checks.published(f"Exp1 block {b} reaching trials", float(got), want, 0,
                     "Methods, Exp 1 (59 trials per block, 44 in block 4, minus the 4 light-reflex trials of each block)")
checks.published("Exp1 share of channel trials", float((d1.trial_type == "channel").mean()), 0.20, 0.0005, "Methods, Exp 1 (20% channel trials)")
d3 = df[df.experiment == "Exp3"]
checks.published("Exp3 share of channel trials", float((d3.trial_type == "channel").mean()), 0.20, 0.0005, "Methods, Exp 3 (20% channel trials)")
checks.published("Exp3 reaching trials per participant (7 blocks of 50)", float(len(d3) / 30), 350, 0, "Methods, Exp 3 (7 x 50 trials)")
d4 = df[df.experiment == "Exp4"]
checks.published("Exp4 reaching trials per participant (6 blocks of 50 plus a block of 70)", float(len(d4) / 34), 370, 0,
                 "Methods, Exp 4 (block 7 has 70 trials)")
b7 = d4[d4.block_in_condition == 7]
checks.published("Exp4 block 7 channel trials per participant (all of trials 21-70 plus 20% of the first 20)", float((b7.trial_type == "channel").sum() / 34), 54, 0,
                 "Methods, Exp 4 (20% channel trials, block 7 channel on trials 21-70)")
d2 = df[df.experiment == "Exp2"]
sched = {"A": ([(61, 89), (101, 129), (151, 159), (170, 179), (190, 209), (220, 229)],
               [(90, 100), (130, 150), (160, 169), (180, 189), (210, 219), (230, 239)]),
         "B": ([(70, 79), (90, 109), (120, 129), (140, 189), (201, 229)],
               [(61, 69), (80, 89), (110, 119), (130, 139), (190, 200), (230, 239)]),
         "C": ([(61, 159)], [(160, 170)])}
mism = 0
for k, (cw, ccw) in sched.items():
    exp = np.zeros(251)
    for a, b in cw:
        exp[a:b + 1] = 1
    for a, b in ccw:
        exp[a:b + 1] = -1
    x = d2[(d2.group == k) & (d2.trial_type != "channel")]
    mism += int((np.sign(x.field_coefficient_signed.astype(float).values) != exp[x.trial.values]).sum())
checks.published("Exp2 non-channel trials whose field direction differs from the schedule printed in the Methods (A, B, C)",
                 float(mism), 0, 0, "Methods, Exp 2 (trial ranges of CW and CCW fields for subgroups A, B, C)")
# internal checks
checks.internal("row counts: 28*260, 29*250, 30*350, 34*370", [len(df[df.experiment == f"Exp{e}"]) for e in (1, 2, 3, 4)] ==
                [7280, 7250, 10500, 12580])
checks.internal("every participant has consecutive trials from 1", df.groupby("subject").trial.apply(lambda t: (t.values == np.arange(1, len(t) + 1)).all()).all())
checks.internal("on non-channel trials the field coefficient is non-zero exactly on field trials",
                (df[df.trial_type == "null"].field_coefficient_signed == 0).all() and (df[df.trial_type == "field"].field_coefficient_signed != 0).all())
ff1 = d1[(d1.trial_type != "channel") & (d1.subject == d1.subject.iloc[0])]
checks.internal("Exp1 field timing: block 2 field from its 11th reach, off from the 11th reach of block 4",
                int((ff1[ff1.block_in_condition == 2].trial_type == "field").sum()) > 0
                and (ff1[ff1.block_in_condition == 3].trial_type == "field").all()
                and (ff1[ff1.block_in_condition == 1].trial_type == "null").all()
                and (ff1[ff1.block_in_condition == 5].trial_type == "null").all())
checks.internal("no participant lost: all four experiments keep every participant of the deposit", len(set(df.subject)) == 28 + 29 + 30 + 34)

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Yokoi (NICT Osaka) / Weiler (Western)", data_url="https://github.com/ayokoi/pupilMotor01",
    license="none stated", license_url="",
    changes="four trial tables turned into one table with one row per reach; phases and trial types derived from the "
            "authors' field-coefficient and channel columns; times converted from ms to s; participant codes replaced.",
    unit_note="reach of one participant sequence",
    how=[
        "One row is one reach of one participant, in the authors' order: Experiment 1 (28 participants, 260 reaches, an "
        "abrupt curl field from the 11th reach of block 2 to the 10th of block 4), Experiment 2 (29 participants, 250 reaches, "
        "abrupt fields with reversals, three subgroups A, B, C), Experiment 3 (30 participants, 350 reaches, a field ramped "
        "up gradually and then removed) and Experiment 4 (34 participants, 370 reaches, the same ramp followed by 70 channel "
        "reaches). One in five reaches (more in Experiments 2C and 4) is a channel trial; the 4 light-reflex trials per block "
        "of the original task are not in the deposit.",
        "trial_type is channel where the authors' channel flag is 1, field where their field coefficient is not 0, null "
        "otherwise; phase is baseline before the first non-channel field trial, learning up to the last one, aftereffect "
        "after it. The measures keep the authors' names and values: peak_velocity_error_cm is the sideways error at peak "
        "velocity, force_over_velocity the lateral force over speed at peak velocity (their learning measure on channel "
        "trials), and the pupil columns the normalized baseline diameter and dilation rate.",
        "Signs. The paper states that positive errors and learning indices are rightward and that fields are clockwise (B "
        "positive) or counter-clockwise (B negative), and that the errors and learning indices of participants who trained "
        "with a counter-clockwise field were sign-flipped in Experiments 1, 3 and 4. The deposit agrees: group CW/CCW follows "
        "the sign of the authors' field coefficient (Experiments 1, 3, 4), and the error columns are in the flipped frame "
        "(so a field pushes toward positive values for everyone). In Experiment 2 nothing is flipped and the field "
        "direction changes within the sequence (field_coefficient_signed shows it).",
        "Units are inferred, not stated in the deposit: distances in cm (target 10-12 cm and error values of that size), "
        "velocities in cm/s, RT and MT in ms in the deposit and in s here. Saccade trials the authors excluded (5-17% of "
        "trials) are not marked in the deposit, so excluded_by_authors is False on every row; pupil dilation is empty "
        "where the authors have no value. The pupil time-course files and interview ratings are not included.",
    ],
    differences=["The paper's table gives 59 reaching-plus-light-reflex trials per block; the deposit holds only the 55 "
                 "reaching trials of each block, as the published check on block sizes shows."],
)
for c in stdlib.columns_for("force_field_reaching"):
    if c[1] not in df.columns and c[1] != "person":
        df[c[1]] = ""
df = df[[c[1] for c in stdlib.columns_for("force_field_reaching")]]
stdlib.release(ID, "force_field_reaching", df, meta, checks)
