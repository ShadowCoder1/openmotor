"""Yeo, Wolpert & Franklin (2015), PLoS ONE: coordinate representations for interference reduction in motor learning.
24 participants, three within-participant conditions of 360 trials each (60 null pre-exposure, 200 curl-field exposure,
100 null post-exposure; blocks of 20 trials, 4 of them channel trials), one row per trial.

Run from the site folder: python3 -I tools/std/convert/Yeo2015_PLoSOne.py
Raw files (untrusted data, read only) in raw_extra/Yeo2015_PLoSOne/: mpe.csv and fcom.csv (1080 trials x 24 participants,
NaN = not that trial type) and order.csv (condition order per participant: 1 Cartesian, 2 joint angles, 3 hand orientation;
columns of the matrices follow the same participant order). The deposit's own description gives all of this; the paper
(Methods) gives the 360-trial structure and the channel-trial rule.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Yeo2015_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
COND = {1: "Cartesian", 2: "joint_angles", 3: "hand_orientation"}

order = pd.read_csv(RAW / "order.csv", header=None).iloc[0].astype(int).tolist()
mpe = pd.read_csv(RAW / "mpe.csv", header=None).values
fc = pd.read_csv(RAW / "fcom.csv", header=None).values
nP = len(order)
assert mpe.shape == fc.shape == (1080, nP)
chan = ~np.isnan(fc)
checks.internal("every trial has exactly one of MPE and force compensation", bool(np.all(chan ^ ~np.isnan(mpe))))
checks.internal("channel trials: 4 per block of 20 (216 per participant), never two in a row in the authors' order",
                bool(np.all(chan.sum(0) == 216)) and not bool(np.any(chan[1:] & chan[:-1])))

rows = []
for s in range(nP):
    tr = 0
    for k, c in enumerate(str(order[s])):
        cond = COND[int(c)]
        for t in range(360):
            i = 360 * k + t
            ph = "baseline" if t < 60 else "learning" if t < 260 else "aftereffect"
            ttype = "channel" if chan[i, s] else ("field" if ph == "learning" else "null")
            tr += 1
            rows.append(dict(
                dataset=ID, study=ID, experiment=ID, group=f"order_{order[s]}", subject=f"p{s+1}", person=f"p{s+1}",
                subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                device="vBOT planar robotic manipulandum", session=1, block=f"{cond}_block{t // 20 + 1}", trial=tr, phase=ph,
                time_s="", task="reaching", unit="trial", condition=cond,
                pert_type="force_field" if ttype == "field" else "channel" if ttype == "channel" else "none",
                pert_value=0.13 if ttype == "field" else 0, pert_unit="N s/cm" if ttype == "field" else "",
                feedback="normal", excluded_by_authors=False,
                trial_type=ttype, trial_in_condition=t + 1, block_in_condition=t // 20 + 1,
                mpe_cm=mpe[i, s] if not chan[i, s] else "", force_compensation_pct=fc[i, s] if chan[i, s] else ""))
df = pd.DataFrame(rows)

checks.published("participants", df["subject"].nunique(), 24, 0, "Methods, Participants ('Twenty-four naive subjects')")
checks.published("trials per participant", len(df) / nP, 1080, 0, "Methods ('a total of 1080 trials ... 360 per condition')")
oc = pd.Series(order).value_counts()
checks.published("participants per condition order (each of the 6 orders)", float(oc.min()), 4, 0,
                 "Methods ('four subjects each experienced one of the 6 possible orders')")
checks.internal("six distinct condition orders, each used by 4 participants", len(oc) == 6 and oc.max() == 4)
checks.internal("phase sizes 60 / 200 / 100 per condition", df.groupby(["subject", "condition"])["phase"].apply(
    lambda x: (x == "baseline").sum() * 1e6 + (x == "learning").sum() * 1e3 + (x == "aftereffect").sum()).eq(60200100).all())
ho = df[(df.condition == "hand_orientation") & (df.trial_type == "channel") & (df.block_in_condition.isin([4, 5, 6]))]
fo = df[(df.condition != "hand_orientation") & (df.trial_type == "channel") & (df.block_in_condition.isin([4, 5, 6]))]
a = float(ho.force_compensation_pct.mean())
checks.internal("early-exposure force compensation of the hand-orientation condition (blocks 4-6, a window chosen by us) lies in the paper's 20-25% range "
                "(Results, Learning); the paper gives a range, not a value", 20 <= a <= 25, f"{a:.1f}%")
checks.internal("early-exposure force compensation of the other two conditions below 10%",
                float(fo.groupby("condition").force_compensation_pct.mean().max()) < 10)
pre = df[(df.phase == "baseline") & (df.trial_type == "null")].mpe_cm.mean()
exp0 = df[(df.phase == "learning") & (df.trial_in_condition <= 64) & (df.trial_type == "field")].mpe_cm.mean()
checks.internal("MPE is small before exposure and large at the start of exposure", pre < 1.0 and exp0 > 2.0, f"{pre:.2f} vs {exp0:.2f}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Wolpert & Franklin (Cambridge)", data_url="https://doi.org/10.6084/m9.figshare.1379060",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="the two matrices (trials x participants) and the order file were turned into one table with one row per trial; "
            "phases, blocks and trial types were derived from the paper's trial structure; participant codes replaced.",
    unit_note="trial of one participant sequence",
    how=[
        "One row is one reach of one participant, in the order the participant performed them: three conditions in the "
        "participant's counterbalanced order (group says which order; condition names the way the force field direction "
        "was tied to the arm posture), each of 360 trials: 60 null-field, 200 curl-field exposure, 100 null-field.",
        "Null and field trials carry mpe_cm, the maximum perpendicular error of the hand path from the straight line "
        "(cm; the deposit does not state the unit, the paper measures distances in cm). Channel trials (4 per block of 20) "
        "carry force_compensation_pct, the percentage of the field force pushed against the channel wall.",
        "The direction of the curl field (clockwise or counter-clockwise for each posture) was counterbalanced across "
        "participants and is not in the deposit; pert_value is the force constant 0.13 N s/cm of the paper. The hand "
        "paths themselves were not shared, so there are no hand angles and the reach90 format is not used.",
    ],
)
stdlib.release(ID, "force_field_reaching", df.to_dict("records"), meta, checks)
