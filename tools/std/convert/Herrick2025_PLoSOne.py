"""Herrick & Yakovenko (2025) PLoS ONE 20(4):e0321949: treadmill walking with a stride-length constraint on one leg
(23 adults, 2 groups), one row per stride from the authors' nTable_AI.csv (double-support durations), with the
cumulative ground-reaction-force sums of nTable_CS.csv as participant columns.

Raw files: raw_extra/Herrick2025_PLoSOne/ nTable_AI.csv, nTable_CS.csv (Figshare 10.6084/m9.figshare.25343671, CC BY 4.0);
the authors' MATLAB scripts (main_v9_figshare.m etc.) were read as text only. nTable_ex_GRF.csv is a group-average GRF
curve for a figure (no per-participant values) and is not converted.
Run from the site folder: python3 -I tools/std/convert/Herrick2025_PLoSOne.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Herrick2025_PLoSOne"
raw = stdlib.raw_dir(DS)
GROUP = {1: "kinematic constraint only", 2: "kinematic constraint plus asymmetric loading"}
BLOCK = {"Control": ("baseline", "none"), "Adapt": ("learning", "constraint"), "Post": ("washout", "none")}
STRAP_CM = {1: 2.0, 2: 10.5}  # anterior protrusion of the strap/device, paper Methods
DEVICE = ("Bertec instrumented treadmill (force plates, 1 kHz), both belts at 1.0 m/s; a device on the right leg limited "
          "the stride length of the left leg during the adaptation block")

ai = pd.read_csv(raw / "nTable_AI.csv")
cs = pd.read_csv(raw / "nTable_CS.csv")
ck = stdlib.Checks()

ai["subject"] = "G" + ai.idGroup.astype(str) + "_" + ai.sSession
order = ai.subject.drop_duplicates().tolist()
df = pd.DataFrame(dict(
    dataset=DS, study=DS, experiment="", group=ai.idGroup.map(GROUP), subject=ai.subject,
    subject_excluded_by_authors=False, population="healthy", setting="lab", device=DEVICE, session=1,
    block=ai.sCondition, trial=ai.groupby("subject").cumcount() + 1, phase=ai.sCondition.map(lambda c: BLOCK[c][0]),
    task="walking", unit="stride", condition=ai.sCondition.map(lambda c: {"Control": "control", "Adapt": "left stride-length constraint", "Post": "washout"}[c]),
    pert_type=ai.sCondition.map(lambda c: "stride_length_constraint" if c == "Adapt" else "none"),
    pert_value=[STRAP_CM[g] if c == "Adapt" else 0.0 for g, c in zip(ai.idGroup, ai.sCondition)],
    pert_unit=["cm strap protrusion" if c == "Adapt" else "" for c in ai.sCondition],
    feedback="normal", excluded_by_authors=False, fast_leg="unknown", belt_speed_fast_mps=1.0, belt_speed_slow_mps=1.0,
    walking_context="treadmill", surface_incline_deg=0.0,
    double_support_left_leading_s=ai.nDS_Left_Leading_s, double_support_left_trailing_s=ai.nDS_Left_Trailing_s))

df = df.reindex(columns=[c[1] for c in stdlib.columns_for('walking') if c[1] != 'person'])

# ---- participants.csv: nTable_CS (cumulative sum of the vertical GRF during stance, per condition and leg)
cs["subject"] = "G" + cs.idGroup.astype(str) + "_" + cs.sSession
piv = cs.pivot_table(index="subject", columns=["sCondition", "sLeg"], values="nCumulSum")
G2A = ["S22", "S23", "S24", "S25", "S26", "S27", "S28"]  # main_v9_figshare.m: idSessionList.G2a, "sessions that showed adaptation"
part, pcols = [], [("showed_adaptation", "true/false", "",
                    "Group 2 only: the authors' list of participants who showed adaptation (main_v9_figshare.m, idSessionList.G2a); "
                    "empty for group 1.")]
names = {("Control", "L"): "control_left", ("Control", "R"): "control_right", ("AsymL", "L"): "constraint_left",
         ("AsymL", "R"): "constraint_right", ("Washout", "L"): "washout_left", ("Washout", "R"): "washout_right"}
for k, n in names.items():
    pcols.append((f"cumulative_grf_{n}", "number", "body weights x steps",
                  f"Cumulative sum of the vertical ground reaction force of the {'left' if k[1] == 'L' else 'right'} leg during stance "
                  f"in the {k[0].lower()} condition, normalized to the participant's weight and to the number of steps in the "
                  "condition (nCumulSum in the deposit)."))
for s in order:
    r = dict(person=s, showed_adaptation=(s.split("_")[1] in G2A) if s.startswith("G2") else "")
    for k, n in names.items():
        r[f"cumulative_grf_{n}"] = piv.loc[s, k]
    part.append(r)

# ---- checks
g = ai.drop_duplicates("subject").idGroup.value_counts().to_dict()
ck.internal("rows kept: 43,977 strides, none dropped; 15 left-leading and 8 left-trailing values are empty in the deposit",
            len(df) == 43977 and ai.nDS_Left_Leading_s.isna().sum() == 15 and ai.nDS_Left_Trailing_s.isna().sum() == 8)
ck.internal("each participant has contiguous Control, Adapt, Post blocks in that order",
            all(list(dict.fromkeys(x.sCondition)) == ["Control", "Adapt", "Post"] and (x.sCondition != x.sCondition.shift()).sum() == 3
                for _, x in ai.groupby("subject")))
ck.internal("CS table has the same 23 participants and 6 values each", cs.subject.nunique() == 23 and len(cs) == 138 and set(cs.subject) == set(order))
ck.published("participants, group 1", g[1], 11, 0, "Methods (Participants)")
ck.published("participants, group 2", g[2], 12, 0, "Methods (Participants)")
ck.published("participants in total", len(order), 23, 0, "Methods (Participants)")
ck.published("group 2 participants who showed adaptation", sum(p["showed_adaptation"] is True for p in part), 7, 0, "Results (58% of group 2)")

ai["a"] = (ai.nDS_Left_Leading_s - ai.nDS_Left_Trailing_s) / (ai.nDS_Left_Leading_s + ai.nDS_Left_Trailing_s)


def absai(sub, cond, sl=slice(None)):
    return sub[sub.sCondition == cond].groupby("subject").a.apply(lambda x: np.nanmean(np.abs(x.iloc[sl])))


for gid, vals in {1: dict(ctrl=(0.022, "Control"), ew=(0.026, 0), lw=(0.021, 0)), 2: dict(ew=(0.044, 0), lw=(0.024, 0))}.items():
    sub = ai[ai.idGroup == gid]
    e, l = absai(sub, "Post", slice(0, 20)), absai(sub, "Post", slice(-20, None))
    ck.published(f"group {gid} early washout |AI| (first 20 strides), mean", e.mean(), vals["ew"][0], 0.001, "Results, aftereffects (AI, mean over participants)")
    ck.published(f"group {gid} late washout |AI| (last 20 strides), mean", l.mean(), vals["lw"][0], 0.001, "Results, aftereffects (AI, mean over participants)")
    if gid == 1:
        ck.published("group 1 early washout |AI|, SD", e.std(), 0.022, 0.001, "Results, aftereffects")
        ck.published("group 1 control |AI|, mean", absai(sub, "Control").mean(), 0.022, 0.001, "Results, aftereffects")
        ck.published("group 1 late washout |AI|, SD", l.std(), 0.015, 0.001, "Results, aftereffects")
    else:
        ck.published("group 2 early washout |AI|, SD", e.std(), 0.032, 0.001, "Results, aftereffects")
sub2 = ai[ai.subject.str.split("_").str[1].isin(G2A) & (ai.idGroup == 2)]
ck.published("group 2 adapting subgroup control |AI|, mean", absai(sub2, "Control").mean(), 0.023, 0.001, "Results, aftereffects")
ck.published("group 2 adapting subgroup washout |AI|, mean", absai(sub2, "Post").mean(), 0.022, 0.001, "Results, aftereffects")
for gid, leg, m, sd in [(1, "L", 458, 64), (1, "R", 453, 54), (2, "L", 420, 35), (2, "R", 617, 68)]:
    v = cs[(cs.idGroup == gid) & (cs.sCondition == "AsymL") & (cs.sLeg == leg)].nCumulSum
    ck.published(f"group {gid} {leg} leg cumulative GRF in the constraint condition, mean", v.mean(), m, 0.5, "Results, loading analysis")
    ck.published(f"group {gid} {leg} leg cumulative GRF in the constraint condition, SD", v.std(), sd, 0.6, "Results, loading analysis")

meta = dict(
    paper=dict(authors="Herrick EM, Yakovenko S", year=2025,
               title="Evidence of sensory error threshold in triggering locomotor adaptations in humans",
               venue="PLoS ONE 20(4):e0321949", doi="10.1371/journal.pone.0321949"),
    lab="Yakovenko (West Virginia University)", data_url="https://doi.org/10.6084/m9.figshare.25343671",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept every stride row of nTable_AI.csv in its order, with participants coded and the group and block named in "
             "the authors' words; the six cumulative-sum values per participant of nTable_CS.csv are in participants.csv. "
             "The group-average GRF curves of nTable_ex_GRF.csv (figure data, no participant values) are not included."),
    how=["One row is one stride of one participant: the double-support duration (seconds) while the left leg leads and while it "
         "trails, as the authors derived them from the treadmill force plates and smoothed with a 20-stride moving average. "
         "The authors' asymmetry index is (leading - trailing) / (leading + trailing), which can be computed from the two columns.",
         "Participants walked on a tied-belt treadmill at 1.0 m/s through a control block (5 min), an adaptation block with "
         "a device on the right leg that limited the stride length of the left leg (10 min) and a washout block (20 min), one "
         "after the other. Group 1 had the kinematic constraint only; group 2 also had to unload the constrained leg, with "
         "verbal feedback. pert_value is the anterior protrusion of the device strap (2 and 10.5 cm, paper Methods).",
         "The deposit shares no age, sex or handedness per participant and no stride times, so those columns are empty; "
         "participants S11-S33 are session numbers of the deposit, recoded here."],
    unit_note="stride",
    differences=["Group 1 mean |AI| in the adaptation block is 0.167 here against 0.171 in the paper, and the group 2 adapting "
                 "subgroup 0.107 against 0.103: the authors trimmed every participant to a common number of strides before "
                 "averaging, which is not reproduced here (not used as a check). The paper's AI equation is an image; the "
                 "formula from the deposit's README was used."])
stdlib.release(DS, "walking", df, meta, ck, participants=part, participant_columns=pcols)
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
