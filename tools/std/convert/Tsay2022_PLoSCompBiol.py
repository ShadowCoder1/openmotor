"""Tsay, Haith, Ivry & Kim (2022), PLoS Comput Biol: interactions between sensory prediction error (error-clamped cursor,
SPE) and task error (target jump, TE) in implicit motor learning. Six of the seven trial-level files of the authors' GitHub
repository, one row per reach, in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Tsay2022_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Tsay2022_PLoSCompBiol/, from https://github.com/xiaotsay2015/target_jump
(TJ_GitHubData/). The authors' Analysis/TargetJump_5Feb2022.Rmd was read as text only (never run) to learn how the columns are
used. Files converted (file -> paper experiment, by participant and trial counts and by the Rmd):
  TargetJumpClamp_data   12 people, 100 + 804 trials   Exp 1A (in person)
  TE_SPE_data            20 people, 100 + 724 trials   Exp 2, Set A (in person; the Rmd calls SN <= 20 'Set A')
  ControlJump_data       20 people, 100 + 724 trials   Exp 2, Set B (in person)
  NoFB_TJ_Control        50 people, 20 + 209 trials    online no-feedback target-jump control (Exp 1B design; the paper has 87)
  discretejump           100 people, 28 + 120 trials   Exp 3 (online)
  parajump               210 people, 28 + 252 trials   Exp 4 (online)
ConsistJump_5Feb2022.csv ('Supp1', 10 people, 900 trials) is NOT converted: the paper does not describe it and neither the
repository nor the paper says where it was run or where the target was.

Signs. 'Hand' is hand angle relative to the target; clamp_size, Rotation, tgt_jump_size and targetjump are in the same frame
(tgt_error = clamp_size - tgt_jump_size in the files; the Rmd flips the sign of Hand_Diff and of the jump for positive clamps
because the hand change opposes the clamp). Positive is counter-clockwise: the paper's Fig 5 legend says 'counterclockwise (+)
clamped feedback', and the online files have Hand = Hand.Angle - ti with Hand.Angle a standard angle from the right. The
checks below confirm that the trial-to-trial change of Hand opposes the clamp in every experiment, as the paper states
('a CW clamp eliciting a CCW change in hand angle').
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tsay2022_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}

FILES = dict(
    TJ="TargetJumpClamp_data_5Feb2022.csv", TE="TE_SPE_data_5Feb2022.csv", CJ="ControlJump_data_5Feb2022.csv",
    NF="NoFB_TJ_Control.csv", DJ="discretejump_5Feb2022.csv", PJ="parajump_5Feb2022.csv")
D = {}
for k, fn in FILES.items():
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
    D[k] = pd.read_csv(RAW / fn)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- one frame per file with a common set of fields
frames = {}
for k in ("TE", "TJ", "CJ"):  # in person, KINARM, target straight ahead at 10 cm
    d = D[k].copy()
    d["_sub"] = d["SN"].astype(int)
    d["_hand"] = d["Hand_raw"]
    d["_out"] = d["Hand"].isna()
    d["_clamp"] = d["clamp_size"].astype(float)
    d["_jump"] = d["tgt_jump_size"].astype(float)
    d["_flicker"] = False
    d["_target"] = 90.0
    d["_base"] = (d.groupby("_sub").cumcount() < 100)
    d["_fb"] = np.where(d["_base"], "veridical", "clamped")
    d["_cn"] = d.groupby("_sub").cumcount()
    d["_tn"] = d["TN"]
    frames[k] = d
# TJ (Exp 1A) baseline = trial_type 0 (100 trials); the other trial types are the two kinds of mini-block
frames["TJ"]["_block"] = frames["TJ"]["trial_type"].map({0: "baseline", 1: "trial_type1_miniblock", 2: "trial_type2_miniblock"})
for k in ("TE", "CJ"):
    frames[k]["_block"] = np.where(frames[k]["_base"], "baseline", "perturbation")

d = D["NF"].copy()  # online, mixed feedback per trial
d["_sub"] = d["SN"].astype(int)
d = d.sort_values(["_sub", "CN"], kind="stable").reset_index(drop=True)
d["_hand"] = wrap(d["Hand.Angle"] - d["ti"])
d["_out"] = d["Hand"].isna()
d["_clamp"] = np.where(d["Cursor.FB"] == "clamped_fb", d["Rotation"], np.nan).astype(float)
d["_jump"] = d["targetjump"].astype(float)
d["_flicker"] = False
d["_target"] = d["ti"].astype(float)
d["_base"] = d["CN"] <= 20
d["_fb"] = d["Cursor.FB"].map({"online_fb": "veridical", "clamped_fb": "clamped", "no_fb": "none"})
d["_cn"] = d["CN"] - 1
d["_tn"] = d["TN"]
d["_block"] = np.where(d["_base"], "baseline", "perturbation")
frames["NF"] = d

for k in ("DJ", "PJ"):  # online, trackpad, target at 45 deg and 6 cm; subject = (Exp label, SN)
    d = D[k].copy()
    d["_sub"] = d["Exp"] + "_" + d["SN"].astype(str)
    d["_hand"] = wrap(d["Hand.Angle"] - d["ti"])
    d["_out"] = d["Hand"].isna()
    d["_clamp"] = d["clamp_size"].astype(float)
    d["_flicker"] = (k == "DJ") & (d["tgt_jump_size"] == 1)
    d["_jump"] = np.where(d["_flicker"], 0.0, d["tgt_jump_size"]).astype(float)
    d["_target"] = d["ti"].astype(float)
    d["_cn"] = d.groupby("_sub").cumcount()
    d["_base"] = d["_cn"] < 28
    d["_fb"] = np.where(d["_base"], "veridical", "clamped")
    d["_tn"] = d["TN"]
    d["_block"] = np.where(d["_base"], "baseline", "perturbation")
    frames[k] = d

META = dict(  # experiment label, setting, device, baseline length, group column, id type
    TJ=("TargetJumpClamp", "lab", "kinarm", 100, None, "lab"),
    TE=("TE_SPE", "lab", "kinarm", 100, None, "lab"),
    CJ=("ControlJump", "lab", "kinarm", 100, None, "lab"),
    NF=("NoFB_TJ_Control", "online", "trackpad", 20, None, "unknown"),
    DJ=("discretejump", "online", "trackpad", 28, "Exp", "unknown"),
    PJ=("parajump", "online", "trackpad", 28, "Exp", "unknown"),
)

# ---------------------------------------------------------------- structure checks
for k, d in frames.items():
    nrows = D[k].shape[0]
    checks.internal(f"{k}: every row of the authors' file is kept", len(d) == nrows)
    base_len = META[k][3]
    pert_first = d[(d["_clamp"].fillna(0) != 0) | (d["_jump"] != 0) | d["_flicker"]].groupby("_sub")["_cn"].min()
    checks.internal(f"{k}: no clamp or jump before row {base_len} and a perturbation soon after, for everyone",
                    (pert_first >= base_len).all() and (pert_first <= base_len + 12).all(),
                    f"first perturbed row {pert_first.min()} to {pert_first.max()}")

per = {k: d.groupby("_sub").size() for k, d in frames.items()}
checks.published("Exp 1A: participants", len(per["TJ"]), 12, 0, "Methods, Experiment 1A")
checks.published("Exp 1A: trials per participant (100 baseline + 804 perturbation)", per["TJ"].iloc[0], 904, 0, "Methods, Experiment 1A")
checks.internal("Exp 1A: every participant has 904 rows", set(per["TJ"]) == {904})
checks.published("Exp 2: participants (Set A + Set B)", len(per["TE"]) + len(per["CJ"]), 40, 0, "Results, Experiment 2")
checks.published("Exp 2: trials per participant (100 baseline + 724 perturbation)", per["TE"].iloc[0], 824, 0, "Methods, Experiment 2")
checks.internal("Exp 2: every participant has 824 rows", set(per["TE"]) == {824} and set(per["CJ"]) == {824})
checks.published("Exp 3: participants", len(per["DJ"]), 100, 0, "Results, Experiment 3")
checks.published("Exp 3: trials per participant (28 baseline + 120 perturbation)", per["DJ"].iloc[0], 148, 0, "Methods, Experiment 3")
checks.internal("Exp 3: every participant has 148 rows", set(per["DJ"]) == {148})
checks.published("Exp 4: participants", len(per["PJ"]), 210, 0, "Results, Experiment 4")
checks.published("Exp 4: trials per participant (28 baseline + 252 perturbation)", per["PJ"].iloc[0], 280, 0, "Methods, Experiment 4")
checks.internal("Exp 4: every participant has 280 rows", set(per["PJ"]) == {280})
checks.published("Exp 1B design: trials per participant in the no-feedback file (20 baseline + 200 perturbation, plus 9 extra)",
                 sum(frames["NF"]["_cn"] < 20) / 50, 20, 0, "Methods, Experiment 1B (baseline block of 20 trials)")
cnt = frames["NF"][~frames["NF"]["_base"]].groupby("_sub").size()
checks.internal("Exp 1B file: 209 perturbation rows per participant (the paper counts 200)", set(cnt) == {209}, str(cnt.unique()))

# repetitions per condition (Methods, Experiment 2: 80 per clamp x jump combination, 84 for 0/0)
for k in ("TE", "CJ"):
    p = frames[k][~frames[k]["_base"]]
    n = p.groupby(["_sub", "_clamp", "_jump"]).size().unstack([1, 2])
    n00 = n[(0.0, 0.0)]
    others = n.drop(columns=[(0.0, 0.0)])
    checks.published(f"Exp 2 {'Set A' if k == 'TE' else 'Set B'}: repetitions of the 0 deg clamp / 0 deg jump trial", n00.iloc[0], 84, 0,
                     "Results, Experiment 2 ('80 repetitions per type, 84 for the 0/0 type')")
    checks.published(f"Exp 2 {'Set A' if k == 'TE' else 'Set B'}: repetitions of every other combination (min and max)",
                     float(others.min().min() + others.max().max()) / 2, 80, 0, "Results, Experiment 2")
    checks.internal(f"Exp 2 {k}: all other combinations exactly 80 per participant", (others == 80).all().all())

# the authors' outlier removal: Hand is missing where a trial deviated by more than 3 SD from a moving window
for k, rep, sd_rep, label in (("DJ", 0.8, 0.8, "Exp 3"), ("PJ", 1.1, None, "Exp 4")):
    r = 100 * frames[k].groupby("_sub")["_out"].mean()
    checks.published(f"{label}: mean percent of trials removed as outliers per participant", r.mean(), rep, 0.05,
                     "Methods, Outlier removal (Exp 3: 0.8 +- 0.8 %; Exp 4: 1.1 +- 0.1 %)")
    if sd_rep:
        checks.published(f"{label}: SD of that percent across participants", r.std(), sd_rep, 0.05, "Methods, Outlier removal")

# ---------------------------------------------------------------- sign check: the hand change opposes the clamp
for k in frames:
    d = frames[k]
    changes = {+1: [], -1: []}
    for s, g in d.groupby("_sub"):
        g = g[g["_fb"].isin(["clamped"])].copy()
        h = d[d["_sub"] == s]["_hand"].to_numpy()
        idx = d.index[d["_sub"] == s]
        pos = {i: j for j, i in enumerate(idx)}
        dn = []
        for i, c in zip(g.index, g["_clamp"]):
            j = pos[i]
            if j + 1 < len(h) and c != 0 and not d.loc[i, "_out"] and not d.loc[idx[j + 1], "_out"]:
                dn.append(-math.copysign(1, c) * (h[j + 1] - h[j]))  # clamp-opposing change is positive
        if dn:
            changes[1].append(np.median(dn))
    m = np.mean(changes[1])
    checks.internal(f"{k}: the median trial-to-trial hand change after a clamp trial opposes the clamp (mean over participants)",
                    m > 0.3 and np.mean(np.array(changes[1]) > 0) > 0.7,
                    f"mean {m:.2f} deg, {np.mean(np.array(changes[1]) > 0):.2f} of participants positive")

# ---------------------------------------------------------------- rows
rows = []
tids = {}
for k, d in frames.items():
    exp, setting, device, nb, gcol, idt = META[k]
    for s, g in d.groupby("_sub", sort=False):
        grp = g[gcol].iloc[0] if gcol else ""
        pid = f"{k}_{s}"
        for j, (i, r) in enumerate(g.iterrows()):
            base = bool(r["_base"])
            fbk = r["_fb"]
            c = r["_clamp"]
            jump = r["_jump"]
            ha = float(r["_hand"])
            ta = float(r["_target"]) % 360.0
            tid = tids.setdefault((k, round(ta, 6)), len(tids))
            row = dict.fromkeys(names, "")
            row.update(
                dataset=ID, experiment=f"{ID}_{exp}", study=ID, subject=pid, person=pid, id_type=idt,
                person_link="assumed_unique", group=grp, prior_exposure="unknown", subject_excluded_by_authors="False",
                handedness="right", population="healthy", device=device, setting=setting,
                trial=str(j), trial_orig=str(int(r["_tn"])), session="0", block=r["_block"],
                phase="baseline" if base else "learning", trial_kind="reach", break_before="True" if j == 0 else "False",
                target_angle="%.8g" % ta, target_id=str(tid), target_distance_norm="1",
                target_visible_during_move="True",
                target_event="flicker" if r["_flicker"] else ("jump" if jump != 0 else "none"),
                target_jump="%.8g" % jump, probe_kind="none",
                instr_goal="hand_to_target", instr_strategy="unspecified" if base else "forbidden",
                instr_disclosure="none" if base else "exists", instr_report="none",
                hand_angle="%.8g" % ha, hand_angle_fb="%.8g" % ha, hand_measure="fixed_distance",
                excluded_by_authors="True" if r["_out"] else "False",
                fb_endpoint_transform_only="False", fb_delay_ms="" if fbk == "none" else "0",
                fb_cued="" if base else "False",
            )
            rt = r["RT"]
            mt = r["MT"]
            row["rt_ms"] = "" if pd.isna(rt) else "%.8g" % rt
            row["mt_ms"] = "%.8g" % mt
            if fbk == "none":
                row.update(fb_visible="none", fb_mode="none", fb_form="none", online_cursor="none")
            elif fbk == "veridical":
                row.update(fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical", rotation="0",
                           cursor_error="%.8g" % ha, outcome_channels="cursor")
            else:
                row.update(fb_visible="online", fb_mode="clamped", fb_form="cursor", online_cursor="perturbed",
                           clamp="%.8g" % c, cursor_error="%.8g" % c, outcome_channels="cursor")
            rows.append(row)

checks.internal("every kept row is a reach with hand_angle in (-180, 180] and a target angle in [0, 360)",
                all(-180 < float(r["hand_angle"]) <= 180 and 0 <= float(r["target_angle"]) < 360 for r in rows))
checks.internal("rows written = rows of the six converted files", len(rows) == sum(len(D[k]) for k in FILES))

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k_: paper[k_] for k_ in ("authors", "year", "title", "venue", "doi")},
    lab="Ivry (UC Berkeley) / Kim (University of Delaware)",
    data_url="https://github.com/xiaotsay2015/target_jump",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format and participant codes replaced; hand angle taken from the authors' Hand_raw "
            "(in person) or from Hand.Angle minus the target direction (online) so that outlier trials are kept and flagged; "
            "the authors' derived columns (Hand_dt, RT_dt, Hand_Diff, RT_Diff) and Search.Time are not included.",
    unit_note="reach of one participant",
    how=[
        "One row is one reach. The file holds six experiments in the order of the authors' files: in person on a KINARM "
        "with the target 10 cm straight ahead (TargetJumpClamp = Exp 1A, TE_SPE and ControlJump = Exp 2 Sets A and B) and online "
        "on a trackpad with the target at 45 deg and 6 cm (discretejump = Exp 3, parajump = Exp 4, NoFB_TJ_Control = the "
        "no-feedback control, with targets at several directions). The authors' seventh file (ConsistJump, 10 people) is not "
        "included because the paper does not describe it.",
        "Participants were told to ignore the cursor and any change of the target and to reach to the original target. After "
        "the baseline (veridical cursor) each trial had an error-clamped cursor (clamp, deg) and/or a target jump (target_jump, "
        "deg, at movement onset; a jump in place is target_event flicker). Clamp, jump and hand angle share one sign: "
        "counter-clockwise is positive (see the signs note), so the hand change after a clamp is opposite in sign to it.",
        "hand_angle is the hand direction relative to the target when the movement reached the target distance. The authors' "
        "outlier rule (more than 3 SD from a moving 5-trial window) is marked in excluded_by_authors; their own Hand column is "
        "missing on those trials, here the value is kept. In the no-feedback control (NoFB_TJ_Control) the cursor was absent on "
        "target-jump trials (fb_mode none) and a few trials carry large clamps or jumps that the paper does not describe.",
        "In the in-person files a target jump with a 0 deg clamp is a clamped cursor that always passes through the original "
        "target (cursor_error 0). Group holds the authors' Exp label for discretejump and parajump (3d/7d and the jump range); "
        "what 3d and 7d stand for is not documented. trial_orig is the authors' TN, which restarts or skips in some files.",
    ],
    differences=[
        "The paper reports 87 participants for the online Experiment 1B; the deposit's no-feedback file has 50 and also holds "
        "rotated-clamp and large-jump trials, so it is converted as the no-feedback control and its identification with "
        "Experiment 1B is not certain.",
        "Outlier removal: the paper reports 0.2 +- 0.2 % of trials per participant for Exp 1 and 0.1 +- 0.2 % for Exp 2; the "
        "deposit's missing Hand values give about 0.14 % (Exp 1A) and 0.23 % (Exp 2), and 1.1 % +- 0.8 (not +- 0.1) for Exp 4, "
        "so only the Exp 3 and Exp 4 means are listed as reproduced.",
        "Positive = counter-clockwise is inferred from the paper's Fig 5 legend and from the direction of the hand change; "
        "the deposit has no readme.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
