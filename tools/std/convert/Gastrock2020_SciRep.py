"""Gastrock, Modchalingam, 't Hart & Henriques (2020), Sci Rep: external error attribution and hand localization.
90 right-handed students in four groups (Control = '30implicit', Instructed = '30explicit', Cursor Jump, Hand View) trained with a
30 deg clockwise cursor rotation. Raw files (untrusted data, read only) in raw_extra/Gastrock2020_SciRep/ from https://osf.io/xdgh6/:
*_learningcurves_long.csv (reach deviation, 90 rotated cursor-training trials), *_localization.csv, *_nocursor.csv,
participant_demographics.csv, ppawareness_scores.csv. Aligned-session and top-up training trials are not in the deposit.

Signs: the paper (Methods, Procedure) says the cursor was rotated 30 deg CW so correcting needs reaches 30 deg CCW; the learning
curves rise to about +30, so reachdev is CCW positive, as reach90 requires. Localization angles are polar angles from the home
position (0 deg = right, CCW positive; targets/arcs at 45/90/135 and 50/90/130 deg), the screen frame.
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gastrock2020_SciRep"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
GROUPS = [("30implicit", "Control"), ("30explicit", "Instructed"), ("cursorjump", "Cursor Jump"), ("handview", "Hand View")]
rd = lambda f: pd.read_csv(RAW / f)  # noqa: E731


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


demo = rd("participant_demographics.csv").set_index("participant")
aware = rd("ppawareness_scores.csv").set_index("participant")
LC = {g: rd(f"{g}_learningcurves_long.csv") for g, _ in GROUPS}
LO = {g: rd(f"{g}_localization.csv") for g, _ in GROUPS}
NC = {g: rd(f"{g}_nocursor.csv") for g, _ in GROUPS}
rows, loc_extra = [], []


def base(pid, gname, seq, experiment):
    r = dict.fromkeys(names, "")
    r.update(dataset=ID, experiment=experiment, study=ID, subject=f"{pid}_{seq}", person=pid, id_type="lab",
             person_link="assumed_unique", group=gname, prior_exposure="unknown", subject_excluded_by_authors="False",
             age=num(demo.age[pid]), sex=demo.sex[pid], handedness="right", device="robot", setting="lab", population="healthy",
             trial_kind="reach", target_distance_norm="1", hand_used="right", excluded_by_authors="False")
    return r


for g, gname in GROUPS:
    d = LC[g]
    for pid in d.participant.drop_duplicates():
        x = d[d.participant == pid].sort_values("trial", kind="stable")
        assert list(x.trial) == list(range(1, len(x) + 1))
        for j, (_, t) in enumerate(x.iterrows()):
            r = base(pid, gname, "learn", f"{ID}_RotatedTraining")
            miss = t.reachdev != t.reachdev
            r.update(session="1", trial=str(j), trial_orig=str(int(t.trial)), block="rotated_cursor_training", phase="learning",
                     break_before="True" if j == 0 else "False", target_visible_during_move="True", target_event="none",
                     target_jump="0", probe_kind="none", instr_goal="cursor_to_target",
                     instr_strategy="allowed", instr_disclosure="exact" if g == "30explicit" else "exists", instr_report="none",
                     hand_angle=num(t.reachdev), hand_measure="peak_velocity", fb_visible="online", fb_mode="rotated", fb_form="cursor",
                     online_cursor="perturbed" if g == "cursorjump" else "perturbed", fb_endpoint_transform_only="False",
                     rotation="-30", excluded_by_authors="True" if miss else "False")
            rows.append(r)
n_learn = len(rows)

for g, gname in GROUPS:
    d = LO[g]
    for pid in d.participant.drop_duplicates():
        x = d[d.participant == pid].reset_index(drop=True)
        checks.internal(f"{pid}: aligned localization rows come before rotated ones", bool((np.diff(x.rotated.values) >= 0).all()))
        for j, t in x.iterrows():
            passive, rot = int(t.passive) == 1, int(t.rotated) == 1
            r = base(pid, gname, "loc", f"{ID}_Localization")
            r.update(session="1" if rot else "0", trial=str(j), trial_orig=str(int(t.trial)),
                     block=f"{'rotated' if rot else 'aligned'}_{'passive' if passive else 'active'}_{int(t.block)}",
                     phase="probe" if rot else "baseline", trial_kind="proprio_probe" if passive else "reach",
                     break_before="True" if j == 0 else "False", probe_kind="passive_robot" if passive else "active_post_reach",
                     probe_location_deg=num(t.hand_angle), instr_report="hand_passive" if passive else "hand_post",
                     hand_report=num(t.tap_angle - t.hand_angle), target_event="none", hand_extent="1", fb_visible="none",
                     fb_mode="none", fb_form="none", online_cursor="none")
            rows.append(r)
            loc_extra.append([r["subject"], j, int(t.trial), int(t.rotated), int(t.passive), int(t.block), int(t.arc), t.hand_angle, t.tap_angle])

# ---------------------------------------------------------------- published checks (expected values read from the paper)
S = "Methods, Participants"
pers = demo.loc[sorted({r["person"] for r in rows})]
checks.published("participants", len(pers), 90, 0, S)
for g, gname in GROUPS:
    pass
n = pers.groupby("group").size()
fem = pers[pers.sex == "F"].groupby("group").size()
for g, gname, nn, ff in (("30implicit", "Control", 20, 14), ("30explicit", "Instructed", 21, 13), ("cursorjump", "Cursor Jump", 20, 14),
                         ("handview", "Hand View", 29, 23)):
    checks.published(f"{gname} group, participants", n[g], nn, 0, S)
    checks.published(f"{gname} group, women", fem[g], ff, 0, S)
checks.published("women in all groups", (pers.sex == "F").sum(), 64, 0, S)
checks.published("mean age (years)", pers.age.mean(), 20.8, 0.1, S)
checks.published("SD of age (years)", pers.age.std(), 3.88, 0.01, S)
mis = sum(1 for r in rows[:n_learn] if r["excluded_by_authors"] == "True") / n_learn * 100
checks.published("percent of cursor-training trials removed as outliers (paper: 0.94% over aligned and rotated sessions; here the rotated block only)",
                 mis, 0.94, 0.6, "Methods, Data analysis, learning during adaptation training")
checks.internal("90 rotated training trials per participant", n_learn == 90 * 90, str(n_learn))
checks.internal("every localization row kept", len(rows) - n_learn == sum(len(v) for v in LO.values()))
lat = pd.Series([float(r["hand_angle"]) for r in rows[:n_learn] if r["hand_angle"]])
checks.internal("late rotated training deviations rise toward +30 (counter-clockwise compensation): mean of last 15 trials in 15 to 40 deg",
                15 < np.mean([float(r["hand_angle"]) for r in rows[:n_learn] if r["hand_angle"] and int(r["trial"]) >= 75]) < 40)

# ---------------------------------------------------------------- extra tables
code = {}
for r in rows:
    code.setdefault(r["subject"], f"{ID}_S{len(code) + 1:04d}")
b = io.StringIO()
w = csv.writer(b, lineterminator="\r\n")
w.writerow(["subject", "trial", "trial_in_session", "rotated_session", "passive", "block", "arc_center_deg", "hand_angle_deg", "tap_angle_deg"])
for r in loc_extra:
    w.writerow([code[r[0]]] + [num(v) for v in r[1:]])
loc_txt = b.getvalue()

ppl = []
for pid in dict.fromkeys(r["person"] for r in rows):
    g = demo.group[pid]
    p = dict(person=pid, group=dict(GROUPS)[g], awareness_points=aware.points.get(pid), awareness_version=aware.version.get(pid))
    nc = NC[g]
    for _, t in nc[nc.participant == pid].iterrows():
        for k in ("aligned", "exclusive", "inclusive"):
            p[f"nocursor_{k}_{int(t.target_angle)}"] = t[k]
    ppl.append(p)
pcols = [("group", "text", "", "Group: Control, Instructed, Cursor Jump or Hand View."),
         ("awareness_points", "number", "points", "The authors' awareness score from ppawareness_scores.csv (0 to 3 in the file); the paper's text does not define it further."),
         ("awareness_version", "text", "", "The authors' version code (ie or ei) of the awareness task; probably the order of the with- and without-strategy no-cursor variations, which the paper says was counterbalanced (inferred).")]
for k, lab in (("aligned", "aligned session"), ("exclusive", "rotated session, without strategy"), ("inclusive", "rotated session, with strategy")):
    for a in (45, 90, 135):
        pcols.append((f"nocursor_{k}_{a}", "number", "degrees",
                      f"No-cursor reach deviation to the {a} deg target, {lab}, as the authors share it (one value per participant and target; counter-clockwise positive; the file does not say if it is a mean of several reaches or baseline-corrected)."))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Henriques & 't Hart (York University)", data_url="https://osf.io/xdgh6/", license="none stated", license_url="",
    changes="converted to the OpenMotor format; participant codes replaced; the authors' reach deviation, localization angles and "
            "no-cursor values kept in degrees; the hand-minus-tap difference computed as hand_report.",
    unit_note="rotated-session cursor training reach, or hand localization trial",
    how=[
        "One row is one trial of a 30 degree visuomotor rotation experiment on a robot manipulandum (right hand, 12 cm out-and-back reaches). "
        "Rotated training rows (session 1, experiment ..._RotatedTraining) are the first 90 cursor-training trials of the rotated session; "
        "hand_angle is the authors' reach deviation (angle of the hand at peak velocity from the target, from the home position), counter-clockwise "
        "positive, and the cursor rotation is -30 (clockwise). The three targets (45, 90 and 135 deg) were shuffled and the deposit does not say which "
        "trial had which target, so target_angle is empty. Rows with no deviation (0.9 percent: trials the authors removed as outliers) are kept "
        "and flagged excluded_by_authors.",
        "Localization rows (experiment ..._Localization; session 0 aligned, session 1 rotated) are active (probe_kind active_post_reach, the person "
        "chose where to cross a 12 cm arc) and passive (passive_robot, the robot moved the hand to a place chosen earlier) localization trials. "
        "The person then tapped where the unseen hand crossed the arc with the left hand. probe_location_deg is the polar angle of the hand and "
        "hand_report is the tap angle minus the hand angle, counter-clockwise positive. localization_details.csv has the arc, hand angle and tap angle.",
        "Groups: Control (file name 30implicit), Instructed (30explicit: told about the rotation and given a strategy at the break), Cursor Jump "
        "(cursor jumped to the rotated path after a third of the reach) and Hand View (the illuminated hand was visible along with the cursor). "
        "In Cursor Jump the cursor is veridical for the first third of the reach, then rotated: online_cursor is coded perturbed. "
        "Instruction coding is read from the paper's Procedure: Instructed instr_disclosure exact; the others exists (told the cursor would move differently); "
        "the strategy angle given to the Instructed group is not in the deposit or main text. Aligned-session and top-up training trials are not shared.",
        "No-cursor reaches are shared only as one value per participant and target (aligned, without strategy, with strategy) and are in participants.csv.",
    ],
    differences=["The paper's learning-curve statistics use baseline-corrected deviations over aligned and rotated sessions, which the deposit "
                 "does not hold, so only the participant, group, sex, age and outlier-rate numbers are checked. The outlier rate of the paper (0.94 percent) covers both sessions; "
                 "the rotated block alone is compared here with a wide tolerance."],
)
stdlib.release(ID, "reach90", rows, meta, checks, participants=ppl, participant_columns=pcols,
               extra_files=[("localization_details.csv", loc_txt)])
