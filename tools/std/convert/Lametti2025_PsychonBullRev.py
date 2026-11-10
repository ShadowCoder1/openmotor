"""Lametti, Vaillancourt, Whitman & Skipper (2025), Psychon Bull Rev: joystick reaching with a 30 deg visuomotor rotation in which the
reach target is a spoken word (Experiment 1: 'bad' versus a green circle; Experiment 2: 'tap' versus 'hit') or a circle; 60 participants
(30 per experiment, two groups of 15 each), one row per reach, 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Lametti2025_PsychonBullRev.py
Raw files (untrusted data, read only) in raw_extra/Lametti2025_PsychonBullRev/, from https://osf.io/t7jdq/: P01_EX1.mat ... P30_EX2.mat
(one MATLAB file per participant and experiment). The authors' Lametti_et_al_2023a_figures_stats.m was read as text (never run) for the
variables: errpeakvel (angular error at peak hand velocity, degrees), afind_T1 / afind_T2 (their after-effect trials), condition
(Experiment 1: 1 = word, 2 = not word; Experiment 2: 1 = tap, 2 = hit), trialtype, rotation, targetdir, target, feedback, xpos / ypos.

Sign convention (inferred, see the README). The authors' errpeakvel has the opposite sign to the OpenMotor convention: with their
rotation of -30 the adaptation error is negative at first and the after-effect is positive. The paper's Methods say the rotation was
counter-clockwise, so cursor = hand + 30 deg here (rotation +30), and
    hand_angle = errpeakvel's rotation column (-30 or 0) - errpeakvel         (counter-clockwise positive, relative to the target)
    cursor_error = -errpeakvel                                                  (the authors' error is the cursor error, sign reversed)
The raw coordinates confirm this: on trials without a cursor or without rotation the direction of (xpos, ypos) at peak speed equals
hand_angle, and on cursor trials during adaptation it equals the cursor direction, hand_angle + 30 (checked below).
"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Lametti2025_PsychonBullRev"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]


def num(x):
    if x is None or x != x:
        return ""
    return "%.8g" % (0.0 if x == 0 else float(x))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def peak_speed_angle(x, y, t):
    """Direction of (x, y) at the sample after peak speed, using the authors' rule (speed from 10 x position, first 25 samples zeroed,
    peak in the first 150 samples); counter-clockwise degrees from the x axis."""
    x, y, t = (np.asarray(a, float) for a in (x, y, t))
    v = np.diff(np.hypot(10 * x, 10 * y)) / np.diff(t)
    v[:25] = 0
    k = int(np.argmax(v[:150]))
    return float(np.degrees(np.arctan2(y[k + 1], x[k + 1])))


files = sorted(RAW.glob("P??_EX?.mat"), key=lambda p: (p.stem[-1], p.stem[:3]))
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("60 files, all with the downloaded hash",
                len(files) == 60 and all(stdlib.sha256(p) == src[p.name]["sha256"] for p in files))

GROUP = {("EX1", 1): "adapt_with_speech", ("EX1", 2): "adapt_without_speech", ("EX2", 1): "adapt_with_tap", ("EX2", 2): "adapt_with_hit"}
LABEL_CUE = {"BAD": "word:BAD", "DOT": "circle", "TAP": "word:TAP", "HIT": "word:HIT"}
WARN = {"Just right!": 0, "Move sooner!": 1, "Move later!": 1, "Don't speak!": 1, "No response": 1}

rows = []
fdata = []  # per file: dict for checks
dev_free = []   # |direction from coordinates - hand_angle| on trials where the logged position is the hand
dev_cursor = []  # |direction from coordinates - (hand_angle + 30)| on cursor trials during adaptation
ok_rot_vec = True
for fi, p in enumerate(files):
    m = sio.loadmat(p, squeeze_me=True)
    exp = p.stem[-3:]
    cond = int(m["condition"])
    err = np.asarray(m["errpeakvel"], float)
    n = len(err)
    tt = np.asarray(m["trialtype"])
    td = np.asarray(m["targetdir"], float)
    tg = [str(v) for v in np.asarray(m["target"])]
    fbk = [str(v) for v in np.asarray(m["feedback"])]
    a_idx = sorted(set(np.asarray(m["afind_T1"]).tolist()) | set(np.asarray(m["afind_T2"]).tolist()))
    a_idx = [int(i) for i in a_idx]
    first_ae, last_ae = min(a_idx), max(a_idx)
    rot_vec = np.asarray(m["rotation"], float)
    # phase of each 1-based row: baseline = first 80 rows, adaptation up to the row before the first after-effect row,
    # after-effect rows = the authors' afind rows (a trial of the after-effect block in between that the authors left out stays
    # in the after-effect block), the rest = unlearning
    blocks, rot_a = [], []
    for r1 in range(1, n + 1):
        if r1 <= 80:
            b = "baseline"
        elif r1 < first_ae:
            b = "adaptation"
        elif r1 <= last_ae:
            b = "after-effect"
        else:
            b = "unlearning"
        blocks.append(b)
        rot_a.append(-30.0 if b == "adaptation" else 0.0)
    if n >= 480 or p.stem != "P18_EX1":
        ok_rot_vec &= bool(np.all(np.asarray(rot_a) == rot_vec[:n]))
    nocursor = tt != 0
    xs, ys, ts = m["xpos"], m["ypos"], m["time"]
    for i in range(n):
        if np.isnan(err[i]) or not len(xs[i]):
            continue
        rel = wrap(peak_speed_angle(xs[i], ys[i], ts[i]) - td[i])
        hand = rot_a[i] - err[i]
        if blocks[i] == "adaptation" and not nocursor[i]:
            dev_cursor.append(abs(wrap(rel - (hand + 30.0))))
        else:
            dev_free.append(abs(wrap(rel - hand)))
    sub = f"{ID}_{p.stem}"
    sex = {"female": "F", "male": "M"}[str(m["sex"])]
    for i in range(n):
        r1 = i + 1
        b = blocks[i]
        vis = not nocursor[i]
        ra = rot_a[i]
        e = err[i]
        hand = ra - e if e == e else float("nan")
        rot_screen = -ra  # +30 counter-clockwise on adaptation rows
        label = tg[i]
        row = dict.fromkeys(names, "")
        row.update(
            dataset=ID, experiment=f"{ID}_{exp}", study=ID, subject=sub, person=sub, id_type="lab", person_link="assumed_unique",
            group=GROUP[(exp, cond)], prior_exposure="unknown", subject_excluded_by_authors="False",
            age=num(float(m["age"])), sex=sex, handedness="right", device="other", setting="lab", population="healthy",
            trial=str(i), trial_orig=str(r1), session="0", block=b,
            phase={"baseline": "baseline", "adaptation": "learning", "after-effect": "aftereffect", "unlearning": "washout"}[b],
            trial_kind="reach", break_before="True" if i == 0 else "False",
            target_angle=num(td[i]), target_id=str(int(td[i] // 60)), target_distance_norm="1", target_event="none", target_jump="0",
            probe_kind="none", instr_goal="cursor_to_target", cue_elements=LABEL_CUE[label], hand_used="right",
            hand_angle=num(hand), hand_angle_fb=num(hand), hand_measure="peak_velocity",
            excluded_by_authors="True" if (e != e or abs(e) > 40) else "False",
            fb_visible="online" if vis else "none",
            fb_mode=("rotated" if ra else "veridical") if vis else "none",
            fb_form="cursor" if vis else "none",
            online_cursor=("perturbed" if ra else "veridical") if vis else "none",
            fb_endpoint_transform_only="False",
            rotation=num(rot_screen) if vis else "",
            cursor_error=num(-e) if (vis and e == e) else "",
            outcome_channels="cursor" if vis else "",
            speed_warning="0", other_warning=str(WARN[fbk[i]]),
        )
        rows.append(row)
    fdata.append(dict(file=p.stem, exp=exp, cond=cond, age=float(m["age"]), sex=sex, n=n))

checks.internal("rows kept: every trial the authors shared (57 files with 480 trials, P02_EX1 and P07_EX1 with 479, P18_EX1 with 460)",
                len(rows) == sum(d["n"] for d in fdata) and Counter(d["n"] for d in fdata) == Counter({480: 57, 479: 2, 460: 1}),
                f"{len(rows)} rows")
checks.internal("the authors' rotation column (-30 on 320 trials) equals the rule used here (rows 81 to the first after-effect row - 1) "
                "in every file except the short P18_EX1", ok_rot_vec)
checks.internal("on trials where the logged position is the hand (no cursor, or no rotation) its direction at peak speed equals hand_angle "
                "(median difference below 2 deg, 95th percentile below 12 deg)",
                np.median(dev_free) < 2 and np.percentile(dev_free, 95) < 12,
                f"n={len(dev_free)}, median {np.median(dev_free):.2f}, 95th percentile {np.percentile(dev_free, 95):.1f} deg")
checks.internal("on cursor trials during adaptation the logged position is the cursor: its direction equals hand_angle + 30 "
                "(median difference below 2 deg, 95th percentile below 12 deg)",
                np.median(dev_cursor) < 2 and np.percentile(dev_cursor, 95) < 12,
                f"n={len(dev_cursor)}, median {np.median(dev_cursor):.2f}, 95th percentile {np.percentile(dev_cursor, 95):.1f} deg")

# ------------------------------------------------------------------ published numbers, from the converted rows
by_sub = {}
for r in rows:
    by_sub.setdefault(r["subject"], []).append(r)


def auth_err(r):
    """The authors' errpeakvel, rebuilt from the converted row: minus the cursor direction relative to the target
    (hand_angle plus the screen rotation on adaptation rows)."""
    if r["hand_angle"] == "":
        return float("nan")
    return -(float(r["hand_angle"]) + (30.0 if r["block"] == "adaptation" else 0.0))


def nanmean(v):
    v = [x for x in v if x == x]
    return sum(v) / len(v)


def stats(sub_rows):
    e = [auth_err(r) for r in sub_rows]
    base = nanmean(e[:80])
    out = dict(start=nanmean(e[80:100]) - base, end=nanmean(e[380:400]) - base)
    for lab, cue in (("BAD", "word:BAD"), ("DOT", "circle"), ("TAP", "word:TAP"), ("HIT", "word:HIT")):
        sel = [auth_err(r) - base for r in sub_rows if r["block"] == "after-effect" and r["cue_elements"] == cue]
        sel = [x for x in sel if x == x]
        out["ae_" + lab] = sum(sel) / len(sel) if sel else float("nan")
    return out


S = {k: stats(v) for k, v in by_sub.items()}
grp_of = {k: v[0]["group"] for k, v in by_sub.items()}
exp_of = {k: v[0]["experiment"].split("_")[-1] for k, v in by_sub.items()}


def gmean(group, key):
    return st.mean(S[k][key] for k in S if grp_of[k] == group and S[k][key] == S[k][key])


for g in ("adapt_with_speech", "adapt_without_speech", "adapt_with_tap", "adapt_with_hit"):
    checks.published(f"participants in group {g}", sum(1 for k in grp_of if grp_of[k] == g), 15, 0,
                     "Methods, Participants and Experimental Design (15 per group)")
checks.published("participants, both experiments", len(by_sub), 60, 0, "Methods, Participants (60 adults, 30 per experiment)")
checks.published("women among the 60 participants", sum(d["sex"] == "F" for d in fdata), 42, 0, "Methods, Participants (42 female, 18 male)")
checks.published("youngest participant (years)", min(d["age"] for d in fdata), 18, 0, "Methods, Participants (age 18 to 26)")
checks.published("oldest participant (years)", max(d["age"] for d in fdata), 26, 0, "Methods, Participants (age 18 to 26)")
checks.published("median age (years)", st.median(d["age"] for d in fdata), 20, 0, "Methods, Participants (median 20)")
# Results, Experiment 1 and 2: mean angular error (authors' sign) in the first and last 20 adaptation trials (relative to baseline)
# and the after-effect, per group. All printed to 2 decimals.
REP = {
    "adapt_with_speech": dict(start=-19.40, end=-4.41, ae_BAD=19.27, ae_DOT=17.40),
    "adapt_without_speech": dict(start=-20.74, end=-3.57, ae_DOT=19.41, ae_BAD=15.02),
    "adapt_with_tap": dict(start=-21.64, end=-3.38, ae_TAP=21.86, ae_HIT=20.40),
    "adapt_with_hit": dict(start=-19.83, end=-4.03, ae_HIT=22.00, ae_TAP=20.98),
}
WHERE = {"adapt_with_speech": "Results, Experiment 1, group 1", "adapt_without_speech": "Results, Experiment 1, group 2",
         "adapt_with_tap": "Results, Experiment 2, tap-adapted group", "adapt_with_hit": "Results, Experiment 2, hit-adapted group"}
differences = []
for g, rep in REP.items():
    for key, val in rep.items():
        comp = gmean(g, key)
        what = f"{g}: mean {'error in the first 20 adaptation trials' if key == 'start' else 'error in the last 20 adaptation trials' if key == 'end' else 'after-effect, ' + key[3:] + ' trials'} (deg, authors' sign)"
        ok = abs(comp - val) <= 0.005
        if ok:
            checks.published(what, comp, val, 0.005, WHERE[g])
        else:
            differences.append(f"{what}: the paper prints {val}, the shared rows give {comp:.3f}.")
            print("NOT REPRODUCED:", what, comp, val)

# ------------------------------------------------------------------ release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Lametti (Acadia University)",
    data_url="https://osf.io/t7jdq/",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, the sign of the angles reversed to counter-clockwise "
            "positive (see below), and the hand angle rebuilt as the authors' rotation column minus their angular error.",
    unit_note="reach (trial) of one participant in one experiment",
    how=[
        "One row is one joystick reach (5 cm centre-out to one of four targets at 0, 60, 120 or 180 deg) of one participant in one "
        "experiment: 80 baseline, 320 adaptation with a 30 deg rotation of the cursor, 40 after-effect trials without cursor, "
        "40 unlearning trials with veridical cursor. The target is a spoken word (cue_elements word:BAD, word:TAP or word:HIT) or "
        "a green circle (circle); on 32 adaptation trials per person (one in ten) and in the after-effect block the cursor is hidden.",
        "hand_angle is the authors' angular error at peak hand velocity, re-signed: hand direction minus target direction, counter-clockwise "
        "positive, the same quantity as in the other reaching studies. The authors store it with the opposite sign (their rotation is "
        "-30 and their after-effects are positive); here rotation is +30 (counter-clockwise, as the Methods say), cursor_error is "
        "minus the authors' value, and hand_angle is cursor_error minus the rotation. This sign was inferred from the paper's wording and "
        "confirmed with the logged joystick coordinates, whose direction at peak speed equals hand_angle on trials without cursor and "
        "hand_angle + 30 on cursor trials during adaptation (the logged position on those trials is the rotated cursor).",
        "Groups follow the authors' condition variable (Experiment 1: 1 = adapted mostly with the spoken word, 2 = adapted mostly with "
        "the circle; Experiment 2: 1 = tap, 2 = hit). The paper's Methods and Results describe the two Experiment 2 groups in different "
        "orders; the shared data show condition 1 with 304 'tap' trials and condition 2 with 304 'hit' trials, and the converted rows "
        "reproduce the paper's Results numbers with that assignment. excluded_by_authors marks trials without a measured error and "
        "trials whose error exceeds 40 deg (the paper removes errors beyond 40 deg, under 1 percent of the data; the authors' script "
        "does not apply this rule to the shared values). other_warning marks the timing messages shown after the trial "
        "(Move sooner, Move later, Don't speak, No response); the movement time, speech timing, peak velocity and the joystick "
        "traces the deposit also holds are not included.",
        "P02_EX1 and P07_EX1 have 479 trials and P18_EX1 460: the deposit's arrays are shorter than the 480 trials of the design; rows "
        "are kept as shared, and the phase of each row follows the authors' after-effect indices (afind_T1, afind_T2).",
    ],
    differences=differences,
)
stdlib.release(ID, "reach90", rows, meta, checks)
