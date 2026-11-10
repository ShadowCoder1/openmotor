"""Mehler, Reichenbach, Klein & Diedrichsen (2017), PLoS ONE: reward-based learning of a wrist movement through injected
endpoint noise during 3-joint (shoulder, elbow, wrist) reaching, 41 participants x 630 reaches, in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Mehler2017_PLoSOne.py
Raw files (untrusted data, read only) in raw_extra/Mehler2017_PLoSOne/, from https://osf.io/36kyr/:
  DOF3rfl1_alldat_OSF.mat   one row per reach (25830 x variables), DOF3rfl1_alltraj_OSF.mat   block-mean time courses
  DOF3_rfl1_data_documentation.pdf (read as text, not part of the release)

hand_angle is not a stored variable: it is the direction of the final fingertip position seen from the start box (0, -10 cm)
relative to the direction start box -> target box (7.5, 3 cm), i.e. 60.02 deg, counter-clockwise positive with x to the right
and y away from the body (inferred: the paper puts the target 15 cm diagonally right of the start box).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mehler2017_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
START = np.array([0.0, -10.0])
TARGET = np.array([7.5, 3.0])
DIST = float(np.hypot(*(TARGET - START)))
TARGET_ANGLE = math.degrees(math.atan2(*(TARGET - START)[::-1]))
GROUP = {1: "flexion_reward", 2: "extension_reward", 3: "control_constant_noise_2cm", 4: "control_no_added_noise"}


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


m = sio.loadmat(RAW / "DOF3rfl1_alldat_OSF.mat")


def g(k):
    a = m[k].astype(float)
    return a.ravel() if a.shape[1] == 1 else a


SN, CN, BN, TN, BT = (g(k) for k in ("SN", "CN", "BN", "TN", "BT"))
RT, MT, PTS, GOOD, AW, AFC = (g(k) for k in ("RT", "MT", "points", "good", "Aware", "AFC"))
fp = np.c_[g("finalPosX"), g("finalPosY")]
fd = np.c_[g("finalPosXDisp"), g("finalPosYDisp")]
noise = g("noiseAdd")
nd = g("noiseAddDist")
tc = g("thetaEndChange")
th_end = g("thetaEnd")
ph_end = g("phiEnd")
n = len(SN)

# ---------------------------------------------------------------- structure checks
subs = list(dict.fromkeys(SN.tolist()))  # order of first appearance
checks.internal("rows are grouped by participant and ordered by block then trial", all(
    np.all(np.diff(BN[SN == s] * 1000 + TN[SN == s]) > 0) for s in subs))
checks.internal("every participant has 9 blocks x 70 trials = 630 rows, block 1 is the baseline (BT 3)", all(
    (SN == s).sum() == 630 and set(BT[(SN == s) & (BN == 1)]) == {3.0} and set(BT[(SN == s) & (BN > 1)]) == {4.0} for s in subs))
checks.internal("start (0,-10) and target (7.5,3) cm constant on every row", bool(
    np.all(g("startX") == 0) and np.all(g("startY") == -10) and np.all(g("targetX") == 7.5) and np.all(g("targetY") == 3)))
checks.internal("displayed endpoint minus actual endpoint equals the injected noise on every row that has a stored noise value",
                float(np.nanmax(np.abs((fd - fp) - noise))) < 1e-9)
d_disp = np.hypot(*(fd - TARGET).T)
strict = (MT < 700) & (g("maxSpeed") >= 45) & (g("maxSpeed") <= 100)
checks.internal("reward = valid trial (MT < 700 ms, speed 45-100 cm/s, the paper's in-task criteria) with the displayed endpoint "
                "within 1 cm of the target, on 99.9 % of rows",
                float(np.mean(PTS == (strict & (d_disp <= 1.0)))) > 0.999, f"{np.mean(PTS == (strict & (d_disp <= 1.0))):.5f}")

cnt = {c: len({s for s in subs if CN[SN == s][0] == c}) for c in GROUP}
aware = {s: AW[SN == s][0] for s in subs}
unaware = lambda c: [s for s in subs if CN[SN == s][0] == c and aware[s] != 1]  # noqa: E731
SRC = "Methods, Participants"
checks.published("flexion group, participants", cnt[1], 13, 0, SRC + " (flexion experimental group, n = 13)")
checks.published("extension group, participants in the deposit (13 recruited, 1 excluded before analysis)", cnt[2], 12, 0,
                 SRC + " (extension group n = 13; one excluded for changing body position)")
checks.published("low-noise control group, participants", cnt[4], 9, 0, SRC)
checks.published("high-noise control group, participants", cnt[3], 7, 0, SRC)
checks.published("aware participants in the flexion group", sum(aware[s] == 1 for s in subs if CN[SN == s][0] == 1), 2, 0,
                 "Results, awareness (2 flexion aware)")
checks.published("aware participants in the extension group", sum(aware[s] == 1 for s in subs if CN[SN == s][0] == 2), 3, 0,
                 "Results, awareness (3 extension aware)")
checks.published("unaware flexion participants", len(unaware(1)), 11, 0, "Results (final sample 11 flexion)")
checks.published("unaware extension participants", len(unaware(2)), 9, 0, "Results (final sample 9 extension)")
checks.published("high-noise control, participants left after the 2 who detected the manipulation", len(unaware(3)), 5, 0,
                 "Results, control groups (2 of 7 excluded)")
corr2afc = sum(AFC[SN == s][0] == 1 for s in subs if CN[SN == s][0] in (1, 2) and aware[s] != 1)
checks.published("unaware experimental participants answering the 2AFC question correctly", corr2afc, 10, 0,
                 "Results, awareness ('only 10 participants answered correctly')")


def block_means(cn, b, col, only_good=True, aware_ok=False):
    out = []
    for s in subs:
        if CN[SN == s][0] != cn or (not aware_ok and aware[s] == 1):
            continue
        j = (SN == s) & (BN == b) & ((GOOD == 1) if only_good else True)
        out.append(np.nanmean(col[j]))
    return np.array(out)


def tstat(x):
    return float(np.mean(x) / (np.std(x, ddof=1) / math.sqrt(len(x))))


# the authors' 'good' trials are the ones their block means use (t values below only match with them)
EXT = "Results, Learning (extension group vs 0, df = 8)"
checks.published("extension (unaware), change in wrist angle vs 0, block B7, t(8)", tstat(block_means(2, 8, tc[:, 2])), 3.561, 0.0006, EXT)
checks.published("extension (unaware), change in wrist angle vs 0, block B8, t(8)", tstat(block_means(2, 9, tc[:, 2])), 3.159, 0.0006, EXT)
hi = block_means(3, 2, tc[:, 2])
checks.published("high-noise control (unaware), change in wrist angle block B1, mean (deg)", hi.mean(), 2.32, 0.005, "Results, control groups")
checks.published("high-noise control (unaware), change in wrist angle block B1, SEM (deg)", np.std(hi, ddof=1) / math.sqrt(len(hi)), 0.82,
                 0.005, "Results, control groups")
checks.published("high-noise control (unaware), block B1 vs 0, t(4)", tstat(hi), 2.823, 0.0006, "Results, control groups")
lo = block_means(4, 9, tc[:, 2], aware_ok=True)
checks.published("low-noise control, block B8 vs 0, t(8)", tstat(lo), -0.650, 0.001, "Results, control groups")

# ---------------------------------------------------------------- rows
ang_hand = np.array([wrap(math.degrees(math.atan2(y - START[1], x - START[0])) - TARGET_ANGLE) for x, y in fp])
ang_disp = np.array([wrap(math.degrees(math.atan2(y - START[1], x - START[0])) - TARGET_ANGLE) for x, y in fd])
ext_hand = np.hypot(*(fp - START).T) / DIST
HALF = math.degrees(math.atan2(1.0, DIST))
rows, supp = [], []
for i_s, s in enumerate(subs, start=1):
    code = f"{ID}_S{i_s:04d}"
    cn = int(CN[SN == s][0])
    excl_sub = aware[s] == 1
    prev_b = None
    for j, i in enumerate(np.where(SN == s)[0]):
        b = int(BN[i])
        row = dict.fromkeys(names, "")
        row.update(
            dataset=ID, experiment=ID, study=ID, subject=code, person=f"{ID}_P{i_s:04d}", id_type="lab",
            person_link="assumed_unique", group=GROUP[cn], prior_exposure="unknown",
            subject_excluded_by_authors="True" if excl_sub else "False",
            subject_exclusion_reason="classified as aware of the noise manipulation (debriefing)" if excl_sub else "",
            handedness="right", device="robot", setting="lab", population="healthy",
            trial=str(j), trial_orig=str(int(TN[i])), session="0", block=f"B{b - 1}", phase="baseline" if b == 1 else "learning",
            trial_kind="reach", break_before="True" if j == 0 else "False",
            target_angle=num(TARGET_ANGLE), target_id="0", target_size_deg=num(math.degrees(math.atan2(0.5, DIST))),
            target_distance_norm="1", target_event="none", target_jump="0", probe_kind="none",
            instr_goal="cursor_to_target", instr_strategy="unspecified", instr_disclosure="none",
            instr_report="none", hand_used="right", hand_angle=num(ang_hand[i]), hand_angle_fb=num(ang_disp[i]),
            hand_measure="endpoint", hand_extent=num(ext_hand[i]), rt_ms=num(RT[i]), mt_ms=num(MT[i]) if MT[i] > 0 else "",
            excluded_by_authors="True" if GOOD[i] == 0 else "False",
            fb_visible="endpoint", fb_mode="other", fb_form="cursor", online_cursor="none", fb_endpoint_transform_only="False",
            cursor_error=num(ang_disp[i]), cursor_extent_error=num(np.hypot(*(fd[i] - START)) / DIST - 1.0),
            outcome_channels="cursor|points", reward=num(PTS[i]), score=num(PTS[i]), score_max="1",
            reward_space="cursor", reward_center_deg="0", reward_halfwidth_deg=num(HALF), hit=num(float(d_disp[i] <= 1.0)),
        )
        rows.append(row)
        supp.append([code, int(BN[i]) - 1, int(TN[i]), int(BT[i]), int(g("max_noise")[i]), int(GOOD[i]), AW[i], AFC[i],
                     fp[i, 0], fp[i, 1], fd[i, 0], fd[i, 1], noise[i, 0], noise[i, 1], nd[i],
                     th_end[i, 0], th_end[i, 1], th_end[i, 2], ph_end[i, 0], ph_end[i, 1], ph_end[i, 2],
                     tc[i, 0], tc[i, 1], tc[i, 2], g("maxSpeed")[i], g("maxVelF")[i]])
checks.internal("every row the authors shared is kept, in order (25830 = 41 x 630)", len(rows) == n == 41 * 630)
checks.internal("hand angles finite and within (-180, 180]; MT blanked only where the authors stored -1 (3 rows)",
                all(-180 < float(r["hand_angle"]) <= 180 for r in rows) and sum(r["mt_ms"] == "" for r in rows) == 3)
checks.internal("hand angle (actual endpoint) has median ~0 deg relative to the target and the displayed endpoint is wider",
                abs(float(np.median(ang_hand))) < 0.5 and np.std(ang_disp) > np.std(ang_hand))

SUPP_HEAD = ["subject", "block", "trial_in_block", "block_type_code", "max_noise_cm", "good_trial", "aware", "afc_correct",
             "final_x_cm", "final_y_cm", "shown_x_cm", "shown_y_cm", "noise_x_cm", "noise_y_cm", "noise_total_cm",
             "theta_end_shoulder_deg", "theta_end_elbow_deg", "theta_end_wrist_deg", "phi_end_shoulder_deg",
             "phi_end_elbow_deg", "phi_end_wrist_deg", "theta_change_shoulder_deg", "theta_change_elbow_deg",
             "theta_change_wrist_deg", "max_speed", "max_endpoint_velocity"]
import csv  # noqa: E402
import io  # noqa: E402


def fmt(v):
    if isinstance(v, str):
        return v
    return "" if v != v else ("%.8g" % v)


bio = io.StringIO()
w = csv.writer(bio, lineterminator="\r\n")
w.writerow(SUPP_HEAD)
for r in supp:
    w.writerow([r[0]] + [fmt(float(x)) for x in r[1:]])

# block-mean time courses
tr = sio.loadmat(RAW / "DOF3rfl1_alltraj_OSF.mat")
vars_ = [k for k in tr if not k.startswith("__") and k not in ("SN", "CN", "BN", "t")]
code_of = {s: f"{ID}_S{i:04d}" for i, s in enumerate(subs, start=1)}
tb = io.StringIO()
tw = csv.writer(tb, lineterminator="\r\n")
tw.writerow(["subject", "block", "time_ms"] + vars_)
for r in range(tr["SN"].shape[0]):
    s = float(tr["SN"][r, 0])
    for k in range(tr["t"].shape[1]):
        tw.writerow([code_of[s], int(tr["BN"][r, 0]) - 1, int(tr["t"][r, k])] + [fmt(float(tr[v][r, k])) for v in vars_])
checks.internal("block-mean time courses: 369 = 41 x 9 rows of 149 samples", tr["SN"].shape[0] == 369 and tr["t"].shape[1] == 149)

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Diedrichsen (UCL / Western)",
    data_url="https://osf.io/36kyr/",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format, participant numbers replaced by codes, hand angle derived from the final fingertip "
            "position, a missing movement time (-1) left empty; joint angles and block-mean time courses are in two extra tables.",
    unit_note="reach of one participant sequence",
    how=[
        "One row is one reach by a participant whose right hand was attached to a 3-joint (shoulder, elbow, wrist) robot "
        "manipulandum, 70 reaches in each of 9 blocks (B0 baseline, B1 to B8 learning), from a start box to a target box 15 cm "
        "away (60 deg in the authors' x-y plane). The cursor was hidden during the movement and shown at the end with "
        "injected Gaussian noise; reward was one point for a valid reach ending within 1 cm of the target on the screen.",
        "hand_angle is the direction of the actual final fingertip position, hand_angle_fb that of the displayed (noisy) endpoint, "
        "both relative to the target direction in degrees. The sign (counter-clockwise positive, x right, y away from the body) is "
        "inferred, because the paper only says that the target is diagonally to the right of the start box.",
        "The groups are the flexion and extension experimental groups (noise smaller when the wrist moves 5 deg beyond baseline in "
        "that direction) and two controls (constant 2 cm noise; no added noise). The group codes were read from the data: group 3 "
        "has 7 participants with constant 2 cm noise and group 4 has 9 with none.",
        "The authors' learning variable is the change of the joint angles from baseline (theta_change_wrist_deg, positive = extension). "
        "It, the joint angles, the injected noise and the 'good' flag (excluded_by_authors is True where good = 0; the authors' block "
        "means use only good trials) are in trial_details.csv; block_mean_trajectories.csv has the block-averaged time courses "
        "(5 ms steps, variables as the authors named them).",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks, extra_files=[("trial_details.csv", bio.getvalue()),
                                                              ("block_mean_trajectories.csv", tb.getvalue())])
