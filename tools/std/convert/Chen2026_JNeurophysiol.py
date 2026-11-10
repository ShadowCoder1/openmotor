"""Chen & Taylor (2026), J Neurophysiol: 45 deg visuomotor rotation with aim reports (aiming group) and process
dissociation probes (Exclusion / Inclusion trials), 60 participants, one row per reach, in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Chen2026_JNeurophysiol.py
Raw files (untrusted data, read only) in raw_extra/Chen2026_JNeurophysiol/, from https://osf.io/zt8pd/:
  all_raw_df.csv              trial-level data of the paper (400 trials x 60 participants)
  all_control_survey_df.csv,  post-experiment instruction check; read only to reproduce the authors' exclusion
  all_aiming_survey_df.csv
The authors' data_analysis.ipynb and helper_function.py were read as text to learn the variables, never run.
all_raw_pilot_df.csv (the pilot of the authors' 2025 CogSci paper, a separate study) is not converted here.

Sign convention. In the authors' file final_HA and aim_angle are flipped per participant so that compensation for the
rotation is positive: final_HA = flip * wrap(atan2(final_hand_Y, final_hand_X) - target_angle), with flip = +1 when the
rotation on the last training trial is negative (clockwise, 'cw -> -45' in the notebook) and -1 otherwise (notebook cells
28-31; paper, Methods, Data Analysis). Multiplying by the same flip gives back counter-clockwise-positive angles in the
frame of target_angle and rotation, which is the OpenMotor convention. The checks below confirm this from the raw
coordinates and from where participants tapped their aim reports on the screen.
"""
import csv
import json
import math
import statistics as st
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Chen2026_JNeurophysiol"
RAW = stdlib.raw_dir(ID)
csv.field_size_limit(2 ** 31 - 1)  # the authors' hand_X / hand_Y trajectory cells are long strings (not converted)

SET_STARTS = [241, 256, 271, 286, 301, 316, 331, 346]  # first trial of each 5-trial Exclusion or Inclusion set
LOC = {0: "training_location", 1: "aiming_location"}    # third digit of the authors' set label
DELAY = {0: "no_delay", 1: "1min_delay"}                 # second digit of the authors' set label
SURVEY_THRESHOLD = round(math.degrees(math.atan(4 / 70)) * 2, 1)  # 6.5 deg: 0.8 cm target width on the 7 cm ring
TARGET_RADIUS_DEG = math.degrees(math.atan(0.5 / 7.0))            # 1 cm target at 7 cm (Methods, Task and Apparatus)

checks = stdlib.Checks()


def num(x):
    """Number format of the 97 standardized studies: up to 8 significant digits, no trailing zeros."""
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


def tf(x):
    return "True" if x else "False"


def wrap(a):
    """Degrees into (-180, 180]."""
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def fl(s):
    return None if s in ("", "nan", "NaN", "None") else float(s)


# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for fn in ("all_raw_df.csv", "all_control_survey_df.csv", "all_aiming_survey_df.csv"):
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
raw = list(csv.DictReader(open(RAW / "all_raw_df.csv", newline="", encoding="utf-8")))
subs = OrderedDict()
for r in raw:
    subs.setdefault(r["sub_ID"], []).append(r)
group = {s: v[0]["Group"] for s, v in subs.items()}
checks.internal("24000 rows = 60 participants x 400 trials, trials 1-400 in order, rows grouped by participant",
                len(raw) == 24000 and len(subs) == 60
                and all([int(r["trial"]) for r in v] == list(range(1, 401)) for v in subs.values()))
checks.internal("group constant within participant: 30 Aiming, 30 Control",
                all(len({r["Group"] for r in v}) == 1 for v in subs.values())
                and Counter(group.values()) == Counter({"Aiming": 30, "Control": 30}))
checks.published("participants recruited", len(subs), 60, 0, "Methods, Participants ('Sixty participants were recruited')")


# ---------------------------------------------------------------- trial structure
def kind_of(t, r):
    if t <= 40:
        return "baseline"
    if t <= 240:
        return "training"
    if t >= 361:
        return "washout"
    lab = int(float(r["original_ite_label"])) // 100
    return {5: "exclusion", 6: "inclusion"}.get(lab, "topup")


set_trials = {t for s in SET_STARTS for t in range(s, s + 5)}
fb_trials = set(range(1, 11)) | set(range(21, 241)) | (set(range(241, 361)) - set_trials)
ok_kind = ok_fb = ok_flags = ok_target = ok_rot = ok_blocks = True
cond = {}       # participant -> {pdp block 1..4: (location digit, delay digit)}
rot_train = {}  # participant -> rotation of the training block (the authors' flip uses trial 240)
for s, v in subs.items():
    rot_train[s] = fl(v[239]["rotation"])
    ok_rot &= rot_train[s] in (-45.0, 45.0) and all(fl(r["rotation"]) == rot_train[s] for r in v[40:240])
    cond[s] = {}
    for k in range(4):
        labs = [int(float(v[SET_STARTS[2 * k + j] - 1]["original_ite_label"])) for j in (0, 1)]
        ok_blocks &= sorted(x // 100 for x in labs) == [5, 6] and labs[0] % 100 == labs[1] % 100
        cond[s][k + 1] = (labs[0] % 10, (labs[0] // 10) % 10)
    ok_blocks &= len(set(cond[s].values())) == 4
    for r in v:
        t = int(r["trial"])
        kd = kind_of(t, r)
        ok_kind &= (kd in ("exclusion", "inclusion")) == (t in set_trials)
        on, end = fl(r["online_feedback"]), fl(r["endpoint_feedback"])
        ok_fb &= on == end == (1.0 if t in fb_trials else 0.0)
        if kd in ("topup", "baseline", "washout", "exclusion"):
            ok_rot &= fl(r["rotation"]) == (rot_train[s] if kd == "topup" else 0.0)
        if t in set_trials:
            loc, dl = cond[s][(t - 241) // 30 + 1]
            first = t in SET_STARTS
            ok_flags &= int(r["target_at_aim"]) == loc and int(r["delay_trial"]) == (60 if (first and dl) else 0)
            ok_target &= (abs(fl(r["target_angle"]) - 90) < 1e-9) == (loc == 0)
        else:
            ok_flags &= int(r["target_at_aim"]) == 0 and int(r["delay_trial"]) == 0
            ok_target &= abs(fl(r["target_angle"]) - 90) < 1e-9
checks.internal("Exclusion/Inclusion sets are trials 241-245, 256-260, ... 346-350 for everyone (rest of 241-360 are Top-up)",
                ok_kind)
checks.internal("cursor shown (online and endpoint) on trials 1-10, 21-240 and Top-up trials, never on trials 11-20, "
                "sets or washout (Methods)", ok_fb)
checks.internal("each PDP block (30 trials) has one Exclusion and one Inclusion set in the same condition, "
                "and each participant has the four Target Location x Delay conditions once", ok_blocks)
checks.internal("target_at_aim and the 60 s delay flag agree with the set labels; delay only before a set's first trial",
                ok_flags)
checks.internal("target at 90 deg on all trials except the sets probed at the aiming location", ok_target)
checks.internal("rotation -45 or +45 on all 200 training trials and on Top-up trials, 0 on baseline, Exclusion and washout",
                ok_rot)
checks.internal("rotation direction counterbalanced: 15 clockwise and 15 counter-clockwise per group",
                Counter((group[s], rot_train[s]) for s in subs) == Counter({("Aiming", -45.0): 15, ("Aiming", 45.0): 15,
                                                                         ("Control", -45.0): 15, ("Control", 45.0): 15}))

# ---------------------------------------------------------------- signs
flip = {s: (1.0 if rot_train[s] < 0 else -1.0) for s in subs}  # the authors' flip_fac
maxdiff = 0.0
for s, v in subs.items():
    for r in v:
        recomputed = wrap(math.degrees(math.atan2(fl(r["final_hand_Y"]), fl(r["final_hand_X"]))) - fl(r["target_angle"]))
        maxdiff = max(maxdiff, abs(wrap(recomputed - flip[s] * fl(r["final_HA"]))))
checks.internal("hand angle recomputed from final hand x, y and target_angle equals the flipped-back final_HA on every row",
                maxdiff < 1e-9, f"max difference {maxdiff:.2e} deg")
late = {s: st.mean(flip[s] * fl(r["final_HA"]) for r in v[230:240]) for s, v in subs.items()}
checks.internal("clockwise (-45) participants compensate counter-clockwise in late training (mean hand angle > 0), "
                "counter-clockwise (+45) participants clockwise (< 0), all 60",
                all((late[s] > 0) == (rot_train[s] < 0) for s in subs))
# aim taps are in screen pixels (y down); solve the start position (cx, cy) that the authors' aim angles refer to
A = [[0.0, 0.0], [0.0, 0.0]]
bv = [0.0, 0.0]
taps = []
for s, v in subs.items():
    for r in v:
        if r["aim_report"] == "1.0":
            th = math.radians(fl(r["target_angle"]) + flip[s] * fl(r["aim_angle"]))
            x, y = fl(r["aim_X"]), fl(r["aim_Y"])
            sn, cs = math.sin(th), math.cos(th)
            A[0][0] += sn * sn; A[0][1] += sn * cs; A[1][1] += cs * cs  # noqa: E702
            bv[0] += sn * (x * sn + y * cs); bv[1] += cs * (x * sn + y * cs)  # noqa: E702
            taps.append((s, int(r["trial"]), x, y, th))
det = A[0][0] * A[1][1] - A[0][1] ** 2
cx = (bv[0] * A[1][1] - bv[1] * A[0][1]) / det
cy = (A[0][0] * bv[1] - A[0][1] * bv[0]) / det
resid = max(abs(wrap(math.degrees(math.atan2(cy - y, x - cx) - th))) for s, t, x, y, th in taps)
checks.internal("one fixed start position on the screen fits every reported aim angle (9900 aim taps)", resid < 1e-6,
                f"start at ({cx:.1f}, {cy:.1f}) px, max residual {resid:.1e} deg")
late_taps = [(s, x < cx) for s, t, x, y, th in taps if 231 <= t <= 240]
left_cw = st.mean(lft for s, lft in late_taps if rot_train[s] < 0)
left_ccw = st.mean(lft for s, lft in late_taps if rot_train[s] > 0)
checks.internal("on the screen, clockwise-rotation participants tap their aim left of the start in late training "
                "(counter-clockwise of the 90 deg target), counter-clockwise-rotation participants right of it",
                left_cw > 0.9 and left_ccw < 0.1, f"share of taps left of start: {left_cw:.3f} vs {left_ccw:.3f}")

# ---------------------------------------------------------------- the authors' exclusions, reproduced from their rules
excluded = {}
for fn in ("all_control_survey_df.csv", "all_aiming_survey_df.csv"):
    for r in csv.DictReader(open(RAW / fn, newline="", encoding="utf-8")):
        if "Exclusion" in r["Condition"] and abs(float(r["AimAngle"]) - float(r["TargetAngle"])) > SURVEY_THRESHOLD:
            where = "aiming" if "Aiming_Location" in r["Condition"] else "training"
            excluded.setdefault(r["sub_ID"], "post-experiment survey: aimed more than 6.5 deg away from the target on "
                                             f"the Exclusion question with the target at the {where} location")
for lab in (600, 601, 610, 611):
    means = {s: st.mean(fl(r["aim_angle"]) for r in v if int(float(r["original_ite_label"])) == lab)
             for s, v in subs.items() if group[s] == "Aiming"}
    m, sd = st.mean(means.values()), st.stdev(means.values())
    for s, x in means.items():
        if abs(x - m) > 4 * sd:
            excluded.setdefault(s, "aim reports more than 4 SD from the group mean in the Inclusion set with the target "
                                   f"at the {LOC[lab % 10].split('_')[0]} location, {DELAY[(lab // 10) % 10].replace('_', ' ')}")
for g in ("Aiming", "Control"):
    means = {s: st.mean(fl(r["final_HA"]) for r in v if int(float(r["original_ite_label"])) // 100 == 6)
             for s, v in subs.items() if group[s] == g}
    m, sd = st.mean(means.values()), st.stdev(means.values())
    for s, x in means.items():
        if abs(x - m) > 4 * sd:
            excluded.setdefault(s, "Inclusion hand angle more than 4 SD from the group mean")
checks.internal("the authors' criteria flag exactly two participants, one per group (Methods, Testing Participants' "
                "Understanding of the Task)", Counter(group[s] for s in excluded) == Counter({"Aiming": 1, "Control": 1}))
for g in ("Aiming", "Control"):
    checks.published(f"participants analysed, {g.lower()} group", sum(group[s] == g and s not in excluded for s in subs),
                     29, 0, "Methods, Participants (n = 29 per group)")

# ---------------------------------------------------------------- rows
names = [c[1] for c in stdlib.reach90_columns()]
target_ids = {}
rows = []
for i, (s, v) in enumerate(subs.items(), start=1):
    aiming = group[s] == "Aiming"
    prev_instruction = None
    for j, r in enumerate(v):
        t = int(r["trial"])
        kd = kind_of(t, r)
        fb = t in fb_trials
        rot = fl(r["rotation"])
        ta = fl(r["target_angle"]) % 360.0
        tid = target_ids.setdefault(round(ta, 6), len(target_ids))
        ha = wrap(flip[s] * fl(r["final_HA"]))
        has_aim = aiming and r["aim_report"] == "1.0"
        if kd in ("exclusion", "inclusion", "topup"):
            k = (t - 241) // 30 + 1
            loc, dl = cond[s][k]
            block = f"pdp{k}_{kd}_{LOC[loc]}_{DELAY[dl]}"
        else:
            block = kd
        phase = {"baseline": "baseline", "training": "learning", "topup": "learning", "exclusion": "probe",
                 "inclusion": "probe", "washout": "aftereffect"}[kd]
        if kd == "exclusion" or kd == "washout":
            goal, strategy = "hand_to_target", "remove"
        elif kd == "inclusion":
            goal, strategy = "cursor_to_target", "maintain"
        elif kd in ("training", "topup"):
            goal, strategy = "cursor_to_target", "allowed"
        else:  # baseline: cursor trials, and the 10 no-cursor trials 11-20
            goal, strategy = ("cursor_to_target" if fb else "hand_to_target"), "unspecified"
        row = dict.fromkeys(names, "")
        row.update(
            dataset=ID, experiment=f"{ID}_Main", study=ID, subject=f"{ID}_S{i:04d}", person=f"{ID}_P{i:04d}",
            id_type="lab", person_link="assumed_unique", group=group[s], prior_exposure="unknown",
            subject_excluded_by_authors=tf(s in excluded), subject_exclusion_reason=excluded.get(s, ""),
            handedness="right", device="tablet", setting="lab", population="healthy",
            trial=str(j), trial_orig=str(t), session="0", block=block, phase=phase, trial_kind="reach",
            break_before=tf(j == 0 or (prev_instruction or 0) != 0),
            target_angle=num(ta), target_id=str(tid), target_size_deg=num(TARGET_RADIUS_DEG), target_distance_norm="1",
            target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
            instr_goal=goal, instr_strategy=strategy, instr_disclosure="exists",
            instr_report="aim_pre" if has_aim else "none", aim_report_timing="pre" if has_aim else "",
            cue_elements="target_color:red" if kd == "exclusion" else "", fb_cued="True",
            hand_used="right", hand_angle=num(ha), hand_angle_fb=num(ha), hand_measure="fixed_distance",
            rt_ms=num(1000 * fl(r["rt"])), mt_ms=num(1000 * fl(r["movement_time"])),
            aim_report=num(wrap(flip[s] * fl(r["aim_angle"]))) if has_aim else "",
            excluded_by_authors="False",
            fb_visible="online" if fb else "none",
            fb_mode=("veridical" if rot == 0 else "rotated") if fb else "none",
            fb_form="cursor" if fb else "none",
            online_cursor=("veridical" if rot == 0 else "perturbed") if fb else "none",
            fb_endpoint_transform_only="False",
            rotation=num(rot) if fb else "",
            cursor_error=num(wrap(ha + rot)) if fb else "",
            fb_delay_ms="0" if fb else "",
            outcome_channels="cursor" if fb else "",
        )
        rows.append(row)
        prev_instruction = fl(r["instruction"])
checks.internal("every row the authors shared is kept, in order", len(rows) == len(raw))
checks.internal("hand_angle and aim_report within (-180, 180], target_angle within [0, 360), rt and mt finite",
                all(-180 < float(r["hand_angle"]) <= 180 for r in rows)
                and all(-180 < float(r["aim_report"]) <= 180 for r in rows if r["aim_report"])
                and all(0 <= float(r["target_angle"]) < 360 for r in rows)
                and all(math.isfinite(float(r["rt_ms"])) and math.isfinite(float(r["mt_ms"])) for r in rows))
checks.internal("aim reports on trials 31-360 of the 30 aiming-group participants only (9900 rows)",
                sum(1 for r in rows if r["aim_report"]) == 9900
                and all(31 <= int(r["trial_orig"]) <= 360 for r in rows if r["aim_report"]))
checks.internal("instruction screens give pauses before trials 11, 21, 41, 361 and every set and Top-up run "
                "(and 31 in the aiming group)",
                Counter(int(r["trial_orig"]) for r in rows if r["break_before"] == "True")
                == Counter({t: n for t, n in [(1, 60), (11, 60), (21, 60), (31, 30), (41, 60), (361, 60)]
                            + [(x, 60) for x in SET_STARTS] + [(x + 5, 60) for x in SET_STARTS]}))

# ---------------------------------------------------------------- published numbers, from the converted rows
by_sub = OrderedDict()
for r in rows:
    by_sub.setdefault(r["subject"], []).append(r)


def comp(v, select, col="hand_angle"):
    """Participant mean of a column over selected rows, signed so that compensation for the rotation is positive."""
    sign = 1.0 if float(next(r["rotation"] for r in v if r["block"] == "training")) < 0 else -1.0
    return st.mean(sign * float(r[col]) for r in v if select(r))


def slope(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def sem(x):
    return st.stdev(x) / math.sqrt(len(x))


def tr(r):
    return int(r["trial_orig"])


def kind(r):
    return r["block"].split("_")[1] if r["block"].startswith("pdp") else r["block"]


def cnd(r):
    return r["block"].split("_", 2)[2] if r["block"].startswith("pdp") else ""


# The authors' Top-up measure = the 5 cursor trials just before each set (their 7xxx labels; trials 236-240 before the
# first set are the last training trials). next_set gives the set that a pre-set trial leads into.
PRE_TOPUP = {t for s0 in SET_STARTS for t in range(s0 - 5, s0)}


def next_set(t):
    return min(x for x in SET_STARTS if x > t)


RES = "Results, 'Participants successfully adapted'"
SUM = "Results, 'Summation test'"
reported = {
    "Aiming": dict(late=(43.75, 0.36), topup_tn=(43.89, 0.49), incl_tn=(44.01, 0.89), total=(43.79, 2.02),
                   implicit=(13.18, 6.20), explicit=(26.16, 11.6), D=(4.44, 10.26), t=2.33, slope=-0.281),
    "Control": dict(late=(43.15, 0.46), topup_tn=(43.47, 0.30), incl_tn=(44.46, 0.82), total=(43.95, 1.25),
                    implicit=(13.46, 7.65), explicit=(23.15, 10.9), D=(7.34, 9.29), t=4.25, slope=-0.388),
}
for g, rep in reported.items():
    vs = [v for v in by_sub.values() if v[0]["group"] == g and v[0]["subject_excluded_by_authors"] == "False"]
    lt = [comp(v, lambda r: 231 <= tr(r) <= 240) for v in vs]
    checks.published(f"{g}: mean hand angle, last 10 training trials", st.mean(lt), rep["late"][0], 0.005, RES)
    checks.published(f"{g}: SEM of that mean", sem(lt), rep["late"][1], 0.005, RES)
    tu = []
    for v in vs:
        set_cond = {tr(r): cnd(r) for r in v if tr(r) in SET_STARTS}
        tu.append(comp(v, lambda r: tr(r) in PRE_TOPUP and set_cond[next_set(tr(r))] == "training_location_no_delay"))
    inc = [comp(v, lambda r: kind(r) == "inclusion" and cnd(r) == "training_location_no_delay") for v in vs]
    checks.published(f"{g}: Top-up, target at training location, no delay (mean)", st.mean(tu), rep["topup_tn"][0], 0.005, RES)
    checks.published(f"{g}: Top-up, training location, no delay (SEM)", sem(tu), rep["topup_tn"][1], 0.005, RES)
    checks.published(f"{g}: Inclusion, training location, no delay (mean)", st.mean(inc), rep["incl_tn"][0], 0.005, RES)
    checks.published(f"{g}: Inclusion, training location, no delay (SEM)", sem(inc), rep["incl_tn"][1], 0.005, RES)
    total = [comp(v, lambda r: tr(r) in PRE_TOPUP) for v in vs]
    implicit = [comp(v, lambda r: 361 <= tr(r) <= 365) for v in vs]
    explicit = [comp(v, lambda r: kind(r) == "inclusion") - comp(v, lambda r: kind(r) == "exclusion") for v in vs]
    D = [a - (i + e) for a, i, e in zip(total, implicit, explicit)]
    for key, name, x in (("total", "total adaptation (Top-up)", total),
                         ("implicit", "implicit (aftereffect, washout trials 1-5)", implicit),
                         ("explicit", "explicit (Inclusion - Exclusion)", explicit),
                         ("D", "D = A - (I + E)", D)):
        tol_sd = 0.05 if key == "explicit" else 0.005  # the paper gives this SD to one decimal
        checks.published(f"{g}: {name}, mean", st.mean(x), rep[key][0], 0.005, SUM)
        checks.published(f"{g}: {name}, SD", st.stdev(x), rep[key][1], tol_sd, SUM)
    checks.published(f"{g}: t(28) of D against 0", st.mean(D) / sem(D), rep["t"], 0.005, SUM)
    checks.published(f"{g}: slope of implicit (aftereffect) on explicit (Inclusion - Exclusion)", slope(explicit, implicit),
                     rep["slope"], 0.0005, "Table 2, observed slope" if g == "Aiming" else "Table 1 and Table 2, control")
# Table 1, aiming group: implicit = Exclusion hand angle, explicit = aim report on the Inclusion trials, per condition
va = [v for v in by_sub.values() if v[0]["group"] == "Aiming" and v[0]["subject_excluded_by_authors"] == "False"]
for c, rep_slope in (("training_location_no_delay", -0.197), ("training_location_1min_delay", -0.158),
                     ("aiming_location_no_delay", -0.588), ("aiming_location_1min_delay", -0.401)):
    x = [comp(v, lambda r: kind(r) == "inclusion" and cnd(r) == c, "aim_report") for v in va]
    y = [comp(v, lambda r: kind(r) == "exclusion" and cnd(r) == c) for v in va]
    checks.published(f"Aiming: slope of Exclusion hand angle on Inclusion aim report, {c}", slope(x, y), rep_slope, 0.0005,
                     "Table 1")

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Taylor (Princeton)",
    data_url="https://osf.io/zt8pd/",
    license="CC BY-NC 4.0",
    license_url="https://creativecommons.org/licenses/by-nc/4.0/",
    changes="converted to the OpenMotor format, participant codes replaced, and hand and aim angles flipped back from "
            "the authors' compensation-positive sign to counter-clockwise positive.",
    unit_note="trial of one participant sequence",
    how=[
        "One row is one reach: 400 per participant (40 baseline, 200 with a 45 deg rotation, four 30-trial process "
        "dissociation blocks of Exclusion, Top-up and Inclusion trials, 40 no-cursor washout trials); all 60 recruited "
        "participants are kept and the 2 the authors excluded are flagged.",
        "The authors stored hand angles and aim reports flipped so that compensation is positive; they were flipped back "
        "with the authors' own rule, so hand_angle and aim_report are counter-clockwise positive like target_angle and "
        "rotation (rotation -45 is clockwise).",
        "This sign was checked three ways: hand angles recomputed from the authors' hand x, y and target angle match on "
        "every row, participants with the -45 rotation reach about +43 deg late in training, and their aim taps lie left "
        "of the start position on the screen.",
        "hand_angle is the hand direction where the hand passed the 7 cm target distance (the paper's endpoint); block "
        "names the process dissociation condition (target at the training or the aiming location, no delay or a 1 min "
        "delay); the deposit's pilot data (a separate study) and hand trajectories are not included.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
