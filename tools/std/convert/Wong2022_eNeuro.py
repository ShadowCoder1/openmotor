"""Wong, Green & Isaacs (2022), eNeuro: "go before you know" reaching with two possible targets of unequal reward frequency (Experiment 1,
24 participants) or probability (Experiment 2, 17 participants) and value, one row per reach, in the 90-column reaching format.
Experiment 3 (discrete choice, 28 participants, gbyk_discretechoice_alldata_clean.mat) is converted too (added 11-09): one row per reach,
direct (to a near target) or intermediate (through the rectangle toward a far target), see the README.

Run from the site folder: python3 -I tools/std/convert/Wong2022_eNeuro.py
Raw files (untrusted data, read only) in raw_extra/Wong2022_eNeuro/, from https://osf.io/jn79v/:
  gbyk_rwdfreq_alldata_clean.mat (Experiment 1), gbyk_rwdProb_alldata_clean.mat (Experiment 2),
  gbyk_discretechoice_alldata_clean.mat (Experiment 3).
The authors' analysis scripts (gbykRwdFreqAllSubj_clean.m, gbykRwdProbAllSubj_clean.m, gbykDiscreteChoiceAllSubj_clean.m) were read as
text, never run.

Per participant the deposit has, for each condition cell, vectors of trials in time order within the cell (the order of the cells
within the session is not stored). InitDir is the direction of the hand velocity 100 ms after movement onset, "0 = midway between the two
targets", positive to the right: it equals 90 - atan2(vy, vx) of the authors' stored paths at sample 100 (verified on trials of
the stored trajectories). hand_angle here is the direction relative to the target in the OpenMotor frame:
hand_angle = (90 - InitDir) - TargetDir, with TargetDir = 67.509 (right target) or 112.491 deg (left target), counter-clockwise positive.
"""
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wong2022_eNeuro"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
TARGET_RADIUS_DEG = math.degrees(math.atan2(1.0, 20.0))  # 2 cm diameter targets at 20 cm


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


def isnum(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and x == x


rows, details = [], []
MEDIAN = {}
exp_info = {}
removed = {}


def build(file, exp_label, ratio_name, ratio_field):
    m = sio.loadmat(RAW / file, struct_as_record=False, squeeze_me=True)
    Rc = m["Reach"]
    n_sub = int(np.asarray(Rc[0, 0].InitDir).shape[1])
    prev = {}
    for si in range(n_sub):
        sid = f"{exp_label}_{si}"
        removed[sid] = [0, 0]
        j = 0
        for a in range(Rc.shape[0]):
            for b in range(Rc.shape[1]):
                r = Rc[a, b]
                ratio = getattr(r, ratio_field)
                if np.size(ratio) == 0:
                    continue
                F, V = float(ratio), float(r.ValRatio)
                succ = np.asarray(r.Succes, float)[:, si]
                idir = np.asarray(r.InitDir, float)[:, si]
                keep = np.where(~np.isnan(succ) | ~np.isnan(idir))[0]
                valid = ~np.isnan(idir)
                kvalid = -1
                layout = "A" if (ratio_field == "FreqRatio" or a < 3) else "B"
                block = (f"freq{F:g}_val{V:g}" if ratio_field == "FreqRatio" else f"prob{round(F, 2):g}_val{V:g}_layout{layout}")
                for t in keep:
                    ok = bool(valid[t])
                    if ok:
                        kvalid += 1
                    tgt = float(np.asarray(r.TargetDir, float)[t, si])
                    ha = (90.0 - idir[t]) - tgt if ok else float("nan")
                    row = dict.fromkeys(names, "")
                    row.update(
                        dataset=ID, experiment=f"{ID}_{exp_label}", study=ID, subject=sid, person=sid, id_type="lab",
                        person_link="assumed_unique", prior_exposure="unknown", subject_excluded_by_authors="False",
                        handedness="right", device="kinarm", setting="lab", population="healthy",
                        trial=str(j), trial_orig=str(t + 1), session="0", block=block, phase="other", trial_kind="reach",
                        break_before="True" if j == 0 else "False",
                        target_angle=num(tgt) if ok else "", target_id=("0" if abs(tgt - 67.509) < 0.01 else "1") if ok else "",
                        target_size_deg=num(TARGET_RADIUS_DEG), target_distance_norm="1", target_visible_during_move="True",
                        target_event="none", target_jump="0", probe_kind="none", instr_goal="hand_to_target",
                        instr_strategy="unspecified", cue_elements=f"probability_cue:{round(F, 2):g}" if ratio_field == "ProbRatio" else "",
                        hand_used="right", hand_angle=num(ha), hand_measure="initial_direction",
                        rt_ms=num(np.asarray(r.Latency, float)[t, si]) if ok else "",
                        excluded_by_authors="False" if ok else "True",
                        fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical",
                        fb_endpoint_transform_only="False", rotation="0", fb_delay_ms="0",
                    )
                    rows.append(row)
                    removed[sid][0] += 0 if ok else 1
                    removed[sid][1] += 1
                    tcc = lambda arr: (float(np.asarray(arr, float)[kvalid, si]) if ok and kvalid < np.asarray(arr).shape[0] else float("nan"))  # noqa: E731
                    details.append([sid, exp_label, j, a, b, F, V, t + 1, "" if not ok else idir[t],
                                    np.asarray(r.InitDirErr, float)[t, si], np.asarray(r.EndptErr, float)[t, si], succ[t],
                                    np.asarray(r.VPeak, float)[t, si], tcc(r.TimeChoose), tcc(r.TCclass), tcc(r.TCclassdir)])
                    j += 1
        prev[sid] = j
    exp_info[exp_label] = (n_sub, prev)
    return m


m1 = build("gbyk_rwdfreq_alldata_clean.mat", "RwdFreq", "FreqRatio", "FreqRatio")
m2 = build("gbyk_rwdProb_alldata_clean.mat", "RwdProb", "ProbRatio", "ProbRatio")

# ---------------------------------------------------------------- checks
S1 = "Methods, Participants, Experiment 1 (24 recruited)"
checks.published("Experiment 1, participants", exp_info["RwdFreq"][0], 24, 0, S1)
checks.published("Experiment 2, participants", exp_info["RwdProb"][0], 17, 0, "Methods, Participants, Experiment 2 (17 recruited)")
checks.internal("every participant has rows; Experiment 1 has 24 sequences, Experiment 2 has 17",
                len(exp_info["RwdFreq"][1]) == 24 and len(exp_info["RwdProb"][1]) == 17 and min(
                    min(exp_info["RwdFreq"][1].values()), min(exp_info["RwdProb"][1].values())) > 200)
for lab, rep, sec in (("RwdFreq", 6.8, "Results, Experiment 1 (median 6.8 % of reaches removed)"),
                      ("RwdProb", 11.4, "Results, Experiment 2 (median 11.4 % of reaches removed)")):
    pct = [100.0 * v[0] / v[1] for k, v in removed.items() if k.startswith(lab)]
    MEDIAN[lab] = float(np.median(pct))
hand = np.array([float(r["hand_angle"]) for r in rows if r["hand_angle"] != ""])
checks.internal("hand_angle within (-180, 180] and every row without hand_angle is an authors' exclusion",
                bool(np.all(np.abs(hand) < 180)) and all(r["excluded_by_authors"] == "True" for r in rows if r["hand_angle"] == ""))
TA = sorted({round(float(r["target_angle"]), 2) for r in rows if r["target_angle"]})
checks.internal("target angles are only 67.51 and 112.49 deg (two targets 45 deg apart, midline at 90)", TA == [67.51, 112.49], str(TA))
checks.internal("authors' latencies are 150 to 600 ms on kept rows (their script removes others)",
                all(150 <= float(r["rt_ms"]) <= 600 for r in rows if r["rt_ms"]))


import collections  # noqa: E402
cell_n = collections.Counter((r["experiment"], r["subject"], r["block"]) for r in rows)
f_max = {}
p_max = {}
for (e, sj, bl), n in cell_n.items():
    (f_max if e.endswith("RwdFreq") else p_max)[bl] = max((f_max if e.endswith("RwdFreq") else p_max).get(bl, 0), n)
checks.published("Experiment 1, largest block (trials in one frequency x value cell)", max(f_max.values()), 44, 0,
                 "Methods, Experiment 1 ('Blocks had 30-44 trials')")
checks.published("Experiment 1, smallest of the blocks' largest sizes (the 1:1 and 2:1 frequency blocks have 30 trials)",
                 min(f_max.values()), 30, 0, "Methods, Experiment 1 ('Blocks had 30-44 trials')")
checks.published("Experiment 2, trials in one probability-ratio cell (70:30 or 80:20) per participant and reward ratio, largest",
                 max(v for k, v in p_max.items() if not k.startswith("prob1_") and "_val1_" not in k), 20, 0,
                 "Methods, Experiment 2 (20 trials per probability ratio)")

# ---------------------------------------------------------------- Experiment 3 (discrete choice)
# reaches[a, b, c]: a = 1 direct / 2 intermediate, b = frequency ratio (1, 2, 4), c = reward ratio (1, 2, 4) of the two sides (the authors'
# script comments and loops). Per cell the deposit has arrays of 150 x 28 (trials x participants), valid trials first, NaN elsewhere.
# InitDir is the direction of the hand velocity 100 ms after movement onset in degrees from the midline, in the authors' aligned frame
# (the deposit is aligned so that positive = toward the more rewarded side, negative = toward the more frequent side; see the README).
m3 = sio.loadmat(RAW / "gbyk_discretechoice_alldata_clean.mat", struct_as_record=False, squeeze_me=True)
R3 = m3["reaches"]
n3 = int(m3["nsubj"])
TP = {k: np.asarray(m3[k], float) for k in ("t1pos", "t2pos", "t3pos", "t4pos")}
ang = lambda p: math.degrees(math.atan2(p[1], p[0]))  # noqa: E731
TARGET_ANGLE = {(0, 1): ang(TP["t1pos"]), (0, 2): ang(TP["t2pos"]), (1, 1): ang(TP["t3pos"]), (1, 2): ang(TP["t4pos"])}  # (type, correct side code)
DIST_CM = {0: 100 * math.hypot(*TP["t1pos"]), 1: 100 * math.hypot(*TP["t3pos"])}
RATIO = {0: 1, 1: 2, 2: 4}
KIND = {0: "direct", 1: "intermediate"}
flag_ok = np.asarray(m3["goodsub_weightfits"], int)
rows3, details3, gaps3 = [], [], {}
e3 = {}
for si in range(n3):
    sid = f"DiscChoice_{si}"
    j = 0
    e3[sid] = {"direct": 0, "intermediate": 0}
    for b in range(3):
        for c in range(3):
            for a in range(2):
                r = R3[a, b, c]
                cols = {k: np.asarray(getattr(r, k), float)[:, si] for k in ("InitDir", "corr", "corrRL", "tgtChoice", "RT", "rowRatio", "BoxDir")}
                keep = np.where(~np.isnan(cols["InitDir"]) | ~np.isnan(cols["corrRL"]) | ~np.isnan(cols["RT"]) | ~np.isnan(cols["corr"]))[0]
                last = keep[-1] + 1 if len(keep) else 0
                gaps3.setdefault(sid, [0, 0])
                gaps3[sid][0] += last - len(keep)   # blank positions inside the filled part of the array
                gaps3[sid][1] += last
                block = f"freq{RATIO[b]:g}_val{RATIO[c]:g}_{KIND[a]}"
                for t in keep:
                    idir = cols["InitDir"][t]
                    ok = not np.isnan(idir)
                    cs = int(cols["corrRL"][t])
                    tgt = TARGET_ANGLE[(a, cs)]
                    ha = (90.0 - idir) - tgt if ok else float("nan")
                    row = dict.fromkeys(names, "")
                    row.update(
                        dataset=ID, experiment=f"{ID}_DiscChoice", study=ID, subject=sid, person=sid, id_type="lab",
                        person_link="assumed_unique", prior_exposure="unknown", subject_excluded_by_authors="False",
                        handedness="right", device="kinarm", setting="lab", population="healthy",
                        trial=str(j), trial_orig=str(t + 1), session="0", block=block, phase="other", trial_kind="reach",
                        break_before="True" if j == 0 else "False",
                        target_angle=num(tgt), target_id=str((0 if a == 0 else 2) + (0 if cs == 1 else 1)),
                        target_distance_norm=num(DIST_CM[a] / DIST_CM[1]), target_visible_during_move="True",
                        target_event="none", target_jump="0", probe_kind="none", instr_goal="hand_to_target",
                        instr_strategy="unspecified", cue_elements=f"reach_type:{KIND[a]}",
                        hand_used="right", hand_angle=num(ha), hand_measure="initial_direction",
                        rt_ms=num(1000.0 * cols["RT"][t]) if not np.isnan(cols["RT"][t]) else "",
                        excluded_by_authors="False" if ok else "True",
                        fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical",
                        fb_endpoint_transform_only="False", rotation="0", fb_delay_ms="0",
                    )
                    rows3.append(row)
                    e3[sid][KIND[a]] += 1 if ok else 0
                    details3.append([sid, j, KIND[a], RATIO[b], RATIO[c], t + 1, round(float(cols["rowRatio"][t]), 6),
                                     cs, cols["corr"][t], cols["tgtChoice"][t], "" if not ok else idir, cols["BoxDir"][t],
                                     int(flag_ok[si])])
                    j += 1
rows.extend(rows3)

S3 = "Methods, Participants (28 participants in Experiment 3)"
checks.published("Experiment 3, participants", len(e3), 28, 0, S3)
checks.published("Experiment 3, participants who reliably made both direct and intermediate reaches in every condition (authors' flag)",
                 sum(int(flag_ok[i]) for i in range(n3)), 19, 0,
                 "Methods, Participants ('Of these 28 individuals in experiment 3, 19 participants reliably generated both direct and intermediate reaches')")
checks.published("Experiment 3, near targets are 10 cm from the start (cm)", round(DIST_CM[0], 1), 10.0, 0.05, "Methods, Experiment 3 (near pair 10 cm from the start)")
checks.published("Experiment 3, far targets are 15 cm from the start (cm)", round(DIST_CM[1], 1), 15.0, 0.05, "Methods, Experiment 3 (far pair 15 cm from the start)")
sep = lambda a, b: round(abs(ang(TP[a]) - ang(TP[b])), 1)  # noqa: E731
checks.published("Experiment 3, angle between the two near targets (deg)", sep("t1pos", "t2pos"), 90.0, 0.05, "Methods, Experiment 3 (near pair 90 deg apart)")
checks.published("Experiment 3, angle between the two far targets (deg)", sep("t3pos", "t4pos"), 45.0, 0.05, "Methods, Experiment 3 (far pair 45 deg apart)")
rr = sorted({d[6] for d in details3})
checks.published("Experiment 3, largest near-to-far reward ratio in a trial", max(rr), 4.0, 0, "Methods, Experiment 3 (near targets worth one to four times the far targets)")
checks.internal("Experiment 3: every participant has direct and intermediate reaches; rows only where the authors kept data",
                all(v["direct"] > 0 and v["intermediate"] > 0 for v in e3.values()) and min(v["direct"] + v["intermediate"] for v in e3.values()) > 100,
                str({k: e3[k] for k in list(e3)[:3]}))
h3 = np.array([float(r["hand_angle"]) for r in rows3 if r["hand_angle"] != ""])
checks.internal("Experiment 3: initial direction never beyond 75 deg from the midline (the authors' exclusion) and hand_angle within (-180, 180]",
                max(abs(d[10]) for d in details3 if d[10] != "") <= 75.0 and bool(np.all(np.abs(h3) < 180)))
checks.internal("Experiment 3: reaction times are positive and below 2 s (stored in seconds, converted to ms)",
                all(0 < float(r["rt_ms"]) < 2000 for r in rows3 if r["rt_ms"]))
checks.internal("Experiment 3: reward ratio cells are 1, 2, 4 and near-to-far ratios are 1, 2, 4", rr == [1.0, 2.0, 4.0], str(rr))
G3 = float(np.median([100.0 * v[0] / v[1] for v in gaps3.values() if v[1]]))

# ---------------------------------------------------------------- supplementary tables
b = io.StringIO()
w = csv.writer(b, lineterminator="\r\n")
w.writerow(["subject", "experiment", "trial", "cell_row", "cell_col", "frequency_or_probability_ratio", "value_ratio", "trial_in_cell",
            "init_dir_midline_deg_right_positive", "init_dir_err_as_stored", "endpoint_err_as_stored", "success_as_stored", "peak_speed_m_s",
            "time_choose_ms", "class_1_direct_2_intermediate", "class_dir_target_1_left_2_right"])
code = {}
for r in rows:
    code.setdefault(r["subject"], f"{ID}_S{len(code) + 1:04d}")
for d in details:
    d[0] = code[d[0]]
    w.writerow([d[0], d[1]] + [num(v) if not isinstance(v, str) else v for v in d[2:]])

b3 = io.StringIO()
w3 = csv.writer(b3, lineterminator="\r\n")
w3.writerow(["subject", "trial", "reach_type", "frequency_ratio", "reward_ratio_of_the_two_sides", "trial_in_array", "near_to_far_reward_ratio",
             "correct_side_code_1_more_rewarded_2_more_frequent", "chose_correct_side_as_stored", "target_choice_code_as_stored",
             "init_dir_aligned_deg_positive_toward_more_rewarded", "box_cross_dir_deg_as_stored", "authors_flag_weights_estimable"])
for d in details3:
    d[0] = code[d[0]]
    w3.writerow([d[0]] + [num(v) if not isinstance(v, str) else v for v in d[1:]])

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Wong (Moss Rehabilitation Research Institute)",
    data_url="https://osf.io/jn79v/",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format, participant numbers replaced by codes, initial direction converted from the authors' "
            "midline frame (right positive) to counter-clockwise degrees relative to the target; Experiment 3 (discrete choice) added with its own details file.",
    unit_note="reach of one participant sequence (Experiments 1 and 2: one reach to one of two targets; Experiment 3: one direct or intermediate reach)",
    how=[
        "One row is one reach with the right arm in a KINARM exoskeleton: two possible targets (2 cm, 45 deg apart, 20 cm from the start) "
        "were shown, and the correct one was revealed after movement onset. Experiment 1 (RwdFreq) varied how often each target was correct "
        "(frequency ratio 1, 2, 5, 10) and its reward value (1, 2, 5, 10); Experiment 2 (RwdProb) showed the probability on every trial "
        "(ratio 1, 2.33, 4) with value ratios 1, 2, 5. block names the authors' condition cell.",
        "hand_angle is the initial reach direction (hand velocity 100 ms after movement onset) relative to the target, in degrees, "
        "counter-clockwise positive. The authors store it as degrees from the midline between the two targets, positive to the right "
        "(toward the more rewarded target); that value is in reach_details.csv. Target angles are 67.509 (right) and 112.491 deg (left) in "
        "the screen frame (90 = straight ahead), and the sign flip was verified on the stored hand paths.",
        "Reaches that the authors' script removed (empty initial direction, target and latency) are kept as rows with excluded_by_authors "
        "True. The deposit holds the trials of each condition cell in time order within the cell, but not the order of the cells within a "
        "session, so trial counts rows cell by cell in the deposit's order and trial_orig is the position within the cell. In Experiment 2 "
        "the deposit has two sets of cells (layout A and B) for probability ratios 2.33 and 4; in layout B the typical initial direction has "
        "the opposite sign, which suggests the more probable target was on the other side, but the deposit does not say.",
        "Experiment 3 (DiscChoice, 28 participants, KINARM, right arm) shows five targets at once: a near pair 90 deg apart at 10 cm, a "
        "rectangle between them, and a far pair 45 deg apart at 15 cm. Reaching straight to a near target is a direct reach; passing through "
        "the rectangle toward a far target is an intermediate reach, and the deposit stores the two kinds in separate arrays, so block "
        "ends with _direct or _intermediate (also in cue_elements). block starts with the frequency ratio (1, 2, 4) and the reward ratio (1, "
        "2, 4) of the two sides; the near-to-far reward ratio (1, 2, 4) of each trial is in reach_details_exp3.csv. The two targets on one "
        "side are yoked: the correct side is stored per trial (correct_side_code) and target_angle is the target of the reach type on that "
        "side (45 or 135 deg for near, 67.5 or 112.5 deg for far). Latency in the deposit is in seconds and is converted to ms in rt_ms.",
        "Experiment 3 frame: the deposit is aligned so that, in cells with different reward ratios, reaches toward the more rewarded side are "
        "positive and in cells with different frequency ratios reaches toward the more frequent side are negative (this follows the paper's "
        "Figure 5 and 6 convention and the counts of the stored correct side and direction). It is therefore not the physical screen frame: "
        "side code 1 (positive, screen 'right') is the more rewarded side and code 2 (negative, 'left') the more frequent side, "
        "while the actual left or right of the screen varied between blocks and is not stored. hand_angle uses the same aligned frame "
        "(hand_angle = (90 - InitDir) - target_angle), so its sign is relative to that alignment. The deposit also stores tgtChoice, "
        "the authors' target index (1 near right, 2 near left, 3 far right, 4 far left in their script); it is not the chosen side (it does "
        "not follow the sign of the initial direction) and its meaning is not documented, so it is given as stored in reach_details_exp3.csv.",
        "Experiment 3 trials: each cell holds the reaches in time order within the reach type; the order of cells and the interleaving of "
        "direct and intermediate reaches within a session are not stored, so trial counts the rows participant by participant, cell by "
        "cell, direct before intermediate. Reaches the authors excluded are absent (the arrays keep only their data); 28 rows have a "
        "correct-side code and reaction time but no direction and are marked excluded_by_authors. One participant did not finish the "
        "last block (authors' script comment). Reach_details_exp3.csv also gives the authors' stored direction in degrees from the midline, the "
        "stored flag for participants whose weights could be estimated (19, as in the paper), and BoxDir, stored for intermediate reaches only and "
        "not defined in the deposit.",
    ],
    differences=[
        "The paper removes reaches whose initial direction is more than 45 deg from the midline (Experiments 1 and 2) or that were not "
        "completed in time; the authors' script removes |initial direction| > 75 deg, latency outside 150 to 600 ms and peak speed below "
        "0.3 m/s. The deposit already has those reaches blanked, so the excluded rows carry no values. The median share of reaches blanked "
        f"per participant is {MEDIAN['RwdFreq']:.1f} % in Experiment 1 (paper: 6.8 %) and {MEDIAN['RwdProb']:.1f} % in Experiment 2 (paper: "
        "11.4 %); the Experiment 2 deposit has fewer blanked reaches than the paper's criteria remove, so the paper's removals were "
        "probably applied partly in the authors' analysis, not in the deposit.",
        "Experiment 3: the paper reports that a median of 10.5 % of reaches were removed per participant (initial direction beyond 75 deg, "
        "movement not completed in 2500 ms, or incomplete trial). The excluded reaches are not in the deposit, which keeps no row for them, "
        f"so this cannot be reproduced from the rows; the share of blank positions inside the filled part of the arrays is {G3:.1f} % (median over "
        "participants), which is not the same quantity. The paper's median of 6.8 % and 11.4 % in Experiments 1 and 2 are discussed above.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks, extra_files=[("reach_details.csv", b.getvalue()), ("reach_details_exp3.csv", b3.getvalue())])
