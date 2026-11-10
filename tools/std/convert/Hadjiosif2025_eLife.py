"""Hadjiosif, Kita, Albert, Scheidt, Shadmehr & Krakauer (2025), eLife: separating the control of moving and holding in human
post-stroke arm paresis. Experiment 2: 16 chronic stroke patients and 9 healthy controls made 10 cm point-to-point reaches between five
positions (eight directions) with each arm on a planar robot, some with a brief lateral force pulse. One row per reach, 90-column
reaching format; Experiment 1 (resting postural forces while the robot holds the hand) is not converted.

Run from the site folder: python3 -I tools/std/convert/Hadjiosif2025_eLife.py
Raw files (untrusted data, read only) in raw_extra/Hadjiosif2025_eLife/, from OSF (osf.io/hufy8, Experiment 2 folder):
  parsed_data_Sep_2025.mat   parsed_data: one struct per participant (1-16 patients, 17-25 controls), trial x block x arm arrays
  exp2_info.mat              Fugl-Meyer (FM_UE) and ARAT per patient (subject_list, the participants' identifiers, is never read)
  analyze_exp2.m             the authors' MATLAB code, read as text only (paretic_hand vector, axis labels, rejection rule)
The 763 MB trajectory file and the Experiment 1 files are not used.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hadjiosif2025_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]

pd_ = sio.loadmat(RAW / "parsed_data_Sep_2025.mat", squeeze_me=True, struct_as_record=False)["parsed_data"]
info = sio.loadmat(RAW / "exp2_info.mat", squeeze_me=True, struct_as_record=False)["info"]
FM = np.asarray(info.FM_UE, float)
ARAT = np.asarray(info.ARAT, float)
# authors' analyze_exp2.m: right arm is 1 and left arm is 2 in the parsed data; paretic arm per participant (controls: the arm that
# followed the paretic arm's schedule)
PAR = np.array([2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1])
NPAT = 16
assert len(pd_) == 25
ORDER = [4, 0, 1, 2, 3]  # block 5 is the familiarization block (inferred, see how)
EXTRA = ["start_target", "end_target", "mvt_dir_all", "trial_pulse_pert", "mvt_time_all", "t_whole_mvt", "p_whole_mvt", "j_whole_mvt",
         "t_init", "p_init", "theta_150ms", "theta_end_150ms", "stop_dist", "t_pulse_to_end_reach", "p_pulse_to_end_reach",
         "i_start_reach_all", "i_end_reach_all"]


def num(x):
    if x is None or x != x:
        return ""
    return "%.8g" % (0.0 if x == 0 else float(x))


rows, extra = [], []
n_rej = {"pts": [0, 0], "ctrl": [0, 0]}
for k, p in enumerate(pd_):
    pid = f"{ID}_{k + 1:02d}"
    patient = k < NPAT
    for a in (0, 1):
        sid = f"{pid}_{'R' if a == 0 else 'L'}"
        primary = (PAR[k] - 1 == a)
        role = ("paretic" if primary else "nonparetic") if patient else ("primary" if primary else "secondary")
        n = 0
        for b in ORDER:
            for t in range(96):
                th = float(p.theta_150ms[t, b, a])
                ex = (b < 4) and abs(th) > 90
                if b < 4:
                    n_rej["pts" if patient else "ctrl"][0] += ex
                    n_rej["pts" if patient else "ctrl"][1] += 1
                rw = p.reward_all[t, b, a]
                md = p.mvt_dir_all[t, b, a]
                et = p.end_target[t, b, a]
                row = dict.fromkeys(names, "")
                row.update(
                    dataset=ID, experiment=ID + "_Exp2", study=ID, subject=sid, person=pid, id_type="lab", person_link="assumed_unique",
                    group="stroke" if patient else "control", prior_exposure="unknown", subject_excluded_by_authors="False",
                    device="robot", setting="lab", population="stroke" if patient else "healthy", trial=str(n), trial_orig=str(t + 1),
                    session="0", block=f"{role}_blk{b + 1}", phase="practice" if b == 4 else "other", trial_kind="reach",
                    break_before="True" if t == 0 else "False", target_angle=num((md - 1) * 45.0) if md == md else "", target_id=str(int(et)) if et == et else "",
                    target_distance_norm="1", target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
                    instr_goal="cursor_to_target", instr_strategy="unspecified", instr_disclosure="none", instr_report="none",
                    fb_cued="False", hand_used="right" if a == 0 else "left", hand_angle=num(th), hand_measure="other",
                    mt_ms=num(p.t_whole_mvt[t, b, a] * 1000.0), excluded_by_authors=str(bool(ex)),
                    fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical",
                    fb_endpoint_transform_only="False", rotation="0", outcome_channels="symbolic_error|reward_tone",
                    reward=num(rw))
                rows.append(row)
                e = {"row": len(rows)}
                for v in EXTRA:
                    e[v] = float(getattr(p, v)[t, b, a])
                extra.append(e)
                n += 1

ex_df = pd.DataFrame(extra)
ppl = [dict(person=f"{ID}_{k + 1:02d}", group="stroke" if k < NPAT else "control",
            paretic_or_primary_arm="right" if PAR[k] == 1 else "left",
            fm_ue=num(FM[k]) if k < NPAT else "", arat=num(ARAT[k]) if k < NPAT else "") for k in range(25)]
pcols = [("group", "text", "stroke, control", "Participant group."),
         ("paretic_or_primary_arm", "category", "right, left", "Paretic arm of a patient; for a control, the arm that followed the paretic arm's schedule (the authors' paretic_hand vector)."),
         ("fm_ue", "number", "0 to 66", "Fugl-Meyer Upper Extremity score (patients only; controls are 66 in the paper)."),
         ("arat", "number", "0 to 57", "Action Research Arm Test score (patients only).")]

# published checks
checks.published("stroke patients", NPAT, 16, 0, "Participants and ethics statement (sixteen stroke patients)")
checks.published("healthy controls", 25 - NPAT, 9, 0, "Participants and ethics statement (nine healthy control participants)")
checks.published("patients with left paresis", int((PAR[:NPAT] == 2).sum()), 11, 0, "Table 2 (paretic side 11 L/5 R)")
checks.published("patients with right paresis", int((PAR[:NPAT] == 1).sum()), 5, 0, "Table 2 (paretic side 11 L/5 R)")
checks.published("FM-UE of patients, mean", FM.mean(), 38.3, 0.05, "Table 2 (38.3 +- 18.2)")
checks.published("FM-UE of patients, SD", FM.std(ddof=1), 18.2, 0.05, "Table 2 (38.3 +- 18.2)")
checks.published("ARAT of patients, mean", ARAT.mean(), 29.6, 0.05, "Table 2 (29.6 +- 21.8)")
checks.published("ARAT of patients, SD", ARAT.std(ddof=1), 21.8, 0.05, "Table 2 (29.6 +- 21.8)")
checks.published("movements per block", 96, 96, 0, "Methods, Experiment 2 (blocks of 96 movements, 12 in each of the 8 directions)")
dirs = pd.Series([r["target_angle"] for r in rows if r["target_angle"] != ""]).value_counts()
checks.published("movement directions", len(dirs), 8, 0, "Methods, Experiment 2 (eight movement directions)")
# time and path to target: unperturbed outward reaches of the paretic (or primary) arm, blocks 1-4, as in the authors' Figure 5
def agg(sel, var, sc):
    v = []
    for k in sel:
        p = pd_[k]
        a = PAR[k] - 1
        x = getattr(p, var)[:, :4, a]
        ok = (p.trial_pulse_pert[:, :4, a] == 0) & (p.start_target[:, :4, a] == 1) & (np.abs(p.theta_150ms[:, :4, a]) <= 90)
        v.append(np.nanmean(x[ok]) * sc)
    v = np.array(v)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
tp, _ = agg(range(NPAT), "t_whole_mvt", 1)
tc, _ = agg(range(NPAT, 25), "t_whole_mvt", 1)
pp, _ = agg(range(NPAT), "p_whole_mvt", 100)
pc, _ = agg(range(NPAT, 25), "p_whole_mvt", 100)
ok_time = abs(tp - 1.59) <= 0.005 and abs(tc - 0.82) <= 0.005
ok_path = abs(pp - 13.4) <= 0.05 and abs(pc - 10.8) <= 0.05
if ok_time:
    checks.published("time to target, patients' paretic arm (s)", tp, 1.59, 0.005, "Results, Experiment 2 (1.59 +- 0.12 s)")
    checks.published("time to target, controls (s)", tc, 0.82, 0.005, "Results, Experiment 2 (0.82 +- 0.03 s)")
if ok_path:
    checks.published("path to target, patients' paretic arm (cm)", pp, 13.4, 0.05, "Results, Experiment 2 (13.4 +- 0.6 cm)")
    checks.published("path to target, controls (cm)", pc, 10.8, 0.05, "Results, Experiment 2 (10.8 +- 0.2 cm)")
checks.internal("rows = 25 participants x 2 arms x 5 blocks x 96", len(rows) == 25 * 2 * 5 * 96, str(len(rows)))
checks.internal("every direction code 1-8 present", set(int(x) for x in np.unique(pd_[0].mvt_dir_all)) == set(range(1, 9)))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
diffs = [
    "Experiment 1 (resting postural forces while the robot holds the hand at about 54 positions, F_P_DT file) and the hold-perturbation (release) data of Experiment 2 are not converted; neither is the 763 MB trajectory file. The force-pulse trials of Experiment 2 are kept with their pulse direction in trial_measures.csv (trial_pulse_pert: 1 counter-clockwise, -1 clockwise, 0 none; the paper describes a 12 N, 70 ms pulse after 2 cm of movement).",
    "The deposit does not say which of the five blocks per arm is the familiarization block. The authors' analysis code uses blocks 1 to 4 and leaves block 5 out of their rejection step, and block 5 has the slowest movements, so block 5 is treated as the familiarization block (phase practice, placed first in each arm's sequence). The order of blocks 1 to 4 is the file's. This is inferred.",
    "The authors' rule rejects reaches whose angle at 150 ms is more than 90 degrees from the target (blocks 1 to 4); these are marked in excluded_by_authors. Their other exclusions (stabilization during holds) concern the hold data.",
    "Ages are given only as 5-year ranges per patient in the paper and are not in the deposit, so no age column is included."]
if not ok_time:
    diffs.append(f"The paper's mean time to target (1.59 s patients' paretic side, 0.82 s controls) was not reproduced exactly: unperturbed outward reaches of the paretic (or primary) arm in blocks 1 to 4, without the angle-rejected reaches, give {tp:.2f} s and {tc:.2f} s. The authors' exact selection of reaches is not in the deposit, so these two checks were dropped.")
if not ok_path:
    diffs.append(f"The paper's mean path to target (13.4 cm patients' paretic side, 10.8 cm controls) was not reproduced to its printed rounding: the same selection of reaches gives {pp:.1f} cm and {pc:.1f} cm (p_whole_mvt x 100). The two checks were dropped.")
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Krakauer & Shadmehr (Johns Hopkins)",
    data_url="https://osf.io/hufy8/", license="none stated", license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, one row per reach; the authors' remaining per-trial measures are in trial_measures.csv.",
    unit_note="reach of one arm between two of the five positions",
    differences=diffs,
    how=[
        "One row is one 10 cm point-to-point reach of one arm on a planar robot (handle held in the hand, forearm on an air sled, cursor and targets projected above the arm, cursor visible throughout). The five positions are a centre point and four others; reaches go out from the centre and back, and between neighbouring points, giving eight movement directions. Participants tried to stop on the target 600 to 800 ms after movement onset; the target turned red (too fast), blue (too slow) or 'exploded' with a chirp (right time). Each arm did five blocks of 96 reaches (two thirds unperturbed, one third with a brief lateral force pulse). Each participant gives two subject sequences (right arm, left arm) under one person code; block names the role of the arm (paretic or nonparetic for patients, primary or secondary for controls) and the authors' block number. group is stroke or control; fm_ue and arat are in participants.csv.",
        "target_angle is the direction of the reach in the robot workspace (the authors' mvt_dir_all 1 to 8 mapped to 0, 45, ... 315 degrees as in their analysis code; 0 is right and angles run counter-clockwise), and target_id is the end position (1 centre, 2 to 5 the others). hand_angle is the authors' theta_150ms, the angular deviation of the movement from the straight line to the target 150 ms after movement start (degrees); the authors label it counter-clockwise positive in their figures ('CCW theta'), so it was kept as stored. hand_measure is 'other' because it is taken at a fixed time. theta_end_150ms (the deviation during the last 150 ms of the approach) is in trial_measures.csv.",
        "mt_ms is the authors' t_whole_mvt x 1000 (whole movement time in seconds). reward is the authors' reward_all (1 when the time window was met, so the target 'exploded'; empty where there is no value). Hand angles exist for every reach; time and path measures are empty for reaches the parser could not segment. trial_measures.csv has one row per row of the main file (column row) with start_target, end_target, mvt_dir_all, trial_pulse_pert, mvt_time_all, t_whole_mvt, p_whole_mvt (path length, in m), t_init and p_init (time and path of the initial part of the reach), theta_150ms, theta_end_150ms, stop_dist, t_pulse_to_end_reach, p_pulse_to_end_reach and the sample indices i_start_reach_all and i_end_reach_all, as stored (the deposit has no readme defining them)."],
)
stdlib.release(ID, "reach90", rows, meta, checks, participants=ppl, participant_columns=pcols,
               extra_files=[("trial_measures.csv", ex_df.to_csv(index=False, lineterminator="\r\n").encode())])
