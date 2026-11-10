"""Binetti, Mancinelli, Zanon & Bueti (2026) iScience: duration judgements while preparing finger reaches (40 adults, two
tasks of 20), trial-level behaviour from the authors' OSF project 4nz2r (TimeInAction).

Raw files: raw_extra/Binetti2026_iScience/trialData.csv (24,000 rows = 40 participants x 600 trials; read with pandas) and
MTA3-logic.Rmd (the authors' R notebook, read as text only, never run) for the filter the authors apply and the meaning of
several columns. The EEG archive (13 GB) is not used. The deposit's 'Sub' column holds a participant label with initials and
is not carried over.

Run from the site folder: python3 -I tools/std/convert/Binetti2026_iScience.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Binetti2026_iScience"
MODULE = "time_perception_reach"
DEVICE = ("right index finger moving on a Plexiglass surface under a 75 Hz monitor, tracked at 75 Hz by a Leap Motion infrared "
          "camera; left/right button presses of the other hand for the duration judgement")
COND = {"Exp": "cued", "MotorContr": "non-cued", "SensoryContr": "sensory control (no reach)"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    t = pd.read_csv(stdlib.raw_dir(DS) / "trialData.csv", low_memory=False)
    ck = stdlib.Checks()
    n = len(t)
    reach = t.Cond != "SensoryContr"
    # the authors' notebook (MTA3-logic.Rmd) keeps rows with GoalDist < 200 (NA read as -1) and 11 < actStart < 75
    gd = t.GoalDist.fillna(-1)
    keep = (gd < 200) & (t.actStart > 11) & (t.actStart < 75)
    excl = reach & ~keep
    own_map = np.where(t.Task == "Direction", t.DirMap, t.DistMap)
    incong = own_map == "incongruent"
    ck.internal("cursor gain differs from 1 only on incongruent rows of the Distance task",
                bool(((t.Gain != 1) <= ((t.Task == "Distance") & incong)).all()))
    ck.internal("each participant belongs to one task and has 600 rows", t.groupby("SubID").Task.nunique().max() == 1
                and (t.groupby("SubID").size() == 600).all())
    ck.internal("sensory-control rows hold no reach (action latency -1, no hand position)",
                bool(((t.actStart[~reach] == -1) & t.handStopX[~reach].isna()).all()))
    ck.internal("share of 'longer' judgements rises with test duration", bool(t.groupby("probeDur").timeResp.mean().is_monotonic_increasing))

    def r(col, dtype=float):
        v = num(t[col]).astype(float)
        return v.where(reach)

    df = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment=t.Task, group="", subject=[f"s{s}" for s in t.SubID], subject_excluded_by_authors=False,
        age=np.nan, sex="", handedness="right", population="healthy", setting="lab", device=DEVICE, session=np.nan,
        block=[f"block {b}" for b in t.Block], trial=t.groupby("SubID").cumcount().to_numpy() + 1, phase="test", time_s=np.nan,
        task="reaching", unit="trial", condition=t.Cond.map(COND),
        pert_type=np.where(incong, "incongruent_cursor_mapping", "none"), pert_value=np.where(incong, np.nan, 0.0),
        pert_unit="", feedback=np.where(reach, np.where(incong, "altered", "normal"), "none"),
        excluded_by_authors=excl.to_numpy(),
        reach_task=t.Task, mapping=own_map, target_direction=t.Dir, target_distance=t.Dist, cursor_gain=t.Gain, run=t.Run,
        trial_in_run=t.Trial, probe_onset_s=t.probeOnset, probe_duration_s=t.probeDur, judged_longer=t.timeResp.astype(bool),
        response_time_s=t.RT, early_start=t.EarlyStart.astype(bool), late_start=t.LateStart.astype(bool),
        probe_error_samples=t.probeErrorSamp, score=t.score, action_start_frames=t.actStart.where(reach),
        action_stop_frames=t.actStop.where(reach), action_duration_frames=t.actDur.where(reach), peak_velocity=r("peakVel"),
        mean_velocity=r("meanVel"), sse=t.SSE, mse=t.MSE, trajectory_inefficiency=t.trajIneff, hand_error=t.handErr,
        hand_stop_x=t.handStopX, hand_stop_y=t.handStopY, goal_x=t.GoalX, goal_y=t.GoalY, goal_distance_px=t.GoalDist,
        outlier_action=t.outlierAction.map({"True": True, "False": False, True: True, False: False}),
    ))
    ck.internal("every row of the authors' trialData.csv is kept, in order (24,000)", len(df) == n == 24000)
    ck.internal("the authors' filter keeps 19,327 reach rows (15,389 cued, 3,938 non-cued)",
                int((reach & keep).sum()) == 19327)

    # published numbers: Methods, Participants / Tasks and conditions / Trials and timing
    per_task = t.groupby("Task").SubID.nunique()
    ck.published("participants", t.SubID.nunique(), 40, 0, "STAR Methods, Participants")
    ck.published("participants in the Directional task", per_task["Direction"], 20, 0, "STAR Methods, Participants")
    ck.published("participants in the Distance task", per_task["Distance"], 20, 0, "STAR Methods, Participants")
    c = t.groupby(["SubID", "Cond"]).size().unstack()
    ck.published("cued trials per participant (20 blocks of 20)", c["Exp"].mean(), 400, 0, "STAR Methods, Trials and timing")
    ck.published("non-cued trials per participant (5 blocks of 20)", c["MotorContr"].mean(), 100, 0, "STAR Methods, Trials and timing")
    ck.published("sensory-control trials per participant (5 blocks of 20)", c["SensoryContr"].mean(), 100, 0, "STAR Methods, Trials and timing")
    d = sorted(t.probeDur.unique())
    ck.published("number of test durations (5 linearly spaced values)", len(d), 5, 0, "STAR Methods, Duration discrimination task")
    ck.published("shortest test duration, s", d[0], 0.44, 1e-9, "STAR Methods, Duration discrimination task")
    ck.published("longest test duration, s", d[-1], 1.24, 1e-9, "STAR Methods, Duration discrimination task")
    ck.published("target distance in the Directional task, px (205)", t.GoalX[t.Task == "Direction"].abs().mean(), 205, 0.5, "STAR Methods, Tasks")
    ck.published("far target distance in the Distance task, px (410)", t.GoalX[(t.Task == "Distance") & (t.Dist == "far")].mean(), 410, 0.5, "STAR Methods, Tasks")

    meta = dict(
        study=DS,
        paper=dict(authors="Binetti N, Mancinelli F, Zanon M, Bueti D", year=2026,
                   title="Perceptual lengthening of visual time throughout learning of hand-reaching actions",
                   venue="iScience 29(11):117993", doi="10.1016/j.isci.2026.117993"),
        lab="Bueti (SISSA, Trieste)",
        data_url="https://osf.io/4nz2r/",
        license="none stated",
        license_url="",
        changes=("Took the 24,000 rows of trialData.csv in the authors' order, dropped the participant label column that holds "
                 "initials, converted the velocity columns from text to numbers, left the sensory-control rows' missing reach "
                 "values empty (the deposit holds -1 for the latency), marked the rows the authors' R notebook filters out, and "
                 "gave participants codes."),
        how=["One row is one trial of a duration-discrimination task: a test stimulus lasting 0.44-1.24 s was compared with a 0.84 s "
             "standard (probe_duration_s, judged_longer) while the participant prepared or made a right-index-finger reach "
             "(cued: target known during the foreperiod; non-cued: target shown at the go beep) or, in the sensory control, "
             "made no reach. 20 participants did the Directional task (left or right target, cursor mapping congruent or "
             "reversed), 20 the Distance task (near or far target, mapping congruent or changed gain).",
             "Reach measures are the authors' own: latency, duration and stop in frames of the 75 Hz hand tracker, final "
             "finger position and its distance from the target centre in screen pixels, velocities in the authors' units. "
             "judged_longer is coded 1 for 'longer' by inference (the share of 1 rises with test duration); cursor_gain, "
             "hand_error, sse, mse, trajectory_inefficiency and response_time_s are the authors' columns, which neither the paper nor "
             "the code define. The data are not in the reach90 format because they hold no hand angle relative to a target "
             "and are about duration judgements.",
             "excluded_by_authors marks reach rows that the authors' notebook (MTA3-logic.Rmd) removes: reaching error of 200 px or more "
             "or an action latency of 11 frames or fewer or 75 frames or more (the 673 reach rows that fail it; "
             "the sensory-control rows are never marked)."],
        unit_note="trial (one duration judgement with its reach)",
        differences=["The paper describes cued blocks as 20 blocks of 20 trials and non-cued and sensory blocks as 5 blocks of 20; "
                     "the deposit's Block column is the authors' own number (up to 20 per participant, shared across the three "
                     "conditions), so it is not a count of blocks within a condition.",
                     "The paper's PSE and JND statistics rest on psychometric fits that are not rebuilt here."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for k in ck.items:
        if k["kind"] == "published":
            print("published", k["what"], round(k["computed"], 4), k["reported"], k["ok"])
        elif not k["ok"]:
            print("INTERNAL FAIL", k["what"])


if __name__ == "__main__":
    main()
