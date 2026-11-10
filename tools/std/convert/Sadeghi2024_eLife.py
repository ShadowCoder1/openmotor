"""Sadeghi et al. (2024), eLife 12:RP88514: humans performing the critical stability task (CST), balancing a virtual unstable cursor
with the right hand, 18 participants in three instruction groups, three days. One row per trial (CST, reaction-time and tracking trials).

Run from the site folder: python3 -I tools/std/convert/Sadeghi2024_eLife.py
Raw file (untrusted data, read only): raw_extra/Sadeghi2024_eLife/eLife_code_and_data/data/cst_data.mat, one member pulled with HTTP range
requests out of eLife_code_and_data.rar (4 GB, non-solid) of Zenodo record 10972722 (Dryad doi 10.5061/dryad.p2ngf1vzt) and extracted
with the system tar; the README and the authors' MATLAB code (read as text only) are in the same archive. Strings in the MAT file are
MATLAB string objects, decoded by Sadeghi2024_eLife_mcos.py (a copy of the Liu2024 reader with string_object() added).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Sadeghi2024_eLife_mcos import MatObjects, string_object  # noqa: E402

ID = "Sadeghi2024_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
STUDY = "Sadeghi et al. 2024, eLife"
MAT = RAW / "eLife_code_and_data/data/cst_data.mat"
SDIR = RAW / "_samples"
SDIR.mkdir(exist_ok=True)
DEVICE = ("12-camera Qualisys motion capture (250 Hz) tracking a marker on the back of the right hand; the hand's mediolateral position drove an unstable "
          "first-order system whose output was shown as a cursor on a large screen 150 cm away")

o = MatObjects(MAT)
m = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
data = m["data"]
assert len(data) == 13575


def txt(v):
    if type(v).__name__ == "MatlabOpaque":
        return string_object(o, v)
    return str(v)


def num(v):
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0]) if a.size == 1 else np.nan


# instruction group of each participant: instruction_type is -1 on trials where it does not apply
inst_of = {}
for tr in data:
    inst_of.setdefault(txt(tr.subjectID), set()).add(txt(tr.instruction_type))
inst_of = {k: v - {"-1", "-1.0"} for k, v in inst_of.items()}
assert all(len(v) == 1 for v in inst_of.values()), inst_of.values()
rows, extra = [], []
pcode, tcount = {}, {}
dur_success, first_dyn_lambda, n_by = [], {}, {}
x_succ_max, x_fail_max = [], []
for k, tr in enumerate(data):
    sid = txt(tr.subjectID)
    if sid not in pcode:
        pcode[sid] = len(pcode) + 1
    pi = pcode[sid]
    tcount[sid] = tcount.get(sid, 0) + 1
    ttype, mode, tgt = txt(tr.trialType), txt(tr.lambdaUpdateMode), txt(tr.target)
    inst = next(iter(inst_of[sid]))
    lam = num(getattr(tr, "lambda"))
    ses = int(num(tr.sessionNumber))
    suc = num(tr.success)
    t = np.atleast_1d(np.asarray(tr.t, dtype=float))
    x = np.atleast_1d(np.asarray(tr.x, dtype=float))
    u = np.atleast_2d(np.asarray(tr.u, dtype=float))
    assert len(t) == len(x) == len(u)
    back = np.flatnonzero(np.diff(t) < -0.01)
    i0 = int(back[-1]) + 1 if len(back) else 0  # start of the trial proper, as the authors' discardPreTrial.m finds it
    code = f"{ID}_S{pi:04d}"
    fn = f"samples/{code}_trial{tcount[sid]:05d}.csv"
    df = pd.DataFrame({"time_s": t, "cursor_x": x, "hand_x": u[:, 0], "hand_y": u[:, 1], "hand_z": u[:, 2]})
    local = SDIR / f"{code}_trial{tcount[sid]:05d}.csv"
    df.to_csv(local, index=False, float_format="%.8g", lineterminator="\r\n")
    extra.append((fn, local))
    if ttype == "CST":
        if suc == 1:
            dur_success.append(t[-1] - t[i0])
            x_succ_max.append(np.max(np.abs(x[i0:])))
        elif suc == 0:
            x_fail_max.append(np.max(np.abs(x[i0:])))
        if mode == "dynamic" and ses == 1:
            key = pi
            if key not in first_dyn_lambda or tr.trialNumber < first_dyn_lambda[key][0]:
                first_dyn_lambda[key] = (int(tr.trialNumber), lam)
    analysed = (ttype == "CST" and mode == "ramp" and ses >= 1)
    info = [f"trial_type={ttype}", f"trial_number={int(num(tr.trialNumber))}"]
    if ttype == "CST":
        info += [f"lambda_update_mode={mode}", f"success={int(suc)}"]
    wt, rt = num(tr.waitTime), num(tr.reactionTime)
    if wt != -1 and not np.isnan(wt):
        info.append(f"wait_time_s={wt:.6g}")
    if rt != -1 and not np.isnan(rt):
        info.append(f"reaction_time_s={rt:.6g}")
    if tgt:
        info.append(f"target={tgt}")
    ev = f"trial_start={t[i0]:.4f}" if i0 else ""
    n_by[(pi, ses, ttype, mode)] = n_by.get((pi, ses, ttype, mode), 0) + 1
    cst = ttype == "CST"
    rows.append(dict(
        dataset=ID, study=STUDY, experiment="Critical stability task", group={"none": "no instruction", "position": "position control instruction", "velocity": "velocity control instruction"}[inst],
        subject=str(pi), person=str(pi), subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy",
        setting="lab", device=DEVICE, session=ses, block=f"{ttype} {mode}" if cst else ttype, trial=tcount[sid],
        phase="practice" if ses == 0 else "test", time_s="", task="tracking", unit="recording",
        condition=(f"CST {mode}, lambda {lam:.3g}" if cst else ttype),
        pert_type="task_instability" if cst else "none", pert_value=lam if cst else 0, pert_unit="1/s" if cst else "",
        feedback="normal", excluded_by_authors=not analysed,
        recording=tcount[sid], samples_file=fn,
        signal_kind="cursor position (1D) and hand position (3D)" if cst else "cursor position (1D) and hand position (3D), " + ttype.replace("_", " ") + " trial",
        channels="cursor_x;hand_x;hand_y;hand_z", channel_units="unit not documented;unit not documented;unit not documented;unit not documented",
        sample_rate_hz=250, n_samples=len(t), duration_s=len(t) / 250.0, events=ev,
        processing="none (as shared by the authors; samples before the last backward jump of time_s are the hand-centring phase before the trial starts)",
        trial_info=";".join(info)))
D = pd.DataFrame(rows)

# ---- checks
P = checks.published
g = {k: len([s for s, v in inst_of.items() if v == {k}]) for k in ["none", "position", "velocity"]}
P("participants", D.person.nunique(), 18, 0, "Methods, Participants and ethics statement (18 participants)")
P("participants in the position-control group", g["position"], 6, 0, "Results, Experiment 2 (n = 6 per group)")
P("participants in the velocity-control group", g["velocity"], 6, 0, "Results, Experiment 2 (n = 6 per group)")
P("participants in the no-instruction group", g["none"], 6, 0, "Methods (18 participants in three groups; 6 + 6 instructed)")
P("sampling rate (Hz)", 1 / np.median(np.concatenate([np.diff(np.asarray(d.t, dtype=float)) for d in data[:300]])), 250, 1, "Methods, Critical stability task (250 Hz)")
P("days (sessions) of the main experiment", D[D.session > 0].session.nunique(), 3, 0, "Methods, Experimental paradigm (three consecutive days)")
P("reaction-time trials per participant and day", D[(D.session > 0) & (D.block == "reaction_time")].groupby(["person", "session"]).size().mean(), 15, 0, "Methods, Experimental paradigm (15 reaction time trials)")
P("tracking trials per participant and day", D[(D.session > 0) & (D.block == "tracking")].groupby(["person", "session"]).size().mean(), 10, 0, "Methods (10 tracking trials)")
P("Block 1 CST trials per participant (day 1)", D[(D.block == "CST dynamic") & (D.session == 1)].groupby("person").size().mean(), 30, 0, "Methods (Block 1: 30 CST trials)")
P("Block 3 CST trials per participant and day (3 levels x 20)", D[(D.session > 0) & (D.block == "CST rand")].groupby(["person", "session"]).size().mean(), 60, 0, "Methods (Block 3: 20 trials at each of 3 difficulty levels)")
P("duration of a successful CST trial (s, median)", np.median(dur_success), 6, 0.01, "Methods (balance for 6 s for the trial to be successful)")
b2 = D[(D.session > 0) & (D.block == "CST ramp")].groupby(["person", "session"]).size()
checks.internal("Block 2 (ramp) has 120-200 trials per participant and day in most cases (paper: approximately 120-200)", 0.8 < ((b2 >= 100) & (b2 <= 220)).mean(), f"min {b2.min()}, median {b2.median()}, max {b2.max()}")
checks.internal("each participant has one instruction group (instruction_type is -1 on trials where it does not apply)", all(len(v) == 1 for v in inst_of.values()))
checks.internal("13,575 trials, 18 participants", len(D) == 13575 and D.person.nunique() == 18)
checks.internal("every trial has a samples file", len(extra) == len(D))
checks.internal("successful CST trials stay inside |cursor| < 100 and failed ones leave it (workspace +-100 in the cursor's unit)",
                np.max(x_succ_max) < 100 <= np.min(x_fail_max), f"success max {np.max(x_succ_max):.2f}, failure min {np.min(x_fail_max):.2f}")
fd = np.array([v[1] for v in first_dyn_lambda.values()])

meta = dict(
    paper=dict(authors="Sadeghi M, Sharif Razavian R, Bazzi S, Chowdhury RH, Batista AP, Loughlin PJ, Sternad D", year=2024,
               title="Inferring control objectives in a virtual balancing task in humans and monkeys", venue="eLife 12:RP88514", doi="10.7554/elife.88514"),
    lab="Sternad (Northeastern University)",
    data_url="https://doi.org/10.5061/dryad.p2ngf1vzt", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Each of the 13,575 trials of cst_data.mat (the human data; the monkey data of the paper are not in this deposit) became one samples CSV: time_s is the trial's t, "
             "then the cursor position x and the three hand-position columns u. The authors' participant initials were replaced by codes. "
             "Strings were decoded from MATLAB string objects. The fields lambdaWidth, CIV and feedback_type of the file are not described in the authors' README and are not included. "
             "Age and sex are not shared per participant (the paper gives 8 females of 18, ages 18 to 25, all right-handed). Only the group means in the paper are reproduced; "
             "no simulation or model code was used."),
    how=["One row is one trial of one participant. Most are critical stability task (CST) trials: the participant held the right hand above a table and moved it sideways to keep an unstable "
         "cursor (dx/dt = lambda x + u, with u the mediolateral hand position) inside the workspace for 6 s; a larger lambda is harder. The trial ends in success after 6 s or in failure when the cursor leaves "
         "the workspace. The file also holds the 15 reaction-time and 10 tracking trials the participants did each day, which the paper does not report. pert_value is lambda for CST trials.",
         "block is the trial type (CST with the way lambda was set: dynamic = up-down staircase in Block 1, ramp = stepwise increase in Block 2, rand = three fixed difficulty levels in Block 3; "
         "reaction_time; tracking). session 0 is the familiarisation (5 CST trials, 5 reaction-time trials and 3 tracking trials per person on average; phase practice) and sessions 1 to 3 are the three days. "
         "excluded_by_authors is true for rows the paper does not analyse: everything except CST ramp trials of days 1 to 3.",
         "Each samples file has the whole recorded trial. The time column restarts when the trial proper starts: the samples before the last backward jump of time_s are the phase in which the participant "
         "brought the hand cursor to the centre (the authors' discardPreTrial.m keeps only the samples after it); events gives trial_start, the time_s value at which the trial proper begins. "
         "Units are not documented in the deposit; they are probably millimetres (the workspace limit of the paper, +-10 cm, matches a cursor limit of 100 in the data: successful trials stay below 100 and failed ones reach it) - an inference.",
         "Rows are in the participant's order in the deposit; trial counts the participant's rows in that order, from 1. The group column is the instruction the participant received "
         "(none: do the task without failing; position: keep the cursor at the centre; velocity: keep the cursor still anywhere in the bounds)."],
    differences=["The paper says lambda started at 2.5 in the first Block 1 trial; the deposit's first day-1 dynamic CST trial of every participant (lowest trialNumber) has lambda = %.1f." % float(np.median(fd)),
                 "The paper says the three blocks were repeated on each day; the deposit has the Block 1 (dynamic) staircase trials on day 1 only (30 trials per participant) and Blocks 2 and 3 on all three days.",
                 "The deposit has no demographics, so age and sex are empty. The 18 participants all took part in the study; the number per group is not stated for the no-instruction group, which is inferred as 18 - 12 = 6."],
    unit_note="trial (one recording)",
)
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
