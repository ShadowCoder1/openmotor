"""Mehta et al. (2025), Sci Rep 15:36915: learning a redundant mapping between four upper-limb joint angles (IMUs, left arm) and the position
of a virtual paddle to intercept a falling ball (de novo learning), 560 trials per participant, groups with blocked or random practice and
low or high intrinsic variability. One row per trial, with the trial's samples (50 Hz) in a CSV.

Run from the site folder: python3 -I tools/std/convert/Mehta2025_SciRep.py
Raw files (untrusted data, read only): raw_extra/Mehta2025_SciRep/ RawData__P##_data.mat, TrialSeq_Files__PPData_BAP&BDP_Manuscript.csv, Readme.txt and the
authors' MATLAB scripts (read as text only). The members were pulled one by one with HTTP range requests out of the 1 GB Zenodo zip
(record 15488730). Each P##_data.mat holds the raw table JA (a MATLAB table object; not read) and the authors' trial-wise struct array 'trialdata'
(plain MATLAB structs, read with scipy), which is what is released.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mehta2025_SciRep"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
STUDY = "Mehta et al. 2025, Sci Rep"
SDIR = RAW / "_samples"
SDIR.mkdir(exist_ok=True)
FS = 50
DEVICE = ("four wireless inertial measurement units (Xsens MTw Awinda, 50 Hz) on sternum, upper arm, lower arm and hand of the non-dominant (left) arm; "
          "four joint angles drove a paddle on an 86-inch vertical screen at 2.5 m")
EXCLUDED_BY_AUTHORS = {23, 46}  # BCLEARNSYN_OutlierRemoverAndJAanalysis.m: 23 very high error, 46 no learning (3, 10, 21 have no data; 40 is left-handed and has no file)

info = pd.read_csv(RAW / "TrialSeq_Files__PPData_BAP&BDP_Manuscript.csv", sep=";")
info.columns = [c.strip() for c in info.columns]
info = info.set_index("PP")
files = sorted(RAW.glob("RawData__P*_data.mat"), key=lambda p: int(re.search(r"P(\d+)_data", p.name).group(1)))
pps = [int(re.search(r"P(\d+)_data", p.name).group(1)) for p in files]
assert len(pps) == 44, pps
SUBJ = {pp: i + 1 for i, pp in enumerate(pps)}
EXT = {"B": "blocked practice", "V": "random practice"}
INT = {"L": "low", "H": "high"}
PHASE = {0: ("pre-test", "baseline"), 1: ("practice phase 1", "learning"), 2: ("practice phase 2", "learning"), 3: ("practice phase 3", "learning"),
         4: ("practice phase 4", "learning"), 5: ("post-test", "test"), 6: ("transfer test", "transfer")}
TESTCOND = {"Blo": "blocked test schedule", "Var": "random test schedule"}


def scal(v):
    return float(np.asarray(v, dtype=float).ravel()[0])


rows, extra = [], []
for f, pp in zip(files, pps):
    td = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)["trialdata"]
    g = info.loc[pp, "Group"]
    code = f"{ID}_S{SUBJ[pp]:04d}"
    for t in td:
        assert int(t.PP) == pp and str(t.grp) == g
        trno, ph = int(t.trno), int(t.ph)
        n = len(np.atleast_1d(t.ppos))
        fn = f"samples/{code}_trial{trno:03d}.csv"
        df = pd.DataFrame({"time_s": np.arange(n) / FS, "ball_x": t.bpox, "ball_y": t.bpoy, "joint_scx": t.scx, "joint_ely": t.ely, "joint_elz": t.elz,
                           "joint_wry": t.wry, "paddle_x": t.ppos})
        local = os.path.join(SDIR, f"{code}_trial{trno:03d}.csv")
        df.to_csv(local, index=False, float_format="%.8g", lineterminator="\r\n")
        extra.append((fn, local))
        block, phase = PHASE[ph]
        cond = TESTCOND.get(str(t.testcond), EXT[g[0]]) if ph in (0, 5, 6) else EXT[g[0]]
        err, aerr = scal(t.error), scal(t.abserror)
        excl = bool(abs(scal(t.strtppos)) > 2) or pp in EXCLUDED_BY_AUTHORS
        rows.append(dict(
            dataset=ID, study=STUDY, experiment="Learning task: joint angles to paddle mapping",
            group=f"{EXT[g[0]]}, {INT[g[1]]} intrinsic variability", subject=str(pp), person=str(pp),
            subject_excluded_by_authors=pp in EXCLUDED_BY_AUTHORS, age=int(info.loc[pp, "Age"]), sex=info.loc[pp, "Sex"], handedness="right", population="healthy",
            setting="lab", device=DEVICE, session=1, block=block, trial=trno, phase=phase, time_s="", task="reaching", unit="recording",
            condition=cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=excl,
            recording=trno, samples_file=fn, signal_kind="ball position, four joint angles of the left arm (fractions of range of motion) and paddle position, from the start of the ball's descent to interception",
            channels="ball_x;ball_y;joint_scx;joint_ely;joint_elz;joint_wry;paddle_x", channel_units="Unity units;Unity units;fraction of range of motion;fraction of range of motion;fraction of range of motion;fraction of range of motion;Unity units",
            sample_rate_hz=FS, n_samples=n, duration_s=n / FS, events="",
            processing="none on the sample columns (unfiltered); the recording is the part of the trial in which the ball is between y = +10 and y = -10, up to the sample before interception (authors' trial split)",
            trial_info=(f"ball_departure_x={scal(t.bdp):g};ball_arrival_x={scal(t.bap):g};paddle_start_x={scal(t.strtppos):.8g};paddle_end_x={scal(t.endppos):.8g};"
                        f"error={err:.8g};abs_error={aerr:.8g};hit={int(scal(t.hit))};ball_touched_in_unity={int(scal(t.unityhit))}")))
D = pd.DataFrame(rows)
D["_hit"] = D.trial_info.str.extract(r"hit=(\d)")[0].astype(int)
D["_ae"] = D.trial_info.str.extract(r"abs_error=([-\d.e+]+)")[0].astype(float)

P = checks.published
kept = D[~D.subject_excluded_by_authors]
P("participants analysed in the learning task", kept.person.nunique(), 42, 0, "Results (42 participants analysed for the learning task)")
for ph, val in (("baseline", 14.98), ("test", 66.31), ("transfer", 56.39)):
    pooled = 100 * kept[(kept.phase == ph) & (~kept.excluded_by_authors)]._hit.mean()
    P(f"success rate, {ph} (% of trials with absolute error below 1.25 units)", pooled, val, 0.005, "Results, Learning task (14.98% pre-test, 66.31% post-test, 56.39% transfer)")
checks.internal("hit equals absolute error <= 1.25 (the authors' definition) on every trial", ((D._ae <= 1.25).astype(int) == D._hit).all())
n_by = kept.groupby("person").size()
checks.internal("560 planned trials per participant: 16 pre, 512 practice (4 x 128), 16 post, 16 transfer; a few trials lost to recording problems",
                n_by.max() == 560 and n_by.min() >= 540, f"min {n_by.min()}, max {n_by.max()}")
checks.internal("44 participants with data, 6 participants without (3, 10, 21: no data; 40 left-handed; 23, 46 excluded by authors)", D.person.nunique() == 44)
checks.internal("every recording has a samples file", len(extra) == len(D))
D = D.drop(columns=["_hit", "_ae"])

meta = dict(
    paper=dict(authors="Mehta A, Smith J, Travieso D, Bongers RM", year=2025,
               title="Extrinsic variability but not intrinsic variability affects the search for synergies in de-novo motor learning",
               venue="Sci Rep 15(1):36915", doi="10.1038/s41598-025-20895-w"),
    lab="Bongers (University of Groningen) / Mehta", data_url="https://doi.org/10.5281/zenodo.15488730", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Only the learning task (ball interception with a joint-angle to paddle mapping) is released; the deposit's raw tables JA (continuous recordings, MATLAB table objects) are not used because the authors' own trial-wise "
             "struct array trialdata, saved inside every P##_data.mat, holds the same samples split into trials. Per trial the unfiltered ball position, four joint angles and paddle position were written to a samples CSV; the authors' "
             "filtered and time-normalised copies and their analysis outputs are not included. The categorization (reaching) task data, the UCM output file and the MATLAB scripts of the deposit are not included."),
    how=[
        "One row is one interception trial (560 per participant: 16 pre-test, 4 x 128 practice, 16 post-test, 16 transfer-test trials) of one participant, who moved the non-dominant arm to control a paddle on a vertical screen and "
        "intercept a ball falling from one of four departure positions (x = -14, -7, 7, 14 Unity units) to one of four arrival positions (same values; -16, -9, 9, 16 in the transfer test). Practice was blocked (one arrival position at a time) or random.",
        "Samples (samples/<subject>_trial<nnn>.csv, 50 Hz): time_s, ball_x and ball_y (Unity units, ball moves from y = +10 down to -10), then joint_scx, joint_ely, joint_elz, joint_wry and paddle_x. "
        "The paddle position equals 8 x (sum of the four joint values) - 16 exactly on every sample (checked), so each joint value is a fraction of that person's calibrated range of motion that moves the paddle by 8 units. "
        "The names are the authors' variable names; from the paper's list and order (shoulder abduction-adduction, elbow pronation-supination, elbow flexion-extension, wrist ulnar-radial deviation) they are, in order, SAA, EPS, EFE and WUR; this assignment is inferred from the order, not stated in the deposit.",
        "trial_info holds the trial's ball departure and arrival position, the paddle position at the start and end of the recording, error (ball minus paddle position at the end, in Unity units), abs_error, hit (1 when abs_error <= 1.25, the authors' definition) and the Unity flag for the ball being touched. "
        "excluded_by_authors marks trials whose paddle start position was more than 2 units from the centre (the authors' outlier rule, 1.5% of trials); subject_excluded_by_authors marks the two participants the authors removed after the categorization task (very high error; no learning).",
        "Group is the authors' practice schedule (blocked = low extrinsic variability, random = high) and the intrinsic variability group from the categorization task (median split of the covariation of joint angles in a separate reaching task, not released here). "
        "Age and sex are from the deposit's participant table. Participants 3, 10 and 21 have no data and participant 40 (left-handed) has no file.",
    ],
    differences=["The paper recruited 47 right-handed participants (mean age 22.45, SD 3.39, 28 females); the deposit has trial data for 44, so the participant table cannot reproduce those numbers and they are not used as checks. "
                 "The success rates are computed over all non-outlier trials of the 42 analysed participants; the deposit's data were not re-filtered otherwise. Trials the authors removed by hand when building bapdata (7 trials in 5 participants) are kept here."],
    unit_note="trial (one ball interception)",
)
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
