"""Codol, Kashefi, Forgaard, Galea, Pruszynski & Gribble (2023) eLife: five KINARM experiments on how fast feedback responses
depend on reward (In-Out target, cursor jump, target jump, target selection, proprioception-cued reaction time), per-trial
scalars of 75 participant files.

Raw files: the authors' per-participant MATLAB files (reflx_N, cjump_N, tjump_N, tarsw_N, rtime_N .mat, 40-250 MB each, 8.5 GB in
all). The PC has too little free disk for all of them at once, so each was fetched on its own, read by
Codol2023_eLife_extract.py (scipy.io.loadmat; pure data, no deposit code run) into a small CSV of per-trial scalars under
raw_extra/Codol2023_eLife/scalars/, and then deleted. This converter reads those CSVs (and the authors' tiny files only as
text: full_analysis.m, computePerformance.m, wilcoxon.m, loadData.m). EMG and the 1 kHz position, velocity, force and torque
traces are not released. R_Trial.mat / RP_Trial.mat (choice reaction time, 60 participants) are the data of Codol et al. 2020
(J Neurosci, a separate listing) and are not released here.

Run from the site folder: python3 -I tools/std/convert/Codol2023_eLife.py"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Codol2023_eLife"
MODULE = "feedback_perturbed_reach_reward"
DEVICE = "KINARM exoskeleton robot (arm in the horizontal plane, vision of the arm blocked), 1000 Hz"
EXPS = [("reflx", "In-Out target (SLR and LLR)", 16, "torque_pulse"), ("cjump", "Cursor jump", 15, "cursor_jump"),
        ("tjump", "Target jump", 14, "target_jump"), ("tarsw", "Target selection", 14, "torque_pulse"),
        ("rtime", "Proprioception-cued reaction time", 17, "torque_pulse")]


def wilcoxon_w(d):
    """The authors' wilcoxon.m: flip the sign if fewer than half are positive, then the sum of ranks of the positive differences."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) / 2 > (d > 0).sum():
        d = -d
    return float(stats.rankdata(np.abs(d))[d > 0].sum())


def main():
    sd = stdlib.raw_dir(DS) / "scalars"
    ck = stdlib.Checks()
    frames = []
    for pre, label, n_paper, pert in EXPS:
        files = sorted(sd.glob(f"{pre}_*.csv"), key=lambda p: int(re.search(r"_(\d+)", p.name).group(1)))
        for p in files:
            t = pd.read_csv(p)
            t["pre"], t["label"], t["pert"], t["pnum"] = pre, label, pert, int(re.search(r"_(\d+)", p.name).group(1))
            frames.append(t)
    A = pd.concat(frames, ignore_index=True)
    ck.internal("every file's performance measure is movement time (MT)", bool((A.performance_measure == "MT").all()))
    ck.internal("participant files read: 16 In-Out, 15 cursor jump, 14 target jump, 13 target selection, 17 reaction time",
                A.groupby("pre").pnum.nunique().to_dict() == {"cjump": 15, "reflx": 16, "rtime": 17, "tarsw": 13, "tjump": 14},
                str(A.groupby("pre").pnum.nunique().to_dict()))
    jump = A.pre.isin(["cjump", "tjump"])
    A["pert_value"] = np.where(jump, 10.0 * A.displ, np.nan)
    df = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment=A.label, group="", subject=[f"{p}_{n}" for p, n in zip(A.pre, A.pnum)],
        subject_excluded_by_authors=False, age=np.nan, sex="", handedness="right", population="healthy", setting="lab",
        device=DEVICE, session=np.nan, block=[f"b{int(b)}" for b in A.block], trial=A.groupby(["pre", "pnum"]).cumcount() + 1,
        phase="test", time_s=np.nan, task="reaching", unit="trial",
        condition=np.where(A.reward > 0, "reward", "no reward"), pert_type=A.pert,
        pert_value=A.pert_value, pert_unit=np.where(jump, "deg elbow angle (sign: displ code)", ""), feedback="normal",
        excluded_by_authors=False,
        authors_trial=A.trial, authors_block=A.block, control_flag=A.control, displacement_code=A.displ, n_targets=A.n_targets,
        reward_max_cents=A.reward, reward_tau=A.tau, movement_time_s=A.MT, target_x_m=A.end_x, target_y_m=A.end_y,
        target_angle_deg=A.end_angle, target_radius_m=A.end_vis_rad, jump_x_m=A.jump_x, jump_y_m=A.jump_y, jump_angle_deg=A.jump_angle,
    ))
    ck.internal("no missing movement times", bool(A.MT.notna().all()))

    # ---- the paper's behavioural statistics (Wilcoxon W of the reward effect on movement time), rebuilt as full_analysis.m does
    def per_pt(pre, sel_r, sel_n, absdev=False):
        out = []
        for pn, g in A[A.pre == pre].groupby("pnum"):
            r, n = g.MT[sel_r(g)].median(), g.MT[sel_n(g)].median()
            out.append(abs(r - 0.6) - abs(n - 0.6) if absdev else r - n)
        return np.array(out)
    rw = lambda g: g.reward != 0  # noqa: E731
    nr = lambda g: g.reward == 0  # noqa: E731
    cond_a = lambda g: ((g.control == 1) & (g.displ == 0)) | ((g.control == 0) & (g.displ == 1))  # noqa: E731
    W = {
        "In-Out target (SLR contrast)": (wilcoxon_w(per_pt("reflx", lambda g: rw(g) & cond_a(g), lambda g: nr(g) & cond_a(g))), 136),
        "In-Out target (LLR contrast)": (wilcoxon_w(per_pt("reflx", lambda g: rw(g) & (g.displ == 1), lambda g: nr(g) & (g.displ == 1))), 131),
        "Target selection": (wilcoxon_w(per_pt("tarsw", lambda g: rw(g) & (g.displ == 1), lambda g: nr(g) & (g.displ == 1))), 92),
        "Proprioception-cued RT": (wilcoxon_w(per_pt("rtime", rw, nr)), 153),
        "Cursor jump": (wilcoxon_w(per_pt("cjump", lambda g: rw(g) & (g.displ != 0), lambda g: nr(g) & (g.displ != 0), True)), 108),
        "Target jump": (wilcoxon_w(per_pt("tjump", lambda g: rw(g) & (g.displ != 0), lambda g: nr(g) & (g.displ != 0), True)), 103),
    }
    diffs = []
    for k, (got, rep) in W.items():
        if abs(got - rep) <= 0.01:
            ck.published(f"Wilcoxon W of the reward effect on movement time, {k}", got, rep, 0.01, "Results (Wilcoxon signed-rank W as printed)")
        else:
            diffs.append(f"Wilcoxon W of the reward effect on movement time, {k}: {got:g} here, {rep:g} in the paper")
    cnt = A.groupby("pre").pnum.nunique()
    for pre, label, n_paper, _ in EXPS:
        if cnt[pre] == n_paper:
            ck.published(f"participants, {label}", cnt[pre], n_paper, 0, "Methods, Participants")
        else:
            diffs.append(f"{label}: the paper has {n_paper} participants, the deposit {cnt[pre]} files")
    tr = A.groupby(["pre", "pnum"]).size().groupby("pre").mean()
    ck.published("trials per participant, proprioception-cued RT (27 epochs of 4)", tr["rtime"], 108, 0, "Methods, Proprioception-cued RT")
    ck.published("trials per participant, In-Out target", tr["reflx"], 336, 0, "Methods, In-Out Target")
    ck.published("trials per participant, target selection", tr["tarsw"], 224, 0, "Methods, Target Selection")
    if tr["cjump"] != 312:
        diffs.append(f"cursor and target jump: the paper has 312 trials per participant, the deposit files hold {tr['cjump']:g}")
    ck.published("maximum return g for rewarded trials in the cursor jump experiment, cents", A.reward[A.pre == "cjump"].max(), 10, 0, "Methods, Reward")
    ck.published("maximum return g for rewarded trials in the In-Out experiment, cents", A.reward[A.pre == "reflx"].max(), 15, 0, "Methods, Reward")
    print("not reproduced / differences:", diffs)

    meta = dict(
        study=DS,
        paper=dict(authors="Codol O, Kashefi M, Forgaard CJ, Galea JM, Pruszynski JA, Gribble PL", year=2023,
                   title="Sensorimotor feedback loops are selectively sensitive to reward", venue="eLife 12:e81325",
                   doi="10.7554/elife.81325"),
        lab="Gribble & Pruszynski (Western University)",
        data_url="https://osf.io/7t8yj/",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Read the per-trial scalars (trial and block numbers, control, displacement and reward codes, tau, movement time, "
                 "target positions) from the authors' 75 participant MATLAB files in file order; the EMG and the 1 kHz kinematic, "
                 "force and torque traces (8.5 GB) are not released; participant file numbers are replaced by codes."),
        how=["One row is one trial of a KINARM arm-reaching experiment in which half the trials were rewarded (a '$$$' cue and a "
             "return of up to 10 or 15 cents that decayed with movement time) and half were not. Five experiments are included "
             "(experiment column): In-Out target (a torque pulse at the shoulder and elbow while reaching to an inward or outward target), "
             "cursor jump and target jump (the cursor or the target jumped by 10 degrees of elbow angle in either direction, or not), "
             "target selection (a torque pulse with one or two targets) and proprioception-cued reaction time (a small shoulder torque "
             "pulse cued the movement).",
             "The authors' codes are kept as shared: control_flag, displacement_code, n_targets, reward_max_cents (0 = not rewarded), "
             "reward_tau and movement_time_s; target positions are metres relative to the start. pert_value is 10 times the "
             "displacement code (degrees of elbow angle, signed by the code) in the two jump experiments, empty in the torque experiments. "
             "The tasks are mechanical or visual perturbations in joint space with EMG responses as the main outcome, so the data are not "
             "in the reach90 format (no hand angle per trial is released).",
             "Subjects of different experiments are coded as different people; the deposit does not say whether any participant took part "
             "in more than one."],
        unit_note="trial",
        differences=diffs + ["The paper's choice reaction-time experiment (60 participants) re-analyses Codol et al. 2020 and is not part of this release."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for k in ck.items:
        if k["kind"] == "published":
            print("published", k["what"], round(k["computed"], 4), k["reported"], k["ok"])
        elif not k["ok"]:
            print("INTERNAL FAIL", k["what"])


if __name__ == "__main__":
    main()
