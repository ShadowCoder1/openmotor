"""Wang, Zhu, Kana, Yu, He & Wei (2020) Sci Rep: hand localization by visual matching (Experiment 1: 26 participants x 3
sessions; Experiment 2: 21 participants x 4 sessions), 100 targets per session.

Raw files: raw_extra/Wang2020_SciRep/data_exp1.zip and data_exp2.zip (OSF project jenxa), read with zipfile + h5py (the .mat files
are MATLAB v7.3 = HDF5; pure data, never executed). Only the localization part is released: each file also holds the
trajectory-matching task (moveData: raw stylus samples, with the authors' score function in score_endtrajMatch.m) and a 300-row
block coded 301 that the paper does not describe. The file name gives the session: exp1 subjectN_day1_1, _day1_2, _day2 =
sessions 1-3; exp2 subjectN_day1_1, day1_2, day2_1, day2_2 = sessions 1-4 (the authors' analyse.m reads them in this order).
Each file holds one session's 100 matches in PreProprioceptionData or PostProprioceptionData (the other is all NaN).

Run from the site folder: python3 -I tools/std/convert/Wang2020_SciRep.py"""
import io
import re
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wang2020_SciRep"
MODULE = "hand_localization"
DEVICE = ("digitizing tablet and stylus held in the left hand, hand hidden by a semi-silvered mirror under a 75 Hz projector "
          "screen")
EXPS = [("data_exp1.zip", "experiment 1", re.compile(r"exp1_subject(\d+)_day(\d)(?:_(\d))?\.mat$"), {("1", "1"): 1, ("1", "2"): 2, ("2", None): 3}),
        ("data_exp2.zip", "experiment 2", re.compile(r"control_subject(\d+)_day(\d)_(\d)\.mat$"), {("1", "1"): 1, ("1", "2"): 2, ("2", "1"): 3, ("2", "2"): 4})]


def load(buf):
    f = h5py.File(io.BytesIO(buf), "r")
    L = np.array(f["PositionList"]).T
    if L.shape == (2, 100):  # a few files store it the other way round
        L = L.T
    order = np.array(f["PointOder"]).flatten() if "PointOder" in f else np.full(100, np.nan)  # experiment 2 files have none
    for key in ("PreProprioceptionData", "PostProprioceptionData"):
        P = np.array(f[key]).T
        if np.isfinite(P).sum() > 0:
            return order, L, P
    raise ValueError("no localization data")


def main():
    ck = stdlib.Checks()
    rows = []
    for zname, exp, rx, smap in EXPS:
        z = zipfile.ZipFile(stdlib.raw_dir(DS) / zname)
        files = {}
        for n in z.namelist():
            m = rx.search(n)
            if m and not n.startswith("__MACOSX"):
                files[(int(m.group(1)), smap[(m.group(2), m.group(3))])] = n
        subs = sorted({s for s, _ in files})
        nses = max(smap.values())
        ck.internal(f"{exp}: every participant has all {nses} session files", all((s, k) for s in subs for k in range(1, nses + 1) if (s, k) in files)
                    and len(files) == len(subs) * nses, f"{len(subs)} participants, {len(files)} files")
        for s in subs:
            for k in range(1, nses + 1):
                order, L, P = load(z.read(files[(s, k)]))
                assert L.shape == (100, 2) and P.shape[0] == 100, (exp, s, k, L.shape, P.shape)
                tgt, mat = L / 1000.0, P[:, 2:4] / 1000.0
                for i in range(100):
                    rows.append(dict(exp=exp, s=s, k=k, i=i, grid=order[i], tx=tgt[i, 0], ty=tgt[i, 1], mx=mat[i, 0], my=mat[i, 1],
                                     err=float(np.hypot(*(mat[i] - tgt[i]))), t=P[i, 4]))
    R = pd.DataFrame(rows)
    # the authors' properr3.m drops a match whose squared error is above the session mean + 3 SD (SD with n-1); the paper's
    # session means reproduce only with these matches left out
    d2 = R.err ** 2
    g = d2.groupby([R.exp, R.s, R.k])
    R["out"] = (d2 > g.transform("mean") + 3 * g.transform("std")).to_numpy()
    ck.internal("every match has a finite position", bool(np.isfinite(R[["mx", "my", "tx", "ty"]].to_numpy()).all()))
    ck.internal("each session shows 100 different target positions", bool((R.groupby(["exp", "s", "k"]).apply(lambda g: len(set(zip(g.tx.round(3), g.ty.round(3)))) == 100)).all()))
    df = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment=R.exp, group="", subject=[f"{e[-1]}_{s}" for e, s in zip(R.exp, R.s)],
        subject_excluded_by_authors=False, age=np.nan, sex="", handedness="right", population="healthy", setting="lab",
        device=DEVICE, session=R.k, block=[f"session {k}" for k in R.k], trial=R.i + 1, phase="test", time_s=np.nan,
        task="hand localization", unit="trial", condition="", pert_type="none", pert_value=0.0, pert_unit="", feedback="none",
        excluded_by_authors=R.out, grid_index=R.grid, target_x_cm=R.tx, target_y_cm=R.ty, matched_x_cm=R.mx, matched_y_cm=R.my,
        localization_error_cm=R.err, matching_time_s=R.t))
    df["trial"] = df.groupby("subject").cumcount() + 1
    e1 = R[R.exp == "experiment 1"]
    e2 = R[R.exp == "experiment 2"]
    ck.published("participants in Experiment 1", e1.s.nunique(), 26, 0, "Results, Experiment 1 / Methods, Participants")
    ck.published("participants in Experiment 2", e2.s.nunique(), 21, 0, "Results, Experiment 2 / Methods, Participants")
    ck.published("participants in total", e1.s.nunique() + e2.s.nunique(), 47, 0, "Methods, Participants")
    ck.published("targets per session", R.groupby(["exp", "s", "k"]).size().mean(), 100, 0, "Methods, Visual-matching task")
    ck.published("Experiment 1 sessions", e1.k.nunique(), 3, 0, "Methods, Experimental protocols")
    ck.published("Experiment 2 sessions", e2.k.nunique(), 4, 0, "Methods, Experimental protocols")
    for k, rep in [(1, 3.098), (2, 2.944), (3, 2.420)]:
        v = e1[(e1.k == k) & ~e1.out].groupby("s").err.mean().mean()
        ck.published(f"Experiment 1 mean localization error in session {k}, cm", v, rep, 0.0015, "Results, Experiment 1 (Fig. 1), matches flagged by the authors' outlier rule left out")
    w = R.groupby("exp").tx.agg(["min", "max"])
    ck.published("target workspace width in cm (48.76) from the extreme target columns plus one column step", (R.tx.max() - R.tx.min()) + 2.438, 48.76, 0.05, "Methods, Visual-matching task")

    meta = dict(
        study=DS,
        paper=dict(authors="Wang T, Zhu Z, Kana I, Yu Y, He H, Wei K", year=2020,
                   title="Accuracy of hand localization is subject-specific and improved without performance feedback",
                   venue="Sci Rep 10:19188", doi="10.1038/s41598-020-76220-0"),
        lab="Wei (Peking University)",
        data_url="https://osf.io/jenxa/",
        license="none stated",
        license_url="",
        changes=("Read the localization block of each of the authors' .mat files (their trial order), converted positions from the "
                 "tablet's 0.01 mm units to cm, added each match's distance from its target, marked the matches the authors' properr3.m leaves out, and gave participants codes. "
                 "The file name gives the session."),
        how=["One row is one hand-localization trial: a target dot (50 mm) was shown on a 5 x 20 grid of 100 positions in random "
             "order and the participant, who could not see the hand, moved the stylus held in the left hand until its tip was on "
             "the target and pressed the space bar. No performance feedback was given within a session (only the 16 familiarization "
             "trials before some sessions, which are not in the deposit, showed the stylus position).",
             "target_x/y_cm are the authors' PositionList and matched_x/y_cm the stylus position at the key press; "
             "localization_error_cm is the distance between them, whose mean over 100 targets is the paper's accuracy measure "
             "(the paper's error vector is target minus stylus).",
             "Experiment 1 (26 participants) has 3 sessions (two on day 1, one 24 h later) and Experiment 2 (21 participants) 4 sessions "
             "(two per day). Not included: the trajectory-matching task of the same files and a 300-row block the paper does not describe. "
             "The deposit does not say whether a participant in experiment 1 is also in experiment 2; they are coded as different people."],
        unit_note="hand-localization trial (one target and the stylus position placed on it)",
        differences=["The paper's session means (3.098, 2.944 and 2.420 cm) reproduce only after leaving out the matches whose squared error is above the session mean + 3 SD (the rule in the authors' properr3.m, which is not mentioned in the paper); those matches are marked excluded_by_authors. With all matches the means are 3.167, 3.014 and 2.470 cm.",
                     "The deposit's .mat files also hold raw stylus samples; the paper's trajectory-matching errors are not rebuilt here."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for k in ck.items:
        if k["kind"] == "published":
            print("published", k["what"], round(k["computed"], 4), k["reported"], k["ok"])
        elif not k["ok"]:
            print("INTERNAL FAIL", k["what"])


if __name__ == "__main__":
    main()
