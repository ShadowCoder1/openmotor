"""Codol, Holland, Manohar & Galea (2020), J Neurosci: reward-based improvements in motor control (OSF 7as8g, CC BY 4.0).
Raw (untrusted, read only): R_Trial.mat (Experiment 1, reward magnitude, 30 people x 192 trials) and RP_Trial.mat
(Experiment 2, reward vs punishment, 30 x 256), each with a struct T whose fields are [trial, participant] arrays.
The 1.2 and 2 GB R_Data/RP_Data hand-path files and the stiffness-experiment files are not converted.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Codol2020_JNeurosci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
rows = []
EXP = {"R_Trial.mat": ("Exp1_reward_magnitude", 192, 0.3), "RP_Trial.mat": ("Exp2_reward_punishment", 256, 0.7)}
excl_pct = {}
for f, (exp, nt, _) in EXP.items():
    T = sio.loadmat(RAW / f, struct_as_record=False, squeeze_me=True)["T"]
    assert T.trialno.shape == (nt, 30)
    n_ex = 0
    for p in range(30):
        pid = int(T.S[0, p])
        assert (T.S[:, p] == pid).all()
        for t in range(nt):
            g = lambda k: getattr(T, k)[t, p]  # noqa: E731
            rt, err, ang, d = float(g("RT")), float(g("err")), float(g("errangle")), int(g("D"))
            bad = rt < 0.2 or rt > 1 or (err > 6 and d == 1) or (ang > 20 and d == 1)  # the authors' rule (MS_R.m)
            n_ex += bad
            pot = float(g("potrwd"))
            jit = float(g("targjitter"))
            rows.append(dict(
                dataset=ID, study=ID, experiment=exp, group="", subject=f"{exp}_{pid}", person=f"p{pid}_{exp[:4]}",
                subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population="healthy", setting="lab",
                device="KINARM endpoint robot", session=1, block=f"block{int(g('b'))}", trial=t + 1, phase="learning",
                time_s=np.nan, task="reaching", unit="trial", condition=f"potential_reward_{pot:g}", pert_type="none", pert_value=0,
                pert_unit="", feedback="normal", excluded_by_authors=bool(bad),
                target_angle_deg=float(g("targ_angle")), trial_orig=int(g("trialno")), block_orig=int(g("b")),
                reward_context="punishment" if pot < 0 else "reward", potential_reward_pence=pot, score_pence=float(g("score")) + 0.0,
                tau1_ms=float(g("tau1")), tau2_ms=float(g("tau2")), reaction_time_s=rt, movement_time_s=float(g("MT")),
                peak_velocity_m_s=float(g("peakVel")), radial_error_cm=err, angular_error_deg_abs=ang,
                distractor_present=bool(g("DistOn")), not_distracted=bool(d), distractor_delay_s=jit))
    pct = 100 * n_ex / (nt * 30)
    excl_pct[exp] = pct
    checks.published(f"{exp}: participants", 30, 30, 0, "Methods, Experiment " + exp[3])
    checks.published(f"{exp}: percent of trials removed by the authors' rules", pct, EXP[f][2], 0.15, "Methods, Experiment " + exp[3] + ", exclusions")
    nd = np.mean(T.DistOn[T.b == (2 if exp.startswith("Exp1") else 2)] == 1) if False else None
    main = T.b >= 2 if exp.startswith("Exp1") else np.isin(T.b, (2, 4))
    if exp.startswith("Exp1"):
        checks.published("Exp1: share of distractor trials in the 168-trial main block (%)", 100 * np.mean(T.DistOn[main]), 42.86, 0.05, "Methods, Experiment 1")
        checks.published("Exp1: trials per person (24-trial block + 168-trial main block)", nt, 192, 0, "Methods, Experiment 1")
    else:
        checks.published("Exp2: share of distractor trials in the distractor blocks (%)", 100 * np.mean(T.DistOn[main]), 42.86, 0.05, "Methods, Experiment 2")
    checks.internal(f"{exp}: no-distractor blocks have no distractors", not T.DistOn[~main].any())
checks.internal("rows", len(rows) == 30 * 192 + 30 * 256, str(len(rows)))

paper = dict(authors="Codol O, Holland PJ, Manohar SG, Galea JM", year=2020,
             title="Reward-Based Improvements in Motor Control Are Driven by Multiple Error-Reducing Mechanisms",
             venue="J Neurosci 40(18):3604-3620", doi="10.1523/JNEUROSCI.2646-19.2020")
meta = dict(
    study=ID, paper=paper, lab="Galea (University of Birmingham)", data_url="https://osf.io/7as8g/", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format; participant numbers replaced by codes; the authors' per-trial values kept in their units; "
            "the authors' outlier rule applied as a flag, not a deletion; no values recoded.",
    unit_note="trial (one reach) of one participant",
    how=[
        "One row is one trial of a KINARM endpoint-robot reaching task to one of four targets (22.5, 67.5, 112.5, 157.5 deg from the start), "
        "in which a monetary reward (Experiment 1: potential 0, 10 or 50 pence) or reward and punishment blocks (Experiment 2) depended on a speed "
        "criterion that adapted to each person; on some trials a distractor target appeared. 30 people took part in each experiment.",
        "The deposit's R_Trial.mat and RP_Trial.mat hold the per-trial measures only (timing, peak velocity, radial error, angular error, reward); "
        "the 48 training trials per person are not in it, and the full hand paths (1.2 and 2 GB) were not converted. Angular error is unsigned in the "
        "deposit, so there is no signed reach direction and the reach90 format could not be used; a task module holds the authors' measures instead.",
        "excluded_by_authors applies the rule in the authors' script: reaction time under 0.2 s or over 1 s, or on non-distracted trials radial error over 6 cm "
        "or angular error over 20 deg. Reaction time, error and reward values of excluded trials are kept.",
        "Experiment 2 has 256 rows per person (blocks of 16, 112, 16, 112 trials) where the paper's text describes 12 and 112 trials per block. "
        "Age and sex are not in the deposit (paper: Exp 1 median age 19, 2 men; Exp 2 median 20.5, 4 men).",
    ],
    differences=[],
)
stdlib.release(ID, "reward_speed_accuracy_reaching", rows, meta, checks)
