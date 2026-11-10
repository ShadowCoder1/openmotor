"""Wang, Avraham, Tsay, Abram & Ivry (2024), PLoS Comput Biol: perturbation variability and implicit adaptation (OSF exg4c).
Raw (untrusted, read only): ErrVar_share.zip; only Our_Exp/{low12,high12,low30,high30}.mat are the new experiments of this paper
(AlbertNHB, TBT and models hold re-analysed data of other papers and simulations, not converted).
Each .mat: hand_common_angle (4 targets, 80 cycles, N people, NaN = invalid trial, includes people the authors dropped),
hand_common_bin_angle (80 cycles, kept people; mean over the 4 targets), mt, rt (kept people, one value each, ms).
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wang2024_PLoSCompBiol"
Z = zipfile.ZipFile(stdlib.raw_dir(ID) / "ErrVar_share.zip")
names = [c[1] for c in stdlib.reach90_columns()]
checks = stdlib.Checks()
# file, experiment, group, clamp description, fixed clamp size
G = [("low12", "Exp1_12deg", "Zero_variance_12", 12.0), ("high12", "Exp1_12deg", "High_variance_12", None),
     ("low30", "Exp2_30deg", "Zero_variance_30", 30.0), ("high30", "Exp2_30deg", "High_variance_15to45", None)]
rows, ppl = [], []
binmeans = {}
for fn, exp, grp, clamp in G:
    m = sio.loadmat(io.BytesIO(Z.read(f"ErrVar_share/Our_Exp/{fn}.mat")))
    H, B = m["hand_common_angle"], m["hand_common_bin_angle"]
    mt, rt = m["mt"].ravel(), m["rt"].ravel()
    N, nk = H.shape[2], B.shape[1]
    # the bin file holds the kept people in the raw order: match them to raw columns in order
    bm = np.nanmean(np.where(np.isnan(H), np.nan, H), axis=0) if False else None
    keep, k = [], 0
    for j in range(nk):
        while True:
            col = np.array([np.nanmean(H[:, c, k]) if not np.isnan(H[:, c, k]).all() else np.nan for c in range(80)])
            if np.array_equal(np.nan_to_num(col, nan=9e9), np.nan_to_num(B[:, j], nan=9e9)):
                keep.append(k); k += 1; break
            k += 1
            assert k < N, (fn, j)
    dropped = [i for i in range(N) if i not in keep]
    binmeans[fn] = np.nanmean(-B[20:80], axis=0)
    for i in range(N):
        pid = f"{fn}_{i + 1}"
        kept = i in keep
        for c in range(80):
            for t in range(4):
                v = H[t, c, i]
                j = c * 4 + t
                phase = "baseline" if c < 20 else "learning"
                r = dict.fromkeys(names, "")
                r.update(dataset=ID, experiment=exp, study=ID, subject=pid, person=pid, id_type="prolific", person_link="assumed_unique",
                         group=grp, prior_exposure="unknown", subject_excluded_by_authors=str(not kept),
                         subject_exclusion_reason="" if kept else "not in the authors' analysed set (bin file); reason not stated in the deposit",
                         handedness="right", device="unknown", setting="online", population="healthy",
                         trial=j, trial_orig=j + 1, session=0, block=("no_feedback" if c < 10 else "veridical_feedback" if c < 20 else "clamp"),
                         phase=phase, trial_kind="reach", break_before=str(j == 0), target_id=t + 1, target_distance_norm="1",
                         target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
                         instr_goal="hand_to_target", instr_strategy="forbidden", instr_disclosure="exists", instr_report="none",
                         hand_used="", hand_angle="" if np.isnan(v) else v, hand_measure="endpoint",
                         excluded_by_authors=str(bool(np.isnan(v))))
                if c < 10:
                    r.update(fb_visible="none", fb_mode="none", fb_form="none", online_cursor="none")
                elif c < 20:
                    r.update(fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical", rotation="0")
                else:
                    r.update(fb_visible="online", fb_mode="clamped", fb_form="cursor", online_cursor="perturbed",
                             clamp="" if clamp is None else clamp)
                rows.append(r)
        pos = keep.index(i) if kept else None
        ppl.append(dict(person=pid, group=grp, analysed=kept, mean_rt_ms=rt[pos] if kept else np.nan, mean_mt_ms=mt[pos] if kept else np.nan))
    checks.internal(f"{fn}: rows", True, f"{N} people raw, {nk} kept")
    # published: analysed numbers
    if fn == "low12":
        checks.published("Exp1 zero-variance group, people", nk, 37, 0, "Methods, Participants (final N, Experiment 1)")
    if fn == "high12":
        checks.published("Exp1 high-variance group, people", nk, 39, 0, "Methods, Participants (final N, Experiment 1)")
checks.published("trials per person (40 no-feedback + 40 veridical + 240 clamp)", 320, 320, 0, "Methods, trial structure")
n2 = len(binmeans["low30"]) + len(binmeans["high30"])
checks.published("Exp2 analysed people from the t-test degrees of freedom (t(61) -> 63 people)", n2 - 2, 61, 0, "Results, Experiment 2, averaged hand angle")
t1 = stats.ttest_ind(binmeans["high12"], binmeans["low12"]).statistic
t2 = stats.ttest_ind(binmeans["high30"], binmeans["low30"]).statistic
print("t exp1", t1, "t exp2", t2)
checks.published("Exp1 averaged hand angle, High minus Zero variance (t)", abs(t1), 3.4, 0.5, "Results, Experiment 1")
checks.published("Exp2 averaged hand angle, High minus Zero variance (t)", abs(t2), 0.24, 0.3, "Results, Experiment 2")
checks.internal("adaptation is negative in the file frame (late clamp mean below -10 deg)", all(v.mean() > 10 for v in binmeans.values()))

import pandas as pd  # noqa: E402
pcols = [("group", "text", "", "Group: zero or high perturbation variance, and clamp size."),
         ("analysed", "true/false", "", "The person is in the authors' analysed set (bin file)."),
         ("mean_rt_ms", "number", "ms", "The authors' 'rt': one mean reaction time per analysed person, ms (not defined further in the deposit)."),
         ("mean_mt_ms", "number", "ms", "The authors' 'mt': one mean movement time per analysed person, ms.")]
paper = dict(authors="Wang T, Avraham G, Tsay JS, Abram SJ, Ivry RB", year=2024,
             title="Perturbation Variability Does Not Influence Implicit Sensorimotor Adaptation",
             venue="PLoS Comput Biol 20(4):e1011951", doi="10.1371/journal.pcbi.1011951")
meta = dict(
    study=ID, paper=paper, lab="Ivry (UC Berkeley)", data_url="https://osf.io/exg4c/", license="none stated", license_url="",
    changes="converted to the OpenMotor format; participant numbers replaced by codes; the authors' hand angles kept in degrees in their sign; "
            "trial order rebuilt as cycle x target; people missing from the authors' analysed set kept and marked.",
    unit_note="trial (one reach) of one participant",
    how=[
        "One row is one reach of an online error-clamp experiment (Prolific, OnPoint): 80 cycles of 4 targets = 320 trials: 40 without "
        "feedback, 40 with veridical cursor feedback, 240 with a clamped cursor, with instructions to ignore the cursor and reach to the target. "
        "Experiment 1 clamps 12 deg (zero-variance, 37 people) or Gaussian mean 12, SD 12 deg (high-variance, 39); Experiment 2 clamps 30 deg "
        "(29 people) or uniform 15 to 45 deg (34).",
        "hand_angle is the authors' 'hand_common_angle': endpoint hand angle minus target angle, in degrees, in one common frame in which learning "
        "is negative (late clamp trials average about -20 deg for the 30 deg clamp). The readme says the sign was reversed for counter-clockwise "
        "rotations, so people trained with both clamp directions are in one frame; the original screen direction is lost. We therefore say clamp "
        "is positive (counter-clockwise) in this frame, inferred from the sign of the learning and not stated by the authors. clamp is empty for "
        "the high-variance groups (per-trial clamp values are not in the deposit).",
        "The deposit does not say which of the 4 targets (at +-45 and +-135 deg) is which index, so target_id is 1 to 4 and target_angle is empty. "
        "Empty (NaN) trials are the authors' invalid trials (movement time over 500 ms or angle error over 70 deg) and are flagged excluded_by_authors.",
        "Per-trial reaction and movement times are not in the deposit: participants.csv holds one mean per analysed person. People in the raw file "
        "but not in the authors' bin file (reason not stated; their share of invalid trials is small, so it is not the 70% rule) are kept with subject_excluded_by_authors True.",
    ],
    differences=[
        "The paper reports final N of 38 (zero) and 40 (high) for Experiment 2; the deposit holds 30 and 37 people (29 and 34 analysed, which agrees with "
        "the reported t(61)). Experiment 1 agrees (37 and 39 analysed). The paper's text also gives t(71) for Experiment 1 where 37 + 39 people give 74 df.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks, participants=ppl, participant_columns=pcols)
