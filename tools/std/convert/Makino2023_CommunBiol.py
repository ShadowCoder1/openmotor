"""Makino, Hayashi & Nozaki (2023) Commun Biol: divisively normalized processing of uncertain visual feedback (visuomotor learning).

Raw files: raw_extra/Makino2023_CommunBiol/exp1_af.mat, exp1_fbr.mat, exp2_af.mat, exp2_fbr.mat, exp3_afre.mat (Zenodo 8343357;
MATLAB v7.3 or v5, read with h5py / scipy only; figure scripts were read as text, never run). Each holds a trial-kind x participant matrix
of the authors' learning response (af, probe trials) or feedback response (fbr, perturbation trials). The force waveform files
(exp*_waves.mat) are not used.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Makino2023_CommunBiol"
R = stdlib.raw_dir(DS)


def load(name):
    p = R / f"{name}.mat"
    try:
        m = sio.loadmat(p)
        k = [k for k in m if not k.startswith("__")][0]
        return np.asarray(m[k], float)
    except NotImplementedError:
        with h5py.File(p) as h:
            k = list(h.keys())[0]
            return np.array(h[k]).T  # MATLAB orientation: kinds x participants


OFF = [-45, -30, -15, -7.5, 0, 7.5, 15, 30, 45]
DB = [-45, -30, -15, 0, 15, 30, 45]


def kind1(i):  # i = 1..64, per the authors' script comments
    if i <= 9:
        return f"single cursor, visual rotation {OFF[i - 1]:g} deg"
    if i <= 58:
        g, p = divmod(i - 10, 7)
        return f"double cursor, group '{DB[g]} double', second cursor {DB[p]} deg"
    return f"triple cursor, kind {i - 58}"


def kind23(i):  # 1..15, per the authors' script comments
    mu = {1: 0, 2: 0, 3: 0, 4: -14, 6: -14, 8: -14, 5: 14, 7: 14, 9: 14, 10: -40, 12: -40, 14: -40, 11: 40, 13: 40, 15: 40}[i]
    sd = {1: 0, 2: 7, 3: 20, 4: 0, 6: 7, 8: 20, 5: 0, 7: 7, 9: 20, 10: 0, 12: 7, 14: 20, 11: 0, 13: 7, 15: 20}[i]
    return f"cursor cloud mean {mu} deg, SD {sd} deg"


EXPS = [("experiment 1", "exp1_af", "exp1_fbr", kind1, set()),
        ("experiment 2", "exp2_af", "exp2_fbr", kind23, {15}),
        ("experiment 3", "exp3_afre", None, kind23, {7, 9})]
MEAS = {"af": ("learning_response", "Lateral force impulse on a force-channel probe trial, integrated from 200 ms before movement onset to the "
                                    "time of peak handle velocity (the authors' 'learning response'), as stored; the experiment 1 values are "
                                    "symmetrized by the authors, af(x) = (af(x) - af(-x))/2"),
        "fb": ("feedback_response", "Lateral force on the perturbation trial integrated from 100 ms before movement onset to the time of peak "
                                    "velocity (the authors' 'feedback response'), as stored")}
ck = stdlib.Checks()
rows, mats = [], {}
for ename, fa, ff, kf, excl in EXPS:
    for key, fn in (("af", fa), ("fb", ff)):
        if fn is None:
            continue
        M = load(fn)
        mats[(ename, key)] = M
        for s in range(M.shape[1]):
            for i in range(M.shape[0]):
                v = M[i, s]
                if np.isnan(v):
                    continue
                rows.append(dict(
                    dataset=DS, study="Makino 2023 Commun Biol", experiment=ename, group=None, subject=f"{ename[-1]}_{s + 1}",
                    person=f"{ename[-1]}_{s + 1}", subject_excluded_by_authors=s in excl, age=None, sex=None, handedness="right",
                    population="healthy", setting="lab", device="planar robot handle with force channel, virtual cursors", session=1,
                    block=None, trial=None, phase="other", time_s=None, task="reaching", unit="participant summary",
                    condition=kf(i + 1), pert_type="visuomotor_rotation", pert_value=None, pert_unit="deg", feedback="altered",
                    excluded_by_authors=s in excl, measure=MEAS[key][0], measure_description=MEAS[key][1] + f"; trial kind {i + 1}",
                    statistic="mean", value=v, value_unit="Nms", n_trials=None, session_context=None))
df = pd.DataFrame(rows)
ck.internal("participants kept: 8 + 16 + 12", df.groupby("experiment").person.nunique().tolist() == [8, 16, 12])
ck.internal("shapes", mats[("experiment 1", "af")].shape == (64, 8) and mats[("experiment 2", "af")].shape == (15, 16)
            and mats[("experiment 3", "af")].shape == (15, 12))
a = mats[("experiment 1", "af")]
for tgt, t in ((7.5, 6.354), (15, 3.564), (30, 6.504), (45, 3.354)):
    ck.published(f"experiment 1 single cursor {tgt} deg, one-sample t (df 7; magnitude, stored sign is negative)",
                 abs(stats.ttest_1samp(a[OFF.index(tgt)], 0).statistic), t, 0.0005, "Results, experiment 1")
nsub = df[~df.subject_excluded_by_authors].groupby("experiment").person.nunique()
ck.published("experiment 1 participants analysed", nsub["experiment 1"], 8, 0, "Methods, Participants")
ck.published("experiment 2 participants analysed", nsub["experiment 2"], 15, 0, "Methods, Participants (16 recruited, 1 excluded)")
ck.published("experiment 3 participants analysed", nsub["experiment 3"], 10, 0, "Methods, Participants (12 recruited, 2 excluded)")
ck.published("experiment 2 participants recruited", df[df.experiment == "experiment 2"].person.nunique(), 16, 0, "Methods, Participants")
ck.published("experiment 3 participants recruited", df[df.experiment == "experiment 3"].person.nunique(), 12, 0, "Methods, Participants")

meta = dict(
    paper=dict(authors="Makino Y, Hayashi T, Nozaki D", year=2023,
               title="Divisively normalized neuronal processing of uncertain visual feedback for visuomotor learning",
               venue="Commun Biol 6(1):1286", doi="10.1038/s42003-023-05578-4"),
    lab="Nozaki (University of Tokyo)", data_url="https://doi.org/10.5281/zenodo.8343357",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Turned the authors' trial-kind x participant matrices into one long table (a row per participant, trial kind and measure), "
             "labelled trial kinds from the comments in the authors' figure scripts, and replaced participant columns with codes. The "
             "force waveform files are not included."),
    how=["One row is the learning response (probe trial) or feedback response (perturbation trial) of one participant for one kind of "
         "visual perturbation, averaged over that participant's trials of the kind by the authors. Unit Nms is taken from the axis label in the "
         "authors' figure code (the paper gives none). Values are as stored; positive visual rotations give negative values.",
         "Trial-kind labels are inferred from the authors' script comments: experiment 1 kinds 1-9 are single cursors at -45, -30, -15, -7.5, 0, "
         "7.5, 15, 30, 45 deg, kinds 10-58 are seven groups of seven double-cursor conditions (second cursor -45 to 45 deg), and kinds 59-64 are "
         "triple cursors (not further described); experiments 2 and 3 have 15 cursor-cloud kinds (mean 0, +-14, +-40 deg; SD 0, 7, 20 deg). "
         "pert_value is empty because kinds differ in size.",
         "Experiment 2 participant 16 and experiment 3 participants 8 and 10 were left out of the authors' analysis (script comments: they "
         "failed to adapt to the single error); they are kept and flagged. Experiment 3 has learning responses only."],
    unit_note="measure of one participant for one kind of perturbation",
    differences=["The paper describes 39 perturbation types in experiment 1; the file has 64 trial kinds (mirrored conditions included), "
                 "so kinds do not match the paper's count. The paper's t statistics are given as positive; the stored values are negative.",
                 "Which participants are the same across the learning- and feedback-response files is assumed from column order."],
)
stdlib.release(DS, "summary", df, meta, ck)
