"""Chan, Wiechmann & Verwey (2022), PsyArXiv: dance-step Discrete Sequence Production task, 24 participants, one row per step.

Raw files (untrusted data, read only) from https://osf.io/9x2te/ (Datasets_Analysis Scripts folder): raw_extra/Chan2022_PsyArXiv/
Concat1.csv (E-Prime step-level data, the released table), DFrdy_all.csv (the same data summed per sequence execution, used for
checks), MVN_AllSubsV0.2.csv (centre-of-mass summaries of two participants; not released). The deposit's R markdown (Methods.rmd)
was read as text. The E-Prime and Xsens files of the example participant are not needed and were not fetched.
Run from the site folder: python3 -I tools/std/convert/Chan2022_PsyArXiv.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Chan2022_PsyArXiv"
B = stdlib.raw_dir(ID)
c = pd.read_csv(B / "Concat1.csv")
d = pd.read_csv(B / "DFrdy_all.csv")
ck = stdlib.Checks()
n = len(c)

# ---------------------------------------------------------------- structure
k = c[["Subject", "Session", "Block"]].to_numpy()
ck.internal("rows are in order of participant, session and trial, each participant contiguous",
            bool(((k[1:, 0] != k[:-1, 0]) | (k[1:, 1] > k[:-1, 1]) | ((k[1:, 1] == k[:-1, 1]) & (k[1:, 2] >= k[:-1, 2]))).all())
            and c.Subject.nunique() == int((np.r_[True, k[1:, 0] != k[:-1, 0]]).sum()))
ck.internal("rows come in groups of six steps (SubTrial 8-13), one group per sequence execution",
            n % 6 == 0 and bool((c.SubTrial.to_numpy().reshape(-1, 6) == np.arange(8, 14)).all()))
ex = np.repeat(np.arange(n // 6), 6)                       # sequence execution index (file order)
first = c.iloc[::6].reset_index(drop=True)
ck.internal("each group of six has one participant, session and trial number",
            bool((c.groupby(ex)[["Subject", "Session", "Block"]].nunique() == 1).all().all()))
ck.internal("feedback.CRESP equals the cued key and ACC equals (response == cued key) on every row",
            bool((c["feedback.CRESP"] == c["correctanswer[SubTrial]"]).all()
                 and ((c["feedback.ACC"] == 1) == (c["feedback.RESP"] == c["feedback.CRESP"])).all()))
seqkey = c.groupby(ex)["correctanswer[SubTrial]"].agg("".join).to_numpy()
seq_acc = c.groupby(ex)["feedback.ACC"].min().to_numpy()
seq_rt = c.groupby(ex).RT.sum().to_numpy()
mm = first[["Subject", "Session", "Block"]].assign(acc=seq_acc, rtsum=seq_rt).merge(
    d[["Subject", "Session", "Block", "TrialAcc", "SumRT", "Trial Nr"]], on=["Subject", "Session", "Block"], how="outer", indicator=True)
ck.internal("the sequence-level file (DFrdy_all) holds exactly these executions, with the same accuracy and summed RT",
            bool((mm._merge == "both").all() and (mm.acc == mm.TrialAcc).all() and (mm.rtsum == mm.SumRT).all()))
rep = (mm.set_index(["Subject", "Session", "Block"])["Trial Nr"]).reindex(pd.MultiIndex.from_frame(first[["Subject", "Session", "Block"]])).to_numpy()
ck.internal("Trial Nr runs 1-48 within each participant and session", bool(
    (pd.Series(rep).groupby([first.Subject.to_numpy(), first.Session.to_numpy()]).apply(lambda x: (x.to_numpy() == np.arange(1, 49)).all())).all()))

pers = c.Subject.map(lambda s: f"{ID}_p{s}")
correct = c["feedback.ACC"] == 1
key_text = c["correctanswer[SubTrial]"]
t = pd.DataFrame(dict(
    dataset=ID, study=ID, experiment="dance-step DSP", group="", subject=pers, subject_excluded_by_authors=False,
    age=np.nan, sex="", handedness="", population="healthy", setting="lab",
    device="dance mat (D-Force) read as keys W, A, S, D through JoyToKey, E-Prime task, 7 Xsens MTw sensors",
    session=1, block=c.Session.astype(str), trial=c.groupby("Subject").cumcount() + 1, phase="learning", time_s=np.nan,
    task="sequence learning", unit="keypress", condition="", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
    # the authors' learning models use accurate sequence executions only (Methods.rmd, df2 / Figs 5-6)
    excluded_by_authors=np.repeat(seq_acc != 1, 6),
    block_number=c.Session, block_type="training", epoch=np.nan, stimulus=np.nan, correct_key=key_text,
    key_pressed=c["feedback.RESP"], correct=correct, rt_ms=c.RT.astype(float), rt_correct_ms=np.nan, interval_ms=np.nan,
    element_type="sequence", sequence=np.repeat([("-".join(s)) for s in seqkey], 6),
    sequence_position=c.SubTrial - 7, repetition=np.repeat(rep, 6), triplet_type="", trial_in_block=c.Block,
))
for col in stdlib.load_module("keypress"):
    if col[1] not in t.columns:
        t[col[1]] = np.nan
ck.internal("every row kept, in the authors' order", len(t) == n)
ck.internal("24 participants x 8 sessions x 48 executions x 6 steps", n == 24 * 8 * 48 * 6)
ck.internal("each participant practised two different six-step sequences",
            bool((first.assign(s=seqkey).groupby("Subject").s.nunique() == 2).all()))
ck.internal("RT >= 0 on every step", bool((c.RT >= 0).all()))

# ---------------------------------------------------------------- published checks
S31 = "Section 3.1, Participants (24 participants took part)"
ck.published("participants", int(c.Subject.nunique()), 24, 0, S31)
ck.published("training blocks per participant", int(c.groupby("Subject").Session.nunique().max()), 8, 0,
             "Section 2.1 (participants practiced over the course of eight training blocks)")
ck.published("sequence executions per participant (48 per block x 8 blocks)", int(first.groupby("Subject").size().mean()), 384, 0,
             "Section 3.2, Fig. 4 caption (raw response time for 384 trials)")
ck.published("steps per sequence", int(c.groupby(ex).size().max()), 6, 0, "Section 2.1 (six stimuli, six steps per trial)")
ck.published("sequence executions per block", int(first.groupby(["Subject", "Session"]).size().max()), 48, 0,
             "Section 2.1 (a block is 48 trials: 24 for each of the two sequences)")
ck.published("distinct counterbalanced sequences across participants", int(len(set(seqkey))), 8, 0,
             "Section 2.1 (both sequences rotated four times, eight counterbalanced sequences)")
ck.published("distinct keys", int(c["correctanswer[SubTrial]"].nunique()), 4, 0, "Section 2.1 (arrows mapped to W, S, D, A)")

diffs = [
    "The paper reports 19 of 24 participants female, a mean age of 20.5 +/- 2.3 years and 87.5% right-footed; the deposit has no age, "
    "sex or footedness, so these columns are empty.",
    "The deposit's score1 and score2 columns (one pair of numbers per participant and block, undocumented; they look like the "
    "block feedback shown by the task) and the E-Prime timing columns were not released.",
    "The deposit's centre-of-mass summaries (MVN_AllSubsV0.2.csv, two participants, one row per sequence) and the Xsens recordings "
    "are not released; this file holds the E-Prime step data of all 24 participants only.",
    "The deposit's Block column is the trial number within the session (1-52, with the 4 no-go trials missing), not the paper's block; "
    "it is trial_in_block here, and block is the authors' Session (1-8, the eight training blocks).",
    "RT is the E-Prime response time of each step; for the first step it starts at the Go signal and for later steps after the "
    "previous step (the deposit does not say; the paper reports step-level response times).",
    "excluded_by_authors marks all six steps of a sequence execution with any wrong step: the authors' linear mixed models use "
    "accurate sequence executions only (Methods.rmd; Figs 5-6 of the preprint). The R markdown has a second model that keeps "
    "wrong steps.",
    "The title of the OSF project is 'Multimodal mobile brain and body imaging for the quantification of dance motor sequence "
    "learning'; the listed paper is the methods preprint (PsyArXiv, 2022, doi 10.31234/osf.io/ypt7n, CC BY 4.0 per Crossref). "
    "The OSF project states no license.",
]
meta = dict(
    study="Chan et al. 2022",
    paper=dict(authors="Chan RW, Wiechmann E, Verwey WB", year=2022,
               title="Motor Sequencing Learning from Dance Step: A whole-body version of the Discrete Sequence Production Task",
               venue="PsyArXiv preprint", doi="10.31234/osf.io/ypt7n"),
    lab="Verwey (University of Twente)", data_url="https://osf.io/9x2te/", license="none stated", license_url="",
    changes=("Converted to the OpenMotor format with participant numbers replaced by codes. The deposit's block-level columns "
             "score1 and score2 and the E-Prime timestamp columns (onset time, onset delay, response time stamp) were dropped; "
             "the sequence text and the execution number were derived from the cued keys and the deposit's trial numbers."),
    how=["One row is one step of a six-step sequence in a dance-step Discrete Sequence Production task: after watching six boxes "
         "light up, participants waited for a Go signal and then reproduced the sequence by stepping with a foot on the matching "
         "area (up, left, down, right = keys W, A, S, D) of a dance mat. Each participant did 8 training blocks (block, the "
         "authors' Session) of 48 sequence executions, 24 of each of two sequences (4 additional no-go trials are not in the data).",
         "stimulus is empty; correct_key and key_pressed are the cued and the registered key. rt_ms is the E-Prime step response time in "
         "ms. sequence gives the six cued keys of the execution joined by dashes, sequence_position is the step (1-6), and repetition is "
         "the execution number within the session (1-48, the deposit's Trial Nr). Rows keep the deposit's order.",
         "correct is the step-level accuracy; excluded_by_authors is true for all six steps of an execution with a wrong step."],
    unit_note="step in a dance-step sequence production trial",
    differences=diffs,
)
stdlib.release(ID, "keypress", t, meta, ck)
for kk in ck.items:
    if kk["kind"] == "published":
        print(kk["what"], kk["computed"], kk["reported"], kk["ok"])
