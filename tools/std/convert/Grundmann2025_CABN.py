"""Grundmann, Arndt, Ebrahimi, Musial, Bode, Schlagenhauf & Endrass (2025), Cogn Affect Behav Neurosci: habit formation through
motor sequence learning (a 9-element sequence on four boxes; learning runs, then tests with skipped elements), 29 participants.

Run from the site folder: python3 -I tools/std/convert/Grundmann2025_CABN.py
Raw files (untrusted data, read only) from https://osf.io/3cksz/ : Code/behavioral/learning.RData and test.RData (the authors' tidy
trial tables `learn` and `test`, plus the run-level tables their analysis script MSLT_behavioral_analyses_OSF.R uses; the script was read
as text, never run). The per-participant .mat files in 'behavioral data' hold the same sessions and are not needed.
Published checks: the authors' own run-level tables (MeanRT, MeanRT_correct, RelErrors, Nmiss; MeanRT and NSoA of the test runs)
are recomputed from the converted rows, and the correlations that the script reports in its comments (MeanRT x NSoA, r = -0.80, -0.83).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ID = "Grundmann2025_CABN"
D = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(D / "SOURCES.json"))}
for f in ("learning.RData", "test.RData"):
    checks.internal(f"{f} matches the downloaded hash", stdlib.sha256(D / f) == src[f]["sha256"])
RL, RT = pyreadr.read_r(str(D / "learning.RData")), pyreadr.read_r(str(D / "test.RData"))
L, T = RL["learn"].copy(), RT["test"].copy()
print("tables", list(RL), list(RT), "| extra column in test:", sorted(set(T.columns) - set(L.columns)))
L["Code"], T["Code"] = L.Code.astype(str), T.Code.astype(str)
L["taskRun"], T["taskRun"] = L.taskRun.astype(str), T.taskRun.astype(str)
checks.internal("29 participants in both tables", L.Code.nunique() == 29 and set(L.Code) == set(T.Code))
checks.internal("Accuracy is 1 exactly when the box pressed is the cued box (misses are Choice 0)",
                bool(((L.Choice == L.Position).astype(int) == L.Accuracy).all() and ((T.Choice == T.Position).astype(int) == T.Accuracy).all()))

# ---------------------------------------------------------------- order: within each session the learning runs, then the test run
RUN_ORDER = {"learning1_fMRI": (1, 1), "learning2_lab": (1, 2), "test1": (1, 3), "learning3_fMRI": (2, 1), "test2": (2, 2)}
L["tab"], T["tab"] = 0, 1
allr = pd.concat([L, T], ignore_index=True)
allr["_sess"] = allr.taskRun.map(lambda r: RUN_ORDER[r][0])
allr["_run"] = allr.taskRun.map(lambda r: RUN_ORDER[r][1])
allr["_seq"] = np.arange(len(allr))
first = {c: i for i, c in enumerate(dict.fromkeys(L.Code))}           # participants in order of first appearance
allr["_p"] = allr.Code.map(first)
allr = allr.sort_values(["_p", "_sess", "_run", "Block", "Trial_block"], kind="stable").reset_index(drop=True)
checks.internal("Session in the tables matches the run names (learning1, learning2, test1 in Session 1; learning3, test2 in Session 2)",
                bool((allr.Session == allr._sess).all()))
key = allr.taskRun + "|" + allr.Block.astype(int).astype(str)
allr["_blk"] = (key != key.groupby([allr.Code, allr.Session]).shift()).astype(int).groupby([allr.Code, allr.Session]).cumsum()
allr["_trial"] = allr.groupby("Code").cumcount() + 1
skip = (allr.tab == 1) & (allr.lastSkipPos > 0)

n = len(allr)
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": "", "subject": allr.Code,
    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
    "device": "four response keys (one per box), fMRI scanner or lab computer", "session": allr.Session,
    "block": allr.taskRun, "trial": allr._trial, "phase": np.where(allr.tab == 0, "learning", "test"), "time_s": "",
    "task": "sequence learning", "unit": "keypress", "condition": np.where(skip, "skip_box_" + allr.lastSkipBox.astype(str), ""),
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": "",
    "block_number": allr._blk, "block_type": np.where(allr.tab == 0, "training", "test"), "epoch": "",
    "stimulus": allr.Position, "correct_key": allr.Position.astype(str),
    "key_pressed": np.where(allr.Choice > 0, allr.Choice.astype(str), ""), "correct": allr.Accuracy == 1, "rt_ms": allr.RT,
    "rt_correct_ms": "", "interval_ms": "", "element_type": "sequence", "sequence": "", "sequence_position": allr.SequenceElement,
    "repetition": allr.SequenceIteration, "triplet_type": "", "time_in_block_ms": "",
})
for c in stdlib.columns_for("keypress"):
    assert c[1] == "person" or c[1] in t.columns, c[1]
checks.internal("every trial of both tables kept", len(t) == len(L) + len(T) == 134433 + 87145)
checks.internal("the 27 blocks of each run are numbered 1-27 within every participant and run",
                bool((allr.groupby(["Code", "taskRun"]).Block.nunique() == 27).all()))
print("block_number max per session", allr.groupby("Session")._blk.max().to_dict())

# ---------------------------------------------------------------- checks against the authors' run-level tables and script comments
def block_median_mean(d):
    """Mean over the 27 blocks of the block's median RT (the authors' MeanRT), for all trials."""
    return d.groupby(["Code", "taskRun", "Block"], observed=True).RT.median().groupby(["Code", "taskRun"], observed=True).mean()


def run_table(d):
    g = d.groupby(["Code", "taskRun"], observed=True)
    return pd.DataFrame({"MeanRT": block_median_mean(d), "MeanRT_correct": block_median_mean(d[d.Accuracy == 1]),
                         "RelErrors": g.Accuracy.apply(lambda a: (a == 0).mean()), "Nmiss": g.Choice.apply(lambda a: (a == 0).sum())})


SRC = "authors' learn_subjects_run table in learning.RData (used by MSLT_behavioral_analyses_OSF.R)"
mine = run_table(L)
theirs = RL["learn_subjects_run"].assign(Code=lambda d: d.Code.astype(str), taskRun=lambda d: d.taskRun.astype(str)).set_index(["Code", "taskRun"])
for m in ("MeanRT", "MeanRT_correct", "RelErrors", "Nmiss"):
    dif = (mine[m] - theirs[m].reindex(mine.index)).abs()
    print(m, "max abs diff", float(dif.max()), "n", len(dif))
    checks.published(f"{m} per participant and learning run (87 values): largest difference to the authors' table", float(dif.max()), 0.0, 1e-6, SRC)
checks.published("learning runs per participant", L.groupby("Code").taskRun.nunique().median(), 3, 0, "analysis script: 'LME RTs run (3) x block (27)'")
checks.published("blocks per learning run", L.groupby(["Code", "taskRun"], observed=True).Block.nunique().median(), 27, 0, "analysis script: 'LME RTs run (3) x block (27)'")
tr = RT["test_subjects_run"].assign(Code=lambda d: d.Code.astype(str), taskRun=lambda d: d.taskRun.astype(str)).set_index(["Code", "taskRun"])
print("test_subjects_run columns", list(tr.columns))
Ts = T[T.lastSkipPos > 0]
v_err = Ts.groupby(["Code", "taskRun"], observed=True).Accuracy.apply(lambda a: (a == 0).sum())
v_box = Ts.groupby(["Code", "taskRun"], observed=True).apply(lambda g: (g.Choice == g.lastSkipBox).sum(), include_groups=False)
mrt = block_median_mean(T)
dif = lambda x, y: float((x - y.reindex(x.index)).abs().max())
print("test MeanRT diff", dif(mrt, tr.MeanRT), "| errors-on-skip vs NSoA", dif(v_err, tr.NSoA), "vs Nskiperrors", dif(v_err, tr.Nskiperrors),
      "| box-chosen vs NSoA", dif(v_box, tr.NSoA), "| Nskiptrials", dif(Ts.groupby(["Code", "taskRun"], observed=True).size(), tr.Nskiptrials))
sk = t[t.condition != ""].assign(slip=lambda d: d.key_pressed == d.condition.str[-1])      # a press of the skipped element's box
nsoa = sk.groupby(["subject", "block"]).slip.sum()
nsoa.index.names = ["Code", "taskRun"]
checks.published("mean RT of the test runs (58 values): largest difference to the authors' test_subjects_run", dif(mrt, tr.MeanRT), 0.0, 1e-6, "test_subjects_run in test.RData")
checks.published("action slips (NSoA) of the test runs (58 values): largest difference to the authors' test_subjects_run", dif(nsoa, tr.NSoA), 0.0, 1e-6, "test_subjects_run in test.RData")
for sess, run, rep in ((1, "test1", -0.80), (2, "test2", -0.83)):
    a, b = mrt.xs(run, level="taskRun"), nsoa.xs(run, level="taskRun")
    r = np.corrcoef(a.loc[b.index], b)[0, 1]
    checks.published(f"correlation of mean RT and number of action slips across participants, Session {sess}", r, rep, 0.01,
                     "analysis script comment: cor.test(MeanRT, NSoA) in test_subjects_run")

# ---------------------------------------------------------------- release
meta = dict(
    study="Grundmann et al. 2025",
    paper=dict(authors="Grundmann CC, Arndt VA, Ebrahimi C, Musial MPM, Bode EL, Schlagenhauf F, Endrass T", year=2025,
               title="Studying human habit formation through motor sequence learning",
               venue="Cogn Affect Behav Neurosci 25(4):941-951", doi="10.3758/s13415-025-01300-5"),
    lab="Endrass (TU Dresden) / Schlagenhauf (Charite)",
    data_url="https://osf.io/3cksz/",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced. The authors' tidy trial tables `learn` and `test` are "
             "merged (a `test` row has phase test) and ordered by participant, session, run, block and trial. Their columns "
             "ResponseLimit, ResponseBeforeTarget, lastSkipPos, lastSkipBox, Choice_FB, RT_FB, RTdelta, calcRespLimit and error "
             "are not in the format (lastSkipPos > 0 and lastSkipBox are written into condition); the per-participant .mat files, the fMRI maps and "
             "the code are not converted."),
    how=[
        "One row is one trial (one cued box and the response to it) of a sequence task with four boxes and a 9-element sequence "
        "(stimulus 1-4; sequence_position is the authors' SequenceElement 1-9 and repetition their SequenceIteration). Each "
        "participant has two sessions: Session 1 = learning1_fMRI, learning2_lab and test1, Session 2 = learning3_fMRI and test2, "
        "each run having 27 blocks; block is the run name and block_number counts the blocks within the session.",
        "correct is the authors' Accuracy (1 when the box pressed is the cued box; a miss counts as wrong and has no "
        "key_pressed or rt_ms) and rt_ms is their RT in ms, with the few negative values of presses before the target kept "
        "as given. In the test runs, condition skip_box_N marks the trials with lastSkipPos > 0, which the authors count as skip "
        "trials, N being their lastSkipBox, the box of the skipped element; pressing box N on such a trial is the action slip "
        "(NSoA) of their analysis (a wrong press of another box is an error but no slip).",
        "The deposit's two tables have no clock, so the order of a session's test run after its learning runs follows the run "
        "names, and trial counts the participant's rows in that order.",
    ],
    unit_note="trial (one cued box and its response in the sequence task)",
)
stdlib.release(ID, "keypress", t, meta, checks, participants=None)
