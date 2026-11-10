"""Cousins et al. (2016), PLoS Biol: cued reactivation (targeted memory reactivation) of a serial reaction time task during sleep, 22 healthy
adults. Per-participant source data of the paper (S1_Data.xlsx, figshare 3101953, Apache 2.0): reaction times and error rates of the cued and
uncued sequences (early and late blocks, learning, retest, improvement), free recall, sleepiness scale, block-by-block learning RTs, age and sex.
Raw (untrusted, read only): raw_extra/Cousins2016_PLoSBiol/S1_Data.xlsx. Run: python3 -I tools/std/convert/Cousins2016_PLoSBiol.py"""
import sys
import warnings
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

warnings.filterwarnings("ignore")
ID = "Cousins2016_PLoSBiol"
wb = openpyxl.load_workbook(stdlib.raw_dir(ID) / "S1_Data.xlsx", read_only=True, data_only=True)
S = lambda n: [r for r in wb[n].iter_rows(values_only=True)]  # noqa: E731

demo = {}
for r in S("Demographics")[2:]:
    if r[0] is None:
        break
    demo[int(r[0])] = (str(r[1]).strip(), r[2])
assert len(demo) == 22

BASE = dict(dataset=ID, study="Cousins et al. 2016 PLoS Biol", experiment="serial reaction time task with cued sleep reactivation", group="",
            subject_excluded_by_authors=False, handedness="", population="healthy", setting="lab", device="", session=1, block="", trial=None,
            phase="other", time_s=None, task="sequence learning", unit="participant summary", condition="", pert_type="none", pert_value=0,
            pert_unit="", feedback="normal", excluded_by_authors=False, statistic="mean", n_trials=None)
rows, cnt = [], {}


def add(sid, measure, desc, value, unit, **kw):
    if value is None or isinstance(value, str):
        return
    cnt[sid] = cnt.get(sid, 0) + 1
    sx, age = demo[sid]
    r = dict(BASE, subject=f"P{sid}", person=f"P{sid}", age=age, sex=sx, trial=cnt[sid], measure=measure, measure_description=desc,
             value=value, value_unit=unit)
    r.update(kw)
    rows.append(r)


def grid(sheet, window, measure, desc, unit):
    s = S(sheet)
    top, sub = [], []
    t = sb = None
    for j in range(len(s[0])):
        t = s[0][j] if s[0][j] is not None else t
        sb = s[1][j] if (len(s[1]) > j and s[1][j] is not None) else sb if s[0][j] is None else None
        top.append(t)
        sub.append(sb if sb is not None else "Sequence")
    for r in s[3:]:
        if not isinstance(r[0], (int, float)):
            break
        for j in range(1, len(s[2])):
            cued = s[2][j]
            if cued not in ("Cued", "Uncued") or top[j] is None:
                continue
            stage = {"LEARN": "learning", "RETEST": "retest", "IMPROVEMENT": "improvement"}[top[j]]
            add(int(r[0]), measure, desc.format(stage=stage), r[j], unit, block=window, condition=f"{cued.lower()} {str(sub[j]).lower()}",
                session=2 if stage == "retest" else 1, phase="learning" if stage == "learning" else "test" if stage == "retest" else "other")


RTD = "Mean reaction time of correct responses, {stage}, of the cued or uncued sequence (or random blocks, or the difference), as in the deposit sheet"
ERD = "Percentage of erroneous responses, {stage}, of the cued or uncued sequence (or random blocks, or the difference), as in the deposit sheet"
grid("RT_Early", "early blocks", "mean_rt", RTD, "ms")
grid("RT_late", "late blocks", "mean_rt", RTD, "ms")
grid("Accuracy_Early", "early blocks", "error_rate", ERD, "percent")
grid("Accuracy_Late", "late blocks", "error_rate", ERD, "percent")
for r in S("Explicit")[2:]:
    if not isinstance(r[0], (int, float)):
        break
    for c, v in zip(["cued", "uncued", "cued minus uncued"], r[1:4]):
        add(int(r[0]), "explicit_recall_score", "Explicit sequence knowledge: free recall of the sequence order written on paper after the retest, score as in the deposit",
            v, "score", condition=c, statistic="value")
for r in S("SSS")[2:]:
    if not isinstance(r[0], (int, float)):
        break
    for c, v, ses in [("learning", r[1], 1), ("retest", r[2], 2)]:
        add(int(r[0]), "stanford_sleepiness", "Stanford Sleepiness Scale before the learning or retest session", v, "scale 1-7", block=c, session=ses, statistic="value")
# block-by-block RTs
s = S("Block RT_Learn")
title = None
nblock_rows = 0
for r in s:
    if r[0] in ("Cued Sequence", "Uncued Sequence"):
        title = r[0].split()[0].lower()
        continue
    if title and r[0] in ("A", "B") and isinstance(r[1], (int, float)):
        nblock_rows += 1
        for j in range(2, len(r)):
            if r[j] is not None:
                add(int(r[1]), "block_mean_rt", "Mean reaction time in the learning block for the cued or uncued sequence, block number in column block",
                    r[j], "ms", block=str(j - 1), condition=f"{title} sequence {r[0]}", phase="learning")

checks = stdlib.Checks()
subs = {r["subject"] for r in rows}
vals = lambda meas, blk, cond, sess_phase: [r["value"] for r in rows if r["measure"] == meas and r["block"] == blk and r["condition"] == cond and r["phase"] == sess_phase]  # noqa: E731
m = lambda v: sum(v) / len(v)  # noqa: E731
checks.published("participants analysed", len(subs), 22, 0, "Materials & Methods, Participants (remaining twenty-two)")
checks.published("male participants", sum(1 for s_, a in demo.values() if s_ == "M"), 14, 0, "Materials & Methods, Participants (22, 14 male)")
checks.published("mean age", m([a for _, a in demo.values()]), 23.5, 0.05, "Materials & Methods, Participants (mean age = 23.5 y)")
checks.published("SD age", (sum((a - 23.5) ** 2 for _, a in demo.values()) / 21) ** 0.5, 4.3, 0.05, "Materials & Methods, Participants (SD 4.3)")
checks.published("late-block RT improvement, cued sequence, ms", m(vals("mean_rt", "late blocks", "cued sequence", "other")), 50.0, 0.05, "Table 1 (Late, Sequence, Improvement, cued)")
checks.published("late-block RT improvement, uncued sequence, ms", m(vals("mean_rt", "late blocks", "uncued sequence", "other")), 46.8, 0.05, "Table 1 (Late, Sequence, Improvement, uncued)")
checks.published("late-block RT improvement, cued random, ms", m(vals("mean_rt", "late blocks", "cued random", "other")), 29.3, 0.05, "Table 1 (Late, Random, Improvement, cued)")
missing = [(k, i) for k in ["mean_rt", "error_rate", "explicit_recall_score", "stanford_sleepiness", "block_mean_rt"] for i in range(1, 23) if not any(r["measure"] == k and r["subject"] == f"P{i}" for r in rows)]
checks.internal("every participant in every sheet used except participant 1 who has no error rates in the deposit", missing == [("error_rate", 1)], str(missing))
paper = dict(authors="Cousins JN, El-Deredy W, Parkes LM, Hennies N, Lewis PA", year=2016,
             title="Cued Reactivation of Motor Learning during Sleep Leads to Overnight Changes in Functional Brain Activity and Connectivity",
             venue="PLoS Biol 14(5):e1002451", doi="10.1371/journal.pbio.1002451")
meta = dict(paper=paper, lab="Lewis (University of Manchester)", data_url="https://doi.org/10.6084/m9.figshare.3101953.v1", license="Apache 2.0",
            license_url="https://www.apache.org/licenses/LICENSE-2.0.html",
            changes="Wide spreadsheet tables turned into long form (one row per participant, measure and condition); mean and SD rows, sleep-stage, spindle, slow-oscillation and alpha tables and the three participants who were tested but not analysed were left out; nothing recomputed.",
            how=["One row is one value for one participant: the mean reaction time (ms) or error rate (percent) of the cued or uncued sequence, or of random blocks, in early or late blocks at "
                 "learning, at retest after a night with sequence-cued sleep, and the improvement (the authors' columns, including their differences); the explicit-recall score; Stanford "
                 "Sleepiness Scale; and the block-by-block mean reaction times of the learning session (cued and uncued sequence A or B).",
                 "Early and late blocks are the first four and last four blocks of each sequence (paper, Table 1 footnote, given for retest; the same windows are assumed for learning, which is inferred). "
                 "Which of the 20 learning blocks are sequence or random blocks is not stated in the sheet, so the block-wise rows keep the deposit's block numbers.",
                 "Age and sex are those of the 22 analysed participants (the deposit's final group). The sheets use subject numbers 1 to 22, replaced here by OpenMotor codes."],
            unit_note="participant, measure and condition")
stdlib.release(ID, "summary", rows, meta, checks)
