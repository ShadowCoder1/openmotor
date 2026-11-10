"""Berlot, Popp & Diedrichsen (2020), eLife 9:e55241: trained and untrained 9-digit finger sequences in four 7T fMRI sessions.

Run from the site folder: python3 -I tools/std/convert/Berlot2020_eLife.py
Raw files (untrusted data, read only): raw_extra/Berlot2020_eLife/sub-NN__ses-NN__func__*_events.tsv (behavioural events of the
BIDS dataset ds002776, CC0, https://openneuro.org/datasets/ds002776), participants.tsv and task-motorseq_events.json.
No imaging data are read. Expected values of published checks come from the paper (Methods, Participants and Task design),
https://elifesciences.org/articles/55241.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Berlot2020_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

parts = pd.read_csv(RAW / "participants.tsv", sep="\t")
ids = parts.participant_id.tolist()
dup = [i for i in set(ids) if ids.count(i) > 1]
# participants.tsv lists sub-08 twice and has no sub-09: the second row sits in sub-09's place (rows are in id order)
fixed, seen = [], set()
for i in ids:
    if i in seen:
        fixed.append("sub-09")
    else:
        fixed.append(i)
        seen.add(i)
parts["participant_id"] = fixed
checks.internal("participants.tsv: one duplicated id (sub-08) and no sub-09; the second one is read as sub-09, ids 01-26 then complete",
                dup == ["sub-08"] and sorted(fixed) == [f"sub-{k:02d}" for k in range(1, 27)])
pinfo = parts.set_index("participant_id")

files = sorted(RAW.glob("sub-*__ses-*__func__*_events.tsv"))
frames = []
for f in files:
    m = re.match(r"sub-(\d+)__ses-(\d+)__func__sub-\1_ses-\2_task-motorseq_run-(\d+)_events\.tsv", f.name)
    assert m, f.name
    x = pd.read_csv(f, sep="\t", na_values=["n/a"])
    x["sub"], x["ses"], x["run"] = int(m[1]), int(m[2]), int(m[3])
    frames.append(x)
a = pd.concat(frames, ignore_index=True)
n = len(a)
checks.internal("26 participants x 4 sessions x 8 runs = 832 event files", len(files) == 832 and a["sub"].nunique() == 26
                and bool((a.groupby("sub").ses.nunique() == 4).all()) and bool((a.groupby(["sub", "ses"]).run.nunique() == 8).all()))
per_run = a.groupby(["sub", "ses", "run"]).size()
checks.internal("828 of the 832 runs hold 72 executions (12 sequences x 3 pairs x 2 repetitions)", int((per_run == 72).sum()) == 828,
                f"other runs hold {sorted(per_run[per_run != 72].tolist())} rows (aborted or restarted runs, all kept)")
checks.internal("events fields as documented: is_error 0/1, points 0 or 3 matching is_error, sequences 1-12, type 1 trained/2 untrained, repetition 1-2",
                bool(a.is_error.isin([0, 1]).all() and ((a.points == 3) == (a.is_error == 0)).all() and a.sequence_num.between(1, 12).all()
                     and a.sequence_type.isin([1, 2]).all() and a.repetition.isin([1, 2]).all()))
checks.internal("sequences 1-6 are the trained ones for every participant and 7-12 the untrained ones",
                bool(((a.sequence_num <= 6) == (a.sequence_type == 1)).all()))
checks.internal("error trials may carry movement time 0 (aborted)", bool((a.movement_time[a.is_error == 0] > 0).all()),
                f"{int(((a.is_error == 1) & (a.movement_time == 0)).sum())} error trials with movement time 0")

sub = a["sub"].to_numpy()
codes = np.array([f"{ID}_S{s:02d}" for s in sub])
pid = [f"sub-{s:02d}" for s in sub]
phase = np.select([a.ses == 1, a.ses.isin([2, 3])], ["baseline", "learning"], "test")
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1",
    "group": ["group" + str(pinfo.loc[p, "group"]) for p in pid], "subject": codes, "person": codes,
    "subject_excluded_by_authors": False,
    "age": [pinfo.loc[p, "age"] for p in pid], "sex": [pinfo.loc[p, "sex"] for p in pid], "handedness": "right",
    "population": "healthy", "setting": "lab", "device": "MRI-compatible five-key keyboard, right hand (7 T scanner)",
    "session": a.ses.to_numpy(), "block": [f"ses{s:02d}_run{r:02d}" for s, r in zip(a.ses, a.run)],
    "trial": a.groupby("sub").cumcount().to_numpy() + 1, "phase": phase, "time_s": "", "task": "sequence learning",
    "unit": "sequence", "condition": np.where(a.sequence_type == 1, "trained", "untrained"), "pert_type": "none", "pert_value": 0,
    "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
    "block_number": a.run.to_numpy(), "block_type": "test", "epoch": "", "stimulus": "", "correct_key": "", "key_pressed": "",
    "correct": (a.is_error == 0).to_numpy(), "rt_ms": "", "rt_correct_ms": "", "element_type": "sequence",
    "sequence": ["S" + str(s) for s in a.sequence_num], "sequence_position": "", "repetition": a.repetition.to_numpy(),
    "triplet_type": "none", "time_in_run_s": a.onset.to_numpy(),
    "movement_time_ms": np.rint(a.movement_time.to_numpy() * 1000), "points": a.points.to_numpy(),
    "pacing": np.where(a.ses == 4, "full speed", "metronome"),
})
for col in [c[1] for c in stdlib.columns_for("keypress")]:
    if col not in t.columns:
        t[col] = ""
checks.internal("every row kept, in the authors' order (participant, session, run, event)", len(t) == n == 59929)

# ---------------------------------------------------------------- published checks (paper, Participants)
pp = pinfo.reset_index()
checks.published("participants analysed", len(pp), 26, 0, "Methods, Participants (27 recruited, 1 excluded, 26 remain)")
checks.published("women among the participants", (pp.sex == "F").sum(), 17, 0, "Methods, Participants")
checks.published("men among the participants", (pp.sex == "M").sum(), 9, 0, "Methods, Participants")
checks.published("mean age (years)", pp.age.mean(), 22.2, 0.05, "Methods, Participants (22.2 years)")
checks.published("SD of age (years)", pp.age.std(), 3.3, 0.05, "Methods, Participants (SD = 3.3)")
checks.published("fMRI sessions per participant", a.groupby("sub").ses.nunique().iloc[0], 4, 0, "Methods, Task design (four 7 T sessions)")
checks.published("runs per session", a.groupby(["sub", "ses"]).run.nunique().iloc[0], 8, 0, "Methods, Task design (eight runs per session)")
checks.published("trained sequences per participant", a[a.sequence_type == 1].sequence_num.nunique(), 6, 0, "Methods, Task design (six trained sequences)")
checks.published("untrained sequences per participant", a[a.sequence_type == 2].sequence_num.nunique(), 6, 0, "Methods, Task design (six untrained sequences)")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))
if not all(c["ok"] for c in checks.items):
    sys.exit(1)

meta = dict(
    study="Berlot et al. 2020",
    paper=dict(authors="Berlot E, Popp NJ, Diedrichsen J", year=2020,
               title="A critical re-evaluation of fMRI signatures of motor sequence learning",
               venue="eLife 9:e55241", doi="10.7554/eLife.55241"),
    lab="Diedrichsen (Western University)",
    data_url="https://openneuro.org/datasets/ds002776",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted to the OpenMotor format from the BIDS events files (behaviour only; the 7 T images are not included); "
             "participant codes replaced; movement time converted from seconds to milliseconds. participants.tsv lists sub-08 "
             "twice and has no sub-09, so the second row was read as sub-09."),
    how=[
        "One row is one execution of a 9-digit finger sequence, pressed with the right hand on an MRI keyboard, in one of four 7 T scanning "
        "sessions (session 1 before training, 2 after one week, 3 and 4 after five weeks; session 4 was done at full speed, sessions 1-3 "
        "were paced to a metronome). Each run has 6 trained and 6 untrained sequences, each executed twice in a row (repetition 1-2).",
        "sequence is the authors' label S1-S12 (S1-S6 trained, S7-S12 untrained, as numbered in the events files); condition is trained or "
        "untrained; group is the participant's assigned group (which half of the 12 sequences was trained). The digit order of each "
        "sequence is not in the deposit.",
        "correct is true when the execution had no error; movement_time_ms is the time to execute the sequence (0 on some aborted error "
        "trials, as the authors recorded); points are the authors' reward points; time_in_run_s is the onset within the scanning run.",
        "The authors included error trials in their main analyses, so no row is marked excluded_by_authors.",
    ],
    unit_note="sequence execution (one 9-digit key-press sequence)",
    differences=[
        "The paper's movement-time and error-rate numbers were not reproduced from the events files and are not used as checks: error rates "
        "of trained and untrained sequences by session are 0.13/0.14, 0.06/0.11, 0.05/0.08 and 0.14/0.13 here (all trials, participant "
        "means), against 0.14/0.15, 0.08/0.14, 0.06/0.09 and 0.14/0.13 in the paper; full-speed trained versus untrained is 335 ms faster "
        "(t(25) = 9.95, correct executions) against 338 ms (t(25) = 9.92); full speed versus baseline is 1.19 s faster against 1.01 s. "
        "The deposit does not say how the paper's behavioural summary was computed, so the cause is unknown.",
    ],
)
stdlib.release(ID, "keypress", t, meta, checks,
               participants=pp.assign(person=[f"{ID}_S{int(p[4:]):02d}" for p in pp.participant_id], group=["group" + str(g) for g in pp.group])[["person", "age", "sex", "group"]],
               participant_columns=[("age", "number", "years", "Age from the deposit's participants.tsv."),
                                    ("sex", "category", "F, M", "Sex from participants.tsv."),
                                    ("group", "text", "", "Assigned group (which half of the 12 sequences was trained).")])
