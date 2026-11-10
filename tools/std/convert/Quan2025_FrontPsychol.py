"""Quan, Wang, Wang & Kang (2025), Front Psychol: reward x voluntary choice in a serial reaction time task, 110 participants.

Run from the site folder: python3 -I tools/std/convert/Quan2025_FrontPsychol.py
Raw file (untrusted data, read only): raw_extra/Quan2025_FrontPsychol/quan_workbook.xlsx from https://osf.io/ejhfr/
(Sheet1 = one row per trial, Sheet2 = one row per participant with age and a five-item questionnaire).
The paper's numbers that can be recomputed from the rows are the sample (110 people in four groups, age mean and SD, overall
and by group) and the questionnaire cell means it prints; the trial counts follow its design (10 sessions of 72 trials).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ID = "Quan2025_FrontPsychol"
RAWFILE = stdlib.raw_dir(ID) / "quan_workbook.xlsx"
SRC1 = "Quan et al. 2025, Participants"
SRC2 = "Quan et al. 2025, Methods (task and sessions)"
SRC3 = "Quan et al. 2025, Results (questionnaire)"

checks = stdlib.Checks()

# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(stdlib.raw_dir(ID) / "SOURCES.json"))}
checks.internal("raw file matches the downloaded hash", stdlib.sha256(RAWFILE) == src["quan_workbook.xlsx"]["sha256"])
raw = pd.read_excel(RAWFILE, sheet_name=0)
par = pd.read_excel(RAWFILE, sheet_name=1)
h = [str(c) for c in raw.columns]
assert h[0].startswith("Participants") and h[1].startswith("Groups") and h[2].startswith("Reward") and h[3].startswith("Choice"), h
assert h[4].startswith("Sequence") and h[5].startswith("Sessions") and h[6].startswith("Accuracy") and h[7].startswith("RT"), h
assert str(par.columns[0]).startswith("Participants") and str(par.columns[1]) == "Group" and str(par.columns[2]) == "Age", list(par.columns)[:3]
raw.columns = ["pid", "grp", "rew", "ch", "seq", "sess", "acc", "rt"]
par.columns = ["pid", "grp", "age", "q1", "q2", "q3", "q4", "q5"]
n = len(raw)

# ---------------------------------------------------------------- structure
GROUPS = {1: "reward_choice", 2: "reward_no-choice", 3: "no-reward_choice", 4: "no-reward_no-choice"}   # the authors' header
SEQ = {"A": "3-4-2-3-1-2-1-4-3-2-4-1", "B": "1-2-1-4-2-3-4-1-3-2-4-3"}                                    # Methods, Task
BLOCK = {1: "s1", 2: "S2", 3: "S3", 4: "S4", 5: "S5", 6: "S6", 7: "s7", 8: "r8", 9: "s9", 10: "s10"}      # the authors' header
BLOCK_TYPE = {1: "test", 2: "training", 3: "training", 4: "training", 5: "training", 6: "training", 7: "test", 8: "random",
              9: "test", 10: "test"}
PHASE = {1: "baseline", 2: "learning", 3: "learning", 4: "learning", 5: "learning", 6: "learning", 7: "test", 8: "test",
         9: "test", 10: "test"}
first_seen = pd.factorize(raw.pid)[0]
checks.internal("each participant's rows are contiguous and in session order",
                bool(((raw.pid != raw.pid.shift()).sum() == raw.pid.nunique()) and
                     (raw.groupby("pid").sess.apply(lambda s: bool((np.diff(s.to_numpy()) >= 0).all()))).all()))
checks.internal("110 participants x 10 sessions x 72 trials = 79,200 rows",
                n == 79200 and raw.pid.nunique() == 110 and (raw.groupby(["pid", "sess"]).size() == 72).all() and raw.sess.nunique() == 10)
checks.internal("group, reward flag, choice flag and sequence are constant within a participant, and the flags follow the group",
                bool((raw.groupby("pid")[["grp", "rew", "ch", "seq"]].nunique() == 1).all().all()) and
                bool((raw.groupby("grp")[["rew", "ch"]].nunique() == 1).all().all()) and
                raw.groupby("grp").rew.first().to_dict() == {1: 1, 2: 1, 3: 0, 4: 0} and raw.groupby("grp").ch.first().to_dict() == {1: 1, 2: 0, 3: 1, 4: 0})
checks.internal("every trial without an RT is an incorrect trial (a missed response)", bool((raw.rt.isna() <= (raw.acc == 0)).all()),
                f"{int(raw.rt.isna().sum())} trials without RT")
checks.internal("RTs lie between 0 and the 1000 ms response window", bool((raw.rt.dropna() >= 0).all() and (raw.rt.dropna() <= 1.0).all()))
checks.internal("sheet 2 has the same 110 participants and groups as sheet 1",
                set(par.pid) == set(raw.pid) and bool((par.set_index("pid").grp == raw.drop_duplicates("pid").set_index("pid").grp).all()))
assert set(raw.sess) == set(BLOCK) and set(raw.seq) == {"A", "B"} and set(raw.acc) == {0, 1}

# ---------------------------------------------------------------- the OpenMotor table
age = raw.pid.map(par.set_index("pid").age)
sess = raw.sess.to_numpy()
trial = raw.groupby("pid").cumcount().to_numpy() + 1
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": raw.grp.map(GROUPS).to_numpy(),
    "subject": "p" + raw.pid.astype(str), "subject_excluded_by_authors": False, "age": age.to_numpy(), "sex": "",
    "handedness": "right", "population": "healthy", "setting": "lab",
    "device": "laptop keyboard (D, F, J, K keys), 14.2-inch screen at 60 Hz, PsychoPy", "session": np.where(sess == 10, 2, 1),
    "block": pd.Series(sess).map(BLOCK).to_numpy(), "trial": trial, "phase": pd.Series(sess).map(PHASE).to_numpy(),
    "time_s": "", "task": "sequence learning", "unit": "keypress", "condition": "",
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": "",
    "block_number": sess, "block_type": pd.Series(sess).map(BLOCK_TYPE).to_numpy(), "epoch": "", "stimulus": "",
    "correct_key": "", "key_pressed": "", "correct": (raw.acc == 1).to_numpy(), "rt_ms": (raw.rt * 1000).to_numpy(),
    "rt_correct_ms": "", "interval_ms": "", "element_type": np.where(sess == 8, "random", "sequence"),
    "sequence": np.where(sess == 8, "random", raw.seq.map(SEQ).to_numpy()), "sequence_position": "", "repetition": "",
    "triplet_type": "", "time_in_block_ms": "",
})
for c in stdlib.columns_for("keypress"):
    assert c[1] == "person" or c[1] in t.columns, c[1]
checks.internal("every row kept, in the authors' order, with the trial count per participant 1..720",
                len(t) == n and int(t.trial.max()) == 720 and bool((t.groupby("subject").trial.apply(lambda s: s.is_monotonic_increasing)).all()))

# ---------------------------------------------------------------- checks against the paper
persons = par.set_index("pid")
checks.published("participants analysed", raw.pid.nunique(), 110, 0, SRC1)
for g, nn in {1: 26, 2: 27, 3: 28, 4: 29}.items():
    checks.published(f"participants in group {GROUPS[g]}", int((persons.grp == g).sum()), nn, 0, SRC1)
checks.published("age, all 110 participants: mean (years)", persons.age.mean(), 20.05, 0.01, SRC1)
checks.published("age, all 110 participants: SD (years)", persons.age.std(), 2.32, 0.01, SRC1)
for g, (m, s) in {1: (20.27, 2.48), 2: (19.70, 1.77), 3: (20.43, 2.91), 4: (None, 2.02)}.items():
    a = persons.age[persons.grp == g]
    if m is not None:
        checks.published(f"age, group {GROUPS[g]}: mean (years)", a.mean(), m, 0.01, SRC1)
    checks.published(f"age, group {GROUPS[g]}: SD (years)", a.std(), s, 0.01, SRC1)
for item, g, val in [("q4", 1, 5.54), ("q4", 2, 4.89), ("q5", 1, 2.85), ("q5", 2, 3.81)]:
    nm = {"q4": "satisfaction (item 4)", "q5": "nervousness (item 5)"}[item]
    checks.published(f"questionnaire {nm}, group {GROUPS[g]}: mean", persons[persons.grp == g][item].mean(), val, 0.006, SRC3)
checks.published("trials per session (12-element sequence x 6 repetitions)", int(raw.groupby(["pid", "sess"]).size().iloc[0]), 72, 0, SRC2)
checks.published("sessions per participant (s1, S2-S6, s7, r8, s9, s10)", raw.sess.nunique(), 10, 0, SRC2)

# ---------------------------------------------------------------- participants table
parts = pd.DataFrame({"person": "p" + persons.index.astype(str), "age": persons.age.to_numpy(), "group": persons.grp.map(GROUPS).to_numpy(),
                      "q1_choice": persons.q1.to_numpy(), "q2_competence": persons.q2.to_numpy(), "q3_interest": persons.q3.to_numpy(),
                      "q4_satisfaction": persons.q4.to_numpy(), "q5_nervous": persons.q5.to_numpy()})
pcols = [("age", "number", "years", "Age in years."),
         ("group", "text", "", "Reward x choice group, in the authors' naming."),
         ("q1_choice", "integer", "1..7", "Questionnaire item 1 (translated from the Chinese): I had the power of choice throughout the experiment."),
         ("q2_competence", "integer", "1..7", "Item 2: I am fully capable of this task."),
         ("q3_interest", "integer", "1..7", "Item 3: I found the experimental task interesting."),
         ("q4_satisfaction", "integer", "1..7", "Item 4: I am satisfied with my performance throughout the experiment."),
         ("q5_nervous", "integer", "1..7", "Item 5: I felt nervous while doing the task.")]

# ---------------------------------------------------------------- release
meta = dict(
    study="Quan et al. 2025",
    paper=dict(authors="Quan Y, Wang J, Wang Y, Kang G", year=2025,
               title="The effect of reward and voluntary choice on the motor learning of serial reaction time task",
               venue="Front Psychol 15:1493434", doi="10.3389/fpsyg.2024.1493434"),
    lab="Kang (Beijing Sport University)",
    data_url="https://osf.io/ejhfr/",
    license="none stated", license_url="",
    changes=("Converted to the OpenMotor format with participant codes replaced; RT converted from seconds to milliseconds; "
             "the sequence label A or B is written out as the 12 cued positions given in the paper; the five questionnaire "
             "items, translated from Chinese, are in participants.csv."),
    how=[
        "One row is one trial (one cued key press) of a serial reaction time task with four locations and keys D, F, J, K. "
        "Each participant has 720 rows: 10 sessions of 72 trials (a 12-element sequence repeated six times), kept in the "
        "authors' order. block is the authors' session label (s1 pre-test, S2-S6 training, s7 immediate post-test, r8 random "
        "sequence, s9 delayed post-test, s10 retention test 24 h later) and block_number counts them 1-10; session is 1 for "
        "s1-s9 and 2 for s10.",
        "group is the authors' reward x choice group (reward or no reward; voluntary choice of a task-irrelevant colour before "
        "each training block, or a colour assigned by the computer). sequence is the practised 12-element sequence (half of each "
        "group practised A = 3-4-2-3-1-2-1-4-3-2-4-1, half B = 1-2-1-4-2-3-4-1-3-2-4-3, positions numbered left to right) and "
        "'random' in r8. Fixed-sequence sessions started at a different position of the sequence, which the file does not give, "
        "so stimulus, sequence_position and repetition are empty.",
        "correct is the authors' accuracy (1 = correct; wrong keys and missed responses are both 0) and rt_ms is their RT times "
        "1000, empty for the trials without an RT, all of them incorrect (no key within the 1000 ms response window). The file "
        "gives the 110 analysed participants only, so subject_excluded_by_authors is false for everyone; the authors' RT analysis "
        "used correct trials within 3 SD of each participant's mean, a rule that could not be reproduced exactly, so "
        "excluded_by_authors is left empty.",
    ],
    unit_note="trial (one cued key press in the serial reaction time task)",
    differences=[
        "Group age: the paper prints a mean of 20.05 for the no-reward/no-choice group; the deposit gives 19.83 (its SD of 2.02 matches, "
        "and 20.05 is the paper's mean for all participants, so it is probably a copy slip). The other three group means match.",
        "The paper reports a mean error rate of 2.5% (SD 2.2); counting every incorrect trial (wrong key or miss) in the deposit "
        "gives 3.5% (SD 1.7 across participants), and leaving out the random session gives 3.3%.",
        "The paper reports that 96.1% of trials remained after removing RTs beyond 3 SD of each participant's mean (correct trials); "
        "the deposit gives 98.5% of correct trials (95.1% of all trials) with a per-participant mean and SD, so the exact rule is unknown.",
    ],
)
stdlib.release(ID, "keypress", t, meta, checks, participants=parts, participant_columns=pcols)
