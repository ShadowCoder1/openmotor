"""Maceira-Elvira, Timmermann, Popa, Schmid, Krakauer, Morishita, Wessel & Hummel (2022), Sci Adv: a nine-element finger
sequence-tapping task trained for five days (with follow-ups on days 10 and 60) by young, middle-aged and older adults,
without stimulation (Experiment 1) or with verum or placebo anodal tDCS over the motor cortex (Experiment 2). One row per
key press.

Run from the site folder: python3 -I tools/std/convert/MaceiraElvira2022_SciAdv.py
Raw files (untrusted data, read only, never extracted) in raw_extra/MaceiraElvira2022_SciAdv/:
  zenodo6490799.zip (https://doi.org/10.5281/zenodo.6490799, CC BY 4.0):
    Data/Behavioral/MS_Data_All_Participants.csv  one row per key press (participantID, age, ageGroup, day, blockType,
                                                  blockNr, randomBlock, sequenceNr, pressNr, time, deltaTime, answer, error)
    Data/Behavioral/Sequence_key.txt              the baseline, training and five catch sequences
    Data/Metadata/Unblinded_stim.csv              per participant: age, age group, stimulation code
    Data/Elecrophysiology/Python_AP2_SICIrest_*   SICI ratios labelled Verum/Sham (used only to decode the stimulation codes)
  code_zenodo6490843.zip (https://doi.org/10.5281/zenodo.6490843, the authors' code; its User_Guide.docx and
    Chunk_extraction.py were read as text, never run): blockType 0 = baseline, 1 = training, 2 = follow-up; randomBlock 1 =
    catch block; the authors' chunk analysis uses days 1-5, 10 and 60.
The published checks reproduce the paper's participant counts per experiment, age group and stimulation, and its mean ages.
"""
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "MaceiraElvira2022_SciAdv"
RAW = stdlib.raw_dir(ID)
ZIP = RAW / "zenodo6490799.zip"
P = "10.5281zenodo.6490799/Data/"
AGE_GROUP = {0: "young", 1: "middle-aged", 2: "older"}
STIM = {1: "verum", 2: "placebo"}          # 3 = no stimulation (Experiment 1); decoded below from the SICI file's labels
TYPE = {0: "baseline", 1: "training", 2: "follow-up"}  # the authors' User_Guide.docx

checks = stdlib.Checks()

# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("zenodo6490799.zip matches the downloaded hash", stdlib.sha256(ZIP) == src["zenodo6490799.zip"]["sha256"])
z = zipfile.ZipFile(ZIP)
raw = pd.read_csv(io.BytesIO(z.read(P + "Behavioral/MS_Data_All_Participants.csv")))
assert list(raw.columns) == ["participantID", "age", "ageGroup", "day", "blockType", "blockNr", "randomBlock", "sequenceNr",
                             "pressNr", "time", "deltaTime", "answer", "error"], list(raw.columns)
meta_p = pd.read_csv(io.BytesIO(z.read(P + "Metadata/Unblinded_stim.csv")), skipinitialspace=True)
meta_p.columns = [c.strip() for c in meta_p.columns]
sici = pd.read_csv(io.BytesIO(z.read(P + "Elecrophysiology/Python_AP2_SICIrest_14_10_2021.txt")), sep="\t")
keytxt = z.read(P + "Behavioral/Sequence_key.txt").decode("utf-8", "replace").replace("\r", "")
n = len(raw)

# sequences from the authors' key file: a label line ("Baseline sequence:", "Training sequence:", "Catch block k:") then digits
SEQ, label = {}, None
for line in keytxt.split("\n"):
    s = line.strip()
    if s.endswith(":"):
        label = s[:-1].lower()
    elif label and re.fullmatch(r"[2-5]( [2-5]){8}", s):
        SEQ[label] = [int(x) for x in s.split()]
        label = None
assert sorted(SEQ) == ["baseline sequence", "catch block 1", "catch block 2", "catch block 3", "catch block 4", "catch block 5",
                       "training sequence"], sorted(SEQ)

# ---------------------------------------------------------------- participants
pid = raw.participantID.to_numpy()
order = list(dict.fromkeys(pid))
runs = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
checks.internal("each participant's rows are contiguous in the authors' file", len(runs) == len(order) == 114, f"{len(order)} participants")
per = raw.groupby("participantID", sort=False).agg(age=("age", "nunique"), ag=("ageGroup", "nunique"))
checks.internal("age and age group are constant within each participant", bool((per.age == 1).all() and (per.ag == 1).all()))
first_age = raw.groupby("participantID", sort=False)[["age", "ageGroup"]].first()
mp = meta_p.set_index("Code")
missing = [p for p in order if p not in mp.index]
checks.internal("every participant of the stimulation list has key presses; one participant with key presses is not in the "
                "list (the participant the authors excluded, Methods: one middle-aged participant with arthritis)",
                len(mp) == 113 and set(mp.index) <= set(order) and len(missing) == 1
                and first_age.loc[missing[0], "ageGroup"] == 1)
EXCL = set(missing)
# stimulation codes: the SICI file labels each Experiment 2 participant Verum or Sham
lab = sici.dropna(subset=["ID"]).groupby("ID").S.agg(lambda s: set(s))
code_of = mp.stim.to_dict()
dec = Counter((code_of.get(int(i)), next(iter(v))) for i, v in lab.items() if len(v) == 1)
checks.internal("stimulation code 1 = verum and 2 = placebo (sham): every participant in the authors' SICI file labelled "
                "Verum has code 1 and every one labelled Sham has code 2", set(dec) == {(1, "Verum"), (2, "Sham")},
                str(dict(dec)))
# experiment: code 3 (no stimulation) = Experiment 1; 1 and 2 = Experiment 2. The excluded participant is not in the list; their
# participant code has the digit that every Experiment 2 code has (and no Experiment 1 code), so they are put in Experiment 2.
digit = {p: str(p)[2] for p in order}
exp_digit = {e: {digit[p] for p in mp.index if (mp.stim[p] == 3) == (e == 1)} for e in (1, 2)}
checks.internal("participant codes of the two experiments differ in one digit (used only to place the excluded participant)",
                len(exp_digit[1]) == 1 and len(exp_digit[2]) == 1 and exp_digit[1] != exp_digit[2]
                and all(digit[p] in exp_digit[2] for p in EXCL))
exp_of = {p: (1 if mp.stim[p] == 3 else 2) if p in mp.index else 2 for p in order}
# age group: the behavioral file's ageGroup gives the paper's group sizes (one participant aged 65/66 is 'older' there and
# 'middle-aged' in the stimulation list); ages: the stimulation list's ages give the paper's mean ages (three differ from the
# behavioral file's by 1, 1 and 7 years). The excluded participant's age comes from the behavioral file.
ag_of = first_age.ageGroup.to_dict()
age_of = {p: (int(mp.age[p]) if p in mp.index else int(first_age.age[p])) for p in order}
n_age_diff = sum(1 for p in mp.index if int(mp.age[p]) != int(first_age.age[p]))
n_ag_diff = sum(1 for p in mp.index if int(mp.age_group[p]) != ag_of[p] + 1)
checks.internal("the behavioral file and the stimulation list agree on age for all but 3 participants and on age group for "
                "all but 1", n_age_diff == 3 and n_ag_diff == 1, f"{n_age_diff} ages, {n_ag_diff} age group")


def group_name(p):
    g = AGE_GROUP[ag_of[p]]
    if p in EXCL:
        return f"{g}, stimulation not shared"
    s = int(mp.stim[p])
    return g if s == 3 else f"{g}, {STIM[s]}"


# ---------------------------------------------------------------- blocks, runs and sequences
day = raw.day.to_numpy()
bt = raw.blockType.to_numpy()
bn = raw.blockNr.to_numpy()
rb = raw.randomBlock.to_numpy()
ans = raw.answer.to_numpy()
err = raw.error.to_numpy()
t = raw.time.to_numpy()
dt = raw.deltaTime.to_numpy()
pn = raw.pressNr.to_numpy()
sn = raw.sequenceNr.to_numpy()
bkey = pd.Series(list(zip(pid, day, bt, bn)))
k = pd.factorize(bkey)[0]
keystart = np.r_[True, k[1:] != k[:-1]]
checks.internal("block type and catch flag are constant within each block label (participant, day, block type, block "
                "number)", bool((pd.Series(rb).groupby(k).nunique() == 1).all()))
checks.internal("days run 1, 2, 3, 4, 5, 10, (20), 60 in file order for every participant",
                all((np.diff(g.day.to_numpy()) >= 0).all() for _, g in raw.groupby("participantID", sort=False)))

# block layout per (participant, day): order of first appearance of each block label
lay = {}
for (p, d, b_t, b_n, r_b) in pd.DataFrame({"p": pid, "d": day, "t": bt, "n": bn, "r": rb})[keystart].drop_duplicates(
        ["p", "d", "t", "n"]).itertuples(index=False):
    lay.setdefault((p, d), []).append((b_t, b_n, r_b))
CATCH_AT = {1: 4, 2: 3, 3: 5, 4: 3, 5: 4}
lay_ok = Counter()
for (p, d), L in lay.items():
    if d == 1:
        ok = L[0] == (0, 1, 0) and [x[:2] for x in L[1:]] == [(1, b) for b in range(1, 8)]
    elif d in (2, 3, 4, 5):
        ok = [x[:2] for x in L] == [(1, b) for b in range(1, 8)]
    else:
        ok = [x[:2] for x in L] == [(2, b) for b in range(1, 4)] and d in (10, 20, 60)
    catch_ok = d not in CATCH_AT or [x[1] for x in L if x[2] == 1] == [CATCH_AT[d]]
    lay_ok[(ok, catch_ok)] += 1
checks.internal("each training day has seven blocks (day 1 after one baseline block) with one catch block, at block 4, 3, 5, "
                "3 and 4 on days 1-5; each follow-up day has three blocks (Methods, Motor task); one participant has no "
                "catch block on days 1 and 2", lay_ok[(True, True)] == len(lay) - 2 and lay_ok[(True, False)] == 2,
                str(dict(lay_ok)))
days_of = raw.groupby("participantID", sort=False).day.agg(lambda s: tuple(sorted(set(s))))
checks.internal("visits: every participant has days 1-5 and 10, all but one day 60, and 49 participants also day 20 (not "
                "reported in the paper)", Counter(days_of) == Counter({(1, 2, 3, 4, 5, 10, 60): 64,
                                                                       (1, 2, 3, 4, 5, 10, 20, 60): 49,
                                                                       (1, 2, 3, 4, 5, 10): 1}), str(Counter(days_of)))

# runs: a block label can appear more than once (a single press logged before the block restarts at press 1, or a short
# extra run of 4-19 presses filed later under the same label); rows stay in the authors' order and keep their label
reset = np.r_[False, (pn[1:] <= pn[:-1]) & ~keystart[1:]]
after_ph = np.r_[False, keystart[:-1] & (ans[:-1] == -1)]
newrun = keystart | (reset & ~after_ph)
run = np.cumsum(newrun) - 1
labels_in_runs = pd.Series(k[newrun]).value_counts()
checks.internal("block labels in more than one run of rows", True,
                f"{int((labels_in_runs > 1).sum())} of {len(labels_in_runs)} labels; {int(newrun.sum())} runs")

# time unit: the authors' time stamps are in 0.1 ms (90-s blocks end near 900000)
run_end = pd.Series(t).groupby(run).max()
run_len = pd.Series(t).groupby(run).size()
full = run_end[run_len >= 40]
checks.internal("time is in units of 0.1 ms: the last press of 99% of block runs falls 83-91 s after the block start "
                "(90-s blocks, Methods)", ((full >= 830000) & (full <= 910000)).mean() >= 0.99,
                f"median {np.median(full) / 1e4:.2f} s, max {full.max() / 1e4:.2f} s over {len(full)} runs")

# placeholders: answer -1 is a row the authors inserted for a sequence element without a key press; it sits at the midpoint
# between the presses around it, and the press after it gets the same interval
ph = ans == -1
nxt = np.r_[ph[1:], False]
inner = ph & ~newrun & ~np.r_[keystart[1:], True]
checks.internal("answer -1 rows (placeholders) are always errors, never two in a row, and the next row repeats their interval",
                bool((err[ph] == 1).all() and not (ph & np.r_[ph[1:], False]).any() and (dt[np.flatnonzero(ph) + 1] == dt[ph]).all()))
mid = (t[inner] - t[np.flatnonzero(inner) - 1] == dt[inner]) & (np.abs(t[np.flatnonzero(inner) + 1] - t[inner] - dt[inner]) <= 1)
checks.internal("placeholders inside a block split the interval between the presses around them evenly (within 1 unit)",
                mid.mean() >= 0.995, f"{mid.sum()} of {inner.sum()} inside a block; {ph.sum()} placeholders in all")
diff_t = np.r_[t[0], np.diff(t)]
same = (dt == diff_t) | (newrun & (dt == t))
explained = same | np.r_[False, ph[:-1]] | ph
checks.internal("deltaTime = time minus the previous press's time (= time at a block's first press), except next to "
                "placeholders", (~explained).sum() <= 12, f"{(~explained).sum()} other rows (block restarts, a clock jump)")
checks.internal("time decreases within a run on a few rows (kept as the authors' values)", True,
                f"{int((np.r_[False, np.diff(t) < 0] & ~newrun).sum())} rows")

# sequences: the authors' segmentation (block label + sequenceNr, contiguous rows); positions from the key file
skey = pd.factorize(pd.Series(list(zip(k, sn))))[0]
segstart = np.r_[True, skey[1:] != skey[:-1]]
seg = np.cumsum(segstart) - 1
seg_size = np.bincount(seg)[seg]
pos = np.arange(n) - np.flatnonzero(segstart)[seg]
seg_end_row = np.r_[np.flatnonzero(segstart)[1:] - 1, n - 1]      # last row of each segment
run_end_row = np.r_[run[1:] != run[:-1], True]                      # row is the last of its run
seg_last = run_end_row[seg_end_row][seg]                             # the segment ends its run (cut off by the block end)
seq_name = np.where(bt == 0, "baseline sequence", np.where(rb == 1, np.array([f"catch block {d}" for d in day]), "training sequence"))
seq_arr = {nm: np.array(v) for nm, v in SEQ.items()}
for d in CATCH_AT:
    m = (rb == 1) & (day == d) & (seg_size == 9)
    agree = {c: float(np.mean(err[m] == (ans[m] != seq_arr[c][pos[m]]))) for c in SEQ}
    checks.internal(f"day {d} catch blocks use 'Catch block {d}' of the key file (its elements fit the authors' error flags "
                    "best)", max(agree, key=agree.get) == f"catch block {d}", f"{agree[f'catch block {d}']:.3f}")
filled = (seg_size == 9) | (seg_last & (seg_size < 9))
expected = np.full(n, -9)
for nm in SEQ:
    m = filled & (seq_name == nm)
    expected[m] = seq_arr[nm][pos[m]]
agree9 = err[filled] == (ans[filled] != expected[filled])
checks.internal("in complete nine-press sequences and the cut-off last sequence of a block, the authors' error flag = (key "
                "pressed != the sequence element at that position) on at least 99.9% of presses; elsewhere (sequences with "
                "an extra press) the authors aligned presses to elements themselves, so no element is given",
                agree9.mean() >= 0.999, f"{agree9.mean():.5f} of {filled.sum()} presses ({n - filled.sum()} presses without "
                                        f"an element)")

# ---------------------------------------------------------------- the OpenMotor table
session_of = {}
sess = np.zeros(n, dtype=int)
for i, (p, d) in enumerate(zip(pid, day)):
    session_of.setdefault(p, {})
    if d not in session_of[p]:
        session_of[p][d] = len(session_of[p]) + 1
    sess[i] = session_of[p][d]
blocknum = {}
for (p, d), L in lay.items():
    for j, (b_t, b_n, _) in enumerate(L, 1):
        blocknum[(p, d, b_t, b_n)] = j
block_number = np.array([blocknum[x] for x in zip(pid, day, bt, bn)])
block_label = np.array([f"day {d} {TYPE[b_t]} block {b_n}" + (" (catch)" if r == 1 else "")
                        for d, b_t, b_n, r in zip(day, bt, bn, rb)])
phase = np.where(bt == 0, "baseline", np.where(bt == 2, "test", np.where(rb == 1, "transfer", "learning")))
block_type = np.where(bt == 0, "practice", np.where((bt == 2) | (rb == 1), "test", "training"))
seq_text = {nm: "-".join(str(x) for x in v) for nm, v in SEQ.items()}
excluded_row = np.isin(pid, list(EXCL)) | (day == 20)
trial = raw.groupby("participantID", sort=False).cumcount().to_numpy() + 1

tab = pd.DataFrame({
    "dataset": ID, "study": ID,
    "experiment": [f"{ID}_Exp{exp_of[p]}" for p in pid],
    "group": pd.Series(pid).map({p: group_name(p) for p in order}).to_numpy(),
    "subject": pid.astype(str), "subject_excluded_by_authors": np.isin(pid, list(EXCL)),
    "age": pd.Series(pid).map(age_of).to_numpy(), "sex": "", "handedness": "right", "population": "healthy",
    "setting": "lab", "device": "four-button box (buttons 2-5 under the index to little finger), left hand",
    "session": sess, "block": block_label, "trial": trial, "phase": phase, "time_s": "",
    "task": "finger tapping and timing", "unit": "keypress", "condition": "",
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": excluded_row,
    "block_number": block_number, "block_type": block_type, "epoch": "",
    "stimulus": pd.Series(expected, dtype=object).where(filled, None).to_numpy(),
    "correct_key": pd.Series(expected.astype(str), dtype=object).where(filled, None).to_numpy(),
    "key_pressed": ans.astype(str), "correct": err == 0, "rt_ms": "", "rt_correct_ms": "",
    "interval_ms": dt / 10.0, "element_type": "sequence",
    "sequence": pd.Series(seq_name).map(seq_text).to_numpy(),
    "sequence_position": pd.Series(pos + 1, dtype=object).where(filled, None).to_numpy(),
    "repetition": sn, "triplet_type": "", "time_in_block_ms": t / 10.0,
})
checks.internal("every row kept, in the authors' order; trial counts each participant's rows from 1",
                len(tab) == n and bool((tab.groupby("subject", sort=False).trial.apply(lambda s: (np.diff(s) == 1).all())).all()))
checks.internal("ranges: keys pressed 2-5 or the placeholder -1; intervals 0-35 s apart from one clock jump; times 0-100 s "
                "into the block", set(np.unique(ans)) == {-1, 2, 3, 4, 5} and (dt >= -120025).all() and dt.max() <= 350000
                and t.min() >= 0 and t.max() <= 1000000,
                f"interval {dt.min() / 10:.1f} to {dt.max() / 10:.1f} ms; time {t.min() / 10:.1f}-{t.max() / 10:.1f} ms")
checks.internal("excluded_by_authors rows: the excluded participant and the day-20 visits", True,
                f"{int(excluded_row.sum())} of {n} rows ({100 * excluded_row.mean():.1f}%)")

# ---------------------------------------------------------------- published checks, from the converted rows
pp = tab.drop_duplicates("subject")[["subject", "experiment", "group", "age", "subject_excluded_by_authors"]]
an = pp[~pp.subject_excluded_by_authors]
an = an.assign(ag=an.group.str.split(",").str[0], exp=an.experiment.str[-1].astype(int),
               stim=np.where(an.group.str.contains("verum"), "verum", np.where(an.group.str.contains("placebo"), "placebo", "")))
SRC_M = "Methods, Participants"
checks.published("participants analysed (113 volunteers, after excluding one)", len(an), 113, 0, SRC_M)
checks.published("participants the authors excluded", int(pp.subject_excluded_by_authors.sum()), 1, 0, SRC_M)
for g, nn, mean in (("young", 41, 24.5), ("middle-aged", 34, 57.7), ("older", 38, 72.3)):
    sub = an[an.ag == g]
    checks.published(f"{g} participants, both experiments", len(sub), nn, 0, SRC_M)
    checks.published(f"{g} mean age, both experiments (the paper's value matches the rows truncated to one decimal; "
                     "tolerance one unit of the last digit)", sub.age.mean(), mean, 0.1, SRC_M)
SRC1 = "Results, 'Age-related behavioral differences in the execution and practice of a sequence-tapping task'"
SRC2 = "Results, 'Motor training in combination with atDCS in an aging population'"
checks.published("Experiment 1 participants", int((an.exp == 1).sum()), 52, 0, SRC1)
checks.published("Experiment 2 participants", int((an.exp == 2).sum()), 61, 0, SRC2)
for e, src_, vals in ((1, SRC1, (("young", 22, 24.7), ("middle-aged", 15, 57.4), ("older", 15, 74.1))),
                      (2, SRC2, (("young", 19, 24.4), ("middle-aged", 19, 58.0), ("older", 23, 71.2)))):
    for g, nn, mean in vals:
        sub = an[(an.exp == e) & (an.ag == g)]
        checks.published(f"Experiment {e} {g} participants", len(sub), nn, 0, src_)
        checks.published(f"Experiment {e} {g} mean age (tolerance one unit of the last digit)", sub.age.mean(), mean, 0.1, src_)
for s, vals in (("verum", (10, 9, 14)), ("placebo", (9, 10, 9))):
    for g, nn in zip(("young", "middle-aged", "older"), vals):
        checks.published(f"Experiment 2 {g} participants with {s} stimulation", int(((an.exp == 2) & (an.ag == g) & (an.stim == s)).sum()),
                         nn, 0, SRC2)
# design numbers from the rows
tr = tab[(tab.block_type == "training")]
bpd = tab[tab.phase.isin(["learning", "transfer"])].groupby(["subject", "session"]).block_number.nunique()
checks.published("blocks per training day (six training blocks and one catch block)", float(bpd.mode()[0]), 7, 0,
                 "Methods, Motor task")
fu = tab[tab.phase == "test"].groupby(["subject", "session"]).block_number.nunique()
checks.published("blocks per follow-up day", float(fu.mode()[0]), 3, 0, "Methods, Motor task")
checks.published("elements per sequence", len(SEQ["training sequence"]), 9, 0, "Methods, Motor task (a nine-digit sequence)")

for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    else:
        print(f"  [{'ok' if ch['ok'] else 'FAILED'}] {ch['what']} {ch.get('detail', '')}")

# ---------------------------------------------------------------- release
meta = dict(
    study="Maceira-Elvira et al. 2022",
    paper=dict(authors="Maceira-Elvira P, Timmermann JE, Popa T, Schmid AC, Krakauer JW, Morishita T, Wessel MJ, Hummel FC",
               year=2022, title="Dissecting motor skill acquisition: Spatial coordinates take precedence",
               venue="Sci Adv 8(29):eabo3505", doi="10.1126/sciadv.abo3505"),
    lab="Hummel (EPFL)",
    data_url="https://doi.org/10.5281/zenodo.6490799",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced; time stamps and intervals converted from the "
             "authors' 0.1-ms units to ms; ages taken from the authors' stimulation list (Unblinded_stim.csv), which "
             "reproduces the paper's mean ages; the cued element of each press added from the authors' sequence key; the "
             "visual analogue scales of attention and fatigue and the SICI ratios (TMS) were not converted."),
    how=[
        "One row is one key press in a 90-s block of the sequence-tapping task: participants typed a nine-digit sequence "
        "shown on a screen (sequence) as fast and as accurately as possible on four buttons labelled 2-5 (index to little "
        "finger of the left hand); days 1-5 had seven blocks, six of the training sequence and one catch block with a new "
        "sequence (phase transfer), day 1 began with a baseline block of another sequence, and the follow-up visits (days 10 "
        "and 60, and day 20 for 49 participants) had three blocks of the training sequence (phase test).",
        "key_pressed is the authors' answer and correct their error flag; -1 is a placeholder row the authors inserted for a "
        "sequence element without a press (always an error, placed midway between the presses around it, which share the "
        "interval); repetition is their sequence number within the block and stimulus, correct_key and sequence_position "
        "give the element at the press's place in a complete sequence (or the cut-off last sequence of a block), left empty "
        "in sequences with an extra press, where the authors aligned presses to elements themselves.",
        "time_in_block_ms is the authors' time stamp from the start of the block and interval_ms their deltaTime, the "
        "interval from the previous press (from the block start for a block's first press); both are in ms, converted from "
        "0.1-ms units, and 43 block labels appear in more than one run of rows (a press logged before a restart, or a short "
        "extra run filed later under the same label), kept in the authors' order.",
        "group gives the age group and, in Experiment 2, the stimulation (verum or placebo anodal tDCS over the right motor "
        "cortex during training, codes 1 and 2, decoded from the authors' SICI file); excluded_by_authors marks the "
        "participant the authors excluded (not in their stimulation list, so the experiment is taken from the participant "
        "code series) and the day-20 visits, which the paper does not report.",
    ],
    unit_note="key press in the finger sequence-tapping task",
    differences=["Mean ages of the two experiments pooled: the rows give 24.59 (young) and 57.76 (middle-aged) years, printed "
                 "as 24.5 and 57.7 (truncated); the per-experiment means round to the printed values.",
                 "The behavioral file's age column differs from the stimulation list for three participants (by 1, 1 and 7 "
                 "years) and its age group for one participant aged 65-66; the paper's numbers follow the list's ages and the "
                 "behavioral file's groups, as here."],
)


def shared_values(s):
    codes, uniques = pd.factorize(pd.Series(s), use_na_sentinel=True)
    u = np.empty(len(uniques) + 1, dtype=object)
    u[:-1] = uniques.tolist()
    u[-1] = None
    return u[codes].tolist()


class Rows:
    def __init__(self, df):
        self.names = list(df.columns)
        self.cols = [shared_values(df[c]) for c in self.names]
        self.n = len(df)

    def __len__(self):
        return self.n

    def __iter__(self):
        for vals in zip(*self.cols):
            yield dict(zip(self.names, vals))


rows = Rows(tab)
del tab, raw, tr
stdlib.release(ID, "keypress", rows, meta, checks)
