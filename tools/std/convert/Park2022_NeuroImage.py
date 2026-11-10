"""Park, Janacsek, Nemeth & Jeon (2022), NeuroImage 260:119459: ASRT in the MRI scanner, 35 participants, one row per cue.

Run from the site folder: python3 -I tools/std/convert/Park2022_NeuroImage.py
Raw files (untrusted data, read only): raw_extra/Park2022_NeuroImage/sub-NN_task-asrt_run-R_events.tsv and participants.tsv from
https://openneuro.org/datasets/ds003401 (CC0). Behaviour only: no imaging data are read.
Expected values of the published checks come from the paper (Methods 2.1, 3.1, Results 4.1), read at
https://gala.gre.ac.uk/id/eprint/37707/ (publisher version, open copy).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Park2022_NeuroImage"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

parts = pd.read_csv(RAW / "participants.tsv", sep="\t")
frames = []
for f in sorted(glob.glob(str(RAW / "sub-*_events.tsv"))):
    x = pd.read_csv(f, sep="\t", na_values=["n/a"])
    name = Path(f).name
    assert int(name[4:6]) == x.Subject.iloc[0] == x.Subject.max() and int(name.split("run-")[1][0]) == x.Run.iloc[0]
    frames.append(x)
a = pd.concat(frames, ignore_index=True)
n = len(a)
checks.internal("35 participants x (1445 + 1020 + 1020) trials",
                a.Subject.nunique() == 35 and n == 35 * 3485 and bool((a.groupby(["Subject", "Run"]).size().unstack().eq([1445, 1020, 1020]).all().all())))
checks.internal("firstACC is 1 exactly when the pressed key equals the cued position", bool(((a.event == a.firstRESP) == (a.firstACC == 1)).all()))
checks.internal("cues 1-4, responses 1-4 or none, RT 0-499 ms", bool(a.event.between(1, 4).all() and a.firstRESP.dropna().between(1, 4).all() and a.firstRT.between(0, 499).all()))
noresp = a.firstRESP.isna().to_numpy()
checks.internal("no-response trials: RT is 0 and accuracy 0",
                bool((a.firstRT[noresp] == 0).all() and (a.firstACC[noresp] == 0).all()), f"{noresp.sum()} trials, RT 0 with a response on {((a.firstRT == 0) & ~noresp).sum()} trials")

# ------------------------------------------------------------ structure
block_number = a.BlocksInRun.to_numpy() + np.select([a.Run == 2, a.Run == 3], [17, 29], 0)
a["block_number"] = block_number
checks.internal("41 blocks of 85 trials per participant (5 practice blocks + 36 task blocks)",
                a.groupby(["Subject", "block_number"]).size().eq(85).all() and a.block_number.max() == 41)
practice_block = (a.Run == 1) & (a.BlocksInRun <= 5)
checks.internal("blocks 1-5 are all Prac trials; every task block starts with 5 Prac trials then 40 P and 40 R alternating, pattern first",
                bool((a.TrialType[practice_block] == "Prac").all()
                     and (a.TrialType[~practice_block & (a.SubTrial <= 5)] == "Prac").all()
                     and (a.TrialType[~practice_block & (a.SubTrial > 5)] == np.where(a.SubTrial[~practice_block & (a.SubTrial > 5)] % 2 == 0, "P", "R")).all()))
k = a.SubTrial.to_numpy() - 1
seq_pos = np.where(a.TrialType == "Prac", 0, (k - 5) % 8 + 1)
rep = np.where(a.TrialType == "Prac", 0, (k - 5) // 8 + 1)
pat = a[a.TrialType == "P"]
ppos = seq_pos[pat.index]
patterns = pat.assign(pp=(ppos + 1) // 2).groupby(["Subject", "pp"]).event.nunique()
checks.internal("each participant's four pattern positions are the same in every cycle", bool((patterns == 1).all()))
pseq = pat.assign(pp=(ppos + 1) // 2).groupby(["Subject", "pp"]).event.first().unstack()
seq_text = {s: "-".join(f"{int(v)}-r" for v in pseq.loc[s]) for s in pseq.index}
cond = a.Condition
checks.internal("Condition is set on all task-block trials except the first two of each block",
                int(cond.notna().sum()) == int((a.TrialType != "Prac").sum()) - 35 * 36 * 2, f"{cond.notna().sum()} coded trials")
checks.internal("PatternHigh <-> P, RandomLow/RandomHigh <-> R", bool((cond.dropna().str.startswith("Pattern") == (a.TrialType[cond.notna()] == "P")).all()))

# ------------------------------------------------------------ exclusions as in Methods 3.1
mainx = a[a.TrialType != "Prac"]
acc_sub = mainx.groupby("Subject").firstACC.mean()
bad = set(acc_sub[acc_sub < 0.8].index)
checks.published("participants with accuracy below 80% excluded", len(bad), 4, 0, "Methods 2.1 (four participants excluded)")
good = acc_sub.index[acc_sub >= 0.8]
checks.published("participants analysed", len(good), 31, 0, "Methods 2.1")

corr = mainx[(mainx.firstACC == 1) & mainx.Subject.isin(good)]
g = corr.groupby("Subject").firstRT
inlier = (corr.firstRT - g.transform("mean")).abs() <= 2 * g.transform("std")
kept = corr[inlier].copy()
excluded = np.ones(n, dtype=bool)
excluded[kept.index] = False

# ------------------------------------------------------------ the OpenMotor table
people = {int(r.participant_id[4:]): r for r in parts.itertuples()}
subj = a.Subject.to_numpy()
codes = np.array([f"{ID}_S{s:03d}" for s in subj])
element = a.TrialType.map({"P": "sequence", "R": "random", "Prac": "warm-up"}).to_numpy()
trip = cond.map({"PatternHigh": "high", "RandomHigh": "high", "RandomLow": "low"}).fillna("none").to_numpy()
blocktype = np.where(practice_block, "practice", "training")
resp = a.firstRESP.astype("Int64")
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": "", "subject": codes, "person": codes,
    "subject_excluded_by_authors": np.isin(subj, list(bad)),
    "age": [people[s].age for s in subj], "sex": [people[s].sex for s in subj], "handedness": "right",
    "population": "healthy", "setting": "lab", "device": "MRI-compatible four-button response box (keys 1-4)",
    "session": 1, "block": [f"run{r}_block{b}" for r, b in zip(a.Run, a.BlocksInRun)],
    "trial": a.groupby("Subject").cumcount().to_numpy() + 1, "phase": np.where(practice_block, "practice", "learning"),
    "time_s": "", "task": "sequence learning", "unit": "keypress",
    "condition": cond.fillna("").to_numpy(), "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal",
    "excluded_by_authors": excluded,
    "block_number": block_number, "block_type": blocktype, "epoch": a.Run.to_numpy(), "stimulus": a.event.to_numpy(),
    "correct_key": a.event.to_numpy().astype(str), "key_pressed": resp.astype(object).where(resp.notna(), "").astype(str).to_numpy(),
    "correct": (a.firstACC == 1).to_numpy(), "rt_ms": np.where(noresp, np.nan, a.firstRT), "rt_correct_ms": np.nan,
    "element_type": element,
    "sequence": [seq_text[s] for s in subj],
    "sequence_position": pd.Series(seq_pos, dtype=object).where(seq_pos > 0, None).to_numpy(),
    "repetition": pd.Series(rep, dtype=object).where(rep > 0, None).to_numpy(),
    "triplet_type": trip, "time_in_run_s": a.onset.to_numpy(),
})
t.loc[t.correct & t.rt_ms.notna(), "rt_correct_ms"] = t.rt_ms
checks.internal("every row kept in the authors' order", len(t) == n)

# ------------------------------------------------------------ published checks from the converted rows
tt = t[t.element_type != "warm-up"]
tt = tt[~tt.subject_excluded_by_authors]
checks.published("mean accuracy over the 31 analysed participants, task blocks (%)", 100 * tt.correct.mean(), 91.4, 0.05, "Results 4.1.1 (91.4%)")
for r, rep_ in [(1, 91.4), (2, 92.0), (3, 90.8)]:
    checks.published(f"accuracy in run {r} (%)", 100 * tt[tt.epoch == r].correct.mean(), rep_, 0.05, "Results 4.1.1")
t["z"] = np.nan
c = t[~t.excluded_by_authors & (t.element_type != "warm-up")]
gz = c.groupby("subject").rt_ms
z = (c.rt_ms - gz.transform("mean")) / gz.transform("std")
c = c.assign(z=z)
def sl(d):
    m = d.groupby(["subject", "condition"]).z.mean().unstack()
    return m["RandomLow"] - m["RandomHigh"]
for runs, rep_, lab in [([1, 2, 3], 11.21, "across three runs"), ([1], 7.26, "run 1"), ([2], 7.18, "run 2"), ([3], 13.08, "run 3")]:
    s = sl(c[c.epoch.isin(runs)])
    tv = stats.ttest_1samp(s, 0)
    checks.published(f"statistical learning score (normalised RT, Random-Low minus Random-High) t(30), {lab}", tv.statistic, rep_, 0.005,
                     "Results 4.1.2 (two SD outlier cut per participant, correct trials)")
    checks.internal(f"SL score df = 30 ({lab})", len(s) == 31)
# block-wise SL scores and the one-way (linear) ANOVA on learning time
bs = c.groupby(["subject", "block_number", "condition"]).z.mean().unstack()
bs = (bs["RandomLow"] - bs["RandomHigh"]).dropna()
x = bs.index.get_level_values("block_number").to_numpy()
lr = stats.linregress(x, bs.to_numpy())
F = (lr.rvalue ** 2) / (1 - lr.rvalue ** 2) * (len(bs) - 2)
print("block-wise SL scores", len(bs), "F", F, "df2", len(bs) - 2)
checks.published("one-way ANOVA of block-wise SL scores on block number, F(1,1114)", F, 19.65, 0.005, "Results 4.1.2 (F(1,1114) = 19.65)")
checks.internal("block-wise SL scores: 31 participants x 36 blocks (df 1114)", len(bs) == 1116)
rt_c = c.rt_ms
print("mean correct RT (after cut)", rt_c.mean(), c.groupby("epoch").rt_ms.mean().to_dict(),
      "raw correct", tt[tt.correct].rt_ms.mean())

ptab = parts.assign(person=[f"{ID}_S{int(p[4:]):03d}" for p in parts.participant_id])[["person", "age", "sex"]]
ptab["excluded_low_accuracy"] = [int(p[-3:]) in bad for p in ptab.person]
checks.published("participants in the study", len(ptab), 35, 0, "Methods 2.1")
checks.published("mean age of the 35 participants (years)", ptab.age.mean(), 23.1, 0.05, "Methods 2.1")
checks.published("SD of age, 35 participants", ptab.age.std(), 2.8, 0.05, "Methods 2.1")
checks.published("females among the 35", (ptab.sex == "F").sum(), 17, 0, "Methods 2.1")
an = ptab[~ptab.excluded_low_accuracy]
checks.published("mean age of the 31 analysed (years)", an.age.mean(), 23.0, 0.05, "Methods 2.1")
checks.published("females among the 31 analysed", (an.sex == "F").sum(), 14, 0, "Methods 2.1")

for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))
if not all(ch["ok"] for ch in checks.items):
    sys.exit(1)

meta = dict(
    study="Park et al. 2022",
    paper=dict(authors="Park J, Janacsek K, Nemeth D, Jeon HA", year=2022,
               title="Reduced functional connectivity supports statistical learning of temporally distributed regularities",
               venue="NeuroImage 260:119459", doi="10.1016/j.neuroimage.2022.119459"),
    lab="Jeon (DGIST) with Nemeth & Janacsek",
    data_url="https://openneuro.org/datasets/ds003401",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted to the OpenMotor format from the BIDS events files (behaviour only; the fMRI images are not included); "
             "participant codes replaced. Dropped as redundant or imaging-only: the E-Prime clock values (firstON, ScaningOnset), "
             "RandomDelay/PatternDelay, and the block and rest onset/duration columns. The authors' Condition label is kept in condition."),
    how=[
        "One row is one cued response in the Alternating Serial Reaction Time task done in an MRI scanner: 41 blocks of 85 cues "
        "per participant (5 random practice blocks, then 36 task blocks in three scanning runs of 12). Each task block starts with 5 "
        "random warm-up cues followed by 10 cycles of an 8-cue sequence in which pattern and random cues alternate (pattern first).",
        "block_number counts the 41 blocks; block is run and block within the run; epoch is the run (1-3). stimulus is the cued "
        "position 1-4; key_pressed is the button pressed (empty when no response came within the 500 ms the cue stayed on); "
        "rt_ms is the time from cue onset (ms), empty without a response; rt_correct_ms equals rt_ms on correct presses.",
        "condition is the authors' label (PatternHigh, RandomHigh, RandomLow); triplet_type is high for PatternHigh and RandomHigh "
        "and low for RandomLow. The authors' statistical learning score is the normalised mean RT of Random-Low minus Random-High.",
        "excluded_by_authors is true for warm-up and practice-block cues, errors and misses, and RTs more than 2 SD from the "
        "participant's mean correct RT (task blocks); subject_excluded_by_authors marks the four participants below 80% accuracy. "
        "time_in_run_s is the cue onset from the start of its scanning run.",
    ],
    unit_note="cued response (key press) in the ASRT task",
    differences=[],
)
meta["differences"] = [
    "Mean correct RT after the 2-SD cut is 349.6 ms here against 349.9 ms in the paper (runs 356.2, 347.1, 345.6 ms against 356.6, 347.2, 346.1); the difference is larger than the paper's rounding, so these were not used as checks.",
    "The SD of age of the 31 analysed participants is 2.86 years from the deposit's participants.tsv, against 2.8 in the paper.",
]
t = t.drop(columns=["z"])
for col in [c[1] for c in stdlib.columns_for("keypress")]:
    if col not in t.columns:
        t[col] = ""
stdlib.release(ID, "keypress", t, meta, checks, participants=ptab,
               participant_columns=[("age", "number", "years", "Age from the deposit's participants.tsv."),
                                    ("sex", "category", "F, M", "Sex from participants.tsv."),
                                    ("excluded_low_accuracy", "true/false", "", "Accuracy below 80% in the task blocks; the authors left these participants out of the analysis.")])
