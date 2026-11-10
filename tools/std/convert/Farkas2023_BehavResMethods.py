"""Farkas, Krajcsi, Janacsek & Nemeth (2023), Behav Res Methods: ASRT learning session, 180 participants, one row per key press.

Run from the site folder: python3 -I tools/std/convert/Farkas2023_BehavResMethods.py
Raw file (untrusted data, read only): raw_extra/Farkas2023_BehavResMethods/ASRT_master.csv from https://osf.io/9szk7/.
The published checks re-run the deterministic estimates of the authors' ASRT_reliability_main.R (read as text, never run)
on the converted rows: split-half correlations and Cronbach's alphas of RT and accuracy triplet learning scores.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Farkas2023_BehavResMethods"
RAWFILE = stdlib.raw_dir(ID) / "ASRT_master.csv"
N_BLOCKS, TRIALS_PER_BLOCK, N_WARMUP, CYCLE = 45, 85, 5, 8

checks = stdlib.Checks()

# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(stdlib.raw_dir(ID) / "SOURCES.json"))}
checks.internal("raw file matches the downloaded hash", stdlib.sha256(RAWFILE) == src["ASRT_master.csv"]["sha256"])
raw = pd.read_csv(RAWFILE, na_values=["NA"], keep_default_na=False,
                  dtype={"Subject": int, "Block": int, "ACCmean": float, "RTmean": float, "CorrectResponse": str,
                         "event": int, "finalRT": int, "firstACC": int, "firstRESP": str, "firstRT": int,
                         "LearnTrialList.Cycle": float, "TrialType": str, "TripC": str, "TT": str, "epoch": int})
n = len(raw)

# ---------------------------------------------------------------- structure (positions within participant and block)
first_seen = pd.factorize(raw.Subject)[0]                       # participants in order of first appearance
pos = raw.groupby("Subject", sort=False).cumcount().to_numpy()  # 0-based order within participant
block_number = pos // TRIALS_PER_BLOCK + 1
subj_arr = raw.Subject.to_numpy()
run_start = np.flatnonzero(np.r_[True, subj_arr[1:] != subj_arr[:-1]])
checks.internal("the authors' file lists blocks 1-15 of every participant, then blocks 16-30, then 31-45, "
                "with participants in the same order each time (rows are kept in this order)",
                len(run_start) == 540 and bool((np.diff(np.r_[run_start, n]) == 15 * TRIALS_PER_BLOCK).all()
                                               and (first_seen[run_start] == np.tile(np.arange(180), 3)).all()
                                               and ((block_number[run_start] - 1) // 15 == np.repeat(np.arange(3), 180)).all()))
k = pos % TRIALS_PER_BLOCK                                      # 0-based trial within block
is_warm = k < N_WARMUP
seq_pos = np.where(is_warm, 0, (k - N_WARMUP) % CYCLE + 1)
rep = np.where(is_warm, 0, (k - N_WARMUP) // CYCLE + 1)

per_subj = raw.groupby("Subject", sort=False).size()
checks.internal("180 participants (Methods, Participants)", raw.Subject.nunique() == 180, f"{raw.Subject.nunique()}")
checks.internal("every participant has 45 blocks x 85 trials = 3825 rows (Methods, The ASRT task)",
                bool((per_subj == N_BLOCKS * TRIALS_PER_BLOCK).all()))
checks.internal("nine epochs of five blocks", bool((raw.epoch.to_numpy() == (block_number - 1) // 5 + 1).all()))
checks.internal("authors' Block label restarts every 15 blocks (block_number = 15 x run + Block)",
                bool((block_number == 15 * ((raw.epoch.to_numpy() - 1) // 3) + raw.Block.to_numpy()).all()))
checks.internal("each block = 5 warm-up trials then 80 trials alternating random/pattern",
                bool(((raw.TrialType == "Prac").to_numpy() == is_warm).all()
                     and ((raw.TrialType == "R").to_numpy() == (~is_warm & (seq_pos % 2 == 1))).all()))
cyc = raw["LearnTrialList.Cycle"].to_numpy()
checks.internal("authors' cycle counter = (Block - 1) x 10 + repetition (repetition is lossless)",
                bool(np.isnan(cyc[is_warm]).all()
                     and (cyc[~is_warm] == (raw.Block.to_numpy()[~is_warm] - 1) * 10 + rep[~is_warm]).all()))

# ---------------------------------------------------------------- responses
keymap = raw.groupby("CorrectResponse").event.agg(lambda x: sorted(set(x)))
checks.internal("keys Z, C, B, M map one to one to positions 1-4 left to right (Methods)",
                keymap.to_dict() == {"b": [3], "c": [2], "m": [4], "z": [1]}, str(keymap.to_dict()))
correct = raw.firstACC.to_numpy() == 1
checks.internal("correct == (key pressed is the correct key)",
                bool((correct == (raw.firstRESP == raw.CorrectResponse).to_numpy()).all()))
checks.internal("finalRT equals firstRT on correct presses and exceeds it on every error (time to the correct key)",
                bool((raw.finalRT[correct] == raw.firstRT[correct]).all() and (raw.finalRT[~correct] > raw.firstRT[~correct]).all()))
checks.internal("ranges: stimulus 1-4, firstRT 0-5000 ms, finalRT 1-6000 ms",
                bool(raw.event.between(1, 4).all() and raw.firstRT.between(0, 5000).all() and raw.finalRT.between(1, 6000).all()),
                f"firstRT {raw.firstRT.min()}-{raw.firstRT.max()}, finalRT {raw.finalRT.min()}-{raw.finalRT.max()}, "
                f"firstRT <= 100 ms on {(raw.firstRT <= 100).sum()} rows")

# ---------------------------------------------------------------- triplets and sequences
ev = raw.event.to_numpy()
blk_key = raw.Subject.to_numpy() * 100 + block_number
p1 = pd.Series(ev).groupby(blk_key).shift(1).to_numpy()
p2 = pd.Series(ev).groupby(blk_key).shift(2).to_numpy()
coded = (raw.TT != "X").to_numpy()
last3 = (p2[coded] * 100 + p1[coded] * 10 + ev[coded]).astype(int)
checks.internal("authors' triplet code = the last three stimuli in the block, on every coded row (dropped as redundant)",
                bool((last3 == raw.TripC[coded].astype(int).to_numpy()).all() and (raw.TripC[~coded] == "XXX").all()
                     and (k[~coded] < N_WARMUP + 2).all() and (k[coded] >= N_WARMUP + 2).all()))
a, b, c = p2, p1, ev
checks.internal("trills and repetitions are coded as the authors define them",
                bool((((raw.TT == "R").to_numpy()) == (coded & (a == b) & (b == c))).all()
                     and (((raw.TT == "T").to_numpy()) == (coded & (a == c) & (a != b))).all()))

pat = raw[(raw.TrialType == "P").to_numpy()]
pat_key = pat.Subject.to_numpy() * 10000 + block_number[pat.index] * 100 + rep[pat.index]
cycles = pd.Series(pat.event.to_numpy()).groupby(pat_key).agg(tuple)
cyc_subj = cycles.index.to_numpy() // 10000
per_subject_patterns = pd.Series(list(cycles.values)).groupby(cyc_subj).nunique()
checks.internal("each participant's four pattern positions are identical in all 450 cycles",
                bool((per_subject_patterns == 1).all()) and len(per_subject_patterns) == 180)
pattern = pd.Series(list(cycles.values)).groupby(cyc_subj).first()          # Subject -> (p1, p2, p3, p4)
seq_text = {s: "-".join(x for p in t for x in ("r", str(p))) for s, t in pattern.items()}
succ = {s: {t[i]: t[(i + 1) % 4] for i in range(4)} for s, t in pattern.items()}
hl = raw.TT.isin(["H", "L"]).to_numpy()
is_high = np.array([succ[s][int(x)] == int(y) for s, x, y in zip(raw.Subject.to_numpy()[hl], a[hl], c[hl])])
checks.internal("authors' high/low triplet labels follow from each participant's sequence",
                bool(((raw.TT.to_numpy()[hl] == "H") == is_high).all()))

# block-level columns the authors' file carries; they are not the participants' own values
blockvals = raw.groupby([raw.Subject, block_number])[["ACCmean", "RTmean"]].first().unstack()
checks.internal("ACCmean and RTmean (dropped) are not per-participant: at most two 45-block vectors across 180 people",
                blockvals.apply(tuple, axis=1).nunique() <= 2)

# ---------------------------------------------------------------- the OpenMotor table
codes = np.array([f"{ID}_S{i + 1:03d}" for i in range(raw.Subject.nunique())])
element_type = raw.TrialType.map({"P": "sequence", "R": "random", "Prac": "warm-up"}).to_numpy()
triplet_type = raw.TT.map({"H": "high", "L": "low", "T": "trill", "R": "repetition", "X": "none"}).to_numpy()
rt = raw.firstRT.to_numpy()
subject = codes[first_seen]

# authors' row exclusions (Methods): non-triplet rows, trills, repetitions, RT <= 100 ms, RT >= mean + 3 SD of the
# participant (computed, as in their sequence-wise and RT-accuracy code, over non-warm-up presses with RT > 100 ms)
ref = (~is_warm) & (rt > 100)
mu = pd.Series(rt[ref]).groupby(subject[ref]).mean()
sd = pd.Series(rt[ref]).groupby(subject[ref]).std(ddof=1)
cut = pd.Series(subject).map(mu + 3 * sd).to_numpy()
excluded = (~hl) | (rt <= 100) | (rt >= cut)

t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": "", "subject": subject,
    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
    "setting": "lab", "device": "keyboard (Z, C, B, M keys)", "session": 1, "block": raw.Block.astype(str).to_numpy(),
    "trial": pos + 1, "phase": "learning", "time_s": "", "task": "finger/hand", "unit": "keypress", "condition": "",
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": excluded,
    "block_number": block_number, "block_type": "training", "epoch": raw.epoch.to_numpy(), "stimulus": ev,
    "correct_key": raw.CorrectResponse.to_numpy(), "key_pressed": raw.firstRESP.to_numpy(), "correct": correct,
    "rt_ms": rt, "rt_correct_ms": raw.finalRT.to_numpy(), "interval_ms": "", "element_type": element_type,
    "sequence": raw.Subject.map(seq_text).to_numpy(),
    "sequence_position": pd.Series(seq_pos, dtype=object).where(~is_warm, None).to_numpy(),
    "repetition": pd.Series(rep, dtype=object).where(~is_warm, None).to_numpy(),
    "triplet_type": triplet_type,
})
checks.internal("every row kept, in the authors' order", len(t) == n and bool((np.diff(t.trial.to_numpy())[
    np.diff(first_seen) == 0] == 1).all()))
checks.internal("excluded_by_authors share", True,
                f"{excluded.sum()} of {n} rows ({100 * excluded.mean():.1f}%); of high/low rows, "
                f"{((rt <= 100) & hl).sum()} at <= 100 ms and {((rt > 100) & (rt >= cut) & hl).sum()} at or above "
                f"mean + 3 SD")


# ---------------------------------------------------------------- published checks, from the converted rows
def above_cut(d):
    g = d.groupby("subject").rt_ms
    return d.rt_ms < d.subject.map(g.mean() + 3 * g.std(ddof=1))


def halves(d, split):
    d = d.copy()
    pattern_ = ["S1", "S1", "S2", "S2"] if split == "trial" else ["S1"] * 8 + ["S2"] * 8
    d["half"] = np.resize(np.array(pattern_), len(d))   # assigned in row order across all participants, as in the R code
    return d


def scores(d, measure, two_stage, split=True):
    """Triplet learning score per participant (and half): median RT low - high, or mean accuracy high - low."""
    d = d[d.triplet_type.isin(["high", "low"])]
    val = "rt_ms" if measure == "rt" else "correct"
    if measure == "acc":
        d = d.assign(correct=d.correct.astype(float))
    keys = ["subject"] + (["epoch"] if two_stage else []) + (["half"] if split else []) + ["triplet_type"]
    g = d.groupby(keys)[val]
    m = g.median() if measure == "rt" else g.mean()
    if two_stage:
        m = m.groupby([x for x in keys if x != "epoch"]).mean()
    m = m.unstack("triplet_type")
    ls = (m["low"] - m["high"]) if measure == "rt" else (m["high"] - m["low"])
    if not split:
        return ls.to_numpy()
    w = ls.unstack("half")
    return w["S1"].to_numpy(), w["S2"].to_numpy()


slim = t[["subject", "epoch", "element_type", "triplet_type", "correct", "rt_ms", "excluded_by_authors"]]


def reliability(measure, split, two_stage):
    d = slim[slim.triplet_type.isin(["high", "low"])] if split == "trial" else slim[slim.element_type != "warm-up"]
    d = halves(d, split)
    d = d[(d.rt_ms > 100) & d.correct] if measure == "rt" else d[d.rt_ms > 100]
    d = d[above_cut(d)]
    s1, s2 = scores(d, measure, two_stage)
    assert len(s1) == 180
    return np.corrcoef(s1, s2)[0, 1], 4 * np.cov(s1, s2, ddof=1)[0, 1] / np.var(s1 + s2, ddof=1)


SRC_RT = "Results, 'Reaction time-based learning scores' and Fig. 2"
SRC_ACC = "Results, 'Accuracy-based learning scores' and Fig. 3"
r, al = reliability("rt", "sequence", False)
checks.published("RT learning score split-half r, sequence-wise split, single-stage average", r, 0.606, 0.0005, SRC_RT)
checks.published("RT learning score Cronbach alpha, sequence-wise split, single-stage average", al, 0.754, 0.0005, SRC_RT)
r, al = reliability("rt", "trial", True)
checks.published("RT learning score split-half r, trial-wise split, two-stage average (0.65449 here, 0.6545 to four "
                 "decimals; the paper's .655 matches rounding that, so the tolerance is one unit of the third decimal)",
                 r, 0.655, 0.001, SRC_RT)
checks.published("RT learning score Cronbach alpha, trial-wise split, two-stage average", al, 0.791, 0.0005, SRC_RT)
r, al = reliability("acc", "trial", False)
checks.published("accuracy learning score split-half r, trial-wise split, single-stage average", r, 0.598, 0.0005, SRC_ACC)
checks.published("accuracy learning score Cronbach alpha, trial-wise split, single-stage average", al, 0.747, 0.0005, SRC_ACC)

# accuracy, sequence-wise, single stage, using the excluded_by_authors flag itself (its 3-SD cut is this analysis's cut)
d = halves(slim[slim.element_type != "warm-up"], "sequence")
s1, s2 = scores(d[~d.excluded_by_authors], "acc", False)
r = np.corrcoef(s1, s2)[0, 1]
checks.published("accuracy learning score split-half r, sequence-wise split, single-stage average "
                 "(rows with excluded_by_authors false)", r, 0.531, 0.0005, SRC_ACC)
checks.published("accuracy learning score Cronbach alpha, sequence-wise split, single-stage average "
                 "(rows with excluded_by_authors false)", 4 * np.cov(s1, s2, ddof=1)[0, 1] / np.var(s1 + s2, ddof=1),
                 0.690, 0.0005, SRC_ACC)

# correlation of RT and accuracy learning scores, two-stage average over all non-warm-up presses
base = slim[slim.element_type != "warm-up"]
d = base[base.rt_ms > 100]
acc_ls = scores(d[above_cut(d)], "acc", True, split=False)
d = base[(base.rt_ms > 100) & base.correct]
rt_ls = scores(d[above_cut(d)], "rt", True, split=False)
r = np.corrcoef(acc_ls, rt_ls)[0, 1]
checks.published("correlation of RT and accuracy learning scores", r, 0.09, 0.005,
                 "Results, 'Dissociation of RT- and accuracy-based learning scores' (r = .09)")
checks.internal("same correlation as the value in the authors' script (r_xy = 0.08985844)", abs(r - 0.08985844) < 1e-7,
                f"{r:.8f}")

for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.5f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))

# ---------------------------------------------------------------- release
meta = dict(
    study="Farkas et al. 2023",
    paper=dict(authors="Farkas BC, Krajcsi A, Janacsek K, Nemeth D", year=2023,
               title="The complexity of measuring reliability in learning tasks: An illustration using the Alternating "
                     "Serial Reaction Time Task",
               venue="Behav Res 56(1):301-317", doi="10.3758/s13428-022-02038-5"),
    lab="Nemeth (Lyon Neuroscience Research Center / ELTE)",
    data_url="https://osf.io/9szk7/",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced; the authors' triplet code column (the "
             "last three stimulus positions) and their block-level ACCmean and RTmean columns (the same for every "
             "participant, not their own values) were dropped."),
    how=[
        "One row is one key press in the learning session of the Alternating Serial Reaction Time task: 45 blocks of 85 "
        "trials, each block five random warm-up trials followed by ten repetitions of an 8-element cycle in which random "
        "(r) and pattern elements alternate; block is the authors' label, which restarts at 1 every 15 blocks, and "
        "block_number counts the 45 blocks.",
        "Rows keep the authors' file order, which lists blocks 1-15 of every participant, then blocks 16-30, then blocks "
        "31-45, so sort by subject and trial to read each participant's sequence.",
        "stimulus is the cued position (1-4, left to right; keys Z, C, B, M), correct and rt_ms are the authors' first "
        "response (firstACC, firstRT, ms from stimulus onset), rt_correct_ms is their finalRT, the time to the correct "
        "key, which the stimulus waited for, and sequence gives each participant's cycle with the pattern positions read "
        "from the pattern trials (the same in all 450 cycles).",
        "excluded_by_authors is true for rows outside the authors' learning scores: warm-up trials, the first two trials "
        "of each block, trills, repetitions, first-press RTs of 100 ms or less, and RTs at or above the participant's "
        "mean + 3 SD over non-warm-up trials with RT above 100 ms (their trial-wise code takes this cut over high and "
        "low triplets only, which moves about 0.07% of rows); RT learning scores used correct first presses only.",
    ],
    unit_note="key press (one cued response in the ASRT task)",
)

def shared_values(s):
    """Column as a list of native Python values (bool, int, str, None) that share one object per distinct value."""
    codes, uniques = pd.factorize(s)            # missing values get code -1
    u = np.empty(len(uniques) + 1, dtype=object)
    u[:-1] = uniques.tolist()
    u[-1] = None                                # code -1 -> None
    return u[codes].tolist()


class Rows:
    """The table as a re-iterable sequence of row dicts made on the fly: 688,500 dicts at once would need ~2 GB."""

    def __init__(self, df):
        self.names = list(df.columns)
        self.cols = [shared_values(df[col]) for col in self.names]
        self.n = len(df)

    def __len__(self):
        return self.n

    def __iter__(self):
        for vals in zip(*self.cols):
            yield dict(zip(self.names, vals))


rows = Rows(t)
del t, slim, raw, base, d
stdlib.release(ID, "keypress", rows, meta, checks)
