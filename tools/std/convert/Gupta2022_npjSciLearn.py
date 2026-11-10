"""Gupta & Rickard (2022), npj Sci Learn: finger-tapping sequence task, massed (30 s trials) vs spaced (10 s trials) training
with 10 s or 30 s breaks, a 300 s rest and a post-rest test. One row per key press, 4 files (one per group), 160 people.

Run from the site folder: python3 -I tools/std/convert/Gupta2022_npjSciLearn.py
Raw files (untrusted data, read only): raw_extra/Gupta2022_npjSciLearn/br{M10,M30,S10,S30}RT.csv from https://osf.io/khaqv/
(OSF component j3hn6). The authors' analysis script (FTT-Mass-Space-Online-OSF.Rmd, osf.io/zbyfm) was read as text, never run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gupta2022_npjSciLearn"
RD = stdlib.raw_dir(ID)
EXCLUDED_PERSON = "MWG998155"   # removed in the authors' script (10 s trials, 10 s breaks group)
# file -> (group label, trial length s, break s, n in paper after removals)
FILES = {
    "brM10RT.csv": ("massed (30 s trials), 10 s breaks", 30, 10, 41),
    "brM30RT.csv": ("massed (30 s trials), 30 s breaks", 30, 30, 37),
    "brS10RT.csv": ("spaced (10 s trials), 10 s breaks", 10, 10, 42),
    "brS30RT.csv": ("spaced (10 s trials), 30 s breaks", 10, 30, 39),
}
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RD / "SOURCES.json"))}


def block_label(trial, tl):
    """Authors' block names (their figure code): Warmup_1, Training_1-10, Pretest_1-2, Warmup_2, Posttest_1-2; the 10 s
    trials are grouped three to a label, so each label covers 30 s of task time."""
    if tl == 30:
        w2, last_train, post0, per = 14, 13, 14, 1
    else:
        w2, last_train, post0, per = 38, 37, 38, 3
    if trial == 1:
        return "Warmup_1", "practice", "practice"
    if trial <= last_train:
        u = (trial - 2) // per + 1
        return (f"Training_{u}", "training", "learning") if u <= 10 else (f"Pretest_{u - 10}", "training", "learning")
    if trial == w2:
        return "Warmup_2", "practice", "practice"
    return f"Posttest_{(trial - post0 - 1) // per + 1}", "test", "test"


frames = []
raws = {}
for fn, (grp, tl, br, n_paper) in FILES.items():
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RD / fn) == src[fn]["sha256"])
    d = pd.read_csv(RD / fn, dtype={"keyPress": str}, na_values=["NA"], keep_default_na=False)
    raws[fn] = d
    d["_file"], d["_grp"], d["_tl"], d["_br"] = fn, grp, tl, br
    frames.append(d)
raw = pd.concat(frames, ignore_index=True)
n = len(raw)

# ------------------------------------------------------------------ structure checks
per_id = raw.groupby(["_file", "ID"], sort=False).size()
checks.internal("each file lists a participant's rows in one run (no interleaving)",
                all((f.ID != f.ID.shift()).sum() == f.ID.nunique() for f in raws.values()))
checks.internal("160 participant sequences in the four files (37 + 41 + 43 + 39)", len(per_id) == 160, f"{len(per_id)}")
dup_ids = raw.groupby("ID")._file.nunique()
checks.internal("exactly one ID code occurs in two files (two separate sequences; kept as two participants)", int((dup_ids > 1).sum()) == 1)
checks.internal("RT is the time since the previous key press (first press of a trial: since trial start): "
                "RT sums to the last totalTime in the trial for every trial of the two massed files",
                all((f.groupby(["ID", "trial"]).RT.sum() == f.groupby(["ID", "trial"]).totalTime.max()).all()
                    for k, f in raws.items() if k[2] == "M"))
tn = raw.groupby("_file").trial.max().to_dict()
checks.internal("16 trials in the massed files, 44 in the spaced files",
                tn == {"brM10RT.csv": 16, "brM30RT.csv": 16, "brS10RT.csv": 44, "brS30RT.csv": 44}, str(tn))
checks.internal("keys are 1-4 (plus 56 presses of the letters a-d, all coded incorrect)",
                set(raw.keyPress) <= set("1234abcd") and raw[raw.keyPress.isin(list("abcd"))].keyCorrect.isin([0, "0", 0.0]).all())

# ------------------------------------------------------------------ table
pos = raw.groupby(["_file", "ID"], sort=False).cumcount().to_numpy()
lab = [block_label(int(t), int(tl)) for t, tl in zip(raw.trial, raw._tl)]
block = np.array([x[0] for x in lab])
btype = np.array([x[1] for x in lab])
phase = np.array([x[2] for x in lab])
kc = pd.to_numeric(raw.keyCorrect, errors="coerce")
sc = pd.to_numeric(raw.seqComp, errors="coerce")
correct = kc.map({1.0: True, 0.0: False}).astype(object).where(kc.notna(), None)
excluded = (btype == "practice") | (kc.to_numpy() != 1) | (sc.to_numpy() == 1)
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": raw._grp.to_numpy(),
    "subject": raw._file.str[2:5] + "_" + raw.ID, "subject_excluded_by_authors": (raw.ID == EXCLUDED_PERSON).to_numpy(),
    "age": "", "sex": "", "handedness": "right", "population": "healthy", "setting": "online",
    "device": "keyboard (keys 1-4, left hand)", "session": 1, "block": block, "trial": pos + 1, "phase": phase,
    "time_s": "", "task": "finger/hand", "unit": "keypress",
    "condition": np.where(raw._tl == 30, "30 s trials", "10 s trials"),
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": excluded,
    "block_number": raw.trial.to_numpy(), "block_type": btype, "epoch": "", "stimulus": "", "correct_key": "",
    "key_pressed": raw.keyPress.to_numpy(), "correct": correct, "rt_ms": "", "rt_correct_ms": "",
    "interval_ms": raw.RT.to_numpy(), "element_type": "", "sequence": "4-1-3-2-4", "sequence_position": "",
    "repetition": sc.astype("Int64").astype(object).where(sc.notna(), None), "triplet_type": "",
    "time_in_block_ms": raw.totalTime.to_numpy(), "hand": "", "direction": "", "hold_ms": "", "flight_ms": "", "finger": "",
})
checks.internal("every row kept, in the authors' order", len(t) == n == 244454, f"{n}")

# ------------------------------------------------------------------ published numbers (authors' script, Rmd lines ~669-750)
g = raw.assign(key=lambda x: x._file.str[2:5])


def cells(d, by):
    return d[(pd.to_numeric(d.keyCorrect) == 1) & (pd.to_numeric(d.seqComp) != 1)].groupby(by, dropna=False).RT.mean().reset_index()


res = {}
for fn, (grp, tl, br, n_paper) in FILES.items():
    d = g[g._file == fn]
    c = cells(d, ["trial", "Bin", "ID"])
    if tl == 30:
        c = c[(c.trial >= 12) & (c.trial != 14)]
        c["pp"] = np.where(c.trial >= 15, "post", "pre")
    else:
        c = c[(c.trial >= 30) & (c.trial != 36)]
        c["pp"] = np.where(c.trial >= 37, "post", "pre")
    m = c.groupby(["pp", "ID"]).RT.mean().unstack(0)
    m = m.drop(index=EXCLUDED_PERSON, errors="ignore")
    tt = stats.ttest_rel(m["post"], m["pre"])
    res[fn] = (m, tt)
    checks.internal(f"{grp}: {n_paper} participants after the authors' removals", len(m) == n_paper, f"{len(m)}")

SRC = "Results, 'Post-rest improvement' (t-tests of trials 11-12 vs 13-14 in each group)"
for fn, tval, dval in [("brM10RT.csv", -9.01, -1.41), ("brM30RT.csv", -3.97, -0.65),
                       ("brS30RT.csv", -0.70, -0.11), ("brS10RT.csv", -7.9, -1.22)]:
    m, tt = res[fn]
    checks.published(f"{FILES[fn][0]}: post-rest vs last training trials, paired t (df {len(m) - 1})", tt.statistic, tval, 0.006 if abs(tval) < 5 else 0.06, SRC)
    checks.published(f"{FILES[fn][0]}: its effect size d", tt.statistic / np.sqrt(len(m)), dval, 0.006, SRC)

# 2 x 2 ANOVA of the pre - post difference
rows = []
for fn, (grp, tl, br, n_paper) in FILES.items():
    m, _ = res[fn]
    for v in (m["pre"] - m["post"]).to_numpy():
        rows.append((v, f"{tl}s", f"{br}s"))
an = pd.DataFrame(rows, columns=["diff", "trials", "breaks"])
import statsmodels.api as sm  # noqa: E402
from statsmodels.formula.api import ols  # noqa: E402
tab = sm.stats.anova_lm(ols("diff ~ C(trials) * C(breaks)", an).fit(), typ=1)   # R aov order: trials, breaks, interaction
SRC2 = "Results, 'Post-rest improvement' (2 x 2 ANOVA on RT improvement; sequential sums of squares as in R aov)"
checks.internal("ANOVA error df = 155", int(tab.loc["Residual", "df"]) == 155)
ss_res = tab.loc["Residual", "sum_sq"]
# The paper's text gives F = 14.49 (eta2 = .08) for 'break time' and F = 5.79 (eta2 = .04) for 'on-task trial time'; in the
# data it is the on-task trial length that gives F = 14.49 and eta2 = .08, and break time gives F = 6.99, eta2 = .04.
checks.published("ANOVA: effect of on-task trial length, F(1, 155) (the paper labels this value 'break time')", tab.loc["C(trials)", "F"], 14.49, 0.01, SRC2)
checks.published("ANOVA: its partial eta squared", tab.loc["C(trials)", "sum_sq"] / (tab.loc["C(trials)", "sum_sq"] + ss_res), 0.08, 0.006, SRC2)
checks.published("ANOVA: effect of break length, partial eta squared (paper labels it 'on-task trial time')", tab.loc["C(breaks)", "sum_sq"] / (tab.loc["C(breaks)", "sum_sq"] + ss_res), 0.04, 0.005, SRC2)
checks.published("ANOVA: interaction F(1, 155)", tab.loc["C(trials):C(breaks)", "F"], 1.217, 0.001, SRC2)
ANOVA_BREAK_F = tab.loc["C(breaks)", "F"]

# reactive inhibition within 3.33 s thirds of the 10 s trials (script lines 726-750)
for fn, tval, dval in [("brS30RT.csv", -8.2, -1.32), ("brS10RT.csv", -6.3, -0.98)]:
    d = g[g._file == fn]
    d = d[(pd.to_numeric(d.keyCorrect) == 1) & (d.trial != 1) & (d.trial < 38) & (pd.to_numeric(d.seqComp) != 1)
          & (pd.to_numeric(d.Bin2) != 2)]
    d = d[d.ID != EXCLUDED_PERSON]
    x = d.groupby(["ID", "Bin2"]).RT.mean().unstack()
    tt = stats.ttest_rel(x[1], x[3])
    checks.published(f"{FILES[fn][0]}: first vs third 3.33 s of a trial, paired t (reactive inhibition)", tt.statistic, tval, 0.06,
                     "Results, 'Reactive inhibition'")
    checks.published(f"{FILES[fn][0]}: its effect size d", tt.statistic / np.sqrt(len(x)), dval, 0.006, "Results, 'Reactive inhibition'")

checks.published("participants analysed (abstract: N = 159)", sum(len(v[0]) for v in res.values()), 159, 0, "Abstract")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))

meta = dict(
    study="Gupta & Rickard 2022",
    paper=dict(authors="Gupta MW, Rickard TC", year=2022,
               title="Dissipation of reactive inhibition is sufficient to explain post-rest improvements in motor sequence learning",
               venue="npj Sci Learn 7(1):25", doi="10.1038/s41539-022-00140-z"),
    lab="Rickard (UC San Diego)", data_url="https://osf.io/khaqv/", license="none stated", license_url="",
    changes=("Converted to the OpenMotor format with participant codes replaced. The authors' columns Bin (which third of "
             "the trial time), Bin2 (thirds of the 10 s trials) and prevIncorrect were dropped; they follow from time in "
             "the trial and from the key coding. Block names are the authors' figure labels, derived from their trial number."),
    how=[
        "One row is one key press in a finger-tapping task: participants typed the sequence 4-1-3-2-4 with keys 1-4 of the "
        "keyboard, left (non-dominant) hand, as fast and accurately as possible. Four groups crossed trial length (30 s "
        "'massed' or 10 s 'spaced') with the break between trials (10 s or 30 s); 360 s of training, a 300 s rest with "
        "double-digit addition, then 60 s of test trials, with the same trial and break lengths.",
        "interval_ms is the authors' RT: time since the previous key press (for the first press of a trial, since trial start); "
        "time_in_block_ms is the press time from trial start; block_number is the authors' trial number and trial counts rows "
        "per participant. correct is the authors' keyCorrect (whether the key was the next key of the sequence) and repetition "
        "is their seqComp (which attempt at the sequence the press belongs to, empty where they left it missing).",
        "Trial 1 and the trial before the post-rest test are warm-up trials (block Warmup_1, Warmup_2, phase practice); the "
        "final 30 s of training are labelled Pretest by the authors. excluded_by_authors marks warm-up trials, incorrect "
        "presses and the first sequence of each trial (seqComp 1), which the authors removed before analysing RT; "
        "subject_excluded_by_authors marks one participant that the authors' script removes in the 10 s trials, 10 s breaks group.",
        "Only the key pressed and its timing are recorded; no text, age or sex was shared per participant (groups were 67-80% female, "
        "mean age about 20-21 years). The files do not say what the key letters a-d, pressed 56 times, were.",
    ],
    unit_note="key press",
    differences=["The ANOVA of the post-rest RT improvement: the paper reports F(1, 155) = 14.49 (eta2 = .08) for break time and 5.79 (eta2 = .04) for on-task trial time. In the converted rows 14.49 and eta2 = .08 belong to the on-task trial length (30 s vs 10 s trials) and eta2 = .04 to the break length, so the labels appear swapped in the text; the break-length F is 6.99 here, not 5.79 (probably a typing slip). The interaction F = 1.217 matches.",
                 "The paper's reactive-inhibition t-tests for the 30 s trials (t(36) = -3.42 and t(40) = -3.26) cannot be reproduced: "
                 "the authors' script has no code for them, and the closest version computed here gives t = -4.19 and -3.49. "
                 "The same tests for the 10 s trials, the four post-rest t-tests and the ANOVA reproduce.",
                 "The authors' script treats trial 38 (the second warm-up of the spaced groups) as part of the post-rest window for the "
                 "paired t-tests (trial >= 37); that is reproduced as is in the checks."],
)
stdlib.release(ID, "keypress", t, meta, checks)
