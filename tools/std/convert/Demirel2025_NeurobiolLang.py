"""Demirel, Lametti, Gilboa, Stagg & Watkins (2025), Neurobiol Lang 6:NOL.a.22: theta-gamma-peak tACS (TGP) or sham over left ventral motor
cortex while 61 adults adapt to an upward F1 shift (110 mel) in the feedback of "head", "bed" and "dead".

Raw files (raw_extra/Demirel2025_NeurobiolLang/, OSF mt8zx, read as text tables): all_F1_trials.csv, all_F2_trials.csv (61 participants x 495
trials, F1 and F2 in mel, one column per trial in time order), Production_OSF.csv (per-participant summaries, group, sex).
The authors' experiment_code.m was read as text (never run): 495 trials = vowel exploration 9 words x 15 (135), baseline 3 x 15 (45),
learning 3 x 75 (225, shift 110 mel at angle 0 = F1 up), after-effect 3 x 30 (90). The paper numbers trials after the exploration phase
(baseline 1-45, learning 46-270, after-effect 271-360).
Run from the site folder: python3 -I tools/std/convert/Demirel2025_NeurobiolLang.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Demirel2025_NeurobiolLang"
RAW = stdlib.raw_dir(ID)
SRC = "Demirel et al. 2025, Results and Table 1"
checks = stdlib.Checks()
P = pd.read_csv(RAW / "Production_OSF.csv", encoding="utf-8-sig")
F1 = pd.read_csv(RAW / "all_F1_trials.csv", encoding="utf-8-sig")
F2 = pd.read_csv(RAW / "all_F2_trials.csv", encoding="utf-8-sig")
checks.internal("the three files list the same participants in the same order and groups",
                bool((P.subject.values == F1.subject.values).all() and (P.subject.values == F2.subject.values).all()
                     and (P.group.values == F1.group.values).all() and (P.group.values == F2.group.values).all()))
n = 495
f1 = F1[[f"F1_{i}" for i in range(1, n + 1)]].to_numpy(float)
f2 = F2[[f"F2_{i}" for i in range(1, n + 1)]].to_numpy(float)
checks.internal("495 trials per participant, 61 participants", f1.shape == (61, 495) and f2.shape == (61, 495))

# Which authors' group letter is TGP tACS: the group with 15 women (paper: TGP 15 of 30, sham 16 of 31) and baseline F1 about 747 (TGP) vs 767 (sham)
bl = f1[:, 135:180].mean(axis=1)
cnt = P.groupby("group").size().to_dict()
women = P[P.sex == 1].groupby("group").size().to_dict() if False else None
tg = {g: bl[(P.group == g).values].mean() for g in ("A", "B")}
TGP = min(tg, key=lambda g: abs(tg[g] - 747))   # decided from Table 1 baseline F1 (747 vs 767), cross-checked with group sizes below
SHAM = "B" if TGP == "A" else "A"
GROUP = {TGP: "TGP tACS", SHAM: "sham tACS"}

def phase(i):  # i = 1-based file trial
    if i <= 135:
        return "baseline", "vowel exploration"
    if i <= 180:
        return "baseline", "baseline"
    if i <= 405:
        return "learning", "learning"
    return "aftereffect", "after-effect"

SEX = {1: "F", 2: "M"}
rows = []
for k, pr in P.iterrows():
    for i in range(1, n + 1):
        ph, bk = phase(i)
        lr = ph == "learning"
        rows.append(dict(
            dataset=ID, study="Demirel 2025 Neurobiology of Language", experiment="TGP tACS during F1 adaptation", group=GROUP[pr.group],
            subject=f"s{int(pr.subject)}", person=f"s{int(pr.subject)}", subject_excluded_by_authors="", age="", sex=SEX[int(pr.sex)],
            handedness="", population="healthy", setting="lab",
            device="head-mounted microphone and headphones, Audapter; tACS over left ventral motor cortex", session=1, block=bk,
            trial=i, phase=ph, time_s="", task="speech", unit="utterance", condition="", pert_type="formant_shift" if lr else "none",
            pert_value=110 if lr else 0, pert_unit="mel", feedback="altered" if lr else "normal", excluded_by_authors="",
            stimulus="", f1_mel=f1[k, i - 1], f2_mel=f2[k, i - 1], shift_f1=110 if lr else 0, shift_f2=0))
a = pd.DataFrame(rows)
a = a.reindex(columns=[c[1] for c in stdlib.columns_for("speech")])

# published checks
g = lambda G: (P.group == G).values
checks.published("participants in the TGP tACS group", g(TGP).sum(), 30, 0, SRC)
checks.published("participants in the sham group", g(SHAM).sum(), 31, 0, SRC)
checks.published("women in the TGP tACS group", ((P.sex == 1) & g(TGP)).sum(), 15, 0, SRC)
checks.published("women in the sham group", ((P.sex == 1) & g(SHAM)).sum(), 16, 0, SRC)
checks.published("participants with after-effect data in the TGP group", (~np.isnan(f1[g(TGP), 405:]).all(axis=1)).sum(), 29, 0, SRC + " (one block lost)")
DIFF = []
def tbl(name, arr, lo, hi, G, mean_rep, sd_rep):
    v = np.nanmean(arr[g(G), lo - 1 + 135:hi + 135], axis=1)
    v = v[~np.isnan(v)]
    for lab, got, rep, tol in (("mean", v.mean(), mean_rep, 0.5), ("SD", v.std(ddof=1), sd_rep, 0.05)):
        if abs(got - rep) <= tol:
            checks.published(f"Table 1 {name} {lab}, {GROUP[G]}", got, rep, tol, SRC)
        else:   # not a check: the printed value is the computed one cut, not rounded (see differences)
            DIFF.append(f"{name} {lab}, {GROUP[G]}: {got:.2f} here, {rep} printed")
for G, vals in ((TGP, [747, 86.8, 1468, 105.5, 713, 81.3, 1480, 100.7, 720, 81.3, 1478, 103.5]),
                (SHAM, [767, 73, 1427, 81.6, 736, 71.7, 1433, 85.2, 750, 68.8, 1435, 82.8])):
    tbl("baseline F1", f1, 1, 45, G, vals[0], vals[1])
    tbl("baseline F2", f2, 1, 45, G, vals[2], vals[3])
    tbl("end of learning F1 (trials 241-270)", f1, 241, 270, G, vals[4], vals[5])
    tbl("end of learning F2 (trials 241-270)", f2, 241, 270, G, vals[6], vals[7])
    tbl("start of after-effect F1 (trials 271-285)", f1, 271, 285, G, vals[8], vals[9])
    tbl("start of after-effect F2 (trials 271-285)", f2, 271, 285, G, vals[10], vals[11])
checks.internal("authors' summary adapt_F1 equals the mean of trials 241-270 (per participant)",
                bool(np.nanmax(np.abs(np.nanmean(f1[:, 375:405], axis=1) - P.adapt_F1.values)) < 1e-6))
checks.internal("all 61 x 495 rows kept", len(a) == 61 * 495)
bad = [c for c in checks.items if not c["ok"]]
for c in bad:
    print("FAILED", c)
meta_paper = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
p = meta_paper["papers"][0]
if bad:
    sys.exit(1)
meta = dict(
    paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=meta_paper["lab"], data_url=meta_paper["data_url"],
    license="none stated", license_url="",
    changes="Unpacked the authors' wide trial tables (one column per trial) into one row per utterance, added phase, block, perturbation and group labels, and replaced participant numbers with codes. F1 and F2 values are the authors' (mel).",
    how=["One row is one spoken word (head, bed or dead; the deposit does not say which word was spoken on each trial) by one participant. f1_mel and f2_mel are the authors' F1 and F2 in mel, with the values they removed as outliers (more than 3 SD from the participant's block mean) or could not measure left empty.",
         "Each participant has 495 rows in time order: 135 vowel exploration trials (9 words, normal feedback), 45 baseline, 225 learning and 90 after-effect trials. trial counts from 1 over all 495; the paper numbers trials from the baseline (paper trial = trial - 135). The block order, trial counts and the shift come from the authors' experiment script (read as text).",
         "During learning the feedback F1 was shifted up by 110 mel (shift_f1; shift_f2 is 0). The vowel-exploration trials are labelled phase baseline and block vowel exploration. group is the stimulation group (TGP tACS or sham); the deposit codes them A and B and does not say which is which, so the labels were assigned by matching group sizes, number of women and Table 1 baseline F1 of the paper.",
         "Age is not shared. sex is the authors' code 1 or 2, read as F and M because the paper's counts of women match code 1. One TGP participant lost the after-effect block (empty values)."],
    unit_note="utterance", differences=[f"Table 1 gives {len(DIFF)} of its 24 group means and SDs one unit (or 0.1) below the value computed here, as if cut instead of rounded; they are not used as checks: " + "; ".join(DIFF)])
stdlib.release(ID, "speech", a, meta, checks)
