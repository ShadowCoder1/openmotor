"""Cheng, Niziolek & Parrell (2026), J Speech Lang Hear Res 69(3):920-931: adaptation to a nonuniform "centralization"
formant perturbation in 43 speakers aged 18-73, and generalization of the learning to untrained vowels.

Raw files, raw_extra/Cheng2026_JSLHR/ (OSF jx6fy, fetched with tools/std/fetch.py):
  vsaGen2vowelspaceTable_50_100_window2025-02-19.csv   one row per trial: subj, trialnum (1-510), vowel (ARPABET code of the
      word), condition (the five phases), F1, F2 (mels, mean over 50-100 ms after vowel onset), FC1, FC2 (the participant's
      baseline vowel-space center, mels), FPertC1, FPertC2 (the center of the participant's perturbation field, mels),
      vowel_dur (seconds); trials the authors discarded have empty F1 and F2
  subject_age.csv     subj, age, age_group, gender for the 43 participants
  hearingdata.csv     pure-tone thresholds (dB HL) per ear at 250, 500, 1000, 2000 and 4000 Hz for 50 listeners (the column
                      for 2000 Hz is labelled 20000; the authors' code reads it as 2000 Hz); 43 of them are in the trial table
  vowel_space_adapt_generalization_analysis.Rmd   the authors' analysis, read as text to reproduce the paper's numbers
What the design is (paper, Method; table of phases): generalization baseline (6 untrained words x 15, noise-only feedback,
trials 1-90), baseline (4 trained corner-vowel words x 10, veridical feedback, 91-130), ramp (x 10, perturbation 0 to 50%,
131-170), hold (x 70, perturbation 50%, 171-450), generalization (6 untrained words x 10, noise-only feedback, 451-510).
The perturbation moves F1 and F2 toward the center of the participant's vowel space by a percentage of the distance to it.
Run from the site folder: python3 -I tools/std/convert/Cheng2026_JSLHR.py
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cheng2026_JSLHR"
SRC = "Cheng, Niziolek & Parrell 2026, J Speech Lang Hear Res"
FIRST = 30  # the authors' code drops trialnum <= 30 (the paper's text says the first 25 trials; see differences)
# ARPABET code of the authors' table -> (word, vowel, trained?), words and vowels from the paper's Methods
WORD = {"iy": ("bead", "vowel /i/", True), "ae": ("bad", "vowel /æ/", True), "aa": ("bod", "vowel /ɑ/", True),
        "uw": ("booed", "vowel /u/", True), "ih": ("bid", "vowel /ɪ/", False), "ey": ("bayed", "diphthong /eɪ/", False),
        "eh": ("bed", "vowel /ɛ/", False), "ah": ("bud", "vowel /ʌ/", False), "ow": ("bode", "diphthong /oʊ/", False),
        "ay": ("bide", "diphthong /ɑɪ/", False)}
TRAINED = ["iy", "ae", "aa", "uw"]
UNTRAINED = ["ih", "ey", "eh", "ah", "ow", "ay"]
NOISE_ONLY = ("speech-shaped noise only, no speech feedback; the noise amplitude was scaled in real time by the "
              "participant's speech amplitude (about 80 dB)")
NOISE_MIX = "perturbed speech feedback mixed with speech-shaped noise at 60 dB SPL"
# condition -> (phase, pert_type, pert_value, feedback, masking_noise)
BLOCK = {"baselineGeneralize": ("baseline", "none", 0, "masked", NOISE_ONLY),
         "baselineTrain": ("baseline", "none", 0, "normal", ""),
         "ramp": ("learning", "formant_centralization", "", "altered", NOISE_MIX),
         "train": ("learning", "formant_centralization", 50, "altered", NOISE_MIX),
         "generalization": ("transfer", "none", 0, "masked", NOISE_ONLY)}
PERT_UNIT = "% of the distance to the vowel-space center"
DEVICE = ("AKG C520 head-mounted microphone, Focusrite Scarlett 2i2 interface, Audapter, "
          "Beyerdynamic DT 770 closed-back headphones")
REPS = {"baselineGeneralize": 15, "baselineTrain": 10, "ramp": 10, "train": 70, "generalization": 10}
FREQS = ["250", "500", "1000", "20000", "4000"]  # the authors' header "20000" is 2000 Hz in their code
HZ = {"250": 250, "500": 500, "1000": 1000, "20000": 2000, "4000": 4000}


def tol_of(printed):
    """Half a unit of the last printed digit."""
    s = str(printed)
    return 0.5 * 10 ** -(len(s.split(".")[1]) if "." in s else 0) + 1e-6


def tol_unit(printed):
    """One unit of the last printed digit: the paper truncates in places (5.0989 is printed 5.09)."""
    s = str(printed)
    return 10 ** -(len(s.split(".")[1]) if "." in s else 0) + 1e-6


def tol_se(printed):
    s = str(printed)
    return 0.5 + 1e-6 if (s.endswith(".0") or "." not in s) else tol_unit(s)


def tol_mean(printed):
    """The paper prints most means in % as whole numbers (4.0, 5.0, 7.0, 10, 13): tolerance half a percentage point."""
    s = str(printed)
    return 0.5 + 1e-6 if (s.endswith(".0") or "." not in s) else tol_of(s)


def ntile(n, k):
    return np.floor(k * np.arange(n) / n).astype(int) + 1  # dplyr::ntile on an ordered vector of unique trial numbers


def authors_analysis(a, first=FIRST):
    """The authors' pipeline (their Rmd): trials above `first`, 7 bins of the 280 hold trials, mean F1 and F2 per participant,
    phase, bin and vowel, average pairwise distance between the vowels' means (AVS) normalized to the baseline phase, and the
    distance of each vowel's productions to the baseline center, normalized the same way. Returns normalized values."""
    d = a[a.trial > first].copy()
    d["bin"] = 1
    for _, g in d[d.block == "train"].groupby("subject", sort=False):
        d.loc[g.index, "bin"] = ntile(len(g), 7)
    d["dist"] = np.hypot(d.f1 - d.fc1, d.f2 - d.fc2)
    m = d.groupby(["subject", "block", "bin", "vowel"], sort=False).agg(F1=("f1", "mean"), F2=("f2", "mean"),
                                                                       dist=("dist", "mean")).reset_index()

    def avs(g):
        p = g[["F1", "F2"]].to_numpy()
        return float(np.mean([np.hypot(*(p[i] - p[j])) for i in range(len(p)) for j in range(i + 1, len(p))]))

    out = {}
    for name, blocks, vowels, base in (("trained", ["baselineTrain", "ramp", "train"], TRAINED, "baselineTrain"),
                                       ("untrained", ["baselineGeneralize", "generalization"], UNTRAINED[:-1], "baselineGeneralize"),
                                       ("untrained_all", ["baselineGeneralize", "generalization"], UNTRAINED, "baselineGeneralize")):
        s = m[m.block.isin(blocks) & m.vowel.isin(vowels)].groupby(["subject", "block", "bin"]).apply(avs, include_groups=False)
        s = s.rename("avs").reset_index()
        b = s[(s.block == base) & (s.bin == 1)].set_index("subject").avs
        s["normed"] = s.avs / s.subject.map(b)
        out[name] = s
    for name, vowels, base in (("v_trained", TRAINED, "baselineTrain"), ("v_untrained", UNTRAINED, "baselineGeneralize")):
        s = m[m.vowel.isin(vowels)].copy()
        b = s[(s.block == base) & (s.bin == 1)][["subject", "vowel", "dist"]].rename(columns={"dist": "base"})
        s = s.merge(b, on=["subject", "vowel"], how="left")
        s["normed"] = s.dist / s.base
        out[name] = s
    return out


def pct(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    mean, se = (x.mean() - 1) * 100, x.std(ddof=1) / math.sqrt(len(x)) * 100
    return mean, se, mean / se


def published_checks(a, ages, checks):
    P = f"{SRC}, Method (Participants)"
    ppl = a.drop_duplicates("subject")
    checks.published("participants", len(ppl), 43, 0, P)
    A = ages.set_index("subj")
    for grp, n, lo, hi, mean, sd, f, mm in (("younger", 16, 18, 29, "21.2", "2.91", 15, 1), ("middle aged", 13, 33, 54, "41.5", "7.82", 7, 6),
                                            ("older", 14, 58, 73, "65", "5.07", 9, 5)):
        g = A[A.age_group == grp]
        checks.published(f"{grp}: participants", len(g), n, 0, P)
        checks.published(f"{grp}: youngest (years)", g.age.min(), lo, 0, P)
        checks.published(f"{grp}: oldest (years)", g.age.max(), hi, 0, P)
        checks.published(f"{grp}: mean age (years)", g.age.mean(), float(mean), tol_of(mean), P)
        checks.published(f"{grp}: SD of age (years)", g.age.std(ddof=1), float(sd), tol_of(sd), P)
        checks.published(f"{grp}: females", (g.gender == "F").sum(), f, 0, P)
        checks.published(f"{grp}: males", (g.gender == "M").sum(), mm, 0, P)
    checks.published("overall age range, youngest (years)", ages.age.min(), 18, 0, f"{SRC}, Abstract")
    checks.published("overall age range, oldest (years)", ages.age.max(), 73, 0, f"{SRC}, Abstract")
    T = f"{SRC}, Method (Procedure)"
    per = a[a.subject == a.subject.iloc[0]].groupby("block", sort=False).size()
    for blk, n in (("baselineGeneralize", 90), ("baselineTrain", 40), ("ramp", 40), ("train", 280), ("generalization", 60)):
        checks.published(f"trials per participant in the {blk} phase", per[blk], n, 0, T)
    D = f"{SRC}, Method (Data analysis)"
    pc = 100 * a.groupby("subject").f1.apply(lambda s: s.isna().mean())
    checks.published("discarded trials, % of all trials (the paper's 1.8 is 1.865 here; see differences)",
                     100 * a.f1.isna().mean(), 1.8, 0.1, D)
    checks.published("discarded trials, lowest % across participants", pc.min(), 0, 0, D)
    checks.published("discarded trials, highest % across participants (6)", pc.max(), 6, 0.5, D)

    R = f"{SRC}, Results"
    res = authors_analysis(a)
    tr, un, ua = res["trained"], res["untrained"], res["untrained_all"]
    for lab, s, rep in (("early adaptation (hold, first 10 repetitions)", tr[(tr.block == "train") & (tr.bin == 1)], ("4.0", "0.9", "4.64")),
                        ("late adaptation (last 10 repetitions of the hold)", tr[(tr.block == "train") & (tr.bin == 7)], ("5.0", "1.0", "5.18")),
                        ("generalization to the untrained words", un[un.block == "generalization"], ("7.0", "1.3", "5.62"))):
        m_, se, t = pct(s.normed)
        for what, v, r, tl in (("mean AVS change (%)", m_, rep[0], tol_mean(rep[0])), ("SE (%)", se, rep[1], tol_se(rep[1])),
                               ("t(42)", t, rep[2], tol_unit(rep[2]))):
            checks.published(f"{lab}: {what}", v, float(r), tl, f"{R}, Sensorimotor adaptation / Generalization")
    ta, ga = tr[(tr.block == "train") & (tr.bin == 7)].set_index("subject").normed, \
        un[un.block == "generalization"].set_index("subject").normed
    rho = stats.spearmanr(ta.loc[ga.index], ga).statistic
    checks.published("Spearman correlation of adaptation with generalization", rho, 0.45, tol_of("0.45"), f"{R}, Generalization")
    vt, vu = res["v_trained"], res["v_untrained"]
    names = {"iy": "/i/", "ae": "/æ/", "aa": "/ɑ/", "uw": "/u/", "ih": "/ɪ/", "ey": "/eɪ/", "eh": "/ɛ/", "ah": "/ʌ/", "ow": "/oʊ/"}
    table = [("initial learning", vt[(vt.block == "train") & (vt.bin == 1)], {"iy": ("5.0", "1.1", "4.25"), "ae": ("10", "1.9", "5.09"),
                                                                              "aa": ("4.0", "1.2", "2.97"), "uw": ("2.0", "2.7", "0.8")}),
             ("maximal adaptation", vt[(vt.block == "train") & (vt.bin == 7)], {"iy": ("5.0", "1.3", "4.04"), "ae": ("13", "1.7", "7.64"),
                                                                                 "aa": ("8.0", "1.4", "5.88"), "uw": ("-0.9", "3.0", "-0.31")}),
             ("generalization", vu[vu.block == "generalization"], {"ih": ("9.0", "2.1", "4.44"), "ey": ("7.0", "2.0", "3.49"),
                                                                    "eh": ("14", "2", "5.51"), "ah": ("11", "1.5", "7.59"),
                                                                    "ow": ("5.0", "2.1", "2.46")})]
    for lab, s, rep in table:
        for v, (rm, rse, rt) in rep.items():
            m_, se, t = pct(s[s.vowel == v].normed)
            for what, x, r, tl in (("distance to the center, change (%)" + (" (sign taken from the reported t)" if rm == "-0.9" else ""), m_, rm, tol_mean(rm)), ("SE (%)", se, rse, tol_se(rse)),
                                   ("t(42)", t, rt, tol_unit(rt))):
                checks.published(f"{lab}, {names[v]}: {what}", x, float(r), tl, f"{R}, Vowel-specific {lab}")
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    d = pd.read_csv(raw / "vsaGen2vowelspaceTable_50_100_window2025-02-19.csv")
    ages = pd.read_csv(raw / "subject_age.csv")
    hear = pd.read_csv(raw / "hearingdata.csv")
    d["subj"] = d.subj.astype(str)
    ages["subj"] = ages.subj.astype(str)
    hear["subj"] = hear.subj.astype(str)

    # internal checks on the table
    checks.internal("43 participants x 510 trials, trials 1-510 in order, participants in blocks",
                    d.subj.nunique() == 43 and (d.groupby("subj").size() == 510).all()
                    and all(list(g.trialnum) == list(range(1, 511)) for _, g in d.groupby("subj", sort=False))
                    and (d.subj != d.subj.shift()).sum() == 43)
    order = ["baselineGeneralize", "baselineTrain", "ramp", "train", "generalization"]
    ranges = {"baselineGeneralize": (1, 90), "baselineTrain": (91, 130), "ramp": (131, 170), "train": (171, 450), "generalization": (451, 510)}
    checks.internal("phases in the order of the paper's table, at trials 1-90, 91-130, 131-170, 171-450 and 451-510",
                    all(g.condition.drop_duplicates().tolist() == order for _, g in d.groupby("subj", sort=False))
                    and all(d[d.condition == c].trialnum.between(*r).all() and (d.condition == c).sum() == 43 * (r[1] - r[0] + 1)
                            for c, r in ranges.items()))
    ct = pd.crosstab(d.condition, d.vowel)
    checks.internal("repetitions per word and participant: 15, 10, 10, 70 and 10 in the five phases; trained words in the three "
                    "middle phases, untrained words in the two noise-only phases",
                    all((ct.loc[c, TRAINED if c in ("baselineTrain", "ramp", "train") else UNTRAINED] == 43 * REPS[c]).all()
                        and (ct.loc[c, UNTRAINED if c in ("baselineTrain", "ramp", "train") else TRAINED] == 0).all() for c in order))
    na = d[["F1", "F2", "vowel_dur"]].isna()
    odd = d[na.F1 & ~na.vowel_dur]
    checks.internal("F1 and F2 are missing together on 409 trials; vowel duration is also missing on all but one of them",
                    bool((na.F1 == na.F2).all() and na.F1.sum() == 409 and len(odd) == 1 and not (na.vowel_dur & ~na.F1).any()),
                    f"{len(odd)} trial has a duration but no formants")
    checks.internal("the four center coordinates are constant within a participant and never missing",
                    bool(d.groupby("subj")[["FC1", "FC2", "FPertC1", "FPertC2"]].nunique().eq(1).all().all()
                         and not d[["FC1", "FC2", "FPertC1", "FPertC2"]].isna().any().any()))
    for c, (lo, hi) in dict(F1=(250, 1300), F2=(700, 2000), vowel_dur=(0.01, 1.0)).items():
        v = d[c].dropna()
        checks.internal(f"{c} within {lo}-{hi}", bool(v.between(lo, hi).all()), f"{v.min():.3f}-{v.max():.3f}")
    bm = d[d.condition.isin(["baselineTrain", "baselineGeneralize"])].groupby("vowel")[["F1", "F2"]].mean()
    checks.internal("vowel codes match the words: at baseline /i/ has the lowest F1 and highest F2 of the trained words, /ɑ/ and /æ/ "
                    "the highest F1, /u/ and /ɑ/ the lowest F2",
                    bool(bm.F1[TRAINED].idxmin() == "iy" and set(bm.F1[TRAINED].nlargest(2).index) == {"aa", "ae"}
                         and bm.F2[TRAINED].idxmax() == "iy" and set(bm.F2[TRAINED].nsmallest(2).index) == {"uw", "aa"}))
    checks.internal("every participant is in the age table and in the hearing table",
                    set(d.subj) == set(ages.subj) and set(d.subj) <= set(hear.subj))
    checks.internal("the hearing table has one left and one right ear row per listener", bool((hear.groupby("subj").ears.nunique() == 2).all()))

    A = ages.set_index("subj")
    rows = []
    for r in d.itertuples(index=False):
        phase, ptype, pval, fb, noise = BLOCK[r.condition]
        word, seg, trained = WORD[r.vowel]
        miss = pd.isna(r.F1)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "", "group": A.age_group[r.subj], "subject": r.subj, "person": r.subj,
            "subject_excluded_by_authors": False, "age": int(A.age[r.subj]), "sex": A.gender[r.subj], "handedness": "",
            "population": "healthy", "setting": "lab", "device": DEVICE, "session": 1, "block": r.condition,
            "trial": int(r.trialnum), "phase": phase, "time_s": "", "task": "speech", "unit": "utterance",
            "condition": "trained word" if trained else "untrained word", "pert_type": ptype, "pert_value": pval,
            "pert_unit": PERT_UNIT, "feedback": fb, "excluded_by_authors": bool(miss or r.trialnum <= FIRST),
            "stimulus": word, "segment": seg, "measure_window": "50-100 ms after vowel onset",
            "duration_ms": "" if pd.isna(r.vowel_dur) else float(r.vowel_dur) * 1000.0,
            "f1_mel": "" if miss else float(r.F1), "f2_mel": "" if miss else float(r.F2), "masking_noise": noise,
            "center_f1_mel": float(r.FC1), "center_f2_mel": float(r.FC2),
            "pert_center_f1_mel": float(r.FPertC1), "pert_center_f2_mel": float(r.FPertC2),
        })
    a = pd.DataFrame(rows)
    a["vowel"] = d.vowel.values  # the authors' code for the word, for the reproduction of their analysis
    a["f1"], a["f2"] = pd.to_numeric(a.f1_mel.replace("", np.nan)), pd.to_numeric(a.f2_mel.replace("", np.nan))
    a["fc1"], a["fc2"] = a.center_f1_mel.astype(float), a.center_f2_mel.astype(float)
    checks.internal("excluded_by_authors = no formants or trial 1-30",
                    bool((a.excluded_by_authors == (a.f1.isna() | (a.trial <= FIRST))).all()),
                    f"{int(a.excluded_by_authors.sum())} rows marked")
    res = published_checks(a, ages, checks)
    # the paper's text and the authors' code differ on how many generalization baseline trials were dropped: record which one
    # reproduces the published generalization numbers
    alt = authors_analysis(a, first=25)["untrained"]
    alt = alt[alt.block == "generalization"]
    base = res["untrained"][res["untrained"].block == "generalization"]
    ua = res["untrained_all"]
    ua = ua[ua.block == "generalization"]
    print(f"generalization AVS change with the diphthong /ɑɪ/ included: {pct(ua.normed)[0]:.4f} (SE {pct(ua.normed)[1]:.4f}, t {pct(ua.normed)[2]:.4f})")
    print(f"generalization AVS change: first {FIRST} trials dropped {pct(base.normed)[0]:.4f} (t {pct(base.normed)[2]:.4f}); "
          f"first 25 dropped {pct(alt.normed)[0]:.4f} (t {pct(alt.normed)[2]:.4f})")
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    paper = {k: s["papers"][0][k] for k in ("authors", "year", "title", "venue", "doi")}
    # participants.csv: hearing thresholds of the 43 listeners
    pcols = []
    for ear in ("left", "right"):
        for f in FREQS:
            pcols.append((f"hearing_{ear}_{HZ[f]}hz", "number", "dB HL",
                          f"Pure-tone threshold of the {ear} ear at {HZ[f]} Hz, from the authors' hearing table."
                          + (" The authors' file labels this column 20000; their code uses it as 2000 Hz." if f == "20000" else "")))
    pcols.append(("pta_db_hl", "number", "dB HL",
                  "Pure-tone average of the thresholds at 500, 1000 and 2000 Hz over both ears (the mean of the six values, "
                  "ignoring missing ones), the hearing measure of the paper, computed as the authors' code does."))
    participants = []
    for subj in d.subj.drop_duplicates():
        h = hear[hear.subj == subj].set_index("ears")
        rec = dict(person=subj)
        for ear in ("left", "right"):
            for f in FREQS:
                rec[f"hearing_{ear}_{HZ[f]}hz"] = h.loc[ear, f]
        rec["pta_db_hl"] = float(pd.Series(h[["500", "1000", "20000"]].to_numpy(dtype=float).ravel()).mean())
        participants.append(rec)
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Took the authors' per-trial table of all 43 participants as shared (rows, order, mel values), converted vowel "
                 "duration from seconds to ms, named the words and vowels from the paper's Methods, added age, sex and age group "
                 "from their age table, marked the 409 trials without formants and the first 30 trials of each participant as "
                 "excluded (the authors' code drops them), put the hearing thresholds in participants.csv and replaced "
                 "participant identifiers with codes."),
        how=[
            "One row is one word spoken by one participant, in the authors' trial order (trial 1-510): 90 trials of six untrained "
            "words with noise-only feedback (block baselineGeneralize), 40 baseline trials of the four trained corner-vowel words "
            "with normal feedback (baselineTrain), 40 ramp trials (ramp) and 280 hold trials (train) with the feedback shifted "
            "toward the center of the participant's vowel space, then 60 test trials of the untrained words with noise-only "
            "feedback (generalization).",
            "f1_mel and f2_mel are the authors' F1 and F2 averaged over 50-100 ms after vowel onset, in mels; empty values are "
            "trials the authors discarded. center_f1_mel and center_f2_mel (the authors' baseline vowel-space center) and "
            "pert_center_f1_mel and pert_center_f2_mel (the center of the perturbation field) are constant per participant. "
            "group is the authors' age group.",
            "pert_value is the shift in % of the distance to the vowel-space center: 50 in the hold, 0 in the other phases "
            "(on noise-only trials no speech feedback was heard); during the ramp it grew from 0 to 50 and the authors did not "
            "share the trial-by-trial value, so it is empty. The shift in mels depends on where the produced vowel lies.",
            "excluded_by_authors marks the trials without formants and trials 1-30 (the authors' code drops the first five "
            "repetitions of each generalization baseline word; the paper's text says the first 25 trials). The paper reports a "
            "processing latency under 18 ms; masking_noise is empty for the baseline phase because the paper does not say.",
        ],
        unit_note="spoken word (utterance) of one participant",
        differences=[
            "Discarded trials: the paper gives 1.8% in total; the table has 409 of 21,930 trials without formants, 1.865%.",
            "The paper's text says the first 25 trials of the generalization baseline were excluded; the authors' code drops "
            "trials 1-30, which is what excluded_by_authors marks (dropping 30 reproduces the paper's generalization t = 5.62, "
            "dropping 25 gives 5.65).",
            "Many mean changes in % are printed as whole percents (for example 7.0% for generalization, where the rows give 7.5% "
            "with the paper's own SE of 1.3 and t of 5.62), so those means are checked to half a percentage point; the t "
            "statistics and SEs agree to the printed decimals, a few of them truncated rather than rounded (t = 5.0989 is "
            "printed 5.09, SE = 2.76 is printed 2.7).",
            "Vowel /u/, maximal adaptation: the change is -0.93% here, with the paper's t of -0.31; the sign of the paper's "
            "mean was taken from that t.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks, participants=participants, participant_columns=pcols)


if __name__ == "__main__":
    main()
