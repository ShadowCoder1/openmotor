"""Murphy, Cheng, Niziolek & Parrell (2026), JASA Express Letters 6(7):075205: vowel formants of 62 speakers producing
heed, who'd, had and hod without and then with a bite block (small 2 mm horizontal block for had/hod, large 17.5 mm
vertical block for heed/who'd), with auditory feedback masked by loud noise on every trial.

Raw files, raw_extra/Murphy2026_JASAEL/ (OSF kzhtm, fetched with tools/std/fetch.py):
  biteblock_metricsTable_2025-11-17.csv   subj, trialnum, vowel, condition, F1, F2, vowel_dur   62 participants x 96 trials
  processed_biteBlock_fromVSA_data.csv    the same trials of the 55 analysed participants with demographics and the
                                          authors' derived columns (baseline medians, differences, detrended SDs, ...)
  biteBlock_demos.csv                     age, sex, gender, ethnicity, race, parent study and a date per participant
                                          (only age, sex and parent study are used; dates and ethnicity are not copied)
What the values are (paper, Methods 2.2-2.4): F1 and F2 in mels, the mean over the middle 50% of the vowel (Praat LPC
in wave_viewer, boundaries from the Montreal Forced Aligner, hand-checked); vowel_dur in seconds; condition baseline
(trials 1-32, 8 of each word), vert = large bite block (heed, who'd) and horiz = small bite block (had, hod), 32 trials
each in counterbalanced order; empty formants = trials the authors could not measure.
One participant appears in the metrics table under a different identifier than in the processed table and the
demographics (all 96 trials identical); the converter gives that participant the processed table's identifier.
Run from the site folder: python3 -I tools/std/convert/Murphy2026_JASAEL.py
"""
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Murphy2026_JASAEL"
WORD = {"iy": "heed", "uw": "who'd", "ae": "had", "aa": "hod"}
SEG = {"iy": "vowel /i/", "uw": "vowel /u/", "ae": "vowel /æ/", "aa": "vowel /ɑ/"}
BLOCK = {"baseline": ("baseline", "none", 0.0, "baseline"),
         "vert": ("large bite block", "bite_block", 17.5, "learning"),
         "horiz": ("small bite block", "bite_block", 2.0, "learning")}
CRIT = {"iy": "vert", "uw": "vert", "ae": "horiz", "aa": "horiz"}  # the block in which each word was spoken
DEVICE = "Sennheiser MKE 600 microphone, Audapter (masking noise), Beyerdynamic DT 770 Pro headphones"
MASKING = ("speech-shaped noise, about 80 dB SPL, on every trial, scaled in real time by the speech amplitude so that it "
           "played only while the participant spoke; no speech feedback")
SRC = "Murphy et al. 2026, JASA Express Letters"


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def load(checks):
    raw = stdlib.raw_dir(ID)
    m = pd.read_csv(raw / "biteblock_metricsTable_2025-11-17.csv")
    p = pd.read_csv(raw / "processed_biteBlock_fromVSA_data.csv")
    d = pd.read_csv(raw / "biteBlock_demos.csv")
    analysed = set(p.subj)
    only_m = [s for s in m.subj.drop_duplicates() if s not in analysed and s not in set(d.SubjID)]
    only_p = [s for s in p.subj.drop_duplicates() if s not in set(m.subj)]
    ok = len(only_m) == 1 and len(only_p) == 1
    if ok:
        a = m[m.subj == only_m[0]].sort_values("trialnum")
        b = p[p.subj == only_p[0]].sort_values("trialnum")
        ok = all(np.allclose(a[c].values, b[c].values, equal_nan=True) for c in ("F1", "F2", "vowel_dur")) and \
            (a.trialnum.values == b.trialnum.values).all() and (a.vowel.values == b.vowel.values).all() and \
            (a.condition.values == b.condition.values).all()
    checks.internal("one participant has a different identifier in the metrics table than in the processed table and the "
                    "demographics, with all 96 trials identical; it is matched", ok)
    assert ok
    m["pid"] = m.subj.replace({only_m[0]: only_p[0]})
    dd = d.groupby("SubjID").agg(Age=("Age", "first"), Sex=("Sex", "first"), Study=("Q20", "first"),
                                 nAge=("Age", "nunique"), nSex=("Sex", "nunique"))
    checks.internal("demographics cover all 62 participants and repeated rows agree",
                    set(m.pid) <= set(dd.index) and (dd.nAge == 1).all() and (dd.nSex == 1).all())
    pa = p.drop_duplicates("subj").set_index("subj")
    checks.internal("age, sex and parent study in the processed table equal the demographics file",
                    bool((pa.Age == dd.loc[pa.index, "Age"]).all() and (pa.Sex == dd.loc[pa.index, "Sex"]).all()
                         and (pa.Study == dd.loc[pa.index, "Study"]).all()))
    mm = m[m.pid.isin(analysed)].merge(p[["subj", "trialnum", "vowel", "condition", "F1", "F2", "vowel_dur"]],
                                       left_on=["pid", "trialnum"], right_on=["subj", "trialnum"], suffixes=("", "_p"))
    same = len(mm) == len(p) and all(np.allclose(mm[c], mm[c + "_p"], equal_nan=True) for c in ("F1", "F2", "vowel_dur")) \
        and (mm.vowel == mm.vowel_p).all() and (mm.condition == mm.condition_p).all()
    checks.internal("the 55 analysed participants' trials in the metrics table equal the processed table (5280 trials)",
                    bool(same), f"{len(mm)} matched")
    return m, p, dd, analysed


def build(m, dd, analysed):
    rows = []
    for r in m.itertuples(index=False):
        cond, ptype, pval, phase = BLOCK[r.condition]
        demo = dd.loc[r.pid]
        f1, f2, dur = (float(r.F1), float(r.F2), float(r.vowel_dur))
        missing = math.isnan(f1)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "", "group": "", "subject": r.pid, "person": r.pid,
            "subject_excluded_by_authors": r.pid not in analysed, "age": int(demo.Age),
            "sex": {"Female": "F", "Male": "M"}[demo.Sex], "handedness": "", "population": "healthy", "setting": "lab",
            "device": DEVICE, "session": 1, "block": r.condition, "trial": int(r.trialnum), "phase": phase, "time_s": "",
            "task": "speech", "unit": "utterance", "condition": cond, "pert_type": ptype, "pert_value": pval,
            "pert_unit": "mm", "feedback": "masked", "excluded_by_authors": missing,
            "stimulus": WORD[r.vowel], "segment": SEG[r.vowel], "measure_window": "middle 50% of the vowel",
            "duration_ms": "" if math.isnan(dur) else dur * 1000.0, "f0_hz": "", "f1_hz": "", "f2_hz": "",
            "f1_mel": "" if missing else f1, "f2_mel": "" if math.isnan(f2) else f2, "intensity_au": "",
            "shift_f1": "", "shift_f2": "", "masking_noise": MASKING, "feedback_latency_ms": "",
        })
    return rows


def internal_checks(a, checks, raw):
    checks.internal("5952 rows = 62 participants x 96 trials, trials 1-96 complete and in order",
                    len(a) == 5952 and a.subject.nunique() == 62
                    and all(list(g.trial) == list(range(1, 97)) for _, g in a.groupby("subject", sort=False)))
    ph = a.groupby(["subject", "block"]).trial.agg(["min", "max", "count"]).reset_index()
    bl = ph[ph.block == "baseline"]
    bb = ph[ph.block != "baseline"]
    checks.internal("baseline = trials 1-32, each bite block 32 trials at 33-64 or 65-96",
                    bool((bl["min"] == 1).all() and (bl["max"] == 32).all() and (bb["count"] == 32).all()
                         and bb["min"].isin([33, 65]).all()))
    base = a[a.block == "baseline"].groupby("subject").stimulus.value_counts()
    checks.internal("baseline: 8 trials of each of the 4 words per participant", bool((base == 8).all() and len(base) == 62 * 4))
    w = a[a.block != "baseline"].copy()
    w["match"] = w.stimulus.map({v: k for k, v in WORD.items()}).map(CRIT) == w.block
    per = w.groupby("subject").match.mean()
    odd = per[per < 1]
    checks.internal("bite block phases hold their two words (16 trials each) for every participant but one excluded "
                    "participant, whose block labels and words are swapped (the paper: bite blocks of incorrect sizes)",
                    bool(len(odd) == 1 and odd.iloc[0] == 0 and a[a.subject == odd.index[0]].subject_excluded_by_authors.all()
                         and (w.groupby(["subject", "block", "stimulus"]).size() == 16).all()))
    checks.internal("formants and duration are missing together, on 122 trials, which are marked excluded_by_authors",
                    bool(((a.f1_mel == "") == (a.f2_mel == "")).all() and ((a.f1_mel == "") == (a.duration_ms == "")).all()
                         and (a.f1_mel == "").sum() == 122 and (a.excluded_by_authors == (a.f1_mel == "")).all()))
    v = a[a.f1_mel != ""]
    for c, (lo, hi) in dict(f1_mel=(250, 1300), f2_mel=(600, 2100), duration_ms=(20, 1100)).items():
        x = v[c].astype(float)
        checks.internal(f"{c} within {lo}-{hi}", bool(x.between(lo, hi).all()), f"{x.min():.2f}-{x.max():.2f}")
    s = pd.read_csv(raw / "postdrop_demo_summary.csv").iloc[0]
    an = a[~a.subject_excluded_by_authors].drop_duplicates("subject")
    checks.internal("the authors' summary of the 55 analysed participants (postdrop_demo_summary.csv): 36 from vsaGeneralize2, "
                    "19 from vsaRetention, 43 F / 12 M, mean age and SD",
                    bool(an.study_.value_counts().to_dict() == {"vsaGeneralize2": int(s.vsaGen2), "vsaRetention": int(s.vsaRetention)}
                         and (an.sex == "F").sum() == s.Sex_F and (an.sex == "M").sum() == s.Sex_M
                         and abs(an.age.mean() - s.Age_mean) < 1e-9 and abs(an.age.std() - s.Age_std) < 1e-9))


def published_checks(a, checks):
    from scipy import stats
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multitest import multipletests

    ppl = a.drop_duplicates("subject")
    an = ppl[~ppl.subject_excluded_by_authors]
    M = "Methods 2.1"
    checks.published("participants who completed the bite block task", len(ppl), 62, 0, f"{SRC}, {M}")
    checks.published("participants excluded", ppl.subject_excluded_by_authors.sum(), 7, 0, f"{SRC}, {M}")
    checks.published("participants analysed", len(an), 55, 0, f"{SRC}, {M}")
    checks.published("all 62: mean age (years)", ppl.age.mean(), 39.3, 0.05, f"{SRC}, {M}")
    checks.published("all 62: SD of age (years)", ppl.age.std(), 19.3, 0.05, f"{SRC}, {M}")
    checks.published("all 62: youngest (years)", ppl.age.min(), 18, 0, f"{SRC}, {M}")
    checks.published("all 62: oldest (years)", ppl.age.max(), 78, 0, f"{SRC}, {M}")
    checks.published("all 62: females", (ppl.sex == "F").sum(), 48, 0, f"{SRC}, {M}")
    checks.published("all 62: males", (ppl.sex == "M").sum(), 14, 0, f"{SRC}, {M}")
    checks.published("analysed 55: mean age (years)", an.age.mean(), 37.2, 0.05, f"{SRC}, {M}")
    checks.published("analysed 55: SD of age (years)", an.age.std(), 18.8, 0.05, f"{SRC}, {M}")
    checks.published("analysed 55: females", (an.sex == "F").sum(), 43, 0, f"{SRC}, {M}")
    checks.published("analysed 55: males", (an.sex == "M").sum(), 12, 0, f"{SRC}, {M}")
    checks.published("baseline trials per participant (4 words x 8)", a[a.block == "baseline"].groupby("subject").size().max(), 32, 0,
                     f"{SRC}, Methods 2.2")

    w = a[~a.subject_excluded_by_authors].copy()
    for c in ("f1_mel", "f2_mel", "duration_ms"):
        w[c] = pd.to_numeric(w[c].replace("", np.nan))
    w["vowel"] = w.stimulus.map({v: k for k, v in WORD.items()})
    med = w[w.block == "baseline"].groupby(["subject", "vowel"])[["f1_mel", "f2_mel"]].median()
    # Table 1: first bite block trial of each word minus the baseline median (F1, F2: one-sample t; ED: paired t against
    # the median baseline distance), Holm correction over the 12 tests; the paper's d is Hedges-corrected (1 - 3/(4 df - 1))
    t1 = {"iy": [(37.278, 7.942, 0.624, 4.694), (-86.851, 11.680, -0.989, -7.436), (87.077, 12.085, 0.958, 7.206)],
          "uw": [(32.738, 8.914, 0.488, 3.672), (42.670, 21.811, 0.260, 1.956), (111.890, 14.343, 1.037, 7.801)],
          "ae": [(-32.622, 7.089, -0.612, -4.602), (-35.520, 7.726, -0.611, -4.598), (49.176, 6.807, 0.961, 7.225)],
          "aa": [(-18.869, 7.646, -0.328, -2.468), (-32.754, 8.129, -0.536, -4.029), (47.538, 7.081, 0.893, 6.713)]}
    pvals, labels = [], []
    first = {}
    for v, blk in CRIT.items():
        g = w[(w.block == blk) & (w.vowel == v)].sort_values("trial")
        first[v] = g.groupby("subject").head(1).set_index("subject")
        b = med.xs(v, level="vowel").loc[first[v].index]
        bl = w[(w.block == "baseline") & (w.vowel == v)].merge(
            med.xs(v, level="vowel").rename(columns=lambda c: c + "_m"), left_on="subject", right_index=True)
        ed_base = np.hypot(bl.f1_mel - bl.f1_mel_m, bl.f2_mel - bl.f2_mel_m).groupby(bl.subject).median()
        diffs = [first[v].f1_mel - b.f1_mel, first[v].f2_mel - b.f2_mel,
                 np.hypot(first[v].f1_mel - b.f1_mel, first[v].f2_mel - b.f2_mel) - ed_base.loc[first[v].index]]
        for (meas, x), (rm, rse, rd, rt) in zip(zip(("F1", "F2", "Euclidean distance"), diffs), t1[v]):
            n = x.notna().sum()
            se = x.std(ddof=1) / math.sqrt(n)
            g_ = x.mean() / x.std(ddof=1) * (1 - 3 / (4 * (n - 1) - 1))
            lab = f"Table 1, {SEG[v]} {meas}"
            checks.published(f"{lab}: mean change at the first bite block trial (mel)", x.mean(), rm, 0.0005, f"{SRC}, Table 1")
            checks.published(f"{lab}: SE (mel)", se, rse, 0.0005, f"{SRC}, Table 1")
            checks.published(f"{lab}: d", g_, rd, 0.0005, f"{SRC}, Table 1")
            # /u/ F1 t is 3.6725 here and 3.672 printed (see differences); all other values agree to the printed decimals
            checks.published(f"{lab}: t", x.mean() / se, rt, 0.001 if (v, meas) == ("uw", "F1") else 0.0005, f"{SRC}, Table 1")
            pvals.append(stats.ttest_1samp(x, 0, alternative="greater" if meas.startswith("Euclid") else "two-sided").pvalue)
            labels.append(lab)
    padj = dict(zip(labels, multipletests(pvals, method="holm")[1]))
    for lab, rep in ((f"Table 1, {SEG['uw']} F1", 0.002), (f"Table 1, {SEG['uw']} F2", 0.056), (f"Table 1, {SEG['aa']} F1", 0.034)):
        checks.published(f"{lab}: Holm-corrected p", padj[lab], rep, 0.0005, f"{SRC}, Table 1")
    checks.published("Table 1: tests with Holm-corrected p < .001", sum(v < 0.001 for v in padj.values()), 9, 0, f"{SRC}, Table 1")

    # adaptation: absolute deviation from the baseline median at the first and last bite block trial of each word
    rows = []
    for v, blk in CRIT.items():
        g = w[(w.block == blk) & (w.vowel == v)].sort_values("trial")
        for tp, sel in (("initial", g.groupby("subject").head(1)), ("final", g.groupby("subject").tail(1))):
            s = sel.set_index("subject")
            b = med.xs(v, level="vowel").loc[s.index]
            for f, col in (("F1", "f1_mel"), ("F2", "f2_mel")):
                rows += [dict(subject=k, vowel=v, formant=f, timepoint=tp, dev=x) for k, x in (s[col] - b[col]).abs().items()]
            rows += [dict(subject=k, vowel=v, formant="ED", timepoint=tp, dev=x)
                     for k, x in np.hypot(s.f1_mel - b.f1_mel, s.f2_mel - b.f2_mel).items()]
    r = pd.DataFrame(rows)
    checks.internal("no missing formants on the first or last bite block trial of any word (complete rmANOVA cells)",
                    bool(r.dev.notna().all() and len(r) == 55 * 4 * 3 * 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa = AnovaRM(r[r.formant != "ED"], "dev", "subject", within=["formant", "vowel", "timepoint"]).fit().anova_table
        ea = AnovaRM(r[r.formant == "ED"], "dev", "subject", within=["vowel", "timepoint"]).fit().anova_table
    R = f"{SRC}, Results (adaptation rmANOVA; F unaffected by the Greenhouse-Geisser df correction)"
    for term, rep in (("formant", 35.379), ("vowel", 13.696), ("formant:vowel", 24.853), ("timepoint", 0.043),
                      ("formant:vowel:timepoint", 0.649)):
        checks.published(f"F1/F2 deviation from baseline, rmANOVA F for {term.replace(':', ' x ')}", fa.loc[term, "F Value"], rep,
                         0.0005, R)
    for term, rep in (("vowel", 15.997), ("timepoint", 0.104), ("vowel:timepoint", 0.420)):
        checks.published(f"Euclidean distance from baseline, rmANOVA F for {term.replace(':', ' x ')}", ea.loc[term, "F Value"],
                         rep, 0.0005, R)

    # vowel duration: median per participant, word and phase (each word in its own bite block)
    w["bb"] = np.where(w.block == "baseline", "baseline", "bite block")
    keep = (w.block == "baseline") | (w.block == w.vowel.map(CRIT))
    md = w[keep].groupby(["subject", "vowel", "bb"]).duration_ms.median().reset_index()
    D = f"{SRC}, Results (exploratory: vowel duration)"
    checks.published("vowel duration with a bite block, mean of per-phase medians (ms)", md[md.bb == "bite block"].duration_ms.mean(),
                     449, 0.5, D)
    checks.published("vowel duration at baseline, mean of per-phase medians (ms)", md[md.bb == "baseline"].duration_ms.mean(),
                     365, 0.5, D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        da = AnovaRM(md, "duration_ms", "subject", within=["vowel", "bb"]).fit().anova_table
    for term, rep in (("bb", 92.692), ("vowel", 6.909), ("vowel:bb", 2.869)):
        checks.published(f"vowel duration rmANOVA F for {term.replace('bb', 'phase').replace(':', ' x ')}", da.loc[term, "F Value"],
                         rep, 0.0005, D)

    # variability: SD of residuals from a straight line over each word's repetitions within a phase (gaps kept for
    # missing trials), the authors' detrending (equal to their phase_var_F1/F2 columns in all 440 cells)
    out = []
    for (s, v, b), g in w[keep].sort_values("trial").groupby(["subject", "vowel", "bb"]):
        rec = dict(subject=s, vowel=v, bb=b)
        for f in ("f1_mel", "f2_mel"):
            y = g[f].values
            k = np.arange(len(y))
            ok = ~np.isnan(y)
            rec[f] = np.std(y[ok] - np.polyval(np.polyfit(k[ok], y[ok], 1), k[ok]), ddof=1)
        out.append(rec)
    o = pd.DataFrame(out)
    V = f"{SRC}, Results (exploratory: formant variability)"
    wide = o.pivot_table(index=["subject", "vowel"], columns="bb", values=["f1_mel", "f2_mel"])
    inc = {f: wide[(f, "bite block")] - wide[(f, "baseline")] for f in ("f1_mel", "f2_mel")}
    checks.published("increase in F1 SD with a bite block (mel)", inc["f1_mel"].mean(), 5.2, 0.05, V)
    checks.published("increase in F2 SD with a bite block (mel)", inc["f2_mel"].mean(), 11.25, 0.005, V)
    for v, rep in (("iy", 13.5), ("uw", 19.0), ("ae", 4.9), ("aa", 7.6)):
        checks.published(f"increase in F2 SD with a bite block, {SEG[v]} (mel)", inc["f2_mel"].xs(v, level="vowel").mean(), rep, 0.05, V)
    lo = o.melt(id_vars=["subject", "vowel", "bb"], value_vars=["f1_mel", "f2_mel"], var_name="formant", value_name="sd")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        va = AnovaRM(lo, "sd", "subject", within=["formant", "vowel", "bb"]).fit().anova_table
        v1 = AnovaRM(lo[lo.formant == "f1_mel"], "sd", "subject", within=["vowel", "bb"]).fit().anova_table
        v2 = AnovaRM(lo[lo.formant == "f2_mel"], "sd", "subject", within=["vowel", "bb"]).fit().anova_table
    checks.published("formant SD rmANOVA F for phase", va.loc["bb", "F Value"], 91.771, 0.0005, V)
    checks.published("F1 SD rmANOVA F for phase", v1.loc["bb", "F Value"], 28.294, 0.0005, V)
    checks.published("F2 SD rmANOVA F for phase", v2.loc["bb", "F Value"], 82.162, 0.0005, V)
    checks.published("F1 SD rmANOVA F for vowel", v1.loc["vowel", "F Value"], 5.565, 0.0005, V)
    checks.published("F2 SD rmANOVA F for vowel", v2.loc["vowel", "F Value"], 53.902, 0.0005, V)
    checks.published("F2 SD rmANOVA F for phase x vowel", v2.loc["vowel:bb", "F Value"], 6.629, 0.0005, V)
    return o


def main():
    checks = stdlib.Checks()
    m, p, dd, analysed = load(checks)
    rows = build(m, dd, analysed)
    a = pd.DataFrame(rows)
    a["study_"] = a.subject.map(dd.Study)
    internal_checks(a, checks, stdlib.raw_dir(ID))
    published_checks(a, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s, paper = paper_meta()
    ppl = a.drop_duplicates("subject")
    participants = [dict(person=r.subject, parent_study=r.study_) for r in ppl.itertuples()]
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC0 1.0 Universal",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/legalcode",
        changes=("Took the authors' per-trial formant table of all 62 participants, gave the one participant listed there "
                 "under a different identifier the identifier of the authors' processed table (same 96 trials), converted "
                 "vowel duration from seconds to ms, added age, sex and the bite block size, marked the 7 participants and "
                 "122 unmeasured trials the authors left out, and replaced participant identifiers with codes."),
        how=[
            "One row is one word spoken by one participant, in the authors' trial order (trial 1-96): 32 baseline trials "
            "without a bite block (8 each of heed, who'd, had, hod), then 32 trials with each bite block in counterbalanced "
            "order (block vert = large 17.5 mm vertical block, words heed and who'd; horiz = small 2 mm horizontal block, "
            "words had and hod).",
            "f1_mel and f2_mel are the authors' formants in mels, averaged over the middle 50% of the vowel; duration_ms is "
            "the vowel duration (the authors shared seconds). Empty values are trials the authors could not measure; they "
            "are marked excluded_by_authors.",
            "Loud speech-shaped masking noise (about 80 dB SPL) was played on every trial, so feedback is masked throughout; "
            "pert_value is the bite block's jaw aperture in mm (0 at baseline). One excluded participant has swapped block "
            "labels and words in the authors' file (the paper says this person received bite blocks of incorrect sizes); "
            "their labels and pert_value are kept as the file gives them.",
            "subject_excluded_by_authors marks the 7 participants the paper excludes (5 said a wrong word on critical trials, "
            "1 had incorrect bite block sizes, 1 had breathy phonation); participants.csv gives the lab study during whose "
            "session each participant did the task.",
        ],
        unit_note="spoken word (utterance) of one participant",
        differences=[
            "Table 1, vowel /u/ F1: the t statistic is 3.6725 here and 3.672 in the paper (rounding to three decimals gives "
            "3.673); every other mean, SE, d, t and Holm-corrected p in Table 1 agrees with the printed decimals.",
            "Vowel duration: the means of the per-phase medians reproduce (449 and 365 ms), the paper's +-2.1 and +-1.3 ms "
            "do not match an SE across participants and words (7.8 and 4.8 ms here); the paper does not say what they are.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks, participants=participants,
                          participant_columns=[("parent_study", "text", "",
                                                "Lab study in whose session the participant did the bite block task "
                                                "(vsaGeneralize2 or vsaRetention), from the authors' demographics file.")])


if __name__ == "__main__":
    main()
