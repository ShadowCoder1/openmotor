"""Murphy, Holt & Nozari (2025), PsyArXiv: "Exposure to an Accent Transfers to Speech Production in a Single Shot".
Two online experiments in which people passively heard sequences of beer/pier syllables that followed English
regularities (Canonical) or a reversed, accented F0 x VOT relationship (Reverse), categorized an ambiguous test word, and on
some trials said it aloud. One row is one spoken production with its F0.

Raw files, raw_extra/Murphy2025_PsyArXiv/ (OSF cgp7u, fetched with tools/std/fetch.py):
  Overt_Exp1_Production_Data.xlsx   ID, trial_num, response (beer/pier keypress just before the production), stat_reg
                                    (Canonical/Reverse), test_stimulus_f0, block_num, is_tenth, prod_f0 (log F0 of the first
                                    40 ms of voicing, z-scored within participant), trial_identity (exposure sequence label),
                                    group (overt_100 = Full, a production on every trial; overt_10 = Sparse, every tenth trial;
                                    the file keeps the productions of trials 10, 20, ... 360 for both groups), age, sex, language
  Overt_Exp2_Production_Data.xlsx   the same columns without group, age, sex, language; block_num 1 = Critical Canonical block
                                    (productions 1-6), 2 = Critical Reverse block (production 1), 3 = Normalization Canonical block
                                    (productions 7-36; the authors use it only to z-score F0)
  Overt_Exp1_*_Production.csv, Overt_Exp2_Production.csv   the authors' lmer coefficient tables (Tables 2 and 4 of the paper)
  Overt_Production_Exp*.Rmd         the authors' analysis scripts, read as text for their exclusions and models
Exclusions come from the scripts: Experiment 1 drops five participants (four poor-audio participants that have no production
values in the file, and one with 32 rows) and the unambiguous test words (beer, pier); Experiment 2 drops one participant
named in the script and the participants with no production in the Critical Reverse block, uses only blocks 1 and 2 and
drops the unambiguous test words. Excluded participants are recognised here by a hash of their identifier, so no identifier is
written into this file or the output.
Run from the site folder: python3 -I tools/std/convert/Murphy2025_PsyArXiv.py
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Murphy2025_PsyArXiv"
SRC = "Murphy, Holt & Nozari 2025, PsyArXiv"
EXCL1 = {"d59bcec33b8eba4a", "1609ba11b2ee9680", "ea579ea92137e6b7", "0513db5fac04b919", "10a62b8ad544db37"}
EXCL2 = {"c4621682cd006ae7", "69620d04dcd429a0"}
DEVICE = "participant's own computer in Chrome (Gorilla), built-in microphone, wired headphones"
TOKEN = {"LowF0": "beer-pier test word, VOT 20 ms, F0 180 Hz", "HighF0": "beer-pier test word, VOT 20 ms, F0 250 Hz",
         "Beer": "unambiguous beer, VOT 0 ms, F0 180 Hz", "Pier": "unambiguous pier, VOT 40 ms, F0 250 Hz"}


def h(x):
    return hashlib.sha256(str(x).encode()).hexdigest()[:16]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["venue"] = p["venue"] or "PsyArXiv preprint"
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def load(checks):
    raw = stdlib.raw_dir(ID)
    d1 = pd.read_excel(raw / "Overt_Exp1_Production_Data.xlsx")
    d2 = pd.read_excel(raw / "Overt_Exp2_Production_Data.xlsx")
    d1["ID"], d2["ID"] = d1.ID.astype(str), d2.ID.astype(str)
    empty = d1[d1.prod_f0.isna()]
    checks.internal("Experiment 1: the 7 rows with no values are single rows of participants that have no other row; 4 of the "
                    "5 participants the script excludes are among them",
                    bool(len(empty) == 7 and all((d1.ID == i).sum() == 1 for i in empty.ID)
                         and sum(h(i) in EXCL1 for i in empty.ID) == 4), f"{len(empty)} rows")
    d1 = d1[d1.prod_f0.notna()].copy()
    checks.internal("Experiment 1: 1838 productions of 53 participants (27 Sparse, 26 Full), group, age and sex constant within "
                    "participant", bool(len(d1) == 1838 and d1.ID.nunique() == 53
                                        and d1.drop_duplicates("ID").group.value_counts().to_dict() == {"overt_10": 27, "overt_100": 26}
                                        and (d1.groupby("ID")[["group", "age", "sex"]].nunique(dropna=True) <= 1).all().all()))
    return d1, d2


def build(d1, d2, checks):
    rows = []
    # ---- Experiment 1: order = authors' trial number (block_num = ceil(trial_num / 60), checked)
    checks.internal("Experiment 1: block_num = ceil(trial_num / 60) on every row, trial numbers 10 ... 360 in steps of 10",
                    bool((np.ceil(d1.trial_num / 60) == d1.block_num).all() and (d1.trial_num % 10 == 0).mean() > 0.99
                         and d1.trial_num.min() == 10 and d1.trial_num.max() == 360))
    d1 = d1.assign(_o=np.arange(len(d1))).sort_values(["ID", "trial_num", "_o"], kind="stable")
    ex1 = d1.ID.map(h).isin(EXCL1)
    amb1 = d1.test_stimulus_f0.isin(["LowF0", "HighF0"])
    d1["analysed"] = amb1 & ~ex1
    for pid, g in d1.groupby("ID", sort=False):
        for k, r in enumerate(g.itertuples(index=False), 1):
            b = int(r.block_num)
            rows.append(_row("Experiment 1", r.group, "E1_" + pid, bool(h(pid) in EXCL1), r.age, r.sex, k, f"block {b}",
                             "baseline" if b == 1 else ("learning" if r.stat_reg == "Reverse" else "washout"), r, bool(not r.analysed)))
    # ---- Experiment 2: order = block, then the authors' production number
    d2 = d2.assign(_o=np.arange(len(d2))).sort_values(["ID", "block_num", "trial_num", "_o"], kind="stable")
    nonx = d2[~d2.ID.map(h).isin(EXCL2) & (d2.block_num < 3) & ~d2.test_stimulus_f0.isin(["Beer", "Pier"])]
    keep = set(nonx[nonx.test_stimulus_f0.str.endswith("_crit")].ID)
    d2["analysed"] = d2.ID.isin(keep) & ~d2.ID.map(h).isin(EXCL2) & (d2.block_num < 3) & ~d2.test_stimulus_f0.isin(["Beer", "Pier"])
    out_ids = set(d2.ID) - keep
    for pid, g in d2.groupby("ID", sort=False):
        for k, r in enumerate(g.itertuples(index=False), 1):
            b = int(r.block_num)
            lab = {1: "critical canonical", 2: "critical reverse", 3: "normalization canonical"}[b]
            rows.append(_row("Experiment 2", "", "E2_" + pid, pid in out_ids, np.nan, None, k, lab,
                             {1: "baseline", 2: "learning", 3: "other"}[b], r, bool(not r.analysed)))
    return rows, d1, d2, out_ids


def _row(exp, group, subject, subj_excl, age, sex, k, block, phase, r, excluded):
    tok = r.test_stimulus_f0.replace("_crit", "")
    crit = r.test_stimulus_f0.endswith("_crit")
    return {
        "dataset": ID, "study": ID, "experiment": exp, "group": group, "subject": subject, "person": subject,
        "subject_excluded_by_authors": subj_excl, "age": "" if pd.isna(age) else int(age),
        "sex": {"Female": "F", "Male": "M"}.get(sex, "") if sex else "", "handedness": "", "population": "healthy",
        "setting": "online", "device": DEVICE, "session": 1, "block": block, "trial": k, "phase": phase, "time_s": "",
        "task": "speech", "unit": "utterance", "condition": r.stat_reg, "pert_type": "none", "pert_value": 0, "pert_unit": "",
        "feedback": "normal", "excluded_by_authors": excluded,
        "stimulus": TOKEN[tok] + (" (critical trial: first production after the Reverse regularity)" if crit else "") + ", said aloud",
        "segment": "word onset, start of the voiced part", "measure_window": "first 40 ms of voicing",
        "f0_z": float(r.prod_f0), "perceptual_response": r.response, "exposure_set": r.trial_identity,
        "authors_trial_number": int(r.trial_num), "masking_noise": "none",
    }


def fit_checks(a, d1, d2, checks):
    import statsmodels.formula.api as smf
    warnings.simplefilter("ignore")
    A = a.copy()
    A["f0"] = A.f0_z.astype(float)
    A["sr"] = np.where(A.condition == "Canonical", 0.5, -0.5)          # R: contrasts(stat_reg) <- c(.5, -.5), levels Canonical, Reverse
    A["rs"] = np.where(A.perceptual_response == "Beer", 0.5, -0.5)     # levels Beer, Pier
    A["gr"] = np.where(A.group == "overt_10", 0.5, -0.5)               # levels overt_10, overt_100
    e1 = A[(A.experiment == "Experiment 1") & ~A.excluded_by_authors]
    e2 = A[(A.experiment == "Experiment 2") & ~A.excluded_by_authors]
    checks.internal("Experiment 1: 1610 productions enter the authors' models (724 Full + 886 Sparse), as the degrees of freedom "
                    "of the authors' result tables imply (720 + 4, 882 + 4, 1602 + 8)",
                    bool(len(e1) == 1610 and (e1.group == "overt_100").sum() == 724 and (e1.group == "overt_10").sum() == 886))
    T2 = "Table 2 (Experiment 1, transfer to production)"
    T2v = [("Intercept", 0.04, 0.02, 1.79), ("sr", 0.08, 0.04, 1.72), ("rs", -0.76, 0.04, -16.93), ("gr", 0.04, 0.04, 0.90),
           ("sr:rs", -0.53, 0.09, -5.93), ("sr:gr", 0.17, 0.09, 1.94), ("rs:gr", -0.23, 0.09, -2.51), ("sr:rs:gr", -0.28, 0.18, -1.58)]
    names = {"Intercept": "intercept", "sr": "Statistical Regularity", "rs": "Response", "gr": "Group", "sr:rs": "Regularity x Response",
             "sr:gr": "Regularity x Group", "rs:gr": "Response x Group", "sr:rs:gr": "Regularity x Response x Group"}
    # The authors' random-intercept lmer fits are singular (intercept variance 0, df = N - p exactly in their result tables),
    # so they equal ordinary least squares; the mixed fit below confirms the variance is zero.
    mm = smf.mixedlm("f0 ~ sr*rs*gr", e1, groups=e1.ID_.astype(str)).fit(reml=True)
    checks.internal("Experiment 1: the participant random-intercept variance of the authors' model is zero (singular fit), so the "
                    "least-squares fit equals their lmer fit", bool(float(mm.cov_re.iloc[0, 0]) < 1e-4), f"{float(mm.cov_re.iloc[0, 0]):.2g}")
    m = smf.ols("f0 ~ sr*rs*gr", e1).fit()
    for k, est, se, t in T2v:
        checks.published(f"{T2}, {names[k]}: beta", m.params[k], est, 0.0051, f"{SRC}, {T2}")
        checks.published(f"{T2}, {names[k]}: SE", m.bse[k], se, 0.0051, f"{SRC}, {T2}")
        checks.published(f"{T2}, {names[k]}: t", m.tvalues[k], t, 0.011, f"{SRC}, {T2}")
    for grp, lab, (t_r, t_i) in (("overt_100", "Full", (-9.51, -2.87)), ("overt_10", "Sparse", (-14.77, -5.70))):
        g = e1[e1.group == grp]
        mg = smf.ols("f0 ~ sr*rs", g).fit()
        checks.published(f"{lab} group: t of Response", mg.tvalues["rs"], t_r, 0.011, f"{SRC}, Experiment 1, Results (post-hoc tests)")
        checks.published(f"{lab} group: t of Regularity x Response", mg.tvalues["sr:rs"], t_i, 0.011,
                         f"{SRC}, Experiment 1, Results (post-hoc tests)")
    T4 = "Table 4 (Experiment 2, transfer to production)"
    m2 = smf.mixedlm("f0 ~ sr*rs", e2, groups=e2.ID_.astype(str), re_formula="~rs").fit(reml=True)
    for k, est, se, t in (("Intercept", 0.18, 0.04, 4.32), ("sr", -0.03, 0.08, -0.45), ("rs", -0.68, 0.09, -7.96), ("sr:rs", -0.45, 0.15, -2.98)):
        checks.published(f"{T4}, {names[k]}: beta", m2.params[k], est, 0.0051, f"{SRC}, {T4}")
        checks.published(f"{T4}, {names[k]}: SE", m2.bse[k], se, 0.0051, f"{SRC}, {T4}")
        checks.published(f"{T4}, {names[k]}: t", m2.tvalues[k], t, 0.011, f"{SRC}, {T4}")
    return e1, e2


def main():
    checks = stdlib.Checks()
    d1, d2 = load(checks)
    rows, d1, d2, out2 = build(d1, d2, checks)
    a = pd.DataFrame(rows)
    # participant key without the E1_/E2_ prefix is not needed; the subject column holds the (pre-code) key for the model grouping
    a["ID_"] = a.subject
    ppl1 = a[a.experiment == "Experiment 1"].drop_duplicates("subject")
    an1 = ppl1[~ppl1.subject_excluded_by_authors]
    ppl2 = a[a.experiment == "Experiment 2"].drop_duplicates("subject")
    an2 = ppl2[~ppl2.subject_excluded_by_authors]
    S1, S2 = "Experiment 1, Participants", "Experiment 2, Participants"
    checks.published("Experiment 1: participants in the final sample", len(an1), 52, 0, f"{SRC}, {S1}")
    checks.published("Experiment 1: mean age of the final sample (years)", an1.age.astype(float).mean(), 28.7, 0.05, f"{SRC}, {S1}")
    checks.published("Experiment 1: SD of age of the final sample (years)", an1.age.astype(float).std(), 6.9, 0.05, f"{SRC}, {S1}")
    checks.published("Experiment 2: participants in the final sample", len(an2), 217, 0, f"{SRC}, {S2}")
    checks.published("Experiment 1: trials in the experiment (6 blocks x 60), from the highest production trial number",
                     pd.to_numeric(a[a.experiment == "Experiment 1"].authors_trial_number).max(), 360, 0,
                     f"{SRC}, Experiment 1, Procedure (360 trials Full group, 36 productions Sparse group)")
    checks.published("Experiment 1: distinct production trials in the file (every tenth trial of 360, 10 to 360)",
                     pd.to_numeric(a[a.experiment == "Experiment 1"].authors_trial_number).nunique(), 36, 0,
                     f"{SRC}, Experiment 1, Procedure")
    checks.internal("Experiment 2: 7961 productions of 223 participants; 217 enter the analyses (1425 productions: 1208 Canonical, "
                    "217 Reverse), 6 participants have no production in the Critical Reverse block or are named in the script",
                    bool(len(a[a.experiment == "Experiment 2"]) == 7961 and len(ppl2) == 223 and len(an2) == 217
                         and ((a.experiment == "Experiment 2") & ~a.excluded_by_authors).sum() == 1425))
    z = a.assign(f=a.f0_z.astype(float)).groupby("subject").f.agg(["mean", "std"])
    checks.internal("F0 is z-scored within participant over the shared productions: mean 0 and SD 1 for every participant",
                    bool((z["mean"].abs() < 0.01).all() and (z["std"].sub(1).abs() < 0.05).all()),
                    f"max |mean| {z['mean'].abs().max():.2g}, SD {z['std'].min():.3f}-{z['std'].max():.3f}")
    checks.internal("f0_z within +-5 and every row has a perceptual response and an exposure sequence label",
                    bool(a.f0_z.astype(float).abs().max() < 5 and (a.perceptual_response != "").all() and (a.exposure_set != "").all()),
                    f"{a.f0_z.astype(float).min():.2f} to {a.f0_z.astype(float).max():.2f}")
    fit_checks(a, d1, d2, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s, paper = paper_meta()
    rows_out = a.drop(columns=["ID_"]).to_dict("records")
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Took the authors' two production tables (Experiment 1 and 2) and kept every production row; ordered each "
                 "participant's rows by block and the authors' trial number; mapped the authors' z-scored F0, perceptual "
                 "response and exposure sequence label to columns; added the test word and the regularity; marked the "
                 "participants and productions the authors' scripts leave out; dropped 7 rows of participants who have no "
                 "production value at all; replaced participant identifiers with codes."),
        how=[
            "One row is one spoken production of one participant, ordered by block and then the authors' own trial number (trial "
            "is the order among the rows shared; authors_trial_number is the authors' number). On each trial the participant "
            "passively heard 8 beer/pier syllables that followed English regularities (condition Canonical) or an accented, "
            "reversed F0-by-VOT relationship (Reverse), categorized an ambiguous test word with a keypress (perceptual_response: "
            "beer or pier) and, on prompted trials, heard it again and said it aloud.",
            "Experiment 1 has two groups: overt_100 (Full), prompted to speak on every trial of 360 (6 blocks of 60, regularity "
            "alternating, Canonical first), and overt_10 (Sparse), prompted every tenth trial; the authors' file keeps the "
            "productions of trials 10, 20, ... 360 for both groups. Experiment 2 has a Critical Canonical block (productions "
            "1-6), a Critical Reverse block (one production), and a Normalization Canonical block (productions 7-36) that the "
            "authors use only to z-score F0; its age and sex were not shared.",
            "f0_z is the authors' F0 of the first 40 ms of voicing, log-transformed, outliers beyond 3 SD of the participant's mean "
            "removed, z-scored within participant (0 is the participant's mean F0 over all their productions, in standard "
            "deviations; the Hz value is not shared). No auditory feedback manipulation: feedback is normal and pert_value is 0; "
            "the manipulation is the regularity of the passively heard syllables (condition).",
            "excluded_by_authors marks productions the authors' scripts do not analyse (unambiguous test words, Experiment 2's "
            "Normalization block, and all rows of excluded participants); subject_excluded_by_authors marks the participants "
            "the scripts leave out.",
        ],
        unit_note="spoken production (utterance) of one participant",
        differences=[
            "Experiment 1 sample: the paper reports 39 women among the 52 final participants; the shared sex column gives 28 women "
            "and 24 men for the same 52 (age, 28.7 +- 6.9, reproduces).",
            "Seven Experiment 1 rows hold only a participant identifier with no values (four of them are participants the paper "
            "excludes for poor audio); they are not copied. The script excludes one further participant who has 32 productions.",
            "Tables 2 and 4 were refitted in Python. The Experiment 1 model of the authors has zero participant-intercept variance "
            "(a singular fit), so least squares gives their coefficients, SEs and t; Experiment 2 was refitted by REML with a random "
            "intercept and a random Response slope per participant. All agree with the printed values to the printed decimals; "
            "p-values were not refitted.",
        ],
    )
    return stdlib.release(ID, "speech", rows_out, meta, checks)


if __name__ == "__main__":
    main()
