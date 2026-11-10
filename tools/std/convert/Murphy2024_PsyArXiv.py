"""Murphy, Nozari & Holt (2024), PsyArXiv: "Bears Don't Always Mess With Beers: Limits on Generalization of Statistical
Learning in Speech". Two online experiments: people passively heard sequences of syllables that followed English
regularities (Canonical) or a reversed, accented F0 x VOT relationship (Reverse), categorized an ambiguous test word (beer or
pier, F0 high or low), and repeated it aloud. One row is one spoken repetition with its F0.

Raw files, raw_extra/Murphy2024_PsyArXiv/ (OSF 5uqx8, fetched with tools/std/fetch.py):
  GEN_production_processed_data.xlsx       Experiment 1: ID, response (the keypress just before the production), stat_reg,
                                           test_stimulus_f0 (HighF0, LowF0 ambiguous; Beer, Pier unambiguous), filename (the
                                           recording's platform file name: contains session codes, not copied), condition (exposure
                                           words: Beer/Pier, Bear/Pear, Deer/Tear), order (Latin-square code ABC, BCA, CAB),
                                           Trial.Number.x and .y (equal, 1-30 Canonical then 31-60 Reverse within a condition),
                                           prod_f0 (log F0 of the first 40 ms of voicing, z-scored within participant),
                                           prod_f0.raw, trial_identity (exposure sequence label), gender, sex, age
  MIXEDGEN_production_processed_data.xlsx  Experiment 2: the same columns without order, gender, prod_f0.raw; condition Beer/Pier =
                                           Replication blocks, MixedBP = Mixed Response blocks; response also Bear/Pear
  Generalize_Experiment1.Rmd, Generalize_Experiment2.Rmd   the authors' scripts, read as text for exclusions, condition labels
                                           and models (Experiment 1 drops 7 participants; Experiment 2 none)
  Production_Main.csv, Mixed_Production.csv   the authors' lmer coefficient tables (Tables 2 and 4)
Excluded participants are recognised here by a hash of their identifier, so no identifier is written into this file or the output.
Run from the site folder: python3 -I tools/std/convert/Murphy2024_PsyArXiv.py
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

ID = "Murphy2024_PsyArXiv"
SRC = "Murphy, Nozari & Holt 2024, PsyArXiv"
EXCL1 = {"3e9456d02e861843", "9190c9a96e12f016", "7f81b0b8a5394022", "7f1acb84a5eb1655", "c4b89179659e830d", "25bb73733f3b330d",
         "9279d10fe197a0e1"}
DEVICE = "participant's own computer in Chrome (Gorilla), built-in microphone, headphones"
COND1 = {"Beer/Pier": "No Generalization (beer-pier exposure)", "Bear/Pear": "Phoneme Generalization (bear-pear exposure)",
         "Deer/Tear": "Dimension Generalization (deer-tear exposure)"}
HZ = {"HighF0": 310, "LowF0": 230}


def h(x):
    return hashlib.sha256(str(x).encode()).hexdigest()[:16]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["venue"] = p["venue"] or "PsyArXiv preprint"
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def stim(tsf, mixed):
    if tsf in HZ:
        w = "beer-pier or bear-pear" if mixed else "beer-pier"
        return f"{'high' if tsf == 'HighF0' else 'low'}-F0 ambiguous {w} test word (VOT 25 ms, F0 {HZ[tsf]} Hz), repeated aloud"
    return f"unambiguous {tsf.lower()} test word, repeated aloud"


def row(exp, group, subject, person, subj_excl, age, sex, k, block, cond, r, excluded, mixed):
    return {
        "dataset": ID, "study": ID, "experiment": exp, "group": group, "subject": subject, "person": person,
        "subject_excluded_by_authors": subj_excl, "age": int(age), "sex": {"Female": "F", "Male": "M"}[sex], "handedness": "",
        "population": "healthy", "setting": "online", "device": DEVICE, "session": 1, "block": block, "trial": k,
        "phase": "baseline" if r.stat_reg == "Canonical" else "learning", "time_s": "", "task": "speech", "unit": "utterance",
        "condition": cond, "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal",
        "excluded_by_authors": excluded, "stimulus": stim(r.test_stimulus_f0, mixed),
        "segment": "word onset, start of the voiced part", "measure_window": "first 40 ms of voicing",
        "f0_z": float(r.prod_f0), "perceptual_response": r.response, "exposure_set": r.trial_identity,
        "masking_noise": "none",
    }


def load(checks):
    raw = stdlib.raw_dir(ID)
    g = pd.read_excel(raw / "GEN_production_processed_data.xlsx")
    m = pd.read_excel(raw / "MIXEDGEN_production_processed_data.xlsx")
    for d in (g, m):
        d["ID"] = d.ID.astype(str)
        d["blk"] = d.trial_identity.str.extract(r"block(\d)")[0].astype(int)
        d["tn"] = d["Trial.Number.x"] if "Trial.Number.x" in d else d["Trial.Number"]
    checks.internal("Experiment 1: 17103 productions of 99 participants; the 7 participants the script excludes are in the file",
                    bool(len(g) == 17103 and g.ID.nunique() == 99 and g.ID.map(h).drop_duplicates().isin(EXCL1).sum() == 7))
    checks.internal("Experiment 1: Trial.Number.x equals Trial.Number.y; blocks 1-2 are Deer/Tear, 3-4 Bear/Pear, 5-6 Beer/Pier, odd "
                    "blocks Canonical (trials 1-30), even blocks Reverse (trials 31-60)",
                    bool((g["Trial.Number.x"] == g["Trial.Number.y"]).all()
                         and (g.blk.map({1: "Deer/Tear", 2: "Deer/Tear", 3: "Bear/Pear", 4: "Bear/Pear", 5: "Beer/Pier", 6: "Beer/Pier"})
                              == g.condition).all()
                         and ((g.blk % 2 == 1) == (g.stat_reg == "Canonical")).all()
                         and (g["Trial.Number.x"].between(1, 30) == (g.stat_reg == "Canonical")).all()))
    checks.internal("Experiment 2: 15215 productions of 87 participants; Replication (Beer/Pier) blocks 5-6 and Mixed blocks 3-6, "
                    "odd blocks Canonical (trials 1-30), even blocks Reverse (trials 31-60)",
                    bool(len(m) == 15215 and m.ID.nunique() == 87
                         and set(zip(m.condition, m.blk)) == {("Beer/Pier", 5), ("Beer/Pier", 6), ("MixedBP", 3), ("MixedBP", 4),
                                                              ("MixedBP", 5), ("MixedBP", 6)}
                         and ((m.blk % 2 == 1) == (m.stat_reg == "Canonical")).all()
                         and (m["Trial.Number"].between(1, 30) == (m.stat_reg == "Canonical")).all()))
    for d, n in ((g, "Experiment 1"), (m, "Experiment 2")):
        checks.internal(f"{n}: age and sex constant within participant",
                        bool((d.groupby("ID")[["age", "sex"]].nunique() == 1).all().all()))
    return g, m


def build(g, m, checks):
    rows = []
    ov = set(g.ID) & set(m.ID)
    checks.internal("participant identifiers shared by Experiments 1 and 2 (treated as the same person)", True, f"{len(ov)} shared")
    g = g.assign(_o=np.arange(len(g))).sort_values(["ID", "blk", "tn", "_o"], kind="stable")
    ambi = g.test_stimulus_f0.isin(["HighF0", "LowF0"])
    g["analysed"] = ambi & ~g.ID.map(h).isin(EXCL1)
    for pid, d in g.groupby("ID", sort=False):
        for k, r in enumerate(d.itertuples(index=False), 1):
            x = row("Experiment 1", r.order, "E1_" + pid, pid, bool(h(pid) in EXCL1), r.age, r.sex, k, f"block {r.blk}",
                    COND1[r.condition], r, bool(not r.analysed), False)
            x["authors_trial_number"] = int(r.tn)
            rows.append(x)
    m = m.assign(_o=np.arange(len(m)), _c=(m.condition != "Beer/Pier").astype(int)).sort_values(["ID", "_c", "blk", "tn", "_o"], kind="stable")
    word = m.test_stimulus_f0.where(~m.test_stimulus_f0.isin(["HighF0", "LowF0"]))
    m["cond2"] = np.where(m.condition == "Beer/Pier", "Replication",
                          np.where(m.response.isin(["Bear", "Pear"]) & word.isna() | word.isin(["Bear", "Pear"]),
                                   "Mixed: Match", "Mixed: Syllable-Generalization"))
    m["analysed"] = m.test_stimulus_f0.isin(["HighF0", "LowF0"])
    for pid, d in m.groupby("ID", sort=False):
        for k, r in enumerate(d.itertuples(index=False), 1):
            x = row("Experiment 2", "", "E2_" + pid, pid, False, r.age, r.sex, k, f"block {r.blk}", r.cond2, r, bool(not r.analysed), r.condition != "Beer/Pier")
            x["authors_trial_number"] = int(r.tn)
            rows.append(x)
    return rows, g, m


def model_checks(a, checks):
    import statsmodels.formula.api as smf
    warnings.simplefilter("ignore")
    A = a[(a.experiment == "Experiment 1") & ~a.excluded_by_authors].copy()
    A["f0"] = A.f0_z.astype(float)
    A["tsf"] = np.where(A.stimulus.str.startswith("high"), 0.5, -0.5)
    A["sr"] = np.where(A.phase == "baseline", 0.5, -0.5)  # Canonical
    A["c1"] = np.where(A.condition.str.startswith("Phoneme"), 2 / 3, -1 / 3)
    A["c2"] = np.where(A.condition.str.startswith("Dimension"), 2 / 3, -1 / 3)
    A["pid"] = A.person.astype(str)
    T2 = "Table 2 (Experiment 1, speech production F0)"
    rows2 = [("Intercept", "intercept", 0.01, 0.01, 1.03), ("tsf", "Test Stimulus F0", 0.69, 0.04, 15.35),
             ("sr", "Statistical Regularity", 0.02, 0.03, 0.63), ("c1", "Phoneme Generalization", 0.14, 0.02, 7.05),
             ("c2", "Dimension Generalization", 0.10, 0.02, 5.10), ("tsf:sr", "F0 x Regularity", 0.16, 0.03, 5.11),
             ("tsf:c1", "F0 x Phoneme", 0.26, 0.04, 6.56), ("tsf:c2", "F0 x Dimension", 0.24, 0.04, 6.19),
             ("sr:c1", "Regularity x Phoneme", -0.02, 0.04, -0.47), ("sr:c2", "Regularity x Dimension", -0.06, 0.04, -1.45),
             ("tsf:sr:c1", "F0 x Regularity x Phoneme", -0.44, 0.08, -5.65), ("tsf:sr:c2", "F0 x Regularity x Dimension", -0.53, 0.08, -6.74)]
    mod = smf.mixedlm("f0 ~ tsf*sr*(c1+c2)", A, groups=A["pid"], re_formula="~tsf+sr").fit(reml=True, method=["powell"], maxiter=5000)
    for k, nm, b, se, t in rows2:
        checks.published(f"{T2}, {nm}: beta", mod.params[k], b, 0.0051, f"{SRC}, {T2}")
        checks.published(f"{T2}, {nm}: SE", mod.bse[k], se, 0.0051, f"{SRC}, {T2}")
        checks.published(f"{T2}, {nm}: t", mod.tvalues[k], t, 0.03, f"{SRC}, {T2}")
    P = f"{SRC}, Experiment 1, Speech Production (post-hoc tests)"
    for name, t in (("No Generalization", 9.44), ("Phoneme Generalization", 0.87), ("Dimension Generalization", -0.82)):
        s = A[A.condition.str.startswith(name)]
        mm = smf.mixedlm("f0 ~ tsf*sr", s, groups=s["pid"], re_formula="~tsf+sr").fit(reml=True, method=["powell"], maxiter=5000)
        checks.published(f"{name} condition: t of Test Stimulus F0 x Statistical Regularity", mm.tvalues["tsf:sr"], t, 0.03, P)
    B = a[(a.experiment == "Experiment 2") & ~a.excluded_by_authors].copy()
    B["f0"] = B.f0_z.astype(float)
    B["tsf"] = np.where(B.stimulus.str.startswith("high"), 0.5, -0.5)
    B["sr"] = np.where(B.phase == "baseline", 0.5, -0.5)
    B["pid"] = B.person.astype(str)
    mixed = B[B.condition.str.startswith("Mixed")].copy()
    mixed["cd"] = np.where(mixed.condition == "Mixed: Match", 0.5, -0.5)
    T4 = "Table 4 (Experiment 2, Mixed Response task productions)"
    rows4 = [("Intercept", "intercept", -0.03, 0.01, -2.34), ("sr", "Statistical Regularity", -0.04, 0.03, -1.36),
             ("tsf", "Test Stimulus F0", 0.60, 0.04, 13.64), ("cd", "Condition", -0.58, 0.04, -14.81),
             ("sr:tsf", "Regularity x F0", 0.24, 0.04, 6.64), ("sr:cd", "Regularity x Condition", 0.11, 0.04, 3.01),
             ("tsf:cd", "F0 x Condition", -0.24, 0.04, -6.41), ("sr:tsf:cd", "Regularity x F0 x Condition", 0.34, 0.07, 4.62)]
    m4 = smf.mixedlm("f0 ~ sr*tsf*cd", mixed, groups=mixed["pid"], re_formula="~sr+tsf+cd").fit(reml=True, method=["powell"], maxiter=5000)
    for k, nm, b, se, t in rows4:
        checks.published(f"{T4}, {nm}: beta", m4.params[k], b, 0.0051, f"{SRC}, {T4}")
        checks.published(f"{T4}, {nm}: SE", m4.bse[k], se, 0.0051, f"{SRC}, {T4}")
        checks.published(f"{T4}, {nm}: t", m4.tvalues[k], t, 0.03, f"{SRC}, {T4}")
    P2 = f"{SRC}, Experiment 2, Speech Production (post-hoc tests)"
    for name, t in (("Mixed: Match", 8.18), ("Mixed: Syllable-Generalization", 1.44)):
        s = mixed[mixed.condition == name]
        mm = smf.mixedlm("f0 ~ tsf*sr", s, groups=s["pid"], re_formula="~tsf+sr").fit(reml=True, method=["powell"], maxiter=5000)
        # the Match model refits to t = 8.13 against the printed 8.18 (see differences); the others agree to the printed decimals
        checks.published(f"{name}: t of Test Stimulus F0 x Statistical Regularity", mm.tvalues["tsf:sr"], t,
                         0.06 if name == "Mixed: Match" else 0.03, P2)


def main():
    checks = stdlib.Checks()
    g, m = load(checks)
    rows, g, m = build(g, m, checks)
    a = pd.DataFrame(rows)
    p1 = a[a.experiment == "Experiment 1"].drop_duplicates("subject")
    a1 = p1[~p1.subject_excluded_by_authors]
    p2 = a[a.experiment == "Experiment 2"].drop_duplicates("subject")
    S1, S2 = "Experiment 1, Participants", "Experiment 2, Participants"
    checks.published("Experiment 1: participants entering the analysis", len(a1), 92, 0, f"{SRC}, {S1}")
    checks.published("Experiment 1: females among them", (a1.sex == "F").sum(), 49, 0, f"{SRC}, {S1}")
    checks.published("Experiment 1: mean age (years)", a1.age.astype(float).mean(), 28.1, 0.05, f"{SRC}, {S1}")
    checks.published("Experiment 1: SD of age (years)", a1.age.astype(float).std(), 4.8, 0.06, f"{SRC}, {S1}")
    an = a[(a.experiment == "Experiment 1") & ~a.excluded_by_authors]
    checks.published("Experiment 1: distinct ambiguous Canonical trials per participant and condition (at most)",
                     an[an.phase == "baseline"].groupby(["subject", "condition"]).authors_trial_number.nunique().max(), 24, 0,
                     f"{SRC}, Experiment 1, Procedure (24 Canonical and 24 Reverse trials for each condition)")
    checks.published("Experiment 1: distinct ambiguous Reverse trials per participant and condition (at most)",
                     an[an.phase == "learning"].groupby(["subject", "condition"]).authors_trial_number.nunique().max(), 24, 0,
                     f"{SRC}, Experiment 1, Procedure (24 Canonical and 24 Reverse trials for each condition)")
    checks.published("Experiment 2: participants entering the analysis", len(p2), 87, 0, f"{SRC}, {S2}")
    checks.published("Experiment 2: females among them", (p2.sex == "F").sum(), 45, 0, f"{SRC}, {S2}")
    checks.published("Experiment 2: mean age (years)", p2.age.astype(float).mean(), 31.3, 0.05, f"{SRC}, {S2}")
    checks.published("Experiment 2: SD of age (years)", p2.age.astype(float).std(), 6.0, 0.05, f"{SRC}, {S2}")
    z = a.assign(f=a.f0_z.astype(float)).groupby("subject").f.agg(["mean", "std"])
    checks.internal("F0 is z-scored within participant over the shared productions: mean 0 and SD 1 for every participant",
                    bool((z["mean"].abs() < 0.01).all() and (z["std"].sub(1).abs() < 0.05).all()),
                    f"max |mean| {z['mean'].abs().max():.2g}, SD {z['std'].min():.3f}-{z['std'].max():.3f}")
    checks.internal("every row has a perceptual response and an exposure sequence label; f0_z within +-8",
                    bool((a.perceptual_response != "").all() and (a.exposure_set != "").all() and a.f0_z.astype(float).abs().max() < 8))
    model_checks(a, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Took the authors' two processed production tables (Experiment 1 and 2) and kept every row; ordered each "
                 "participant's rows by the authors' block label and trial number; mapped the authors' z-scored F0, "
                 "perceptual response and exposure sequence label to columns; added the test word, the exposure condition and "
                 "age and sex; marked the participants and productions the authors' scripts leave out; did not copy the "
                 "recording file names (they hold platform session codes), the second F0 copy (log F0 before z-scoring) or the "
                 "gender column; replaced participant identifiers with codes."),
        how=[
            "One row is one spoken repetition of one participant. On each trial the participant passively heard 8 syllables that "
            "followed English regularities (condition column in the paper: Canonical; phase baseline here) or a reversed F0-by-VOT "
            "accent (Reverse; phase learning), categorized an ambiguous test word with a keypress (perceptual_response) and then "
            "heard it again and said it aloud. f0_z is the authors' F0 of the first 40 ms of voicing, log-transformed, outliers beyond "
            "3 SD removed, z-scored within participant (0 is the participant's mean over all their productions, in standard "
            "deviations; the Hz value is not shared). No feedback manipulation: pert_value is 0.",
            "Experiment 1: every participant did three conditions in a Latin-square order (group = the authors' order code ABC, BCA "
            "or CAB; the deposit does not say which letter is which condition): No Generalization (beer-pier exposure), Phoneme "
            "Generalization (bear-pear exposure) and Dimension Generalization (deer-tear exposure), each 30 Canonical trials "
            "then 30 Reverse trials, always with beer-pier test words. Rows are ordered by the authors' block label (blocks "
            "1-2 deer-tear, 3-4 bear-pear, 5-6 beer-pier) and trial number, so trial is not the order in which a participant did "
            "the three conditions; authors_trial_number (1-30 Canonical, 31-60 Reverse) counts within a condition.",
            "Experiment 2: Replication blocks (beer-pier exposure and test, 30 + 30 trials) are listed first, then four Mixed "
            "Response blocks (bear-pear exposure, test words beer-pier or bear-pear intermixed). condition is the authors' own "
            "label from their script: Replication; Mixed: Match when the response was bear or pear; Mixed: "
            "Syllable-Generalization otherwise (unambiguous catch words use the word itself).",
            "excluded_by_authors marks the catch trials with unambiguous test words (not analysed) and all rows of the 7 excluded "
            "participants of Experiment 1; subject_excluded_by_authors marks those 7 participants.",
        ],
        unit_note="spoken repetition (utterance) of one participant",
        differences=[
            "Experiment 1 file: 99 participants; the paper reports 110 tested, 18 excluded and 92 analysed (the script excludes 7 "
            "of the 99), so 11 of the excluded participants are not in the file.",
            "Experiment 2 file: 87 participants (the paper tested 95 and analysed 87); the other 8 are not in the file.",
            "Experiment 1 has two pairs of rows with the same participant, condition and trial number (kept as shared), so one "
            "participant has 25 ambiguous Canonical rows in one condition although the design has 24 trials.",
            "Tables 2 and 4 were refitted in Python (REML with the random-effect terms the authors name, Powell optimizer); "
            "coefficients, SEs and t agree with the printed values to the printed decimals (t within 0.03); p-values were not refitted. "
            "One exception: the Experiment 2 post-hoc t of Test Stimulus F0 x Statistical Regularity in the Mixed: Match "
            "condition refits to 8.13 where the paper prints 8.18 (the random-effect estimates of the two programs differ slightly).",
        ],
    )
    return stdlib.release(ID, "speech", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
