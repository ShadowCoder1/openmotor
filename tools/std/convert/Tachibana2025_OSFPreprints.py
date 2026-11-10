"""Tachibana & Iimura (2025), OSF Preprints: vocal pitch compensation and f0 variability in adults who stutter (AWS) and
adults who do not (ANS), 16 per group (OSF 67xkw).

Raw files (raw_extra/Tachibana2025_OSFPreprints/, fetched with tools/std/fetch.py):
  alldata.csv   Subject, isAWS, Sustained, Within, Between, CompRate, CmpAmntN300..P300   one row per participant x adaptive shift
  indivdata.csv Subject, isAWS, SSI, MusExp                                               one row per participant
What the values are (preprint, Methods 2.1-2.4, read as text):
  Sustained  the constant f0 shift of the auditory feedback in the adaptation phase, cents (0, +400, -400)
  Within     f0 variability index of the adaptation phase: SD of f0 over 100-350 ms from vocal onset, cents (the authors' name is
             'Within'; the paper calls it the variability index; the correlation with CompRate reproduces the paper's r)
  Between    a second variability column of the deposit that neither the preprint nor the deposit defines (kept, flagged)
  CompRate   compensation ratio, sign-reversed slope of f0 change (250-350 ms) against test-phase shift (0, +-100, +-200 cents)
  CmpAmnt*   mean f0 change 250-350 ms after onset for test shifts of -300..+300 cents (N = minus, P = plus)
  SSI        Stuttering Severity Instrument score (AWS only); MusExp music experience (years assumed, not stated)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tachibana2025_OSFPreprints"
SRC = "Tachibana & Iimura 2025 preprint (OSF ceurq_v1), Results"
SHIFTS = [("N300", -300), ("N200", -200), ("N100", -100), ("0", 0), ("P100", 100), ("P200", 200), ("P300", 300)]
CTX = "Vowel /a/ held 0.5 s with a guide tone; feedback f0 shifted by a constant amount in the adaptation phase, then by test shifts."


def build():
    raw = stdlib.raw_dir(ID)
    a = pd.read_csv(raw / "alldata.csv")
    ind = pd.read_csv(raw / "indivdata.csv").set_index("Subject")
    rows = []

    def add(sub, aws, measure, desc, stat, val, unit, cond, pt, pv, sess):
        if pd.isna(val):
            return
        rows.append({
            "dataset": ID, "study": ID, "experiment": "", "group": "AWS" if aws else "ANS", "subject": sub,
            "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
            "population": "stuttering" if aws else "healthy", "setting": "lab", "device": "", "session": sess, "block": "",
            "trial": 0, "phase": "other", "time_s": "", "task": "speech", "unit": "participant summary", "condition": cond,
            "pert_type": pt, "pert_value": pv, "pert_unit": "cents", "feedback": "altered" if pv != 0 else "normal",
            "excluded_by_authors": False, "measure": measure, "measure_description": desc, "statistic": stat,
            "value": float(val), "value_unit": unit, "n_trials": "", "session_context": CTX})

    for r in a.itertuples(index=False):
        aws = bool(r.isAWS)
        sh = int(r.Sustained)
        cond = f"adaptive shift {sh:+d} cents" if sh else "adaptive shift 0 cents"
        sess = {0: 1, 400: 2, -400: 3}[sh]  # order not stated; session only labels the shift condition
        pt = "pitch_shift" if sh else "none"
        add(r.Subject, aws, "f0_variability_within", "Authors' column Within: f0 variability index of the adaptation phase "
            "(SD of f0 over 100-350 ms from vocal onset, low-pass filtered at 20 Hz)", "sd", r.Within, "cents", cond, pt, sh, sess)
        add(r.Subject, aws, "variability_between", "Authors' column Between; not defined in the preprint or the deposit", "other",
            r.Between, "cents", cond, pt, sh, sess)
        add(r.Subject, aws, "compensation_ratio", "Sign-reversed slope of the f0 change (250-350 ms) against the test-phase shift "
            "(0, +-100, +-200 cents); 1 is complete compensation, 0 none", "slope", r.CompRate, "ratio", cond, pt, sh, sess)
        for k, s in SHIFTS:
            add(r.Subject, aws, f"compensation_amount_test_{'+' if s > 0 else ''}{s}",
                f"Mean f0 change 250-350 ms after vocal onset in the test phase for a {s:+d} cent feedback shift, trend removed; "
                "negative when the participant opposes a positive shift", "mean", getattr(r, "CmpAmnt" + k), "cents", cond, pt, sh, sess)
    for sub, r in ind.iterrows():
        aws = bool(r.isAWS)
        add(sub, aws, "SSI_score", "Stuttering Severity Instrument score (SSI-4)", "value", r.SSI, "score", "", "none", 0, 1)
        add(sub, aws, "music_experience", "Authors' column MusExp, duration of music experience (unit not stated; years assumed)",
            "value", r.MusExp, "years", "", "none", 0, 1)
    return pd.DataFrame(rows)


def main():
    checks = stdlib.Checks()
    df = build()
    n = df.groupby("group").subject.nunique()
    checks.published("ANS participants analysed", n["ANS"], 16, 0, SRC + " (n = 16)")
    checks.published("AWS participants analysed", n["AWS"], 16, 0, SRC + " (n = 16)")
    z = pd.read_csv(stdlib.raw_dir(ID) / "alldata.csv")
    z = z[z.Sustained == 0]
    for g, rep, nm in ((0, 0.58, "ANS"), (1, -0.27, "AWS")):
        x = z[z.isAWS == g]
        checks.published(f"{nm}: Pearson r of variability and compensation ratio, 0-cent adaptive shift",
                         x.Within.corr(x.CompRate), rep, 0.005, SRC + " (Fig 3c,d text)")
    x = z[z.isAWS == 1]
    x = x.drop(x.CompRate.idxmax())
    checks.published("AWS: same r without the outlier (largest compensation), n = 15", x.Within.corr(x.CompRate), 0.08, 0.005,
                     SRC + " (Fig 3d text)")
    checks.internal("96 participant x shift cells, 32 people", df.groupby(["subject", "condition"]).ngroups >= 96 and df.subject.nunique() == 32)
    checks.internal("no missing values in value", bool(df.value.notna().all()))
    s, p = None, None
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    meta = dict(
        study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC-By Attribution 4.0 International (the OSF preprint; the data project states no license)",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Reshaped the authors' two wide per-participant tables into one long table and replaced participant labels with codes.",
        how=["One row is one measure of one participant in one adaptive-shift condition (a constant f0 shift of 0, +400 or -400 cents "
             "heard during the adaptation phase); SSI and music experience have one row per participant.",
             "Values are in cents (compensation_ratio unitless). compensation_amount_test_* is the f0 change 250-350 ms after vocal onset "
             "for the test shift named; opposing a positive shift gives a negative value (inferred from the group means and the "
             "sign-reversed slope that defines the compensation ratio).",
             "Session numbers only label the three shift conditions (their order is not given). Age and sex were not shared. "
             "The 'Between' column is kept as shared but is undefined in the deposit and preprint."],
        unit_note="participant, measure and adaptive shift",
        differences=["The preprint is a preprint (2025); the 4 participants per group it excluded are not in the deposit."])
    return stdlib.release(ID, "summary", df.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
