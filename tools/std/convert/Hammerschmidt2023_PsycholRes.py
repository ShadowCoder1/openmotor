"""Hammerschmidt & Woellner (2022/2023), Psychological Research 87(1):327-338: spontaneous motor tempo (SMT) measured four times
a day for seven days (28 sessions) in 36 adults, by tapping with the index finger on a smartphone screen for 25 s.

Raw files, raw_extra/Hammerschmidt2023_PsycholRes/ (Zenodo 5947002, CC BY 4.0):
  smt_esm_data.csv            1008 rows = 36 participants x 28 sessions, semicolon-separated, latin-1; one row per session with
                              the mean inter-tap interval (smt_ms), its coefficient of variation (smt_cv), bpm = 60000 / smt_ms,
                              the number of tries, test duration, and the questionnaire answers of that moment
  smt_esm_variable_list.txt   codebook (read as text)
The deposit holds one summary per tapping session, not single taps, so the summary format is used (one row = one measure of one
session). Participant-level covariates go in participants.csv. userID, the date and the clock time are not carried.
Exclusions: the paper (Data processing) removed 62 incomplete or missed tests and then, per participant, tests whose ITI mean was an
outlier by the 1.5 IQR rule; the deposit does not flag them, so the flag is rebuilt here: no SMT, or outside 1.5 IQR of the
participant's own SMT values.
Run from the site folder: python3 -I tools/std/convert/Hammerschmidt2023_PsycholRes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hammerschmidt2023_PsycholRes"
SRC = "Hammerschmidt & Woellner 2022, Psychological Research 87:327"
RAW = stdlib.raw_dir(ID)

# (column, measure, description, statistic, unit)
MEASURES = [
    ("smt_ms", "smt_mean_iti_ms", "Spontaneous motor tempo: mean inter-tap interval over about 25 s of even, comfortable index-finger tapping on a smartphone screen (column smt_ms).", "mean", "ms"),
    ("smt_cv", "smt_cv", "Coefficient of variation (SD / mean) of the inter-tap intervals of the session, as a proportion (column smt_cv; sessions above 0.30 were repeated).", "value", "proportion"),
    ("bpm", "smt_bpm", "Tapping rate in beats per minute (column bpm; 60000 divided by smt_ms).", "value", "beats per minute"),
    ("tries", "tapping_tries", "Number of attempts needed for an acceptable tapping run (CV of intervals below 0.30 and at least 8 taps) (column tries).", "count", "attempts"),
    ("exp_duration", "test_duration_ms", "Time the participant took for the whole test of the session (column exp_duration).", "value", "ms"),
    ("stress_level", "stress_level", "Stress felt at the moment, 1 (not at all) to 7 (very) (column stress_level).", "value", "rating 1-7"),
    ("paq_total", "paq_total", "Physiological Arousal Questionnaire total at the moment of the test; higher means more arousal (column paq_total).", "value", "score"),
    ("emotion_intensity", "emotion_intensity", "Intensity of the emotion reported at the moment, 1 to 7 (column emotion_intensity).", "value", "rating 1-7"),
]
PCOLS = [
    ("chronotype_category", "text", "morning, evening, neither", "Chronotype category from the D-MEQ (column chrono_cat)."),
    ("chronotype_score", "number", "D-MEQ points (16-86)", "Morningness-eveningness score of the D-MEQ (column chrono_score)."),
    ("gold_msi_general", "number", "points", "Gold-MSI general musical sophistication score (column gold_msi)."),
    ("best_instrument", "text", "", "Best played instrument, or none (column instrument)."),
    ("education_code", "integer", "1-6", "School degree code of the codebook (column education)."),
    ("training_code", "integer", "1-10", "Higher education / vocational training code of the codebook (column training)."),
    ("profession_code", "integer", "1-14", "Job field code of the codebook (column profession)."),
    ("employment_code", "integer", "1-6", "Employment status code of the codebook (column employment)."),
]
TOD = ["morning", "afternoon", "evening", "night"]


def iqr_out(g):
    q1, q3 = g.quantile(.25), g.quantile(.75)
    i = q3 - q1
    return (g < q1 - 1.5 * i) | (g > q3 + 1.5 * i)


def ctx(r):
    bits = []
    for k, c in (("location", "location"), ("activity", "activity"), ("company", "company"), ("listening to music", "music_listening")):
        if isinstance(r[c], str) and r[c].strip():
            bits.append(f"{k}={r[c].strip()}")
    return "; ".join(bits)


def main():
    checks = stdlib.Checks()
    d = pd.read_csv(RAW / "smt_esm_data.csv", sep=";", encoding="latin-1")
    d["miss"] = d.smt_ms.isna()
    d["outl"] = d.groupby("subj_no").smt_ms.transform(iqr_out)
    d["excl"] = d.miss | d.outl
    rows, parts = [], []
    for subj, g in d.groupby("subj_no", sort=True):
        g = g.sort_values("order")
        f = g.iloc[0]
        code = f"P{int(subj):02d}"
        n = 0
        base = {"dataset": ID, "study": ID, "experiment": "", "group": "", "subject": code, "person": code,
                "subject_excluded_by_authors": False, "age": float(f.age), "sex": "F" if f.gender == "female" else "M",
                "handedness": "", "population": "healthy", "setting": "home",
                "device": "index-finger tapping on a smartphone touchscreen in a web application"}
        for r in g.itertuples(index=False):
            r = r._asdict()
            for col, name, desc, stat, unit in MEASURES:
                v = r[col]
                if pd.isna(v):
                    continue
                n += 1
                rows.append({**base, "session": int(r["dayN"]) + 1, "block": str(r["day_time"]), "trial": n, "phase": "test",
                             "time_s": "", "task": "finger tapping and timing", "unit": "participant summary",
                             "condition": str(r["time"]), "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal",
                             "excluded_by_authors": bool(r["excl"]), "measure": name, "measure_description": desc, "statistic": stat,
                             "value": float(v), "value_unit": unit, "n_trials": "", "session_context": ctx(r)})
        parts.append({"person": code, "chronotype_category": f.chrono_cat, "chronotype_score": f.chrono_score,
                      "gold_msi_general": f.gold_msi, "best_instrument": f.instrument, "education_code": int(f.education),
                      "training_code": int(f.training), "profession_code": int(f.profession), "employment_code": int(f.employment)})
    out = pd.DataFrame(rows)
    # checks
    smt = out[out.measure == "smt_mean_iti_ms"]
    kept = d[~d.excl]
    checks.published("participants", d.subj_no.nunique(), 36, 0, f"{SRC}, Method (Participants)")
    checks.published("female participants", d.groupby("subj_no").gender.first().eq("female").sum(), 21, 0, f"{SRC}, Method (Participants)")
    a = d.groupby("subj_no").age.first()
    checks.published("age mean", a.mean(), 28.64, 0.005, f"{SRC}, Method (Participants)")
    checks.published("age SD", a.std(), 5.00, 0.005, f"{SRC}, Method (Participants)")
    checks.published("tests (36 x 28)", len(d), 1008, 0, f"{SRC}, Data processing")
    checks.published("Gold-MSI general: mean over participants", d.groupby("subj_no").gold_msi.first().mean(), 71.11, 0.01, f"{SRC}, Method (Participants)")
    checks.published("Gold-MSI general: SD", d.groupby("subj_no").gold_msi.first().std(), 23.91, 0.01, f"{SRC}, Method (Participants)")
    checks.published("minimum SMT (ms)", kept.smt_ms.min(), 181, 0.5, f"{SRC}, Results (descriptives)")
    checks.published("maximum SMT (ms)", kept.smt_ms.max(), 1822, 0.5, f"{SRC}, Results (descriptives)")
    checks.published("overall mean SMT (ms)", kept.smt_ms.mean(), 650, 1.5, f"{SRC}, Results (descriptives)")
    checks.published("overall SD of SMT (ms)", kept.smt_ms.std(), 253, 1.5, f"{SRC}, Results (descriptives)")
    checks.internal("rebuilt exclusions leave about the 946 tests the paper analysed (not exact)", abs(len(kept) - 946) <= 15, str(len(kept)))
    checks.internal("1008 sessions = 36 x 28", d.groupby("subj_no").size().eq(28).all())
    checks.internal("bpm = 60000 / smt_ms in the rows", bool(((d.bpm * d.smt_ms - 60000).abs() < 1).mean() > .95))
    checks.internal("every smt_ms row has an output row", len(smt) == d.smt_ms.notna().sum())
    checks.internal("no lost participants", out.person.nunique() == 36)
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Stacked the 1008 sessions into long form (one row per measure that has a value, in the authors' row order), moved age, sex "
                 "and the setting words into standard columns, put the chronotype, musical sophistication and demographic codes into "
                 "participants.csv, dropped the user ID, the date and clock time, the free-time-perception and physical-activity "
                 "questionnaire columns, and recoded participants as codes. Flagged the sessions the paper excluded (see how)."),
        how=["One row is one measure of one tapping session (28 per participant: four a day, morning to night, for seven days from a "
             "Monday). The tapping measures are the mean inter-tap interval smt_ms (milliseconds, the spontaneous motor tempo), its "
             "coefficient of variation (a proportion), the rate in beats per minute and the number of tries; stress, arousal and emotion "
             "intensity are the ratings given at the same moment. Block holds the authors' day_time label (day, then time-slot), "
             "condition the time of day.",
             "The deposit does not mark excluded sessions. excluded_by_authors is rebuilt from the paper's Data processing section: "
             "sessions with no SMT, and sessions whose SMT lies outside 1.5 interquartile ranges of that participant's own SMT values.",
             "There are no per-tap data in the deposit."],
        unit_note="measure of one tapping session", differences=[
            "The paper analysed N = 946 tests; the rebuilt exclusion flag (no SMT, per-participant 1.5 IQR outlier) leaves %d, because the "
            "deposit does not say which further 13 or so tests (for example missing questionnaire answers or a negative time of day) were left out." % len(kept),
            "The paper reports a mean tapping variability (CV) of 9.25%% (SD 5.46); the rows give %.2f%% (SD %.2f) for the sessions kept, so "
            "this number was not reproduced." % (100 * kept.smt_cv.mean(), 100 * kept.smt_cv.std())])
    stdlib.release(ID, "summary", out, meta, checks, participants=pd.DataFrame(parts), participant_columns=PCOLS)


main()
