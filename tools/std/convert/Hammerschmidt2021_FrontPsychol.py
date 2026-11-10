"""Hammerschmidt, Frieler & Wollner (2021), Front Psychol: spontaneous motor tempo in a large online tapping experiment.

Run from the site folder: python3 -I tools/std/convert/Hammerschmidt2021_FrontPsychol.py
Raw files (untrusted data, read only) in raw_extra/Hammerschmidt2021_FrontPsychol/, from https://doi.org/10.5281/zenodo.4897921:
  raw_data.csv      ';'-separated, one row per test participation (5,966 plus three empty rows of the text 'undefined'); the
                    inter-tap intervals of the 15-s trial are the ';'-separated list in the 'intervals' column (ms)
  cleaned_data.csv  the 3,986 participations with complete questionnaire data, with numeric codes (variable_list_cleaned_data.txt)
  variable_list_*.txt  the authors' variable lists (UTF-16 text)
Location (lat, long), user agent, date and year of birth are not converted. The published sample (3,576) is the cleaned file
restricted as the paper's Participants section describes: first participation only, mean ITI of at least 100 ms, age within
1.5 IQR, and without one participant who named Antarctica as heritage.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hammerschmidt2021_FrontPsychol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for n in ("raw_data.csv", "cleaned_data.csv"):
    checks.internal(f"{n} matches the downloaded hash", stdlib.sha256(RAW / n) == src[n]["sha256"])

raw = pd.read_csv(RAW / "raw_data.csv", sep=";", encoding="utf-8", dtype=str)
cl = pd.read_csv(RAW / "cleaned_data.csv", sep=";")
empty = raw["subj_no"] == "undefined"
checks.internal("three trailing rows of the raw file are empty ('undefined' in every field) and are not converted",
                int(empty.sum()) == 3 and bool((raw[empty].drop(columns=[]).nunique() <= 1).all()))
raw = raw[~empty].copy()
raw["subj_no"] = raw["subj_no"].astype(int)
checks.internal("subj_no is unique in the raw file and the cleaned file is a subset of it",
                raw.subj_no.is_unique and set(cl.subj_no) <= set(raw.subj_no))
cl = cl.set_index("subj_no")

# ---- the published sample (Participants section)
first = cl[cl.no_entries == 0]
s2 = first[first.smt_ms >= 100]
q1, q3 = s2.age.quantile([0.25, 0.75])
iqr = q3 - q1
s3 = s2[(s2.age >= q1 - 1.5 * iqr) & (s2.age <= q3 + 1.5 * iqr)]
final = s3[s3.continent != 7]
final_ids = set(final.index)

def toint(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return ""


GENDER = {"male": "M", "female": "F", "other": "other"}
CGENDER = {1: "M", 2: "F", 3: "other"}
DEVICE = {1: "mobile", 2: "computer"}
OS = {1: "Android", 2: "Windows", 3: "iOS", 4: "Mac OS X", 5: "other OS"}
SOFT = {1: "Chrome", 2: "Safari", 3: "Firefox", 4: "QQ Browser", 5: "Edge", 6: "Facebook", 7: "other browser"}

rows, parts = [], []
n_undef = 0
tap_means = {}
for r in raw.itertuples(index=False):
    sid = int(r.subj_no)
    ivs = r.intervals.split(";")
    vals = []
    for x in ivs:
        if x.strip() in ("undefined", ""):
            vals.append(None)
            n_undef += 1
        else:
            vals.append(float(x))
    good = [v for v in vals if v is not None]
    tap_means[sid] = sum(good) / len(good) if good else None
    c = cl.loc[sid] if sid in cl.index else None
    dev = "own device and browser, 15 s of self-paced tapping on a key, mouse button or touchscreen"
    if c is not None:
        dev = (f"{DEVICE[int(c.device_type)]}, {OS[int(c.operating_sys)]}, {SOFT[int(c.software)]}; own device and browser, "
               "15 s of self-paced tapping on a key, mouse button or touchscreen")
    excl = sid not in final_ids
    sex = CGENDER[int(c.gender)] if c is not None else GENDER.get(r.gender, "")
    for k, v in enumerate(vals, start=1):
        rows.append(dict(
            dataset=ID, study="Hammerschmidt et al. 2021 Front Psychol", experiment="online spontaneous motor tempo", group="",
            subject=f"E{sid}", person=f"E{sid}", subject_excluded_by_authors=excl,
            age=int(c.age) if c is not None else "", sex=sex, handedness="", population="general online sample",
            setting="online", device=dev, session=1, block="15-s self-paced tapping (accepted attempt)", trial=k, phase="test",
            time_s="", task="finger tapping and timing", unit="tap", condition="spontaneous motor tempo",
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=excl,
            tapping_trial=1, event_in_trial=k, stimulus="", iti_ms=v))
    q = [toint(x) for x in (r.q4, r.q5, r.q6, r.q7)]
    p = dict(person=f"E{sid}", n_tries=toint(r.no_tries), previous_participations=r.no_of_entries,
             smt_bpm=float(r.smt_bpm), smt_cv=float(r.smt_cv), heritage=r.heritage, city_size=r.city_size,
             music_making=toint(r.q1), arousal=toint(r.q2), work_hours_rating=toint(r.q3),
             pss_items=";".join(str(x) for x in q) if all(x != "" for x in q) else "",
             in_cleaned_file=c is not None, in_published_sample=not excl)
    if c is not None:
        p.update(continent=int(c.continent), weekday=int(c.weekday), time_of_day=int(c.day_cat), pss_score=int(c.pss_score))
    parts.append(p)

participant_columns = [
    ("n_tries", "integer", "", "Number of tapping attempts until one met the authors' criterion (coefficient of variation of the intervals at most 0.1, at least 8 taps); the rows are the accepted attempt."),
    ("previous_participations", "text", "Never, Once, Twice, More than twice", "How often the person had taken part before (as the authors recorded it)."),
    ("smt_bpm", "number", "beats per minute", "Authors' mean inter-tap interval of the attempt, in beats per minute (60000 / mean interval in ms)."),
    ("smt_cv", "number", "", "Authors' coefficient of variation of the inter-tap intervals."),
    ("heritage", "text", "country code", "Country the participant reported as their heritage, as a code."),
    ("city_size", "text", "", "Size of the place of residence as the participant chose it."),
    ("music_making", "integer", "1 never ... 6 professional", "Answer to 'Do you make music?'."),
    ("arousal", "integer", "1 very calm ... 5 very excited", "Answer to 'How do you feel right now?'."),
    ("work_hours_rating", "integer", "1 none ... 7 more than 50 hours", "Hours of studying or working in a typical week."),
    ("pss_items", "text", "four answers separated by ;, each 1 never ... 5 very often", "Answers to the four Perceived Stress Scale (short form) items q4-q7, in the order asked (items q5 and q6 are reverse-scored in pss_score)."),
    ("in_cleaned_file", "true/false", "", "The participation is in the authors' cleaned file (complete questionnaire data)."),
    ("in_published_sample", "true/false", "", "The participation is in the 3,576 participants of the paper's analysis."),
    ("continent", "integer", "1 Europe, 2 Asia, 3 North America, 4 South America, 5 Africa, 6 Oceania, 7 Antarctica", "Continent of the heritage country (cleaned file only)."),
    ("weekday", "integer", "1 Monday ... 7 Sunday", "Day of the week of the test (cleaned file only)."),
    ("time_of_day", "integer", "1 night, 2 morning, 3 afternoon, 4 evening", "Time of day of the test (cleaned file only)."),
    ("pss_score", "integer", "0-16", "Authors' Perceived Stress Scale short-form score (cleaned file only)."),
]
rows_df = pd.DataFrame(rows)
parts_df = pd.DataFrame(parts)

# ---- checks
checks.published("participations in the raw data", len(raw), 5966, 0, "Participants section (N = 5,966 took part)")
checks.published("participants in the analysed sample", len(final_ids), 3576, 0,
                 "Participants section (3,986 -> 3,704 first participations -> 3,703 -> 3,577 -> 3,576)")
checks.published("participants in the first-participation step", len(first), 3704, 0, "Participants section")
checks.published("participants after the age-outlier step", len(s3), 3577, 0, "Participants section")
checks.published("participants with complete questionnaire data (cleaned file)", len(cl), 3986, 0, "Participants section")
m = pd.Series({s: tap_means[s] for s in final_ids})
checks.published("mean SMT (mean inter-tap interval, ms) in the analysed sample", m.mean(), 780, 0.5, "Results (SMT descriptives)")
checks.published("SD of SMT (ms)", m.std(), 328, 0.5, "Results (SMT descriptives)")
checks.published("median SMT (ms)", m.median(), 729, 0.5, "Results (SMT descriptives)")
checks.published("fastest SMT (ms)", m.min(), 123, 0.5, "Results (SMT descriptives, range 123-2150 ms)")
checks.published("slowest SMT (ms)", m.max(), 2150, 0.5, "Results (SMT descriptives, range 123-2150 ms)")
ages = cl.loc[sorted(final_ids), "age"]
checks.published("mean age (years) of the analysed sample", ages.mean(), 27.64, 0.01, "Participants section")
checks.published("SD of age (years)", ages.std(), 7.61, 0.01, "Participants section")
checks.published("youngest and oldest age", float(ages.min() + ages.max()), 7 + 49, 0, "Participants section (age 7-49)")
rho = pd.Series(m.values).corr(pd.Series(ages.loc[list(m.index)].values), method="spearman")
checks.published("Spearman correlation of SMT with age", rho, 0.12, 0.006, "Table 2 (rho = 0.12)")
sm = pd.Series({s: cl.loc[s, "smt_ms"] for s in cl.index})
checks.internal("the authors' smt_ms (cleaned file) equals the mean of the converted intervals for all 3,986 participations",
                float(np.max(np.abs(sm.values - np.array([tap_means[s] for s in sm.index])))) < 1e-4)
okb = [s for s in tap_means if tap_means[s]]
chk = np.array([float(x) for x in raw.set_index("subj_no").loc[okb].smt_bpm])
checks.internal(f"smt_bpm in the raw file = 60000 / mean of the converted intervals ({len(okb)} participations with a valid interval)",
                float(np.nanmax(np.abs(chk - 60000 / np.array([tap_means[s] for s in okb])))) < 1e-3)
checks.internal("every participation has at least one interval; number of intervals per participation 1-342",
                min(r["event_in_trial"] for r in rows) == 1 and rows_df.groupby("subject").size().max() == 342)
checks.internal("'undefined' interval entries are kept as rows with an empty iti_ms", n_undef == 10 and rows_df.iti_ms.isna().sum() == 10,
                f"{n_undef} entries")

meta = dict(
    study="Hammerschmidt et al. 2021",
    paper=dict(authors="Hammerschmidt D, Frieler K, Wollner C", year=2021,
               title="Spontaneous Motor Tempo: Investigating Psychological, Chronobiological, and Demographic Factors in a Large-Scale Online Tapping Experiment",
               venue="Front Psychol 12:677201", doi="10.3389/fpsyg.2021.677201"),
    lab="Wollner (University of Hamburg)",
    data_url="https://doi.org/10.5281/zenodo.4897921",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced: the ';'-separated list of inter-tap intervals of each "
             "participation was unpacked into one row per interval; the questionnaire answers, the authors' tapping summaries and the "
             "variables of the cleaned file went into participants.csv. The device location (latitude, longitude, accuracy), user-agent "
             "text, test date and year of birth were not converted, and three empty rows at the end of the raw file were dropped."),
    how=[
        "One row is one inter-tap interval of a participation: participants tapped with the index finger for 15 s at a comfortable, "
        "natural pace on a device of their choice (keyboard, mouse or touchscreen) in a web browser, repeating the trial until the "
        "coefficient of variation of the intervals was at most 0.1 with at least 8 taps; the rows are the accepted attempt. iti_ms is "
        "the interval in ms that ends with the tap, event_in_trial counts it from 1. The deposit has no tap times, so tap_time_ms is empty.",
        "All 5,966 participations of the raw file are rows. Participations that the authors did not analyse are marked "
        "subject_excluded_by_authors: those without complete questionnaire data (not in the cleaned file), repeat participations, "
        "mean intervals under 100 ms, age outliers (1.5 IQR) and one person who named Antarctica; the 3,576 that remain reproduce the paper. "
        "The deposit does not link repeat participations to a person, so each participation is its own person.",
        "Age is given only for the 3,986 participations of the cleaned file; sex, device, operating system and browser come from "
        "the cleaned file when it has the participation, else sex from the raw file. participants.csv holds the questionnaire answers, "
        "the authors' mean tempo in beats per minute and coefficient of variation, and the cleaned-file variables (all codes explained "
        "in participants_columns.csv).",
        "Spontaneous motor tempo (SMT) in the paper is each participant's mean inter-tap interval; 10 interval entries are the text "
        "'undefined' in the raw file and are written as rows with an empty iti_ms (the authors' mean ignores them).",
    ],
    unit_note="inter-tap interval of a 15-s self-paced tapping trial (one participation)",
    differences=[],
)
stdlib.release(ID, "tapping", rows, meta, checks, participants=parts_df, participant_columns=participant_columns)
