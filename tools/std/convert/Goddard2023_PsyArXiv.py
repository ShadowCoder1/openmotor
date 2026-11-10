"""Goddard, Pinet, Kartushina, Koutsogiannaki & Martin (2023, PsyArXiv xysc5): feedback alteration (FA) training of the English
vowel /ɪ/ in native Spanish speakers (BCBL), 22 participants, 3 training days, plus a picture-naming vowel production task
before and after each training day and at a one-week retest.

Raw files, raw_extra/Goddard2023_PsyArXiv/ (OSF g5eru, data folder, fetched with tools/std/fetch.py):
  all_sub_M.txt   no header, space separated, one row per training trial: file code (group letter, 2-digit participant,
                  2-digit day), trial (1-250), F1, F2 (Hz produced, Audapter; 0 = no estimate), feedbackF1, feedbackF2 (Hz heard),
                  euclideanDist, avDist (the authors' distance to the target vowel); 65 files x 250 trials
  fa_cleaned.csv  the authors' cleaned version of those rows (FA_1_preprocessing.Rmd): which rows they kept, phase, gender
  VST.csv         vowel production task: one row per utterance (group, subject, session 1-7, word, vowel, F1, F2, F3 in Hz)
What the values are (preprint Methods, Data processing; the authors' code FA_1_preprocessing.Rmd, FA_2_compensation_adaptation.Rmd,
VowelProductionTask.Rmd, read as text): 250 training trials per day: baseline 1-40 (the first 10 dropped by the authors), feedback
41-140 (alteration ramped over 100 steps towards the target, experimental group only), noise-increase 141-170 (alteration held at
its maximum while noise rises), noise-max 171-200 (alteration off, noise at maximum), noise-decrease 201-230, washout 231-250.
The control group heard unaltered feedback with the same noise. FA_2 drops day 2 of participant E02 (missing baseline trials).
The columns F1_central, F2_central, F3_central, duration and duration_central of VST.csv are not described in the deposit or paper
and are left out; the perception files (Raw3AFC.csv, RawAxB.csv) are perceptual tasks and are not included.
Run from the site folder: python3 -I tools/std/convert/Goddard2023_PsyArXiv.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Goddard2023_PsyArXiv"
PAPER_SRC = "Goddard et al. 2023 (PsyArXiv preprint xysc5)"
GROUP = {"C": "Control", "E": "Experimental"}
DEVICE_FA = "Audapter real-time formant shifting, Beyerdynamic DT 770 PRO headphones, Xenyx 502 mixer, MOTU MicroBook II"
DEVICE_VST = "Marantz PMD671 recorder, Beyerdynamic DT 770 PRO headphones"
PHASES = [("Baseline", 0, 40, "baseline", "none"),
          ("Feedback", 40, 140, "learning", "none"),
          ("NoiseBuild", 140, 170, "learning", "noise rising to its maximum"),
          ("NoiseMax", 170, 200, "other", "noise at its maximum"),
          ("NoiseReduce", 200, 230, "other", "noise falling to zero"),
          ("Washout", 230, 250, "washout", "none")]


def phase_of(t):
    for name, lo, hi, std, noise in PHASES:
        if lo < t <= hi:
            return name, std, noise
    raise ValueError(t)


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def num(v):
    return "" if pd.isna(v) else float(v)


def build(checks):
    raw = stdlib.raw_dir(ID)
    r = pd.read_csv(raw / "all_sub_M.txt", sep=r"\s+", header=None,
                    names=["subject", "trial", "F1", "F2", "feedbackF1", "feedbackF2", "euclideanDist", "avDist"])
    fa = pd.read_csv(raw / "fa_cleaned.csv")
    vst = pd.read_csv(raw / "VST.csv")
    kept = set(zip(fa.subject, fa.trial))
    r["code"] = r.subject.str[:3].str.upper()
    r["day"] = r.subject.str[4].astype(int)
    # sex: the authors' VST column per participant; one participant (C02) has no VST rows of that kind, so use the authors' rule
    # from FA_1_preprocessing.Rmd (mean F2 below 2100 Hz = male) for anyone missing
    vst["code"] = vst.group.str.upper() + vst.subject.astype(int).astype(str).str.zfill(2)
    sex = vst.groupby("code").gender.first().map({"Female": "F", "Male": "M"}).to_dict()
    for code, g in r.groupby("code"):
        if code not in sex:
            m = g.F2[g.F2 > 0].mean()
            sex[code] = "M" if m < 2100 else "F"
    rows = []
    for x in r.itertuples(index=False):
        code, grp = x.code, GROUP[x.code[0]]
        name, std, noise = phase_of(int(x.trial))
        f1 = None if x.F1 == 0 else float(x.F1)
        f2 = None if x.F2 == 0 else float(x.F2)
        h1 = None if x.feedbackF1 == 0 else float(x.feedbackF1)
        h2 = None if x.feedbackF2 == 0 else float(x.feedbackF2)
        s1 = "" if f1 is None or h1 is None else h1 - f1
        s2 = "" if f2 is None or h2 is None else h2 - f2
        s1 = 0.0 if s1 != "" and abs(s1) < 1e-6 else s1
        s2 = 0.0 if s2 != "" and abs(s2) < 1e-6 else s2
        shifted = (s1 != "" and s1 != 0) or (s2 != "" and s2 != 0)
        in_cleaned = (x.subject, x.trial) in kept
        excl = (not in_cleaned) or (code == "E02" and x.day == 2)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Feedback alteration training", "group": grp,
            "subject": f"{code}|day{x.day}", "person": code, "subject_excluded_by_authors": code == "C02",
            "age": "", "sex": sex[code], "handedness": "", "population": "healthy (native Spanish speakers learning English)",
            "setting": "lab", "device": DEVICE_FA, "session": int(x.day), "block": name, "trial": int(x.trial), "phase": std,
            "time_s": "", "task": "speech", "unit": "utterance", "condition": grp,
            "pert_type": "formant_shift" if shifted else "none", "pert_value": s1 if s1 != "" else 0.0, "pert_unit": "Hz",
            "feedback": "masked" if "noise" in noise else ("altered" if shifted else "normal"),
            "excluded_by_authors": bool(excl), "stimulus": "pip", "segment": "vowel /ɪ/",
            "measure_window": "formants as tracked by Audapter on the trial",
            "f1_hz": "" if f1 is None else f1, "f2_hz": "" if f2 is None else f2,
            "f1_heard_hz": "" if h1 is None else h1, "f2_heard_hz": "" if h2 is None else h2,
            "shift_f1": s1, "shift_f2": s2, "masking_noise": noise,
            "distance_to_target_hz": num(x.euclideanDist), "distance_to_target_smoothed_hz": num(x.avDist),
        })
    n_fa = len(rows)
    vst["person"] = vst.code
    vst["pos"] = vst.groupby(["person", "session"]).cumcount() + 1
    for x in vst.itertuples(index=False):
        grp = GROUP[x.person[0]]
        before = x.feedback == "before-feedback"
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Vowel production task", "group": grp,
            "subject": f"{x.person}|session{x.session}", "person": x.person, "subject_excluded_by_authors": False,
            "age": "", "sex": sex[x.person], "handedness": "", "population": "healthy (native Spanish speakers learning English)",
            "setting": "lab", "device": DEVICE_VST, "session": int(x.session),
            "block": f"{x.day} {x.feedback}", "trial": int(x.pos),
            "phase": "baseline" if x.session == 1 else "test", "time_s": "", "task": "speech", "unit": "utterance",
            "condition": x.language, "pert_type": "none", "pert_value": 0.0, "pert_unit": "Hz", "feedback": "normal",
            "excluded_by_authors": False, "stimulus": x.word, "segment": f"vowel /{x.vocal}/",
            "measure_window": "formants extracted with Praat for the vowel", "f1_hz": float(x.F1), "f2_hz": float(x.F2),
            "f3_hz": float(x.F3), "masking_noise": "none",
        })
    return rows, r, fa, vst, n_fa, sex


def main():
    checks = stdlib.Checks()
    rows, r, fa, vst, n_fa, sex = build(checks)
    a = pd.DataFrame(rows)
    f = a.iloc[:n_fa]
    v = a.iloc[n_fa:]
    persons = set(a.person)
    checks.internal("training: 65 files x 250 trials, in order", len(f) == 65 * 250 and f.groupby("subject").trial.apply(lambda t: (t.diff().dropna() > 0).all()).all())
    checks.internal("training rows marked kept equal the authors' cleaned file (trials 11-250 after their cleaning)",
                    int((~f.excluded_by_authors).sum()) == len(fa) - int(((fa.subj == "E02") & (fa.day == 2)).sum()),
                    f"{int((~f.excluded_by_authors).sum())} kept")
    ctl = f[(f.group == "Control") & (f.feedback != "masked")]
    checks.internal("control group: feedback equal to production (no shift) on every trial with a valid estimate",
                    bool((pd.to_numeric(ctl.shift_f1, errors="coerce").fillna(0).abs() < 1e-6).all()))
    exp_fb = f[(f.group == "Experimental") & (f.block == "Feedback")]
    checks.internal("experimental group: alteration present in the Feedback phase",
                    float((pd.to_numeric(exp_fb.shift_f2, errors="coerce").abs() > 1).mean()) > 0.5)
    # checks against the paper
    src = f"{PAPER_SRC}, Method"
    checks.published("participants", len(persons), 22, 0, f"{src}, Participants")
    checks.published("females", sum(1 for p in persons if sex[p] == "F"), 14, 0, f"{src}, Participants")
    checks.published("males", sum(1 for p in persons if sex[p] == "M"), 8, 0, f"{src}, Participants")
    checks.published("trials per training day", int(f.groupby("subject").trial.max().max()), 250, 0, f"{src}, Tasks and material")
    ph = f.groupby("block").trial.nunique()
    checks.published("baseline-phase trials", int(ph["Baseline"]), 40, 0, f"{src}, Tasks and material, Feedback alteration (FA) training")
    checks.published("feedback-phase trials", int(ph["Feedback"]), 100, 0, f"{src}, Tasks and material, Feedback alteration (FA) training")
    checks.published("noise-phase trials", int(ph["NoiseBuild"] + ph["NoiseMax"] + ph["NoiseReduce"]), 90, 0,
                     f"{src}, Tasks and material, Feedback alteration (FA) training")
    checks.published("washout-phase trials", int(ph["Washout"]), 20, 0, f"{src}, Tasks and material, Feedback alteration (FA) training")
    pct = 100 * (1 - len(fa) / len(f))  # rows of the authors' cleaned file (before FA_2 drops day 2 of one participant)
    checks.published("share of training data discarded (%)", pct, 35.7, 0.05, f"{src}, Results, Feedback alteration training")
    checks.published("words in the vowel production task (4 Spanish, 5 English)", v.stimulus.nunique(), 9, 0,
                     f"{src}, Tasks and material, Production task")
    checks.internal("vowel production task: 13715 rows in the authors' order, sessions 1-7", len(v) == 13715 and set(v.session) == set(range(1, 8)))
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Stacked the authors' training trials (all_sub_M.txt) and their vowel production task (VST.csv) into one table "
                 "with an experiment label, wrote formant values of 0 (no estimate) as empty, computed the feedback shift as heard "
                 "minus produced formant, marked the rows the authors dropped in their cleaning, and replaced participant codes. "
                 "The perception tasks and undocumented columns are not included."),
        how=[
            "Experiment 'Feedback alteration training': one row is one production of the word 'pip' in a 250-trial training "
            "day (3 days per participant; the experimental group heard /ɪ/ shifted towards the English target, the control group "
            "heard unaltered feedback; both groups had noise in the later phases). f1_hz and f2_hz are the formants Audapter tracked "
            "on the trial, f1_heard_hz and f2_heard_hz the formants of the feedback played back, all in Hz.",
            "shift_f1 and shift_f2 are heard minus produced formant in Hz, computed here from the authors' two columns (empty where "
            "either had no estimate); pert_value repeats shift_f1. distance_to_target_hz and distance_to_target_smoothed_hz are the "
            "authors' euclideanDist and avDist columns (Hz, distance to the English target vowel).",
            "excluded_by_authors marks training rows that are absent from the authors' cleaned file (trials 1-10, formant estimates "
            "of 0, out-of-range values, local outliers, sessions with more than half zeros) and day 2 of one participant that "
            "their analysis drops; subject_excluded_by_authors marks the participant they removed.",
            "Experiment 'Vowel production task': one row is one naming of a picture word (4 Spanish and 5 English words, 10 repetitions "
            "each) with Praat formants f1_hz, f2_hz, f3_hz in Hz, in session 1 (before training day 1), 2 to 6 (after / before each "
            "day's training) and 7 (retest one week later). Feedback was normal. sex is the authors' gender column "
            "(for the training-only participant their F2 rule); age was not shared.",
        ],
        unit_note="spoken word (utterance) of one participant",
        differences=[
            "Zero formant values: the paper reports 2.7% of the data in the sessions it kept; counting the rows of those sessions "
            "here gives 2.85%.",
            "Utterances per session: the paper describes 9 words x 10 repetitions (90 per participant and session); the file has up to "
            "107 rows for some participant-sessions, all kept.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
