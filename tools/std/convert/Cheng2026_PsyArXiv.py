"""Cheng, Murphy, Parrell & Niziolek (2026), PsyArXiv: predictors of adaptation to a non-uniform formant perturbation
and the construct validity of sensory acuity assays. Standardized here: the bite block speech task, the only motor task
the authors shared trial by trial.

Raw files, raw_extra/Cheng2026_PsyArXiv/ (OSF ew7ps, data/):
  vsaGen2biteblockAdaptTable_2025-04-04.csv, vsaRetentionbiteblockAdaptTable_2026-02-12.csv,
  vsaRetentionbiteblockAdaptTable_controls_2026-02-13.csv   subj, trialnum, vowel (iy, ae, uw, aa), condition (baseline,
      vert = large bite block, horiz = small bite block), F1, F2 (mels, mean of the middle 50% of the vowel), vowel_dur (s);
      discarded trials have empty values. The three files are the study's cohorts (bound together in predict_adaptation.Rmd,
      which renames sp891redo to sp891).
  subject_age.csv   subj, age, age_group, gender for all 78 participants.
Not standardized: the adaptation task was shared only as per-participant summaries (vsaPredict_production_superTable),
and the vowel, pure-tone and tactile JND staircases and BrainGauge tables are perception tasks.
Paper (Methods, Bite block task): 8 baseline repetitions of heed, had, who'd, hod without a bite block, then 16 repetitions
of had and hod with the small bite block (2 mm jaw aperture) and of heed and who'd with the large one (17.5 mm), order of
the bite blocks counterbalanced, under speech-shaped masking noise scaled by speech amplitude (~80 dB SPL).
Run from the site folder: python3 -I tools/std/convert/Cheng2026_PsyArXiv.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cheng2026_PsyArXiv"
SRC = "Cheng, Murphy, Parrell & Niziolek 2026, PsyArXiv pzse8 v1"
FILES = {"vsaGen2biteblockAdaptTable_2025-04-04.csv": "vsaGen2 cohort",
         "vsaRetentionbiteblockAdaptTable_2026-02-12.csv": "vsaRetention cohort",
         "vsaRetentionbiteblockAdaptTable_controls_2026-02-13.csv": "vsaRetention controls"}
WORD = {"iy": ("heed", "vowel /i/"), "ae": ("had", "vowel /æ/"), "uw": ("who'd", "vowel /u/"), "aa": ("hod", "vowel /ɑ/")}
BITE = {"baseline": ("none", 0.0, "baseline"), "horiz": ("bite_block", 2.0, "test"), "vert": ("bite_block", 17.5, "test")}
COND = {"baseline": "no bite block", "horiz": "small bite block (2 mm jaw aperture)",
        "vert": "large bite block (17.5 mm jaw aperture)"}
NOISE = "speech-shaped noise scaled in real time by the speech amplitude, about 80 dB SPL at comfortable loudness"


def sex(g):
    g = str(g).strip().lower()
    return "F" if g.startswith("f") else "M" if g.startswith("m") else "other"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    B = pd.concat([pd.read_csv(raw / f).assign(cohort=c) for f, c in FILES.items()], ignore_index=True)
    ages = pd.read_csv(raw / "subject_age.csv")
    ages["subj"] = ages.subj.astype(str)
    B["person"] = B.subj.str.replace("redo", "", regex=False)
    checks.internal("75 participants with 96 trials each (32 baseline, 64 bite block), trials 1-96 in order",
                    B.person.nunique() == 75 and (B.groupby("subj").size() == 96).all()
                    and B.groupby("subj", sort=False).trialnum.apply(lambda s: list(s) == list(range(1, 97))).all())
    ct = pd.crosstab(B.condition, B.vowel)
    checks.internal("8 baseline repetitions of each vowel; 16 of /æ/ and /ɑ/ with the small and of /i/ and /u/ with the large "
                    "bite block", bool((ct.loc["baseline"] == 600).all() and ct.loc["horiz", ["aa", "ae"]].eq(1200).all()
                                       and ct.loc["vert", ["iy", "uw"]].eq(1200).all()
                                       and ct.loc["horiz", ["iy", "uw"]].eq(0).all() and ct.loc["vert", ["aa", "ae"]].eq(0).all()))
    checks.internal("baseline on trials 1-32, bite blocks on 33-96",
                    bool((B.trialnum[B.condition == "baseline"] <= 32).all() and (B.trialnum[B.condition != "baseline"] > 32).all()))
    checks.internal("F1, F2 and duration missing together (discarded trials)",
                    bool((B.F1.isna() == B.F2.isna()).all() and (B.F1.isna() == B.vowel_dur.isna()).all()))
    for c, (lo, hi) in dict(F1=(250, 1300), F2=(600, 2100), vowel_dur=(0.03, 1.5)).items():
        v = B[c].dropna()
        checks.internal(f"{c} within {lo}-{hi}", bool(v.between(lo, hi).all()), f"{v.min():.3f}-{v.max():.3f}")
    checks.internal("every participant has an age in subject_age.csv", set(B.person) <= set(ages.subj))
    g = ages.gender.map(sex).value_counts().to_dict()
    checks.internal("the age table describes the paper's 78 participants (62 F, 14 M, 2 non-binary; mean age 38.88, SD 19.49, "
                    "18-74)", len(ages) == 78 and g == {"F": 62, "M": 14, "other": 2} and round(ages.age.mean(), 2) == 38.88
                    and round(ages.age.std(ddof=1), 2) == 19.49 and ages.age.min() == 18 and ages.age.max() == 74, str(g))
    A = ages.set_index("subj")
    rows = []
    for r in B.itertuples(index=False):
        pt, pv, phase = BITE[r.condition]
        word, seg = WORD[r.vowel]
        miss = pd.isna(r.F1)
        rows.append({
            "dataset": ID, "study": ID, "experiment": r.cohort, "group": "", "subject": r.subj, "person": r.person,
            "subject_excluded_by_authors": False, "age": int(A.age[r.person]), "sex": sex(A.gender[r.person]),
            "handedness": "", "population": "healthy", "setting": "lab",
            "device": "Sennheiser MKE 600 tabletop microphone, Beyerdynamic DT 770 PRO headphones, tongue-depressor bite block",
            "session": 1, "block": r.condition, "trial": int(r.trialnum), "phase": phase, "time_s": "", "task": "speech",
            "unit": "utterance", "condition": COND[r.condition], "pert_type": pt, "pert_value": pv,
            "pert_unit": "mm jaw aperture" if pv else "", "feedback": "masked", "excluded_by_authors": bool(miss),
            "stimulus": word, "segment": seg, "measure_window": "middle 50% of the vowel",
            "duration_ms": "" if miss else float(r.vowel_dur) * 1000, "f0_hz": "", "f1_hz": "", "f2_hz": "",
            "f1_mel": "" if miss else float(r.F1), "f2_mel": "" if miss else float(r.F2), "intensity_au": "",
            "intensity_max_au": "", "f1_heard_hz": "", "f2_heard_hz": "", "shift_f1": "", "shift_f2": "",
            "masking_noise": NOISE, "feedback_latency_ms": "",
        })
    a = pd.DataFrame(rows)
    pct = 100 * a.groupby("person").excluded_by_authors.mean()
    checks.published("bite block task: % of trials discarded, all participants", 100 * a.excluded_by_authors.mean(), 2.5, 0.05,
                     f"{SRC}, Methods, Acoustic analysis of bite block task")
    checks.published("bite block task: % of trials discarded, highest participant", pct.max(), 16, 0.5,
                     f"{SRC}, Methods, Acoustic analysis of bite block task")
    checks.published("bite block task: % of trials discarded, lowest participant", pct.min(), 0, 0,
                     f"{SRC}, Methods, Acoustic analysis of bite block task")
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["venue"] = p["venue"] or "PsyArXiv preprint"
    meta = dict(
        study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="none stated", license_url="",
        changes=("Joined the authors' three bite block tables (one per cohort), added age and sex from their age table, "
                 "converted vowel duration to ms and replaced participant identifiers with codes."),
        how=[
            "One row is one word of the bite block speech task, in the authors' trial order: trials 1-32 are 8 repetitions "
            "each of heed, had, who'd and hod without a bite block, trials 33-96 16 repetitions each of had and hod with a "
            "small bite block (2 mm jaw aperture, pert_value) and of heed and who'd with a large one (17.5 mm), in "
            "counterbalanced order, all spoken under masking noise.",
            "f1_mel and f2_mel are the authors' F1 and F2 averaged over the middle 50% of the vowel, in mels; trials they "
            "discarded have empty values and are marked excluded_by_authors. experiment names the cohort table a "
            "participant came from.",
            "The study's sensorimotor adaptation task was shared only as per-participant summaries, and its vowel, "
            "pure-tone and tactile acuity tasks are perception tasks; neither is included here.",
        ],
        unit_note="word spoken in the bite block task (utterance)",
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
