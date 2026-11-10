"""Eads et al. (2026), Folia Phoniatr Logop: response to an upward F1 shift in 46 children aged 9-15, 20 with residual
speech sound disorder (RSSD, rhotic distortions) and 26 without, plus a discrimination (JND) measure of auditory acuity.

Raw files, raw_extra/Eads2026_FoliaPhoniatr/ (OSF yc5vm):
  AAF_Trial_Response_Data.csv   subID, response (F1, Hz), trial_num (1-72); 46 children x 72 trials; 359 trials have no value
  AudAcuity_AAF_Data.csv        Study_ID, Age, Sex, Group, DISCRIM_JND (steps of a 240-step head-had continuum),
                                Adaptation_during_pert_150, Adaptation_after_pert_150, n_missing_trials
  OSF_Code.Rmd                  the authors' code: Baseline = trials 1-18, Hold = 19-54, After-effect = 55-72; the response is F1
                                from a 150-350 ms window; adaptation is the mean of the first three After-effect trials (55-57)
                                relative to the mean Baseline F1; group is MOSAIC (No RSSD) or RCT (RSSD) in subID
The abstract (Karger, 10.1159/fpl/acvag022) gives the 30% upward F1 shift, N = 46, 20 RSSD (5 female, 15 male) and
26 without (13 female, 13 male), ages 9-15. The full text is behind a paywall.
Run from the site folder: python3 -I tools/std/convert/Eads2026_FoliaPhoniatr.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Eads2026_FoliaPhoniatr"
SRC = "Eads et al. 2026, abstract (Folia Phoniatr Logop, doi 10.1159/fpl/acvag022)"
BLOCK = [(18, "Baseline", "baseline"), (54, "Hold", "learning"), (72, "After-effect", "aftereffect")]


def phase_of(t):
    for end, b, p in BLOCK:
        if t <= end:
            return b, p


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    t = pd.read_csv(raw / "AAF_Trial_Response_Data.csv")
    s = pd.read_csv(raw / "AudAcuity_AAF_Data.csv")
    t["sid"] = t.subID.str.replace(r"\D", "", regex=True).astype(int)
    s = s.set_index("Study_ID")
    rows = []
    for r in t.itertuples(index=False):
        c = s.loc[r.sid]
        b, p = phase_of(r.trial_num)
        hold = b == "Hold"
        miss = pd.isna(r.response)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Altered auditory feedback", "group": c.Group, "subject": f"C{r.sid}",
            "subject_excluded_by_authors": False, "age": float(c.Age), "sex": "F" if c.Sex == "Female" else "M", "handedness": "",
            "population": "healthy" if c.Group == "No RSSD" else "residual speech sound disorder (rhotic distortions)",
            "setting": "lab", "device": "", "session": 1, "block": b, "trial": int(r.trial_num), "phase": p, "time_s": "",
            "task": "speech", "unit": "utterance", "condition": "", "pert_type": "formant_shift" if hold else "none",
            "pert_value": 30.0 if hold else 0.0, "pert_unit": "percent of F1", "feedback": "altered" if hold else "normal",
            "excluded_by_authors": bool(miss), "stimulus": "", "segment": "vowel (head-had contrast)",
            "measure_window": "150-350 ms into the trial", "duration_ms": "", "f0_hz": "", "f1_hz": "" if miss else float(r.response),
            "f2_hz": "", "f1_mel": "", "intensity_au": "", "shift_f1": 30.0 if hold else 0.0, "shift_f2": 0.0,
            "masking_noise": "", "feedback_latency_ms": "",
        })
    a = pd.DataFrame(rows)
    sub = a.drop_duplicates("subject")
    # participant table
    pr = []
    for sid, c in s.iterrows():
        pr.append({"person": f"C{sid}", "age_years": float(c.Age), "sex": "F" if c.Sex == "Female" else "M", "group": c.Group,
                   "discrimination_jnd_steps": float(c.DISCRIM_JND),
                   "adaptation_during_pert_150": float(c.Adaptation_during_pert_150),
                   "adaptation_after_pert_150": float(c.Adaptation_after_pert_150),
                   "n_missing_trials": int(c.n_missing_trials)})
    pcols = [("age_years", "number", "years", "Age at testing."),
             ("sex", "category", "F, M", "Sex as in the deposit."),
             ("group", "text", "", "No RSSD or RSSD (residual speech sound disorder, rhotic distortions)."),
             ("discrimination_jnd_steps", "number", "steps of a 240-step head-had continuum",
              "Just noticeable difference in the discrimination task; smaller means more acute perception."),
             ("adaptation_during_pert_150", "number", "authors' units (as in the deposit)",
              "The authors' file column Adaptation_during_pert_150: response during altered feedback, 150 ms window; its exact unit is not documented."),
             ("adaptation_after_pert_150", "number", "authors' units (as in the deposit)",
              "The authors' file column Adaptation_after_pert_150: response after the shift was removed; its exact unit is not documented."),
             ("n_missing_trials", "integer", "trials", "Number of the 72 trials without an F1 value.")]
    checks.internal("72 trials for each of 46 children", a.groupby("subject").size().eq(72).all() and a.subject.nunique() == 46)
    checks.internal("missing F1 trials equal the authors' n_missing_trials per child",
                    bool((a[a.excluded_by_authors].groupby("subject").size().reindex(sub.subject).fillna(0).values
                          == [s.loc[int(x[1:]), "n_missing_trials"] for x in sub.subject]).all()))
    checks.internal("subID prefix (MOSAIC/RCT) agrees with Group in the acuity file",
                    bool((t.drop_duplicates("sid").subID.str.contains("MOSAIC").values
                          == (s.loc[t.drop_duplicates("sid").sid, "Group"] == "No RSSD").values).all()))
    checks.published("children analysed", sub.shape[0], 46, 0, SRC)
    checks.published("children without RSSD", (sub.group == "No RSSD").sum(), 26, 0, SRC)
    checks.published("children with RSSD", (sub.group == "RSSD").sum(), 20, 0, SRC)
    for g, f, m in (("RSSD", "F", 5), ("RSSD", "M", 15), ("No RSSD", "F", 13), ("No RSSD", "M", 13)):
        checks.published(f"{g}: number of {'female' if f == 'F' else 'male'} children", ((sub.group == g) & (sub.sex == f)).sum(), m, 0, SRC)
    checks.published("youngest child, whole years (ages 9-15)", math.floor(sub.age.min()), 9, 0, SRC)
    checks.published("oldest child, whole years (ages 9-15)", math.floor(sub.age.max()), 15, 0, SRC)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    st = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(st["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    paper["venue"] = "Folia Phoniatr Logop"
    meta = dict(
        study=ID, paper=paper, lab=st["lab"], data_url=st["data_url"], license="none stated", license_url="",
        changes=("Added the phase labels and the feedback shift from the authors' analysis code and the abstract, marked trials "
                 "without an F1 value, joined each child's age and sex, and replaced participant identifiers with codes."),
        how=[
            "One row is one of the 72 trials of one child; f1_hz is the authors' F1 for that trial (Hz, from a 150-350 ms window "
            "of the trial); trials without a value are kept and marked excluded_by_authors.",
            "Trials 1-18 are Baseline, 19-54 Hold (F1 heard 30% higher: pert_value 30, unit percent of F1) and 55-72 After-effect "
            "(feedback normal); the authors define adaptation from the first three After-effect trials.",
            "participants.csv holds, per child, age, sex, group, the auditory discrimination JND (steps of a 240-step head-had "
            "continuum) and the two adaptation columns of the authors' file.",
        ],
        unit_note="trial of one child",
        differences=["The full text is paywalled, so only the abstract's counts (children per group, sex, age range) could be checked; "
                     "the stimulus word, the ramp (if any) of the shift and the units of the two adaptation columns are not documented, "
                     "and the main-table shift is therefore assumed to start at trial 19 as the authors' phase code says."],
    )
    return stdlib.release(ID, "speech", rows, meta, checks, participants=pr, participant_columns=pcols)


if __name__ == "__main__":
    main()
