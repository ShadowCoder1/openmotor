"""Polsterer et al. (OSF project bjn95, publication not located): adaptation of speech to a shifted first formant (F1 raised in the
feedback) across adulthood, 66 adults aged 16 to 75 (a 67th, who failed the hearing screening, is in the raw file only).

Raw files, raw_extra/Polsterer_ageAdaptation_OSF/ (read as text and data only):
  data_raw_shared.xlsx       one row per trial: participant, f1_40120 (produced F1 in Hz), phase, sex, age, trial (1-108)
  dat_shared.Rdata           the authors' analysis table: participant, trial, f1_norm (F1 change from baseline in %), phase, stimulus, ...
  datRM_shared.Rdata         per participant phase means of f1_norm (hold mean = response magnitude)
  participants_shared.xlsx   age, sex, language and music background, hearing screening
  analysis.Rmd, AuditoryMotorAdaptationAcrossAge_public.html   the authors' analysis script and its rendered output (printed numbers)
Run from the site folder: python3 -I tools/std/convert/Polsterer_ageAdaptation_OSF.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Polsterer_ageAdaptation_OSF"
SRC = "the deposit's rendered analysis report AuditoryMotorAdaptationAcrossAge_public.html"
PHASE = {"baseline": ("baseline", "baseline"), "ramp": ("ramp", "learning"), "hold": ("hold", "learning"),
         "after": ("after-effect", "aftereffect"), "after-effect": ("after-effect", "aftereffect")}


def build():
    raw = stdlib.raw_dir(ID)
    par = pd.read_excel(raw / "participants_shared.xlsx", sheet_name=0)
    par.columns = [c.strip() for c in par.columns]
    par["subjectID"] = par.subjectID.str.strip()
    par["hearing"] = par.hearing.str.strip()
    r = pd.read_excel(raw / "data_raw_shared.xlsx")
    dat = pyreadr.read_r(str(raw / "dat_shared.Rdata"))["dat"]
    rm = pyreadr.read_r(str(raw / "datRM_shared.Rdata"))["datRM"]
    dat["participant"] = dat.participant.astype(str)
    dat["trial"] = dat.trial.astype(int)
    rm["participant"] = rm.participant.astype(str)
    outliers = set(rm.loc[rm["hold"] >= 16, "participant"])  # the authors' model criticism: hold < 16 kept (Rmd)
    m = r.rename(columns={"subjectID": "participant"}).merge(
        dat[["participant", "trial", "f1_norm", "stimulus", "phase"]].rename(columns={"phase": "phase_dat"}),
        on=["participant", "trial"], how="outer")
    m["stimulus"] = m.stimulus.astype(object)
    m["phase"] = m.phase.fillna(m.phase_dat)
    only_dat = m.f1_40120.isna()
    assert only_dat.sum() == 1
    pord = {p: i for i, p in enumerate(par.subjectID)}
    m["o"] = m.participant.map(pord)
    m = m.sort_values(["o", "trial"], kind="stable")
    pinfo = par.set_index("subjectID")
    rows = []
    for t in m.itertuples(index=False):
        p = pinfo.loc[t.participant]
        blk, std = PHASE[t.phase]
        altered = blk in ("ramp", "hold")
        failed = p.hearing != "PASS"
        no_norm = pd.isna(t.f1_norm)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "F1 perturbation across age", "group": "",
            "subject": t.participant, "subject_excluded_by_authors": bool(failed or t.participant in outliers),
            "age": int(p.age), "sex": p.sex, "handedness": "", "population": "healthy", "setting": "lab", "device": "",
            "session": 1, "block": blk, "trial": int(t.trial), "phase": std, "time_s": "", "task": "speech", "unit": "utterance",
            "condition": "" if pd.isna(t.stimulus) else str(t.stimulus),
            "pert_type": "formant_shift" if altered else "none", "pert_value": "" if altered else 0.0, "pert_unit": "",
            "feedback": "altered" if altered else "normal", "excluded_by_authors": bool(failed or no_norm),
            "stimulus": "" if pd.isna(t.stimulus) else str(t.stimulus), "segment": "vowel",
            "measure_window": "window named 40-120 in the raw column f1_40120 (units not stated in the deposit)",
            "f1_hz": t.f1_40120, "f1_change_pct": t.f1_norm,
        })
    a = pd.DataFrame(rows)
    parts = []
    for p in par.itertuples(index=False):
        parts.append({"person": p.subjectID, "age_group": p.age_group, "multiple_L1": p.multiple_L1, "dialect": p.dialect, "L2": p.L2,
                      "instruments": p.instruments, "singing": p.singing, "hearing_screen": p.hearing})
    return a, parts, outliers, rm


def main():
    checks = stdlib.Checks()
    a, parts, outliers, rm = build()
    hearing_ok = a[~a.subject.isin([p["person"] for p in parts if p["hearing_screen"] != "PASS"])]
    n_pass = hearing_ok.subject.nunique()
    withnorm = hearing_ok[hearing_ok.f1_change_pct.notna()]
    checks.published("participants who passed the hearing screening", n_pass, 66, 0, SRC + ", length(unique(par$participant))")
    checks.published("trials with an F1 change value (n of the model m0)", len(withnorm), 7087, 0, SRC + ", summary(m0): n = 7087")
    ages = hearing_ok.groupby("subject").age.first()
    checks.published("youngest age", ages.min(), 16, 0, SRC + ", range(dat$age)")
    checks.published("oldest age", ages.max(), 75, 0, SRC + ", range(dat$age)")
    checks.published("distinct ages", ages.nunique(), 43, 0, SRC + ", length(unique(dat$age))")
    sx = hearing_ok.groupby("subject").sex.first().value_counts()
    checks.published("female participants", sx["F"], 38, 0, SRC + ", table(par$sex)")
    checks.published("male participants", sx["M"], 28, 0, SRC + ", table(par$sex)")
    hold = withnorm[withnorm.block == "hold"].groupby("subject").f1_change_pct.mean()
    checks.published("mean of the participants' hold-phase F1 change (%)", hold.mean(), 1.3, 0.05, SRC + ", describe(datRM$hold)")
    checks.published("SD of the hold-phase means", hold.std(), 7.72, 0.005, SRC + ", describe(datRM$hold)")
    checks.published("median of the hold-phase means", hold.median(), 0.29, 0.005, SRC + ", describe(datRM$hold)")
    checks.published("minimum hold-phase mean", hold.min(), -12.94, 0.005, SRC + ", describe(datRM$hold)")
    checks.published("maximum hold-phase mean", hold.max(), 29.49, 0.005, SRC + ", describe(datRM$hold)")
    checks.published("participants removed as hold-phase outliers (hold mean 16 or more)", len(outliers), 3, 0,
                     "the deposit's analysis script analysis.Rmd, model criticism: 'three outliers above 2 sd'")
    checks.published("trials in a full session (24 baseline + 30 ramp + 30 hold + 24 after-effect)", a.trial.max(), 108, 0,
                     "the deposit's analysis script analysis.Rmd, section Validate response types")
    checks.internal("phases cover trials 1-24, 25-54, 55-84, 85-108",
                    bool(a.groupby("block").trial.agg(["min", "max"]).loc[["baseline", "ramp", "hold", "after-effect"]].values.tolist()
                         == [[1, 24], [25, 54], [55, 84], [85, 108]]))
    checks.internal("67 participants and 7222 rows (7221 raw trials plus one trial only in the analysis table)",
                    a.subject.nunique() == 67 and len(a) == 7222)
    checks.internal("raw F1 between 250 and 900 Hz", bool(a.f1_hz.dropna().between(250, 900).all()))
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    paper["authors"] = "Polsterer K, et al."
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the authors' raw F1 table (Hz) and their analysis table (F1 change in %, word) on participant and trial, kept the "
                 "participant who failed the hearing screening (flagged), added age and sex from the participant table and replaced "
                 "participant codes. The language and music questionnaire answers are in participants.csv."),
        how=[
            "One row is one repetition of one of three nonwords (BEVE, DEVE, VEDE) in the authors' four phases: baseline (trials 1-24), "
            "ramp (25-54), hold (55-84) and after-effect (85-108). f1_hz is the raw produced F1 (the authors' column f1_40120; the "
            "deposit does not say what 40120 means, probably the 40-120 ms window of the vowel, so measure_window says only that) and "
            "f1_change_pct is the authors' F1 change from the participant's baseline in percent (their f1_norm).",
            "The deposit says that F1 was raised in the feedback in the ramp and hold phases but gives neither the size of the shift nor "
            "the ramp; pert_value is empty. The after-effect phase is taken to have normal feedback (the usual design; inferred, not stated).",
            "Rows are excluded_by_authors when they have no f1_change_pct in the authors' analysis table (their script drops rows "
            "without it) or belong to the participant who failed the hearing screening. subject_excluded_by_authors also marks the three "
            "participants whose hold-phase mean was 16 % or more, whom the authors removed as outliers from their final models.",
            "The deposit gives age per participant only to the year; trial numbers are the authors' (1-108) and have gaps where trials "
            "are missing.",
        ],
        unit_note="repetition of a nonword by one participant",
        differences=[
            "The deposit's analysis script says 79 participants and ages 16-82, but the shared files and the script's own rendered output "
            "have 66 participants aged 16-75 (a 67th failed the hearing screening); the shared data are a subset, and the numbers "
            "checked here are those printed in the rendered output.",
            "27 trials have raw F1 but no F1 change value in the authors' analysis table; they are marked excluded_by_authors, and one "
            "trial present only in the analysis table has no raw F1.",
        ],
    )
    pc = [("age_group", "category", "<50, >50", "Age group the authors assigned (under or over 50 years)."),
          ("multiple_L1", "category", "N, Y", "Reported more than one first language."),
          ("dialect", "category", "N, Y", "Reported speaking a dialect."),
          ("L2", "category", "N, Y", "Reported knowing a second language."),
          ("instruments", "category", "N, Y", "Plays a musical instrument."),
          ("singing", "category", "N, Y", "Sings."),
          ("hearing_screen", "category", "PASS, FAIL", "Result of the hearing screening.")]
    return stdlib.release(ID, "speech", a.to_dict("records"), meta, checks, participants=parts, participant_columns=pc)


if __name__ == "__main__":
    main()
