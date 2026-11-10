"""Nault, Mitsuya, Purcell & Munhall (2022), Front Hum Neurosci 16:905365: perturbing the consistency of auditory feedback in
speech (Experiment 1: three conditions, 18 speakers; Experiment 2: five conditions, 20 speakers; the word 'head').

Raw file, raw_extra/Nault2022_FrontHumNeurosci/raw.xlsx (OSF n4pgf, 'Experiment1_Experiment2_Raw_Data_2022.06.26.xlsx'):
  sheets 'Experiment 1' and 'Experiment 2', one row per utterance:
  Speaker ID, Condition, Normalized F1, Normalized F2 (Hz, minus the speaker's baseline mean), Trial, Phase,
  F1 Perturbation, F2 Perturbation (Hz, the shift applied to the feedback; F2 empty in the F1-only condition).
What the values are (paper, Methods; read as text): formants from the middle 40-80% of the vowel, Burg tracking; trials 1-5 of
each condition were removed by the authors (the file's Trial 1 is their trial 6); 30 F1 and 43 F2 values (Exp 1) and 62 F1 and
191 F2 values (Exp 2) were omitted for tracking problems and are empty in the file.
Run from the site folder: python3 -I tools/std/convert/Nault2022_FrontHumNeurosci.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Nault2022_FrontHumNeurosci"
PAPER_SRC = "Nault et al. 2022"
PHASE = {"Baseline": "baseline", "Perturbation": "learning", "Return": "washout"}
COND = {"F1/F2 Random Perturbation": "F1/F2 Random", "F1-Only Random Perturbation": "F1-Only Random",
        "F1/F2 Coupled Random Perturbation": "F1/F2 Coupled"}
DEVICE = "headset microphone, real-time formant filter (IIR), Sennheiser HD 265 headphones"
MASKING = "speech-shaped noise, 50 dBA SPL"


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def build(checks):
    raw = stdlib.raw_dir(ID)
    rows, info = [], {}
    for e in (1, 2):
        d = pd.read_excel(raw / "raw.xlsx", sheet_name=f"Experiment {e}")
        d.columns = [c.strip() for c in d.columns]
        info[e] = d
        for r in d.itertuples(index=False):
            cond = COND.get(r.Condition, r.Condition)
            f1p, f2p = float(r[6]), r[7]
            f2p = 0.0 if pd.isna(f2p) else float(f2p)
            shifted = f1p != 0 or f2p != 0
            f1, f2 = r[2], r[3]
            person = f"E{e}_S{int(r[0])}"
            rows.append({
                "dataset": ID, "study": ID, "experiment": f"Experiment {e}", "group": "",
                "subject": f"{person}|{cond}", "person": person, "subject_excluded_by_authors": False,
                "age": "", "sex": "F", "handedness": "", "population": "healthy", "setting": "lab", "device": DEVICE,
                "session": 1, "block": r.Phase, "trial": int(r[4]), "phase": PHASE[r.Phase], "time_s": "",
                "task": "speech", "unit": "utterance", "condition": cond,
                "pert_type": "formant_shift" if shifted else "none", "pert_value": f1p, "pert_unit": "Hz",
                "feedback": "altered" if shifted else "normal",
                "excluded_by_authors": bool(pd.isna(f1) and pd.isna(f2)),
                "stimulus": "head", "segment": "vowel /ɛ/", "measure_window": "middle 40-80% of the vowel duration",
                "f1_change_hz": "" if pd.isna(f1) else float(f1), "f2_change_hz": "" if pd.isna(f2) else float(f2),
                "shift_f1": f1p, "shift_f2": f2p, "masking_noise": MASKING,
                "authors_trial_number": int(r[4]) + 5,
            })
    return rows, info


def main():
    checks = stdlib.Checks()
    rows, info = build(checks)
    a = pd.DataFrame(rows)
    a["exp"] = a.experiment.str[-1].astype(int)
    n1, n2 = info[1], info[2]
    checks.internal("Experiment 1 phases are trials 1-15, 16-75, 76-135 of the file (original trials 6-20, 21-80, 81-140)",
                    bool((n1.groupby("Phase")["Trial"].agg(["min", "max"]).loc[["Baseline", "Perturbation", "Return"]].values
                          == np.array([[1, 15], [16, 75], [76, 135]])).all()))
    checks.internal("Experiment 2 phases are trials 1-15, 16-45, 46-75 of the file (original 6-20, 21-50, 51-80)",
                    bool((n2.groupby("Phase")["Trial"].agg(["min", "max"]).loc[["Baseline", "Perturbation", "Return"]].values
                          == np.array([[1, 15], [16, 45], [46, 75]])).all()))
    checks.internal("every sequence is unique by trial and rows kept in the authors' order",
                    not a.duplicated(["subject", "exp", "trial"]).any())
    checks.internal("shift 0 in Baseline rows", bool((a.pert_value[a.block == "Baseline"] == 0).all()
                                                    and (a.shift_f2[a.block == "Baseline"] == 0).all()))
    checks.internal("Experiment 1: three conditions per speaker, 135 trials each", bool(
        (a[a.exp == 1].groupby("person").subject.nunique() == 3).all()
        and (a[a.exp == 1].groupby("subject").size() == 135).all()))
    checks.internal("Experiment 2: five conditions per speaker, 75 trials each", bool(
        (a[a.exp == 2].groupby("person").subject.nunique() == 5).all()
        and (a[a.exp == 2].groupby("subject").size() == 75).all()))
    # published counts (Methods, Participants; Results)
    src = f"{PAPER_SRC}, Methods"
    checks.published("Experiment 1: number of speakers", a[a.exp == 1].person.nunique(), 18, 0, f"{src}, Experiment 1, Participants")
    checks.published("Experiment 2: number of speakers analysed (22 recruited, 2 removed)", a[a.exp == 2].person.nunique(), 20, 0,
                     f"{src}, Experiment 2, Participants")
    checks.published("Experiment 1: number of utterances", int((a.exp == 1).sum()), 7290, 0, f"{src}, Experiment 1, Procedure")
    checks.published("Experiment 2: number of utterances", int((a.exp == 2).sum()), 7500, 0, f"{src}, Experiment 2, Procedure")
    for e, (r1, r2) in ((1, (30, 43)), (2, (62, 191))):
        x = a[a.exp == e]
        checks.published(f"Experiment {e}: F1 values omitted for tracking problems", int((x.f1_change_hz == "").sum()), r1, 0,
                         f"{PAPER_SRC}, Methods, Experiment {e}, Acoustic analysis")
        checks.published(f"Experiment {e}: F2 values omitted for tracking problems", int((x.f2_change_hz == "").sum()), r2, 0,
                         f"{PAPER_SRC}, Methods, Experiment {e}, Acoustic analysis")
    # model N (Results, Tables 1 to 4): F1 observations and F2 observations
    for e, (n_f1, n_f2) in ((1, (7260, 7247)), (2, (7438, 7309))):
        x = a[a.exp == e]
        checks.published(f"Experiment {e}: F1 observations in the mixed model", int((x.f1_change_hz != "").sum()), n_f1, 0,
                         f"{PAPER_SRC}, Results, Experiment {e}, Table {1 if e == 1 else 3}")
        checks.published(f"Experiment {e}: F2 observations in the mixed model", int((x.f2_change_hz != "").sum()), n_f2, 0,
                         f"{PAPER_SRC}, Results, Experiment {e}, Table {2 if e == 1 else 4}")
    # within-speaker variability of F1 (Results, Variability)
    def sd_by_phase(d):
        g = d.groupby(["Speaker ID", "Condition", "Phase"])["Normalized F1"].std(ddof=1)
        return g.groupby("Phase").mean()
    for lab, d, rep in (("Experiment 1", n1, dict(Baseline=20.72, Perturbation=24.05, Return=24.47)),
                        ("Experiment 2", n2, dict(Baseline=28.47, Perturbation=31.64, Return=29.75))):
        sd = sd_by_phase(d)
        for ph, r in rep.items():
            checks.published(f"{lab}: mean within-speaker SD of F1, {ph} (Hz)", sd[ph], r, 0.005,
                             f"{PAPER_SRC}, Results, {lab}, Variability")
    sdc = sd_by_phase(n2[n2.Condition == "Control"])
    for ph, r in dict(Baseline=30.42, Perturbation=29.16, Return=28.65).items():
        checks.published(f"Experiment 2: mean within-speaker SD of F1, Control condition, {ph} (Hz)", sdc[ph], r, 0.005,
                         f"{PAPER_SRC}, Results, Experiment 2, Variability")
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Stacked the authors' two sheets (Experiments 1 and 2) into one table, wrote the empty F2 shift of the F1-only "
                 "condition as 0, and replaced speaker numbers with codes."),
        how=[
            "One row is one utterance of the word 'head' by one female speaker, in the authors' order. Each speaker did every "
            "condition of their experiment, so each speaker x condition is a separate sequence (subject) and the speaker is the person.",
            "f1_change_hz and f2_change_hz are the authors' normalized formants in Hz: the production minus the speaker's baseline mean "
            "(from the middle 40-80% of the vowel). The absolute formants are not shared. Empty cells are values the authors omitted for "
            "formant-tracking problems.",
            "shift_f1 and shift_f2 (Hz) are the perturbations of the feedback on that trial as the authors list them; pert_value repeats "
            "shift_f1. Positive F1 raises the vowel towards 'had'. In the F1-only condition F2 was not shifted (0 here, empty in the file).",
            "trial is the authors' trial number in the file; the authors removed the first 5 trials of every condition, so "
            "authors_trial_number (trial + 5) is the position in the full 140-trial or 80-trial run.",
        ],
        unit_note="spoken word (utterance) of one speaker",
        differences=[],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
