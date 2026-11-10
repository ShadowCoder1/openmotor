"""Hilger, Cole & Larson (2023), Research Square preprint: pitch-shift reflex in people with cerebellar ataxia and matched
controls, sentence-production task (+/-200 cent pitch perturbation 50 ms into a stressed word, new vs corrective focus).

Raw file, raw_extra/Hilger2023_ResSquare/ (OSF cu5wy): Hilger_Cole_Larson_Data.csv, one row per participant x perturbation direction x
sentence focus x response direction (opposing or following): Abs_Mag (cents), Latency (s), Magnitude (signed cents).
Paper (preprint PDF on the same OSF project, read as text): Methods, Data analysis; Results.
Run from the site folder: python3 -I tools/std/convert/Hilger2023_ResSquare.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hilger2023_ResSquare"
P = "Hilger, Cole & Larson 2023 preprint, "


def build():
    d = pd.read_csv(stdlib.raw_dir(ID) / "Hilger_Cole_Larson_Data.csv")
    d["subj"] = d.subj.str.strip("_ ").astype("object")
    rows = []
    n = {}
    for r in d.itertuples(index=False):
        sub = r.subj
        pv = 200.0 if r.Perturbation == "UP" else -200.0
        base = {
            "dataset": ID, "study": ID, "experiment": "Sentence production (stressed-word focus)", "group": "",
            "subject": sub, "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
            "population": "cerebellar ataxia" if r.Group == "Ataxia" else "healthy", "setting": "lab",
            "device": "Etymotic ER2-14A insert earphones, over-ear microphone, real-time pitch shifter", "session": 1,
            "block": "", "phase": "test", "time_s": "", "task": "speech", "unit": "participant summary",
            "condition": f"{r.Focus} focus, {r.Perturbation.lower()} shift, {r.Response_direction.lower()} response",
            "pert_type": "pitch_shift", "pert_value": pv, "pert_unit": "cents", "feedback": "altered",
            "excluded_by_authors": False, "statistic": "mean", "n_trials": "", "session_context": "",
        }
        for meas, desc, val, unit in (
            ("pitch_shift_response_signed", "Mean change of voice f0 after the perturbation relative to the 50 ms before it (400 ms window, "
             "cents); positive when f0 rose, so an opposing response to an upward shift is negative", r.Magnitude, "cents"),
            ("pitch_shift_response_magnitude", "Absolute size of the same change (the authors' Abs_Mag), cents", r.Abs_Mag, "cents"),
            ("pitch_shift_peak_latency", "Time from perturbation onset to the peak of the response, seconds (the paper reports ms)", r.Latency, "s"),
        ):
            n[sub] = n.get(sub, 0) + 1
            rows.append({**base, "trial": n[sub], "measure": meas, "measure_description": desc, "value": float(val), "value_unit": unit})
    return rows, d


def main():
    checks = stdlib.Checks()
    rows, d = build()
    k = d.groupby("Focus").Abs_Mag.mean()
    checks.published("participants (group df F(1, 47) plus the two groups)", d.subj.nunique(), 49, 0, P + "Results, response magnitude by task and group (F(1, 47))")
    checks.published("mean absolute response magnitude, corrective focus (cents)", k["Corrective"], 136.71, 0.01, P + "Results, section c")
    checks.published("mean absolute response magnitude, new information focus (cents)", k["New"], 102.81, 0.01, P + "Results, section c")
    g = d.groupby(["Group", "Focus"]).Abs_Mag.mean()
    checks.published("ataxia, corrective focus (cents)", g["Ataxia", "Corrective"], 146.98, 0.01, P + "Results, section c")
    checks.published("ataxia, new focus (cents)", g["Ataxia", "New"], 115.89, 0.01, P + "Results, section c")
    checks.published("control, new focus (cents)", g["Control", "New"], 93.12, 0.01, P + "Results, section c")
    checks.internal("signed and absolute magnitude agree", bool(((d.Magnitude.abs() - d.Abs_Mag).abs() < 1e-9).all()))
    checks.internal("rows = 383 and latency within 0.05-0.30 s", len(d) == 383 and d.Latency.between(0.05, 0.3).all())
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    meta = dict(
        study=ID, paper={k2: p0[k2] for k2 in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="none stated on the OSF project (the preprint is CC BY 4.0)", license_url="",
        changes="Reshaped the authors' per-participant response table into long form (three measures per row) and replaced participant codes.",
        how=[
            "The authors shared one averaged pitch-shift response per participant, perturbation direction (+200 or -200 cents), sentence focus "
            "(new or corrective) and response direction (opposing or following the shift), from the sentence-production task only. "
            "Each row is one such cell and one measure; the three measures of a cell are on three consecutive rows.",
            "pert_value is +200 for the upward and -200 for the downward shift of the voice fundamental. pitch_shift_response_signed is the authors' "
            "Magnitude (cents, positive when f0 rose, so opposing responses to an upward shift are negative); the magnitude is its absolute value; "
            "latency is in seconds as shared. Sign and unit follow the paper's cents formula (1200 log2(f2/f1)), checked against Abs_Mag.",
        ],
        unit_note="participant x perturbation x focus x response direction, one measure",
        differences=[
            "The file holds 21 ataxia and 28 control participants (49, matching the F(1, 47) degrees of freedom) while the Methods text reports 22 and 29.",
            "Control group, corrective focus: file 129.10 cents, paper 129.89; group means over both focus types: file 131.15 (ataxia) and 110.79 (control), "
            "paper 123.24 and 106.41 (the paper's group values appear to be model-based). The sustained-vowel task is not in the deposit.",
        ],
    )
    return stdlib.release(ID, "summary", rows, meta, checks)


if __name__ == "__main__":
    main()
