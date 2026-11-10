"""Barchet, Henry, Pelofi & Rimmele (2024), Commun Psychol 2(1):2: accelerated speech-to-speech synchronization (SSS) test, adapted to whispering
or tapping in time with accelerating syllable or piano-tone sequences at a fast and a slow rate (66 adults, 62 after exclusions).

Raw files, raw_extra/Barchet2024_CommunPsychol/ (OSF 9qthr):
  data_plv_trial.csv           PLV per participant, condition and trial (2 trials per condition)
  data_plv_trial_shuffle.csv   PLV of the permutation baseline (shuffled auditory signal) for the same cells
  plv_corrected.csv            baseline-corrected PLV averaged over the 2 trials (used as an internal check)
  data_plv.csv, msi_data.csv   condition means (not carried); questionnaire (not carried)
  Synchronization_Analysis_final.R   the authors' exclusion rules (read as text)
The perception task file (data_perception.csv) is a listening task and is not carried.
Paper read in PubMed Central (PMC11332030).
Run from the site folder: python3 -I tools/std/convert/Barchet2024_CommunPsychol.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Barchet2024_CommunPsychol"
P = "Barchet et al. 2024, "
LOUD = {"sub_37", "sub_60"}  # spoke loudly instead of whispering (authors' script and Methods)


def main():
    raw = stdlib.raw_dir(ID)
    tr = pd.read_csv(raw / "data_plv_trial.csv").set_index("Subjects")
    sh = pd.read_csv(raw / "data_plv_trial_shuffle.csv").set_index("Subjects").loc[tr.index, tr.columns]
    # R write.table output: header has one name fewer than the fields, so pandas takes the row number as the index
    cor = pd.read_csv(raw / "plv_corrected.csv", sep=r"\s+", quotechar='"')
    cor["Subjects"] = cor.index if cor.index.dtype == object else cor["Subjects"]
    base = tr - sh
    # the authors' consistency rule (their script): per condition regress trial 2 on trial 1 of the baseline-corrected PLV over the
    # participants who did not speak loudly; a participant is inconsistent if trial 2 lies outside the 99% prediction interval
    keep = base.drop(index=list(LOUD))
    conds = sorted({c[:-2] for c in tr.columns})
    incons = set()
    n = len(keep)
    for c in conds:
        x, y = keep[c + "_1"].values, keep[c + "_2"].values
        b, a = np.polyfit(x, y, 1)
        res = y - (a + b * x)
        s = np.sqrt((res ** 2).sum() / (n - 2))
        h = 1 / n + (x - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
        half = stats.t.ppf(0.995, n - 2) * s * np.sqrt(1 + h)
        incons |= set(keep.index[np.abs(res) > half])
    excl = LOUD | incons

    rows = []
    for sub in tr.index:
        for c in conds:
            eff, stim, rate = c.split("_")
            cond = f"{'whispering' if eff == 'whisper' else 'tapping'} to {'piano tones' if stim == 'tones' else 'syllables'}, {rate} rate"
            for t in (1, 2):
                for meas, val, desc in (
                        ("plv", tr.loc[sub, f"{c}_{t}"], "Phase locking value between the envelope of the heard accelerating sequence and the envelope of the motor output (whispered or tapped), 0 to 1"),
                        ("plv_permutation_baseline", sh.loc[sub, f"{c}_{t}"], "PLV of the permutation baseline: motor envelope against a shuffled version of the heard envelope"),
                        ("plv_baseline_corrected", base.loc[sub, f"{c}_{t}"], "plv minus plv_permutation_baseline (the authors' baseline correction, computed per trial in their script)")):
                    rows.append({
                        "dataset": ID, "study": ID, "experiment": "Perception-production synchronization task", "group": "",
                        "subject": sub, "subject_excluded_by_authors": sub in excl, "age": "", "sex": "", "handedness": "",
                        "population": "healthy adults", "setting": "lab",
                        "device": "microphone for whispering; tapping on a table next to a microphone; headphones",
                        "session": 1, "block": c, "trial": 0, "phase": "test", "time_s": "",
                        "task": "speech" if eff == "whisper" else "finger tapping and timing", "unit": "participant summary",
                        "condition": cond, "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal",
                        "excluded_by_authors": sub in excl, "measure": meas, "measure_description": desc, "statistic": "value",
                        "value": val, "value_unit": "PLV (-1 to 1 after correction)" if meas.endswith("corrected") else "PLV (0 to 1)",
                        "n_trials": 1, "session_context": f"synchronization trial {t} of 2"})
    df = pd.DataFrame(rows)
    df["trial"] = df.groupby("subject").cumcount() + 1

    ch = stdlib.Checks()
    ch.published("participants who took the synchronization task", df.subject.nunique(), 66, 0, P + "Participants (66 participated)")
    ch.published("participants who spoke loudly", len(LOUD), 2, 0, P + "Participants (two excluded for speaking loudly)")
    ch.published("participants inconsistent between trials (authors' rule)", len(incons), 2, 0, P + "Participants (additional 2 excluded for inconsistency)")
    ch.published("final synchronization sample", df[~df.excluded_by_authors].subject.nunique(), 62, 0, P + "Participants and Table 1 (N = 62)")
    cor = cor.set_index("Subjects")
    mean2 = pd.DataFrame({c: base[[c + "_1", c + "_2"]].mean(axis=1) for c in conds})
    ch.internal("mean of the two corrected trials equals the authors' plv_corrected.csv",
                bool(((mean2.loc[cor.index, conds] - cor[conds]).abs().max().max()) < 1e-6))
    ch.internal("66 participants x 16 cells x 3 measures", len(df) == 66 * 16 * 3)
    ch.internal("raw PLV in 0..1", df[df.measure == "plv"].value.between(0, 1).all())
    for c in ch.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {x: p0[x] for x in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Reshaped the authors' wide per-trial PLV tables into long form (three measures per participant, condition and trial), subtracted the permutation baseline per trial as the authors' script does, marked the authors' exclusions, and replaced participant codes. The perception task and questionnaire files were not carried.",
        how=["Participants whispered the syllable 'teh' or tapped on a table in time with accelerating sequences of syllables or piano tones at a fast or a slow rate (two trials per condition). Each row is one measure for one participant, condition and trial.",
             "plv is the phase locking value between the envelope of the heard sequence and of the motor output; plv_permutation_baseline is the same value against a shuffled heard envelope; plv_baseline_corrected is their difference. Raw audio was not shared.",
             "excluded_by_authors marks the two participants who spoke loudly (named in the authors' script) and the two the authors' consistency rule flags; the rule was re-applied here from their script (regression of trial 2 on trial 1, 99% prediction interval, per condition on baseline-corrected PLV) and gives the paper's count of two. No age or sex was shared with the PLV files."],
        unit_note="participant, condition and trial",
        differences=[])
    return stdlib.release(ID, "summary", df.to_dict("records"), meta, ch)


if __name__ == "__main__":
    main()
