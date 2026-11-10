"""Wiltshire et al. (2024), PLoS ONE 19(10):e0309612: speaking to a metronome, vocal tract real-time MRI, 24 people who stutter (PWS)
and 16 controls (OSF 79j4b).

Raw files (raw_extra/Wiltshire2024_PLoSOne/): Sent_Met_data_OSF.csv (one row per participant x sentence x condition x region),
Demographics_VT_sent_met_OSF.csv (one row per participant: sex, handedness, SSI score; no age), the authors' R script (read as text).
What the values are (paper Methods; R script vtMRI_sentmet_OSF_CODE.R): three sentences each repeated 10 times with (met) and without
(nomet) a 2 Hz metronome; movement size = sum of the constriction values across MRI frames (33.3 frames/s) at the alveolar, palatal
or velar constriction location; mean_summed_amp is its mean over the repetitions, COV its coefficient of variation (SD / mean),
mean_duration the mean utterance duration (frames, inferred), num_trials the repetitions kept. The authors' script drops rows with
num_trials <= 5 ('The one's that are excluded').
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wiltshire2024_PLoSOne"
SRC = "Wiltshire et al. 2024, Methods (Participants)"
CTX = "Sentence spoken in a vocal tract real-time MRI scanner, ten repetitions, with or without a 2 Hz metronome."


def main():
    raw = stdlib.raw_dir(ID)
    a = pd.read_csv(raw / "Sent_Met_data_OSF.csv", encoding="utf-8-sig")
    g = pd.read_csv(raw / "Demographics_VT_sent_met_OSF.csv", encoding="utf-8-sig").set_index("p_code")
    a["sentence"] = a.sentence.str.capitalize()
    rows = []
    for r in a.itertuples(index=False):
        d = g.loc[r.p_code]
        pws = r.group == "PWS"
        ex = bool(r.num_trials <= 5)
        cond = "metronome" if r.condition == "met" else "no metronome"
        for m, desc, stat, val, unit in (
            ("movement_size", f"Mean over repetitions of the summed constriction at the {r.region} location across MRI frames "
             f"(authors' mean_summed_amp; tier {r.tier})", "mean", r.mean_summed_amp, "summed constriction (authors' units)"),
            ("movement_size_cov", "Coefficient of variation (SD / mean) of the movement size across the repetitions, "
             f"{r.region} location (authors' COV); the authors' variability measure", "other", r.COV, "ratio"),
            ("utterance_duration", "Mean duration of the sentence over repetitions (authors' mean_duration; frames at 33.3/s inferred)",
             "mean", r.mean_duration, "frames")):
            rows.append({
                "dataset": ID, "study": ID, "experiment": "", "group": r.group, "subject": str(r.p_code),
                "subject_excluded_by_authors": False, "age": "", "sex": {"men": "M", "women": "F"}.get(d.Gender, ""),
                "handedness": str(d.Handedness).lower(), "population": "stuttering" if pws else "healthy", "setting": "lab",
                "device": "real-time vocal tract MRI", "session": 1, "block": r.sentence, "trial": 0, "phase": "other",
                "time_s": "", "task": "speech", "unit": "participant summary", "condition": cond, "pert_type": "none",
                "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": ex,
                "measure": f"{m}_{r.region}", "measure_description": desc, "statistic": stat, "value": float(val),
                "value_unit": unit, "n_trials": int(r.num_trials), "session_context": CTX})
    df = pd.DataFrame(rows)
    checks = stdlib.Checks()
    n = df.groupby("group").subject.nunique()
    checks.published("people who stutter analysed", n["PWS"], 24, 0, SRC)
    checks.published("controls analysed", n["CON"], 16, 0, SRC)
    sent = a.drop_duplicates(["p_code", "sentence", "condition"])
    checks.published("sentences x conditions per full participant (3 sentences, 2 conditions)",
                     sent.groupby("p_code").size().max(), 6, 0, SRC + " (60 trials = 3 sentences x 10 x 2 conditions)")
    checks.published("most repetitions kept per sentence", a.num_trials.max(), 10, 0, SRC + " (each sentence repeated 10 times)")
    checks.internal("every participant is in the demographics file", set(a.p_code) <= set(g.index))
    checks.internal("three regions per row group", a.groupby(["p_code", "sentence", "condition"]).region.nunique().max() == 3)
    checks.internal("1 rows x 3 measures", len(df) == 3 * len(a), str(len(df)))
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    meta = dict(
        study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC BY 4.0 (the PLoS ONE article; the OSF project states no license)",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Reshaped the authors' table into long form (three measures per row of theirs), added sex and handedness from the demographics table, replaced participant codes, and wrote the sentence 'peter'/'Peter' spelling the same way.",
        how=["One row is one measure of one participant for one sentence (block), metronome condition and constriction location "
             "(alveolar, palatal or velar, in the measure name); n_trials is the repetitions the authors averaged.",
             "movement_size is the authors' mean_summed_amp (summed constriction across frames, their units), movement_size_cov their "
             "COV, utterance_duration their mean_duration (frames at 33.3 per second, inferred from the paper's frame rate).",
             "excluded_by_authors marks rows with 5 or fewer usable repetitions, which the authors' R script removes. Age was not shared. "
             "The paper reports 24 + 16 analysed participants and the deposit holds exactly those."],
        unit_note="participant, sentence, condition and location",
        differences=["The paper's exclusion shares (4.2% of control and 2% of PWS sentences) were not used as checks; the deposit's "
                     "num_trials is the only exclusion information in it."])
    return stdlib.release(ID, "summary", df.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
