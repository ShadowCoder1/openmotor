"""Gunzel, Tuomainen, Bross, Hanne & Wartenburger (2026), J Speech Lang Hear Res: tapping to metronomes and music in 16 individuals with aphasia and
32 age-matched neurotypical controls (the tapping half of a tapping plus speech-to-speech synchronization study).

Raw files, raw_extra/Guenzel2026_JSLHR/ (OSF e5ym2):
  results_aphasia.json, results_control.json   per participant and stimulus: tapping summary measures (mean/SD of asynchrony and inter-tap interval, MVL, PLV)
  SYNC_VP_info.csv                              demographics (age, gender, group), semicolon-separated
  SYNC_Analysis.Rmd                             the authors' analysis (read as text; printed values in its comments are used for checks)
Not carried: the speech-sync results (results_*.pkl are Python pickles and were not opened), SYNC_Data.RData (not readable without R), QS_overview.csv (prosody perception).
Run from the site folder: python3 -I tools/std/convert/Guenzel2026_JSLHR.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Guenzel2026_JSLHR"
P = "Gunzel et al. 2026, "
MEAS = [
    ("mean_async", "mean_asynchrony", "Mean signed asynchrony between taps and stimulus onsets (negative: taps lead the stimulus)", "mean", "ms"),
    ("mean_abs_async", "mean_absolute_asynchrony", "Mean absolute asynchrony between taps and stimulus onsets", "mean", "ms"),
    ("std_async", "sd_asynchrony", "Standard deviation of the signed asynchrony", "sd", "ms"),
    ("std_abs_async", "sd_absolute_asynchrony", "Standard deviation of the absolute asynchrony", "sd", "ms"),
    ("mean_iti", "mean_inter_tap_interval", "Mean interval between successive taps", "mean", "ms"),
    ("std_iti", "sd_inter_tap_interval", "Standard deviation of the inter-tap interval", "sd", "ms"),
    ("mvl", "mean_vector_length", "Mean vector length of the relative tap phases (0 to 1; higher is more consistent synchronization)", "value", "0 to 1"),
    ("plv", "phase_locking_value", "Phase locking value between tap times and stimulus onsets (0 to 1)", "value", "0 to 1"),
]


def load(path):
    t = open(path, encoding="utf-8").read().replace("NaN", "null")
    return json.loads(t)["participant"]


def main():
    raw = stdlib.raw_dir(ID)
    info = {r["participant"]: r for r in csv.DictReader(open(raw / "SYNC_VP_info.csv", encoding="utf-8-sig", errors="replace", newline=""), delimiter=";")}
    rows = []
    ages = {"control": [], "aphasia": []}
    npart = {"control": 0, "aphasia": 0}
    for grp in ("aphasia", "control"):
        for pid, items in load(raw / f"results_{grp}.json").items():
            v = info[pid]
            assert v["group"] == grp and v["informed_consent"] == "1"
            npart[grp] += 1
            ages[grp].append(float(v["age"]))
            for item, res in items.items():
                name = item.replace("_results", "")
                if name.startswith("iso"):
                    cond = f"metronome, inter-onset interval {name.split('_')[1]} ms"
                    kind = "metronome"
                else:
                    cond = f"music excerpt {name.replace('music_train', '')}"
                    kind = "music"
                for key, mname, desc, stat, unit in MEAS:
                    val = res.get(key)
                    if val is None:
                        continue
                    rows.append({
                        "dataset": ID, "study": ID, "experiment": "Tapping to metronome and music", "group": grp, "subject": pid,
                        "subject_excluded_by_authors": False, "age": float(v["age"]), "sex": {"w": "F", "m": "M"}.get(v["gender"], ""),
                        "handedness": "", "population": "individuals with aphasia" if grp == "aphasia" else "neurotypical older adults",
                        "setting": "lab", "device": "tapping recorded with an audio card, stimuli over headphones or speakers", "session": 1,
                        "block": name, "trial": 0, "phase": "test", "time_s": "", "task": "finger tapping and timing", "unit": "participant summary",
                        "condition": cond, "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
                        "measure": mname, "measure_description": desc + f" ({kind} stimulus)", "statistic": stat, "value": val, "value_unit": unit,
                        "n_trials": "", "session_context": ""})
    df = pd.DataFrame(rows)
    df["trial"] = df.groupby("subject").cumcount() + 1

    ch = stdlib.Checks()
    ch.published("individuals with aphasia", npart["aphasia"], 16, 0, P + "Abstract (sixteen individuals with aphasia)")
    ch.published("neurotypical controls", npart["control"], 32, 0, P + "Abstract (32 age-matched controls)")
    ch.published("mean age, controls", np.mean(ages["control"]), 67.1, 0.05, "SYNC_Analysis.Rmd printed age comparison (control M = 67.1)")
    ch.published("SD of age, controls", np.std(ages["control"], ddof=1), 6.93, 0.005, "SYNC_Analysis.Rmd printed age comparison (control SD = 6.93)")
    ch.published("mean age, aphasia", np.mean(ages["aphasia"]), 61.4, 0.05, "SYNC_Analysis.Rmd printed age comparison (aphasia M = 61.4)")
    ch.published("SD of age, aphasia", np.std(ages["aphasia"], ddof=1), 6.25, 0.005, "SYNC_Analysis.Rmd printed age comparison (aphasia SD = 6.25)")
    ch.internal("every participant has 9 stimuli", bool((df.groupby("subject").block.nunique() == 9).all()))
    ch.internal("PLV and MVL within 0..1", bool(df[df.measure.isin(["phase_locking_value", "mean_vector_length"])].value.between(0, 1).all()))
    ch.internal("rows kept (rows with a missing value are not written)", len(df) > 3000, str(len(df)))
    for c in ch.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {x: p0[x] for x in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Reshaped the authors' nested tapping summaries (participant, stimulus, measure) into long form, added age, sex and group from the demographics table, and replaced participant codes. Free-text and clinical columns of the demographics table were not carried.",
        how=["Participants tapped along to isochronous metronomes (inter-onset interval 450, 600 and 750 ms) and to six music excerpts. Each row is one summary measure for one participant and stimulus, as computed by the authors' tapping pipeline: asynchrony and inter-tap interval means and SDs in ms, and the mean vector length and phase locking value (0 to 1).",
             "Asynchrony sign (negative when taps lead) is the usual convention and was not stated in the deposit, so it is inferred. The number of taps behind each value was not shared.",
             "The speech-to-speech synchronization results of the same study are only in Python pickle files, which were not opened, so they are not included."],
        unit_note="participant, stimulus and measure",
        differences=["The paper's full text was not available, so published checks use the abstract and the age summary printed in the authors' analysis script comments.",
                     "The authors removed about 12.5% of the aphasia tapping PLV values for data quality in their analysis (script comment); the shared JSON still contains them and they are not marked here.",
                     "The demographics file lists one further person with aphasia who is not in the tapping results (the script excludes them on aetiology); the demographics file also has an exclude flag set for two controls whose meaning is not documented, and it was not applied."])
    return stdlib.release(ID, "summary", df.to_dict("records"), meta, ch)


if __name__ == "__main__":
    main()
