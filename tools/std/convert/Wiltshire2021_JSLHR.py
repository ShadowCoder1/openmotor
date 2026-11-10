"""Wiltshire et al. (2021), J Speech Lang Hear Res 64(7):2438-2452: speech movement variability in people who stutter (PWS) and
typically fluent speakers (PWTF), vocal tract real-time MRI (OSF 3qdnv).

Raw files (raw_extra/Wiltshire2021_JSLHR/): vtMRI_nonwords_manuscript_data.csv (one row per participant x pseudoword) and the
authors' OSF_data_README.rtf, which defines the columns: group 3 = PWS, 4 = typically fluent; word 1-5 = pseudowords of 1, 2, 3
syllables, 4 = four syllables (complex), 5 = four syllables (simple); SSI = Stuttering Severity Instrument; mean/std/cov_duration =
mean, SD and coefficient of variation of utterance length; XX_mean/std/cov_summed_amp = mean, SD and CoV over the 10 repetitions
of the summed amplitude (the value of each frame added together for each utterance); LA lip aperture, VD velum distance, TB tongue
body. The paper gives duration as the number of frames from utterance start to end (frames inferred as the unit).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wiltshire2021_JSLHR"
SRC = "Wiltshire et al. 2021, Methods (Participants)"
WORD = {1: "word 1 (1 syllable)", 2: "word 2 (2 syllables)", 3: "word 3 (3 syllables)", 4: "word 4 (4 syllables, complex)",
        5: "word 5 (4 syllables, simple)"}
ART = {"LA": "lip aperture", "VD": "velum distance", "TB": "tongue body"}
STAT = {"mean": ("mean", "Mean"), "std": ("sd", "Standard deviation"), "cov": ("other", "Coefficient of variation (SD / mean)")}
CTX = "Pseudoword read 10 times in random order in a vocal tract real-time MRI scanner."


def main():
    raw = stdlib.raw_dir(ID)
    a = pd.read_csv(raw / "vtMRI_nonwords_manuscript_data.csv", encoding="utf-8-sig")
    rows = []
    for r in a.to_dict("records"):
        pws = r["group"] == 3
        items = [(f"utterance_duration_{k}", f"{STAT[k][1]} of the utterance duration over the repetitions (frames)", STAT[k][0],
                  r[f"{k}_duration"], "ratio" if k == "cov" else "frames") for k in STAT]
        for ab, nm in ART.items():
            for k in STAT:
                items.append((f"{ab}_{k}_summed_amp", f"{STAT[k][1]} over the repetitions of the summed amplitude of {nm} "
                              "(each frame's value added across the utterance; authors' units)", STAT[k][0],
                              r[f"{ab}_{k}_summed_amp"], "ratio" if k == "cov" else "summed amplitude (authors' units)"))
        for m, desc, stat, val, unit in items:
            if pd.isna(val):
                continue
            rows.append({
                "dataset": ID, "study": ID, "experiment": "", "group": "PWS" if pws else "PWTF", "subject": str(r["p_code"]),
                "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
                "population": "stuttering" if pws else "healthy", "setting": "lab", "device": "real-time vocal tract MRI",
                "session": 1, "block": WORD[int(r["word"])], "trial": 0, "phase": "other", "time_s": "", "task": "speech",
                "unit": "participant summary", "condition": "", "pert_type": "none", "pert_value": 0.0, "pert_unit": "",
                "feedback": "normal", "excluded_by_authors": False, "measure": m, "measure_description": desc,
                "statistic": stat, "value": float(val), "value_unit": unit, "n_trials": "", "session_context": CTX})
        if pws and not pd.isna(r["SSI"]):
            rows.append({**rows[-1], "block": WORD[int(r["word"])], "measure": "SSI_score",
                         "measure_description": "Stuttering Severity Instrument score of the participant (repeated on each word)",
                         "statistic": "value", "value": float(r["SSI"]), "value_unit": "score"})
    df = pd.DataFrame(rows)
    checks = stdlib.Checks()
    n = df.groupby("group").subject.nunique()
    checks.published("people who stutter analysed", n["PWS"], 28, 0, SRC + " (28 PWS after exclusions)")
    checks.published("typically fluent speakers analysed", n["PWTF"], 20, 0, SRC + " (20 PWTF)")
    ssi = a[a.group == 3].drop_duplicates("p_code").SSI
    checks.published("median SSI score of PWS", ssi.median(), 28.5, 0.05, SRC + ", Table 1")
    checks.published("SSI range minimum", ssi.min(), 16, 0, SRC + ", Table 1 (range 16-40)")
    checks.published("SSI range maximum", ssi.max(), 40, 0, SRC + ", Table 1 (range 16-40)")
    full = a.groupby("p_code").size()
    checks.published("participants with partial data (6 PWS and 2 PWTF)", (full < 5).sum(), 8, 0, SRC + " (partial datasets)")
    checks.internal("five pseudowords per complete participant", (full == 5).sum() == 40)
    checks.internal("no duplicate subject x word", not a.duplicated(["p_code", "word"]).any())
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    meta = dict(
        study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="none stated", license_url="",
        changes="Reshaped the authors' wide participant x pseudoword table into long form and replaced participant codes.",
        how=["One row is one measure of one participant for one pseudoword (block, 10 repetitions summarized): utterance duration "
             "(frames) and the summed amplitude of lip aperture (LA), velum distance (VD) and tongue body (TB), each as mean, SD and "
             "coefficient of variation over the repetitions.",
             "Words 1-3 have one to three syllables, word 4 four syllables (complex) and word 5 four syllables (simple), as the authors' "
             "README says. Duration unit (frames) comes from the paper's methods, not the README.",
             "SSI is repeated for the 28 PWS as a row on each word. Age and sex were not shared. Eight participants (the paper's partial "
             "datasets) have fewer than five words."],
        unit_note="participant, pseudoword and measure",
        differences=["The paper prints group statistics only in figures and models, so no group mean or CoV could be reproduced."])
    return stdlib.release(ID, "summary", df.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
