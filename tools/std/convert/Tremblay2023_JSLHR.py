"""Tremblay, Gagnon, Roy & Arseneault (2023), J Speech Lang Hear Res 66(11):4332-4352: speech and voice production in 78 healthy adults
(38 amateur singers, 40 non-singing controls, aged 20-88): sustained vowels (maximal phonation time, voice quality), passage reading and
a diadochokinetic (DDK) task with words and non-words of varied complexity and frequency, at natural and maximal speed.

Raw files, raw_extra/Tremblay2023_JSLHR/ (Borealis doi:10.5683/SP3/GHD3HE): DDK.csv, MPT.csv, Voice.csv (semicolon, decimal comma),
Passages.csv (tab), README.docx (the data dictionary, read as text). The authors share aggregated values per participant, not recordings.
Run from the site folder: python3 -I tools/std/convert/Tremblay2023_JSLHR.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tremblay2023_JSLHR"
P = "Tremblay et al. 2023 data README, "
DDK = [
    ("Acc_W", "ddk_word_accuracy", "Percent of the stimulus (word or non-word) produced correctly in the DDK task (the authors' Acc_W)", "percent"),
    ("Acc_Syll", "ddk_syllable_accuracy", "Percent of syllables produced correctly in the DDK task (Acc_Syll; not analysed in the paper)", "percent"),
    ("Acc_S1", "ddk_syllable1_accuracy", "Percent of the first syllables produced correctly in the DDK task (Acc_S1; not analysed in the paper)", "percent"),
    ("Acc_S2", "ddk_syllable2_accuracy", "Percent of the second syllables produced correctly in the DDK task (Acc_S2; not analysed in the paper)", "percent"),
    ("Rate", "ddk_articulation_rate", "Articulatory rate in the DDK task (the authors' Rate)", "syllables per second"),
    ("nPVI", "ddk_npvi", "Normalized vocalic pairwise variability index of the DDK production (nPVI), a measure of rhythmic variability", "nPVI"),
]
VOICE = [
    ("Relative_jitter", "relative_jitter", "Relative jitter of the sustained vowel", "ratio"),
    ("Shimmer_dB", "shimmer", "Shimmer of the sustained vowel", "dB"),
    ("HNR", "hnr_mean", "Average harmonics-to-noise ratio over the whole trial", "dB"),
    ("HNR_SD", "hnr_sd", "Standard deviation of the harmonics-to-noise ratio over the trial", "dB"),
    ("MeanF0", "f0_mean", "Mean fundamental frequency over the trial", "Hz"),
    ("MeanF0_SD", "f0_sd", "Standard deviation of the fundamental frequency over the trial", "Hz"),
    ("Minpitch", "f0_min", "Minimal pitch of the trial (not analysed in the paper)", "Hz"),
    ("Maxpitch", "f0_max", "Maximal pitch of the trial (not analysed in the paper)", "Hz"),
]


def num(v):
    return None if pd.isna(v) else float(v)


def build():
    r = stdlib.raw_dir(ID)
    ddk = pd.read_csv(r / "DDK.csv", sep=";", decimal=",", encoding="utf-8-sig")
    mpt = pd.read_csv(r / "MPT.csv", sep=";", decimal=",", encoding="utf-8-sig")
    voi = pd.read_csv(r / "Voice.csv", sep=";", decimal=",", encoding="utf-8-sig")
    pas = pd.read_csv(r / "Passages.csv", sep="\t", encoding="utf-8-sig")
    rows = []

    def add(c, exp, block, cond, measure, desc, stat, val, unit, ntr=""):
        if pd.isna(val):
            return
        rows.append({
            "dataset": ID, "study": ID, "experiment": exp, "group": "singers" if c["Group"] == 1 else "controls",
            "subject": str(c["SID"]), "subject_excluded_by_authors": False, "age": c["Age"], "sex": "F" if c["Sex"] == 1 else "M",
            "handedness": "", "population": "healthy", "setting": "lab", "device": "microphone", "session": 1, "block": block,
            "trial": 0, "phase": "test", "time_s": "", "task": "speech", "unit": "participant summary", "condition": cond,
            "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
            "measure": measure, "measure_description": desc, "statistic": stat, "value": float(val), "value_unit": unit,
            "n_trials": ntr, "session_context": ""})

    for c in ddk.to_dict("records"):
        cond = "%s, %s, %s, %s speed" % ("word" if c["Type"] == 1 else "non-word", "complex" if c["Complexity"] == 1 else "simple",
                                         "rare" if c["Frequency"] == 1 else "frequent", "maximal" if c["Speed"] == 1 else "natural")
        for col, m, d, u in DDK:
            add(c, "Diadochokinetic task", "DDK", cond, m, d, "value", c[col], u)
    for c in mpt.to_dict("records"):
        add(c, "Maximal phonation time", "MPT", "trial %d" % c["Trial"], "maximal_phonation_time",
            "Duration of the longest sustained vowel of the trial (the authors' Duration)", "value", c["Duration"], "s", 1)
    for c in voi.to_dict("records"):
        for col, m, d, u in VOICE:
            add(c, "Voice quality", "Voice", "trial %d" % c["Trial"], m, d, "value", c[col], u, 1)
    for c in pas.to_dict("records"):
        add(c, "Passage reading", "Passages", "", "reading_rate", "Articulatory rate while reading the standardized passage (the authors' ReadingRate)",
            "value", c["ReadingRate"], "syllables per second")
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, ddk


def main():
    checks = stdlib.Checks()
    a, ddk = build()
    p = ddk.drop_duplicates("SID")
    checks.published("participants", len(p), 78, 0, P + "Sample (78 healthy adults)")
    checks.published("females", (p.Sex == 1).sum(), 44, 0, P + "Sample (44 females)")
    checks.published("males", (p.Sex == 0).sum(), 34, 0, P + "Sample (34 males)")
    checks.published("singers", (p.Group == 1).sum(), 38, 0, P + "Sample (38 amateur singers)")
    checks.published("controls", (p.Group == 0).sum(), 40, 0, P + "Sample (40 controls)")
    checks.published("mean age (years)", p.Age.mean(), 58.35, 0.005, P + "Sample (58.35)")
    checks.published("SD of age (years)", p.Age.std(), 18, 0.5, P + "Sample (SD 18, printed as an integer)")
    checks.published("youngest age", p.Age.min(), 20, 0, P + "Sample (20-88 years)")
    checks.published("oldest age", p.Age.max(), 88, 0, P + "Sample (20-88 years)")
    checks.internal("78 persons in the rows", a.subject.nunique() == 78)
    checks.internal("16 DDK cells per participant", (ddk.groupby("SID").size() == 16).all())
    checks.internal("MPT durations are positive", (a[a.measure == "maximal_phonation_time"].value > 0).all())
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY-NC 4.0",
        license_url="http://creativecommons.org/licenses/by-nc/4.0",
        changes=("Merged the authors' four aggregated tables (DDK, maximal phonation time, voice, passage reading) into one long table, "
                 "labelled the DDK conditions from the data dictionary, and replaced participant codes. Their language, hearing, "
                 "dementia-risk, depression and cognition covariates were not carried (only age, sex and group)."),
        how=[
            "The authors shared aggregated values, not recordings, so each row is one measure of one participant in one condition. "
            "Sustained-vowel measures (maximal phonation time in seconds, and jitter, shimmer, harmonics-to-noise ratio and f0 of "
            "the vowel) have one row per trial (three trials, n_trials = 1); DDK measures have one row per participant and "
            "stimulus condition (word or non-word, simple or complex, frequent or rare, natural or maximal speed), and passage "
            "reading rate has one row per participant.",
            "group is singers (amateur singers) or controls (cognitively active non-singers). DDK accuracies are percentages and the rate is in "
            "syllables per second as the data dictionary states.",
            "Units of the voice measures are inferred (dB for shimmer and harmonics-to-noise ratio, Hz for f0, a ratio for relative jitter): "
            "the data dictionary names the measures but gives no units.",
            "Blank cells in the authors' files (a few DDK, voice and MPT values) have no row. The authors state that their original recordings "
            "were not shared because participants had not consented to individual data sharing; these aggregated values are the deposit as published.",
        ],
        unit_note="participant, measure and condition (trial for sustained vowels)",
        differences=["The README gives the group breakdown as 23 females among 38 singers aged 20-87 and 19 females among 40 controls aged 23-88; the shared rows give 24 and 20 females and a singer aged 88 and a control aged 20 (the totals of 44 females, 38 singers, 40 controls and ages 20-88 match), so the group labels of one or two participants differ from the README, and the rows follow the shared files.", "The paper's effect sizes and mixed-model results were not recomputed; checks are limited to the sample description in the deposit's README."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
