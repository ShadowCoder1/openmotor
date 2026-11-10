"""Beach, Tang, Kiran & Niziolek (2024), Neurobiology of Language 5(2):454-483: speech vowel centering and the speaking-induced
suppression (SIS) of the MEG M100 response in 15 people with chronic left-hemisphere aphasia and 15 age-matched controls
(saying "eat", "Ed" and "add" while MEG was recorded, and listening to playback of their own productions).

Raw files, raw_extra/Beach2024_NeurobiolLang/ (OSF hjwsx, fetched with tools/std/fetch.py; no license stated):
  data.csv    30 rows (s01-s30), one per participant, 46 columns, no codebook; the meaning of each column is taken from the
              paper (Methods "Acoustic analysis" and "MEG analysis", Table 1) and checked against the rows:
              speech: dinit_of_all (mean distance of every production's initial F1/F2 from its vowel's median, mels),
                dinit_of_periphest (the same for the 125 most peripheral productions), centering_of_periphest (initial minus
                mid-utterance distance of those 125, mels), centering_of_periphest_per (that centering / dinit_of_periphest,
                checked), dinitmid_of_all (initial minus mid-utterance distance over all productions: read from the name)
              MEG: <hemi>_peak_all_<speak|listen> and <hemi>_peaktime_all_<speak|listen> (M100 peak amplitude, latency in ms),
                <hemi>_sis_all_mag = listen minus speak peak (checked), <hemi>_sis_center_mag / _periph_mag (SIS for the 125 most
                central and most peripheral productions), <hemi>_sisfalloff_centerperiph_mag = center minus periph SIS (checked)
              descriptors (Table 1): age, male, aphasia, mpo (months post-stroke), lesion_vol (mm3), if_oper / heschl / sup_temp
                (percent of pars opercularis / Heschl's gyrus / superior temporal gyrus spared), anomic / brocas / apraxia,
                wabaq, palpa7, palpa8, pal7, pavg (mean of the three repetition scores, checked), nmegtrials
              not carried (meaning not documented in the paper or deposit): right50010002000 and left50010002000 (a hearing
                measure, probably a pure-tone average), om_mis_dist_other and mis_dist_other (speech-error counts), lat_* (laterality
                indices, formula not stated)
  M100.mat    MEG source time courses (1101 samples x 15 participants, by group, hemisphere and condition): neural recordings,
              not motor behaviour, so not converted
Run from the site folder: python3 -I tools/std/convert/Beach2024_NeurobiolLang.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Beach2024_NeurobiolLang"
SRC = "Beach, Tang, Kiran & Niziolek 2024, Neurobiology of Language 5:454"
PERIPH = "the 125 productions of each vowel with the largest initial distance from that vowel's median (peripheral trials)"
CENTER = "the 125 productions of each vowel with the smallest initial distance from that vowel's median (center trials)"
AMP = "amplitude of the M100 in the authors' source-space region, units not stated in the paper"

# (column, measure, description, statistic, unit, condition, n_trials)
SPEECH = [
    ("dinit_of_all", "initial_variability_all_mel",
     "Variability of all correct productions of eat, Ed and add: mean Euclidean distance in F1-F2 space (mels) of the formants in the initial "
     "window (0-50 ms from sound onset) from the participant's median for the same vowel. Column dinit_of_all.", "mean", "mel", "speak", ""),
    ("dinitmid_of_all", "centering_all_mel",
     "Centering of all correct productions: initial-window distance from the vowel median minus the distance in the mid-utterance window "
     "(middle 50% of the formant trajectory), mean over productions; positive when productions moved toward the median. Column "
     "dinitmid_of_all (meaning read from the name and from the paper's definition of centering).", "mean", "mel", "speak", ""),
    ("dinit_of_periphest", "initial_variability_peripheral_mel",
     f"Initial-window distance from the vowel median (mels), mean over {PERIPH}. Column dinit_of_periphest.", "mean", "mel", "speak", 125),
    ("centering_of_periphest", "centering_peripheral_mel",
     f"Centering of {PERIPH}: initial-window distance minus mid-utterance distance from the vowel median, mean over them (mels). "
     "Column centering_of_periphest.", "mean", "mel", "speak", 125),
    ("centering_of_periphest_per", "centering_peripheral_proportion",
     "centering_peripheral_mel divided by initial_variability_peripheral_mel (the proportion of the initial dispersion removed by "
     "centering; the paper reports it as a percentage). Column centering_of_periphest_per.", "proportion", "proportion", "speak", 125),
]
MEG = []
for h in ("left", "right"):
    for cond in ("speak", "listen"):
        MEG.append((f"{h}_peaktime_all_{cond}", f"{h}_M100_latency_{cond}",
                    f"Latency of the {h}-hemisphere M100 peak in the {cond} condition (ms after sound onset). Column {h}_peaktime_all_{cond}.",
                    "other", "ms", cond, ""))
    for cond in ("speak", "listen"):
        MEG.append((f"{h}_peak_all_{cond}", f"{h}_M100_amplitude_{cond}",
                    f"Peak {AMP}, {h} hemisphere, {cond} condition (all productions). Column {h}_peak_all_{cond}.",
                    "other", "authors' amplitude units", cond, ""))
    MEG.append((f"{h}_sis_all_mag", f"{h}_SIS",
                f"Speaking-induced suppression, {h} hemisphere: listen-evoked minus speak-evoked M100 peak amplitude, all productions. "
                f"Column {h}_sis_all_mag.", "other", "authors' amplitude units", "listen minus speak", ""))
    MEG.append((f"{h}_sis_center_mag", f"{h}_SIS_center",
                f"SIS, {h} hemisphere, for {CENTER}. Column {h}_sis_center_mag.", "other", "authors' amplitude units",
                "listen minus speak", 125))
    MEG.append((f"{h}_sis_periph_mag", f"{h}_SIS_peripheral",
                f"SIS, {h} hemisphere, for {PERIPH}. Column {h}_sis_periph_mag.", "other", "authors' amplitude units",
                "listen minus speak", 125))
    MEG.append((f"{h}_sisfalloff_centerperiph_mag", f"{h}_SIS_falloff",
                f"SIS fall-off, {h} hemisphere: SIS of center trials minus SIS of peripheral trials. Column {h}_sisfalloff_centerperiph_mag.",
                "other", "authors' amplitude units", "listen minus speak", ""))
MEASURES = SPEECH + MEG

PCOLS = [
    ("months_post_stroke", "number", "months", "Time since the stroke (column mpo); empty for controls."),
    ("lesion_volume_mm3", "number", "mm3", "Total lesion volume (Table 1; column lesion_vol); 0 for controls."),
    ("ifo_spared_pct", "number", "%", "Percent of the inferior frontal gyrus pars opercularis spared by the lesion (column if_oper)."),
    ("heschl_spared_pct", "number", "%", "Percent of Heschl's gyrus spared by the lesion (column heschl)."),
    ("stg_spared_pct", "number", "%", "Percent of the superior temporal gyrus spared by the lesion (column sup_temp)."),
    ("anomic_aphasia", "true/false", "", "Clinical diagnosis of anomic aphasia (Table 1 subtype; column anomic); empty for controls."),
    ("brocas_aphasia", "true/false", "", "Clinical diagnosis of Broca's aphasia (Table 1 subtype; column brocas); empty for controls."),
    ("apraxia_of_speech", "true/false", "", "Comorbid apraxia of speech (Table 1 subtype; column apraxia); empty for controls."),
    ("wab_aq", "number", "points", "Western Aphasia Battery Aphasia Quotient (column wabaq); empty for controls."),
    ("palpa7_prop", "number", "proportion correct", "PALPA 7 word repetition (column palpa7)."),
    ("palpa8_prop", "number", "proportion correct", "PALPA 8 nonword repetition (column palpa8)."),
    ("pal7_prop", "number", "proportion correct", "PAL 7 word and nonword repetition (column pal7)."),
    ("repetition_mean_prop", "number", "proportion correct", "Mean of palpa7, palpa8 and pal7 (column pavg; checked on the rows)."),
    ("n_meg_trials", "integer", "", "Number of MEG trials analysed (column nmegtrials)."),
]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def flag(v):
    return None if pd.isna(v) else bool(int(v))


def build(d):
    rows, parts = [], []
    for r in d.itertuples(index=False):
        aph = int(r.aphasia) == 1
        pers = {"dataset": ID, "study": ID, "experiment": "", "group": "aphasia" if aph else "control",
                "subject": r.subject, "subject_excluded_by_authors": False, "age": float(r.age),
                "sex": "M" if int(r.male) == 1 else "F", "handedness": "right",
                "population": "chronic left-hemisphere stroke with aphasia" if aph else "healthy (age-matched control)",
                "setting": "lab", "device": "MEG with microphone recording of speech (speak and listen conditions)", "session": 1}
        for k, (col, name, desc, stat, unit, cond, ntr) in enumerate(MEASURES, start=1):
            rows.append({**pers, "block": "", "trial": k, "phase": "test", "time_s": "", "task": "speech",
                         "unit": "participant summary", "condition": cond, "pert_type": "none", "pert_value": 0, "pert_unit": "",
                         "feedback": "normal" if cond == "speak" else "", "excluded_by_authors": False,
                         "measure": name, "measure_description": desc, "statistic": stat, "value": float(getattr(r, col)),
                         "value_unit": unit, "n_trials": ntr})
        parts.append({"person": r.subject, "months_post_stroke": r.mpo, "lesion_volume_mm3": r.lesion_vol,
                      "ifo_spared_pct": r.if_oper, "heschl_spared_pct": r.heschl, "stg_spared_pct": r.sup_temp,
                      "anomic_aphasia": flag(r.anomic), "brocas_aphasia": flag(r.brocas), "apraxia_of_speech": flag(r.apraxia),
                      "wab_aq": r.wabaq, "palpa7_prop": r.palpa7, "palpa8_prop": r.palpa8, "pal7_prop": r.pal7,
                      "repetition_mean_prop": r.pavg, "n_meg_trials": int(r.nmegtrials)})
    return rows, parts


def check_all(d, rows, checks):
    R = f"{SRC}, Results"
    a, c = d[d.aphasia == 1], d[d.aphasia == 0]
    checks.published("participants with aphasia", len(a), 15, 0, f"{SRC}, Methods (Participants)")
    checks.published("age-matched controls", len(c), 15, 0, f"{SRC}, Methods (Participants)")
    checks.published("males among the people with aphasia", int(a.male.sum()), 11, 0, f"{SRC}, Methods (Participants)")
    checks.published("males among the controls", int(c.male.sum()), 11, 0, f"{SRC}, Methods (Participants)")
    checks.published("people with aphasia: mean months post-stroke", a.mpo.mean(), 108, 0.5, f"{SRC}, Methods (Participants)")
    checks.published("... SD", a.mpo.std(ddof=1), 111, 0.5, f"{SRC}, Methods (Participants)")
    checks.published("... minimum", a.mpo.min(), 19, 0, f"{SRC}, Methods (Participants)")
    checks.published("... maximum", a.mpo.max(), 470, 0, f"{SRC}, Methods (Participants)")
    checks.published("peripheral trials: mean initial distance from the median (mels)", d.dinit_of_periphest.mean(), 62, 0.5, f"{R}, centering")
    checks.published("... mean distance at mid-utterance (mels)", (d.dinit_of_periphest - d.centering_of_periphest).mean(), 42, 0.5,
                     f"{R}, centering")
    checks.published("centering of peripheral trials: mean (mels)", d.centering_of_periphest.mean(), 20, 0.5, f"{R}, centering")
    checks.published("... SD", d.centering_of_periphest.std(ddof=1), 7, 0.5, f"{R}, centering")
    checks.published("... minimum", d.centering_of_periphest.min(), 8, 0.5, f"{R}, centering")
    checks.published("... maximum", d.centering_of_periphest.max(), 35, 0.5, f"{R}, centering")
    checks.published("centering as a share of the initial dispersion: mean (%)", 100 * d.centering_of_periphest_per.mean(), 32, 0.5,
                     f"{R}, centering")
    checks.published("... SD (%)", 100 * d.centering_of_periphest_per.std(ddof=1), 9, 0.5, f"{R}, centering")
    checks.published("... minimum (%)", 100 * d.centering_of_periphest_per.min(), 14, 0.5, f"{R}, centering")
    checks.published("... maximum (%)", 100 * d.centering_of_periphest_per.max(), 55, 0.5, f"{R}, centering")
    checks.published("correlation of centering with initial variability of peripheral trials (all participants)",
                     stats.pearsonr(d.centering_of_periphest, d.dinit_of_periphest).statistic, 0.65, 0.005, f"{R}, centering")
    t1 = stats.ttest_ind(a.centering_of_periphest, c.centering_of_periphest)
    checks.published("centering, aphasia vs control: |t(28)|", abs(t1.statistic), 0.74, 0.005, f"{R}, centering")
    t2 = stats.ttest_ind(a.dinit_of_all, c.dinit_of_all)
    checks.published("initial variability of all trials, aphasia vs control: |t(28)|", abs(t2.statistic), 1.59, 0.005, f"{R}, variability")
    for h in ("left", "right"):
        checks.internal(f"{h}_sis_all_mag = listen minus speak peak; falloff = center minus peripheral SIS (as the paper defines them)",
                        bool(np.allclose(d[f"{h}_sis_all_mag"], d[f"{h}_peak_all_listen"] - d[f"{h}_peak_all_speak"], atol=1e-6)
                             and np.allclose(d[f"{h}_sisfalloff_centerperiph_mag"], d[f"{h}_sis_center_mag"] - d[f"{h}_sis_periph_mag"], atol=1e-6)))
    ok = d[["palpa7", "palpa8", "pal7"]].dropna()
    checks.internal("pavg is the mean of the three repetition scores", bool(np.allclose(d.loc[ok.index, "pavg"], ok.mean(axis=1), atol=1e-3)))
    checks.internal("centering_of_periphest_per = centering / initial distance of the peripheral trials",
                    bool(np.allclose(d.centering_of_periphest_per, d.centering_of_periphest / d.dinit_of_periphest, atol=2e-3)))
    checks.internal("one row per participant and measure (30 x %d = %d), no missing values" % (len(MEASURES), 30 * len(MEASURES)),
                    len(rows) == 30 * len(MEASURES) and not any(pd.isna(r["value"]) for r in rows))
    return t1, t2


def main():
    checks = stdlib.Checks()
    d = pd.read_csv(stdlib.raw_dir(ID) / "data.csv")
    rows, parts = build(d)
    t1, t2 = check_all(d, rows, checks)
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Stacked the 30 participants' values into long form (one row per participant and measure, in the order of the "
                 "authors' columns), moved age, sex and group into the standard columns and the lesion, language-test and trial-count "
                 "values into participants.csv, converted nothing else, and replaced participant numbers with codes. Left out the columns "
                 "whose meaning neither the deposit nor the paper states (right50010002000, left50010002000, om_mis_dist_other, "
                 "mis_dist_other and the lat_* columns) and the MEG time courses in M100.mat."),
        how=[
            "One row is one measure of one participant (the deposit has one value per participant and no per trial data). Speech "
            "measures are the vowel variability and centering of the correct productions of eat, Ed and add recorded during MEG: "
            "distances in F1-F2 space in mels from the participant's median for the same vowel, in an initial window (0-50 ms from "
            "sound onset) and a mid-utterance window (middle 50% of the trajectory); centering is the initial minus the mid-utterance "
            "distance, positive when productions moved toward the median. Peripheral trials are the 125 with the largest initial distance.",
            "MEG measures are the peak latency (ms) and amplitude of the M100 in each hemisphere when the participant spoke and when "
            "they listened to playback of their own productions, and speaking-induced suppression (SIS = listen minus speak peak "
            "amplitude) for all productions, the 125 most central and the 125 most peripheral ones. The paper does not state the amplitude "
            "unit; values are kept as the authors shared them.",
            "Group is aphasia (15 people with chronic left-hemisphere stroke, all right-handed) or control (15 age-matched, right-handed); "
            "sex and age are from the deposit. Feedback was normal in the speak condition (no perturbation); rows of the listen condition "
            "and of listen-minus-speak differences have no feedback entry.",
        ],
        unit_note="measure of one participant (long form)",
        differences=[
            f"The group comparisons reproduce in size (centering |t(28)| = {abs(t1.statistic):.2f}, initial variability |t(28)| = "
            f"{abs(t2.statistic):.2f}, as in the paper); their sign here is aphasia minus control. The paper's MEG statistics (mixed "
            "ANOVAs, lesion-brain correlations) were not recomputed.",
        ],
    )
    return stdlib.release(ID, "summary", rows, meta, checks, participants=parts, participant_columns=PCOLS)


if __name__ == "__main__":
    main()
