"""Rao, Gendron, Manning & Ostry (2026), PNAS 123(17):e2525468123: TMS (continuous theta burst) over STG, S1, M1 or no TMS after F1-shift
speech learning; retention tested 24 h later. 15 different participants per condition (60 in the analysis).

Raw files, raw_extra/Rao2026_PNAS/ (OSF 4n2ux, Data_RaoEtAl_PNAS_2026), each a table of 15 participants x 4 conditions
(Control, STG, M1, S1): {F1,F2}_{DuringBaseline_NoiseFeedback, LearningAsymptote_SpeechFeedback, BeforeRetention_NoiseFeedback,
Retention_SpeechFeedback, RelearningAsymptote_SpeechFeedback}.csv, values = percent change in the formant from baseline.
Paper read in PubMed Central (PMC13123799), Methods and Results. Run: python3 -I tools/std/convert/Rao2026_PNAS.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Rao2026_PNAS"
P = "Rao et al. 2026, "
GROUPS = ["Control", "STG", "M1", "S1"]
SITE = {"Control": "no TMS", "STG": "cTBS over superior temporal gyrus", "M1": "cTBS over primary motor cortex",
        "S1": "cTBS over posterior primary somatosensory cortex"}
# file stem -> (visit, phase, feedback, measure suffix, description)
STAGES = [
    ("DuringBaseline_NoiseFeedback", 1, "baseline", "masked", "baseline_noise_trials",
     "Mean of the three speech-masked noise feedback trials interspersed in the visit 1 baseline (no perturbation)"),
    ("LearningAsymptote_SpeechFeedback", 1, "learning", "altered", "learning_asymptote",
     "Mean over the last 10 bins (30 trials) of visit 1 with F1 raised 30% in the feedback"),
    ("BeforeRetention_NoiseFeedback", 2, "test", "masked", "noise_trials_before_retention",
     "Mean of the three speech-masked noise feedback trials at the start of visit 2, 24 h later"),
    ("Retention_SpeechFeedback", 2, "test", "altered", "retention",
     "Mean of three speech-feedback trials following the first speech feedback trial of visit 2 (retention)"),
    ("RelearningAsymptote_SpeechFeedback", 2, "learning", "altered", "relearning_asymptote",
     "Mean over the last 10 bins (30 trials) of visit 2 (relearning)"),
]


def load():
    raw = stdlib.raw_dir(ID)
    out = {}
    for fm in ("F1", "F2"):
        for stem, *_ in STAGES:
            d = pd.read_csv(raw / f"{fm}_{stem}.csv", skiprows=1, header=0)
            d = d[["Participant"] + GROUPS].apply(pd.to_numeric)
            out[fm, stem] = d
    return out


def main():
    checks = stdlib.Checks()
    tabs = load()
    rows = []
    n = {}
    for (fm, stem), d in tabs.items():
        _, visit, phase, fb, short, desc = next(s for s in STAGES if s[0] == stem)
        for g in GROUPS:
            for r in d.itertuples(index=False):
                pid = int(r.Participant)
                sub = f"{g}_{pid}"
                v = getattr(r, g)
                if pd.isna(v):
                    continue
                n[sub] = n.get(sub, 0) + 1
                rows.append({
                    "dataset": ID, "study": ID, "experiment": "cTBS after F1-shift speech learning", "group": SITE[g], "subject": sub,
                    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
                    "device": "microphone and headphones with real-time formant shifting; Magstim Rapid 2 cTBS", "session": visit, "block": stem,
                    "trial": n[sub], "phase": phase, "time_s": "", "task": "speech", "unit": "participant summary", "condition": stem.split("_")[1],
                    "pert_type": "formant_shift" if fb == "altered" else "none", "pert_value": 30.0 if fb == "altered" else 0.0,
                    "pert_unit": "percent F1 increase", "feedback": fb, "excluded_by_authors": False,
                    "measure": f"{fm.lower()}_percent_change_{short}",
                    "measure_description": f"Percent change in {fm} of the produced vowel from the participant's baseline mean. " + desc,
                    "statistic": "mean", "value": float(v), "value_unit": "percent", "n_trials": "", "session_context": "",
                })
    ret = tabs["F1", "Retention_SpeechFeedback"]
    checks.published("participants analysed (15 per condition x 4)", len({r["subject"] for r in rows}), 60, 0, P + "Methods, Participants (60 participants)")
    tS = stats.ttest_ind(ret["STG"], ret["Control"])
    checks.published("retention, STG vs no TMS, independent t(28)", tS.statistic, 3.474, 0.001, P + "Results, retention")
    tS1 = stats.ttest_ind(ret["S1"], ret["Control"])
    checks.published("retention, S1 vs no TMS, independent t(28)", tS1.statistic, 2.861, 0.001, P + "Results, retention")
    F = stats.f_oneway(*[ret[g] for g in GROUPS])
    checks.published("main effect of TMS condition on retention, F(3, 56)", F.statistic, 5.006, 0.001, P + "Results, retention")
    F2 = stats.f_oneway(*[tabs["F2", "Retention_SpeechFeedback"][g] for g in GROUPS])
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    meta = dict(
        study=ID, paper={k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Reshaped the authors' ten participant-by-condition tables into one long table and replaced participant numbers with codes.",
        how=[
            "Each row is one participant's percent change in F1 or F2 from his or her own baseline in one stage of the experiment: the three "
            "noise-feedback trials of the baseline, the learning asymptote of visit 1 (F1 raised 30% in the feedback), the noise-feedback trials "
            "24 h later, retention (first speech-feedback trials of visit 2) and the relearning asymptote of visit 2.",
            "Each condition (no TMS, or theta-burst TMS over STG, M1 or S1 after learning) has 15 different participants, so the participant "
            "numbers in the shared tables are not the same people across columns; subject codes combine condition and number. A negative F1 change "
            "means compensation for the raised F1.",
        ],
        unit_note="participant, stage and formant",
        differences=["The 11 participants the authors excluded for lack of learning on visit 1 are not in the shared tables, "
                     "which hold the 60 analysed participants. Only the retention statistics were recomputed."],
    )
    return stdlib.release(ID, "summary", rows, meta, checks)


if __name__ == "__main__":
    main()
