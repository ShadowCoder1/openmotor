"""Kim & Max (2021), Eur J Neurosci 53(9):3093-3108: speech auditory-motor adaptation to a 400-cent upward F1 shift in 12 adults who
stutter (AWS) and 12 who do not (AWNS); words "bed" and "pet"; intent-to-change ratings and auditory-target choices.

Raw files, raw_extra/Kim2021_EurJNeurosci/ (OSF s84df; the authors' key readmes were read as text):
  Adaptation_NormalizedCents_BED / _PET.txt  80 trials x 24 participants, F1 in cents (columns 1-12 AWNS, 13-24 AWS); trials 1-10
        baseline, 11-70 perturbation, 71-80 after-effects; normalized to the mean of baseline trials 6-10 (paper, Methods)
  explicit_BED_PET.txt  80 trials x 48: intent ratings, columns 1-24 "bed", 25-48 "pet", same participant order
  Target_BED / _PET.txt  17 selections x 24: auditory target (F1 cents of the chosen stimulus), rows 1-3 baseline, 4-15 perturbation,
        16-17 after-effects
Run from the site folder: python3 -I tools/std/convert/Kim2021_EurJNeurosci.py
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

ID = "Kim2021_EurJNeurosci"
SRC = "Kim & Max 2021, Eur J Neurosci, "
WORDS = {"BED": "bed", "PET": "pet"}


def grp(j):
    return "adults who do not stutter" if j < 12 else "adults who stutter"


def base_row(j, word, kind):
    return {
        "dataset": ID, "study": ID, "experiment": kind, "group": grp(j), "subject": f"P{j + 1:02d}|{word}|{kind}",
        "person": f"P{j + 1:02d}", "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
        "population": "healthy" if j < 12 else "stuttering", "setting": "lab",
        "device": "AKG C544 headset microphone, RME Babyface Pro, Audapter, Etymotic ER-3A insert earphones", "session": 1,
        "time_s": "", "task": "speech", "unit": "trial", "stimulus": word, "segment": "vowel /ɛ/",
        "masking_noise": "none", "feedback_latency_ms": 15.3,
    }


def build():
    raw = stdlib.raw_dir(ID)
    rows = []
    ex = np.loadtxt(raw / "explicit_BED_PET.txt")
    cents, tgt = {}, {}
    for W, word in WORDS.items():
        cents[word] = np.loadtxt(raw / f"Adaptation_NormalizedCents_{W}.txt")
        tgt[word] = np.loadtxt(raw / f"Target_{W}.txt")
    for j in range(24):
        for wi, word in enumerate(WORDS.values()):
            for t in range(80):
                ph = "baseline" if t < 10 else ("learning" if t < 70 else "aftereffect")
                pert = 400.0 if ph == "learning" else 0.0
                r = base_row(j, word, "Adaptation")
                v = cents[word][t, j]
                r.update({
                    "block": ("Baseline", "Perturbation", "After-effects")[(t >= 10) + (t >= 70)], "trial": t + 1, "phase": ph,
                    "condition": word, "pert_type": "formant_shift" if pert else "none", "pert_value": pert, "pert_unit": "cents",
                    "feedback": "altered" if pert else "normal", "excluded_by_authors": False,
                    "measure_window": "middle 20% of the vowel (40-60%)", "shift_f1": pert, "shift_f2": 0.0,
                    "f1_cents": "" if np.isnan(v) else v,
                    "change_reference": "mean F1 of baseline trials 6-10 of the same word",
                    "intent_rating": ex[t, wi * 24 + j],
                })
                rows.append(r)
            for t in range(17):
                ph = "baseline" if t < 3 else ("learning" if t < 15 else "aftereffect")
                pert = 400.0 if ph == "learning" else 0.0
                r = base_row(j, word, "Auditory target")
                r["unit"] = "trial"
                r.update({
                    "block": ("Baseline", "Perturbation", "After-effects")[(t >= 3) + (t >= 15)], "trial": t + 1, "phase": ph,
                    "condition": word, "pert_type": "formant_shift" if pert else "none", "pert_value": pert, "pert_unit": "cents",
                    "feedback": "altered" if pert else "normal", "excluded_by_authors": False,
                    "shift_f1": pert, "shift_f2": 0.0, "auditory_target_cents": tgt[word][t, j],
                    "measure_window": "not an acoustic measure: the participant picked the best-sounding F1 shift of a recording of their own word",
                })
                rows.append(r)
    return rows, cents, ex, tgt


def anova_cells(cells, g):
    """cells: n x 4 (phase x word: early-bed, early-pet, late-bed, late-pet); g: 0/1 group. Returns F for the 7 effects, df (1, n-2)."""
    eff = {"Group": np.array([1, 1, 1, 1]) / 4, "Phase": np.array([-1, -1, 1, 1]) / 2, "Word": np.array([-1, 1, -1, 1]) / 2,
           "PhaseXWord": np.array([1, -1, -1, 1]) / 2}
    out = {}
    for n, c in eff.items():
        s = cells @ c
        if n == "Group":
            out["Group"] = stats.f_oneway(s[g == 0], s[g == 1]).statistic
            continue
        out[n] = len(s) * s.mean() ** 2 / s.var(ddof=1)  # F of the within-subject main effect, df (1, n-1)
        out["Group x " + n] = stats.f_oneway(s[g == 0], s[g == 1]).statistic
    return out


def main():
    checks = stdlib.Checks()
    rows, cents, ex, tgt = build()
    a = pd.DataFrame(rows)
    ad = a[a.experiment == "Adaptation"]
    checks.published("participants who do not stutter", ad[ad.group == "adults who do not stutter"].person.nunique(), 12, 0, SRC + "Methods, Participants")
    checks.published("participants who stutter", ad[ad.group == "adults who stutter"].person.nunique(), 12, 0, SRC + "Methods, Participants")
    checks.published("trials per word in the adaptation task", int(ad.groupby("subject").size().iloc[0]), 80, 0,
                     SRC + "Methods (10 baseline, 60 perturbation, 10 after-effects trials per word)")
    checks.internal("perturbation phase has 60 trials per word", bool((ad[ad.pert_value == 400].groupby("subject").size() == 60).all()))
    nn = int(ad.f1_cents.eq("").sum())
    checks.internal("missing F1 values", nn == 4, f"{nn} missing")
    # repeated-measures ANOVA on extent of adaptation (Results, Auditory-motor adaptation)
    g = np.array([0] * 12 + [1] * 12)
    cells = np.column_stack([cents[w][r, :].mean(0) if False else np.nanmean(cents[w][rr, :], axis=0)
                             for rr in (slice(10, 15), slice(65, 70)) for w in ("bed", "pet")])
    F = anova_cells(cells, g)
    for k, rep in (("Group", 13.426), ("Group x Phase", 1.447), ("Group x Word", 0.198), ("Group x PhaseXWord", 1.105)):
        checks.published(f"repeated-measures ANOVA on extent of adaptation (early and late, bed and pet): {k}, F(1, 22)", float(F[k]), rep, 0.0005,
                         SRC + "Results, Auditory-motor adaptation")
    rep_t = {("baseline", 0): 0.506, ("baseline", 1): -0.641, ("early", 0): 1.399, ("early", 1): 1.053, ("late", 0): -0.665, ("late", 1): 0.524}
    for lab, rr in (("baseline", slice(5, 10)), ("early", slice(10, 15)), ("late", slice(65, 70))):
        rt = (ex[rr, :24].mean(0) + ex[rr, 24:].mean(0)) / 2
        for gi, gl in ((0, "adults who do not stutter"), (1, "adults who stutter")):
            checks.published(f"intent-to-change rating vs 0, {lab}, {gl}: t(11)", float(stats.ttest_1samp(rt[g == gi], 0).statistic),
                             rep_t[(lab, gi)], 0.0005, SRC + "Results, Explicit strategy use")
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s_ = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    pp = s_["papers"][0]
    meta = dict(
        study=ID, paper={k: pp[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s_["lab"], data_url=s_["data_url"],
        license="none stated", license_url="",
        changes=("Reshaped the authors' participant-by-trial text tables into one long table, joined the intent ratings to the adaptation "
                 "trials, added the feedback shift from the paper, and replaced the column positions with participant codes."),
        how=[
            "Experiment 'Adaptation': one row is one trial of one word (bed or pet) of one participant, trial 1-80 counted within the word "
            "(the files do not give the order in which the two words alternated). f1_cents is F1 in cents relative to the mean of baseline "
            "trials 6-10 of that word, as the authors normalized it; rejected trials (8.7% of bed, 11.8% of pet) were interpolated by the "
            "authors, and 4 bed trials of one participant are empty in the deposit. Trials 11-70 had F1 raised 400 cents.",
            "intent_rating is the slider answer after the same trial, as shared (fractions of the -100 to +100% slider; negative is toward "
            "the direction that opposes the shift, 'more like bid / pit').",
            "Experiment 'Auditory target': one row is one selection of the best-sounding F1 shift of a recording of the participant's own word "
            "(auditory_target_cents), 17 selections per word: 1-3 baseline, 4-15 during the perturbation, 16-17 after-effects.",
            "Participants 1-12 of every file are the adults who do not stutter, 13-24 the adults who stutter (readme); ages, sexes and "
            "stuttering severity are not in the deposit. The adults who stutter were tested at a conference venue, the others in a lab "
            "room without a sound booth; setting is 'lab' for both.",
        ],
        unit_note="trial (or target selection) of one word of one participant",
        differences=[
            "The ANOVA main effect of Phase (paper F = 18.417), of Word (0.149) and the Phase x Word interaction (1.565) came out as "
            "18.07, 0.155 and 1.558 from the shared trial values, so they were not used as checks; the Group effect, Group x Phase, "
            "Group x Word and the three-way interaction match to the printed digits.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
