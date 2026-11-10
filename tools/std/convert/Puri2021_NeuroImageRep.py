"""Puri, Hinder, Kruger & Summers (2021), Neuroimage: Reports 1(1):100009: timing of single-session anodal M1 tDCS (before, during or
after practice) and explicit serial reaction time task (SRTT) performance in 48 healthy older adults (each with an anodal and a sham
session), plus a pilot of 12 younger and 12 older adults on implicit and explicit SRTT versions (Supplementary 1).

Raw files, raw_extra/Puri2021_NeuroImageRep/ (OSF pk8cv, CC-By 4.0), from main_expt/ and supplementary_1/:
  demographics.csv, checks.csv (block 1 and sleep), acquisition.csv + acquisition_after.csv (blocks 2-8), retentions_and_offlineLearning.csv
  (blocks 9-13), errors.csv, supp_1.csv.
The TMS/EMG files (neuro_*.csv, MEP and SICI), tDCS sensation and awareness questionnaires (supp2.csv) and the by-repetition file
(supp_1b.csv) are not carried. Paper read through PubMed Central (PMC12172943, NCBI efetch).
Run from the site folder: python3 -I tools/std/convert/Puri2021_NeuroImageRep.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Puri2021_NeuroImageRep"
P = "Puri et al. 2021, "
R = stdlib.raw_dir(ID)
GROUPS = {"B": "tDCS before practice", "D": "tDCS during practice", "A": "tDCS after practice"}
STIM = {"A": "anodal tDCS", "S": "sham tDCS"}
LEARN = ("Sequence-specific learning of the explicit SRTT block: (mean random RT - mean sequence RT) / mean random RT, in percent "
         "(random RT from the 12 random trials before and after the sequence, sequence RT from the 120 sequence trials; higher is "
         "more sequence-specific improvement)")
STAGE = {"acq": "motor acquisition (blocks 2-8)", "ir": "immediate retention (blocks 9-10)", "dr": "delayed retention (blocks 11-13)"}


def rd(name):
    return pd.read_csv(R / name, encoding="utf-8-sig")


def base(code, grp_label, age, sex, cond, block, phase, trial, exp, group):
    return {"dataset": ID, "study": ID, "experiment": exp, "group": group, "subject": code, "subject_excluded_by_authors": False,
            "age": age, "sex": sex, "handedness": "", "population": "healthy older adults" if "older" in exp or "tDCS" in exp else "",
            "setting": "lab", "device": "keypad (four keys, one per finger of the dominant hand)", "session": 1, "block": block,
            "trial": trial, "phase": phase, "time_s": "", "task": "finger tapping and timing", "unit": "participant summary", "condition": cond,
            "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False}


def build():
    dem = rd("main_expt__1_demographics_and_checks__demographics.csv").set_index("Participant")
    chk = rd("main_expt__1_demographics_and_checks__checks.csv").set_index("Participant Code")
    acq = pd.concat([rd("main_expt__2_behaviour__acquisition.csv"), rd("main_expt__2_behaviour__acquisition_after.csv")]).set_index("Participant Code")
    ret = rd("main_expt__2_behaviour__retentions_and_offlineLearning.csv").set_index("Participant Code")
    err = rd("main_expt__2_behaviour__errors.csv").set_index("Participant")
    rows = []
    for code in dem.index:
        suffix = code.split("_")[1]
        g = GROUPS[suffix]
        age, sex = dem.loc[code, "Age"], "F" if dem.loc[code, "Gender"] == "W" else "M"
        n = 0
        for s in "AS":
            vals = [(1, chk.loc[code, f"{s}_Blk1"])] + [(b, acq.loc[code, f"{s}_Blk{b}"]) for b in range(2, 9)] + \
                   [(b, ret.loc[code, f"{s}_Blk{b}"]) for b in range(9, 14)]
            for b, v in vals:
                n += 1
                r = base(code, g, age, sex, STIM[s], f"block {b}", "practice" if b == 1 else ("learning" if b <= 8 else "test"), n,
                         "Explicit SRTT with tDCS timing", g)
                r.update(measure="sequence_specific_learning", measure_description=LEARN, statistic="value", value=v, value_unit="percent",
                         n_trials="", session_context="")
                rows.append(r)
        for s in "as":
            for st in STAGE:
                for kind, col in (("random", f"{s}_{st}_rand"), ("sequence", f"{s}_{st}_se" if st == "acq" else f"{s}_{st}_seq")):
                    n += 1
                    r = base(code, g, age, sex, STIM["A" if s == "a" else "S"], STAGE[st], "other", n, "Explicit SRTT with tDCS timing", g)
                    r.update(measure=f"error_rate_{kind}_trials", measure_description=f"Percent of {kind}-position key presses that were errors, over the blocks of the stage",
                             statistic="proportion", value=err.loc[code, col], value_unit="percent", n_trials="", session_context="")
                    rows.append(r)
    s1 = rd("supplementary_1__supp_1.csv")
    for c in s1.to_dict("records"):
        code, n = c["Participant_Code"], 0
        grp = "younger adults" if c["Age Group"] == "Y" else "older adults"
        for kind in ("Implicit", "Explicit"):
            for b in range(1, 11):
                n += 1
                r = base(code, grp, c["Age"], "F" if c["Gender"] == "W" else "M", f"{kind.lower()} SRTT", f"block {b}", "learning", n,
                         "Pilot: implicit and explicit SRTT (Supplementary 1)", grp)
                r.update(population=grp, handedness="right" if c["Handedness"] == "R" else "left", measure=f"{kind.lower()}_srtt_block_score",
                         measure_description="Block score of the pilot SRTT as given in supp_1.csv; the deposit does not define the unit, "
                         "but the values match the main experiment's sequence-specific learning percentage (random vs sequence RT)",
                         statistic="value", value=c[f"{kind}_Blk{b}"], value_unit="percent (inferred)", n_trials="", session_context="")
                rows.append(r)
    return dem, err, s1, rows


def main():
    checks = stdlib.Checks()
    dem, err, s1, rows = build()
    a = pd.DataFrame(rows)
    main_ = a[a.experiment == "Explicit SRTT with tDCS timing"]
    checks.published("participants completing both sessions", main_.subject.nunique(), 48, 0, P + "Participants (48 after one withdrawal)")
    for suf, nm, n, mean, sd, women in (("B", "BEFORE", 16, 65.6, 3.2, 11), ("D", "DURING", 16, 64.4, 3.9, 6), ("A", "AFTER", 16, 66.9, 6.0, 9)):
        g = dem[dem.index.str.endswith("_" + suf)]
        checks.published(f"{nm} group size", len(g), n, 0, P + "Results")
        checks.published(f"{nm} mean age", g.Age.mean(), mean, 0.05, P + "Results")
        checks.published(f"{nm} SD of age", g.Age.std(), sd, 0.05, P + "Results")
        checks.published(f"{nm} women", (g.Gender == "W").sum(), women, 0, P + "Results")
    for s, kind, rep in (("anodal tDCS", "random", 3.68), ("anodal tDCS", "sequence", 2.25), ("sham tDCS", "random", 4.10), ("sham tDCS", "sequence", 2.35)):
        sub = a[(a.measure == f"error_rate_{kind}_trials") & (a.condition == s)]
        checks.published(f"mean error rate, {kind} trials, {s} (%)", sub.groupby("block").value.mean().mean(), rep, 0.005, P + "Data processing (error rates)")
    checks.internal("13 learning blocks per participant and stimulation", (main_[main_.measure == "sequence_specific_learning"].groupby(["subject", "condition"]).size() == 13).all())
    checks.internal("pilot has 12 younger and 12 older participants with 20 block scores each", len(s1) == 24 and (a[a.experiment.str.startswith("Pilot")].groupby("subject").size() == 20).all())
    checks.internal("no missing values in the carried files", a.value.notna().all())
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = s["papers"][0]
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC-By Attribution 4.0 International",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Joined the authors' per-participant behaviour files into long form (one row per participant, stimulation, block and measure), "
                "added age and sex, and replaced participant codes. TMS/EMG measures, questionnaires and the pilot's by-repetition file were not carried.",
        how=["Each participant did an explicit serial reaction time task (12-element sequence, 4 keys) in two sessions at least a week apart, "
             "once with anodal and once with sham tDCS, applied before, during or after the acquisition blocks (groups of 16). The authors shared "
             "one value per participant, stimulation and block: sequence-specific learning in percent (positive: sequence trials faster than "
             "random trials). Blocks: 1 practice, 2-8 acquisition, 9-10 immediate retention, 11-13 delayed retention 24 h later. Error rates (percent) "
             "are given per stage for random and sequence trials. No single key presses were shared.",
             "Which session came first and which sequence (original or mirrored) was used is not in the files, so session is 1 for every row and "
             "stimulation is in condition. Two left-handed participants (AFTER group) are not identified in the files, so handedness is empty in the main experiment.",
             "The pilot (Supplementary 1: 12 younger and 12 older adults, implicit and explicit SRTT versions, 10 blocks each) is a separate experiment; its "
             "block scores are not defined in the deposit (assumed to be the same percentage) and the by-repetition file was not carried."],
        unit_note="participant, stimulation, block and measure",
        differences=["The paper's Bayesian ANOVA results were not recomputed. The error-rate means were checked as means over the participant stage values of each stimulation."])
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
