"""Humiston & Wamsley (2018), Neurobiol Learn Mem 155:1-6: motor sequence task (typing a 5-digit sequence with the non-dominant hand,
12 training and 12 test trials of 30 s) before and after 15 min of eyes-closed rest or a distractor game (Snood), tested immediately
(Study 1) or 4 hours later (Study 2), in 49 undergraduates (35 in the paper's analyses).

Raw files, raw_extra/Humiston2018_NeurobiolLearnMem/ (OSF c4zmt): combined.sav (Combined data Study 1 and 2.sav: one row per
participant, plus a blank first row), syntax.sps and excl.sps (the authors' SPSS syntax, read as text only), manual1.docx (not used).
The EEG (.edf) files, the sleep questionnaires, the Epworth and sleepiness scales, the thought-content questionnaires and the
EEG band powers in the SPSS file are not carried. Paper read in PubMed Central (PMC6281812).
Run from the site folder: python3 -I tools/std/convert/Humiston2018_NeurobiolLearnMem.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyreadstat
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Humiston2018_NeurobiolLearnMem"
P = "Humiston & Wamsley 2018, "
SESSION_CTX = {1: "Study 1: test trials immediately after the 15 min retention interval",
               2: "Study 2: test trials 4 hours after the 15 min retention interval"}
D_TRIAL = "Number of times the 5-digit sequence was typed correctly in the 30 s trial, as the authors' column (values can be fractional; the deposit does not say why)"


def build():
    d, meta = pyreadstat.read_sav(str(stdlib.raw_dir(ID) / "combined.sav"))
    d = d[d["Participant"].notna()].reset_index(drop=True)  # the file starts with one blank row
    rows = []
    for c in d.to_dict("records"):
        st = int(c["Study"])
        excl = int(c["filter_$"]) != 1
        base = {
            "dataset": ID, "study": ID, "experiment": "Study 1 (immediate test)" if st == 1 else "Study 2 (test after 4 hours)", "group": "",
            "subject": f"S{st}_{int(c['Participant'])}", "subject_excluded_by_authors": excl, "age": c["Age"], "sex": "M" if c["Sex"] == 1 else "F",
            "handedness": {1: "right", 2: "left"}.get(c["DominantHand"], ""), "population": "healthy", "setting": "lab",
            "device": "computer keyboard, non-dominant hand", "session": 1, "time_s": "", "task": "sequence learning", "unit": "participant summary",
            "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": excl}
        order = ["rest", "snood"] if c["condition.order"] == 1 else ["snood", "rest"]
        for cond in order:
            for part, tag in (("training", "I"), ("test", "F")):
                for t in range(1, 13):
                    v = c[f"MST{tag}{cond}{t}"]
                    if pd.isna(v):
                        continue
                    rows.append({**base, "block": f"{cond} {part}", "phase": "learning" if part == "training" else "test", "condition": cond,
                                 "measure": f"correct_sequences_{part}_trial", "measure_description": D_TRIAL, "statistic": "count",
                                 "value": v, "value_unit": "sequences per 30 s trial", "n_trials": "",
                                 "session_context": SESSION_CTX[st] + f"; MST trial {t} of 12 of the {part} block; " + ("rest condition first" if order[0] == "rest" else "Snood condition first"), "trial": 0})
            for kind, col, unit in (("raw_change", f"{cond.capitalize()}.raw.change", "sequences per 30 s trial"),
                                    ("percent_change", f"{cond.capitalize()}.percent.change", "proportion of the last 3 training trials")):
                v = c[col]
                if pd.isna(v):
                    continue
                rows.append({**base, "block": f"{cond} change", "phase": "other", "condition": cond, "measure": f"consolidation_{kind}",
                             "measure_description": ("Offline gain: mean of the first 3 test trials minus mean of the last 3 training trials" if kind == "raw_change" else
                                                     "Offline gain relative to training: the raw change divided by the mean of the last 3 training trials") + " (the authors' " + col + ")",
                             "statistic": "value", "value": v, "value_unit": unit, "n_trials": 3, "session_context": SESSION_CTX[st], "trial": 0})
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, d


def main():
    checks = stdlib.Checks()
    a, d = build()
    k = d[d["filter_$"] == 1]
    s1, s2 = k[k.Study == 1], k[k.Study == 2]
    checks.published("Study 1 participants analysed", len(s1), 18, 0, P + "Participants / Table 1 (n = 18)")
    checks.published("Study 2 participants analysed", len(s2), 17, 0, P + "Participants / Table 1 (n = 17)")
    checks.published("Study 1 men", (s1.Sex == 1).sum(), 7, 0, P + "Participants (7 male)")
    checks.published("Study 2 men", (s2.Sex == 1).sum(), 4, 0, P + "Participants (4 male)")
    checks.published("Study 1 mean age", s1.Age.mean(), 20.50, 0.005, P + "Participants / Table 2")
    checks.published("Study 1 SD age", s1.Age.std(), 1.10, 0.005, P + "Participants / Table 2")
    checks.published("Study 2 mean age", s2.Age.mean(), 19.29, 0.005, P + "Participants / Table 2")
    checks.published("Study 2 SD age", s2.Age.std(), 1.31, 0.005, P + "Participants / Table 2")
    for nm, g, (sr, ss, rr, sr_sd, rp, sp, rp_sd, sp_sd) in (("immediate", s1, (6.6, 4.6, 2.5, 3.4, 31, 22, 14, 17)), ("4hr delay", s2, (1.3, 1.4, 3.0, 2.5, 7.7, 7.4, 14, 13))):
        rest_raw, snood_raw, rest_sd, snood_sd, rest_p, snood_p, rest_psd, snood_psd = sr, ss, rr, sr_sd, rp, sp, rp_sd, sp_sd
        checks.published(f"{nm}: rest raw change mean", g["Rest.raw.change"].mean(), rest_raw, 0.05, P + "Table 1")
        checks.published(f"{nm}: rest raw change SD", g["Rest.raw.change"].std(), rest_sd, 0.05, P + "Table 1")
        checks.published(f"{nm}: distractor raw change mean", g["Snood.raw.change"].mean(), snood_raw, 0.05, P + "Table 1")
        checks.published(f"{nm}: distractor raw change SD", g["Snood.raw.change"].std(), snood_sd, 0.05, P + "Table 1")
        checks.published(f"{nm}: rest percent change mean (%)", g["Rest.percent.change"].mean() * 100, rest_p, 0.5, P + "Table 1")
        checks.published(f"{nm}: distractor percent change mean (%)", g["Snood.percent.change"].mean() * 100, snood_p, 0.5, P + "Table 1")
        checks.published(f"{nm}: rest percent change SD (%)", g["Rest.percent.change"].std() * 100, rest_psd, 0.5, P + "Table 1")
        checks.published(f"{nm}: distractor percent change SD (%)", g["Snood.percent.change"].std() * 100, snood_psd, 0.5, P + "Table 1")
    t = stats.ttest_rel(s1["Rest.raw.change"], s1["Snood.raw.change"])
    checks.published("immediate: paired t, rest vs distractor raw change", t.statistic, 2.28, 0.005, P + "Results (t(17) = 2.28)")
    t = stats.ttest_rel(s1["Rest.percent.change"], s1["Snood.percent.change"])
    checks.published("immediate: paired t, rest vs distractor percent change", t.statistic, 2.17, 0.005, P + "Results (t(17) = 2.17)")
    # internal: the authors' change scores follow from the trial values
    import numpy as np
    for cond in ("rest", "snood"):
        tr = d[[f"MSTI{cond}{i}" for i in (10, 11, 12)]].mean(axis=1)
        te = d[[f"MSTF{cond}{i}" for i in (1, 2, 3)]].mean(axis=1)
        checks.internal(f"{cond} raw change = first 3 test minus last 3 training trials", bool(np.allclose((te - tr).fillna(0), d[f"{cond.capitalize()}.raw.change"].fillna(0))))
    checks.internal("49 participants, 14 excluded by the authors' criteria", a.subject.nunique() == 49 and a[a.subject_excluded_by_authors].subject.nunique() == 14)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {q: p0[q] for q in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Reshaped the authors' wide SPSS table into a long table (the 48 trial values and the four change scores per participant), "
                 "added age, sex and handedness, dropped the blank first row, and replaced participant numbers with codes. The EEG recordings, "
                 "EEG band powers, sleep, sleepiness and thought-content questionnaires, and instrument history are not carried."),
        how=[
            "One row is one motor sequence task (MST) trial value, or one change score, of one participant. In the MST the participant types a "
            "5-digit sequence (41324 or 23142, one per condition) with the non-dominant hand for 30 s, followed by 30 s rest, for 12 training "
            "trials; then 15 min of eyes-closed rest or a distractor game (Snood), in counterbalanced order on one day; then 12 test trials. "
            "Study 1 tested immediately after the interval, Study 2 4 hours later. condition is rest or snood; rows are in the order the "
            "participant did the two conditions.",
            "The trial value is the authors' number of correct sequences in the trial (it can be fractional; the deposit does not explain "
            "this). consolidation_raw_change and consolidation_percent_change are the authors' own columns; the rows show they equal the mean of the "
            "first 3 test trials minus the mean of the last 3 training trials (and that divided by the training mean). The paper's text phrases it "
            "differently; the data were checked against the rows.",
            "excluded_by_authors marks participants outside the paper's sample: those with 2 or more minutes of stage N2 sleep or more than "
            "5 min of N1 sleep during the rest (per the authors' exclusion syntax), and those with a sleepiness rating above 4 before the rest. 35 of the 49 "
            "are in the paper's analyses (18 immediate, 17 delayed). Sex is coded 1 male and 2 female and handedness 1 right and 2 left in the SPSS value labels.",
        ],
        unit_note="participant, condition and trial or change score",
        differences=["The paper's delayed-group paired t of rest against distractor raw change, t(16) = .13, gives 0.125 here (it would round to .12), so it is not used as a check.",
                     "The paper's text and Table 1 give 17 participants in the delayed group and Table 2 gives 18; the data have 17 (the Table 1 value is used).",
                     "Epworth sleepiness, baseline MST (pre_interval) and EEG numbers of Table 2 are not carried, so they were not checked."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
