"""Smith, Dokovova, Lawson, Kuschmann & Cleland (2023), Proc 20th ICPhS: tongue shape variability (mean nearest neighbour distance,
NND, between ultrasound tongue splines of repeated /aCa/ productions) in 11 typically developing children recorded at a museum event
and 10 age-matched children with speech sound disorder (SSD).

Raw files, raw_extra/Smith2023_ICPhS/ (OSF 3byvp): icphs2023.csv (one row per child and consonant), icphs2023_OSF.R (the authors' analysis, read as text).
Paper read as the PDF on the OSF node. Run from the site folder: python3 -I tools/std/convert/Smith2023_ICPhS.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Smith2023_ICPhS"
P = "Smith et al. 2023, "
CONS = {"t": "/t/", "k": "/k/", "r": "/ɹ/", "s": "/s/"}


def build():
    d = pd.read_csv(stdlib.raw_dir(ID) / "icphs2023.csv", index_col=0)
    rows = []
    for c in d.to_dict("records"):
        ssd = c["developmental_conditions"] == "SSD"

        def add(measure, desc, stat, val, unit):
            if pd.isna(val):
                return
            rows.append({
                "dataset": ID, "study": ID, "experiment": "Tongue shape variability", "group": c["developmental_conditions"],
                "subject": str(c["id"]), "subject_excluded_by_authors": False, "age": c["age_y"], "sex": c["sex"], "handedness": "",
                "population": "speech sound disorder" if ssd else "healthy", "setting": "clinic" if ssd else "lab",
                "device": "ultrasound probe on a headset, 5-8 MHz microconvex, about 100 fps, with microphone",
                "session": 1, "block": "", "trial": 0, "phase": "test", "time_s": "", "task": "speech",
                "unit": "participant summary", "condition": CONS[c["consonant"]] + " in /aCa/", "pert_type": "none",
                "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False, "measure": measure,
                "measure_description": desc, "statistic": stat, "value": float(val), "value_unit": unit, "n_trials": "",
                "session_context": ("recorded at a public event in a museum" if not ssd else "recorded in a clinic (UltraSuite corpus)")
                + ("; minimum spline tracking confidence %d" % c["min_confidence"] if pd.notna(c["min_confidence"]) else "")})

        add("nnd_mean", "Mean nearest neighbour distance (NND) between the tongue spline of each repetition of the consonant and the others, "
            "averaged over the child's repetitions (up to 10); larger means more variable tongue shapes (the authors' mean_NND_cm)", "mean", c["mean_NND_cm"], "cm")
        add("nnd_sd", "Standard deviation of the NND over the child's repetitions of the consonant (the authors' sd_NND)", "sd", c["sd_NND"], "mm")
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, d


def main():
    checks = stdlib.Checks()
    a, d = build()
    p = d.drop_duplicates("id")
    td, ssd = p[p.developmental_conditions == "TD"], p[p.developmental_conditions == "SSD"]
    checks.published("typically developing children", len(td), 11, 0, P + "4.1 (leaving 11 TD participants)")
    checks.published("TD girls", (td.sex == "F").sum(), 5, 0, P + "4.1 (11 TD participants, 5 female)")
    checks.published("children with SSD", len(ssd), 10, 0, P + "3.1 (10 children with SSD)")
    checks.published("girls with SSD", (ssd.sex == "F").sum(), 2, 0, P + "3.1 (10 children with SSD, 2 female)")
    checks.published("TD mean age (years)", td.age_y.mean(), 9, 0.5, P + "4.1 (mean age 9 years, printed as an integer)")
    checks.published("TD youngest age", td.age_y.min(), 7, 0, P + "4.1 (range 7-14)")
    checks.published("TD oldest age", td.age_y.max(), 14, 0, P + "4.1 (range 7-14)")
    t = d[d.developmental_conditions == "TD"].copy()
    t["ag"] = pd.cut(t.age_y, [0, 8, 11, 15], labels=["7-8", "9-11", "14"])
    for ag, cons, rep in (("7-8", "k", 0.10), ("9-11", "k", 0.07), ("9-11", "r", 0.07), ("9-11", "s", 0.05), ("14", "k", 0.07),
                          ("14", "r", 0.07), ("14", "s", 0.07), ("14", "t", 0.05)):
        checks.published("mean NND of age group %s, %s (cm)" % (ag, CONS[cons]), t[(t.ag == ag) & (t.consonant == cons)].mean_NND_cm.mean(),
                         rep, 0.005, P + "4.1, Table 1 (printed to two decimals)")
    for ag, cons, rep in (("7-8", "k", 4), ("7-8", "t", 5), ("9-11", "s", 5)):
        checks.published("participants in age group %s with %s" % (ag, CONS[cons]), t[(t.ag == ag) & (t.consonant == cons)].id.nunique(), rep, 0,
                         P + "4.1, Table 1 (n participants)")
    checks.internal("21 children", a.subject.nunique() == 21)
    checks.internal("NND is positive", (a[a.measure == "nnd_mean"].value > 0).all())
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    paper["authors"] = "Smith A, Dokovova M, Lawson E, Kuschmann A, Cleland J"
    meta = dict(
        study=ID, paper=paper, lab="Strathclyde (Speech and Language Therapy, University of Strathclyde)", data_url=s["data_url"],
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Reshaped the authors' table into one long table (the mean and the SD of NND as separate rows), named the consonants in IPA, and replaced child codes.",
        how=[
            "Children produced up to 10 repetitions of /t/, /k/, /r/ and /s/ in /aCa/ while an ultrasound probe imaged the tongue; the authors fitted a tongue "
            "spline at the maximal gesture of each repetition and measured the mean nearest neighbour distance (NND) between splines. The deposit has one "
            "value per child and consonant, so each row is the child's NND mean (cm) or SD of NND for one consonant.",
            "nnd_sd is the authors' sd_NND; its unit (mm) is inferred, because the authors' mean_NND column is exactly ten times mean_NND_cm. The class of the "
            "children is in group: TD typically developing (recorded at a museum event, setting recorded as lab because the field has no event category) and SSD speech sound disorder (clinic).",
            "The authors' 'usability' flag is yes on every row and their min_confidence column is 80 where it is filled (it is empty on some rows, noted in session_context only when given); the TD child aged 14 has no age-matched "
            "SSD child, and the authors repeated the group comparison without that child.",
        ],
        unit_note="child and consonant",
        differences=["Table 1 prints mean NND per age group and consonant; from the shared rows /ɹ/ and /t/ at 7-8 years (printed 0.10 and 0.08; rows give 0.083 and 0.068) and /t/ at 9-11 years (printed 0.07; rows give 0.060) differ, and /s/ at 7-8 years (0.07 printed, 0.065 here) is at the edge of rounding, so these cells are not used as checks. The model coefficients (age slope, diagnosis effect) were not recomputed."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
