"""Messing & Apthorp (2023), PeerJ 11:e15406: online finger tapping task (space bar, index finger, taps in 10 s, dominant and
non-dominant hand) and the AQ-10 autism-trait screen in 159 adults aged 18 to 78.

Raw files, raw_extra/Messing2023_PeerJ/ (OSF mqyg9): FTT_AQ_data.csv (one row per participant), fingertapping.iqx (Inquisit task
code, read as text only), Reproducible_Results.Rmd and Import_Match_Inquisit_Qualtrics.R (analysis scripts, read as text only).
Paper read in PubMed Central (PMC10198151).
Run from the site folder: python3 -I tools/std/convert/Messing2023_PeerJ.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Messing2023_PeerJ"
P = "Messing & Apthorp 2023, "
DESC_TAP = ("Final tapping score of the hand: number of space bar taps in a 10 s block, as the Inquisit task computed it (mean of "
            "the five blocks whose scores lay within 5 taps of each other, or of all 10 blocks if no such five existed)")


def build():
    d = pd.read_csv(stdlib.raw_dir(ID) / "FTT_AQ_data.csv")
    rows = []
    for i, c in enumerate(d.to_dict("records")):
        excl = c["values.score_dominant"] <= 3  # the authors' script drops participants with 3 or fewer taps (too few taps)
        adhd, neuro = int(c["ADD/ADHD"]) == 1, int(c["Neurological Condition"]) == 1
        pop = "healthy" if not (adhd or neuro) else " and ".join(x for x, f in (("ADD/ADHD", adhd), ("neurological condition", neuro)) if f)
        base = {
            "dataset": ID, "study": ID, "experiment": "Online finger tapping and AQ-10", "group": "", "subject": f"row{i + 1}",
            "subject_excluded_by_authors": bool(excl), "age": c["Age"], "sex": {1: "F", 2: "M"}[int(c["Gender"])], "handedness": "",
            "population": pop, "setting": "online", "device": f"computer ({c['computer.platform']}), space bar", "session": 1, "block": "",
            "phase": "test", "time_s": "", "task": "finger tapping and timing", "unit": "participant summary",
            "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": bool(excl)}

        def add(measure, desc, stat, val, unit, cond, n=None):
            rows.append({**base, "condition": cond, "measure": measure, "measure_description": desc, "statistic": stat, "value": val,
                         "value_unit": unit, "n_trials": n if n is not None else "",
                         "session_context": "", "trial": 0})
        add("tapping_score", DESC_TAP, "mean", c["values.score_dominant"], "taps per 10 s", "dominant hand")
        add("tapping_score", DESC_TAP, "mean", c["values.score_nondominant"], "taps per 10 s", "non-dominant hand")
        add("aq10_score", "AQ-10 total score (0 to 10; one point per item answered in the autistic-trait direction, items 2, 3, 4, 5, 6 and 9 reverse-keyed), as in the authors' file",
            "value", c["AQ-10 score"], "score", "")
        for k in range(1, 11):
            add(f"aq10_item_{k}", f"Answer to AQ-10 item {k} on the four-point scale, coded 1 to 4 as in the authors' file (before reverse-keying)",
                "value", c[f"AQ{k}"], "rating 1-4", "")
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, d


def main():
    checks = stdlib.Checks()
    a, d = build()
    k = d[d["values.score_dominant"] > 3]
    checks.published("participants completing all tasks", len(d), 159, 0, P + "Participants (159 completed)")
    checks.published("participants analysed", len(k), 158, 0, P + "Participants / Table 1 (n = 158)")
    checks.published("women among analysed", (k.Gender == 1).sum(), 111, 0, P + "Table 1 (111 female)")
    checks.published("men among analysed", (k.Gender == 2).sum(), 47, 0, P + "Table 1 (47 male)")
    checks.published("women among all 159", (d.Gender == 1).sum(), 112, 0, P + "Participants (112 female)")
    checks.published("mean age, all 159 (years)", d.Age.mean(), 35.42, 0.005, P + "Participants (M = 35.42)")
    checks.published("mean age, analysed (years)", k.Age.mean(), 35.41, 0.005, P + "Table 1")
    checks.published("SD of age, analysed", k.Age.std(), 14.54, 0.005, P + "Table 1")
    checks.published("minimum age", d.Age.min(), 18, 0, P + "Participants (aged 18 to 78)")
    checks.published("maximum age", d.Age.max(), 78, 0, P + "Participants (aged 18 to 78)")
    checks.published("dominant hand score, mean", k["values.score_dominant"].mean(), 66.38, 0.005, P + "Table 1")
    checks.published("dominant hand score, SD", k["values.score_dominant"].std(), 9.75, 0.005, P + "Table 1")
    checks.published("non-dominant hand score, mean", k["values.score_nondominant"].mean(), 61.07, 0.005, P + "Table 1")
    checks.published("non-dominant hand score, SD", k["values.score_nondominant"].std(), 8.32, 0.005, P + "Table 1")
    checks.published("AQ-10 score, mean", k["AQ-10 score"].mean(), 3.08, 0.005, P + "Table 1")
    checks.published("AQ-10 score, SD", k["AQ-10 score"].std(), 1.82, 0.005, P + "Table 1")
    checks.published("dominant hand score, men, mean", k[k.Gender == 2]["values.score_dominant"].mean(), 70.91, 0.005, P + "Table 1")
    checks.published("dominant hand score, women, mean", k[k.Gender == 1]["values.score_dominant"].mean(), 64.46, 0.005, P + "Table 1")
    checks.internal("one excluded participant", int(a.groupby("subject").excluded_by_authors.first().sum()) == 1)
    checks.internal("13 rows per participant", (a.groupby("subject").size() == 13).all())
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {q: p0[q] for q in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Reshaped the authors' wide participant table into a long table (tapping scores of the two hands, AQ-10 total, ten AQ-10 "
                 "items), dropped the Qualtrics response code and the two session durations, recoded gender to F and M, and marked the "
                 "participants the authors excluded."),
        how=[
            "One row is one measure of one participant. The tapping task asked for as many index finger taps on the space bar as "
            "possible in 10 s blocks, dominant hand and then non-dominant hand (the paper does not say which hand went first); the "
            "tapping score is the authors' final score per hand, in taps per 10 s. The task ran online in the participant's browser on "
            "their own computer; the device column names the operating system the task recorded.",
            "excluded_by_authors marks the one participant the authors' script removed because the dominant hand score was 3 taps or "
            "fewer (paper: scores of 2.0 and 2.2); the other 158 are the analysed sample. Gender is coded 1 (female) and 2 (male) in "
            "the authors' file, as the paper states for its attrition test. Handedness is not recorded per participant (the task asks "
            "about the dominant hand only). population says 'healthy' unless the participant reported ADD/ADHD or a neurological condition.",
            "The AQ-10 items are the authors' four-point answers, coded 1 to 4, before the reverse-keying that gives the total score; "
            "which end of the scale is 1 is inferred (not stated in the deposit), so use the total score for analysis.",
        ],
        unit_note="participant and measure",
        differences=["The paper's text gives the non-dominant hand SD as 9.75, the same as the dominant hand; its Table 1 gives 8.32, which the rows reproduce, so Table 1 is the check.",
                     "The deposit holds 159 rows with 158 unique Qualtrics codes (one code appears twice); both rows are kept as two participants, as the authors' count of 159 does."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
