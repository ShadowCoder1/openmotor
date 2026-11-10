"""Myers, Olson & Scapetis-Tycer (2024), Open Mind 8:1084-1106: individual differences in accent imitation. 92 adults imitated three
accented talkers (Yorkshire, Edinburgh, Eastern Cape) and did a maximal syllable repetition (DDK) task; independent online raters judged
how similar each imitation was to the target accent (1 to 7).

Raw files, raw_extra/Myers2024_OpenMind/ (OSF 8erq3): STEAM_Data_PenPaper_Only.csv (one row per imitator: age, sex, DDK counts, questionnaires),
all.ratings.csv (one row per rater judgement of an imitation), AccentImitation.Rmd (the authors' manuscript source, read as text), README.
steam_alldata_vas.csv (perception task) and the per-participant workbooks were not used. Paper read in PubMed Central (PMC11370968).
Run from the site folder: python3 -I tools/std/convert/Myers2024_OpenMind.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Myers2024_OpenMind"
P = "Myers et al. 2024, "
ACC = {"England": "Yorkshire (England) accent", "Scotland": "Edinburgh (Scotland) accent", "SouthAfrica": "Eastern Cape (South Africa) accent"}
DDK = [("DDK_P", "/pa/"), ("DDK_T", "/ta/"), ("DDK_K", "/ka/"), ("DDK_PTK", "/pataka/")]


def build():
    r = stdlib.raw_dir(ID)
    s = pd.read_csv(r / "STEAM_Data_PenPaper_Only.csv", encoding="utf-8-sig")
    rt = pd.read_csv(r / "all.ratings.csv", encoding="utf-8-sig")
    g = rt.groupby(["imitator", "accent"]).rating.agg(["mean", "size"]).reset_index()
    rows = []
    for c in s.to_dict("records"):
        def add(cond, measure, desc, stat, val, unit, ntr="", phase="test"):
            if pd.isna(val):
                return
            rows.append({
                "dataset": ID, "study": ID, "experiment": "Accent imitation and individual differences", "group": "",
                "subject": str(c["Participant"]), "subject_excluded_by_authors": False, "age": c["Age"],
                "sex": "M" if c["GenderM1F2"] == 1 else "F", "handedness": "", "population": "healthy", "setting": "lab",
                "device": "microphone", "session": 1, "block": "", "trial": 0, "phase": phase, "time_s": "", "task": "speech",
                "unit": "participant summary", "condition": cond, "pert_type": "none", "pert_value": 0.0, "pert_unit": "",
                "feedback": "normal", "excluded_by_authors": False, "measure": measure, "measure_description": desc,
                "statistic": stat, "value": float(val), "value_unit": unit, "n_trials": ntr, "session_context": ""})

        for col, syl in DDK:
            add("repeat %s as fast as possible for 5 s" % syl, "ddk_repetitions",
                "Number of repetitions of the syllable (or of /pataka/) produced in five seconds, scored off-line from the recording (the authors' %s); higher is faster" % col,
                "count", c[col], "repetitions in 5 s")
        for _, q in g[g.imitator == c["imitator"]].iterrows():
            add(ACC[q["accent"]], "imitation_rating",
                "Mean similarity rating (1 least to 7 most like the target accent) that independent online raters gave the participant's imitations of the accent, over all phrases and raters",
                "mean", q["mean"], "rating (1 to 7)", int(q["size"]))
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, s, rt


def main():
    checks = stdlib.Checks()
    a, s, rt = build()
    checks.published("imitators", len(s), 92, 0, P + "Methods, Participants (n = 92)")
    checks.published("mean age (years)", s.Age.mean(), 20.86, 0.005, P + "Methods, Participants (mean = 20.86)")
    checks.published("SD of age (years)", s.Age.std(), 6.84, 0.005, P + "Methods, Participants (SD = 6.84)")
    checks.published("youngest age", s.Age.min(), 18, 0, P + "Methods, Participants (age range 18-58)")
    checks.published("oldest age", s.Age.max(), 58, 0, P + "Methods, Participants (age range 18-58)")
    checks.published("women", (s.GenderM1F2 == 2).sum(), 60, 0, P + "Methods, Participants (60 women)")
    checks.published("men", (s.GenderM1F2 == 1).sum(), 32, 0, P + "Methods, Participants (32 men)")
    checks.published("professional actors", (s.Expert1 == 1).sum(), 6, 0, P + "Methods, Participants (6 professional actors)")
    checks.published("musicians", (s.Musician1 == 1).sum(), 45, 0, P + "Methods, Participants (45 musicians)")
    checks.published("monolingual participants", (s.Monolingual1Multi2 == 1).sum(), 22, 0, P + "Methods, Participants (22 monolingual)")
    checks.internal("every imitator has 3 accent ratings", (a[a.measure == "imitation_rating"].groupby("subject").size() == 3).all())
    checks.internal("92 persons", a.subject.nunique() == 92)
    checks.internal("ratings within 1-7", a[a.measure == "imitation_rating"].value.between(1, 7).all())
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s0 = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s0["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s0["lab"], data_url=s0["data_url"], license="none stated", license_url="",
        changes=("Took the speech-production measures from the authors' files: the participant table gave age, sex and the four syllable repetition counts, "
                 "and the 47,223 rater judgements were averaged per imitator and accent. "
                 "Personality, vocabulary, music-perception and perception-task variables, and the raw rater trials, were not carried. Participant codes were replaced."),
        how=[
            "Each row is one measure of one participant: the number of repetitions of /pa/, /ta/, /ka/ or /pataka/ produced as fast as possible in five seconds "
            "(a maximal speech rate, scored by the authors from the recordings), or the mean rating of the participant's imitations of one of three accents.",
            "imitation_rating is the mean over all raters and phrases of the 1 to 7 similarity ratings (7 most like the target accent) in the authors' all.ratings.csv, "
            "with the number of judgements in n_trials; the authors' own analysis modelled single ratings, so this per-accent mean is a summary of theirs, not their model input.",
            "Sex is coded from the authors' GenderM1F2 column (1 = male, 2 = female), confirmed by the paper's 60 women and 32 men. One participant has no /pa/ count.",
        ],
        unit_note="participant, measure and accent or syllable",
        differences=["The paper's mixed-model and correlation results were not recomputed. The OSF node states no licence."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
