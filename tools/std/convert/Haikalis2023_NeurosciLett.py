"""Haikalis et al. (2023), Neurosci Lett 814:137442: placebo effects of tDCS on learning a spoon-and-beans transfer task,
68 participants x 30 trials of total trial time, in the 'game' module (one row per trial).

Run from the site folder: python3 -I tools/std/convert/Haikalis2023_NeurosciLett.py
Raw files (untrusted data, read only) in raw_extra/Haikalis2023_NeurosciLett/, from https://doi.org/10.5281/zenodo.8011365 (CC BY 4.0):
  AllGroups.csv (all 68), TDCSGroupsOnly.csv (the 40 stimulated participants, with belief questionnaires), README.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Haikalis2023_NeurosciLett"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
differences = []
A = pd.read_csv(RAW / "AllGroups.csv")
T = pd.read_csv(RAW / "TDCSGroupsOnly.csv")
checks.internal("AllGroups has 2040 rows, 68 participants x 30 trials", len(A) == 2040 and A.subid.nunique() == 68)
checks.internal("every participant has trials 1..30 in order", all(list(g.TrialNum) == list(range(1, 31)) for _, g in A.groupby("subid", sort=False)))
m = T.merge(A, on=["subid", "TrialNum"], suffixes=("_t", ""))
same = (m.Performance_t.fillna(-1) == m.Performance.fillna(-1)).all() and (m.AssignGrp_t == m.AssignGrp).all()
checks.internal("TDCSGroupsOnly trials are identical to the same trials in AllGroups", len(m) == len(T) and same)
for c in ("Age", "Sex", "AssignGrp", "RASex"):
    checks.internal(f"{c} constant within participant", (A.groupby("subid")[c].nunique() == 1).all())

tinfo = T.groupby("subid", sort=False).first()
rows = []
for sid, g in A.groupby("subid", sort=False):
    for _, r in g.iterrows():
        rows.append(dict(
            dataset=ID, study=ID, experiment="tDCS placebo", group=r.AssignGrp, subject=sid, subject_excluded_by_authors=False,
            age=r.Age, sex=r.Sex, handedness="right", population="healthy", setting="lab",
            device="spoon, beans and four cups; trial time by chronometer", session=1, block="practice", trial=int(r.TrialNum), time_s=None,
            phase="learning", task="sports and games", unit="trial", condition="", pert_type="none", pert_value=0, pert_unit="",
            feedback="normal", excluded_by_authors=False, game="Spoon bean transfer", attempt=int(r.TrialNum),
            duration_s=r.Performance if pd.notna(r.Performance) else None))
checks.internal("2 trials have no time in the deposit and are kept empty", sum(r["duration_s"] is None for r in rows) == 2)

P = A.groupby("subid", sort=False).first()
checks.published("participants", len(P), 68, 0, "Abstract / Methods")
for g, n in (("Ctrl", 28), ("Active", 20), ("Sham", 20)):
    checks.published(f"participants in {g}", (P.AssignGrp == g).sum(), n, 0, "Methods, Table 1")
for g, mu, sd in (("Ctrl", 21, 3.6), ("Sham", 21.25, 3.9), ("Active", 23.5, 4.5)):
    s = P[P.AssignGrp == g].Age
    checks.published(f"mean age {g}", s.mean(), mu, 0.05, "Table 1")
    checks.published(f"SD age {g}", s.std(), sd, 0.05, "Table 1")
Q = tinfo
checks.published("belief drop pre to post (points)", (Q.PostExp - Q.PreExp).mean(), -1.23, 0.005, "Results, belief change (t(39) = 3.44)")
checks.published("participants in belief analysis (df 39 + 1)", len(Q), 40, 0, "Results, belief change t(39)")
differences.append("The paper's Methods text says 29 participants were female, but its Table 1 counts add to 35 female and 33 male; "
                   "the deposit has 35 female and 33 male, which matches the table. Not used as a check.")
differences.append("The Methods give age for all participants as 22.3 +/- 4.14 years; the 68 participants in the deposit give 21.81 +/- 4.09 (the three group means and SDs in Table 1 do reproduce). Not used as a check.")
differences.append("Two of the 2,040 trials have no trial time in the deposit and are kept with an empty duration_s.")
differences.append("The paper does not state the unit of trial time in its Methods; seconds is taken from its belief-model section "
                   "and the deposit README (Total trial time in seconds).")
checks.internal("sex counts match Table 1 (35 F)", (P.Sex == "F").sum() == 35)

parts = []
for sid, r in P.iterrows():
    d = dict(person=sid, research_assistant_sex=r.RASex)
    if sid in tinfo.index:
        t = tinfo.loc[sid]
        d.update(expectancy_pre=t.PreExp, expectancy_post=t.PostExp, suggestibility=t.Suggestibility,
                 prior_knowledge_tdcs=t.PriorKnow, prior_tdcs_study=t.PriorStudy)
    parts.append(d)
pcols = [("research_assistant_sex", "category", "M, F", "Sex of the research assistant who ran the participant (authors' RASex)."),
         ("expectancy_pre", "number", "scale points", "Expectancy of tDCS benefit before practice (authors' PreExp); only the 40 stimulated participants."),
         ("expectancy_post", "number", "scale points", "Expectancy after practice (authors' PostExp); only the 40 stimulated participants."),
         ("suggestibility", "number", "score", "Suggestibility score (authors' Suggestibility); only the 40 stimulated participants."),
         ("prior_knowledge_tdcs", "category", "Yes, No", "Prior knowledge of tDCS (authors' PriorKnow); only stimulated participants."),
         ("prior_tdcs_study", "category", "Yes, No", "Earlier participation in a tDCS study (authors' PriorStudy); only stimulated participants.")]

meta = dict(
    paper=dict(authors="Haikalis NK, Hooyman A, Wang P, Daliri A, Schaefer SY", year=2023,
               title="Placebo effects of transcranial direct current stimulation on motor skill acquisition",
               venue="Neuroscience Letters 814:137442", doi="10.1016/j.neulet.2023.137442"),
    lab="Schaefer (Arizona State University)", data_url="https://doi.org/10.5281/zenodo.8011365", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Long-format rows kept in the authors' order; participant codes replaced; column names mapped; belief and prior-knowledge "
            "scores moved to participants.csv.",
    how=["One row is one trial of the spoon task: the non-dominant (left) hand moves two beans at a time with a spoon from a home cup "
         "to three cups in left, middle, right order until the home cup is empty (15 reaches), 30 trials per participant.",
         "duration_s is the total trial time in seconds (timed by hand with a chronometer); lower is faster. Groups are Active tDCS, "
         "Sham tDCS and Ctrl (no stimulation). Stimulation was applied to right primary motor cortex; all participants were right-handed "
         "(handedness filled from the paper).",
         "Sex and age are as in the deposit. Expectancy, suggestibility and prior-knowledge scores exist only for the stimulated groups."],
    unit_note="trial", differences=differences)
stdlib.release(ID, "game", rows, meta, checks, participants=parts, participant_columns=pcols)
