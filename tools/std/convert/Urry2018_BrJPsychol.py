"""Urry, Burns & Baetu (2018), Br J Psychol: sequence learning in 43 younger and 35 older adults on a serial reaction time
task (SRTT) and a predictive sequence learning task (PSLT). The deposit has one row per participant with the authors' learning
scores, so the release is the 'summary' format (one row per participant and measure); the cognitive test scores go to
participants.csv.

Raw file (untrusted data, read only): raw_extra/Urry2018_BrJPsychol/data.csv from https://osf.io/vhk3m/.
Measure definitions: paper Table 2 (accepted manuscript at the University of Adelaide repository, hdl 2440/124019).
Run from the site folder: python3 -I tools/std/convert/Urry2018_BrJPsychol.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Urry2018_BrJPsychol"
raw = pd.read_csv(stdlib.raw_dir(ID) / "data.csv")
ck = stdlib.Checks()
n = len(raw)

MEASURES = [  # (column, task, measure, description, unit)
    ("SRTT_TotalLearning", "SRTT", "total_learning", "RT in block 1 minus RT in block 6 (first and last practice blocks before the random block), all trials", "ms"),
    ("SRTT_SequenceLearning", "SRTT", "sequence_learning", "RT in random block 7 minus RT in sequence block 8, all trials", "ms"),
    ("SRTT_RatioRT", "SRTT", "ratio_rt", "(RT block 7 - RT block 8) / RT block 7", "ratio"),
    ("PSLT_TotalLearning", "PSLT", "total_learning", "RT in block 1 minus RT in block 6, all trials (RT = time to respond after the feedback of the previous response ended)", "ms"),
    ("PSLT_SequenceLearning", "PSLT", "sequence_learning", "RT in random block 7 minus RT in sequence block 8, all trials", "ms"),
    ("PSLT_RatioRT", "PSLT", "ratio_rt", "(RT block 7 - RT block 8) / RT block 7", "ratio"),
    ("PSLT_MeanErrorScore", "PSLT", "mean_error_score", "Mean distance between the clicked and the correct location over all sequence blocks (100 px = two adjacent squares; 0 = correct); lower is better", "pixels"),
    ("PSLT_SpeedAccuracyTradeoff", "PSLT", "speed_accuracy_tradeoff", "1 / (mean error score x mean RT in all sequence blocks); higher is better", "score"),
    ("PSLT_GenerationScore", "PSLT", "generation_score", "Number of learning trials needed before the participant could generate the entire sequence correctly; lower is better", "trials"),
]
# the Table 2 definition of total learning reads "RT Block 1 - RT Block 6"
for c in [m[0] for m in MEASURES]:
    assert raw[c].notna().all()

codes = [f"{ID}_p{i + 1:03d}" for i in range(n)]
sex = raw.Gender.map({"F": "F", "M": "M"})
hand = raw.Handedness.map({"R": "right", "L": "left", "A": "ambi"})
rows = []
for i, r in raw.iterrows():
    for k, (col, task, meas, desc, unit) in enumerate(MEASURES):
        rows.append(dict(
            dataset=ID, study=ID, experiment="SRTT and PSLT, one session", group=r.Group, subject=codes[i],
            subject_excluded_by_authors=False, age=float(r.Age), sex=sex[i], handedness=hand[i], population="healthy",
            setting="lab", device="computer, mouse clicks on a 5 x 5 grid", session=1, block="whole task", trial=k + 1,
            phase="learning", time_s=None, task="sequence learning", unit="participant summary", condition=task,
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            measure=f"{task}_{meas}", measure_description=desc, statistic="value", value=float(r[col]), value_unit=unit,
            n_trials=None, session_context=""))
checks_df = pd.DataFrame(rows)

pcols = [
    ("age_group", "category", "Younger, Older", "Recruitment group (younger 18-30 years, older 63-84 years)."),
    ("education_level", "integer", "2-6", "Education level in the authors' category code (the deposit and paper do not give the code book)."),
    ("education_years", "integer", "years", "Years of education."),
    ("smmse", "integer", "score", "Standardized Mini-Mental State Examination score."),
    ("word_meanings", "number", "number correct of 34", "Mill Hill Vocabulary (senior, multiple choice) 'Word Meanings', crystallized ability. Empty when missing."),
    ("dot_matrix", "integer", "score", "Dot Matrix (spatial verification span), working memory."),
    ("apm", "integer", "number correct", "Short-form Raven's Advanced Progressive Matrices (column Ravens in the deposit), reasoning."),
    ("inspection_time_ms", "number", "ms", "Inspection time task, processing speed; lower is faster. Empty when missing."),
    ("symbol_digit", "number", "number correct", "Symbol digit coding task, processing speed. Empty when missing."),
]
part = pd.DataFrame(dict(
    person=codes, age_group=raw.Group, education_level=raw["Education Level"], education_years=raw["Education Years"],
    smmse=raw["Mini Mental"], word_meanings=raw["Word Meanings"], dot_matrix=raw["Dot Matrix"], apm=raw["Ravens"],
    inspection_time_ms=raw["Inspection Time"], symbol_digit=raw["Symbol Digit Coding"]))

# ---------------------------------------------------------------- checks
ck.internal("one row per participant in the deposit, all 78 kept", n == 78 and raw["Participant ID"].is_unique)
ck.internal("every participant has all nine learning scores", bool(raw[[m[0] for m in MEASURES]].notna().all().all()))
ck.internal("released rows = 78 participants x 9 measures", len(checks_df) == 78 * 9)
g = part.groupby("age_group")
SRC1 = "Table 1"
ck.published("participants, younger", int((raw.Group == "Younger").sum()), 43, 0, "Methods, Participants")
ck.published("participants, older", int((raw.Group == "Older").sum()), 35, 0, "Methods, Participants")
a = raw.groupby("Group").Age
for grp, m, sd in (("Younger", 22.6, 3.47), ("Older", 70.9, 5.01)):
    ck.published(f"age mean, {grp.lower()}", a.mean()[grp], m, 0.05, SRC1)
    ck.published(f"age SD, {grp.lower()}", a.std()[grp], sd, 0.005, SRC1)
T1 = [("education_years", "Education (years)", (14.4, 2.56), (15.7, 3.75), 1),
      ("smmse", "SMMSE", (29.3, 0.95), (29.2, 0.87), 1),
      ("word_meanings", "Word Meanings", (19.3, 3.95), (25.4, 4.91), 1),
      ("apm", "APM", (6.70, 3.07), (4.06, 2.46), 2),
      ("dot_matrix", "Dot Matrix", (38.1, 9.59), (22.2, 9.75), 1),
      ("symbol_digit", "Symbol Digit", (92.6, 16.5), (54.2, 7.28), 1),
      ("inspection_time_ms", "IT (ms)", (54.2, 29.6), (83.9, 23.3), 1)]
for col, lab, y, o, dec in T1:
    for grp, (m, sd) in (("Younger", y), ("Older", o)):
        x = part.loc[part.age_group == grp, col]
        tol = 0.5 * 10 ** -dec
        sdtol = 0.5 * 10 ** -(2 if sd < 100 and len(str(sd).split(".")[1]) == 2 else 1)
        ck.published(f"{lab} mean, {grp.lower()}", x.mean(), m, tol, SRC1)
        ck.published(f"{lab} SD, {grp.lower()}", x.std(), sd, sdtol, SRC1)
ck.published("participants with inspection time, younger", int(part.loc[part.age_group == "Younger", "inspection_time_ms"].notna().sum()), 42, 0, "Table 1, note b (n = 42; the deposit has 41)")
# correlations with age and fluid abilities (Tables 4 and 5), whole sample, 2 decimals
corr = [("SRTT_RatioRT", [-.28, .30, .38, .30, -.19]), ("PSLT_RatioRT", [-.37, .21, .33, .37, -.23]),
        ("SRTT_SequenceLearning", [-.01, .11, .21, .04, -.01]), ("PSLT_TotalLearning", [-.18, .01, .19, .05, -.16]),
        ("PSLT_MeanErrorScore", [.57, -.47, -.57, -.56, .37]), ("PSLT_GenerationScore", [.60, -.44, -.59, -.57, .35]),
        ("PSLT_SpeedAccuracyTradeoff", [-.59, .55, .61, .65, -.39])]
preds = [("Age", raw.Age), ("APM", raw.Ravens), ("Dot Matrix", raw["Dot Matrix"]), ("Symbol Digit", raw["Symbol Digit Coding"]),
         ("IT", raw["Inspection Time"])]
for meas, vals in corr:
    for (pn, p), rep in zip(preds, vals):
        ck.published(f"correlation of {meas} with {pn}", raw[meas].corr(p), rep, 0.005, "Table 4 / Table 5")

diffs = []
bad = [k for k in ck.items if k["kind"] == "published" and not k["ok"]]
for k in bad:
    diffs.append(f"{k['what']}: the paper reports {k['reported']:g} ({k['source']}); computed from the deposit {k['computed']:.6g}.")
    ck.items.remove(k)
ng = pd.crosstab(raw.Group, raw.Gender)
diffs.insert(0, "Sex: the paper reports 19 women and 24 men (younger) and 12 women and 23 men (older); the deposit's Gender column "
                f"has F = {ng.loc['Younger', 'F']} and M = {ng.loc['Younger', 'M']} (younger), F = {ng.loc['Older', 'F']} and M = "
                f"{ng.loc['Older', 'M']} (older), the reverse. The deposit's labels are kept; the F/M coding is probably swapped in "
                "the deposit or the paper, so do not rely on sex.")
for k in bad:
    print("DROPPED", k["what"], k["computed"], k["reported"])
meta = dict(
    study="Urry et al. 2018",
    paper=dict(authors="Urry K, Burns NR, Baetu I", year=2018,
               title="Age-related differences in sequence learning: Findings from two visuo-motor sequence learning tasks",
               venue="British J of Psychology 109(4):830-849", doi="10.1111/bjop.12299"),
    lab="Baetu (University of Adelaide)", data_url="https://osf.io/vhk3m/", license="none stated", license_url="",
    changes=("Participant IDs replaced by codes. The deposit's wide table (one row per participant) was turned into long form, one row "
             "per participant and learning score; the cognitive test scores and group are in participants.csv. Handedness R, L, A "
             "were written right, left, ambi."),
    how=["The authors shared one row per participant, not trials: nine sequence-learning scores (three from the serial reaction time "
         "task SRTT, six from the predictive sequence learning task PSLT) and the scores of cognitive tests. One row here is one "
         "learning score of one participant (measure, with its definition in measure_description).",
         "Both tasks used an 8-element sequence on a 5 x 5 grid, 9 blocks of 10 repetitions; block 7 was random. Sequence learning is "
         "the RT of random block 7 minus sequence block 8 (ms), ratio_rt divides it by the RT of block 7, total_learning is block 1 minus "
         "block 6. PSLT mean_error_score is in pixels (100 = two adjacent squares). Scores are the authors' own; none were recomputed.",
         "Younger adults were 18-30 years, older adults 63-84. age is in years; group is Younger or Older. Which task was done first was "
         "counterbalanced but is not in the deposit."],
    unit_note="learning score of one participant (long form)",
    differences=diffs,
)
stdlib.release(ID, "summary", checks_df, meta, ck, participants=part, participant_columns=pcols)
for k in ck.items:
    if k["kind"] == "published":
        print(k["what"], round(k["computed"], 4), k["reported"], k["ok"])
