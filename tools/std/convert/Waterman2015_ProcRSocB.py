"""Waterman, Havelka, Culmer, Hill & Mon-Williams (2015), Proc R Soc B 282:20140896: visual-motor memory (VMM) of 87 children
(6-11 years) drawing briefly shown shapes from memory on a tablet; one optimised-error (OE) value per shape and per child.

Raw file (untrusted data, read only): raw_extra/Waterman2015_ProcRSocB/VMM_Writing_Reading_Waterman_et_al_Data.csv, fetched from the
Zenodo copy (record 4956602, CC0) of the Dryad deposit doi:10.5061/dryad.t3g5j. README.pdf was read as text.
Run from the site folder: python3 -I tools/std/convert/Waterman2015_ProcRSocB.py
Not carried: ID (original participant label), Date_of_test (a date).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Waterman2015_ProcRSocB"
SRC = "Waterman et al. 2015, Proc R Soc B 282:20140896"
OE = ("Optimised error (OE) of the child's drawing: the point-set registration error between the stylus path and the target path, "
      "unitless; lower is more accurate.")
MEASURES = [("Copying_OE", "copying_OE", f"{OE} Copying task: the same task with the target always visible, so no memory component. Column Copying_OE.", "mean", "copy")]
for k in range(1, 10):
    MEASURES.append((f"Shape{k}_OE", f"shape{k}_OE", f"{OE} Memory task, path {k}: mean OE over the pair of trials that showed this path (the first two trials of each child are practice and not in the file). Column Shape{k}_OE.", "mean", "memory"))
for nm, what in [("simple", "paths with consistent heights and widths"), ("medium", "paths with consistent heights or widths"),
                 ("hard", "paths with inconsistent heights and widths"), ("overall", "all memory-task trials")]:
    MEASURES.append((f"OE_avg_{nm}", f"OE_average_{nm}", f"{OE} Mean OE over the memory-task trials of {what}. Column OE_avg_{nm}.", "mean", "memory"))

PCOLS = [("school_year", "integer", "", "School year group (2, 4 or 6; ages 6-7, 8-9, 10-11)."),
         ("age_category", "text", "years", "Age range of the child (exact age was removed by the authors)."),
         ("reading_level", "text", "UK National Curriculum level", "Reading level assessed by the school (column Reading_Level)."),
         ("reading_converted", "integer", "", "Reading level as consecutive integers, 2c = 1 (column Reading_converted)."),
         ("writing_level", "text", "UK National Curriculum level", "Writing level assessed by the school (column Writing_Level)."),
         ("writing_converted", "integer", "", "Writing level as the authors' integer code (column Writing_converted); the code is not strictly consecutive for the levels written without a letter.")]

d = pd.read_csv(stdlib.raw_dir(ID) / "VMM_Writing_Reading_Waterman_et_al_Data.csv")
checks = stdlib.Checks()
rows, parts = [], []
for i, r in enumerate(d.itertuples(index=False), start=1):
    subj = f"c{i:03d}"
    hand = {"Right": "right", "Left": "left"}.get(r.Hand, "")
    pers = dict(dataset=ID, study=ID, experiment="", group=f"year {r.YearGroup_School}", subject=subj, subject_excluded_by_authors=False,
                age="", sex=r.Sex, handedness=hand, population="healthy (school children)", setting="clinic",
                device="tablet laptop with digitizer stylus (Kinematic Assessment Tool)", session=1)
    for k, (col, name, desc, stat, cond) in enumerate(MEASURES, start=1):
        rows.append({**pers, "block": "", "trial": k, "phase": "test", "time_s": "", "task": "handwriting and drawing",
                     "unit": "participant summary", "condition": cond, "pert_type": "none", "pert_value": 0, "pert_unit": "",
                     "feedback": "normal", "excluded_by_authors": False, "measure": name, "measure_description": desc,
                     "statistic": stat, "value": float(getattr(r, col)), "value_unit": "OE (unitless)", "n_trials": 2 if col.startswith("Shape") or col == "Copying_OE" and False else ""})
    parts.append(dict(person=subj, school_year=r.YearGroup_School, age_category=r.Age_Cat, reading_level=str(r.Reading_Level).replace("4b +", "4b+"),
                      reading_converted=r.Reading_converted, writing_level=r.Writing_Level, writing_converted=r.Writing_converted))

P = "Methods (Participants)"
checks.published("Year 2 children", (d.YearGroup_School == 2).sum(), 33, 0, f"{SRC}, {P}")
checks.published("Year 4 children", (d.YearGroup_School == 4).sum(), 27, 0, f"{SRC}, {P}")
checks.published("females", (d.Sex == "F").sum(), 44, 0, f"{SRC}, Abstract and {P}")
checks.internal("year 6 children: 28 in the file, 27 in the paper (total 88 vs 87, one extra male)", (d.YearGroup_School == 6).sum() == 28)
checks.internal("one row per child and measure (88 x 14)", len(rows) == 88 * 14 == 1232 and not any(pd.isna(x["value"]) for x in rows))
checks.internal("OE_avg_simple equals Shape1_OE (path 1 is the only simple path)", bool((d.OE_avg_simple == d.Shape1_OE).all()))
checks.internal("OE values positive", bool((d[[m[0] for m in MEASURES]] > 0).all().all()))
checks.internal("no child lost: 88 persons", d.shape[0] == 88)

s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
p = s["papers"][0]
meta = dict(
    study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"],
    data_url="https://doi.org/10.5061/dryad.t3g5j (copy used: https://zenodo.org/records/4956602)", license="CC0",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Stacked each child's values into long form (one row per child and measure, in the authors' column order), moved sex, "
             "handedness and school year into the standard columns and reading and writing levels into participants.csv, and "
             "replaced child labels with codes. Left out the child's original label and the test date. Values were not changed. "
             "Downloaded from the Zenodo copy of the Dryad deposit."),
    how=["One row is one measure of one child (the deposit has one value per child and shape; the stylus paths are not shared). The "
         "child drew shapes shown briefly on a tablet from memory (visual-motor memory task) and copied shapes that stayed visible "
         "(copying task). Each value is an optimised error (OE) from point-set registration between the drawn and the target path, "
         "unitless, lower is more accurate. Shape1-9 are the mean OE of the two trials that showed each path; the averages are over simple, "
         "medium and hard paths and over all memory trials.",
         "Age is only given as a category (6 to 7, 8 to 9, 10 to 11 years); the age column is empty. Handedness is the hand "
         "holding the stylus; three children have none recorded."],
    unit_note="measure of one child (OE, unitless)",
    differences=["The paper reports 87 children (33 in Year 2, 27 in Year 4, 27 in Year 6; 44 female); the shared file has 88 rows (28 in Year 6, 44 "
                 "female and 44 male), one more male in Year 6, unexplained in the deposit. Partial correlations of VMM with writing and "
                 "reading (r = -0.42 and -0.32, controlling for exact age and copying OE) could not be reproduced because exact age was removed "
                 "(with school year in its place the values are -0.29 and -0.26). Writing_converted is not strictly consecutive "
                 "(2a=2, 2b=3, 2c=4, 3=6, 3a=5)."])
stdlib.release(ID, "summary", rows, meta, checks, participants=pd.DataFrame(parts), participant_columns=PCOLS)
