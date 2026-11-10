"""Toffoli et al. (2026), PLOS Digit Health 5(7):e0001503: handwriting kinematics of schoolchildren (grades 1-8) from a sensorized ink pen
(9-axis IMU + force sensor, 50 Hz), one-minute BVSCO-3 writing tasks UNO (the word repeated) and NUM (number words in sequence), in block (B)
and cursive (C) letters. The deposit shares 178 pen-derived indicators per child and task (no raw signals).

Raw files (untrusted data, read only), raw_extra/Toffoli2026_PLOSDigitHealth/ (Zenodo 19485897, CC BY 4.0): UNO_B/UNO_C/NUM_B/NUM_C.xlsx
(one sheet each, 11 descriptor columns then 178 indicator columns, one row per child) and indicators_description.docx (a table of 178 indicators
in the same order as the columns: name [unit], domain, explanation; read as text). The files carry no child identifier, so the same child
cannot be linked across the four files: each row is its own person.
Run from the site folder: python3 -I tools/std/convert/Toffoli2026_PLOSDigitHealth.py
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Toffoli2026_PLOSDigitHealth"
RD = stdlib.raw_dir(ID)
SRC = "Toffoli et al. 2026, PLOS Digit Health 5:e0001503"
FILES = [("UNO_B", "UNO", "block"), ("UNO_C", "UNO", "cursive"), ("NUM_B", "NUM", "block"), ("NUM_C", "NUM", "cursive")]
DESC = {"UNO": "writing the word UNO (ONE) repeatedly for one minute", "NUM": "writing the number words UNO, DUE, TRE... in sequence for one minute"}

# ------------------------------------------------------------------ indicator table from the docx (text only)
x = zipfile.ZipFile(RD / "indicators_description.docx").read("word/document.xml").decode("utf-8")
ind = []
for tr in re.findall(r"<w:tr[ >].*?</w:tr>", x, re.S):
    cells = [re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", " ", c)).strip() for c in re.findall(r"<w:tc>.*?</w:tc>", tr, re.S)]
    if len(cells) == 3 and cells[0] != "Indicator":
        ind.append(cells)
frames = {}
for f, task, allo in FILES:
    frames[f] = pd.read_excel(RD / f"{f}.xlsx")
cols = list(frames["UNO_B"].columns)
assert all(list(d.columns) == cols for d in frames.values())
assert cols[10] == "ExecutionTime" and len(cols) - 11 == 178
ind_cols = cols[11:]
assert len(ind) == 178, len(ind)
# execution time is column 10 and described by the text, not the table
checks = stdlib.Checks()
checks.published("pen-derived indicators per sample", len(ind_cols), 178, 0, f"{SRC}, Methods (Indicators)")
pairs = list(zip(ind_cols, ind))
for c, (n, dom, ex) in pairs[:6] + pairs[-3:]:
    print("  pair:", c, "<->", n)


def unit_of(name):
    m = re.search(r"\[([^\]]*)\]\s*$", name)
    return (m.group(1) if m else "") or "unitless"


desc_of = {"ExecutionTime": ("ExecutionTime", "Time", "Total time of the execution, from the file column ExecutionTime (the task lasts about one minute); unit assumed seconds.", "s")}
for c, (n, dom, ex) in pairs:
    desc_of[c] = (re.sub(r"\s*\[[^\]]*\]\s*$", "", n), dom, ex, unit_of(n))
measure_cols = ["ExecutionTime"] + ind_cols

PCOLS = [("school_grade", "integer", "", "School grade from 1 (first grade of primary school) to 8 (third year of lower secondary school), column year."),
         ("task_file", "text", "", "File the row came from: UNO_B, UNO_C, NUM_B or NUM_C (task and letter form). A child cannot be linked across files."),
         ("bvsco3_z", "number", "z-score", "BVSCO-3 writing-speed z-score for grade and time point (column Z)."),
         ("nationality_other", "true/false", "", "Nationality other than Italian (column naz: 0 Italian, 1 other); empty where missing."),
         ("mother_tongue_other", "true/false", "", "Mother tongue other than Italian (column mt)."),
         ("home_language_other", "true/false", "", "Language spoken at home other than Italian (column itadom)."),
         ("handedness_code", "integer", "0 right, 1 left", "Column handness as shared: 0 right, 1 left (the deposit says nothing about the value 2, which occurs in a few rows)."),
         ("laterality_coherent", "true/false", "", "Hand, eye and foot have the same dominance (column laterality_coherence)."),
         ("favourite_allograph_cursive", "true/false", "", "Child's favourite allograph is cursive (column favourite_allograph: 0 block, 1 cursive).")]


def b(v):
    return None if pd.isna(v) else bool(int(v))


rows, parts = [], []
n_missing = 0
for f, task, allo in FILES:
    d = frames[f]
    for i, r in enumerate(d.itertuples(index=False), start=1):
        rr = dict(zip(cols, r))
        subj = f"{f}_{i:04d}"
        hc = rr["handness"]
        hand = "" if pd.isna(hc) else {0: "right", 1: "left"}.get(int(hc), "")
        pers = dict(dataset=ID, study=ID, experiment=task, group=f"grade {int(rr['year'])}", subject=subj, subject_excluded_by_authors=False,
                    age=round(float(rr["age"]), 3), sex="M" if int(rr["sex"]) == 0 else "F", handedness=hand,
                    population="healthy (school children)", setting="clinic",
                    device="sensorized ink pen (9-axis IMU and force sensor, 50 Hz) on ordinary paper", session=1)
        for k, c in enumerate(measure_cols, start=1):
            v = rr[c]
            if pd.isna(v):
                n_missing += 1
                continue
            nm, dom, ex, un = desc_of[c]
            rows.append({**pers, "block": f"{task} {allo}", "trial": k, "phase": "test", "time_s": "", "task": "handwriting and drawing",
                         "unit": "participant summary", "condition": f"{task} {allo}", "pert_type": "none", "pert_value": 0, "pert_unit": "",
                         "feedback": "normal", "excluded_by_authors": False, "measure": c,
                         "measure_description": f"{nm} ({dom} domain; file column {c}). {ex} Task: {DESC[task]} in {allo} letters.",
                         "statistic": "value", "value": float(v), "value_unit": un, "n_trials": ""})
        parts.append(dict(person=subj, school_grade=int(rr["year"]), task_file=f, bvsco3_z=rr["Z"], nationality_other=b(rr["naz"]),
                          mother_tongue_other=b(rr["mt"]), home_language_other=b(rr["itadom"]),
                          handedness_code=None if pd.isna(hc) else int(hc), laterality_coherent=b(rr["laterality_coherence"]),
                          favourite_allograph_cursive=b(rr["favourite_allograph"])))

uc, nc = frames["UNO_C"], frames["NUM_C"]
checks.published("UNO cursive samples with a z-score above zero", int((uc.Z > 0).sum()), 27, 0, f"{SRC}, Results (cursive tasks)")
checks.published("NUM cursive samples with a z-score above zero", int((nc.Z > 0).sum()), 37, 0, f"{SRC}, Results (cursive tasks)")
for f in ("UNO_C", "NUM_C"):
    zg = frames[f].groupby("year").Z.mean()
    checks.internal(f"{f}: mean z-score below -1 in every grade (paper: lower than -1 across all grades)", bool((zg < -1).all()), f"{zg.round(2).to_dict()}")
tot = sum(len(d) for d in frames.values())
checks.internal("every file row is a person (2755 rows)", len(parts) == tot, f"{tot}")
checks.internal("rows = values present (children x 179 columns minus missing)", len(rows) == tot * 179 - n_missing, f"{len(rows)} rows, {n_missing} missing values not written")
checks.internal("grades 1 to 8 in every file", all(sorted(d.year.unique()) == list(range(1, 9)) for d in frames.values()))
checks.internal("participants in the largest file within the paper's 707 recruited", max(len(d) for d in frames.values()) <= 707, f"{max(len(d) for d in frames.values())}")

s = next(z for z in json.load(open(stdlib.HERE / "source/extra_studies.json")) if z["id"] == ID)
p = s["papers"][0]
meta = dict(
    study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Stacked the four spreadsheets (UNO_B, UNO_C, NUM_B, NUM_C) into long form, one row per child, task file and measure, in the "
             "authors' column order; moved grade, age, sex and handedness into the standard columns and the other descriptor columns into "
             "participants.csv; attached each indicator's name, domain and explanation from the authors' indicators_description file. Empty "
             "cells are not written as rows. Replaced rows with codes (the files have no child identifier)."),
    how=["One row is one indicator of one child in one task. Children wrote for one minute with a sensorized ink pen (9-axis IMU and a force "
         "sensor, 50 Hz) on paper: UNO is the word UNO written repeatedly, NUM is the number words UNO, DUE, TRE... in sequence, each in block (B) "
         "or cursive (C) letters. The deposit shares pen-derived indicators only (178: temporal, force, smoothness, kinematic oscillations, "
         "frequency and tilt), not the raw signals; ExecutionTime is the first measure. value_unit is the unit in brackets in the authors' indicator "
         "description, 'unitless' where it is empty.",
         "sex is as given at birth (0 male, 1 female in the files). age is at the time of the test. The four files carry no child identifier, so "
         "a child who did several tasks appears as separate people; the study recruited 707 children, the largest file has 701 rows.",
         "participants.csv holds the BVSCO-3 writing-speed z-score (for grade and time point), language and laterality descriptors."],
    unit_note="indicator of one child in one task (summary)",
    differences=["The deposit has no codebook for ExecutionTime (read as seconds from its values near 60) and the value 2 in the handedness column "
                 "(0 right, 1 left are stated); the value 2 is left unmapped in handedness and kept as shared in participants.csv. Table 1 of the paper "
                 "(sample sizes by grade) was not available as text, so only the cursive z-score counts were checked."],
)
stdlib.release(ID, "summary", rows, meta, checks, participants=pd.DataFrame(parts), participant_columns=PCOLS)
