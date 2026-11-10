"""Lammert, Shadle, Narayanan & Quatieri (2018), PLoS ONE 13(9):e0202180: speed-accuracy tradeoffs (Fitts' law) in speech
production, rtMRI of 10 American English speakers (5 F, 5 M) from USC-TIMIT. Dryad 10.5061/dryad.5pn163j, read from its Zenodo
copy (record 4959487, CC0): complexity_mats_plos.zip with, per speaker, 41 x 41 phoneme-pair matrices
<speaker>_D.csv (distance D, articulatory space), <speaker>_W.csv (width W; constant down each column, so it is the width of the
target phoneme h), <speaker>_T_cat1..5.csv (movement time MT per syllable-position category; 0 = no such task).
Row index = initial phoneme g, column = target phoneme h, in the order of the README's PH list.
Categories (paper, Methods): 1 onset-nucleus, 2 nucleus-coda, 3 onset-onset, 4 coda-coda, 5 coda-onset.
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Lammert2018_PLoSOne"
SRC = "Lammert et al. 2018, Methods (Data)"
CAT = {1: "category 1 onset-nucleus", 2: "category 2 nucleus-coda", 3: "category 3 onset-onset", 4: "category 4 coda-coda",
       5: "category 5 coda-onset"}
CTX = "Midsagittal real-time MRI of read sentences (USC-TIMIT, 23.18 frames/s); one row is one phoneme-pair task averaged by the authors."


def main():
    raw = stdlib.raw_dir(ID)
    txt = (raw / "README.txt").read_text(encoding="utf-8", errors="replace")
    ph = re.findall(r"[a-z]+", txt[txt.index("PH:") + 3:txt.index("%eof")])
    assert len(ph) == 41, len(ph)
    z = zipfile.ZipFile(raw / "c.zip")

    def mat(s, m):
        return np.loadtxt(io.StringIO(z.read(f"complexity_mats_plos/{s}_{m}.csv").decode()), delimiter=",")

    subs = sorted({Path(n).name.split("_")[0] for n in z.namelist() if n.endswith(".csv") and not Path(n).name.startswith(".")})
    subs = [s for s in subs if re.fullmatch(r"[FM]\d", s)]
    rows = []
    for s in subs:
        D, W = mat(s, "D"), mat(s, "W")
        for c in CAT:
            T = mat(s, f"T_cat{c}")
            for g, h in zip(*np.nonzero(T)):
                cond = f"{ph[g]} to {ph[h]}"
                for m, desc, val, unit in (
                    ("movement_time", "Movement time MT of the task from the initial phoneme to the target phoneme (authors' T matrix; "
                     "unit not stated, values 0-2 so seconds inferred)", T[g, h], "s"),
                    ("distance", "Distance D between the initial and target phoneme in articulatory space (authors' D matrix)",
                     D[g, h], "authors' articulatory-space units"),
                    ("target_width", "Width W of the target phoneme (authors' W matrix, a column constant: the width of the target)",
                     W[g, h], "authors' articulatory-space units")):
                    rows.append({
                        "dataset": ID, "study": ID, "experiment": "", "group": "", "subject": s,
                        "subject_excluded_by_authors": False, "age": "", "sex": s[0], "handedness": "",
                        "population": "healthy", "setting": "lab", "device": "real-time MRI", "session": 1, "block": CAT[c],
                        "trial": 0, "phase": "other", "time_s": "", "task": "speech", "unit": "participant summary",
                        "condition": cond, "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal",
                        "excluded_by_authors": False, "measure": m, "measure_description": desc, "statistic": "other",
                        "value": float(val), "value_unit": unit, "n_trials": "", "session_context": CTX})
    df = pd.DataFrame(rows)
    checks = stdlib.Checks()
    checks.published("speakers", df.subject.nunique(), 10, 0, SRC + " (10 speakers of American English)")
    checks.published("female speakers", df[df.sex == "F"].subject.nunique(), 5, 0, SRC + " (five male and five female)")
    checks.published("syllable-position categories", df.block.nunique(), 5, 0, SRC + " (five categories of interest)")
    checks.internal("every task has movement time, distance and width", (df.groupby("measure").size().nunique() == 1))
    checks.internal("movement times positive", bool((df[df.measure == "movement_time"].value > 0).all()))
    checks.internal("41 phonemes in the README", True)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    df["sex"] = df.sex.map({"F": "F", "M": "M"})
    meta = dict(
        study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC0 (Dryad deposit, read from its Zenodo copy)", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes="Turned the authors' 41 x 41 phoneme-pair matrices into long form (only pairs with a nonzero movement time), "
                "named the pairs with the README's phoneme list, took sex from the speaker label and replaced speaker labels with codes.",
        how=["One row is one measure of one phoneme-pair task (condition 'initial to target') of one speaker in one syllable-position "
             "category (block), taken from the authors' matrices: movement_time (MT), distance (D) and target_width (W).",
             "Pairs with a zero in the movement time matrix (no such task) are left out. Movement time is in seconds inferred from its "
             "range (0-2); D and W are in the authors' articulatory-space units, as the README gives no unit.",
             "The matrices are the authors' per-pair values, not single tokens."],
        unit_note="speaker, phoneme-pair task and measure",
        differences=["Table 1 of the paper (number of data points and the MT-ID correlation per speaker and category) does not reproduce from "
                     "the shared matrices: for example speaker M2 has 175, 105, 29, 25 and 349 tasks with nonzero MT against 183, 127, 33, 40 "
                     "and 384 data points in the table, so those numbers are not used as checks. The shared matrices appear to be per-pair values "
                     "and the cause of the difference is not known."])
    return stdlib.release(ID, "summary", df.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
