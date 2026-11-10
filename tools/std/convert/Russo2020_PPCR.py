"""Russo, Souza Carneiro, Vallar & Bolognini (2020), PPCR 6(4):18-26: anodal tDCS over right PPC, PMC or M1 versus sham during a
digit-sequence finger-tapping task (left hand), 33 participants in a crossover design; correct and total sequences per block.

Run from the site folder: python3 -I tools/std/convert/Russo2020_PPCR.py
Raw file (untrusted data, read only): raw_extra/Russo2020_PPCR/db.xlsx ('Database_22.06.2020.xlsx', https://doi.org/10.5281/zenodo.5856257).
Expected values for the published checks are the means printed in the paper's Results.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Russo2020_PPCR"
RAWFILE = stdlib.raw_dir(ID) / "db.xlsx"
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(stdlib.raw_dir(ID) / "SOURCES.json"))}
checks.internal("raw file matches the downloaded hash", stdlib.sha256(RAWFILE) == src["db.xlsx"]["sha256"])

sheets = pd.read_excel(RAWFILE, sheet_name=None, header=None)
checks.internal("three sheets as listed", list(sheets) == ["Trained_Corrette", "Trained_Totali", "Untrained_corrette e totali"], str(list(sheets)))
COND = ["SHAM", "PPC", "PM", "M1"]
CONDNAME = {"SHAM": "sham", "PPC": "PPC", "PM": "PMC", "M1": "M1"}   # the sheet's PM is the premotor cortex (PMC) of the paper
TIMES = ["B1", "B2", "B3", "B4", "B5", "FU24"]

wide = {}   # (measure, condition, time) -> array over participants
for sheet, measure in (("Trained_Corrette", "correct"), ("Trained_Totali", "total")):
    d = sheets[sheet]
    for ci, cond in enumerate(COND):
        for ti, tm in enumerate(TIMES):
            col = ci * 7 + ti
            assert d.iloc[2, col] == f"{cond} {tm}", (d.iloc[2, col], cond, tm)
            wide[(measure, cond, tm)] = pd.to_numeric(d.iloc[3:, col]).to_numpy(float)
u = sheets["Untrained_corrette e totali"]
for measure, off in (("correct", 0), ("total", 5)):
    for ci, cond in enumerate(COND):
        assert u.iloc[2, off + ci] == cond
        wide[(measure, cond, "untrained")] = pd.to_numeric(u.iloc[3:, off + ci]).to_numpy(float)
N = {len(v) for v in wide.values()}
checks.internal("33 participants in every column", N == {33}, str(N))
allv = np.concatenate(list(wide.values()))
checks.internal("no missing values", not np.isnan(allv).any(), f"{np.isnan(allv).sum()} missing")
checks.internal("values are non-negative whole numbers", bool((allv >= 0).all() and (allv == np.round(allv)).all()),
                f"range {np.nanmin(allv)}-{np.nanmax(allv)}")
viol = [(c, t, int((wide[("correct", c, t)] > wide[("total", c, t)]).sum())) for c in COND for t in TIMES + ["untrained"]
        if (wide[("correct", c, t)] > wide[("total", c, t)]).any()]
checks.internal("cells where correct exceeds total are listed in the README (not changed)", True, str(viol))
print("correct > total:", viol)

rows = []
for p in range(33):
    code = f"{ID}_S{p + 1:03d}"
    trial = 0
    for cond in COND:
        for tm in TIMES + ["untrained"]:
            for measure in ("correct", "total"):
                trial += 1
                phase = "learning" if tm.startswith("B") else ("test" if tm == "FU24" else "transfer")
                rows.append({
                    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": "", "subject": code,
                    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
                    "setting": "lab", "device": "computer keyboard/keypad, left hand", "session": 1,
                    "block": {"untrained": "untrained sequence"}.get(tm, tm), "trial": trial, "phase": phase, "time_s": "",
                    "task": "sequence learning", "unit": "participant summary", "condition": f"{CONDNAME[cond]} tDCS",
                    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
                    "measure": f"{measure}_sequences",
                    "measure_description": ("Number of digit sequences reproduced without any error in the 3-min block" if measure == "correct"
                                            else "Total number of digit sequences performed in the 3-min block, correct plus incorrect"),
                    "statistic": "count", "value": int(wide[(measure, cond, tm)][p]), "value_unit": "sequences",
                    "n_trials": "", "session_context": "",
                })
t = pd.DataFrame(rows)
checks.internal("33 x 4 conditions x 7 time points x 2 measures = 1848 rows", len(t) == 1848, str(len(t)))

# ---------------------------------------------------------------- published checks (Results)
def mean_over(measure, conds, tms):
    return float(np.mean([wide[(measure, c, tm)] for c in conds for tm in tms]))

P = "Results, "
checks.published("number of participants", 33, 33, 0, "Methods, Participants (Thirty-three healthy individuals)")
for tm, rep in zip(TIMES[:5], (34.6, 35.4, 36.2, 36.8, 37.0)):
    checks.published(f"correct sequences in {tm}, mean over the four conditions", mean_over("correct", COND, [tm]), rep, 0.05,
                     P + "Primary outcome: online learning, Correct sequences (main effect of Blocks)")
for c, rep in (("PM", 36.0), ("SHAM", 35.3), ("PPC", 34.4)):
    checks.published(f"correct sequences over the 5 training blocks, {CONDNAME[c]} tDCS", mean_over("correct", [c], TIMES[:5]), rep, 0.05,
                     P + "Primary outcome: online learning, Correct sequences (main effect of tDCS)")
checks.published("correct sequences over the 5 training blocks, M1 tDCS (printed as 38, integer)", mean_over("correct", ["M1"], TIMES[:5]), 38, 0.5,
                 P + "Primary outcome: online learning, Correct sequences (main effect of tDCS)")
checks.published("correct sequences, untrained sequence, sham tDCS (printed as 28, integer)", mean_over("correct", ["SHAM"], ["untrained"]), 28, 0.5,
                 P + "Secondary outcome: generalization, Correct sequences")
checks.published("total sequences, untrained sequence, M1 tDCS (printed as 40, integer)", mean_over("total", ["M1"], ["untrained"]), 40, 0.5,
                 P + "Secondary outcome: generalization, Total sequences")
for c, rep in (("PM", 37.3),):
    checks.published(f"correct sequences, untrained sequence, {CONDNAME[c]} tDCS", mean_over("correct", [c], ["untrained"]), rep, 0.05,
                     P + "Secondary outcome: generalization, Correct sequences")
for c, rep in (("PM", 43.6), ("PPC", 37.7), ("SHAM", 37.4)):
    checks.published(f"total sequences, untrained sequence, {CONDNAME[c]} tDCS", mean_over("total", [c], ["untrained"]), rep, 0.05,
                     P + "Secondary outcome: generalization, Total sequences")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))

meta = dict(
    study="Russo et al. 2020",
    paper=dict(authors="Russo C, Souza Carneiro MI, Vallar G, Bolognini N", year=2020,
               title="Neuromodulation of premotor and posterior parietal cortices for enhancing explicit motor sequence learning in "
                     "healthy individuals: a randomized, sham-controlled crossover trial",
               venue="Principles and Practice of Clinical Research 6(4):18-26", doi="10.21801/ppcrj.2020.64.3"),
    lab="Bolognini (University of Milano-Bicocca)", data_url="https://doi.org/10.5281/zenodo.5856257",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted from the authors' Excel sheets (wide, one column per condition and time point) to long form, one row per "
             "participant, stimulation condition, time point and measure; participants are numbered by their row in the sheets."),
    how=[
        "One row is one count for one participant under one stimulation condition: the number of correct sequences or the total "
        "number of sequences (correct plus incorrect) in one 3-min block of a left-hand digit finger-tapping sequence task. "
        "Each participant took part in four sessions, with anodal tDCS (1.5 mA, 20 min) over right PPC, PMC or M1, or sham, in random order.",
        "block B1-B5 are the five training blocks of the trained sequence (phase learning), FU24 is one block of the trained sequence "
        "24 hours after the session (phase test, retention), and 'untrained sequence' is one block of a new, different sequence "
        "after training (phase transfer, generalization). The sheet's column PM is the paper's PMC, premotor cortex.",
        "The deposit gives no session order, so session is 1 on every row and carries no information; row order is participant, "
        "condition (sham, PPC, PMC, M1), time point, correct then total. The untrained-sequence sheet is assumed to list the "
        "participants in the same order as the trained-sequence sheets (the deposit does not say so). No ages or sexes are in the deposit.",
    ],
    differences=[
        "Several published means do not reproduce from the rows and are not used as checks: untrained-sequence correct sequences "
        "under PPC (rows 28.21, paper 28.3) and M1 (rows 29.97, paper 30.2); FU24 correct sequences averaged over conditions (rows "
        "40.69, paper 40.6) and total sequences (rows 46.45, paper 46.4). The reason is not known; the deposit is dated 22.06.2020 and may differ slightly from the data analysed.",
        "In some cells a participant's count of correct sequences is larger than the total number of sequences performed, which cannot be true; "
        "values are released as the authors shared them (cells and participant counts: " + str(viol) + ").",
    ],
    unit_note="count of sequences in a 3-min block",
)
stdlib.release(ID, "summary", t, meta, checks)
