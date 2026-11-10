"""Russo et al. (2021), Exp Brain Res: explicit finger-tapping sequence learning in 19 stroke patients (10 right, 9 left brain damage)
and 18 controls, number of correct sequences per block. Raw (untrusted, read only): raw_extra/Russo2021_ExpBrainRes/Raw data.xlsx (OSF 2su95).
Run: python3 -I tools/std/convert/Russo2021_ExpBrainRes.py"""
import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Russo2021_ExpBrainRes"
wb = openpyxl.load_workbook(stdlib.raw_dir(ID) / "Raw data.xlsx", read_only=True, data_only=True)
data = [r for r in wb.worksheets[0].iter_rows(values_only=True)]
head = list(data[0])
PHASE = {"Pre_training": ("baseline", 1), "B1": ("learning", 2), "B2": ("learning", 3), "B3": ("learning", 4), "B4": ("learning", 5),
         "B5": ("learning", 6), "Post-training": ("test", 7), "FU-90m": ("test", 8), "FU-24h": ("test", 9)}
SESSION = {"Pre_training": 1, "B1": 1, "B2": 1, "B3": 1, "B4": 1, "B5": 1, "Post-training": 1, "FU-90m": 1, "FU-24h": 2}
rows = []
for r in data[1:]:
    if r[0] is None:
        continue
    sid = str(r[0]).strip()
    grp = "Right brain damage" if sid.startswith("Right") else "Left brain damage" if sid.startswith("Left") else "Control"
    for j, h in enumerate(head[1:], start=1):
        if r[j] is None:
            continue
        ph, k = PHASE[h]
        rows.append(dict(dataset=ID, study="Russo et al. 2021 Exp Brain Res", experiment="explicit motor sequence learning (finger tapping task)",
                         group=grp, subject=sid, person=sid, subject_excluded_by_authors=False, age=None, sex="", handedness="",
                         population="healthy" if grp == "Control" else "stroke", setting="lab", device="computer keyboard (E-Prime)",
                         session=SESSION[h], block=h, trial=k, phase=ph, time_s=None, task="sequence learning",
                         unit="participant summary", condition="", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                         excluded_by_authors=False, measure="correct_sequences", statistic="count",
                         measure_description="Number of correctly reproduced sequences within the block (3 min each)",
                         value=r[j], value_unit="sequences", n_trials=None))
checks = stdlib.Checks()
ctl = lambda b: [x["value"] for x in rows if x["group"] == "Control" and x["block"] == b]  # noqa: E731
checks.published("control group, mean correct sequences in B1", sum(ctl("B1")) / len(ctl("B1")), 41.3, 0.05, "Results, Online learning (B1 = 41.3)")
checks.published("control group, mean correct sequences in B5", sum(ctl("B5")) / len(ctl("B5")), 50.2, 0.05, "Results, Online learning (B5 = 50.2)")
n = lambda g: len({x["subject"] for x in rows if x["group"] == g})  # noqa: E731
checks.published("stroke patients", n("Right brain damage") + n("Left brain damage"), 19, 0, "Participants")
checks.published("right brain damage patients", n("Right brain damage"), 10, 0, "Participants (10 right-sided)")
checks.published("left brain damage patients", n("Left brain damage"), 9, 0, "Participants (nine left-sided)")
checks.published("controls", n("Control"), 18, 0, "Participants (eighteen healthy subjects)")
checks.internal("9 values per participant", len(rows) == 37 * 9)
paper = dict(authors="Russo C, Veronelli L, Casati C, Monti A, Perucca L, Ferraro F, Corbo M, Vallar G, Bolognini N", year=2021,
             title="Explicit motor sequence learning after stroke: a neuropsychological study", venue="Exp Brain Res 239(7):2303-2316",
             doi="10.1007/s00221-021-06141-5")
meta = dict(paper=paper, lab="Bolognini & Vallar (University of Milano-Bicocca)", data_url="https://osf.io/2su95/", license="none stated",
            license_url="", changes="Wide table turned into long form (one row per participant and block); nothing recomputed.",
            how=["One row is the number of correct sequences one participant produced in one 3-minute block of the finger tapping task: one pre-training block, five "
                 "training blocks (B1 to B5), a post-training block, and follow-ups after 90 minutes and 24 hours (session 2).",
                 "Patients used their paretic hand (paper, Abstract); controls used their non-dominant hand. Hand, age, sex and clinical data are not in the "
                 "deposit file and are not included.",
                 "Participant codes are those of the deposit's group labels, replaced by OpenMotor codes."],
            unit_note="participant and block")
stdlib.release(ID, "summary", rows, meta, checks)
