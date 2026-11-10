"""Oldrati, Nave, Boscarol, Ciricugno, Arioli, Cattaneo, Urgesi & Finisguerra (2026), Neural Plasticity 2026(1):3295986:
fronto-cerebellar tACS and tVNS (each with a sham session) during a visuomotor drag-and-drop task, 96 healthy adults (48 per
stimulation type, 24 women each). One row per valid trial, module touch_precision.

Run from the site folder: python3 -I tools/std/convert/Oldrati2026_NeuralPlast.py
Raw file (untrusted data, read only): raw_extra/Oldrati2026_NeuralPlast/NIBS_sexdiff.csv from https://zenodo.org/records/17417281
(CC BY 4.0): semicolon-separated, columns subject;group;sex;stimulation;trial;distance;type;PE;RT. Rows of one participant and
one stimulation condition are contiguous; the trial number runs 1-160 twice within each condition (two parts of a session)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Oldrati2026_NeuralPlast"
RAW = stdlib.raw_dir(ID)
ck = stdlib.Checks()
d = pd.read_csv(RAW / "NIBS_sexdiff.csv", sep=";", dtype={"subject": str})
ck.internal("53242 rows, 9 columns, no empty cells", d.shape == (53242, 9) and d.isna().sum().sum() == 0)
ck.internal("96 participants, each in exactly one group and with one sex; every participant's rows for a stimulation condition are contiguous",
            d.subject.nunique() == 96 and (d.groupby("subject")[["group", "sex"]].nunique() == 1).all().all()
            and (d.subject + d.stimulation).ne((d.subject + d.stimulation).shift()).sum() == d.groupby(["subject", "stimulation"]).ngroups)
ck.internal("all rows are valid trials; PE 0 to 0.862, RT 0 to 997", (d.type == "valid").all() and d.PE.between(0, 0.9).all() and d.RT.between(0, 1000).all())

names = [c[1] for c in stdlib.columns_for("touch_precision")]
GROUP = {"tacs": "tACS", "tvns": "tVNS"}
rows = []
for s, g in d.groupby("subject", sort=False):
    order = list(dict.fromkeys(g.stimulation))
    k = 0
    for r in g.itertuples(index=False):
        k += 1
        row = dict.fromkeys(names, "")
        row.update(
            dataset=ID, study=ID, experiment=ID, group=GROUP[r.group], subject=f"{ID}_{s}", person=f"{ID}_{s}",
            subject_excluded_by_authors="False", sex=r.sex, population="healthy", setting="lab",
            device="touchpad, index finger, drag and drop", session=order.index(r.stimulation) + 1,
            block=r.stimulation, trial=k, phase="other", task="visuomotor drag-and-drop with brain stimulation", unit="trial",
            condition=f"{r.stimulation} {GROUP[r.group]}", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors="False", trial_in_part=r.trial, target_distance=r.distance, trial_status=r.type,
            precision_error=r.PE, rt_ms=r.RT)
        rows.append(row)
ck.internal("every row kept, in order", len(rows) == len(d))

# published numbers
per = d.drop_duplicates("subject")
ck.published("participants", len(per), 96, 0, "Abstract and Methods (96 healthy adults)")
ck.published("tACS participants", (per.group == "tacs").sum(), 48, 0, "Methods, Participants")
ck.published("tVNS participants", (per.group == "tvns").sum(), 48, 0, "Methods, Participants")
ck.published("women in the tACS group", ((per.group == "tacs") & (per.sex == "F")).sum(), 24, 0, "Methods, Participants")
ck.published("women in the tVNS group", ((per.group == "tvns") & (per.sex == "F")).sum(), 24, 0, "Methods, Participants")
pe = d.groupby(["group", "distance"]).PE.mean()
for (gr, dist), rep in {("tacs", "near"): 0.11, ("tacs", "far"): 0.15, ("tvns", "near"): 0.11, ("tvns", "far"): 0.14}.items():
    ck.published(f"mean precision error, {GROUP[gr]}, {dist} targets", pe[(gr, dist)], rep, 0.006, "Results, descriptive values of PE by distance")
ck.published("tACS participants with data in both conditions", (d[d.group == "tacs"].groupby("subject").stimulation.nunique() == 2).sum(), 47, 0,
             "Methods, exclusions (one tACS participant had sham data only)")
smr = d.groupby(["group", "subject"]).RT.mean().groupby("group").mean()
ck.published("mean reaction time (ms) of the tACS group (mean of participant means)", smr["tacs"], 293.76, 0.05, "Results, descriptive values of RT by group")
ck.published("mean reaction time (ms) of the tVNS group (mean of participant means)", smr["tvns"], 349.06, 0.05, "Results, descriptive values of RT by group")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Urgesi / Finisguerra (Mondino Foundation, Pavia; Udine)", data_url="https://zenodo.org/records/17417281",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format, participant codes replaced, the authors' group, stimulation and distance labels kept (tacs and tvns written tACS and tVNS).",
    unit_note="valid trial of one participant (one drag-and-drop)",
    how=["One row is one valid trial of a drag-and-drop task: a crumpled-paper object appears at the screen centre, the participant clicks it with the right index finger on a touchpad and drags it into one of eight bins (four near, four far). precision_error is the authors' distance between the drop position and the bin centre and rt_ms is the time to the click, both per trial as shared; the deposit has no drag trajectories, bin identities, blocks or age.",
         "group is the stimulation type (tACS: fronto-cerebellar gamma tACS; tVNS: transcutaneous vagus nerve stimulation); condition is active or sham, each participant doing both in separate sessions (session 1 or 2 by order of appearance in the file). trial is the order within the participant; trial_in_part is the authors' number, which restarts at 1 after trial 160 within each condition.",
         "The deposit shares valid trials only (invalid and overtime trials, and the authors' PE outliers, were removed); precision_error is in the unit of the shared file, values 0 to 0.86; the paper defines PE in pixels, but these values look like fractions of the screen size (the paper itself reports means of 0.10 to 0.15 without a unit), so treat the unit as not stated."],
)
stdlib.release(ID, "touch_precision", rows, meta, ck)
