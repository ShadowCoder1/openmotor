"""Albert et al. (2020), eLife: postural control of the arm through integration of movement commands. Human data only (the
deposit's Figs 1 and 2 are monkey EMG and are not converted). Module force_field_reaching: one row per channel trial (hold-force
change and movement force integral) or per force-field trial (endpoint error).

Run from the site folder: python3 -I tools/std/convert/Albert2020_eLife.py
Raw files (read only) in raw_extra/Albert2020_eLife/ from https://osf.io/yc64a/: data-for-figure-3.xlsx, -4, -6, -7 (the sheets
listed below). Figure 5 (postural-field fit per participant) and the per-participant gain tables of Fig 6 are derived summaries
and are not converted; Figs 1-2 are monkeys.
"""
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Albert2020_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
R = lambda n: RAW / f"data-for-figure-{n}.xlsx"  # noqa: E731
HOLD = "Hold 2 - Hold 1 (force)"
MOVE = "Move force integral"
rows = []
seq = {}


def add(person, sheet_key, experiment, d, trial_type, pert, force_code=None, **kw):
    key = (person, sheet_key)
    seq[key] = seq.get(key, 0) + 1
    r = dict(
        dataset=ID, study=ID, experiment=experiment, group=kw.get("group", ""), subject=f"{person}|{sheet_key}", person=person,
        subject_excluded_by_authors="False", age="", sex="", handedness="", population=kw.get("population", "healthy"), setting="lab",
        device="robotic manipulandum (force-measuring handle)", session="", block=kw.get("block", ""), trial=seq[key], phase=kw.get("phase", "other"),
        time_s="", task="reaching", unit="trial", condition=kw.get("condition", ""), pert_type=pert, pert_value="" if pert != "none" else 0,
        pert_unit="", feedback="normal", excluded_by_authors="False",
        trial_type=trial_type, trial_in_condition=d["Trial number"], movement_duration_s=(d["Duration"] / 1000.0 if "Duration" in d and pd.notna(d["Duration"]) else ""),
        force_field_code=force_code if force_code is not None else "", weight_support=kw.get("weight_support", ""),
        hold_force_change_n=d.get(HOLD, ""), move_force_integral_ns=d.get(MOVE, ""), endpoint_error_cm=d.get("Endpoint error", ""))
    rows.append(r)


def recs(df):
    return df.to_dict("records")


ORI = {0: ("CCW", 3), 1: ("CW", 5)}
f3a = pd.read_excel(R(3), "data-panels-B-through-D")
for d in recs(f3a):
    add(f"G{d['Subject ID']}", "f3a", f"Fig3 B-D, experiment {d['Experiment number']}", d, "channel", "none",
        weight_support="airsled" if d["Weight support"] == 1 else "none", condition="null field channel trials")
f3b = pd.read_excel(R(3), "data-panels-E-through-G")
HP = {0: "start in line with midline", 1: "start left of midline", 2: "start right of midline"}
PER = {0: "field magnitude gradually increasing", 1: "field magnitude gradually decreasing"}
for d in recs(f3b):
    o, code = ORI[d["Force orientation"]]
    add(f"G{d['Subject ID']}", "f3b", f"Fig3 E-G, experiment {d['Experiment number']}", d, "channel", "force_field_curl", code,
        block=f"{o} field, {PER[d['Period']]}", condition=HP[d["Hand position"]])
f4 = {"A": pd.read_excel(R(4), "data-panel-A"), "B": pd.read_excel(R(4), "data-panel-B")}
for k, df in f4.items():
    for d in recs(df):
        add(f"G{d['Subject ID']}", f"f4{k}", f"Fig4 panel {k}, experiment {d['Experiment number']}", d, "channel", "force_field_curl",
            block=f"phase {d['Period'] + 1}", condition="channel trials")
f6ff = pd.read_excel(R(6), "data-panel-D-force-field")
for d in recs(f6ff):
    o, code = ORI[d["Force orientation"]]
    add(f"G{d['Subject ID']}", "f6ff", f"Fig6 D, experiment {d['Experiment number']}", d, "field", "force_field_curl", code,
        block=f"{o} field", phase="learning", condition="force field trials")
f6ch = pd.read_excel(R(6), "data-panel-D-channel")
for d in recs(f6ch):
    o, code = ORI[d["Force orientation"]]
    add(f"G{d['Subject ID']}", "f6ch", f"Fig6 D, experiment {d['Experiment number']}", d, "channel", "force_field_curl", code,
        block=f"{o} field", condition="channel trials in a force field block")
ARM = {0: ("C", "age-matched control", "healthy", "control arm"), 1: ("S", "stroke patient", "stroke (corticospinal tract lesion)", "non-paretic arm"),
       2: ("S", "stroke patient", "stroke (corticospinal tract lesion)", "paretic arm")}
for sheet, lab, fk in (("data-null-field-trials", "null field channel trials", "f7n"), ("data-force-field-trials", "channel trials in the force field block", "f7f")):
    f7 = pd.read_excel(R(7), sheet)
    for d in recs(f7):
        pre, grp, pop, arm = ARM[d["Arm condition"]]
        add(f"{pre}{d['Subject ID']}", f"{fk}{d['Arm condition']}", "Fig7 stroke", d, "channel", "none" if fk == "f7n" else "force_field_curl",
            group=grp, population=pop, condition=f"{arm}, {lab}", weight_support="airsled")

df = pd.DataFrame(rows)

# ---- checks against the paper
def nsub(x):
    return x["person"].nunique()

fig = lambda k: df[df.subject.str.contains(f"\\|{k}")]  # noqa: E731
checks.published("healthy participants with null-field channel trials (Fig 3)", nsub(fig("f3a")), 220, 0, "Results, Fig 3 (220 participants)")
checks.published("participants who adapted to a force field in Fig 3E-G", nsub(fig("f3b")), 32, 0, "Results, Fig 3 (32 participants)")
checks.published("Fig 4A participants", nsub(fig("f4A")), 11, 0, "Results, Fig 4A (n = 11)")
checks.published("Fig 4B participants", nsub(fig("f4B")), 14, 0, "Results, Fig 4B (n = 14)")
checks.published("stroke patients (Fig 7)", df[df.group == "stroke patient"].person.nunique(), 14, 0, "Results, Fig 7 (n = 14)")
import numpy as np  # noqa: E402
r2 = []
for p, d in fig("f3a").groupby("person"):
    d = d[["hold_force_change_n", "move_force_integral_ns"]].apply(pd.to_numeric, errors="coerce").dropna()
    r2.append(np.corrcoef(d.hold_force_change_n, d.move_force_integral_ns)[0, 1] ** 2)
checks.published("variance in hold force explained by the move force integral, null field (mean %)", 100 * st.mean(r2), 39, 0.5,
                 "Results, Fig 3 ('39 ± 2%', mean ± SEM)")
checks.published("... SEM across participants (%)", 100 * st.stdev(r2) / len(r2) ** 0.5, 2, 0.5, "Results, Fig 3 ('39 ± 2%')")
checks.internal("every row of the converted sheets kept", len(df) == 2777 + 3456 + 682 + 868 + 8000 + 1160 + 1064 + 4104, str(len(df)))
checks.internal("Fig 7: 10 controls and 14 patients (both arms)", df[df.group == "age-matched control"].person.nunique() == 10)

df["subject"] = df["subject"].str.replace("|", "_", regex=False)
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Shadmehr & Krakauer (Johns Hopkins)", data_url="https://osf.io/yc64a/", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="human sheets of the figure source files combined into one table, participant codes replaced, movement duration converted from ms to s; no other value changed.",
    unit_note="trial", 
    how=["One row is one channel trial (hold-force change in N and movement force integral in N s, columns hold_force_change_n and move_force_integral_ns) or, for Fig 6 force-field trials, one trial's endpoint error (cm). Subject is one sequence (one participant in one sheet); person is the participant, identified by the authors' subject ID, which is shared across Figs 3, 4 and 6. Fig 7 has its own IDs (10 controls, 14 stroke patients tested on both arms).",
         "Force orientation 0 is a counter-clockwise curl field (code 3) and 1 clockwise (code 5), as the authors' readme states. The authors do not give field strengths or the robot in the deposit; trial_in_condition is the authors' trial number in the sheet (channel trials are numbered separately from field trials).",
         "Not converted: Figs 1-2 (monkey EMG), the Fig 5 postural-field fits and the Fig 6 per-participant gain tables (derived summaries). Age, sex and handedness are not in the deposit. Fig 3A comes from 15 experiments of the authors; the label keeps their experiment number.",
         "Missing values (movement too slow or not measurable) are left empty."],
    differences=["The paper's '48 ± 3%' (variance in change in hold force explained, 32 participants) is not a check: the deposit does not state the exact model."])
stdlib.release(ID, "force_field_reaching", df.to_dict("records"), meta, checks)
