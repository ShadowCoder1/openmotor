"""Wong, Eyssalenne, Carter & Therrien (2024), eNeuro: state estimation when stationary or moving (OSF wt6hz).
Raw (untrusted, read only): the authors' extracted per-participant trial CSVs, copied to raw_extra/Wong2024_eNeuro/ as
E1_static_CNxx.csv, E1_dynamic_CNxx.csv, E2_hstatic_CNxx.csv, E3_dmc_CNxx.csv (81 files). The raw 1 kHz recordings (zip files of
50 to 160 MB per person) are not converted; the CSVs hold the per-trial measures the authors extracted from them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wong2024_eNeuro"
RAW = stdlib.raw_dir(ID)
names = [c[1] for c in stdlib.columns_for("position_report_reaching")]
checks = stdlib.Checks()
VAR = {"static": ("Exp1", "static"), "dynamic": ("Exp1", "dynamic"), "hstatic": ("Exp2", "hstatic"), "dmc": ("Exp3", "dynamic_dmc")}
AMAP = dict(x_endpoint="x_m", x_midpoint="x_m", y_endpoint="y_m", y_midpoint="y_m", x_max_deviation="x_max_deviation_m",
            y_max_deviation="y_max_deviation_m", report_position="report_position_m", ideal_report="ideal_report_m",
            average_unimodal="average_unimodal_m", normalization_factor="normalization_factor_m", bias="bias_m",
            bias_normalized="bias_normalized", movement_time="movement_time_ms", movement_vpeak="peak_velocity_m_s",
            v_midpoint="velocity_midpoint_m_s", v_maxdev="velocity_max_deviation_m_s", offset_1="offset_1_m", offset_2="offset_2_m",
            movement_errX="movement_err_x_m", movement_errY="movement_err_y_m", err="err_flag", isbad="isbad")
rows, persons = [], {}
per = {}
for f in sorted(RAW.glob("E?_*_CN*.csv")):
    tag = f.name.split("_")[1]
    exp, variant = VAR[tag]
    d = pd.read_csv(f)
    sid = str(d.subject_ID.iloc[0])
    assert (d.subject_ID == sid).all() and sid in f.name
    persons.setdefault(exp, set()).add(sid)
    per[(tag, sid)] = d
    for _, t in d.iterrows():
        r = dict.fromkeys(names, "")
        bimodal = t.trialtype in ("VPv", "VPp")
        r.update(dataset=ID, study=ID, experiment=f"{exp}_{variant}", group="", subject=f"{sid}_{variant}", person=sid,
                 subject_excluded_by_authors=False, population="healthy", setting="lab", device="KINARM exoskeleton robot",
                 session=1, block=f"block{int(t.block)}", trial=int(t.trialNumber), phase="baseline" if int(t.block) == 1 else "test",
                 task="reaching", unit="trial", condition=str(t.trialtype), pert_type="cue_conflict" if bimodal else "none",
                 pert_value=6 if bimodal else 0, pert_unit="cm", feedback="altered" if bimodal else "normal",
                 excluded_by_authors=bool(t.get("isbad", 0) == 1),
                 task_variant=variant, task_protocol=t.task_protocol, first_task=t.get("first_task", ""), trial_orig=int(t.trialNumber),
                 block_orig=int(t.block), trial_type=t.trialtype, probed_modality=t.probed_modality, target_index=int(t.tpidx))
        for k, v in AMAP.items():
            if k in d.columns and not (isinstance(t[k], float) and np.isnan(t[k])):
                r[v] = t[k] if k not in ("isbad", "err") else bool(t[k])
        rows.append(r)

allp = persons["Exp1"]
checks.published("Experiment 1 participants analysed", len(allp), 24, 0, "Methods, Participants, Experiment 1")
checks.published("Experiment 2 participants", len(persons["Exp2"]), 20, 0, "Methods, Participants, Experiment 2")
checks.published("Experiment 3 participants analysed", len(persons["Exp3"]), 15, 0, "Methods, Participants, Experiment 3")
ns = len([k for k in per if k[0] == "static"]); nd = len([k for k in per if k[0] == "dynamic"])
checks.published("Experiment 1 people with the static task (22 did both, one only static)", ns, 23, 0, "Methods, Participants, Experiment 1")
checks.published("Experiment 1 people with the dynamic task (22 did both, one only dynamic)", nd, 23, 0, "Methods, Participants, Experiment 1")


def mode_count(tag, col, val, blk=None):
    c = []
    for (tg, s), d in per.items():
        if tg == tag:
            x = d[d.block == blk] if blk else d
            c.append(int((x[col] == val).sum()))
    return pd.Series(c).mode().iloc[0]


checks.published("static task, unimodal block: trials per person (modal)", pd.Series([int((d.block == 1).sum()) for (tg, s), d in per.items() if tg == "static"]).mode().iloc[0], 120, 0, "Methods, static task")
checks.published("static task, multimodal block: VPv trials per person (modal)", mode_count("static", "trialtype", "VPv", 2), 60, 0, "Methods, static task")
checks.published("static task, multimodal block: VPp trials per person (modal)", mode_count("static", "trialtype", "VPp", 2), 60, 0, "Methods, static task")
checks.published("static task, multimodal block: V trials per person (modal)", mode_count("static", "trialtype", "V", 2), 24, 0, "Methods, static task")
checks.published("dynamic task, unimodal block: trials per person (modal)", pd.Series([int((d.block == 1).sum()) for (tg, s), d in per.items() if tg == "dynamic"]).mode().iloc[0], 60, 0, "Methods, dynamic task")
checks.published("dynamic task, multimodal block: VPv trials per person (modal)", mode_count("dynamic", "trialtype", "VPv", 2), 60, 0, "Methods, dynamic task")
pv = np.mean([d.movement_vpeak.mean() for (tg, s), d in per.items() if tg == "dynamic"])
checks.internal("dynamic task: mean peak velocity near the 0.187 m/s the Methods give for the arc movements (within 0.05)", abs(pv - 0.187) < 0.05, f"{pv:.3f}")
checks.internal("rows kept", len(rows) == sum(len(d) for d in per.values()), str(len(rows)))

paper = dict(authors="Wong AL, Eyssalenne AN, Carter L, Therrien AS", year=2024,
             title="Different Sensory Information Is Used for State Estimation when Stationary or Moving",
             venue="eNeuro 11(9):ENEURO.0357-23.2024", doi="10.1523/ENEURO.0357-23.2024")
meta = dict(
    study=ID, paper=paper, lab="Wong & Therrien (Moss Rehabilitation Research Institute)", data_url="https://osf.io/wt6hz/",
    license="none stated", license_url="",
    changes="converted to the OpenMotor format; participant codes replaced; start times and the 1 kHz recordings not included; all per-trial "
            "values kept in the authors' units (m, ms, m/s); the authors' bias columns kept as stored.",
    unit_note="trial (one position-report movement) of one participant",
    how=[
        "One row is one trial of a state-estimation task on a KINARM exoskeleton (arm in the horizontal plane, hand hidden): the person "
        "reports where a visual cue (V), their felt hand (P) or both (bimodal VPv, VPp: the cues disagree by 6 cm) was, by moving the hand "
        "there (static task: movement endpoint; dynamic task: the position of an arc at its midpoint or maximum deviation). Block 1 holds "
        "unimodal trials, block 2 mixes unimodal and bimodal trials. Experiment 1 has a static and a dynamic task (24 people, 22 did both), "
        "Experiment 2 an orthogonal-offset static task (hstatic, 20 people), Experiment 3 a same-side-curvature dynamic task (15 people).",
        "Positions are metres in the apparatus frame, times are ms, as stored. The reach90 format does not apply: there is no hand direction relative to "
        "a target, the reports are positions, so a task module holds the authors' report, ideal report, bias (report minus the unimodal "
        "average, positive toward the other modality) and normalized bias.",
        "Not copied: the targets and control-point coordinates (tgts_*, tgt_*, ctlpts_*, vmidpt_*, pmidpt_*) of Experiments 2 and 3, the "
        "start time of each session, and the raw recordings. The deposit's bias is 0 on all unimodal trials by construction. "
        "Age and sex are not per participant in the deposit (paper: Exp 1 mean age 26.0, Exp 2 23.3, Exp 3 25.1).",
        "excluded_by_authors is the authors' isbad flag where the file has one (Experiments 2 and 3); Experiment 1 files have no flag.",
    ],
    differences=[f"The Methods give a peak velocity of 0.187 m/s for the dynamic task's arcs (the speed of the displayed arc, as we read it); the mean of the participants' own peak velocities in the deposit is {pv:.3f} m/s, so it is not the same quantity."],
)
stdlib.release(ID, "position_report_reaching", rows, meta, checks)
