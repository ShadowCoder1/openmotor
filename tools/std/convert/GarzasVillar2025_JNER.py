"""Garzas-Villar et al. (2025), J NeuroEngineering Rehabil: haptic guidance and personality traits in robot-assisted motor
learning. 40 adults steered a virtual pendulum with a haptic robot to hit targets on approaching walls; one row per wall,
in the 'game' module.

Run from the site folder: python3 -I tools/std/convert/GarzasVillar2025_JNER.py
Raw files (untrusted data, read only) in raw_extra/GarzasVillar2025_JNER/, from https://zenodo.org/records/16539273 (CC BY 4.0):
  SummarizedKinData.csv                       the authors' merged and discretised kinematic data (one row per wall), 40 participants
  DemographicsAndPersonalityTraitsData.csv    age range, gender and questionnaire items (42 people)
  AgencyMotivationData.csv                    agency / motivation / robot-interaction questionnaires (3 stages)
  DataframeDescription.rtf                    variable definitions
Not carried over: the 40 per-participant RAR archives of 1.67 kHz kinematics (about 7.6 GB, over the 5 GB limit); the summary
file is what the authors used for their statistics. Questionnaire dates and language columns are left out.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "GarzasVillar2025_JNER"
RAW = stdlib.raw_dir(ID)
SRC = "Methods, J NeuroEngineering Rehabil 22:238"
checks = stdlib.Checks()
differences = []

k = pd.read_csv(RAW / "SummarizedKinData.csv")
dem = pd.read_csv(RAW / "DemographicsAndPersonalityTraitsData.csv")
agn = pd.read_csv(RAW / "AgencyMotivationData.csv")

ids = list(dict.fromkeys(k.ID.tolist()))  # order of first appearance = order of the person codes
checks.internal("rows of one participant are contiguous in the file", bool((k.ID != k.ID.shift()).sum() == len(ids)))
checks.internal("|Error| equals the authors' ErrorAbs", bool(np.allclose(k.Error.abs(), k.ErrorAbs, atol=1e-6)))
checks.internal("Error = targetPos - ballPosY (sign inferred from the data)", bool(np.allclose(k.Error, k.targetPos - k.ballPosY, atol=2e-5)),
                f"max abs gap {float((k.Error - (k.targetPos - k.ballPosY)).abs().max()):.2e}")
checks.internal("Condition label agrees with Stage and Task columns", bool(
    (k.Condition.where(k.Condition == "training", k.Stage + "_" + k.Task).str.replace("^bl_", "baseline_", regex=True) == k.Condition).all()))
checks.internal("20 walls per set, indices 0-19, in order", bool((k.groupby(["ID", "Condition", "cIndex", "sIndex"]).wIndex.apply(
    lambda s: list(s) == list(range(20)))).all()))
checks.internal("20 participants in each group", k.groupby("ID").Group.first().value_counts().to_dict() == {"E": 20, "C": 20})
checks.internal("every analysed participant has a demographics row and 3 agency/motivation stages",
                set(ids) <= set(dem.ID) and bool(agn[agn.QID.isin(ids)].groupby("QID").size().eq(3).all() and agn.QID.isin(ids).sum() == 3 * len(ids)))

# ---------------------------------------------------------------- published numbers
passed = []


def chk(what, computed, reported, tol, source):
    ok = abs(float(computed) - float(reported)) <= tol
    if ok:
        checks.published(what, computed, reported, tol, source)
        passed.append(what)
    else:
        differences.append(f"{what}: the paper reports {reported:g}, the shared rows give {float(computed):.4g} ({source}).")
    print(f"  {'ok ' if ok else 'DIFF'} {what}: {float(computed):.4g} vs {reported:g}")


per = k.groupby("ID")
chk("participants analysed", k.ID.nunique(), 40, 0, f"Participants, {SRC}")
chk("people who performed the experiment (analysed + 2 excluded)", len(dem) , 42, 0, f"Participants (forty-two performed), {SRC}")
d40 = dem[dem.ID.isin(ids)]
chk("analysed participants identifying as female", (d40.Gender == "Female").sum(), 19, 0, f"Participants, {SRC}")
chk("analysed participants identifying as male", (d40.Gender == "Male").sum(), 21, 0, f"Participants, {SRC}")
chk("walls per set", per.wIndex.max().min() + 1, 20, 0, f"Study protocol (a set comprised 20 targets), {SRC}")
chk("training sets per participant (two rounds of 15)", k[k.Condition == "training"].groupby("ID").size().mean() / 20, 30, 0, f"Study protocol, {SRC}")
for cond in ["baseline_mn", "baseline_t1", "baseline_t2", "str_mn", "str_t1", "str_t2", "ltr_mn", "ltr_t1", "ltr_t2"]:
    pass
chk("sets per task in baseline (two sets of each of the three tasks)", k[k.Stage == "bl"].groupby(["ID", "Task"]).size().mean() / 20, 2, 0,
    f"Study protocol, {SRC}")
chk("sets per task in long-term retention (structured like baseline)", k[k.Stage == "ltr"].groupby(["ID", "Task"]).size().mean() / 20, 2, 0,
    f"Study protocol, {SRC}")
chk("distinct target positions on the walls (centre, 0.12 m left and right)", k.targetPos.round(4).nunique(), 3, 0, f"The pendulum game, {SRC}")
chk("largest target offset from the centreline (m)", k.targetPos.abs().max(), 0.12, 1e-6, f"The pendulum game, {SRC}")
gap = k.groupby(["ID", "Condition", "cIndex", "sIndex"]).rTime.diff().dropna()
chk("median time between consecutive walls (walls 1 m apart approaching at 1 m/s), s", gap.median(), 1.0, 0.05, f"The pendulum game, {SRC}")
chk("lowest score (scores range 0-100)", k.Score.min(), 0, 1, f"The pendulum game, {SRC}")
chk("highest score (scores range 0-100)", k.Score.max(), 100, 1, f"The pendulum game, {SRC}")
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}")
assert passed, "no published number reproduced"

# crude check of the LMM-based error reduction the paper gives (not a check: it is a model estimate)
ma = k[k.Task == "mn"].groupby(["ID", "Stage"]).ErrorAbs.mean().unstack()
crude = float(((ma.bl - ma.str) / ma.bl).mean() * 100)
differences.append(
    f"The paper's headline error reductions (38% baseline to short-term retention, 39% to long-term retention, main task) are linear "
    f"mixed-model estimates for an average participant; the raw mean of the per-participant relative reductions in absolute error is "
    f"{crude:.1f}% (baseline to short-term retention), so the model figure was not used as a check.")
print(differences[-1])

# ---------------------------------------------------------------- rows
dem = dem.set_index("ID")
SEX = {"Female": "F", "Male": "M"}
GRP = {"E": "Experimental (haptic guidance in training)", "C": "Control (no guidance)"}
TASK = {"mn": "main task", "t1": "position transfer task", "t2": "dynamics transfer task"}
STAGE = {"bl": "baseline", "str": "short-term retention", "ltr": "long-term retention"}
rows = []
trial = {}
for r in k.itertuples(index=False):
    trial[r.ID] = trial.get(r.ID, 0) + 1
    train = r.Condition == "training"
    if train:
        phase = "learning"
    elif r.Stage == "bl":
        phase = "baseline"
    else:
        phase = "test" if r.Task == "mn" else "transfer"
    rows.append(dict(
        dataset=ID, study=ID, experiment="pendulum game", group=GRP[r.Group], subject=f"S{r.ID}", person=f"S{r.ID}",
        subject_excluded_by_authors=False, age="", sex=SEX[dem.at[r.ID, "Gender"]], handedness="", population="healthy", setting="lab",
        device="Force Dimension haptic robot, virtual pendulum game on a monitor", session=2 if r.Stage == "ltr" else 1,
        block=r.Condition, trial=trial[r.ID], phase=phase, time_s="", task="robot-assisted training", unit="trial",
        condition="training (main task)" if train else f"{STAGE[r.Stage]}, {TASK[r.Task]}", pert_type="none", pert_value=0, pert_unit="",
        feedback="normal", excluded_by_authors=False,
        game="virtual pendulum steered through a haptic robot", attempt=int(r.wIndex), score=r.Score, score_unit="points (0-100) for this wall",
        outcome_error=r.Error, outcome_error_unit="m", set_index=int(r.sIndex), protocol_block_code=int(r.cIndex),
        target_position_m=r.targetPos, ball_position_m=r.ballPosY, interaction_force_abs_mean_n=r.intForceAbsMean,
        interaction_force_sd_n=r.intForceStd, robot_time_s=r.rTime))

# participants
pcols = [("age_range", "text", "", "Age range in years as the authors give it (the deposit shares ranges only)."),
         ("gender", "text", "", "Self-reported gender as in the deposit (Female, Male)."),
         ("education", "text", "", "Highest studies or training (deposit column StudiesOrTraining).")]
pcols += [(f"ex_{c}", "text", "", f"Answer to questionnaire item EX_{c} in the deposit's wording; the deposit gives no item text. Values like '01/Mar' and '04/Jun' are the deposit's, apparently ranges '1-3' and '4-6' turned into dates by a spreadsheet.") for c in ["1a", "1b", "2a", "2b", "3", "4", "5"]]
pcols += [(f"hand_{i}", "text", "", f"Answer to item HAND_{i} of the handedness questionnaire (the paper used the short-form Edinburgh Handedness Inventory), as worded in the deposit; which hand is preferred for the item is not stated in the deposit.") for i in range(1, 11)]
pcols += [(f"aq_{i}", "integer", "1-7", f"Item AQ_{i}, answered 1-7, of the autotelic personality questionnaire (Transform of Challenge and Transform of Boredom subscales; the deposit does not say which item belongs to which subscale).") for i in range(1, 8)]
pcols += [(f"hq_{i}", "integer", "1-7", f"Item HQ_{i}, answered 1-7, of the Hexad gaming-style questionnaire (Achiever and Free Spirit subscales; the deposit does not say which item belongs to which).") for i in range(1, 11)]
pcols += [(f"loc_{i}", "text", "", f"Statement chosen in forced-choice item LOC_{i} of the Locus of Control questionnaire, as in the deposit.") for i in range(1, 30)]
part = []
for pid in ids:
    d = dem.loc[pid]
    row = dict(person=f"S{pid}", age_range=d.Age.replace(" years old", ""), gender=d.Gender, education=d.StudiesOrTraining)
    for c in ["1a", "1b", "2a", "2b", "3", "4", "5"]:
        row[f"ex_{c}"] = "" if pd.isna(d[f"EX_{c}"]) else str(d[f"EX_{c}"])
    for i in range(1, 11):
        row[f"hand_{i}"] = "" if pd.isna(d[f"HAND_{i}"]) else str(d[f"HAND_{i}"])
    for i in range(1, 8):
        row[f"aq_{i}"] = int(d[f"AQ_{i}"])
    for i in range(1, 11):
        row[f"hq_{i}"] = int(d[f"HQ_{i}"])
    for i in range(1, 30):
        row[f"loc_{i}"] = str(d[f"LOC_{i}"])
    part.append(row)
part = pd.DataFrame(part)
pcode = {pid: f"{ID}_P{i + 1:04d}" for i, pid in enumerate(ids)}  # release() codes persons in order of first appearance

# agency / motivation questionnaires, long form, analysed participants only
a = agn[agn.QID.isin(ids)].copy()
a["person"] = a.QID.map(pcode)
a["_o"] = a.QID.map({p: i for i, p in enumerate(ids)})
a["_s"] = a.Stage.map({"BL": 0, "TR-C": 1, "TR-E": 1, "LTR": 2})
a = a.sort_values(["_o", "_s"], kind="stable")
a = a[["person", "Stage", "QAgency1", "QAgency2"] + [f"Q12_{i}" for i in range(1, 5)] + [f"Q13_{i}" for i in range(1, 5)] +
      [f"Q14_{i}" for i in range(1, 6)] + ["RI_1", "RI_2", "RI_3"]]
a.columns = [c.lower() for c in a.columns]
buf = io.StringIO()
a.to_csv(buf, index=False, lineterminator="\r\n")
desc = ("person: participant code as in the main table. stage: BL baseline, TR-C training (Control), TR-E training (Experimental), LTR "
        "long-term retention. qagency1, qagency2: adapted agency items ('It seemed like I was in control of the pendulum'; 'It seemed "
        "like I was causing the movements of the pendulum'). q12_1 ... q14_5: Intrinsic Motivation Inventory items (subscale "
        "assignments are in the authors' DataframeDescription.rtf; q14_5 is a boring-activity item to be reversed). ri_1, ri_2: robot "
        "interaction, -3 disturbing/restricting to 3 helpful/permitting; ri_3: robot interaction frustrated me, 1-7; ri_* asked only of the "
        "Experimental group at the training stage.")
buf2 = io.StringIO()
buf2.write("file,description\r\nquestionnaires_agency_motivation.csv,\"" + desc.replace('"', "'") + "\"\r\n")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k_: paper[k_] for k_ in ("authors", "year", "title", "venue", "doi")},
    lab="Marchal-Crespo (Delft University of Technology)", data_url="https://zenodo.org/records/16539273", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format from the authors' summarised kinematic file: participant numbers replaced by codes, "
             "Condition/Task/Stage/Group written as block, condition, phase and group, Error kept as outcome_error, targetPos, ballPosY, "
             "the interaction-force mean and SD and rTime kept in module columns; ErrorAbs (equal to |Error|) and the redundant Task, "
             "Stage and Condition fragments were not repeated. Questionnaire dates and language columns were left out and the age ranges "
             "and questionnaire answers went to participants.csv; the agency and motivation questionnaires are in "
             "questionnaires_agency_motivation.csv. The 40 per-participant RAR archives of 1.67 kHz kinematics (about 7.6 GB) were not used."),
    how=[
        "One row is one wall of the pendulum game: participants moved the end-effector of a haptic robot to steer a virtual pendulum so that "
        "its mass crossed each approaching wall (1 m apart, 1 m/s, so about one per second) as near as possible to a target line; sets are "
        "20 walls (attempt is the wall 0-19 and set_index the set, both the authors' indices from 0). Rows keep the authors' file order, "
        "participant by participant; trial counts them from 1.",
        "outcome_error (m) is the authors' Error, which in the shared rows equals target_position_m minus ball_position_m along the "
        "horizontal axis y (checked on all rows); the direction of positive y on screen is not stated in the deposit. score is the 0-100 "
        "score shown for that wall. interaction_force_* are the authors' estimated human-robot interaction force (mean absolute value and "
        "SD) in the window between the midpoints of consecutive walls, in N. robot_time_s is the robot stopwatch (rTime).",
        "block is the authors' Condition label: baseline_*, training, str_* (short-term retention, 'STR') and ltr_* (long-term retention, "
        "'LTR'); *_mn is the main task, *_t1 the position transfer task (targets re-located) and *_t2 the dynamics transfer task "
        "(pendulum rod shortened by 70%). protocol_block_code is the authors' cIndex (day 1: 1-3 baseline, 4-5 the two training rounds, "
        "7-9 short-term retention; day 2: 1-3 long-term retention, same task order). Session 1 is the first day, session 2 the return day "
        "1-3 days later. Baseline and retention have two sets (40 walls) of each task; training has 30 sets of the main task.",
        "The Experimental group received haptic guidance during training, except in 'catch' sets (the first set and once every five sets), "
        "and both groups saw mirrored target sets in training; the shared rows do not flag catch or mirrored sets, so pert_type is 'none' on "
        "every row and the guidance is carried by group. The washout set after training is not in the shared summary file. "
        "Participants 1 and 42 of the original numbering were excluded by the authors (errors three SDs above the mean; corrupted recording) "
        "and are not in the data, so no row is marked excluded.",
        "Age is shared as ranges only (participants.csv). Handedness items are in participants.csv as worded in the deposit; handedness is "
        "left empty in the main table because the deposit gives no scoring.",
    ],
    unit_note="wall of the pendulum game", differences=differences)
stdlib.release(ID, "game", rows, meta, checks, participants=part, participant_columns=pcols,
               extra_files=[(f"questionnaires_agency_motivation.csv", buf.getvalue().encode()),
                            (f"questionnaires_agency_motivation_columns.csv", buf2.getvalue().encode())])
