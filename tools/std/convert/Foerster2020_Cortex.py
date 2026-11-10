"""Foerster, Borghi & Goslin (2020), Cortex: labels and learning to move or use novel tools in virtual reality.

Run from the site folder: python3 -I tools/std/convert/Foerster2020_Cortex.py
Raw files (untrusted data, read only) from the Zenodo copy (10.5281/zenodo.4937820) of the Dryad deposit
10.5061/dryad.gf1vhhmjn: ReactionTimes_dataset.csv (one row per successful trial), SuccessTask_Move_dataset.csv and
SuccessTask_Use_dataset.csv (successes and failures per participant, block and label); the two R scripts were read as text.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Foerster2020_Cortex"
R = stdlib.raw_dir(ID)
ck = stdlib.Checks()
d = pd.read_csv(R / "ReactionTimes_dataset.csv")
mv = pd.read_csv(R / "SuccessTask_Move_dataset.csv")
us = pd.read_csv(R / "SuccessTask_Use_dataset.csv")
EXCL = {6, 34}   # excluded in both authors' R scripts ("very poor performance")

n = len(d)
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": ID, "group": "", "subject": d.Subject.astype(str).to_numpy(),
    "subject_excluded_by_authors": d.Subject.isin(EXCL).to_numpy(), "age": d.Age.to_numpy(),
    "sex": d.Gender.map({"Female": "F", "Male": "M"}).to_numpy(), "handedness": "", "population": "healthy", "setting": "lab",
    "device": "immersive virtual reality (3D tools shown in the headset; the tool meshes are named controllerRenderer in the deposit)",
    "session": 1, "block": ["trial block " + str(b) for b in d.TrialBlock], "phase": "learning", "time_s": "",
    "task": "tool use in virtual reality", "unit": "trial", "condition": d.Task.str.lower() + ", " + np.where(d.Name_learnt == "Yes", "labelled", "unlabelled"),
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
    "tool_id": d.Mesh.str.replace("controllerRenderer_", "", regex=False).to_numpy(), "tool_function": d.Function.str.lower().to_numpy(),
    "tool_name": d.Name.to_numpy(), "labelled": (d.Name_learnt == "Yes").to_numpy(), "action_task": d.Task.str.lower().to_numpy(),
    "trial_number": d.TrialNumber.to_numpy(), "trial_block": d.TrialBlock.to_numpy(),
    "reach_onset_s": d.RTMovement.to_numpy(), "grasp_time_s": d.RTGrasping.to_numpy(), "action_end_s": d.RTAction.to_numpy(),
    "reach_to_grasp_s": d.RTMovementGrasping.to_numpy(), "grasp_to_action_s": d.RTGraspingAction.to_numpy(),
    "trial_success": d.SuccessTask.to_numpy(),
})
t["trial"] = t.groupby("subject").cumcount() + 1
cols = [c[1] for c in stdlib.columns_for("vr_tool_manipulation")]
t["person"] = t["subject"]

# block-level success table in the same subject codes (stdlib numbers subjects by first appearance)
codes = {s: f"{ID}_S{i + 1:04d}" for i, s in enumerate(dict.fromkeys(t.subject))}
sc = pd.concat([mv.assign(task="move"), us.assign(task="use")], ignore_index=True)
assert set(sc.Subject.astype(str)) <= set(codes), "success tables hold a participant missing from the trial table"
succ = pd.DataFrame({"subject": sc.Subject.astype(str).map(codes), "action_task": sc.task, "trial_block": sc.TrialBlock,
                     "labelled": (sc.Name == "Yes"), "n_successful_trials": sc.NumberOfSucceedTrials,
                     "n_failed_trials": sc.NumberOfFailedTrials, "percent_success": sc.PercentOfSuccess})
extra = [("block_success.csv", succ.to_csv(index=False, lineterminator="\r\n")),
         ("block_success_columns.csv", "column,description\r\nsubject,Participant sequence code as in the main table\r\n"
          "action_task,move or use (the deposit's SuccessTask_Move / SuccessTask_Use file)\r\ntrial_block,Trial block 1-3\r\n"
          "labelled,Tools with a label (the deposit's Name = Yes)\r\nn_successful_trials,Successful trials in the block (authors' NumberOfSucceedTrials)\r\n"
          "n_failed_trials,Failed trials in the block (authors' NumberOfFailedTrials)\r\npercent_success,Authors' PercentOfSuccess\r\n")]

# ---------------------------------------------------------------- checks
ck.published("novel tools per participant (6 meshes, each with its own manipulation)", d.Mesh.nunique(), 6, 0,
             "Borghi (2020), J Cogn 3(1), describing this study: 'six novel 3D tools' (the full text of the paper itself was not reachable)")
ck.published("distinct labels (pseudo-words)", d[d.Name_learnt == "Yes"].Name.nunique(), 3, 0, "Abstract: labelled and unlabelled tools; three labels (SNI, UNT, LUM); which tool carries which label differs between participants")
ck.internal("10,392 rows kept; 39 participant codes", len(t) == 10392 and t.subject.nunique() == 39)
ck.internal("RT differences reproduce from the other columns (< 0.5 ms)",
            bool(((d.RTMovementGrasping - (d.RTGrasping - d.RTMovement)).abs().dropna() < 5e-4).all()
                 and ((d.RTGraspingAction - (d.RTAction - d.RTGrasping)).abs() < 5e-4).all()))
m_ = d.RTMovement.notna()
bad = int((d.RTMovement[m_] > d.RTGrasping[m_]).sum() + (d.RTGrasping > d.RTAction).sum())
ck.internal("reach onset <= grasp <= action end on at least 97% of rows", bad / len(d) < 0.03, f"{bad} rows violate the order")
ck.internal("the participants of the success tables are 38 of the trial table's 39, with 2 excluded in both authors' scripts (6 and 34)",
            sc.Subject.nunique() == 38 and EXCL <= set(sc.Subject))
ck.internal("success percentage = successes / (successes + failures)", bool((((sc.NumberOfSucceedTrials / (sc.NumberOfSucceedTrials + sc.NumberOfFailedTrials)) * 100).round(2) - sc.PercentOfSuccess).abs().max() < 0.02))
ck.internal("authors' analysed sample after excluding subjects 6 and 34", d[~d.Subject.isin(EXCL)].Subject.nunique() == 37, "37 participants in the reaction-time analysis")
print({k: v for k, v in zip(["rows", "subjects", "excluded"], [len(t), t.subject.nunique(), len(EXCL)])})

meta = dict(
    paper=dict(authors="Foerster FR, Borghi AM, Goslin J", year=2020, title="Labels strengthen motor learning of new tools",
               venue="Cortex 129:1-10", doi="10.1016/j.cortex.2020.04.006"),
    lab="Goslin (University of Plymouth)", data_url="https://doi.org/10.5061/dryad.gf1vhhmjn",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Participant ids replaced by codes; the test date (Date) and the authors' centred and z-scored copies of the times "
             "(columns ending .c and .z) were not copied; Female/Male became F/M; the authors' success tables went into block_success.csv "
             "with the same participant codes."),
    how=["One row is one successful trial of learning to handle novel tools in virtual reality: the participant reached for a tool, "
         "grasped it and then moved it to another location or used it (its learned manipulation), over three trial blocks. "
         "Six tools, three with a pseudo-word label and three without. The times (seconds) are the authors' RTMovement (reach onset), RTGrasping "
         "(grasp) and RTAction (end of the action), all from one origin that the deposit does not state, plus the two intervals between them.",
         "subject_excluded_by_authors marks the two participants (ids 6 and 34 in the deposit) whom both R scripts remove for very poor performance. "
         "The trial table holds successful trials only; block_success.csv gives successes and failures per participant, block, move/use and label. "
         "On about 2.6% of rows reach onset is later than the grasp time or the grasp later than the action end, kept as given. One participant has an implausible age (53 years) in the deposit and one an id far outside the others; both are kept as given."],
    unit_note="successful trial",
    differences=["The full text of the paper could not be read when this file was made (publisher and repository pages refused automated access), so no "
                 "result statistic of the paper was reproduced; the published checks are the tool design (six tools, three labelled) reported in a "
                 "secondary description of the study. Paper title, venue and DOI were checked against Crossref."],
)
stdlib.release(ID, "vr_tool_manipulation", t[cols].to_dict("records"), meta, ck, extra_files=extra)
