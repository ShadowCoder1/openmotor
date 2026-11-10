"""Wood, Eyssalenne, Therrien & Wong (2026), npj Parkinson's Disease: effort and reward tasks in a Kinarm robot, people with
Parkinson's disease (on and off medication) and age-matched controls. One row per trial, three tasks in one table.

Run from the site folder: python3 -I tools/std/convert/Wood2026_npjParkinsonsDis.py
Raw files (untrusted data, read only) in raw_extra/Wood2026_npjParkinsonsDis/ from https://osf.io/krw49/ (data folder):
  force_matching.csv, reward_utility.csv, effort_reward.csv, demographics.csv
README.docx, supp.docx (supplementary results with a table of the participants with PD) and the authors' data-organisation
notebooks (read as JSON text, never run) document the variables. The article's full text was not reachable (publisher page
returns the abstract only), so published numbers come from the supplementary table. The large posterior files are not converted.
Participant IDs in the deposit are numbered one higher than in the supplementary table (the ages and sexes line up), so the
supplementary table is used only for counts and summaries.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wood2026_npjParkinsonsDis"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for fn in ("force_matching.csv", "reward_utility.csv", "effort_reward.csv", "demographics.csv"):
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])

fm = pd.read_csv(RAW / "force_matching.csv", index_col=0)
ru = pd.read_csv(RAW / "reward_utility.csv", index_col=0)
er = pd.read_csv(RAW / "effort_reward.csv", index_col=0)
dem = pd.read_csv(RAW / "demographics.csv", index_col=0)

# ---------------------------------------------------------------- numbers from the supplementary table (PD participants)
SUPP_AGE = [66, 70, 67, 74, 75, 72, 80, 73, 71, 75, 69, 59, 65, 78, 79, 71, 69, 82, 82, 67, 67, 69, 68, 73, 69, 87, 63, 61, 63, 69, 77, 74, 74, 60, 66, 60, 84]
SUPP_SEX = "FMMMMFMMFMMMMFMMFMMFMMMMMFMFFMMFMMFMF"
SUPP_NO_ON = {18, 36, 37}   # participants with only an OFF row in the table
SUPP_NO_OFF = {6, 8}        # participants with only an ON row
SUPP_LEFT_HAND = {13, 17}
assert len(SUPP_AGE) == len(SUPP_SEX) == 37

pd_dem = dem[dem.group == "PD"]
pd_people = pd_dem.drop_duplicates("sid")
checks.published("participants with Parkinson's disease", pd_people.sid.nunique(), 37, 0, "Supplementary table 1 (PD-001 to PD-037)")
checks.published("PD participants with an ON session", (pd_dem.med_status == "on").sum(), 37 - len(SUPP_NO_ON), 0, "Supplementary table 1 (ON values)")
checks.published("PD participants with an OFF session", (pd_dem.med_status == "off").sum(), 37 - len(SUPP_NO_OFF), 0, "Supplementary table 1 (OFF values)")
checks.published("PD participants with both sessions (complete)", pd_dem[pd_dem.complete].sid.nunique(), 37 - len(SUPP_NO_ON) - len(SUPP_NO_OFF), 0,
                 "Supplementary table 1 (rows with both ON and OFF)")
checks.published("PD participants who are female", (pd_people.sex == "F").sum(), SUPP_SEX.count("F"), 0, "Supplementary table 1 (sex)")
checks.published("PD participants who are left-handed", (pd_people.hand == "left").sum(), len(SUPP_LEFT_HAND), 0, "Supplementary table 1 (handedness)")
checks.published("mean age of PD participants", pd_people.age.mean(), float(np.mean(SUPP_AGE)), 0.005, "Supplementary table 1 (ages, mean computed from the table)")

# ---------------------------------------------------------------- internal checks
ctrl = dem[dem.group == "control"]
checks.internal("39 controls, each with one session", ctrl.sid.nunique() == 39 and len(ctrl) == 39)
for name, t in (("force_matching", fm), ("reward_utility", ru), ("effort_reward", er)):
    checks.internal(f"{name}: same people and sessions as demographics",
                    set(zip(t.sid, t.med_status)) == set(zip(dem.sid, dem.med_status)))
checks.internal("reward utility: 135 trials per session; effort-reward 125; is_valid 0 only when no target was reached",
                bool((ru.groupby(["sid", "med_status"]).size() == 135).all() and (er.groupby(["sid", "med_status"]).size() == 125).all()
                     and (ru.orig_choice[ru.is_valid == 0] == 0).all() and (ru.orig_choice[ru.is_valid == 1] != 0).all()))
checks.internal("force matching goals 20/40/60/80 % MVC (and 50 on 6 trials)", set(fm.goals) == {20, 40, 50, 60, 80})

info = dem.set_index(["sid", "med_status"])
persons = {s: i for i, s in enumerate(dem.sid.drop_duplicates())}
MODCOLS = [c[0] for c in stdlib.load_module("effort_reward_tasks")]


def base(sid, med, task, arm_label):
    r = info.loc[(sid, med)]
    ctl = med == "control"
    ft = r["first_test"]
    session = 1 if (ctl or ft == med or ft != ft) else 2
    return {
        "dataset": ID, "study": "Wood 2026 npj Parkinsons Dis", "experiment": task, "group": "control" if ctl else "PD",
        "subject": f"{sid}|{med}|{task}{arm_label}", "person": sid, "subject_excluded_by_authors": False,
        "age": r["age"], "sex": r["sex"], "handedness": r["hand"], "population": "healthy older adults" if ctl else "Parkinson's disease",
        "setting": "lab", "device": "Kinarm exoskeleton robot", "session": session, "phase": "test", "time_s": "",
        "feedback": "normal", "pert_type": "none", "pert_value": 0, "pert_unit": "", "med_status": med,
        "both_sessions_done": bool(r["complete"]), "task_name": task,
    }


out = []
for r in fm.itertuples(index=False):
    o = base(r.sid, r.med_status, "force_matching", f"|{r.arm}")
    o.update(block=f"{r.arm}_arm", trial=r.trial_num, task="isometric force matching", unit="trial", condition=f"goal {r.goals} % MVC",
             excluded_by_authors=False, active_arm=r.active_arm, fm_arm=r.arm, fm_mvc=r.mvc, fm_goal_pct_mvc=r.goals,
             fm_initial_force_pct_mvc=r.init_forces, fm_match_force_pct_mvc=r.match_forces, fm_survey_report=r.survey_report,
             fm_initial_force_sd=r.init_force_var, fm_initial_stabilize_time=r.init_stabilize_time, fm_initial_hold_time=r.init_hold_time,
             fm_match_force_sd=r.match_force_var, fm_match_rise_time=r.match_rise_time, fm_match_hold_time=r.match_hold_time)
    out.append(o)
for r in ru.itertuples(index=False):
    o = base(r.sid, r.med_status, "reward_utility", "")
    o.update(block="reward_utility", trial=r.trial_num, task="reaching", unit="trial", condition=f"sure {r.fixed_ev:g}, gamble ev {r.gamble_ev:g}",
             excluded_by_authors=not bool(r.is_valid), active_arm=r.active_arm, ru_fixed_value=r.fixed_ev, ru_gamble_expected_value=r.gamble_ev,
             ru_gamble_amount=r.gamble_amount, ru_orig_choice=r.orig_choice, ru_final_choice_gamble=bool(r.choices_end),
             ru_initial_choice_gamble=bool(r.init_choices), ru_mt_ms=r.mt, ru_rt_ms=r.rt, ru_peak_velocity=r.vpeak, ru_amplitude=r.amp)
    out.append(o)
for r in er.itertuples(index=False):
    o = base(r.sid, r.med_status, "effort_reward", "")
    o.update(block="effort_reward", trial=r.trial_num, task="effort-reward decisions", unit="trial",
             condition=f"force {r.force:g}, reward {r.reward:g}", excluded_by_authors=False, active_arm=r.active_arm,
             er_force_level=r.force, er_reward_level=r.reward, er_accept=r.accept)
    out.append(o)
checks.internal("every row of the three files kept", len(out) == len(fm) + len(ru) + len(er) == 3417 + 14580 + 13500)

meta = dict(
    study="Wood et al. 2026",
    paper=dict(authors="Wood JM, Eyssalenne AN, Therrien AS, Wong AL", year=2026,
               title="Impaired effort perception, not reward sensitivity, impacts the effort-reward trade-off in Parkinson's disease",
               venue="npj Parkinsons Dis", doi="10.1038/s41531-026-01528-z"),
    lab="Wong / Therrien (Moss Rehabilitation Research Institute)", data_url="https://osf.io/krw49/",
    license="CC BY-NC 4.0", license_url="https://creativecommons.org/licenses/by-nc/4.0/",
    changes=("Converted to the OpenMotor format with participant IDs replaced by codes; the three trial-level tables (force matching, reward "
             "utility, effort-reward) were stacked into one long table with a task_name column; age, sex and handedness were attached from the "
             "deposit's demographics table. The deposit's posterior-sample files and its other demographic and clinical columns were not converted."),
    how=[
        "One row is one trial of one of three tasks done in a Kinarm exoskeleton robot by 37 people with Parkinson's disease (on and off "
        "dopamine medication, 32 of them in both sessions) and 39 age-matched controls. Rows are in the deposit's order: all force-matching "
        "trials, then reward-utility, then effort-reward. Subject is one participant in one session and task (and arm, in force matching); person is the participant.",
        "force_matching: isometric force matching with the same or the opposite arm at goal forces of 20-80 percent of the participant's maximum "
        "voluntary contraction (fm_ columns; forces are percent MVC as the authors computed them). The deposit lists these rows sorted by goal force; trial is the authors' trial number, "
        "which restarts for each arm. The authors do not state the units of the plateau times and variabilities (times are in the authors' frame or ms counts).",
        "reward_utility: the participant reached to one of two 8 cm targets, a sure reward or a gamble, after a go signal 1800 ms after the targets "
        "appeared (ru_ columns; excluded_by_authors is true where no target was reached, 657 trials). Time and distance units are assumed (ms, m/s, m) from magnitudes. "
        "effort_reward: accept or reject an offer of a force level and a reward level (er_ columns). This is a new module (effort_reward_tasks), not reach90: "
        "there is no target-relative hand angle.",
        "Session is 1 for the session the participant did first (demographics first_test) and 2 for the other; controls have one session.",
    ],
    unit_note="trial",
    differences=[
        "The article's full text could not be read (the publisher page gave the abstract only), so the checks use the supplementary table of participants with PD; the "
        "controls' count (39) and the three tasks' trial numbers are internal checks only.",
        "The deposit's README is titled 'Perception of effort but not reward value is impaired in people with Parkinson's disease'; the published paper has the title given above. "
        "The article carries a CC BY-NC-ND 4.0 licence on Crossref; the OSF data deposit is licensed CC BY-NC 4.0, which is what applies here.",
        "Participant IDs in the deposit are numbered one higher than in the supplementary table; the ages and sexes line up, so this does not affect the checks.",
    ],
)
stdlib.release(ID, "effort_reward_tasks", out, meta, checks)
