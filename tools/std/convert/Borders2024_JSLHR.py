"""Borders, Lowell, Huber, Quinn and Troche (2024), J Speech Lang Hear Res 67(5):1299-1323: voluntary and reflex cough in 20 people with
Parkinson's disease, one session of cough skill training with (BF) or without (NBF) respiratory biofeedback. 'summary' module,
one row per cough trial and measure.

Run from the site folder: python3 -I tools/std/convert/Borders2024_JSLHR.py
Raw files (untrusted data, read only) in raw_extra/Borders2024_JSLHR/ from https://osf.io/3tyhx/ (CC BY 4.0):
  cough_data.csv (one row per cough trial), demographics.csv (one row per participant), script.qmd (the authors' analysis script,
  read as text only, never run; its prose gives the participant counts used as published checks).
Not carried over: participant_calibration_factors.csv (belt calibration factors, 23 rows) and the fitted models."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Borders2024_JSLHR"
SRC = "analysis script text of the OSF deposit (Results, Participants)"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
d = pd.read_csv(RAW / "cough_data.csv", keep_default_na=False, na_values=[""])
g = pd.read_csv(RAW / "demographics.csv")
demo = g.set_index("id")
GROUP = {"BF": "biofeedback", "NBF": "no biofeedback"}
PHASE = {"pre": "baseline", "treatment": "learning", "post": "test"}
MEAS = [
    ("crtot", "coughs_in_trial", "count", "count", "coughs",
     "Number of coughs the participant produced in this trial (authors' crtot)."),
    ("lv_cpd_end", "lung_volume_at_cough_start", "value", "proportion of vital capacity",
     "proportion of vital capacity relative to end-expiratory level",
     "Lung volume at the end of the compression phase, i.e. the start of the expiratory phase of the cough, from rib cage and abdomen belts "
     "(authors' lv_cpd_end, 'LVCI' in the paper); empty trials were excluded by the authors, or the participant was excluded from this measure."),
    ("cev_epoch_l_s_s", "cough_expired_volume", "value", "L", "L",
     "Cough expired volume over the whole cough epoch (authors' cev_epoch_l_s_s)."),
    ("pefr_1", "peak_expiratory_flow_rate", "value", "L/s", "L/s", "Peak expiratory flow rate of the cough (authors' pefr_1)."),
]
rows, seq = [], {}
for r in d.itertuples(index=False):
    pid = int(r.id)
    seq[pid] = seq.get(pid, 0) + 1
    p = demo.loc[pid]
    ctype = r.cough_type
    cond = ctype if pd.isna(r.task) else f"{ctype} {r.task}"
    for col, name, stat, unit, _u, desc in MEAS:
        v = getattr(r, col)
        if pd.isna(v):
            continue
        rows.append(dict(
            dataset=ID, study=ID, experiment=ID, group=GROUP[r.condition], subject=f"{pid}", subject_excluded_by_authors=False, handedness=None, time_s=None,
            age=float(p.age), sex={"Male": "M", "Female": "F"}[p.sex], population="Parkinson's disease", setting="lab",
            device="respiratory inductance plethysmography belts and pneumotach, with a visual biofeedback display in the BF group",
            session=1, block=f"{r.phase} trial {int(r.trial)}", trial=seq[pid], phase=PHASE[r.phase], task="sports and games",
            unit="trial", condition=cond, pert_type="none", pert_value=0, pert_unit="",
            feedback="altered" if (r.condition == "BF" and r.phase == "treatment") else "normal", excluded_by_authors=False,
            measure=name, measure_description=desc, statistic=stat, value=float(v), value_unit=unit,
            n_trials=int(r.crtot) if col == "crtot" else None))
# the task column of the summary rows: the authors' task is voluntary or reflex cough
for x in rows:
    x["task"] = "cough"

per = d.groupby("id").size()
checks.published("participants enrolled and analysed", d.id.nunique(), 20, 0, f"{SRC}: 'Twenty participants ... were enrolled'")
checks.published("participants randomized to biofeedback", int((g.condition == "BF").sum()), 10, 0, f"{SRC}: biofeedback (n = 10)")
checks.published("participants randomized to no biofeedback", int((g.condition == "NBF").sum()), 10, 0, f"{SRC}: no biofeedback (n = 10)")
checks.published("participants with Hoehn and Yahr stage 2", int((g.hoehn_yahr_stage == 2).sum()), 17, 0, f"{SRC}")
checks.published("participants with Hoehn and Yahr stage 3", int((g.hoehn_yahr_stage == 3).sum()), 3, 0, f"{SRC}")
checks.published("mean disease duration (years)", float(g.years_since_diagnosis.mean()), 11.39, 0.005, f"{SRC}")
checks.published("SD of disease duration (years)", float(g.years_since_diagnosis.std()), 7.97, 0.005, f"{SRC}")
checks.published("SD of UPDRS motor score", float(g.updrs_motor.std()), 15, 0.5, f"{SRC}: SD = 15")
checks.published("minimum UPDRS motor score", float(g.updrs_motor.min()), 10, 0, f"{SRC}: range 10-66")
checks.published("maximum UPDRS motor score", float(g.updrs_motor.max()), 66, 0, f"{SRC}: range 10-66")
nl = d.groupby("id").lv_cpd_end.apply(lambda x: x.notna().any())
checks.published("participants with lung-volume data (20 minus 6 excluded)", int(nl.sum()), 14, 0, f"{SRC}: six participants excluded from LVCI")
checks.internal("every participant has 25 voluntary treatment trials", bool((d[d.phase == "treatment"].groupby("id").size() == 25).all()))
checks.internal("one condition per participant matching the demographics file",
                bool((d.groupby("id").condition.first() == demo.condition).all()))
checks.internal("all cough trials carried over (862) and four measures at most per trial", len({(x["subject"], x["trial"]) for x in rows}) == 862)
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}")

PCOLS = [("condition", "text", "", "Randomized group: biofeedback or no biofeedback."),
         ("years_since_diagnosis", "number", "years", "Years since Parkinson's disease diagnosis."),
         ("updrs_motor", "number", "points", "MDS-UPDRS motor score."),
         ("hoehn_yahr_stage", "integer", "", "Hoehn and Yahr stage."),
         ("moca_total", "number", "points", "Montreal Cognitive Assessment total.")]
parts = [dict(person=str(i), condition=GROUP[x.condition], years_since_diagnosis=x.years_since_diagnosis, updrs_motor=x.updrs_motor,
              hoehn_yahr_stage=x.hoehn_yahr_stage, moca_total=x.moca_total) for i, x in demo.iterrows()]

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Troche (Teachers College, Columbia University)", data_url="https://osf.io/3tyhx/", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Converted to the OpenMotor summary format: participant ids replaced by codes, the cough file reshaped to long form (one row per "
            "cough trial and measure, empty measures left out), age and sex joined from the demographics file, phases mapped (pre = baseline, "
            "treatment = learning, post = test).",
    how=["One row is one measure of one cough trial: coughs_in_trial (number of coughs in the trial, also in n_trials), "
         "lung_volume_at_cough_start (proportion of vital capacity relative to end-expiratory level), cough_expired_volume (L, whole epoch) "
         "and peak_expiratory_flow_rate (L/s). trial counts the participant's cough trials in file order and block holds the authors' "
         "phase and trial number; the four rows of a trial share the same trial value.",
         "condition is the cough type and task: voluntary single, voluntary sequential, or reflex cough (200 uM; one reflex_500 trial before "
         "and after training). Voluntary single cough was also trained: 25 treatment trials per participant. The group is biofeedback or no "
         "biofeedback; feedback is marked altered on the biofeedback group's treatment trials.",
         "Task is recorded as cough, which has no OpenMotor task name; the listing's 'sports and games' tag is not appropriate for this deposit."],
    unit_note="cough trial", differences=[
        "The paper text in the deposit's script gives a mean UPDRS motor score of 31 (SD 15); the demographics file gives 30.45 (SD 14.8), "
        "so the mean was not used as a published check.",
        "The script reports 9 participants with an EAT-10 of 3 or more and 18 with an unrestricted diet; the demographics file gives 10 and 16 "
        "(FOIS 7) by the obvious cut-offs, so these were not used as checks.",
        "participant_calibration_factors.csv (belt calibration, 23 rows) and the reliability files are not carried over.",
        "Trials the authors excluded for extraneous movement have empty lung volume or flow values in cough_data.csv and give no row for that "
        "measure."])
stdlib.release(ID, "summary", rows, meta, checks, participants=parts, participant_columns=PCOLS)
