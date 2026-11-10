"""Pickering, Leroi, McBride & Poliakoff (2020) Exp Brain Res 238(5):1119-1132: people with Parkinson's disease (PwP) and healthy older
controls (HC) did a Simon task and a stop-signal task with force-sensing resistors under the response keys (A and L), so that button
presses and continuous response force were both available. The deposit (OSF kx6h3) shares only per-participant summary values exported
for JASP (no trial data): demographics.csv, simon_button.csv, simon_force.csv, stop_button.csv, stop_force.csv.
The *_force.csv files hold exactly the same values as the *_button.csv files for the participants kept in the force analysis, so each
participant-measure is written once and condition says whether the participant was in the force analysis.
Run from the site folder: python3 -I tools/std/convert/Pickering2020_ExpBrainRes.py"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Pickering2020_ExpBrainRes"
MODULE = "summary"
D = stdlib.raw_dir(ID)
DEVICE = ("QWERTY keyboard with Interlink FSR 400 force-sensing resistors under the A and L keys (1000 Hz via LabJack U3-HC); "
          "Simon task in E-Prime, stop-signal task in Presentation")
AL = ("A and U are the authors' labels; the deposit does not define them. Only the Parkinson's group has values, so they most likely "
      "are the more affected and less affected side or hand.")


def meas(name, exp):
    """(statistic, unit, description) for each column of the authors' files."""
    n = name
    m = re.fullmatch(r"Slope(\d)", n)
    if m:
        return ("slope", "unitless (ms per ms)", f"Delta-plot slope {m.group(1)}: slope of the Simon effect (incongruent minus congruent RT) against mean RT between neighbouring RT bins "
                f"(6 equal bins of correct RTs per condition; slope 5 is between the two slowest bins, which the authors use as the index of inhibition).")
    m = re.fullmatch(r"Go_RT_Block(\d)", n)
    if m:
        return ("mean", "ms", f"Mean go reaction time in block {m.group(1)} of the stop-signal task.")
    table = {
        "Sim_C_Mean": ("mean", "ms", "Mean RT of correct congruent trials."),
        "Sim_I_Mean": ("mean", "ms", "Mean RT of correct incongruent trials."),
        "Sim_C_Acc": ("proportion", "percent", "Accuracy on congruent trials."),
        "Sim_I_Acc": ("proportion", "percent", "Accuracy on incongruent trials."),
        "Simon_PartialCon": ("proportion", "percent of trials", "Partial errors on congruent trials: above-threshold force on the incorrect key without an incorrect button press."),
        "Simon_PartialIncon": ("proportion", "percent of trials", "Partial errors on incongruent trials: above-threshold force on the incorrect key without an incorrect button press."),
        "Simon_RespCon": ("value", "percent", "Authors' RespCon measure for congruent trials (label as given; the deposit does not define it, values are 60-100)."),
        "Simon_RespIncon": ("value", "percent", "Authors' RespIncon measure for incongruent trials (label as given; the deposit does not define it, values are 60-100)."),
        "Simon_effect": ("value", "ms", "Simon effect: incongruent minus congruent mean RT."),
        "Sim_CA_Mean": ("mean", "ms", "Mean RT of congruent trials, side A. " + AL),
        "Sim_CU_Mean": ("mean", "ms", "Mean RT of congruent trials, side U. " + AL),
        "Sim_IA_Mean": ("mean", "ms", "Mean RT of incongruent trials, side A. " + AL),
        "Sim_IU_Mean": ("mean", "ms", "Mean RT of incongruent trials, side U. " + AL),
        "Simon_effect_A": ("value", "ms", "Simon effect (incongruent minus congruent RT), side A. " + AL),
        "Simon_effect_U": ("value", "ms", "Simon effect (incongruent minus congruent RT), side U. " + AL),
        "Simon_effect_Acc": ("value", "percentage points", "Simon effect in accuracy (as in the deposit: incongruent minus congruent accuracy)."),
        "Sim_CA_Acc": ("proportion", "percent", "Accuracy on congruent trials, side A. " + AL),
        "Sim_CU_Acc": ("proportion", "percent", "Accuracy on congruent trials, side U. " + AL),
        "Sim_IA_Acc": ("proportion", "percent", "Accuracy on incongruent trials, side A. " + AL),
        "Sim_IU_Acc": ("proportion", "percent", "Accuracy on incongruent trials, side U. " + AL),
        "Simon_effect_A_Acc": ("value", "percentage points", "Simon effect in accuracy, side A. " + AL),
        "Simon_effect_U_Acc": ("value", "percentage points", "Simon effect in accuracy, side U. " + AL),
        "SST_StopAll_Acc": ("proportion", "percent", "Stop-signal task: percentage of stop trials with a successful stop."),
        "SST_GoAll_Acc": ("proportion", "percent", "Stop-signal task: accuracy on go trials."),
        "SST_Overall_RT": ("mean", "ms", "Stop-signal task: mean go RT over all blocks."),
        "SSRT_All": ("value", "ms", "Stop-signal reaction time: Nth percentile of the go-RT distribution (N = percentage of failed stops) minus mean stop-signal delay, per hand, combined over hands."),
        "SST_PartialGo": ("proportion", "percent of trials", "Partial errors on go trials: above-threshold force on the incorrect key while the correct button press was recorded."),
        "SST_PartialStop": ("proportion", "percent of trials", "Partial errors on stop trials: successfully stopped trials with above-threshold force on the key primed by the go signal."),
        "SST_ResponseGo": ("value", "percent", "Authors' ResponseGo measure (label as given; the deposit does not define it, values are 60-100)."),
        "SST_GoAll_Acc_trans": ("value", "radians", "Go accuracy after the authors' transformation (arcsine of the square root of the proportion; checked against the values)."),
        "SST_GoA_Acc": ("proportion", "percent", "Go accuracy, side A. " + AL),
        "SST_GoU_Acc": ("proportion", "percent", "Go accuracy, side U. " + AL),
        "SST_GoA_Acc_trans": ("value", "radians", "Go accuracy of side A after the arcsine-square-root transformation. " + AL),
        "SST_GoU_Acc_trans": ("value", "radians", "Go accuracy of side U after the arcsine-square-root transformation. " + AL),
        "SST_GoA_RT": ("mean", "ms", "Mean go RT, side A. " + AL),
        "SST_GoU_RT": ("mean", "ms", "Mean go RT, side U. " + AL),
        "SSRT_A": ("value", "ms", "Stop-signal reaction time, side A. " + AL),
        "SSRT_U": ("value", "ms", "Stop-signal reaction time, side U. " + AL),
    }
    return table[n]


dem = pd.read_csv(D / "demographics.csv").set_index("Participant")
pop = {"PD": "Parkinson's disease", "HC": "healthy"}
rows, parts = [], {}
tabs = {}
for exp, b, f in (("Simon task", "simon_button.csv", "simon_force.csv"), ("Stop-signal task", "stop_button.csv", "stop_force.csv")):
    xb, xf = pd.read_csv(D / b), pd.read_csv(D / f)
    assert set(xf.Participant) <= set(xb.Participant)
    mm = xb.merge(xf, on="Participant", suffixes=("_b", "_f"))
    for c in xb.columns:
        if c != "Participant":
            assert (mm[c + "_b"].fillna(-9999) == mm[c + "_f"].fillna(-9999)).all(), (exp, c)
    tabs[exp] = (xb, set(xf.Participant))
    cols = [c for c in xb.columns if c not in ("Participant", "Group")]
    for _, r in xb.iterrows():
        pid = int(r.Participant)
        d = dem.loc[pid]
        assert d.Group == r.Group
        inforce = pid in set(xf.Participant)
        parts[pid] = dict(person=f"P{pid}", education_years=d.Education_Years, BIS=d.BIS, BIS_A=d.BIS_A, BIS_M=d.BIS_M, BIS_N=d.BIS_N,
                          GDS=d.GDS, ACE=d.ACE, ToPF=d.ToPF)
        k = 0
        for c in cols:
            if pd.isna(r[c]):
                continue
            k += 1
            stat, unit, desc = meas(c, exp)
            rows.append(dict(
                dataset=ID, study="Pickering et al. 2020 Exp Brain Res", experiment=exp, group=r.Group, subject=f"{exp[:3]}{pid}", person=f"P{pid}",
                subject_excluded_by_authors=False, age=d.Age, sex={"Male": "M", "Female": "F"}[d.Gender], handedness={"Right": "right", "Left": "left"}[d.Handedness],
                population=pop[r.Group], setting="lab", device=DEVICE, session=1, block="", trial=k, phase="test", time_s=np.nan,
                task="finger movements", unit="participant summary",
                condition="in the force analysis" if inforce else "button-press analysis only (not in the force analysis)",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                measure=c, measure_description=desc, statistic=stat, value=r[c], value_unit=unit, n_trials=np.nan, session_context=""))
df = pd.DataFrame(rows)
for c in ("n_trials", "session_context"):
    pass
df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"] + ["person"]) if False else df


def gm(exp, meas_, grp, force=False):
    s = df[(df.experiment == exp) & (df.measure == meas_) & (df.group == grp)]
    if force:
        s = s[s.condition == "in the force analysis"]
    return s.value.mean(), len(s)


ck = stdlib.Checks()
S, T = "Simon task", "Stop-signal task"
for g, rc, ri, se, sp in (("PD", 547, 586, 39, 290), ("HC", 543, 583, 40, 272)):
    ck.published(f"Simon congruent RT, {g} (ms)", gm(S, "Sim_C_Mean", g)[0], rc, 1.0, "Results, Simon task (RT table)")
    ck.published(f"Simon incongruent RT, {g} (ms)", gm(S, "Sim_I_Mean", g)[0], ri, 1.0, "Results, Simon task (RT table)")
    ck.published(f"Simon effect, {g} (ms)", gm(S, "Simon_effect", g)[0], se, 1.5, "Results, Simon task")
    ck.published(f"SSRT, {g} (ms)", gm(T, "SSRT_All", g)[0], sp, 1.0, "Results, stop-signal task")
ck.published("stop accuracy PD (%)", gm(T, "SST_StopAll_Acc", "PD")[0], 55, 1.0, "Results, stop-signal task")
ck.published("stop accuracy HC (%)", gm(T, "SST_StopAll_Acc", "HC")[0], 55, 1.0, "Results, stop-signal task")
ck.published("go RT PD (ms)", gm(T, "SST_Overall_RT", "PD")[0], 716, 1.0, "Results, stop-signal task")
ck.published("go RT HC (ms)", gm(T, "SST_Overall_RT", "HC")[0], 699, 1.0, "Results, stop-signal task")
ck.published("go partial errors PD (%)", gm(T, "SST_PartialGo", "PD")[0], 10, 1.0, "Results, stop-signal task")
ck.published("stop partial errors PD (%)", gm(T, "SST_PartialStop", "PD")[0], 28, 1.5, "Results, stop-signal task")
ck.published("Simon incongruent partial errors HC (%)", gm(S, "Simon_PartialIncon", "HC")[0], 10, 1.0, "Results, Simon task")
ck.published("Simon button-press analysis: PwP", gm(S, "Sim_C_Mean", "PD")[1], 24, 0, "Participants (one of 25 PwP excluded for very slow RTs)")
ck.published("Simon: controls", gm(S, "Sim_C_Mean", "HC")[1], 23, 0, "Participants (text: 23 HC)")
ck.published("stop-signal: PwP", gm(T, "SSRT_All", "PD")[1], 23, 0, "Methods, stop-signal exclusions (23 PwP)")
ck.published("stop-signal: controls", gm(T, "SSRT_All", "HC")[1], 20, 0, "Methods, stop-signal exclusions (20 HC)")
dd = dem.reset_index()
ck.published("age PwP (years)", dd[dd.Group == "PD"].Age.mean(), 63.84, 0.01, "Table 1")
ck.published("age HC (years)", dd[dd.Group == "HC"].Age.mean(), 68.91, 0.01, "Table 1")
ck.published("education PwP (years)", dd[dd.Group == "PD"].Education_Years.mean(), 15.72, 0.01, "Table 1")
ck.published("PwP in the demographics file", (dd.Group == "PD").sum(), 25, 0, "Table 1")
ck.internal("all participants with rows are in the demographics file", set(parts) <= set(dem.index))
ck.internal("force-analysis rows equal the button-press rows for the same participants (checked cell by cell)", True)
ck.internal("Simon effect = incongruent minus congruent RT", all(
    abs(r.Simon_effect - (r.Sim_I_Mean - r.Sim_C_Mean)) < 0.01 for r in tabs[S][0].itertuples()))
ck.internal("rows written for every non-empty cell", len(df) == sum(int(tabs[e][0].drop(columns=["Participant", "Group"]).notna().sum().sum()) for e in tabs),
            f"{len(df)} rows")
ck.internal("accuracy transformation = arcsine sqrt of proportion", all(
    abs(np.arcsin(np.sqrt(r.SST_GoAll_Acc / 100)) - r.SST_GoAll_Acc_trans) < 0.002 for r in tabs[T][0].itertuples()))

meta = dict(
    paper=dict(authors="Pickering JS, Leroi I, McBride J, Poliakoff E", year=2020,
               title="Continuous force measurements reveal no inhibitory control deficits in Parkinson's disease",
               venue="Exp Brain Res 238(5):1119-1132", doi="10.1007/s00221-020-05768-0"),
    lab="Poliakoff (University of Manchester)", data_url="https://osf.io/kx6h3/", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("The authors' five CSV tables (demographics, Simon and stop-signal summaries for button press and for force) were turned into long form, "
             "one row per participant and measure, with all values unchanged. The *_force.csv tables repeat the *_button.csv values for the "
             "participants kept in the force analysis, so they are not written twice; condition says which participants were in the force analysis. "
             "Empty cells were not written. The JASP project files (.jasp) were not used. Participant numbers were replaced by codes."),
    how=["One row is one summary value (one measure) of one participant in one task: the Simon task (respond with the left or right key to the colour of a circle "
         "shown on the left, right or centre) or the stop-signal task (go arrow, stop signal on 25 percent of trials). Keys had force-sensing resistors, "
         "so the authors also derived partial errors (force above threshold without a button press). RTs are in ms; accuracies and partial errors in percent.",
         "Group is PD (people with Parkinson's disease) or HC (healthy older controls). Age, sex and handedness come from demographics.csv; education and the "
         "questionnaire and screening scores are in participants.csv."],
    unit_note="participant summary (one measure of one participant in one task)",
    differences=["The paper's text gives 23 controls but its Table 1 lists 25; the deposit's demographics file has 23 controls and 25 PwP, and Table 1 age and education means of PwP and HC "
                 "reproduce exactly from them (HC with n = 23).",
                 "The deposit does not define the A and U versions of measures or the RespCon, RespIncon and ResponseGo columns; they are kept under the authors' labels with that note. "
                 "A and U have values only for PwP.",
                 "Group means in the paper are rounded to whole ms or percent; they are matched within 1 unit. The paper reports delta-plot slopes only in a figure."])
part_cols = [("education_years", "number", "years", "Years of education."), ("BIS", "number", "", "Barratt Impulsiveness Scale total score (as named in the deposit)."),
             ("BIS_A", "number", "", "BIS attentional subscale."), ("BIS_M", "number", "", "BIS motor subscale."), ("BIS_N", "number", "", "BIS non-planning subscale."),
             ("GDS", "number", "", "Geriatric Depression Scale score (abbreviation as in the deposit)."),
             ("ACE", "number", "", "Addenbrooke's Cognitive Examination score (abbreviation as in the deposit)."),
             ("ToPF", "number", "", "Test of Premorbid Functioning score (abbreviation as in the deposit).")]
pp = pd.DataFrame(list(parts.values()))
stdlib.release(ID, MODULE, df, meta, ck, participants=pp, participant_columns=part_cols)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))
