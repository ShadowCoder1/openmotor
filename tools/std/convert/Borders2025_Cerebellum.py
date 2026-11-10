"""Borders, Kuo and Troche (2025), Cerebellum 24(5):129: a single session of cough skill training (visual biofeedback on peak
expiratory flow, and a no-biofeedback condition, in crossover order) in seven people with cerebellar ataxia. 'summary' module,
one row per cough trial and measure.

Run from the site folder: python3 -I tools/std/convert/Borders2025_Cerebellum.py
Raw files (untrusted data, read only) in raw_extra/Borders2025_Cerebellum/ from https://osf.io/pxsry/ (CC BY 4.0):
  sca_biofeedback_data_raw.xlsx   sheet 'long' (one row per cough trial, with participant details; carried over in full, including the
                                  eighth person the authors left out of the seven analysed). Sheet 'wide' (post-training copy) not used.
  sca_biofeedback_data.xlsx       the authors' analysis file, used to confirm which seven people were analysed.
Not carried over: sca_reliability.xlsx, the fitted models."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Borders2025_Cerebellum"
SRC = "abstract and Methods (Cerebellum 24(5):129), OSF project description"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
a = pd.read_excel(RAW / "sca_biofeedback_data_raw.xlsx", sheet_name="long")
b = pd.read_excel(RAW / "sca_biofeedback_data.xlsx", sheet_name="long")

# which raw ids were analysed: screening date and biofeedback target match the analysis file
key_b = {(str(x.Screening_date)[:10], round(float(x.biofeedback_target_l_s), 2)) for x in b.drop_duplicates("ID").itertuples()}
first = a.drop_duplicates("ID").set_index("ID")
analysed = {i for i, x in first.iterrows() if (str(x.Screening_date)[:10], round(float(x.biofeedback_target_l_s), 2)) in key_b}
excluded = set(first.index) - analysed

MEAS = [  # column, measure, statistic, unit, description
    ("crtot", "coughs_in_trial", "count", "coughs", "Number of coughs in the trial (authors' crtot)."),
    ("civ", "cough_inspired_volume", "value", "L", "Cough inspired volume (authors' civ; negative values as shared)."),
    ("cev_epoch", "cough_expired_volume_epoch", "value", "L", "Cough expired volume over the whole cough epoch (authors' cev_epoch)."),
    ("cpd_1", "compression_phase_duration_cough1", "value", "s", "Compression phase duration of the first cough (authors' cpd_1; unit and name inferred, not defined in the deposit)."),
    ("pefrt_1", "peak_flow_rise_time_cough1", "value", "s", "Peak expiratory flow rise time of the first cough (authors' pefrt_1; unit and name inferred)."),
    ("pefr_1", "peak_expiratory_flow_rate_cough1", "value", "L/s", "Peak expiratory flow rate of the first cough (authors' pefr_1)."),
    ("cev_1", "cough_expired_volume_cough1", "value", "L", "Expired volume of the first cough (authors' cev_1)."),
    ("cpd_2", "compression_phase_duration_cough2", "value", "s", "Compression phase duration of the second cough of a sequence (authors' cpd_2; inferred as for cpd_1)."),
    ("pefrt_2", "peak_flow_rise_time_cough2", "value", "s", "Peak flow rise time of the second cough (authors' pefrt_2; inferred)."),
    ("pefr_2", "peak_expiratory_flow_rate_cough2", "value", "L/s", "Peak expiratory flow rate of the second cough (authors' pefr_2)."),
    ("cev_2", "cough_expired_volume_cough2", "value", "L", "Expired volume of the second cough (authors' cev_2)."),
]
TRAIN = {"BF", "NF"}
NAME = {"BF": "biofeedback", "NF": "no biofeedback"}
rows, seq = [], {}
for r in a.itertuples(index=False):
    pid = int(r.ID)
    seq[pid] = seq.get(pid, 0) + 1
    tp = r.treatment_phase
    if tp == "P1":
        phase, blk = "baseline", "baseline assessment"
    elif tp in TRAIN:
        phase, blk = "learning", f"training with {NAME[tp]}"
    else:
        phase, blk = "test", f"post-training assessment after {NAME.get(r.condition, 'training')} ({tp})"
    cond = r.cough_type if pd.isna(r.task) else f"{r.cough_type} {r.task}"
    if not pd.isna(r.capsaicin_um):
        cond += f" {int(r.capsaicin_um)} uM"
    order = str(r.condition_order).replace("NF", "no biofeedback").replace("BF", "biofeedback").replace("NA", "none").replace("_", " then ")
    blk = blk + ("" if pd.isna(r.block) else f", block {int(r.block)}") + f", trial {int(r.trial)}"
    for col, name, stat, unit, desc in MEAS:
        v = getattr(r, col)
        if pd.isna(v):
            continue
        rows.append(dict(
            dataset=ID, study=ID, experiment=ID, group=f"order: {order}", subject=f"{pid}", subject_excluded_by_authors=pid in excluded,
            age=float(r.age), sex={"male": "M", "female": "F"}[r.sex], handedness=None, time_s=None,
            population="cerebellar ataxia", setting="lab",
            device="respiratory inductance plethysmography belts and pneumotach, with a visual biofeedback target line in the biofeedback condition",
            session=1, block=blk, trial=seq[pid], phase=phase, task="cough", unit="trial", condition=cond,
            pert_type="none", pert_value=0, pert_unit="", feedback="altered" if (phase == "learning" and tp == "BF") else "normal",
            excluded_by_authors=pid in excluded, measure=name, measure_description=desc, statistic=stat, value=float(v), value_unit=unit,
            n_trials=int(r.crtot) if col == "crtot" and not pd.isna(r.crtot) else None))

an = first.loc[sorted(analysed)]
checks.published("people analysed", len(an), 7, 0, f"abstract: seven people with genetically confirmed CA")
checks.published("women among the people analysed", int((an.sex == "female").sum()), 6, 0, "abstract: 6 female, 1 male")
checks.published("men among the people analysed", int((an.sex == "male").sum()), 1, 0, "abstract: 6 female, 1 male")
tr = a[a.ID.isin(analysed) & a.treatment_phase.isin(TRAIN)]
for t in ("BF", "NF"):
    n = int((tr.treatment_phase == t).sum())
    checks.published(f"training trials with {NAME[t]} (7 people x 15)", n, 105, 0, f"Methods: all participants completed every training trial (15 per condition), {SRC}")
checks.internal("analysis file ids match 7 raw ids and the one left out has partial data", len(excluded) == 1 and len(analysed) == 7, str(excluded))
nrow_t = len({(x["subject"], x["trial"]) for x in rows})
checks.internal("cough trials with at least one value are carried over", nrow_t > 0.9 * len(a), f"{nrow_t} of {len(a)} trials have a value")
print("trials without any value (no row):", len(a) - nrow_t)
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}")

PCOLS = [("scd_type", "text", "", "Genetic subtype of ataxia as listed (SCA type number)."),
         ("sara", "number", "points", "Scale for the Assessment and Rating of Ataxia."),
         ("moca", "number", "points", "Montreal Cognitive Assessment total."),
         ("disease_duration_years", "number", "years", "Disease duration at screening."),
         ("biofeedback_target_l_s", "number", "L/s", "Peak flow target line shown in the biofeedback condition (25% above baseline maximum PEFR).")]
parts = [dict(person=str(i), scd_type=str(x.SCA_Type), sara=x.SARA, moca=x.MOCA, disease_duration_years=x.disease_duration_years,
              biofeedback_target_l_s=x.biofeedback_target_l_s) for i, x in first.iterrows()]

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Troche (Teachers College, Columbia University)", data_url="https://osf.io/pxsry/", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Converted to the OpenMotor summary format: participant ids replaced by codes, the 'long' sheet reshaped to one row per cough trial "
            "and measure (empty measures left out), demographics from the same sheet, phases mapped (P1 = baseline, training = learning, "
            "P2 and P3 post-training assessments = test).",
    how=["One row is one measure of one cough trial: counts and volumes in L, flows in L/s, durations in s; for sequential coughs the "
         "second cough has its own columns (names ending 2). trial counts the participant's cough trials in file order and block holds "
         "the session part (baseline, training with or without biofeedback, post-training assessment) and the authors' trial number; the "
         "rows of one trial share the same trial value.",
         "Each person did a baseline assessment (voluntary single and sequential cough, reflex cough with capsaicin), then in crossover "
         "order 15 single voluntary coughs with biofeedback and 15 without, each followed by an assessment. group gives the order. "
         "feedback is altered on biofeedback training trials only.",
         "The eighth person in the raw file (completed only part of the protocol) is kept and marked excluded by the authors; the seven "
         "analysed people were identified by matching screening date and target against the authors' analysis file."],
    unit_note="cough trial", differences=[
        "cpd, pefrt and the second-cough columns are not defined in the deposit; their meaning (compression phase duration, peak flow rise "
        "time, in seconds) is inferred from the lab's usual measures and the names.",
        "Reflex-cough rows are in the raw sheet only (not in the authors' analysis file) and are carried over as shared; capsaicin concentration "
        "is appended to the condition.",
        "The paper's effect sizes are Bayesian model estimates and were not recomputed."])
stdlib.release(ID, "summary", rows, meta, checks, participants=parts, participant_columns=PCOLS)
