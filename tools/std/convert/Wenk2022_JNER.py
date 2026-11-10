"""Wenk et al. (2022), J NeuroEng Rehabil 19:137: touching virtual fruits while counting, under immersive VR, AR and a 2D screen.
Experiment 1: 20 healthy older adults; Experiment 2: 5 subacute brain-injured patients. One row per reach.

Run from the site folder: python3 -I tools/std/convert/Wenk2022_JNER.py
Raw files (untrusted data, read only) in raw_extra/Wenk2022_JNER/ from https://zenodo.org/records/6631771:
  ARVR2D_elderlies_DataCatchBased.csv, ARVR2D_patients_DataCatchBased.csv  (latin-1 text), DataFrameDescription.rtf (read as text)
The participant-based averages and questionnaire files of the deposit are not converted.
Age and sex come from the paper's Tables 1 and 2 (participant # = ParticipantNumber; the three participants the paper excluded,
two females and one male, are numbers 1, 2 and 11 in the deposit, which supports the mapping).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wenk2022_JNER"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}

# paper Table 1 (older adults) and Table 2 (patients): sex, age
OLD = list(zip("MFFFMFFMMMFFMMMFMMMM", [84, 82, 68, 72, 64, 64, 70, 60, 67, 64, 89, 71, 85, 84, 76, 78, 73, 77, 72, 75]))
PAT = list(zip("FMMMM", [36, 38, 69, 58, 48]))
EXPS = [("Exp1", "elderlies", OLD, "healthy older adults", "ARVR2D_elderlies_DataCatchBased.csv"),
        ("Exp2", "patients", PAT, "brain injury (subacute)", "ARVR2D_patients_DataCatchBased.csv")]
EXCLUDED_EXP1 = {1: "technical problems with the AR display", 2: "technical problems with the AR display",
                 11: "unable to stay in contact with the start sphere in the 2D screen condition"}
BLOCKS = {"Exp1": [6, 12, 12, 12, 18, 18, 18, 6], "Exp2": [6, 6, 6, 12, 12, 6]}

frames = []
for exp, tag, demo, pop, fn in EXPS:
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
    d = pd.read_csv(RAW / fn, encoding="latin-1")
    d["exp"] = exp
    d["pop"] = pop
    frames.append((exp, d, demo))
    checks.internal(f"{exp}: rows grouped by participant, display blocks in order", d.groupby("ParticipantNumber").ngroups ==
                    (d.ParticipantNumber != d.ParticipantNumber.shift()).sum())

e1, d1, demo1 = frames[0]
e2, d2, demo2 = frames[1]

# ---------------------------------------------------------------- published checks
checks.published("Experiment 1 participants", d1.ParticipantNumber.nunique(), 20, 0, "Methods, Experiment 1, Participants")
checks.published("Experiment 2 participants", d2.ParticipantNumber.nunique(), 5, 0, "Methods, Experiment 2, Participants")
checks.published("reaching movements in Experiment 1", len(d1), 5886, 0, "Methods, Statistical analyses (5886 reaching movements)")
checks.published("reaches without movement onset (speed never reached 0.2 m/s)", d1["Movement.Onset"].isna().sum(), 280, 0,
                 "Methods, Statistical analyses (280 did not reach the minimum velocity threshold)")
checks.published("fruits per display, Experiment 1 (participant 3, complete)", (d1.ParticipantNumber == 3).sum() / 3, 102, 0,
                 "Methods, Reaching and cognitive tasks (102 fruits)")
checks.published("fruits per display, Experiment 2 (participant 1)", (d2.ParticipantNumber == 1).sum() / 3, 48, 0,
                 "Methods, Experiment 2 (48 fruits per condition)")
for exp, d in (("Exp1", d1), ("Exp2", d2)):
    full = d.groupby(["ParticipantNumber", "Modality"]).size()
    ok = True
    for (p, m), n in full.items():
        if n == sum(BLOCKS[exp]):
            sizes = d[(d.ParticipantNumber == p) & (d.Modality == m)].groupby("TrialIndex").size().tolist()
            ok &= sizes == BLOCKS[exp]
    checks.published(f"{exp}: every complete display condition has blocks of {BLOCKS[exp]} fruits",
                     float(ok), 1.0, 0, "Methods, Reaching and cognitive tasks / Experiment 2")
exc_sex = sorted(demo1[p - 1][0] for p in EXCLUDED_EXP1)
checks.published("participants excluded by the authors that are female (deposit's incomplete conditions: numbers 1, 2, 11)",
                 exc_sex.count("F"), 2, 0, "Methods, Statistical analyses (three excluded: two females, one male)")
inc = d1.groupby(["ParticipantNumber", "Modality"]).size().unstack()
checks.internal("the three authors-excluded participants are exactly those missing part of a display condition",
                set(inc[(inc != 102).any(axis=1) | inc.isna().any(axis=1)].index) == set(EXCLUDED_EXP1))
checks.internal("Experiment 2: 719 of 720 reaches shared (one reach missing in the AR condition of participant 3)",
                len(d2) == 719 and d2[(d2.ParticipantNumber == 3) & (d2.Modality == "AR")].shape[0] == 47)
checks.internal("onset missing only where the threshold was not reached (Exp 1: 280, Exp 2: 144)",
                d1["Movement.Onset"].isna().sum() == 280 and d2["Movement.Onset"].isna().sum() == 144)
checks.internal("peak velocity time normalized = peak velocity time / ideal distance",
                bool(((d1["Movement.PeakVelocityTime"] / d1["Movement.IdealDistance"] - d1["Movement.PeakVelocityTime.Normalized"]).abs() < 1e-3).all()))

# ---------------------------------------------------------------- table
rows = []
for exp, d, demo in frames:
    for pn, g in d.groupby("ParticipantNumber", sort=False):
        sex, age = demo[pn - 1]
        excl = exp == "Exp1" and pn in EXCLUDED_EXP1
        subj = f"{exp}_P{pn:02d}"
        for k, r in enumerate(g.itertuples(index=False), start=1):
            r = r._asdict() if hasattr(r, "_asdict") else r
            rr = dict(zip(g.columns, r.values())) if isinstance(r, dict) else None
            rows.append((subj, k, exp, sex, age, excl, rr))

def b(x):
    return bool(x) if x == x else None

out = []
for subj, k, exp, sex, age, excl, r in rows:
    onset = r["Movement.Onset"]
    out.append({
        "dataset": ID, "study": "Wenk 2022 J NeuroEng Rehabil", "experiment": exp, "group": "", "subject": subj,
        "subject_excluded_by_authors": excl, "age": age, "sex": sex, "handedness": "",
        "population": r["pop"], "setting": "lab", "device": "tracked VR controller (Exp 1) or weight-supported arm device ArmeoSpring (Exp 2)",
        "session": 1, "block": f"{r['Modality']}_block{r['TrialIndex']}", "trial": k, "phase": "test", "time_s": "",
        "task": "reaching", "unit": "trial", "condition": r["Modality"], "pert_type": "none", "pert_value": 0, "pert_unit": "",
        "feedback": "normal", "excluded_by_authors": bool(excl or not r["CatchSuccess"]),
        "display": r["Modality"], "block_index": r["TrialIndex"], "reach_in_block": r["CatchIndex"],
        "loc_x_usage": r["Location.xUsage"], "loc_y_usage": r["Location.yUsage"], "loc_z_usage": r["Location.zUsage"],
        "loc_dimensionality": r["Location.Dimensionality"], "depth_usage": r["Location.DepthUsage.Detailed"],
        "fruit_type": r["Type.Real"], "fruit_number": r["Count.Real"], "catch_success": bool(r["CatchSuccess"]),
        "duration_s": r["Movement.Duration"], "ideal_distance_m": r["Movement.IdealDistance"],
        "duration_normalized": r["Movement.Duration.Normalized"], "path_length_m": r["Movement.Distance"],
        "straightness_ratio": r["Movement.Straightness"], "peak_velocity_m_s": r["Movement.PeakVelocity"],
        "peak_velocity_time_s": r["Movement.PeakVelocityTime"], "peak_velocity_time_norm": r["Movement.PeakVelocityTime.Normalized"],
        "n_velocity_peaks": r["Movement.NbPeaks"], "onset_s": None if onset != onset else onset,
        "endpoint_velocity_m_s": r["Movement.EndPointVelocity"], "count_said": r["Count.Said"],
        "type_said": None if r["Type.Said"] != r["Type.Said"] else r["Type.Said"],
        "fruit_correct": bool(r["Fruit.Correctness.Binary"]), "fruit_correctness": r["Fruit.Correctness.Detailed"],
    })
checks.internal("every row kept: 5886 + 719", len(out) == 5886 + 719)
checks.internal("subjects: 20 + 5", len({o["subject"] for o in out}) == 25)

meta = dict(
    study="Wenk et al. 2022",
    paper=dict(authors="Wenk N, Buetler KA, Penalver-Andres J, Muri RM, Marchal-Crespo L", year=2022,
               title="Naturalistic visualization of reaching movements using head-mounted displays improves movement quality "
                     "compared to conventional computer screens and proves high usability",
               venue="J NeuroEngineering Rehabil 19:137", doi="10.1186/s12984-022-01101-8"),
    lab="Marchal-Crespo (University of Bern)", data_url="https://zenodo.org/records/6631771",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant numbers replaced by codes. Only the two catch-based (one row per "
             "reach) files were converted; the deposit's participant averages, questionnaires and the plane/line indicator "
             "columns (derivable from the axis usage columns) were left out. Age and sex were added from the paper's Tables 1 and 2."),
    how=[
        "One row is one reach to a virtual fruit, in the authors' file order (participant, then display condition in the order "
        "tested, then block and reach). Experiment 1: 20 healthy older adults, 102 fruits per display (VR, AR, 2D screen) in eight "
        "blocks, touched with a tracked hand controller. Experiment 2: 5 subacute brain-injured patients, 48 fruits per display in "
        "six blocks, with the paretic arm supported by an arm-weight-support device. Participants also named each fruit and its running count.",
        "There is no hand angle: the authors' measures are duration (s), path length and ideal distance (m), straightness (path/ideal), "
        "peak velocity (m/s), number of velocity peaks and onset (s, empty where speed never reached 0.2 m/s). This is a new module "
        "(reach_quality_3d), not the reach90 format.",
        "Participants 1, 2 and 11 of Experiment 1 are the three the authors excluded from their statistics (AR device problems; unable to "
        "stay on the start sphere with the screen); they have missing reaches in one condition and are marked subject_excluded_by_authors. "
        "Reaches the researcher skipped (catch_success false) are marked excluded_by_authors. The authors' per-metric removal of extreme "
        "outliers (outside Q1 - 3 IQR, Q3 + 3 IQR) is not applied or flagged.",
        "Age and sex are from the paper's tables, matched to the deposit by participant number. The authors' column PeakVelocityTime.Normalized "
        "is peak velocity time divided by ideal distance in the shared values, although the deposit's description says peak velocity.",
    ],
    unit_note="reach to a virtual fruit",
    differences=[
        "The paper reports the table ages of the 20 older adults as 74.22 +/- 8.11 years; the ages printed in its Table 1 give 73.75 +/- 8.12, and "
        "the patients' Table 2 ages give 49.8 +/- 13.9 against 49.88 +/- 12.55 printed in the text, so these were not used as checks.",
        "The paper says 5886 reaching movements were performed by 17 participants; the deposit's 5886 rows come from all 20 participants "
        "(17 with complete data). The 280 reaches without a movement onset also count all 20 (250 in the 17).",
        "The paper's counts of extreme outliers removed per metric (for example 140 for normalized duration) were not reproduced with a "
        "Q1 - 3 IQR / Q3 + 3 IQR rule per participant or per participant and display, so no outlier flag is given.",
    ],
)
stdlib.release(ID, "reach_quality_3d", out, meta, checks)
