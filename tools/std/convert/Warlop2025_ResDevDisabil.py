"""Warlop, Nijhof, Cracco, Wiersema & Deconinck (2025), Research in Developmental Disabilities 167:105170: visuomotor rotation
adaptation on two days (a night of sleep between) in 21 children with developmental coordination disorder (DCD) and 15 without (TD).

Raw files, raw_extra/Warlop2025_ResDevDisabil/ (OSF ycmw4, fetched with tools/std/fetch.py; no license stated):
  DATA_VMR_SLEEP_KINEMATICS.txt  36 rows (21 DCD, 15 TD), tab separated, decimal commas, 38 columns, no codebook. Meaning taken from
      the column names, the task program sVMR.py (centre-out drawing, 4 targets, 40 trials per block: 2 baseline blocks, 6 rotation
      blocks of 45 degrees, 2 post blocks with the rotation removed; sign of the rotation depends on handedness) and the abstract
      (directional error; actigraphy and PSQI sleep measures). SAVINGS equals EARLY_ADAPTATION1_DAY1 minus EARLY_ADAPTATION1_DAY2
      (checked on all rows where it is given; one child has no day-2 data, so no SAVINGS value).
  DATA_VMR_ERN.txt  EEG error-related negativity per child: a neural measure, not motor behaviour, so not converted.
Run from the site folder: python3 -I tools/std/convert/Warlop2025_ResDevDisabil.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Warlop2025_ResDevDisabil"
SRC = "Warlop et al. 2025, Res Dev Disabil 167:105170"
ERR = ("directional error of the movement in degrees (read from the column name and the abstract; the deposit has no codebook)")

# (column, day, phase, block label, pert, feedback, description)
KIN = []
for day in (1, 2):
    KIN.append((f"BASELINE_DAY{day}", day, "baseline", "baseline", 0, "normal",
                f"Baseline (no rotation) performance on day {day}: {ERR}; presumably averaged over the 2 baseline blocks."))
    for k in (1, 2):
        KIN.append((f"EARLY_ADAPTATION{k}_DAY{day}", day, "learning", f"early adaptation {k}", 45, "altered",
                    f"Early adaptation {k} to the 45 degree rotation on day {day}: {ERR}. The meaning of the suffix {k} is inferred "
                    "(early rotation blocks 1 and 2 of the 6)."))
    for k in (1, 2):
        KIN.append((f"FINAL_LEVEL_ADAPTATION{k}_DAY{day}", day, "learning", f"final level {k}", 45, "altered",
                    f"Final level of adaptation {k} on day {day}: {ERR}. The meaning of the suffix {k} is inferred (late rotation blocks)."))
    for k in (1, 2):
        KIN.append((f"AFTEREFFECTS{k}_DAY{day}", day, "aftereffect", f"after-effect {k}", 0, "normal",
                    f"After-effect {k} on day {day} after the rotation was removed: {ERR}. The suffix {k} is inferred to be the 2 post blocks."))

PCOLS = [
    ("writing_hand", "category", "R, L", "Writing hand (WRITING_HAND)."),
    ("ehi_category", "category", "R, L, AMBI", "Edinburgh Handedness Inventory category (EHI_HANDEDNESS_CAT)."),
    ("ehi_score", "number", "", "Edinburgh Handedness Inventory score (EHI_HANDEDNESS_SCORE)."),
    ("mabc2_total_percentile", "number", "percentile", "Movement ABC-2 total score as a percentile (MABC2_PERC_TOT)."),
    ("mabc2_manual_dexterity_percentile", "number", "percentile", "Movement ABC-2 manual dexterity percentile (MABC2_PERC_FINE; read as the fine-motor component)."),
    ("mabc2_aiming_catching_percentile", "number", "percentile", "Movement ABC-2 aiming and catching percentile (MABC2_PERC_TC; read from the abbreviation as throwing and catching)."),
    ("mabc2_balance_percentile", "number", "percentile", "Movement ABC-2 balance percentile (MABC2_PERC_BALANCE)."),
] + [(f"psqi_component_{i}", "integer", "points 0-3", f"Pittsburgh Sleep Quality Index component {i} (PSQI_COMP{i}).") for i in range(1, 8)] + [
    ("psqi_total", "integer", "points", "PSQI total score (PSQI_TOTAL)."),
    ("actigraphy_sleep_latency", "number", "min", "Actigraphy sleep latency (ACTIGR_LATENCY; unit inferred)."),
    ("actigraphy_sleep_efficiency", "number", "percent", "Actigraphy sleep efficiency (ACTIGR_EFFICIENCY; unit inferred)."),
    ("actigraphy_total_sleep_time", "number", "min", "Actigraphy total sleep time (ACTIGR_TST; unit inferred from values of 330-530)."),
    ("actigraphy_fragmentation_index", "number", "", "Actigraphy sleep fragmentation index (ACTIGR_SLEEP_FRAGM_INDEX)."),
]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def main():
    d = pd.read_csv(stdlib.raw_dir(ID) / "DATA_VMR_SLEEP_KINEMATICS.txt", sep="\t", decimal=",", dtype={"PPN_CODE": str})
    rows, parts = [], []
    for r in d.to_dict("records"):
        pers = {"dataset": ID, "study": SRC, "experiment": "", "group": r["GROUP"], "subject": r["PPN_CODE"],
                "subject_excluded_by_authors": False, "age": r["AGE"], "sex": r["SEX"],
                "handedness": {"R": "right", "L": "left", "AMBI": "ambi"}[r["EHI_HANDEDNESS_CAT"]],
                "population": "developmental coordination disorder" if r["GROUP"] == "DCD" else "typically developing",
                "setting": "lab", "device": "centre-out drawing task with a mouse and EEG, on two days"}
        k = 0
        items = [(c, day, ph, bl, pv, fb, ds, "mean") for c, day, ph, bl, pv, fb, ds in KIN]
        items.append(("SAVINGS", 2, "other", "savings", 45, "altered",
                      "Savings: early adaptation 1 on day 1 minus early adaptation 1 on day 2 (checked equal to EARLY_ADAPTATION1_DAY1 - "
                      "EARLY_ADAPTATION1_DAY2); positive when the child started day 2 with a smaller error. Degrees.", "other"))
        for c, day, ph, bl, pv, fb, ds, stat in items:
            k += 1
            rows.append({**pers, "session": day, "block": bl, "trial": k, "phase": ph, "time_s": "", "task": "reaching",
                         "unit": "participant summary", "condition": f"day {day}", "pert_type": "visuomotor_rotation" if pv else "none",
                         "pert_value": pv, "pert_unit": "deg", "feedback": fb, "excluded_by_authors": False,
                         "measure": c.lower(), "measure_description": ds, "statistic": stat, "value": r[c], "value_unit": "deg", "n_trials": ""})
        p = {"person": r["PPN_CODE"], "writing_hand": r["WRITING_HAND"], "ehi_category": r["EHI_HANDEDNESS_CAT"],
             "ehi_score": r["EHI_HANDEDNESS_SCORE"], "mabc2_total_percentile": r["MABC2_PERC_TOT"],
             "mabc2_manual_dexterity_percentile": r["MABC2_PERC_FINE"], "mabc2_aiming_catching_percentile": r["MABC2_PERC_TC"],
             "mabc2_balance_percentile": r["MABC2_PERC_BALANCE"], "psqi_total": r["PSQI_TOTAL"],
             "actigraphy_sleep_latency": r["ACTIGR_LATENCY"], "actigraphy_sleep_efficiency": r["ACTIGR_EFFICIENCY"],
             "actigraphy_total_sleep_time": r["ACTIGR_TST"], "actigraphy_fragmentation_index": r["ACTIGR_SLEEP_FRAGM_INDEX"]}
        p.update({f"psqi_component_{i}": r[f"PSQI_COMP{i}"] for i in range(1, 8)})
        parts.append(p)
    checks = stdlib.Checks()
    A = "Abstract"
    dc, td = d[d.GROUP == "DCD"], d[d.GROUP == "TD"]
    checks.published("children with DCD", len(dc), 21, 0, f"{SRC}, {A}")
    checks.published("children without DCD", len(td), 15, 0, f"{SRC}, {A}")
    checks.published("DCD mean age (years)", dc.AGE.mean(), 13.7, 0.05, f"{SRC}, {A}")
    checks.published("DCD age SD", dc.AGE.std(ddof=1), 1.5, 0.05, f"{SRC}, {A}")
    checks.published("TD mean age (years)", td.AGE.mean(), 14.2, 0.05, f"{SRC}, {A}")
    checks.published("TD age SD", td.AGE.std(ddof=1), 1.4, 0.05, f"{SRC}, {A}")
    checks.internal("SAVINGS = EARLY_ADAPTATION1_DAY1 - EARLY_ADAPTATION1_DAY2 on every row where SAVINGS is given",
                    bool(np.allclose(d.SAVINGS.dropna(), (d.EARLY_ADAPTATION1_DAY1 - d.EARLY_ADAPTATION1_DAY2)[d.SAVINGS.notna()], atol=1e-4)))
    checks.internal("36 children x 15 measures, one child has no day-2 data (8 empty values)", len(rows) == 36 * 15 and int(pd.DataFrame(rows).value.isna().sum()) == 8, str(len(rows)))
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Converted decimal commas to points, stacked the kinematic columns into long form (one row per child and measure), put "
                 "age, sex, handedness, motor-test, sleep-questionnaire and actigraphy values in the standard columns and participants.csv, "
                 "and replaced participant numbers with codes. Left out DATA_VMR_ERN.txt (EEG error-related negativity: neural, not movement)."),
        how=["One row is one kinematic measure of one child (the deposit has one value per child and phase, no per-trial data). Children drew "
             "from a centre to one of 4 targets on a tablet-like display on two days with a night of sleep between; each day had 2 baseline "
             "blocks, 6 blocks with the cursor rotated by 45 degrees (sign set by handedness) and 2 post blocks without rotation, 40 trials per block "
             "(from the task program sVMR.py). The values are directional errors in degrees.",
             "The deposit has no codebook. That the values are directional errors in degrees, and that the suffixes 1 and 2 of the "
             "early-adaptation, final-level and after-effect columns are the first and second of two blocks, are inferred from the column "
             "names, the task program and the abstract, and are not stated by the authors. The error is possibly an absolute error; "
             "values are all positive except SAVINGS and a few small after-effects. Units of the actigraphy values are inferred."],
        unit_note="kinematic measure of one child (long form)",
        differences=["One child has no day-2 kinematics (and no actigraphy), so that child's 8 day-2 values and SAVINGS are empty, as in the deposit. The paper's group statistics (ANOVAs on directional error, EEG error-related negativity, sleep) were not recomputed; "
                     "only the group sizes and the ages given in the abstract were checked."],
    )
    return stdlib.release(ID, "summary", rows, meta, checks, participants=parts, participant_columns=PCOLS)


if __name__ == "__main__":
    main()
