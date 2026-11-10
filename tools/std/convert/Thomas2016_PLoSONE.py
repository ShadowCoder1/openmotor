"""Thomas, Johnsen, Geertsen, Christiansen, Ritz, Roig & Lundbye-Jensen (2016), PLoS ONE 11(7):e0159589: acute exercise and
consolidation of a visuomotor accuracy tracking (VAT) skill in 36 healthy men (rest control, exercise at 45% and at 90% of Wmax).

Raw file, raw_extra/Thomas2016_PLoSONE/ (Zenodo 57508, CC0): Thomas_et_al_DATA.xlsx. Sheets used: Acquisition (blocks B1-B5),
Retention (B5, R24 = 1 day, R7 = 7 days) and Tracking Saturation (R7, R7Tr), each long form with one row per trial score
(Subject 1-36, Time = block, Group, Block = the participant number again, Score 0-100). Sheets Exercise Data, Lactate and
Questionnaires hold heart rate, effort, blood lactate and mood questionnaires, not motor behaviour, and are not converted.
Run from the site folder: python3 -I tools/std/convert/Thomas2016_PLoSONE.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Thomas2016_PLoSONE"
SRC = "Thomas et al. 2016, PLoS ONE 11:e0159589"
GROUPS = {"CON": "rest control", "EX45": "exercise at 45% of Wmax", "EX90": "exercise at 90% of Wmax"}
# block label -> (session, phase)
BLOCKS = {"B1": (1, "baseline"), "B2": (1, "learning"), "B3": (1, "learning"), "B4": (1, "learning"), "B5": (1, "learning"),
          "R24": (2, "test"), "R7": (3, "test"), "R7Tr": (3, "learning")}
NO_FB = {"R24", "R7"}


def main():
    f = stdlib.raw_dir(ID) / "Thomas_et_al_DATA.xlsx"
    acq = pd.read_excel(f, sheet_name="Acquisition")
    ret = pd.read_excel(f, sheet_name="Retention")
    sat = pd.read_excel(f, sheet_name="Tracking Saturation")
    checks = stdlib.Checks()
    # the Retention sheet repeats B5 and the Tracking Saturation sheet repeats R7: keep them once
    b5a = acq[acq.Time == "B5"].reset_index(drop=True)
    b5r = ret[ret.Time == "B5"].reset_index(drop=True)
    checks.internal("B5 rows of the Retention sheet are identical to B5 of the Acquisition sheet",
                    bool(b5a[["Subject", "Group"]].equals(b5r[["Subject", "Group"]]) and np.allclose(b5a.Score, b5r.Score, equal_nan=True)))
    r7a = ret[ret.Time == "R7"].reset_index(drop=True)
    r7s = sat[sat.Time == "R7"].reset_index(drop=True)
    checks.internal("R7 rows of the Tracking Saturation sheet are identical to R7 of the Retention sheet",
                    bool(r7a[["Subject", "Group"]].equals(r7s[["Subject", "Group"]]) and np.allclose(r7a.Score, r7s.Score, equal_nan=True)))
    checks.internal("Block column equals Subject in every sheet (it is the participant number, not a block)",
                    all((d.Block == d.Subject).all() for d in (acq, ret, sat)))
    d = pd.concat([acq, ret[ret.Time == "R24"], ret[ret.Time == "R7"], sat[sat.Time == "R7Tr"]], ignore_index=True)
    d["_o"] = d.Time.map({k: i for i, k in enumerate(BLOCKS)})
    d = d.sort_values(["Subject", "_o"], kind="stable").reset_index(drop=True)
    rows = []
    for sub, g in d.groupby("Subject", sort=True):
        k = 0
        for r in g.itertuples(index=False):
            k += 1
            sess, ph = BLOCKS[r.Time]
            rows.append(dict(
                dataset=ID, study=ID, experiment="", group=GROUPS[r.Group], subject=f"S{int(sub):02d}", subject_excluded_by_authors=False,
                age="", sex="M", handedness="right", population="healthy", setting="lab",
                device="isometric wrist flexion-extension torque handle moving a cursor on a screen",
                session=sess, block=r.Time, trial=k, phase=ph, time_s="", task="tracking", unit="trial", condition=r.Time,
                pert_type="none", pert_value=0, pert_unit="", feedback="" if r.Time in NO_FB else "normal", excluded_by_authors=False,
                measure="VAT score", measure_description="Visuomotor accuracy tracking score, 0 to 100, from the mean vertical distance "
                "between cursor and the target curve scaled by the distance from target to baseline (100 = no error)",
                statistic="value", value=r.Score, value_unit="score", n_trials=1))
    # published checks: group means of the shared trial scores = the means the paper's text gives (model estimates equal the raw means)
    m = d.groupby(["Group", "Time"]).Score.mean()
    P = f"{SRC}, Results (motor skill)"
    for (grp, blk), rep in [(("EX90", "B5"), 70.48), (("EX45", "R7"), 73.54), (("EX90", "R7"), 73.95), (("CON", "R7"), 69.66),
                            (("EX90", "R7Tr"), 74.17), (("EX45", "R7Tr"), 74.72), (("CON", "R7Tr"), 71.98), (("EX45", "R24"), 71.56)]:
        checks.published(f"mean score, {grp} at {blk}", m[(grp, blk)], rep, 0.005, P)
    checks.published("EX90: R7 minus B5", m[("EX90", "R7")] - m[("EX90", "B5")], 3.47, 0.005, P)
    checks.published("participants", d.Subject.nunique(), 36, 0, f"{SRC}, Methods (Participants)")
    checks.internal("12 participants in each of the three groups", bool((d.groupby("Group").Subject.nunique() == 12).all()))
    checks.internal("19 scores per participant and block, 36 participants, 8 blocks of the design",
                    bool((d.groupby(["Subject", "Time"]).size() == 19).all() and len(d) == 36 * 8 * 19), f"{len(d)} rows")
    checks.internal("score range 0-100", bool(d.Score.min() >= 0 and d.Score.max() <= 100))
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s["papers"][0]
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC0 1.0 (Zenodo: cc-zero)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Stacked the Acquisition sheet (B1-B5), the R24 and R7 blocks of the Retention sheet and the R7Tr block of the Tracking "
                 "Saturation sheet into one table per trial score, in block order; left out the repeated B5 and R7 rows of the later "
                 "sheets (checked identical), left out the exercise, lactate and questionnaire sheets, labelled the sheet's Time column "
                 "as block and replaced participant numbers with codes."),
        how=[
            "One row is one trial of the visuomotor accuracy tracking task: the participant moved a cursor up and down by isometric wrist "
            "flexion and extension to follow a fixed triple sine-wave curve while the cursor travelled left to right in 8 s. Value is the "
            "trial score (0 to 100, 100 = no error; 0 when the RMS error exceeded twice the target-to-baseline distance).",
            "The deposit has 19 scores per block (trials 2 to 20 of the 20 trials the paper describes; trial 1 of each block was not "
            "shared). The trial column counts the shared rows within a participant, and their order within a block is taken to be trial "
            "order (inferred from the layout, not stated). Blocks B1 to B5 are acquisition on the first day (B1 is the baseline block, "
            "mean of its trials 2 to 20), R24 and R7 are retention blocks 1 and 7 days later without feedback per the paper "
            "(the deposit labels the 1-day block R24, the paper R1), and R7Tr is a 20-trial training block with feedback after R7. The "
            "feedback column is empty for the retention blocks.",
            "Group is rest control (remained seated), or cycling at 45% or 90% of maximal power starting 20 min after acquisition. All "
            "participants were healthy right-handed men aged 24 +- 3 years (no per-participant age was shared).",
        ],
        unit_note="trial score of the visuomotor accuracy tracking task",
        differences=[
            "Missing scores are kept as empty cells: 3 single trials in the acquisition blocks and all 19 trials of one participant "
            "in R24. The paper's linear-mixed-model estimates differ slightly from raw group means where scores are missing (for example "
            "EX90 at R24: 72.45 here against 72.63 in the paper), so only means that match to the printed two decimals are used as checks.",
        ],
    )
    return stdlib.release(ID, "summary", rows, meta, checks)


if __name__ == "__main__":
    main()
