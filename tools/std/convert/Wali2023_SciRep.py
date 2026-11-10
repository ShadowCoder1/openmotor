"""Wali, Lee-Miller, Babu and Block (2023), Sci Rep 13:6097: retention of visuo-proprioceptive recalibration. 'summary' module,
one row per participant and measure (recalibration in block 1 and retention in block 2 of vision, proprioception and their sum).

Run from the site folder: python3 -I tools/std/convert/Wali2023_SciRep.py
Raw file (untrusted data, read only) in raw_extra/Wali2023_SciRep/ from https://osf.io/n4tbj/ : MixedAnova_data.xlsx (one row per participant).
Not carried over: the JASP/PDF statistics outputs."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wali2023_SciRep"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
df = pd.read_excel(RAW / "MixedAnova_data.xlsx")

GROUPS = {"AV": "AV (Group 1: active movement, hand visible)", "PV": "PV (Group 2: rest, hand visible; called RV in the paper)",
          "ANV": "ANV (Group 3: active movement, hand not visible)", "PNV": "PNV (Group 4: rest, hand not visible; called RNV in the paper)",
          "24H": "24H (Group 5: 24 h between blocks)"}
MEAS = [("V_ra", "visual_recalibration_block1", "Block 1 (recalibration)", "Visual recalibration in block 1 (authors' V_ra)."),
        ("V_ret", "visual_retention_block2", "Block 2 (retention)", "Visual recalibration still present in block 2 (authors' V_ret)."),
        ("P_ra", "proprioceptive_recalibration_block1", "Block 1 (recalibration)", "Proprioceptive recalibration in block 1 (authors' P_ra)."),
        ("P_ret", "proprioceptive_retention_block2", "Block 2 (retention)", "Proprioceptive recalibration still present in block 2 (authors' P_ret)."),
        ("Total_ra", "total_recalibration_block1", "Block 1 (recalibration)", "Visual plus proprioceptive recalibration in block 1 (authors' Total_ra)."),
        ("Total_ret", "total_retention_block2", "Block 2 (retention)", "Visual plus proprioceptive retention in block 2 (authors' Total_ret).")]
rows = []
for i, r in enumerate(df.itertuples(index=False)):
    g = GROUPS[r.Group]
    for k, (col, name, blk, desc) in enumerate(MEAS):
        v = getattr(r, col)
        if pd.isna(v):
            continue
        rows.append(dict(
            dataset=ID, study=ID, experiment=ID, group=g, subject=f"p{i}", subject_excluded_by_authors=False,
            age=None, sex=None, handedness="right", time_s=None, population="healthy", setting="lab",
            device="2D virtual reality apparatus: two-sided touchscreen, mirror and rear projection screen; right index finger indicates target positions",
            session=1 if r.Group != "24H" or "ra" in col else 2, block=blk, trial=k + 1,
            phase="learning" if col.endswith("_ra") else "test", task="reaching", unit="participant summary", condition="visuo-proprioceptive 70 mm mismatch",
            pert_type="visuo_proprioceptive_offset", pert_value=70, pert_unit="mm", feedback="altered", excluded_by_authors=False,
            measure=name, measure_description=desc, statistic="value", value=float(v), value_unit="mm", n_trials=None))

n = df.Group.value_counts()
checks.published("participants analysed (75 recruited, one excluded each from groups 3 and 4)", len(df), 73, 0, "Methods, Participants and Statistical analysis")
for g, exp in {"AV": 15, "PV": 15, "ANV": 14, "PNV": 14, "24H": 15}.items():
    checks.published(f"participants in group {g}", n[g], exp, 0, "Methods: 5 groups of 15, one excluded each from groups 3 and 4")
checks.published("24H group visual retention (mm)", df[df.Group == "24H"].V_ret.mean(), 3.2, 0.05, "Results, Visual recalibration and retention")
checks.published("mixed ANOVA error degrees of freedom (N - 5 groups)", len(df) - 5, 68, 0, "Results: F(4,68)")
checks.internal("Total_ra = V_ra + P_ra within 0.02 mm", bool(((df.Total_ra - df.V_ra - df.P_ra).abs() < 0.02).all()))
checks.internal("Total_ret = V_ret + P_ret within 0.02 mm", bool(((df.Total_ret - df.V_ret - df.P_ret).abs() < 0.02).all()))
checks.internal("no missing values", int(df.isna().sum().sum()) == 0)
checks.internal("rows = participants x 6 measures", len(rows) == 6 * len(df))
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Block (Indiana University)", data_url="https://osf.io/n4tbj/", license="none stated on the OSF project (the paper is CC BY 4.0)",
    license_url="https://osf.io/n4tbj/",
    changes="Converted to the OpenMotor summary format: the single sheet reshaped to one row per participant and measure; participants "
            "coded; group names kept as in the file with the paper's descriptions added.",
    how=["One row is one measure of one participant, in mm. Recalibration (_ra) is the shift in the finger-pointed position of visual (V) and "
         "proprioceptive (P) targets across the 70 mm visuo-proprioceptive mismatch block (block 1); retention (_ret) is the same shift measured "
         "in block 2 after the intervention (rest or active movement, with or without sight of the hand, or a 24 h gap). Total is V plus P.",
         "Signs are as in the authors' file: positive means a shift in the direction the mismatch pushes the estimate. Rows are in the file's order; "
         "participant age and sex were not shared per person (the paper reports 55 women and 20 men of 75 recruited)."],
    unit_note="participant summary", differences=[
        "The file labels groups AV, PV, ANV, PNV and 24H; the paper calls groups 2 and 4 'rest' (RV, RNV). The file has 73 participants because the "
        "paper's 75 includes one person each from groups 3 and 4 excluded for touchscreen problems; the excluded people are not in the deposit.",
        "Two printed group means did not reproduce within the paper's rounding and are not used as checks: the rest-vision group's visual retention is "
        "36.58 mm from the file (paper: 36.5 mm) and the 24H group's visual recalibration is 38.84 mm from the file (paper: 38.3 mm).",
        "Handedness 'right' is from the Methods (all participants right-handed), not a per-person column."])
stdlib.release(ID, "summary", rows, meta, checks)
