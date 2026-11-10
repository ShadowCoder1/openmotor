"""Massoud et al. (2025) Sci Rep: gait of children with cerebral palsy (Zenodo 12793487).

Raw file: raw_extra/Massoud_CPgait_Zenodo/Data.xlsx, sheets 'Patient before', 'Patient after', 'normal children'. Each sheet holds two
blocks: rows 5-24 'time of angles' (KT1-3, AT1-4, HT1-3) and rows 30-49 'angles' (K1-3, A1-4, H1-3), a right and a left limb row per
child. The patient-name column is not read as a label (children are renumbered). Nothing is executed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Massoud_CPgait_Zenodo"
SHEETS = [("Patient before", "cerebral palsy", "before surgery"), ("Patient after", "cerebral palsy", "after surgery"),
          ("normal children", "typically developing", "")]
JOINT = {"K": "knee", "A": "ankle", "H": "hip"}
TD = {"1": "initial maximum phase", "2": "minimum angle", "3": "secondary maximum phase"}
ANK = {"1": "first low phase", "2": "first high phase", "3": "second low phase", "4": "second high phase"}


def desc(code, time):
    j = JOINT[code[0]]
    n = code[-1]
    ph = (ANK if j == "ankle" else TD)[n]
    return (f"Time of the {ph} of the {j} angle in the gait cycle (percent of the gait cycle, inferred from values 0-100)" if time
            else f"Angle of the {j} during the {ph} (degrees)")


def main():
    path = stdlib.raw_dir(DS) / "Data.xlsx"
    rows, ck = [], stdlib.Checks()
    people = {}
    for sheet, grp, cond in SHEETS:
        d = pd.read_excel(path, sheet_name=sheet, header=None)
        for lo, hi, time in ((4, 24, True), (29, 49, False)):
            blk = d.iloc[lo:hi]
            names = blk[0].ffill()
            heads = [str(x).strip().upper() for x in d.iloc[3 if time else 28, 2:].tolist()]
            for (i, r), nm in zip(blk.iterrows(), names):
                if r[1] not in ("R", "L"):
                    continue
                pid = f"{grp[:2]}_{str(nm).strip()}"
                people.setdefault(grp, set()).add(pid)
                for h, v in zip(heads, r.iloc[2:].tolist()):
                    if h in ("NAN", "") or pd.isna(v):
                        continue
                    rows.append(dict(dataset=DS, study="Massoud 2025 Sci Rep", experiment="gait analysis", group=grp,
                                     subject=pid, person=pid, subject_excluded_by_authors=False, age=None, sex=None,
                                     handedness=None, population=grp if grp != "typically developing" else "healthy",
                                     setting="lab", device="six-camera motion capture", session=None, block=cond or None,
                                     trial=None, phase="baseline" if cond != "after surgery" else "test", time_s=None,
                                     task="walking", unit="participant summary", condition=cond or "typically developing",
                                     pert_type="none", pert_value=0, pert_unit=None, feedback="normal",
                                     excluded_by_authors=False, measure=f"{h}_{r[1]}",
                                     measure_description=("Right limb: " if r[1] == "R" else "Left limb: ") + desc(h, time),
                                     statistic="value", value=v, value_unit="% of gait cycle" if time else "deg"))
    ck.published("children with cerebral palsy", len(people["cerebral palsy"]), 10, 0, "Methods, Participants")
    cp = [r for r in rows if r["group"] == "cerebral palsy"]
    ck.internal("CP rows = 10 children x 2 limbs x 2 sessions x 20 values", len(cp) == 800, str(len(cp)))
    ck.internal("typically developing rows = 9 children x 2 limbs x 20 values", len(rows) - len(cp) == 360, str(len(rows) - len(cp)))
    meta = dict(
        study=DS,
        paper=dict(authors="Massoud S, Ismaiel E, Massoud R, Khadour L, Al-mawaldi M",
                   year=2025, title="Multimodal fuzzy logic-based gait evaluation system for assessing children with cerebral palsy",
                   venue="Sci Rep 15(1):1372", doi="10.1038/s41598-025-85172-2"),
        lab="Massoud (Damascus University)", data_url="https://zenodo.org/records/12793487",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Turned the three sheets (two blocks each) into one long table with one row per child, limb and joint feature; the patient-name column was replaced by codes; no values changed.",
        how=["One row is one joint feature of one limb (R or L) of one child. Features are the knee (K1-K3), ankle (A1-A4) and hip (H1-H3) angle values at the phases of the gait cycle, in degrees, and the times at which they occur (KT, AT, HT). Times are not given a unit in the deposit; values lie between about -70 and 100, so percent of the gait cycle is inferred, and a few negative times appear and are kept.",
             "Ten children with spastic cerebral palsy were measured before and after multilevel surgery (condition); twelve typically developing children are described in the paper. The sheet titled 'Patient before' carries the text 'after surgery' in its title cells; we followed the sheet names. Ages and sexes are not given per child.",
             "Feature definitions follow Table 1 of the paper (K1 angular displacement during initial maximum phase, K2 minimum angle, K3 secondary maximum, etc.)."],
        unit_note="joint feature of one limb of one child",
        differences=["The paper describes 12 typically developing children (and 10 in its tables); the spreadsheet has 9, which is what is released.",
                     "Sheet title cells say 'after surgery' on both patient sheets; the sheet names are used."],
    )
    stdlib.release(DS, "summary", rows, meta, ck)


if __name__ == "__main__":
    main()
