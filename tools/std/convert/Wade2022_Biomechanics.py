"""Wade et al. (2022) Biomechanics: locomotor adaptation training to prevent mobility disability.

Raw file: raw_extra/Wade2022_Biomechanics/Formatted_GAITstudy_data.xlsx (Zenodo 6584790): sheet Data (38 participants x 58 columns,
row 1 = dimension labels, row 2 = column names), sheet Codes. Nothing is executed.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wade2022_Biomechanics"
GROUPS = {1: "control", 2: "traditional treadmill training", 3: "split-belt treadmill training"}
UNIT = {"TrailsA": "s", "TrailsB": "s", "TMT": "s", "TUG": "", "WalkingSpeed": "", "Cadence": "", "StrideLength": "",
        "StrideTime": "", "StepWidth": "", "StanceTime": "% gait cycle"}
DIM = [("MMSE", "Cognitive function"), ("Trails", "Cognitive function"), ("TMT", "Cognitive function"),
       ("SPPB", "Clinical function"), ("DGI", "Clinical function"), ("TUG", "Clinical function"),
       ("Peak", "Kinetic gait parameter"), ("Concentric", "Kinetic gait parameter"), ("VO2", "Cardiovascular fitness"),
       ("HR", "Cardiovascular fitness"), ("RPE", "Cardiovascular fitness"), ("Efficiency", "Cardiovascular fitness")]


def dim(name):
    for k, v in DIM:
        if k in name:
            return v
    return "Spatiotemporal gait parameter"


def main():
    d = pd.read_excel(stdlib.raw_dir(DS) / "Formatted_GAITstudy_data.xlsx", sheet_name="Data", header=1)
    d = d[d.Subject.notna()].reset_index(drop=True)
    cols = [c for c in d.columns if c.startswith(("PRE_", "POST_"))]
    ck = stdlib.Checks()
    ck.internal("38 rows, unique participants, 26 PRE/POST column pairs",
                len(d) == 38 and d.Subject.nunique() == 38 and len(cols) == 52)
    ck.published("participants randomized and analysed", len(d), 38, 0, "Abstract (38 sedentary older adults)")
    rows, parts = [], []
    for _, r in d.iterrows():
        pid = str(r.Subject)
        parts.append(dict(person=pid, mass_kg=r["Mass (kg)"], height_m=r["Height (m)"]))
        for c in cols:
            tp, name = c.split("_", 1)
            v = r[c]
            if pd.isna(v):
                continue
            m = re.match(r"(.*?)\s*\((.*)\)$", name)
            unit = ""
            if m and not m.group(2).startswith("Trails"):
                name, unit = m.group(1), m.group(2)
            else:
                base = name.replace("SD_", "")
                unit = UNIT.get(base, "")
                if name == "TMT (Trails B - Trails A)":
                    unit = "s"
            sd = name.startswith("SD_")
            desc = f"{dim(name)}: {name.replace('_', ' ')}" + (" (standard deviation across strides, the authors' variability measure)" if sd else "")
            g = int(r.Group)
            rows.append(dict(dataset=DS, study="Wade 2022 Biomechanics", experiment="16-week training trial", group=GROUPS[g],
                             subject=pid, person=pid, subject_excluded_by_authors=False, age=r.Age, sex="M" if r.Sex == 1 else "F",
                             handedness=None, population="older adults at risk of mobility disability", setting="lab",
                             device=None, session=1 if tp == "PRE" else 2, block=tp, trial=None,
                             phase="baseline" if tp == "PRE" else "test", time_s=None, task="walking",
                             unit="participant summary", condition=tp, pert_type="none", pert_value=0, pert_unit=None,
                             feedback="normal", excluded_by_authors=False, measure=name,
                             measure_description=desc, statistic="sd" if sd else "value", value=v, value_unit=unit))
    ck.internal("every non-empty PRE/POST cell kept", len(rows) == int(d[cols].notna().sum().sum()), str(len(rows)))
    ck.internal("groups 1/2/3 all present", set(d.Group.astype(int)) == {1, 2, 3})
    meta = dict(
        study=DS,
        paper=dict(authors="Wade F, Baudendistel S, Stone A, Roper J, Raffegeau T, Terza M, Hass C",
                   year=2022, title="Locomotor Adaptation Training to Prevent Mobility Disability",
                   venue="Biomechanics 2(3):395-420", doi="10.3390/biomechanics2030031"),
        lab="Hass (University of Florida)", data_url="https://doi.org/10.5281/zenodo.6584790",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Turned the wide spreadsheet (PRE_ and POST_ columns) into a long table with one row per participant, time point and measure; replaced subject ids with codes; moved mass and height to participants.csv; empty cells were not written.",
        how=["One row is one measure of one participant at one time point (PRE before, POST after a 16-week intervention). Groups: control, traditional treadmill training, split-belt treadmill training.",
             "Measures are grouped in the description as cognitive function (MMSE, Trails A and B, their difference), clinical function (SPPB, DGI, TUG), spatiotemporal gait (self-selected speed, cadence, stride length and time, step width, stance time and the standard deviations of each; statistic is 'sd' for those), kinetic gait and cardiovascular fitness. Units are filled where the column name states them or the dataset description names them (Trails in s, stance time in % of gait cycle, as the Zenodo description says); other units are left empty because the deposit does not state them (for example gait speed is probably m/s).",
             "Sex is coded 0 = female, 1 = male in the deposit's Codes sheet."],
        unit_note="measure of one participant at one time point",
        differences=["The paper text could not be read (publisher site blocked automated access), so only the participant count from the abstract was checked."],
    )
    pc = [("mass_kg", "number", "kg", "Body mass."), ("height_m", "number", "m", "Body height.")]
    stdlib.release(DS, "summary", rows, meta, ck, participants=parts, participant_columns=pc)


if __name__ == "__main__":
    main()
