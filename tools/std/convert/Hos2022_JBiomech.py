"""Hos, van Iersel, van Leeuwen & Bruijn (2022) J Biomech 144:111349: 14 young adults walking on a split-belt treadmill and on a
normal single-belt treadmill, with regular shoes and with LesSchuh (a shoe with a narrow ridge that constrains ankle moments).
The authors share one value per participant, measure and condition (their 'csv files' folder); the Optotrak recordings (.ndf) and
their MATLAB code are not converted.

Raw file: raw_extra/Hos2022_JBiomech/analysis_for_paper.zip (Zenodo 10.5281/zenodo.7188711, CC BY 4.0), the four csv files of its
'csv files' folder read with zipfile; the 'Read me.docx' was read as text. Rows of the csv files are participants, in the authors' order.
Run from the site folder: python3 -I tools/std/convert/Hos2022_JBiomech.py"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Hos2022_JBiomech"
COND = {"splitbelt_normal": ("split-belt treadmill, regular shoes", "split-belt", "regular"),
        "splitbelt_LesSchuh": ("split-belt treadmill, LesSchuh", "split-belt", "LesSchuh"),
        "normbelt_normal": ("normal treadmill, regular shoes", "normal", "regular"),
        "normbelt_LesSchuh": ("normal treadmill, LesSchuh", "normal", "LesSchuh")}
MEASURES = [
    ("toe_angle", "toe_out_angle", "deg", "mean",
     "Toe-out angle of the feet at midstance (positive = toes pointing outward), mean over the strides of the 10-minute analysed part of the condition."),
    ("R2_fishertrans", "foot_placement_R2_fisher", "Fisher z", "value",
     "Degree of foot placement control: explained variance (R2) of a regression of mediolateral foot placement on the mediolateral centre-of-mass "
     "position and velocity (foot placement model of Wang and Srinivasan 2014), Fisher-transformed by the authors (file name R2_fishertrans; "
     "values above 1 therefore occur)."),
    ("stridefreq", "stride_frequency", "Hz", "mean", "Stride frequency, mean over the analysed part of the condition."),
    ("stepwidth", "step_width", "m", "mean", "Step width at midstance: mediolateral distance between the two heel markers, mean over the strides of the condition."),
]


def main():
    raw = stdlib.raw_dir(DS)
    z = zipfile.ZipFile(raw / "analysis_for_paper.zip")
    ck = stdlib.Checks()
    tabs = {}
    for f, mname, unit, stat, desc in MEASURES:
        n = [x for x in z.namelist() if re.search(rf"csv files/{f}_.*\.csv$", x)]
        assert len(n) == 1, (f, n)
        tabs[f] = pd.read_csv(io.BytesIO(z.read(n[0])))
        assert list(tabs[f].columns) == list(COND), tabs[f].columns
    npart = {len(t) for t in tabs.values()}
    assert npart == {14}, npart
    rows = []
    for p in range(14):
        k = 0
        for f, mname, unit, stat, desc in MEASURES:
            for c, (ctext, belt, shoe) in COND.items():
                k += 1
                rows.append(dict(
                    dataset=DS, study=DS, experiment="", group="", subject=f"S{p + 1:02d}", person=f"S{p + 1:02d}",
                    subject_excluded_by_authors=False, population="healthy", setting="lab",
                    device="Single-belt treadmill (C-Mill) and split-belt treadmill, Optotrak cluster markers on the thorax and feet",
                    session=1, block=ctext, trial=k, phase="test", task="walking", unit="participant summary", condition=ctext,
                    pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                    measure=mname, measure_description=desc, statistic=stat, value=tabs[f].loc[p, c], value_unit=unit))
    df = pd.DataFrame(rows)
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for("summary") if c[1] != "person"])
    sf, st, sw, r2 = (tabs[f] for f in ("stridefreq", "toe_angle", "stepwidth", "R2_fishertrans"))
    ck.published("participants", df.subject.nunique(), 14, 0, "Abstract / Methods (fourteen healthy young adults)")
    ck.internal("14 participants x 4 conditions x 4 measures = 224 rows, no missing value", len(df) == 224 and df.value.notna().all())
    lesschuh_sf = (sf.splitbelt_LesSchuh + sf.normbelt_LesSchuh - sf.splitbelt_normal - sf.normbelt_normal).mean() / 2
    belt_sf = (sf.splitbelt_normal + sf.splitbelt_LesSchuh - sf.normbelt_normal - sf.normbelt_LesSchuh).mean() / 2
    ck.published("stride frequency, effect of LesSchuh (Hz)", lesschuh_sf, 0.004, 0.002, "Discussion, limitations (about 0.004 Hz)")
    ck.published("stride frequency, effect of the split-belt treadmill (Hz)", belt_sf, 0.005, 0.002, "Discussion, limitations (about 0.005 Hz)")
    ck.internal("step width is larger with LesSchuh on both treadmills (paper: wider step width on both treadmills)",
                sw.splitbelt_LesSchuh.mean() > sw.splitbelt_normal.mean() and sw.normbelt_LesSchuh.mean() > sw.normbelt_normal.mean())
    ck.internal("step width increase with LesSchuh is larger on the split-belt treadmill",
                (sw.splitbelt_LesSchuh - sw.splitbelt_normal).mean() > (sw.normbelt_LesSchuh - sw.normbelt_normal).mean())
    ck.internal("foot placement R2: lower with LesSchuh on the split-belt, higher on the normal treadmill (paper abstract)",
                r2.splitbelt_LesSchuh.mean() < r2.splitbelt_normal.mean() and r2.normbelt_LesSchuh.mean() > r2.normbelt_normal.mean())
    ck.internal("stride frequency between 0.7 and 1.1 Hz", sf.values.min() > 0.7 and sf.values.max() < 1.1)
    meta = dict(
        paper=dict(authors="Hos M, van Iersel L, van Leeuwen AM, Bruijn SM", year=2022,
                   title="Differential effects of ankle constraints on foot placement control between normal and split belt treadmills",
                   venue="J Biomech 144:111349", doi="10.1016/j.jbiomech.2022.111349"),
        lab="Bruijn (Vrije Universiteit Amsterdam)", data_url="https://doi.org/10.5281/zenodo.7188711",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Reshaped the four csv files of the deposit (toe angle, foot placement R2, stride frequency, step width; one row per "
                 "participant, one column per condition) into one long table, one row per participant, measure and condition, with "
                 "values unchanged. The Optotrak recordings, event files and MATLAB code of the deposit are not included."),
        how=["One row is one measure of one participant in one condition (14 participants x 4 conditions x 4 measures). The measures are the "
             "authors' per-participant values: toe-out angle (deg), the Fisher-transformed explained variance of their foot placement "
             "model, stride frequency (Hz) and step width (m). The conditions cross treadmill (split-belt or normal single-belt) with shoe "
             "(regular or LesSchuh, a shoe with a narrow ridge under the sole that constrains mediolateral ankle moments).",
             "Participants walked 12 minutes per condition (first 2 minutes warm-up) at a speed set from leg length. Age, sex and "
             "condition order are not in the deposit per participant; participant numbers here follow the row order of the authors' csv files. "
             "trial counts the rows in the order measure, then condition, within a participant (not the order in which conditions were walked)."],
        unit_note="measure of one participant in one condition",
        differences=["The deposit holds only summary values; the paper's test statistics (ANOVA interactions) were not recomputed. The checks "
                     "use the paper's participant count and the stride frequency effects it quotes in its limitations section (about "
                     "0.004 Hz for LesSchuh and 0.005 Hz for the split-belt treadmill)."])
    stdlib.release(DS, "summary", df, meta, ck)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))


main()
