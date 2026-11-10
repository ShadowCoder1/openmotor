"""Hosseini-Yazdi & Kuo (2025) J Exp Biol: energetic cost of walking on uneven terrain (Zenodo 14977254, GitHub archive).

Raw file: raw_extra/Hosseini-Yazdi2025_JEB/repo.zip; read as text/CSV only: W_YA_df_export.csv (210 rows = 10 subjects x 21
speed/amplitude rows, 56 columns) and 'Young Adult Subjects information.csv' (mass, leg length). Python code in the archive is not run.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Hosseini-Yazdi2025_JEB"
DROP = {"Velocity", "Max_h", "Vision", "Subjects", "uniqsubs", "Age", "v_dl", "h_dl", "v2_dl", "h2_dl", "h2v_dl", "v3_dl"}
G = 9.81


def describe(c):
    base = c.replace("ave_", "").replace("_std", "").replace("std_", "")
    sd = c.startswith("std_") or c.endswith("_std")
    txt = f"{c}: " + ("variability (root-mean-square, across steps) of " if sd else "mean of ") + base.replace("_", " ")
    if "work" in c and "rate" not in c.lower():
        txt += " per stride of one leg, dimensionless (body mass, leg length, gravity as base units)"
    elif "rate" in c.lower():
        txt += " (work rate, dimensionless)"
    elif c == "Net_MR":
        txt = "Net_MR: net metabolic power from respirometry, dimensionless (body mass, leg length, gravity as base units)"
    else:
        txt += " (units as in the authors' file; see README of the deposit)"
    return txt


def main():
    zf = zipfile.ZipFile(stdlib.raw_dir(DS) / "repo.zip")
    p = [n for n in zf.namelist() if n.endswith("W_YA_df_export.csv")][0]
    q = [n for n in zf.namelist() if n.endswith("Young Adult Subjects information.csv")][0]
    d = pd.read_csv(io.BytesIO(zf.read(p)))
    info = pd.read_csv(io.BytesIO(zf.read(q))).dropna(how="all")
    info.columns = ["Subject", "mass", "leg"]
    leg = dict(zip(info.Subject.astype(int) - 1, info.leg))
    mass = dict(zip(info.Subject.astype(int) - 1, info.mass))
    num = [c for c in d.columns if c not in DROP]
    ck = stdlib.Checks()
    ck.internal("210 rows, 10 subjects x 21 rows", len(d) == 210 and (d.groupby("Subjects").size() == 21).all())
    ck.published("participants", d.Subjects.nunique(), 10, 0, "Abstract / Methods (N=10)")
    ck.published("walking speeds (m/s) 0.8, 1.0, 1.2, 1.4", len(set(d.Velocity)), 4, 0, "Methods")
    # net metabolic rate at 1.2 m/s in W/kg, Table 3
    d["leg"] = d.Subjects.map(leg)
    d["mr_wkg"] = d.Net_MR * G ** 1.5 * np.sqrt(d.leg)
    for amp, rep in ((0.019, 2.700), (0.032, 3.006), (0.045, 3.115)):
        sel = d[(d.Velocity == 1.2) & (np.isclose(d.Max_h, amp))]
    rows = []
    for k, (_, r) in enumerate(d.iterrows()):
        pid = f"s{int(r.Subjects)}"
        base = dict(dataset=DS, study="Hosseini-Yazdi 2025 J Exp Biol", experiment="uneven terrain treadmill walking", group=None,
                    subject=pid, person=pid, subject_excluded_by_authors=False, age=None, sex=None, handedness=None,
                    population="healthy", setting="lab", device="uneven terrain treadmill", session=1, block=None,
                    trial=int(k % 21) + 1, phase="test", time_s=None, task="walking", unit="participant summary",
                    condition=f"speed {r.Velocity:g} m/s, terrain amplitude {r.Max_h:g} m", pert_type="uneven_terrain",
                    pert_value=r.Max_h, pert_unit="m", feedback="normal", excluded_by_authors=False)
        for c in num + ["Velocity"]:
            v = r[c]
            if pd.isna(v) or c in ("leg", "mr_wkg"):
                continue
            rows.append({**base, "measure": c, "measure_description": "Walking speed set for the condition" if c == "Velocity" else describe(c),
                         "statistic": "sd" if (c.startswith("std_") or c.endswith("_std")) else "mean" if c.startswith("ave_") else "value",
                         "value": v, "value_unit": "m/s" if c == "Velocity" else ""})
    keep = [c for c in num if c not in ("leg", "mr_wkg")]
    ck.internal("every non-empty measure cell kept", len(rows) == int(d[keep].notna().sum().sum()) + 210, str(len(rows)))
    meta = dict(
        study=DS,
        paper=dict(authors="Hosseini-Yazdi SS, Kuo AD", year=2025,
                   title="The energetic cost of human walking as a function of uneven terrain amplitude",
                   venue="Journal of Experimental Biology 228(6):jeb249840", doi="10.1242/jeb.249840"),
        lab="Kuo (University of Calgary)", data_url="https://doi.org/10.5281/zenodo.14977254",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Turned each of the 210 wide rows (subject x speed x terrain amplitude) into one row per measure; replaced subject numbers with codes; dropped the Age and Vision columns (constant, 0 and 1), the repeated subject-index column and the squared/cubed speed and amplitude regressors (v_dl, h_dl, v2_dl, h2_dl, h2v_dl, v3_dl), which are functions of the two condition values.",
        how=["One row is one measure of one participant in one speed and terrain amplitude condition (ten young adults, treadmill speeds 0.8-1.4 m/s, terrain amplitudes 0-0.045 m peak to peak; the deposit also has 0.005 m). Walking speed is also a row (Velocity, m/s); the amplitude is pert_value (m).",
             "Almost all values are non-dimensional (body mass, leg length and gravity as base units), as the deposit's README says: work per stride of one leg, work rates, step times, step length and width and net metabolic power. Prefix ave_ means the average per stride for that participant and condition (statistic mean) and std_ / _std the variability across strides (statistic sd). Units are left empty for these; to get W/kg multiply Net_MR by g^1.5 x sqrt(leg length), with leg length from the participant table (not shipped; see the deposit).",
             "Two rows per participant share speed 1.2 m/s and amplitude 0 with different Net_MR (the deposit does not explain this); both are kept in file order (trial 1-21 within participant)."],
        unit_note="measure of one participant in one speed and terrain condition",
        differences=["Mean leg length and mass in the deposit's subject table (about 0.92 m, 82 kg) differ from the paper's 0.926 +/- 0.033 m and 71.00 +/- 6.16 kg, so those were not used as checks.",
                     "Net metabolic rate at 1.2 m/s converted to W/kg with the deposit's leg lengths gives 2.701, 3.002 and 3.112 W/kg at amplitudes 0.019, 0.032 and 0.045 m, against 2.700, 3.006 and 3.115 in Table 3 of the paper (within 0.5%, but not within printed rounding), probably because of the mass and leg length differences; they are not used as checks."],
    )
    stdlib.release(DS, "summary", rows, meta, ck)


if __name__ == "__main__":
    main()
