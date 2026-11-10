"""Ikegami et al. (2021), PLoS Comput Biol: hierarchical motor adaptations negotiate failures during force field learning.
Six conditions (VDCF, LIPF in experiment 1; PSPF, CPVF in experiment 2; LIPF-NULL, LIPF-PEC in experiment 3), 119 participant
sequences, 330 reaches each (the last 20 baseline reaches, then the 310-reach adaptation session), one row per reach with the
authors' lateral deviation (LD) and target error (TE), in the force-field reaching format.

Run from the site folder: python3 -I tools/std/convert/Ikegami2021_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Ikegami2021_PLoSCompBiol/ from https://doi.org/10.5061/dryad.5x69p8d2f
(Zenodo copy, record 4725997): Data_<condition>.mat, each with Data.BL.{LD,TE} (20 x n) and Data.AD.{LD,TE} (310 x n), in m,
trials x participants, and ReadMe.txt. The simulation files, code and Traj.mat (group trajectories at 15 positions, for Fig 3)
are not converted.
Structure of AD (paper, Methods): 5 null reaches, 155 field reaches, 150 de-adaptation reaches (null; in LIPF-PEC the
perpendicular-error-clamp field). Checked below against the paper's first-trial values.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Ikegami2021_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
COND = {  # file -> (condition, experiment, field description, pert_value, pert_unit)
    "VDCF": ("VDCF", "exp1", "velocity-dependent curl field", 14, "N s/m"),
    "LIPF": ("LIPF", "exp1", "position-dependent (limb-impedance) field", 60, "N/m"),
    "PSPF": ("PSPF", "exp2", "position- and speed-dependent field", "", ""),
    "CPVF": ("CPVF", "exp2", "combined position and velocity field", "", ""),
    "LIPFNULL": ("LIPF-NULL", "exp3", "position-dependent field, null de-adaptation", 60, "N/m"),
    "LIPFPEC": ("LIPF-PEC", "exp3", "position-dependent field, perpendicular-error-clamp de-adaptation", 60, "N/m"),
}
rows = []
TEs = {}
for f, (cond, exp, _d, pv, pu) in COND.items():
    d = sio.loadmat(RAW / f"Data_{f}.mat", squeeze_me=True, struct_as_record=False)["Data"]
    BLte, BLld, ADte, ADld = (np.asarray(x, dtype=float) * 1000.0 for x in (d.BL.TE, d.BL.LD, d.AD.TE, d.AD.LD))  # m -> mm
    assert BLte.shape[0] == 20 and ADte.shape[0] == 310 and BLte.shape[1] == ADte.shape[1]
    TEs[f] = ADte
    for p in range(ADte.shape[1]):
        for t in range(330):
            if t < 20:
                te, ld, blk, ph, ttype = BLte[t, p], BLld[t, p], "baseline_last20", "baseline", "null"
            else:
                a = t - 20
                te, ld = ADte[a, p], ADld[a, p]
                if a < 5:
                    blk, ph, ttype = "adaptation_session_null_start", "baseline", "null"
                elif a < 160:
                    blk, ph, ttype = "adaptation", "learning", "field"
                else:
                    blk, ph = "de_adaptation", "washout"
                    ttype = "clamp" if f == "LIPFPEC" else "null"
            miss = not (np.isfinite(te) and np.isfinite(ld))
            fld = ttype == "field"
            rows.append(dict(
                dataset=ID, study=ID, experiment=exp, group=cond, subject=f"{f}_{p+1}", person=f"{f}_{p+1}",
                subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                device="TVINS air-magnet manipulandum", session=1, block=blk, trial=t + 1, phase=ph, time_s="", task="reaching",
                unit="trial", condition=cond, pert_type="force_field" if fld else "none", pert_value=pv if fld else 0,
                pert_unit=pu if fld else "", feedback="normal", excluded_by_authors=miss, trial_type=ttype,
                trial_in_condition=t + 1,
                lateral_deviation_mm="" if miss else ld, target_error_mm="" if miss else te))

n_seq = sum(a.shape[1] for a in TEs.values())
checks.internal("119 sequences x 330 reaches", len(rows) == 119 * 330 and n_seq == 119, f"{n_seq} sequences")
checks.published("participants, experiment 1 (VDCF + LIPF)", TEs["VDCF"].shape[1] + TEs["LIPF"].shape[1], 30, 0, "Methods, Participants (30 in experiment 1)")
checks.published("participants, experiment 2 (PSPF + CPVF, one CPVF participant excluded)", TEs["PSPF"].shape[1] + TEs["CPVF"].shape[1], 29, 0, "Methods, Participants (30, one excluded)")
checks.published("participants in the CPVF group", TEs["CPVF"].shape[1], 14, 0, "Methods, Participants")
checks.published("participants in each experiment 3 condition", TEs["LIPFNULL"].shape[1], 30, 0, "Methods, Participants (30)")
sd = lambda x: float(np.nanstd(x, ddof=1))
a1, d1 = TEs["LIPF"][5], TEs["LIPF"][160]
checks.published("LIPF first adaptation reach, |TE| mean (mm)", abs(np.nanmean(a1)), 112.6, 0.1, "Results, experiment 1 (112.6 +/- 38.0 mm)")
checks.published("LIPF first adaptation reach, TE SD (mm)", sd(a1), 38.0, 0.1, "Results, experiment 1")
checks.published("LIPF first de-adaptation reach, TE mean (mm)", np.nanmean(d1), 44.3, 0.1, "Results, experiment 1 (44.3 +/- 27.7 mm)")
checks.published("LIPF first de-adaptation reach, TE SD (mm)", sd(d1), 27.7, 0.1, "Results, experiment 1")
a2 = TEs["CPVF"][5]
checks.published("CPVF first adaptation reach, |TE| mean (mm)", abs(np.nanmean(a2)), 73.2, 0.1, "Results, experiment 2 (73.2 +/- 50.0 mm)")
checks.published("CPVF first adaptation reach, TE SD (mm)", sd(a2), 50.0, 0.2, "Results, experiment 2")
checks.published("experiment 3 first de-adaptation reach, PEC TE mean (mm)", np.nanmean(TEs["LIPFPEC"][160]), 8.6, 0.1, "Results, experiment 3 (8.6 +/- 1.9 mm)")
checks.published("experiment 3 first de-adaptation reach, PEC TE SD (mm)", sd(TEs["LIPFPEC"][160]), 1.9, 0.1, "Results, experiment 3")
checks.published("experiment 3 first de-adaptation reach, Null TE mean (mm)", np.nanmean(TEs["LIPFNULL"][160]), 39.9, 0.1, "Results, experiment 3 (39.9 +/- 26.5 mm)")
checks.published("experiment 3 first de-adaptation reach, Null TE SD (mm)", sd(TEs["LIPFNULL"][160]), 26.5, 0.1, "Results, experiment 3")
checks.internal("some reaches are empty and flagged", 0 < sum(r["excluded_by_authors"] for r in rows) < 200)

paper = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Kawato (ATR) / Ikegami (NICT)", data_url="https://doi.org/10.5061/dryad.5x69p8d2f",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="the authors' matrices (reaches x participants, in m) were turned into one table with one row per reach, in mm; "
            "the phase of each reach was derived from the paper's trial structure; participant codes replaced.",
    unit_note="reach of one participant",
    how=[
        "One row is one reach of one participant. Each sequence holds the last 20 baseline reaches (null field) and then the "
        "310-reach adaptation session: 5 null reaches, 155 reaches in the force field named in condition (VDCF velocity-dependent "
        "curl, LIPF position-dependent, PSPF position- and speed-dependent, CPVF combined position and velocity) and 150 "
        "de-adaptation reaches (null field; in LIPF-PEC an error clamp that pulls the hand back toward the straight line during "
        "the second half of the reach, trial_type clamp). Reaches are 150 mm forward to a 15 mm target.",
        "lateral_deviation_mm is the authors' LD (deviation from the straight start-target line at the reach midpoint) and "
        "target_error_mm their TE (deviation of the final hand position), both converted from m to mm and kept in the authors' "
        "sign: the position field pushes toward -x, so TE is negative on its first field reaches. The deposit's trial order is "
        "that of the matrices; the structure of the 310 reaches (5 + 155 + 150) is from the paper and is confirmed by the "
        "values (the largest errors on reach 6 and 161 of each session).",
        "Reaches the authors excluded (hand moved less than 75 mm) are empty in the deposit and are rows with empty values and "
        "excluded_by_authors True. The baseline in the deposit is only its last 20 reaches (the paper's baseline is 50). "
        "pert_value is the force constant of the paper (VDCF 14 N s/m, LIPF 60 N/m); it is empty for PSPF and CPVF, which have no "
        "single constant. Participants in experiment 3 did both LIPF-NULL and LIPF-PEC (counterbalanced), and 15 of them also "
        "did the LIPF group of experiment 1; the deposit does not link the columns of the files, so every file's columns are "
        "coded as different people here.",
    ],
    differences=["CPVF first de-adaptation reach: the paper reports 26.0 +/- 22.8 mm; the rows give a mean of about 26.6 mm over the "
                 "participants with a value (one is empty), so that number is not used as a check.",
                 "The deposit holds one CPVF participant fewer than the 15 recruited (14), as the paper says."],
)
stdlib.release(ID, "force_field_reaching", rows, meta, checks)
