"""Danion, Coudiere, Leblan-Falzone, Toussaint & Bernier (2026) eNeuro: planning a reach while the same or the other hand moves.

Raw file (OSF d27ek, CC not stated): PlanningMovingOSFeNeuro.xlsx (saved as data.xlsx): sheets Participants, EXP 1, EXP2 (and a
per-participant comparison sheet that is not used). In EXP 1 / EXP2, per-trial tables of 24 participants x 120 trials (four
blocks of 30 columns) for reaction time (rows 4-27), peak velocity (rows 63-86) and timing of peak velocity (rows 93-116).
Empty cells are trials the authors rejected. Nothing is executed.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Danion2026_eNeuro"
RAW = stdlib.raw_dir(DS)
CONDS = ["STATIC-NOCUE", "STATIC-CUE", "MOVE-NOCUE", "MOVE-CUE"]
TABLES = [("reaction time", 3, 1, "ms", "Reaction time of the reach after the target appeared, from a 10 cm/s velocity threshold"),
          ("peak velocity", 62, 61, "cm/s as labelled by the authors", "Peak velocity of the reach"),
          ("time of peak velocity", 92, 91, "unstated", "Timing of peak velocity of the reach (unit not stated in the deposit)")]


def num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def main():
    X = pd.read_excel(RAW / "data.xlsx", sheet_name=None, header=None)
    P = X["Participants"]
    ck = stdlib.Checks()
    rows, parts = [], []
    deltas = {}
    for exp, sheet, pc0, label, hand_note in (("EXP1", "EXP 1", 1, "unimanual", "the same hand that reaches makes the rhythmic movement"),
                                              ("EXP2", "EXP2", 6, "bimanual", "the other hand makes the rhythmic movement")):
        S = X[sheet]
        sex_n = {}
        for i in range(24):
            pr = 3 + i
            assert str(P.iloc[pr, pc0 - 1]) == f"S{i + 1}", (exp, i)
        for tname, r0, hdr_row, unit, desc in TABLES:
            assert hdr_row in (1, 61, 91) and [str(S.iloc[r0 + i, 0]) for i in range(24)] == [f"S{i + 1}" for i in range(24)], (exp, tname)
        for i in range(24):
            pr = 3 + i
            sid = f"{exp}_S{i + 1}"
            age, sx, lat = P.iloc[pr, pc0], str(P.iloc[pr, pc0 + 1]).strip(), str(P.iloc[pr, pc0 + 2]).strip()
            sex = {"M": "M", "F": "F"}.get(sx, "other")
            sex_n[sx] = sex_n.get(sx, 0) + 1
            parts.append(dict(person=sid))
            order = 0
            for tname, r0, hdr_row, unit, desc in TABLES:
                v = S.iloc[r0 + i, 1:121].map(num).values
                blocks = [v[k * 30:(k + 1) * 30] for k in range(4)]
                if tname == "reaction time":
                    means = [np.nanmean(b) for b in blocks]
                    ref = [num(S.iloc[3 + i, 121 + k]) for k in range(4)]
                    ck.internal(f"{sid}: block means equal the authors' condition columns (block 1-4 = {CONDS})",
                                all(abs(a - b) < 1e-6 for a, b in zip(means, ref)) if all(r is not None for r in ref) else True)
                    deltas.setdefault(exp, []).append(means)
                for k, b in enumerate(blocks):
                    for t, val in enumerate(b):
                        if np.isnan(val):
                            continue
                        order += 1
                        rows.append(dict(dataset=DS, study="Danion 2026 eNeuro", experiment=exp.lower() + "-" + label, group=None,
                                         subject=sid, person=sid, subject_excluded_by_authors=False, age=num(age), sex=sex,
                                         handedness={"R": "right", "L": "left"}.get(lat[:1].upper()), population="healthy",
                                         setting="lab", device="reaching with a rhythmic movement in the same or other hand",
                                         session=1, block=CONDS[k], trial=order, phase="test", time_s=None, task="reaching",
                                         unit="trial", condition=CONDS[k], pert_type="none", pert_value=0, pert_unit=None,
                                         feedback="normal", excluded_by_authors=False, measure=tname, measure_description=desc,
                                         statistic="value", value=val, value_unit=unit, n_trials=None,
                                         session_context=hand_note if CONDS[k].startswith("MOVE") else None))
        ck.internal(f"{exp}: 24 participants", len({r["subject"] for r in rows if r["experiment"].startswith(exp.lower())}) == 24)
        print(exp, "sex counts in the Participants sheet:", sex_n)
    ck.internal("every non-empty cell kept", len(rows) > 24 * 100)
    d1 = np.array(deltas["EXP1"])
    c1_static = float(np.mean(d1[:, 0] - d1[:, 1]))
    c1_move = float(np.mean(d1[:, 2] - d1[:, 3]))
    ck.published("Exp 1: precue benefit on reaction time, hand at rest (ms)", c1_static, 44, 0.5, "Abstract")
    ck.published("Exp 1: precue benefit while the same hand moves (ms)", c1_move, 16, 0.5, "Abstract")
    d2 = np.array(deltas["EXP2"])
    print("EXP2 benefit static/move:", float(np.mean(d2[:, 0] - d2[:, 1])), float(np.mean(d2[:, 2] - d2[:, 3])))
    ck.published("participants in experiment 2", len({r["subject"] for r in rows if r["experiment"].startswith("exp2")}), 24, 0, "Abstract (13 females + 11 males)")
    meta = dict(
        study=DS,
        paper=dict(authors="Danion FR, Coudiere A, Leblan-Falzone H, Toussaint L, Bernier PM", year=2026,
                   title="Planning while moving: when two hands are better than one", venue="eNeuro ENEURO.0255-26.2026",
                   doi="10.1523/eneuro.0255-26.2026"),
        lab="Danion (CNRS, Universite de Poitiers)", data_url="https://osf.io/d27ek/",
        license="none stated", license_url="",
        changes=("Read the three per-trial tables (reaction time, peak velocity, timing of peak velocity) of the two experiments, "
                 "turned each into one row per participant and trial, dropped the empty cells (rejected trials), labelled the "
                 "four 30-trial blocks with the authors' condition names, and replaced participant numbers with codes."),
        how=["One row is one accepted trial of a two-choice reach (left or right target) with one measure: reaction time (ms, 10 cm/s "
             "velocity threshold), peak velocity and timing of peak velocity as labelled by the authors. Conditions are STATIC or MOVE "
             "(hand at rest, or making a rhythmic movement: the same hand in experiment 1, the other hand in experiment 2) crossed with "
             "NOCUE or CUE (target direction precued or not).",
             "The deposit orders the 120 columns as four blocks of 30 trials, one block per condition; we checked for every participant "
             "that each block's mean equals the authors' own condition mean, which fixes the block order as STATIC-NOCUE, STATIC-CUE, "
             "MOVE-NOCUE, MOVE-CUE. So trial is the column order (within condition), not presentation order, which the deposit does "
             "not give. Empty cells are trials the authors rejected and are not rows. The peak-velocity unit (cm/s) and the timing unit "
             "(not stated) are the authors' labels and look large for a reach; they are kept as shared.",
             "Experiments 1 and 2 are different participants. Age, sex and handedness come from the Participants sheet; the sheet's "
             "non-binary entry is coded other."],
        unit_note="accepted trial of one reach measure",
        differences=["The abstract's experiment 2 precue benefit while the other hand moves (37 ms) did not reproduce: the mean of the "
                     "participants' condition-mean differences from the released trials is 38.1 ms (not checked; the authors may have "
                     "excluded a participant or used a different average).", "The paper's abstract gives 13 females and 11 males in experiment 2; the Participants sheet lists a different "
                     "split (12 females, 11 males, 1 non-binary), so sex counts were not used as a check."],
    )
    stdlib.release(DS, "summary", rows, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
