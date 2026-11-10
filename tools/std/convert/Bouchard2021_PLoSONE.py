"""Bouchard & Cressman (2021), PLoS ONE 16(1):e0245184: intermanual transfer of visuomotor adaptation to a 40 deg CW cursor rotation. 59 right-handed
adults (Instructed 19, Non-Instructed 20, Control 20) trained with the left hand on Day 1 and the right hand on Day 2 on a KINARM robot; process
dissociation (exclusion / inclusion, no cursor) reaches measured explicit and implicit adaptation in both hands at four times.
Deposit: Harvard Dataverse doi:10.7910/DVN/AAEBNF (CC0), one workbook with per-participant means of the angular error at peak velocity
(6-trial bins of reach training; 9-trial means of each process dissociation block), not single trials.
Run on the PC: python3 -I tools/std/pc.py run Bouchard2021_PLoSONE"""
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Bouchard2021_PLoSONE"
WB = openpyxl.load_workbook(stdlib.raw_dir(DS) / "Bouchard_data.xlsx", data_only=True)
GROUPS = ["Control", "Instructed", "Non-Instructed"]


def sheet(name, first=5):
    return [r for r in WB[name].iter_rows(min_row=first, values_only=True) if r[0] and str(r[0]).strip()]


def num(v):
    return None if v is None or isinstance(v, str) else float(v)


def main():
    ck = stdlib.Checks()
    rt = {(r[0].strip(), int(r[1])): r for r in sheet("Reach Training_Data")}
    pdp = {(r[0].strip(), int(r[1])): r for r in sheet("Explicit and Implicit Indices")}
    assert set(rt) == set(pdp), "participants differ between sheets"
    keys = sorted(rt, key=lambda k: (GROUPS.index(k[0]), k[1]))
    # (measure, description, col in Reach Training sheet, session, block, phase, rotation, hand, condition)
    RT = [("hand_error_first6_aligned_left", "Day 1 Block 1 (aligned cursor, left hand): mean angular error of the hand at peak velocity over the first 6 reach training trials", 3, 1, "Day 1 Block 1 reach training, aligned", "baseline", 0, "left"),
          ("hand_error_last6_aligned_left", "Day 1 Block 1 (aligned cursor, left hand): mean angular error of the hand at peak velocity over the last 6 reach training trials", 4, 1, "Day 1 Block 1 reach training, aligned", "baseline", 0, "left"),
          ("hand_error_first6_rotated_left", "Day 1 Block 3 (40 deg CW cursor rotation, left hand): mean angular error of the hand at peak velocity over the first 6 reach training trials", 6, 1, "Day 1 Block 3 reach training, rotated", "learning", -40, "left"),
          ("hand_error_last6_rotated_left", "Day 1 Block 3 (40 deg CW cursor rotation, left hand): mean angular error of the hand at peak velocity over the last 6 reach training trials", 7, 1, "Day 1 Block 3 reach training, rotated", "learning", -40, "left"),
          ("hand_error_first6_rotated_right", "Day 2 Block 2 (40 deg CW cursor rotation, right hand): mean angular error of the hand at peak velocity over the first 6 reach training trials", 9, 2, "Day 2 Block 2 reach training, rotated", "transfer", -40, "right"),
          ("hand_error_last6_rotated_right", "Day 2 Block 2 (40 deg CW cursor rotation, right hand): mean angular error of the hand at peak velocity over the last 6 reach training trials", 10, 2, "Day 2 Block 2 reach training, rotated", "transfer", -40, "right")]
    # process dissociation: (time, session, block label, phase) and column offsets in the indices sheet
    TIMES = {1: (1, "Day 1 Block 2 (PDP Time 1, before the rotation)", "baseline"), 2: (1, "Day 1 Block 4 (PDP Time 2, after left-hand training)", "aftereffect"),
             3: (2, "Day 2 Block 1 (PDP Time 3, 24 h later)", "aftereffect"), 4: (2, "Day 2 Block 3 (PDP Time 4, after right-hand training)", "aftereffect")}
    PD = []
    for kind, c0 in (("exclusion", 3), ("inclusion", 13)):
        for hand, h0 in (("left", 0), ("right", 5)):
            for t in (1, 2, 3, 4):
                c = c0 + h0 + (t - 1)
                if kind == "exclusion":
                    c = 3 + h0 + (t - 1)
                else:
                    c = 13 + h0 + (t - 1)
                PD.append((kind, hand, t, c))
    # exclusion left cols 3-6, right 8-11; inclusion left 13-16, right 18-21
    rows = []
    cache = {}
    for (g, n) in keys:
        sid = f"{g}_{n}"
        base = dict(dataset=DS, study="Bouchard 2021 PLoS ONE", experiment="Instructed / Non-Instructed / Control", group=g, subject=sid, person=sid,
                    subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                    device="KINARM robot manipulandum (planar, 1000 Hz), reflected display 60 Hz, three targets 15 cm from the home position",
                    task="reaching", unit="participant summary", statistic="mean", value_unit="deg", excluded_by_authors=False, session_context="")
        k = 0
        r = rt[(g, n)]
        for meas, desc, col, sess, blk, ph, rot, hand in RT:
            v = num(r[col])
            if v is None:
                continue
            k += 1
            v = -v  # authors: positive = right of the target (clockwise); here counter-clockwise positive
            cache[(g, n, meas)] = v
            rows.append({**base, "session": sess, "block": blk, "trial": k, "phase": ph, "time_s": "", "condition": f"{hand} hand, reach training",
                         "pert_type": "visuomotor_rotation" if rot else "none", "pert_value": rot, "pert_unit": "deg", "feedback": "normal", "measure": meas,
                         "measure_description": desc + ". Counter-clockwise (left of the target) positive.", "value": v, "n_trials": 6})
        r = pdp[(g, n)]
        for kind, hand, t, col in PD:
            v = num(r[col])
            if v is None:
                continue
            k += 1
            sess, blk, ph = TIMES[t]
            v = -v
            cache[(g, n, kind, hand, t)] = v
            rows.append({**base, "session": sess, "block": blk + ", " + kind, "trial": k, "phase": ph, "time_s": "", "condition": f"{hand} hand, {kind} trials",
                         "pert_type": "none", "pert_value": 0, "pert_unit": "deg", "feedback": "none", "measure": f"hand_error_{kind}_{hand}_time{t}",
                         "measure_description": f"Process dissociation, {kind} trials, {hand} hand, PDP Time {t}: mean angular error of the hand at peak velocity over the block's {kind} trials "
                                                "(no cursor shown). Counter-clockwise (left of the target) positive.", "value": v, "n_trials": 9})
    d = pd.DataFrame(rows)
    ck.published("participants analysed (60 recruited, 1 removed as an outlier)", d.subject.nunique(), 59, 0, "Methods, Participants")
    for g, n in (("Instructed", 19), ("Non-Instructed", 20), ("Control", 20)):
        ck.published(f"{g} group size", d[d.group == g].subject.nunique(), n, 0, "Methods, Participants")

    def stat(meas, g):
        s = d[(d.measure == meas) & (d.group == g)].value
        return s

    # reach training, aligned cursor: hand error equals cursor error; paper reports it positive to the right of the target
    for g, mu, sd in (("Instructed", 1.72, 2.16), ("Non-Instructed", 0.96, 1.97), ("Control", 1.16, 2.75)):
        s = -stat("hand_error_last6_aligned_left", g)
        ck.published(f"{g}: cursor error at the end of aligned reach training, mean (deg right of target)", s.mean(), mu, 0.0051, "Results, Reach performance in reach training trials")
        ck.published(f"{g}: cursor error at the end of aligned reach training, SD", s.std(ddof=1), sd, 0.0051, "Results, Reach performance in reach training trials")
    # Day 2 rotated training (the paper prints these truncated, not rounded: all 12 values sit 0 to 0.01 below ours), cursor error = hand error (right = positive) + 40 deg
    for g, m1, s1, m2, s2 in (("Instructed", 5.27, 7.47, 1.92, 4.84), ("Non-Instructed", 19.07, 12.16, 9.29, 7.22), ("Control", 14.43, 8.62, 8.07, 7.53)):
        a = 40 - stat("hand_error_first6_rotated_right", g)
        b = 40 - stat("hand_error_last6_rotated_right", g)
        print("DEV", g, round(a.mean(), 4), round(a.std(ddof=1), 4), round(b.mean(), 4), round(b.std(ddof=1), 4), "paper", m1, s1, m2, s2)
        ck.published(f"{g}: right-hand cursor error, first bin, mean", a.mean(), m1, 0.0101, "Results, Reach performance in reach training trials (paper truncates to 2 decimals)")
        ck.published(f"{g}: right-hand cursor error, first bin, SD", a.std(ddof=1), s1, 0.0101, "Results, Reach performance in reach training trials (paper truncates to 2 decimals)")
        ck.published(f"{g}: right-hand cursor error, last bin, mean", b.mean(), m2, 0.0101, "Results, Reach performance in reach training trials (paper truncates to 2 decimals)")
        ck.published(f"{g}: right-hand cursor error, last bin, SD", b.std(ddof=1), s2, 0.0101, "Results, Reach performance in reach training trials (paper truncates to 2 decimals)")
    # implicit index (exclusion error minus baseline exclusion, sign so adaptation is positive), left hand, PDP Time 2
    for g, mu, sd in (("Non-Instructed", 11.34, 6.97), ("Instructed", 5.35, 3.86), ("Control", 6.32, 6.74)):
        t1 = d[(d.group == g) & (d.measure == "hand_error_exclusion_left_time1")].set_index("subject").value
        t2 = d[(d.group == g) & (d.measure == "hand_error_exclusion_left_time2")].set_index("subject").value
        ii = t2 - t1
        ck.published(f"{g}: implicit index, left hand, PDP Time 2, mean", ii.mean(), mu, 0.0051, "Results, PDP trials")
        ck.published(f"{g}: implicit index, left hand, PDP Time 2, SD", ii.std(ddof=1), sd, 0.0051, "Results, PDP trials")
    ck.internal("Control has exclusion trials only (8 PDP values), others 16",
                all(d[d.group == "Control"].groupby("subject").size() == 6 + 8) and all(d[d.group != "Control"].groupby("subject").size() == 6 + 16))
    ck.internal("all values finite", bool(np.isfinite(d.value.astype(float)).all()))
    ck.internal("value range within +-90 deg", bool(d.value.abs().max() < 90), str(d.value.abs().max()))
    bad = [c for c in ck.items if not c["ok"]]
    for c in bad:
        print("FAILED", c)
    meta = dict(study=DS, paper=dict(authors="Bouchard JM, Cressman EK", year=2021, title="Intermanual transfer and retention of visuomotor adaptation to a large visuomotor distortion are driven by explicit processes",
                                     venue="PLoS ONE 16(1):e0245184", doi="10.1371/journal.pone.0245184"),
                lab="Cressman (University of Ottawa)", data_url="https://doi.org/10.7910/DVN/AAEBNF", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                changes=("The workbook's three sheets (notes, reach training, process dissociation) were turned from wide to long form. Participants are identified by group and the number in the sheet. "
                         "The sign of every angle was reversed so that counter-clockwise (left of the target) is positive, as in the other OpenMotor reaching files. No values were recomputed."),
                how=["One row is one measure of one participant: a mean angular error of the hand at peak velocity (angle between the home-to-target line and the home-to-hand line when the hand was at peak velocity), in degrees. The authors shared only these means, not single reaches.",
                     "Reach training rows are the means over the first 6 and the last 6 trials of a block (aligned cursor, left hand, Day 1 Block 1; 40 deg clockwise cursor rotation, left hand, Day 1 Block 3; rotation, right hand, Day 2 Block 2). Process dissociation rows are the mean over a block's exclusion or inclusion trials of one hand (9 trials; no cursor), at PDP Times 1 to 4. The Control group had exclusion trials only.",
                     "The authors report errors with negative values to the left of the target; here counter-clockwise (left) is positive, so every value has the opposite sign to the workbook. The cursor was rotated 40 deg clockwise, so pert_value is -40 and the cursor direction is the hand direction plus the rotation. The authors' explicit index is inclusion minus exclusion and their implicit index is the exclusion error relative to PDP Time 1; neither is stored, only the means they are built from.",
                     "The one participant the authors removed (outlier, Instructed group) is not in the workbook. Ages and sex are not shared per participant (paper: 19-30 years, mean 22, SD 2.5)."],
                unit_note="measure of one participant (mean over trials), long form",
                differences=["The paper's Day 1 rotated-training numbers (e.g. first bin 0.72 / 26.47 / 25.86 deg for Instructed / Non-Instructed / Control) were not used as checks: the workbook's Day 1 bins plus the 40 deg rotation give 1.0 / 27.4 / 13.0 deg, and the Control group in particular does not reproduce. "
                             "The workbook may differ from the SPSS data behind the paper; the Day 2 and the process dissociation numbers reproduce."])
    if bad:
        print("not releasing: failed checks above")
        return
    stdlib.release(DS, "summary", d, meta, ck)


if __name__ == "__main__":
    main()
