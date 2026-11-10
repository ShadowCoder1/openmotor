"""Ebner-Karestinos, Thonnard & Bleyenheuft (2016), PLoS ONE 11(11):e0165549: 12 healthy adults (6 women, 6 men) held a grip-lift manipulandum in the
right hand and stepped down a 17 cm step (8 trials each); grip force (GF), load force (LF), vertical ground reaction force (vGRF) and the vertical
displacement of hand and sacrum were taken at six events. Dryad doi 10.5061/dryad.kf4tr, copy at Zenodo record 4959013 (CC0).
Converted (long 'summary' form): per-participant means (Kinem-Dynam_var(table1), Temporal_var(table2)) and the trial-level tables of
Kinem-Dynam_var(table3) (hand and sacrum displacement, vGRF normalized to the group's mean weight, safety margin).
Run on the PC: python3 -I tools/std/pc.py run EbnerKarestinos2016_PLoSONE"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "EbnerKarestinos2016_PLoSONE"
Z = zipfile.ZipFile(stdlib.raw_dir(DS) / "data2016.zip")
EV7 = ["t0", "LFmin", "GFmin", "vGRFfc", "LFmax", "vGRFmax", "GFmax"]
EVD = {"t0": "the start of the step-down (vGRF equal to body weight)", "LFmin": "the minimum load force", "GFmin": "the minimum grip force",
       "vGRFfc": "foot contact (vGRF at the start of the first double support)", "LFmax": "the maximum load force", "vGRFmax": "the maximum vGRF after foot contact",
       "GFmax": "the maximum grip force"}
DELAYS = ["LFmin_to_vGRFfc", "GFmin_to_vGRFfc", "LFmin_to_LFmax", "GFmin_to_LFmax", "LFmax_to_vGRFmax", "LFmax_to_GFmax"]


def book(n):
    return openpyxl.load_workbook(io.BytesIO(Z.read(f"DATA FOR PLOS_One/{n}")), data_only=True)


def num(v):
    return None if v is None or isinstance(v, str) else float(v)


def main():
    ck = stdlib.Checks()
    t1, t2, t3 = book("Kinem-Dynam_var(table1).xlsx"), book("Temporal_var(table2).xlsx"), book("Kinem-Dynam_var(table3).xlsx")
    subj_ws = t3["Subjects"]
    info = {}
    for r in subj_ws.iter_rows(min_row=3, max_row=14, values_only=True):
        info[int(str(r[0])[1:])] = dict(sex="F" if r[1] == "female" else "M", age=float(r[2]), height=float(r[3]), weight=float(r[4]), wn=float(r[5]))
    assert len(info) == 12
    means = []   # (sid, measure, desc, value, unit)
    kin, dyn = t1["Kinematic "] if "Kinematic " in t1.sheetnames else t1.worksheets[0], t1["Dynamic"]
    blocks = [(kin, 5, "hand_vertical_displacement", "vertical displacement of the hand from t0", "cm"), (kin, 24, "sacrum_vertical_displacement", "vertical displacement of the sacrum from t0", "cm"),
              (dyn, 5, "vGRF", "vertical ground reaction force", "N"), (dyn, 24, "load_force", "load force", "N"), (dyn, 43, "grip_force", "grip force", "N"),
              (dyn, 62, "GF_LF_ratio", "grip force to load force ratio", "ratio"), (dyn, 81, "safety_margin", "safety margin, (GF - GF at slip) / GF at slip", "ratio")]
    for ws, r0, nm, desc, unit in blocks:
        for i in range(12):
            r = r0 + i
            sid = int(ws.cell(row=r, column=1).value)
            assert sid == i + 1
            for e, ev in enumerate(EV7):
                v = num(ws.cell(row=r, column=2 + e).value)
                if v is not None:
                    means.append((sid, f"{nm}_at_{ev}", f"Participant's mean {desc} at {EVD[ev]}", v, unit))
    ws = t2.worksheets[0]
    for i in range(12):
        r = 5 + i
        sid = int(ws.cell(row=r, column=1).value)
        assert sid == i + 1
        for k, dn in enumerate(DELAYS):
            a, b = dn.split("_to_")
            means.append((sid, f"delay_{dn}", f"Participant's mean delay from {EVD[a]} to {EVD[b]}", num(ws.cell(row=r, column=3 + k).value), "ms"))
    trials = []  # (sid, family, k, measure, desc, value, unit)
    cnt = {}

    def add(sid, fam, ev, v, unit, desc):
        if v is None:
            return
        cnt.setdefault((sid, fam, ev), 0)
        cnt[(sid, fam, ev)] += 1
        trials.append((sid, fam, cnt[(sid, fam, ev)], f"{fam}_at_{ev}", desc, v, unit))
    EV6 = EV7[1:]
    ws = t3["Kinematic"]
    for r in ws.iter_rows(min_row=4, values_only=True):
        for off in (0, 11):
            mk = r[off + 2]
            if mk in ("Hand", "Sacrum"):
                fam = "hand_vertical_displacement" if mk == "Hand" else "sacrum_vertical_displacement"
                for e, ev in enumerate(EV6):
                    add(int(r[off + 1]), fam, ev, num(r[off + 3 + e]), "cm", f"Vertical displacement of the {mk.lower()} from t0 at {EVD[ev]}, one trial")
    # trial counters must run per (sid, family) not per event: recompute as ordinal per event, equal across events
    ws = t3["Dynamic"]
    for ri, r in enumerate(ws.iter_rows(min_row=1, values_only=True), 1):
        for off in (0, 10):
            gender, sid = r[off], r[off + 1]
            if not (isinstance(gender, str) and gender.lower() in ("female", "male") and isinstance(sid, (int, float))):
                continue
            if 4 <= ri <= 48:
                fam, unit, d = "vGRF_normalized", "N", "Vertical ground reaction force normalized to the mean weight of the whole group"
            elif ri >= 94:
                fam, unit, d = "safety_margin", "ratio", "Safety margin, (GF - GF at slip) / GF at slip"
            else:
                continue
            for e, ev in enumerate(EV6):
                add(int(sid), fam, ev, num(r[off + 2 + e]), unit, f"{d} at {EVD[ev]}, one trial")
    ids = sorted(info)
    rows = []
    seq = {}

    def base(sid):
        i = info[sid]
        return dict(dataset=DS, study="Ebner-Karestinos 2016 PLoS ONE", experiment="", group="", subject=f"S{sid:02d}", person=f"S{sid:02d}", subject_excluded_by_authors=False,
                    age=i["age"], sex=i["sex"], handedness="right", population="healthy", setting="lab",
                    device="Grip-lift manipulandum (right hand), force platform (1000 Hz), Elite 8-camera motion capture (200 Hz) on hand and sacrum",
                    session=1, block="step-down task", phase="test", time_s="", task="grasping and lifting", condition="stepping down a 17 cm step holding the object",
                    pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False, statistic="", n_trials="", session_context="")
    for sid in ids:
        for s2, nm, desc, v, unit in means:
            if s2 != sid:
                continue
            seq[sid] = seq.get(sid, 0) + 1
            rows.append({**base(sid), "trial": seq[sid], "unit": "participant summary", "measure": nm, "measure_description": desc, "statistic": "mean", "value": v, "value_unit": unit})
        for s2, fam, k, nm, desc, v, unit in trials:
            if s2 != sid:
                continue
            seq[sid] = seq.get(sid, 0) + 1
            rows.append({**base(sid), "block": f"step-down task, {fam} table, trial {k}", "trial": seq[sid], "unit": "trial", "measure": nm, "measure_description": desc, "statistic": "value", "value": v, "value_unit": unit})
    d = pd.DataFrame(rows)
    sm = d[d.unit == "participant summary"]
    tr = d[d.unit == "trial"]
    part = pd.DataFrame([dict(person=f"S{s:02d}", height_cm=info[s]["height"], weight_kg=info[s]["weight"]) for s in ids])
    pcols = [("height_cm", "number", "cm", "Height as listed in the deposit's Subjects sheet."), ("weight_kg", "number", "kg", "Body mass as listed in the deposit.")]

    def mean_sd(meas):
        s = sm[sm.measure == meas].value
        return s.mean(), s.std(ddof=1)
    ck.published("participants (12)", d.subject.nunique(), 12, 0, "Methods, Participants")
    ck.published("women (6)", sum(1 for i in info.values() if i["sex"] == "F"), 6, 0, "Methods, Participants")
    a = np.array([info[s]["age"] for s in ids]); h = np.array([info[s]["height"] for s in ids]); w = np.array([info[s]["weight"] for s in ids])
    for nm, arr, mu, sd, tol_m, tol_s in [("age (years)", a, 29.2, 4.0, 0.05, 0.05), ("height (cm)", h, 169.4, 7.5, 0.05, 0.05)]:
        ck.published(f"mean {nm}", arr.mean(), mu, tol_m, "Methods, Participants")
        ck.published(f"SD {nm}", arr.std(ddof=1), sd, tol_s, "Methods, Participants")
    ck.published("mean body weight of the group in N (646.9)", np.mean([info[s]["wn"] for s in ids]), 646.9, 0.05, "Results, Dynamic variables")
    T1 = [("hand_vertical_displacement_at_LFmin", 4.9, 2.3), ("hand_vertical_displacement_at_GFmax", 13.5, 2.0), ("sacrum_vertical_displacement_at_LFmin", 6.8, 2.9),
          ("sacrum_vertical_displacement_at_GFmax", 16.7, 1.6), ("vGRF_at_t0", 631.8, 117.1), ("vGRF_at_vGRFfc", 560.1, 152.3), ("vGRF_at_vGRFmax", 982.7, 153.5),
          ("load_force_at_t0", 2.8, 0.4), ("load_force_at_LFmax", 3.3, 0.5), ("grip_force_at_t0", 8.1, 4.9), ("grip_force_at_GFmax", 8.4, 4.1),
          ("GF_LF_ratio_at_t0", 3.0, 1.8), ("GF_LF_ratio_at_LFmax", 2.6, 1.3), ("safety_margin_at_LFmin", 1.7, 1.5), ("safety_margin_at_GFmax", 2.1, 1.5)]
    for meas, mu, sd in T1:
        m_, s_ = mean_sd(meas)
        ck.published(f"{meas} mean", m_, mu, 0.051, "Table 1")
        ck.published(f"{meas} SD", s_, sd, 0.051, "Table 1")
    T2 = [("LFmin_to_vGRFfc", 101.3, 63.8), ("GFmin_to_vGRFfc", 92.4, 63.2), ("LFmin_to_LFmax", 226.5, 51.4), ("GFmin_to_LFmax", 217.6, 57.5), ("LFmax_to_vGRFmax", 17.3, 21.2), ("LFmax_to_GFmax", 73.4, 42.2)]
    for dn, mu, sd in T2:
        m_, s_ = mean_sd(f"delay_{dn}")
        ck.published(f"delay {dn} mean (ms)", m_, mu, 0.051, "Table 2")
        ck.published(f"delay {dn} SD (ms)", s_, sd, 0.051, "Table 2")
    ck.internal("six women and six men present in trial-level normalized vGRF", tr[tr.measure == "vGRF_normalized_at_LFmin"].subject.nunique() == 12)
    ck.internal("trial-level rows exist for kinematics of 8 participants", tr[tr.measure == "hand_vertical_displacement_at_LFmin"].subject.nunique() == 8,
                str(tr[tr.measure == "hand_vertical_displacement_at_LFmin"].subject.nunique()))
    ck.internal("participant means exist for all 12 participants (forces)", sm[sm.measure == "load_force_at_t0"].subject.nunique() == 12)
    # Table 3 (sex groups) from the trial-level vGRF normalized rows: female LFmin 540.0 (86.2), male 606.7 (75.0)
    vg = tr[tr.measure == "vGRF_normalized_at_LFmin"].copy()
    vg["sex"] = vg.subject.map(lambda s: info[int(s[1:])]["sex"])
    for sx, mu, sd in (("F", 540.0, 86.2), ("M", 606.7, 75.0)):
        s = vg[vg.sex == sx].value
        ck.published(f"{sx} vGRF* at LFmin mean (trial rows)", s.mean(), mu, 0.051, "Table 3")
        ck.published(f"{sx} vGRF* at LFmin SD (trial rows)", s.std(ddof=1), sd, 0.051, "Table 3")
    meta = dict(study=DS, paper=dict(authors="Ebner-Karestinos D, Thonnard JL, Bleyenheuft Y", year=2016, title="Precision Grip Control while Walking Down a Stair Step",
                venue="PLoS ONE 11(11):e0165549", doi="10.1371/journal.pone.0165549"),
        lab="Bleyenheuft (UCLouvain)", data_url="https://doi.org/10.5061/dryad.kf4tr (copy: https://zenodo.org/records/4959013)", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("The deposit's tables were turned from wide to long form: per-participant means from Kinem-Dynam_var(table1) and Temporal_var(table2) (t0 and six events; delays), and the trial-level tables of "
                 "Kinem-Dynam_var(table3) (hand and sacrum vertical displacement; vGRF normalized to the group's mean weight; safety margin). Measure names became snake_case. Participant S-numbers became codes; "
                 "age and sex come from the Subjects sheet. The deposit's MEAN, SD and MEDIAN rows, the per-participant LF, GF and GF/LF means repeated in table3 (identical to table1) and the "
                 "example time series of Data_Graphs(Fig1) and (Fig2) are not included."),
        how=["Two kinds of rows. unit 'participant summary': one measure of one participant, the mean over his or her step-down trials (table1, table2). unit 'trial': one measure of one trial from table3; block names the table and the trial number within it.",
             "Events: t0 (vGRF equal to body weight, start of the step-down), LFmin, GFmin (minimum load and grip force), vGRFfc (foot contact), LFmax, vGRFmax, GFmax. Displacements are in cm downward from t0, forces in N, delays in ms. GF/LF and safety margin are unitless (the deposit labels them N).",
             "The trial-level sheets are not linked across tables (different numbers of trials per participant, no trial identifier), so trial k of one measure family need not be trial k of another."],
        unit_note="measure of one participant (mean) or one trial, long form",
        differences=["The paper gives mean body mass 66.0 +/- 13.2 kg; the deposit's Subjects sheet gives 65.9 +/- 13.1 kg (so this is not used as a check).", "The hand and sacrum displacement is missing for participants 1, 3, 8 and 10 in table1 (marker data not available), so they have no kinematic rows.",
                     "Table3's normalized vGRF is in N after the authors normalized it to the mean weight (646.9 N) of the whole group (paper, Table 3 footnote); the individual weights are in participants.csv.",
                     "The safety margin is unavailable for participants 1 and 7 in table1."])
    stdlib.release(DS, "summary", rows, meta, ck, participants=part, participant_columns=pcols)


if __name__ == "__main__":
    main()
