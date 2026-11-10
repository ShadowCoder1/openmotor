"""Ebner-Karestinos et al. (2018), PLoS ONE 13(2):e0191684: 25 children with unilateral cerebral palsy (UCP) and 25 typically developing (TD)
children held a grip-lift manipulandum while stepping down a step (5 trials per hand); per-child, per-hand means of the grip force (GF), load
force (LF) and temporal variables, and maximum voluntary contraction (MVC). Dryad doi 10.5061/dryad.vm407, copy at Zenodo record 4987178 (CC0).
Converted: the three per-child tables (Dynam_var, MVC_var, Temporal_var) in long 'summary' form. Fig 2 file (group-mean traces) is not per child and is left out.
Run on the PC: python3 -I tools/std/pc.py run EbnerKarestinos2018_PLoSONE"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "EbnerKarestinos2018_PLoSONE"
Z = zipfile.ZipFile(stdlib.raw_dir(DS) / "data2018.zip")
EV = ["LFmin", "GFmin", "LFmax", "GFmax"]
HANDS = {"UCP": ["more affected hand", "less affected hand"], "TD": ["nondominant hand", "dominant hand"]}
POP = {"UCP": "unilateral cerebral palsy", "TD": "typically developing"}


def sheet(name):
    return openpyxl.load_workbook(io.BytesIO(Z.read(f"DRYAD/{name}")), data_only=True).worksheets[0]


def val(ws, r, c):
    v = ws.cell(row=r, column=c + 1).value
    return None if v is None else float(v)


def main():
    ck = stdlib.Checks()
    recs = []  # (grp, pid, hand_index, measure, desc, value, unit)
    dyn, mvc, tmp = sheet("Dynam_var(table1).xlsx"), sheet("MVC_var(table2).xlsx"), sheet("Temporal_var(table1).xlsx")
    evdesc = {"LFmin": "the minimum load force", "GFmin": "the minimum grip force", "LFmax": "the maximum load force", "GFmax": "the maximum grip force"}
    # dynamic: blocks LF rows 7-31, GF rows 41-65, ratio rows 75-99; UCP cols (1,2),(4,5),(7,8),(10,11); TD cols (15,16),(18,19),(21,22),(24,25)
    for var, r0, vname, unit in [("LF", 7, "load force", "N"), ("GF", 41, "grip force", "N"), ("GFLF", 75, "grip force to load force ratio", "ratio")]:
        for grp, c0 in (("UCP", 1), ("TD", 15)):
            for i in range(25):
                pid = int(val(dyn, r0 + i, c0 - 1 if grp == "UCP" else 14))
                assert pid == i + 1, (var, grp, pid, i)
                for e, ev in enumerate(EV):
                    for h in (0, 1):
                        v = val(dyn, r0 + i, c0 + 3 * e + h)
                        nm = f"{var}_at_{ev}" if var != "GFLF" else f"GF_LF_ratio_at_{ev}"
                        recs.append((grp, pid, h, nm, f"Child's mean {vname} at the moment of {evdesc[ev]} while stepping down", v, unit))
    for i in range(25):
        for h in (0, 1):
            recs.append(("UCP", i + 1, h, "MVC_static", "Child's mean maximum voluntary grip force in standing, mean of three trials", val(mvc, 7 + i, 1 + h), "N"))
            recs.append(("TD", i + 1, h, "MVC_static", "Child's mean maximum voluntary grip force in standing, mean of three trials", val(mvc, 7 + i, 5 + h), "N"))
    for grp, c0 in (("UCP", 1), ("TD", 15)):
        for i in range(25):
            for e, ev in enumerate(EV):
                for h in (0, 1):
                    recs.append((grp, i + 1, h, f"GF_pct_MVC_at_{ev}", f"Grip force at the moment of {evdesc[ev]} as a percentage of the child's MVC", val(mvc, 41 + i, c0 + 3 * e + h), "% of MVC"))
    tnames = ["LFmin_to_GFmin", "LFmin_to_LFmax", "GFmin_to_LFmax", "LFmax_to_GFmax"]
    for grp, r0 in (("UCP", 5), ("TD", 37)):
        for i in range(25):
            assert int(val(tmp, r0 + i, 0)) == i + 1
            for k, tn in enumerate(tnames):
                a, b = tn.split("_to_")
                for h in (0, 1):
                    recs.append((grp, i + 1, h, f"delay_{tn}", f"Child's mean delay from the moment of {evdesc[a]} to the moment of {evdesc[b]} (negative: the second event came first)", val(tmp, r0 + i, 1 + 3 * k + h), "ms"))
    rows = []
    order = {}
    for grp, pid, h, nm, desc, v, unit in recs:
        if v is None or np.isnan(v):
            continue
        subj = f"{grp}{pid:02d}"
        order[subj] = order.get(subj, 0) + 1
        rows.append(dict(dataset=DS, study="Ebner-Karestinos 2018 PLoS ONE", experiment="", group=grp, subject=subj, person=subj,
                         subject_excluded_by_authors=False, age="", sex="", handedness="", population=POP[grp], setting="lab",
                         device="Grip-lift manipulandum (two ATI Mini40 sensors, 800 Hz) held while stepping down a 16.5 x 61 x 30 cm step",
                         session=1, block="step-down task", trial=order[subj], phase="test", time_s="", task="grasping and lifting", unit="participant summary",
                         condition=HANDS[grp][h], pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                         measure=nm, measure_description=desc, statistic="mean", value=v, value_unit=unit, n_trials="", session_context=""))
    import pandas as pd
    d = pd.DataFrame(rows)

    def m(grp, hand, meas, sd=False):
        s = d[(d.group == grp) & (d.condition == HANDS[grp][hand]) & (d.measure == meas)].value
        return s.std(ddof=1) if sd else s.mean()
    ck.published("children with UCP (25)", d[d.group == "UCP"].subject.nunique(), 25, 0, "Abstract; Methods, Participants")
    ck.published("typically developing children (25)", d[d.group == "TD"].subject.nunique(), 25, 0, "Abstract; Methods, Participants")
    T1 = [("UCP", 0, "delay_LFmin_to_GFmin", -42.37, 108.6), ("UCP", 1, "delay_LFmin_to_GFmin", 1.42, 114.3), ("TD", 0, "delay_LFmin_to_GFmin", 36.58, 48.6),
          ("TD", 1, "delay_LFmin_to_GFmin", 30.55, 61.2), ("UCP", 0, "delay_LFmax_to_GFmax", 24.93, 68.5), ("TD", 1, "delay_GFmin_to_LFmax", 208.16, 56.5),
          ("UCP", 0, "LF_at_LFmin", 1.64, 0.7), ("UCP", 1, "LF_at_LFmax", 3.66, 0.6), ("TD", 0, "LF_at_GFmax", 2.13, 0.3),
          ("UCP", 0, "GF_at_LFmin", 7.13, 7.6), ("TD", 1, "GF_at_GFmax", 5.40, 1.9), ("UCP", 1, "GF_at_GFmax", 11.06, 11.7),
          ("UCP", 0, "GF_LF_ratio_at_LFmin", 6.20, 3.6), ("TD", 0, "GF_LF_ratio_at_GFmax", 2.21, 0.8),
          ("UCP", 0, "MVC_static", 16.73, 7.6), ("UCP", 1, "MVC_static", 40.35, 21.7), ("TD", 0, "MVC_static", 37.70, 16.0), ("TD", 1, "MVC_static", 43.19, 15.6),
          ("UCP", 0, "GF_pct_MVC_at_LFmin", 44.61, 34.3), ("TD", 1, "GF_pct_MVC_at_GFmax", 14.19, 7.5)]
    for grp, h, meas, mean, sd in T1:
        ref = "Table 2" if ("MVC" in meas) else "Table 1"
        ck.published(f"{grp} {HANDS[grp][h]} {meas} mean", m(grp, h, meas), mean, 0.0051, ref)
        ck.published(f"{grp} {HANDS[grp][h]} {meas} SD", m(grp, h, meas, True), sd, 0.051, ref)
    ck.internal("every child has data for at least one hand", d.subject.nunique() == 50 and bool(d[d.measure == "LF_at_LFmin"].subject.nunique() == 50))
    meta = dict(study=DS, paper=dict(authors="Ebner-Karestinos D, Flament B, Arnould C, Thonnard JL, Bleyenheuft Y", year=2018,
                title="Precision grip control while walking down a step in children with unilateral cerebral palsy", venue="PLoS ONE 13(2):e0191684", doi="10.1371/journal.pone.0191684"),
        lab="Bleyenheuft (UCLouvain)", data_url="https://doi.org/10.5061/dryad.vm407 (copy: https://zenodo.org/records/4987178)", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("The three per-child tables (dynamic variables, MVC and percentage of MVC, temporal variables) were turned from wide to long form (one row per child, hand and measure); "
                 "measure names were written as snake_case (for example LF_at_GFmin = load force at the moment of the minimum grip force); empty cells (hands without data) produce no row. "
                 "The group-mean force traces of the Fig 2 file are not converted. The deposit's MEAN and SD rows were not copied."),
        how=["One row is one measure of one child for one hand: the child's mean over the step-down trials (5 per hand) of the load force (LF), grip force (GF) or GF/LF ratio at the moments of minimum and maximum LF and GF, the delays between these events (ms), and the maximum voluntary pinch force (mean of three trials) with grip force at the events as % of it.",
             "Hands: children with UCP more affected hand (MAH) and less affected hand (LAH); typically developing children nondominant (NDH) and dominant hand (DH). Each child is one subject and person; trial counts rows within the child.",
             "The deposit labels the GF/LF ratio 'N'; it is unitless. The block labelled 'LOAD FORCE (LF)' lists LF at each of the four events (column heads LFmin, GFmin, LFmax, GFmax name the event), checked against Table 1 of the paper."],
        unit_note="measure of one child and hand (long form)",
        differences=["Some children have values for only one hand (empty cells in the deposit, for example children whose more affected hand could not be measured), so their other-hand rows are absent.", "Age, sex and affected side per child are not in the deposit (the paper gives group means only)."])
    stdlib.release(DS, "summary", rows, meta, ck)


if __name__ == "__main__":
    main()
