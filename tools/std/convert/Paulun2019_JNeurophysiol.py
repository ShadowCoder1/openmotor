"""Paulun, Buckingham, Goodale & Fleming 2019 (J Neurophysiol): the material-weight illusion in objects made of two materials.
Four experiments with the same stone, wood and Styrofoam objects (all the same weight except in Experiment 4) lifted with a precision
grip by a handle holding two ATI Nano17 force-torque sensors; participants rated the heaviness of each half (Experiments 1 and 4) or of
the whole object (Experiments 2 and 3) before and after lifting. One row is one trial (rating before a lift counts as a trial with
lift number 0 and no force data).

Deposit (Zenodo 1345746): Experiment_{1,2,3,4}_Data.csv (one row per trial) and Exeriment_{1,2,3,4}_Description.txt (column codes and row
counts). Experiment 4's CSV has every line wrapped in one pair of quotes, which is undone here. Nothing is executed.

Run from the site folder: python3 -I tools/std/convert/Paulun2019_JNeurophysiol.py"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Paulun2019_JNeurophysiol"
MODULE = "weight_illusion"
RAW = stdlib.raw_dir(DS)
MAT = {"Stone": "stone", "Wood": "wood", "Styrofoam": "Styrofoam"}
DEV = "Handle with two ATI Nano17 six-axis force-torque sensors (1000 Hz) on the object; precision grip of index finger and thumb"
SITE = {1: "Western Ontario", 2: "Western Ontario", 3: "Giessen", 4: "Western Ontario"}


def read(e):
    if e != 4:
        return pd.read_csv(RAW / f"Experiment_{e}_Data.csv", na_values=["NA"])
    rows = []
    for l in (RAW / "Experiment_4_Data.csv").read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l.startswith('"') and l.endswith('"') and l.count('""') > 2:
            l = l[1:-1].replace('""', '"')
        rows.append(next(csv.reader([l])))
    d = pd.DataFrame(rows[1:], columns=rows[0])
    for c in d.columns:
        x = d[c].replace("NA", np.nan)
        try:
            d[c] = pd.to_numeric(x)
        except ValueError:
            d[c] = x
    return d


def v(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else x


def mat(x):
    return "" if not isinstance(x, str) else MAT.get(x, x)


def main():
    ck = stdlib.Checks()
    rows = []
    data = {e: read(e) for e in (1, 2, 3, 4)}
    for e, d in data.items():
        for r in d.to_dict("records"):
            sid = f"E{e}_S{int(r['Subject']):02d}"
            practice = r["TrTp"] == "Practice"
            obj = r.get("Object")
            obj = "wood (practice object)" if (isinstance(obj, float) or obj is None) and practice else obj
            task = r.get("RatingTask")
            task = {"NumbericalRating": "numerical rating", "CoM": "centre-of-mass judgement"}.get(task, "") if isinstance(task, str) else ""
            lift = v(r.get("LiftNum"))
            row = dict(
                dataset=DS, study=DS, experiment=f"Experiment {e}", group=task, subject=sid, person=sid, subject_excluded_by_authors=False,
                age="", sex="", handedness="", population="healthy", setting="lab", device=DEV + f" ({SITE[e]})", session=1,
                block=r["TrTp"], trial=int(r["Trial"]), phase="practice" if practice else "test", time_s="", task="grasping and lifting", unit="trial",
                condition=obj if isinstance(obj, str) else "", pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                rating_task=task, lifting_order=int(r["LiftingOrder"]), lift_number=int(lift) if lift != "" else "", object=obj if isinstance(obj, str) else "",
                material_index=mat(r.get("Material_Finger")), material_thumb=mat(r.get("Material_Thumb")),
                weight_index_kg=v(r.get("Weight_Finger")), weight_thumb_kg=v(r.get("Weight_Thumb")), half_index=v(r.get("Half_Finger")), half_thumb=v(r.get("Half_Thumb")),
                rating_index_z=v(r.get("Rating_Finger")), rating_thumb_z=v(r.get("Rating_Thumb")), rating_object_z=v(r.get("Rating")),
                peak_torque_nmm=v(r.get("Torque_Peak", r.get("Torque"))), hold_torque_nmm=v(r.get("Torque_Hold")),
                peak_grip_force_n=v(r.get("GF")), peak_grip_force_rate_n_s=v(r.get("GFR")), peak_load_force_n=v(r.get("LF")), peak_load_force_rate_n_s=v(r.get("LFR")))
            rows.append(row)
    n = {e: d.Subject.nunique() for e, d in data.items()}
    d1, d2, d3, d4 = data[1], data[2], data[3], data[4]
    # rows per file, as the deposit's own descriptions state them
    for e, want in ((1, 1911), (2, 912), (3, 936), (4, 1475)):
        ck.published(f"Experiment {e} trials (rows) in the deposit's description file ({want})", len(data[e]), want, 0, f"Exeriment_{e}_Description.txt, ROW CODES")
    ck.published("Experiment 1 participants analysed (49)", n[1], 49, 0, "Results, Experiment 1 (49 participants), Methods exclusions")
    ck.published("Experiment 1 numerical-rating participants (24)", d1[d1.RatingTask == "NumbericalRating"].Subject.nunique(), 24, 0, "Methods, Experiment 1 (24 numerical ratings)")
    ck.published("Experiment 1 centre-of-mass participants (25)", d1[d1.RatingTask == "CoM"].Subject.nunique(), 25, 0, "Methods, Experiment 1 (25 CoM judgements)")
    ck.published("Experiment 2 participants (24)", n[2], 24, 0, "Methods, Experiment 2, Participants")
    ck.published("Experiment 3 participants (24)", n[3], 24, 0, "Methods, Experiment 3, Participants")
    m1 = d1[d1.TrTp == "Main"]
    L = pd.concat([m1[["Subject", "LiftNum", "Material_Finger", "Rating_Finger"]].set_axis(["S", "L", "M", "r"], axis=1),
                   m1[["Subject", "LiftNum", "Material_Thumb", "Rating_Thumb"]].set_axis(["S", "L", "M", "r"], axis=1)])
    g = L[L.L >= 1].groupby(["S", "M"]).r.mean().unstack()
    ck.published("Experiment 1 illusion index after lifting (stone - Styrofoam) vs 0, t(48) = 8.03", stats.ttest_1samp(g["Stone"] - g["Styrofoam"], 0).statistic, 8.03, 0.01, "Results, Experiment 1, Perception")
    tq = m1.groupby("Subject").Torque.mean().dropna()
    ck.published("Experiment 1 net peak torque vs 0, t(48) = -0.35", stats.ttest_1samp(tq, 0).statistic, -0.35, 0.01, "Results, Experiment 1, Torque")
    m2 = d2[d2.TrTp == "Main"]
    pre = m2[m2.LiftNum == 0].groupby(["Subject", "Object"]).Rating.mean().unstack().mean()
    post = m2[m2.LiftNum >= 1].groupby(["Subject", "Object"]).Rating.mean().unstack().mean()
    ck.published("Experiment 2 expected heaviness before lifting, Styrofoam (-2.93)", pre["Styrofoam"], -2.93, 0.005, "Results, Experiment 2, Perception")
    ck.published("Experiment 2 expected heaviness before lifting, wood (-1.22)", pre["Wood"], -1.22, 0.005, "Results, Experiment 2, Perception")
    ck.published("Experiment 2 expected heaviness before lifting, stone (0.60)", pre["Stone"], 0.60, 0.005, "Results, Experiment 2, Perception")
    ck.published("Experiment 2 perceived heaviness after lifting, stone (-0.03)", post["Stone"], -0.03, 0.005, "Results, Experiment 2, Perception")
    ck.published("Experiment 2 perceived heaviness after lifting, wood (0.25)", post["Wood"], 0.25, 0.005, "Results, Experiment 2, Perception")
    f1 = m2[m2.LiftNum == 1].groupby(["Subject", "Object"]).GFR.mean().unstack().mean()
    ck.published("Experiment 2 peak grip-force rate in the first lift, Styrofoam (47.96 N/s)", f1["Styrofoam"], 47.96, 0.01, "Results, Experiment 2, Forces")
    ck.published("Experiment 2 peak grip-force rate in the first lift, wood (63.64 N/s)", f1["Wood"], 63.64, 0.01, "Results, Experiment 2, Forces")
    ck.published("Experiment 2 peak grip-force rate in the first lift, stone (78.86 N/s)", f1["Stone"], 78.86, 0.02, "Results, Experiment 2, Forces")
    ck.internal("Experiment 1: 6 practice rows and 33 main rows (3 objects x lift 0-10) per participant", bool((d1.groupby("Subject").size() == 39).all()))
    ck.internal("Experiment 2: 5 practice rows and 33 main rows per participant", bool((d2.groupby("Subject").size() == 38).all()))
    ck.internal("Experiment 4: 25 participants x 59 rows", bool(n[4] == 25 and (d4.groupby("Subject").size() == 59).all()))
    ck.internal("all rows kept: 1911 + 912 + 936 + 1475", len(rows) == 1911 + 912 + 936 + 1475, str(len(rows)))
    meta = dict(
        study=DS,
        paper=dict(authors="Paulun VC, Buckingham G, Goodale MA, Fleming RW", year=2019,
                   title="The material-weight illusion disappears or inverts in objects made of two materials",
                   venue="J Neurophysiol 121(3):996-1010", doi="10.1152/jn.00199.2018"),
        lab="Fleming (Justus Liebig University Giessen) and Goodale (University of Western Ontario)", data_url="https://doi.org/10.5281/zenodo.1345746",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("All rows of the four CSV files were kept in file order, with the column names changed to the module's, 'Finger' renamed index (the index-finger side), Experiment 1's misspelt task label NumbericalRating corrected, "
                 "Experiment 4's quote-wrapped lines unwrapped, and participant numbers turned into codes. Only the per-trial values the authors shared are included: the force and torque signals themselves (1000 Hz) are not in the deposit."),
        how=["One row is one trial: the participant rated the heaviness of the object (before any lift: lift number 0, no forces), or lifted it with a precision grip, held it for 3 s and then rated it. Experiment 1: 49 participants (24 gave a number for the heaviness of each half of three bipartite objects, 25 pointed to the perceived centre of mass; the CoM was converted by the authors to a rating of each half). "
             "Experiment 2: 24 participants lifted three uniform-looking objects (stone, wood, Styrofoam) and rated each whole object, with peak grip and load force and their rates. Experiment 3: 24 participants lifted the bipartite objects of Experiment 1 and rated the whole object. "
             "Experiment 4: participants lifted five objects with real weight differences between the halves (100 g) and rated each half, with torque measures.",
             "Ratings are z-scores within each participant. Torque (N.mm) was aligned by the authors across object orientations so that a positive value is a rotation toward the heavier (Experiment 4) or heavier-looking (Experiments 1, 4 equal-weight object) side. Experiment 1 gives the peak torque during loading; Experiment 4 gives the peak and the median during holding.",
             "Practice trials (5 per participant, with a uniform wood object) are phase practice; the first row of each object has lift number 0. Participants are coded per experiment (the paper says none of Experiment 2's participants took part in Experiment 1; it does not say whether anyone took part in more than one of the others)."],
        unit_note="trial (rating or lift of an object)",
        differences=["Experiment 4: the deposit's CSV holds 25 participants (59 rows each: 6 practice and 53 main rows), whereas the paper reports 24 (15 men, 9 women). All 25 are released; the deposit gives no reason for the extra one.",
                     "Experiment 2: the paper's peak load-force-rate means in the first lift (Styrofoam 44.80 N/s, stone 60.16, wood 59.91) are about twice the deposit's values (22.40, 29.58, 29.95); the deposit's description does not define LFR. The deposit's peak grip force means of the later lifts (7.41, 7.76, 8.04 N) also differ from the paper's overall means (7.14, 7.89, 8.62 N), which the paper does not define precisely.",
                     "Experiment 2: the illusion index after lifting gives t(23) = -3.80 from the rows (all lifts after the first rating), against t(23) = -3.74 in the paper; Experiment 1's reproduces exactly (8.03).",
                     "Experiment 1: the paper reports mean ratings per material (Styrofoam -1.21, stone 0.96, wood -0.04) that were not reproduced from the rows; the way these were averaged is not described."],
    )
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
