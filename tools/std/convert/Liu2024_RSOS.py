"""Liu, Valero-Cuevas & Finley (2024) R Soc Open Sci 11(5):231210: foot placement and centre-of-mass state at midstance for
unperturbed steps and for steps with a belt-speed perturbation, 13 adults walking on a treadmill, one row per step.

Raw files: raw_extra/Liu2024_RSOS/{FP_v_CoM_HO_20221224.mat, Demographics.xlsx} (OSF gv5tq, child component 8gwy2 'Data';
OSF states no licence, the article is CC BY 4.0). The MAT file holds two MATLAB table objects (CoM_FP_data: unperturbed steps;
CoM_FP_PTB_data: perturbed steps, with Mag = perturbation level) in which every step is 100 time-normalized samples
(GaitCycle 1-100). They are read without MATLAB by decoding the file's own object store (Liu2024_RSOS_mcos.py, the reader of
the Gonzalez-Rubio2019 converter); nothing from the deposit is run. This converter keeps GaitCycle 50 (midstance, the point the
paper analyses) of every step; the other 99 samples per step are not included.
Run from the site folder: python3 -I tools/std/convert/Liu2024_RSOS.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Liu2024_RSOS_mcos import MatObjects  # noqa: E402

DS = "Liu2024_RSOS"
MODULE = "walking"
# perturbation level Mag 1-6 -> belt speed change in m/s (paper Methods; order checked against the data: mean change in
# anterior-posterior CoM velocity rises monotonically from Mag 1 to Mag 6, negative for 1-3 and positive for 4-6)
MAG = {1: -0.5, 2: -0.4, 3: -0.3, 4: 0.3, 5: 0.5, 6: 0.7}
KEEP = ["No", "trial_no", "stp_no", "Side", "ML_CoM", "AP_CoM", "VT_CoM", "dCoM_ML", "dCoM_AP", "dCoM_VT", "AP_FP", "ML_FP"]


def table_mid(o, name):
    t = o.variable(o.top[name])
    names = [str(v[0]) if hasattr(v, "__len__") else str(v) for v in t["varnames"]]
    data = dict(zip(names, t["data"]))
    gc = np.asarray(data["GaitCycle"]).ravel()
    sel = np.flatnonzero(gc == 50)
    cols = {}
    for n in KEEP + (["Mag"] if "Mag" in names else []):
        c = data[n]
        cols[n] = [c[i][0] if hasattr(c[i], "__len__") else c[i] for i in sel] if isinstance(c, list) else np.asarray(c).ravel()[sel]
    d = pd.DataFrame(cols)
    d["n_samples"] = len(gc)
    return d, len(gc)


def main():
    raw = stdlib.raw_dir(DS)
    ck = stdlib.Checks()
    o = MatObjects(raw / "FP_v_CoM_HO_20221224.mat")
    u, nu = table_mid(o, "CoM_FP_data")
    q, nq = table_mid(o, "CoM_FP_PTB_data")
    dem = pd.read_excel(raw / "Demographics.xlsx")
    dem = dem.set_index("Sub")
    ck.internal("unperturbed table is 100 samples per step", nu == 100 * len(u), f"{nu} samples, {len(u)} steps")
    ck.internal("perturbed table is 100 samples per step", nq == 100 * len(q), f"{nq} samples, {len(q)} steps")
    key = ["No", "trial_no", "stp_no", "Side"]
    ck.internal("step key (participant, trial, step number, side) is unique in the all-steps table", not u.duplicated(key).any())
    n_dup_q = int(q.duplicated(key).sum())  # one step is listed twice in the perturbed table with two different levels
    m = q.merge(u, on=key, suffixes=("_q", "_u"), how="left", indicator=True)
    ck.internal("every perturbed step is also a row of the unperturbed-model table", (m._merge == "both").all())
    same = all((m[c + "_q"].fillna(-99) == m[c + "_u"].fillna(-99)).all() for c in ["ML_CoM", "AP_CoM", "dCoM_AP", "AP_FP", "ML_FP"])
    ck.internal("the perturbed steps hold the same values in both tables", same)
    # CoM_FP_data holds every step of the session, perturbed steps included; Mag marks the retained perturbed steps
    qm = q.groupby(key, sort=False).Mag.agg(["first", "nunique"]).reset_index()
    qm["Mag"] = qm["first"].where(qm["nunique"] == 1)  # a step listed with two levels keeps no level
    allr = u.merge(qm[key + ["Mag"]], on=key, how="left", indicator=True)
    allr["pert"] = (allr._merge == "both").astype(int)
    allr = allr.sort_values(["No", "trial_no", "stp_no", "Side"], kind="stable").reset_index(drop=True)
    sub = allr["No"].astype(int)
    p = allr.pert == 1
    mag = allr.Mag.where(p)
    ptype = np.where(p, "belt_speed_change", "none")
    pval = np.where(p, mag.map(MAG), 0.0)  # empty for the one step listed with two levels
    out = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment="", group="", subject=[f"{DS}_{s}" for s in sub], subject_excluded_by_authors=False,
        age=sub.map(dem.Age).astype(float), sex=sub.map(dem.Sex).map({"F": "F", "M": "M"}), handedness="",
        population="healthy adults", setting="lab", device="Instrumented treadmill; Qualisys motion capture (100 Hz)",
        session=1, block=[f"trial {int(t)}" for t in allr.trial_no], trial=0, phase="other", time_s=np.nan, task="walking",
        unit="step", condition=np.where(p, "perturbed step", "unperturbed step"), pert_type=ptype, pert_value=pval,
        pert_unit=np.where(p, "m/s", ""), feedback="normal", excluded_by_authors=False,
        walking_speed_mps=sub.map(dem.SSWS).astype(float), walking_trial=allr.trial_no.astype(int),
        heel_strike_side=allr.Side.map({"LHS": "left", "RHS": "right"}),
        com_position_ml_norm=allr.ML_CoM, com_position_ap_norm=allr.AP_CoM, com_position_vt_norm=allr.VT_CoM,
        com_velocity_ml_norm=allr.dCoM_ML, com_velocity_ap_norm=allr.dCoM_AP, com_velocity_vt_norm=allr.dCoM_VT,
        foot_placement_ap_norm=allr.AP_FP, foot_placement_ml_norm=allr.ML_FP))
    out["trial"] = out.groupby("subject", sort=False).cumcount() + 1
    # ---- checks
    ck.internal("13 participants, six perturbation levels", out.subject.nunique() == 13 and set(q.Mag) == set(MAG))
    mm = q.groupby("Mag").dCoM_AP.mean()
    ck.internal("mean AP CoM velocity change rises from Mag 1 to 6 (order of the paper's levels)",
                bool((mm.diff().dropna() > 0).all() and (mm.loc[[1, 2, 3]] < 0).all() and (mm.loc[[4, 5, 6]] > 0).all()), mm.round(3).to_dict().__repr__())
    ck.published("participants", out.subject.nunique(), 13, 0, "Methods, Participants")
    dm = dem.loc[sorted(set(sub))]
    ck.published("female participants", (dm.Sex == "F").sum(), 6, 0, "Methods, Participants (6 female)")
    ck.published("self-selected walking speed mean (m/s)", dm.SSWS.mean(), 0.75, 0.01, "Methods, Participants (0.75 +- 0.25 m/s)")
    ck.published("self-selected walking speed SD (m/s)", dm.SSWS.std(ddof=1), 0.25, 0.01, "Methods, Participants")
    ck.published("mean perturbations kept per participant (120 - 6)", int(out.pert_type.eq("belt_speed_change").sum()) / 13, 114, 1.0, "Methods: on average 6 of 120 perturbations removed")
    cnt = q.groupby(["No", "Mag", "Side"]).size()
    ck.published("median perturbations per participant, level and side", cnt.median(), 10, 0, "Methods: median 10 (IQR 1)")
    ck.published("IQR of perturbations per participant, level and side", cnt.quantile(.75) - cnt.quantile(.25), 1, 0, "Methods: median 10 (IQR 1)")
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = np.nan
    meta = dict(
        study=DS,
        paper=dict(authors="Liu C, Valero-Cuevas FJ, Finley JM", year=2024,
                   title="Generalizability of foot placement control strategies during unperturbed and perturbed gait",
                   venue="R Soc Open Sci 11(5):231210", doi="10.1098/rsos.231210"),
        lab="Finley (University of Southern California)", data_url="https://osf.io/gv5tq/",
        license="none stated (article: CC BY 4.0)", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Decoded the two MATLAB tables of FP_v_CoM_HO_20221224.mat, kept midstance (sample 50 of 100 in each step cycle) "
                 "of every step, merged unperturbed and perturbed steps in step order, replaced the participant ids with codes and "
                 "took age, sex and self-selected speed from Demographics.xlsx. The model-coefficient file is not included."),
        how=["One row is one step of one participant walking on a treadmill: the heel-marker foot placement (AP, ML) and the centre-of-"
             "mass position and velocity (ML, AP, VT) at midstance, all relative to the current stance foot. Perturbed steps carry "
             "the level of the belt-speed change in pert_value (m/s relative to the participant's self-selected speed); "
             "all other steps of the session are listed with no perturbation (the deposit's first table holds every step, perturbed ones included; "
             "about 6 of 120 perturbations per participant were removed by the authors after the fact and their steps stay in as unperturbed).",
             "The authors normalize positions by the participant's CoM height H and velocities by sqrt(g*H) (paper, Methods). The "
             "deposit has no README; the stored values behave as normalized (a belt change of 0.7 m/s moves the AP CoM velocity by "
             "about 0.21, i.e. 0.31 per m/s, close to 1/sqrt(g*H) for H near 1 m), so the columns end in _norm and H is not available "
             "to convert them to metres and m/s.",
             "The deposit has no README, so the six levels (Mag 1-6) are mapped to the paper's belt changes -0.5, -0.4, -0.3, +0.3, +0.5 "
             "and +0.7 m/s in the order the paper lists them, which the data confirm (see differences)."],
        unit_note="step (one foot strike to the next); values at midstance",
        differences=[f"{n_dup_q} step is listed twice in the deposit's perturbed table with two different levels (4 and 1); it is a perturbed step here with an empty pert_value.",
                     "The deposit's step cycle is 100 time-normalized samples; only midstance (sample 50), the point the paper analyses, "
                     "is kept, so the files hold 1 of every 100 samples.",
                     "Perturbation level order (Mag 1-6 = -0.5 ... +0.7 m/s) is inferred from the paper's list and confirmed because the "
                     "mean anterior-posterior CoM velocity at midstance rises monotonically across the six levels, negative for 1-3 and "
                     "positive for 4-6.",
                     "The paper gives participant age as 58 +- 29 years; the demographics sheet gives mean %.1f, SD %.1f (the paper's SD "
                     "looks like a typo)." % (dm.Age.mean(), dm.Age.std(ddof=1))],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        print(("  P " if c["kind"] == "published" else "  I ") + c["what"][:80], c.get("computed", ""), c.get("reported", ""), c.get("detail", "")[:150])


if __name__ == "__main__":
    main()
