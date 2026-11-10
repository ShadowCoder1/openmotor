"""Maiello, Paulun, Klein & Fleming 2019 (i-Perception): reanalysis of the precision-grip data of Paulun, Gegenfurtner, Goodale &
Fleming (2016, Exp Brain Res 234:2253-2265). 14 right-handed participants grasped four cylinders (Styrofoam, beech wood, brass and
Vaseline-covered brass; 10 cm long, 2.5 cm wide) lying at six orientations on a turntable with thumb and index finger, lifted them and
carried them to a goal; Optotrak 3020 markers on the finger nails. One row is one trial (5 practice + 120 recorded per participant).

Deposit (Zenodo 2247284, 335 KB Data_and_Analyses.zip): Paulun2016.dat (tab-separated trial table, 1750 rows x 59 columns),
ParsedPaulun2016Data.mat (the authors' medoid grasps, not used), Script_00_Parse_Paulun2016Data.m and Script_01_Analysis.m (read as
text only; they show the object-centred grasp coordinates and the exclusion rule). Nothing is executed.

Run from the site folder: python3 -I tools/std/convert/Maiello2019_iPerception.py"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Maiello2019_iPerception"
MODULE = "grasp_contacts"
RAW = stdlib.raw_dir(DS)
MATERIAL = {"light": "Styrofoam", "woody": "beech wood", "heavy": "brass", "vasel": "Vaseline-covered brass"}
DEVICE = "Optotrak 3020 with infrared markers on the thumb and index fingernails (100 Hz); cylinder on a turntable"
TXT = ["GrTp", "BlTp", "TrTp", "ObjectType"]


def num(x):
    return "" if pd.isna(x) else float(x)


def medoid_index(G):
    D = np.sqrt(((G[:, None, :] - G[None, :, :]) ** 2).sum(-1))
    return int(np.argmin((D ** 2).sum(0)))


def rotz(a, P):
    t = np.deg2rad(a)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return R @ P


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "Data_and_Analyses.zip")
    d = pd.read_csv(io.BytesIO(z.read("Data_and_Analyses/Paulun2016.dat")), sep="\t", dtype=str)
    for c in d.columns:
        if c not in TXT:
            d[c] = pd.to_numeric(d[c].str.strip(), errors="coerce")
    assert len(d) == 1750
    ctr = d.groupby("Sub").cumcount() + 1
    assert all((g.Trial.diff().dropna() >= 0).all() for _, g in d.groupby("Sub")), "file order is trial order"
    # the paper's long-axis exclusion applies to the 60 deg condition: thumb y (object-centred) below -30 mm
    noct = d.TouchedTime.isna()
    main_ = d.BlTp.isin(["Main1", "Main2"])
    longax = main_ & (d.Angle == 60) & (d.TouchedThumbYn < -30)
    rows = []
    for i, r in d.iterrows():
        sub = int(r.Sub)
        material = MATERIAL[r.ObjectType]
        assert r.TrTp == f"{r.ObjectType}_{int(r.Angle):03d}_deg", r.TrTp
        practice = r.BlTp == "Prac"
        rows.append(dict(
            dataset=DS, study=DS, experiment="Paulun et al. 2016 grasping data (reanalysed)",
            group={"NV": "slippery block last", "VN": "slippery block first"}[r.GrTp],
            subject=f"P{sub}", person=f"P{sub}", subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy",
            setting="lab", device=DEVICE, session=1, block={"Prac": "practice", "Main1": "main 1 (Styrofoam, wood, brass)", "Main2": "main 2 (Vaseline-covered brass)"}[r.BlTp],
            trial=int(ctr[i]), phase="practice" if practice else "test", time_s="", task="grasping and lifting", unit="trial",
            condition=f"{material}, {int(r.Angle)} deg", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=bool(noct[i] or longax[i]),
            object_id=int(r.ObjectNumber), object_angle_deg=float(r.Angle),
            thumb_contact_x_mm=num(r.TouchedThumbX), thumb_contact_y_mm=num(r.TouchedThumbY), thumb_contact_z_mm=num(r.TouchedThumbZ),
            index_contact_x_mm=num(r.TouchedFingerX), index_contact_y_mm=num(r.TouchedFingerY), index_contact_z_mm=num(r.TouchedFingerZ),
            thumb_contact_obj_x_mm=num(r.TouchedThumbXn), thumb_contact_obj_y_mm=num(r.TouchedThumbYn), thumb_contact_obj_z_mm=num(r.TouchedThumbZn),
            index_contact_obj_x_mm=num(r.TouchedFingerXn), index_contact_obj_y_mm=num(r.TouchedFingerYn), index_contact_obj_z_mm=num(r.TouchedFingerZn),
            reaction_time_ms=num(r.RT), reach_duration_ms=num(r.MovementDur), contact_to_lift_ms=num(r.LoadingTime), max_grip_aperture_mm=num(r.MaxAp)))
    # internal checks
    ck.internal("1750 rows = 14 participants x (5 practice + 120 recorded)", len(rows) == 1750 and d.Sub.nunique() == 14 and (d.groupby("Sub").size() == 125).all())
    nv = d.assign(first=main_).groupby("Sub")
    ok = True
    for s, g in d[d.BlTp != "Prac"].groupby("Sub"):
        v_first = g[g.ObjectType == "vasel"].Trial.min() < g[g.ObjectType != "vasel"].Trial.min()
        ok &= (v_first == (g.GrTp.iloc[0] == "VN"))
    ck.internal("GrTp VN = slippery (Vaseline) block before the other trials, NV = after", bool(ok))
    ck.internal("object number matches object type", (d.groupby("ObjectNumber").ObjectType.nunique() == 1).all())
    ck.internal("MovementDur equals TouchedTime - StartTime", ((d.MovementDur - (d.TouchedTime - d.StartTime)).abs().dropna() < 1e-6).all())
    # published checks
    ck.published("participants (14)", d.Sub.nunique(), 14, 0, "Maiello 2019 text: reanalysed data, N = 14")
    mn = d[main_ & d.Angle.isin([150, 60])].groupby(["Sub", "Angle"]).size()
    ck.published("recorded trials per participant at each orientation (20 = 4 materials x 5 repetitions)", mn.mean(), 20, 0, "Maiello 2019 text, 'i.e., 20 trials per participant at each orientation'")
    ck.published("recorded trials per participant (120 = 4 materials x 6 orientations x 5 repetitions)", main_.sum() / 14, 120, 0, "Paulun et al. 2016, Procedure (the source of the data)")
    ck.published("trials without a determinable first contact (1.25%)", 100 * (noct & main_).sum() / main_.sum(), 1.25, 0.05, "Paulun et al. 2016, Data analysis (the source of the data)")
    # Maiello 2019 reanalysis
    G = {}
    tot = exc = 0
    for ang in (150, 60):
        meds = []
        for s, g in d[main_ & (d.Angle == ang) & d.ObjectNumber.isin([1, 2, 3, 4])].groupby("Sub", sort=True):
            A = g[["TouchedThumbXn", "TouchedThumbYn", "TouchedThumbZn", "TouchedFingerXn", "TouchedFingerYn", "TouchedFingerZn"]].to_numpy(float)
            A = A[~np.isnan(A[:, 0])]
            tot += len(A)
            if ang == 60:
                exc += int((A[:, 1] < -30).sum())
                A = A[A[:, 1] >= -30]
            meds.append(A[medoid_index(A)])
        G[ang] = np.array(meds)
    ck.published("grasps excluded as lying along the long axis of the object (4%)", 100 * exc / tot, 4, 0.5, "Maiello 2019 text: 'We excluded from the analysis the 4% of grasps that fell along the long axis'")
    x = np.arange(-50, 51)

    def cand(ang, thumb_y):
        th = rotz(ang, np.array([x, np.full(101, thumb_y), np.zeros(101)])).T
        fi = rotz(ang, np.array([x, np.full(101, -thumb_y), np.zeros(101)])).T
        return np.hstack([th, fi])
    C150, C60 = cand(150, 12.5), cand(60, -12.5)
    M150 = G[150][medoid_index(G[150])]
    imin = int(np.argmin(np.sqrt(((C150 - M150) ** 2).sum(1))))
    reach = np.linspace(0, 1, 101)
    pc150, pc60 = np.linspace(0, 1, 101), np.linspace(1, 0, 101)
    i_reach = int(np.argmin(np.abs(reach - reach[imin])))
    i_pc = int(np.argmin(np.abs(pc60 - pc150[imin])))
    dr = np.sqrt(((G[60] - C60[i_reach]) ** 2).sum(1))
    dp = np.sqrt(((G[60] - C60[i_pc]) ** 2).sum(1))
    t = stats.ttest_1samp(dr - dp, 0)
    ck.published("distance to the minimum-reach prediction vs the object-visibility prediction, paired t(13) = 5.66", t.statistic, 5.66, 0.01, "Maiello 2019 text, Results (paired samples t-test)")
    ck.published("same test, p = 7.8e-5 (p in units of 1e-5)", t.pvalue * 1e5, 7.8, 0.15, "Maiello 2019 text, Results")
    meta = dict(
        study=DS,
        paper=dict(authors="Maiello G, Paulun VC, Klein LK, Fleming RW", year=2019,
                   title="Object Visibility, Not Energy Expenditure, Accounts For Spatial Biases in Human Grasp Selection",
                   venue="i-Perception 10(1):2041669519827608", doi="10.1177/2041669519827608"),
        lab="Fleming (Justus Liebig University Giessen)", data_url="https://doi.org/10.5281/zenodo.2247284",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Every row of the deposit's Paulun2016.dat was kept, in file order (practice trials, the 21 trials without a determined first contact and the grasps along the long axis included, and flagged instead of removed). Only 22 of its 59 columns are in the table: object, orientation, "
                 "the contact points of thumb and index finger (tracker coordinates and object-centred coordinates), reaction time, approach time, handling time and maximum grip aperture. Not included: the frame numbers and times of other movement events "
                 "(lift, goal, drop, transport), the grasp-axis angles and the positions at the goal, whose definitions the deposit does not give; the authors' medoid grasps (ParsedPaulun2016Data.mat) and the MATLAB scripts. Participant numbers became codes."),
        how=["One row is one trial: one participant grasping one cylinder (Styrofoam, beech wood, brass, or Vaseline-covered brass) lying at one of six orientations (0, 30, 60, 90, 120, 150 deg) on a turntable, lifting it and carrying it to a goal position. Each participant has 5 practice trials and 120 recorded trials (4 materials x 6 orientations x 5 repetitions). "
             "The deposit is the data of Paulun et al. (2016); Maiello et al. (2019) reanalysed the 150 deg and 60 deg conditions.",
             "Contact points are the thumb and index fingertip positions at the moment of first contact (the authors' multiple-sources-of-information method), in mm. thumb_contact_x_mm etc. are the tracker's coordinates; the _obj_ columns are the deposit's 'n' (normalised) coordinates relative to the object's centre in the table plane, "
             "the frame in which Maiello et al. compare grasps to the outline of the rotated 100 x 25 mm object. Times are in ms (200 Hz frames of 5 ms in the deposit's time columns; the paper gives 100 Hz for the camera). reaction_time_ms is the start signal to movement onset (hand speed above 0.025 m/s), reach_duration_ms movement onset to first contact, "
             "contact_to_lift_ms first contact to lift, max_grip_aperture_mm the maximum thumb-index distance before contact.",
             "excluded_by_authors is true for trials where first contact could not be determined (21 trials, no contact values) and for 60 deg grasps with the thumb more than 30 mm off the object's midline in y (the 4% along the long axis that Maiello et al. removed). Group says whether the slippery (Vaseline) block came after (NV) or before (VN) the other trials; it was counterbalanced between participants."],
        unit_note="trial (grasp of a cylinder)",
        differences=["The paper (Methods) and Paulun et al. 2016 give 100 Hz for the Optotrak, but the deposit's time columns advance in 5 ms per frame; times are copied as the deposit gives them.",
                     "The age, sex and handedness of the 14 analysed participants are not in the deposit; the original study reports 24 participants (17 women), 14 of whom had usable data, all right-handed by self-report.",
                     "Start position of the hand is a fixed marker and is not in the deposit's table, so start_x/y/z are empty."],
    )
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
