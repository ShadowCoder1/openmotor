"""Reznick et al. (2021) Sci Data: lower-limb kinematics and kinetics of 10 able-bodied adults during continuously varying
locomotion. This release keeps the treadmill WALKING part of Normalized.mat: one row per stride, plus a second table with
the stride-normalized sagittal hip, knee and ankle angle curves.

Raw file (untrusted data, read only): raw_extra/Reznick2021_SciData/Normalized.mat (Figshare 10.6084/m9.figshare.14558184,
CC0), MATLAB v7.3 / HDF5, read with h5py. Normalized/ABnn/Walk/<speed>/<incline>/events/StrideDetails is
4 x n [first frame, last frame, duration in frames at 100 Hz, leg (left 1, right 2)]. The README (figshare 28998039) and the
data paper were read as text. Run from the site folder: python3 -I tools/std/convert/Reznick2021_SciData.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Reznick2021_SciData"
MODULE = "walking"
INCL = {"i0": 0, "i5": 5, "i10": 10, "in5": -5, "in10": -10}
STEADY = {"s0x8": 0.8, "s1": 1.0, "s1x2": 1.2}
ACC = {"a0x2": "accelerating 0.2 m/s2 from rest to 1.2 m/s", "a0x5": "accelerating 0.5 m/s2 from rest to 1.2 m/s",
       "d0x2": "decelerating 0.2 m/s2 from 1.2 m/s", "d0x5": "decelerating 0.5 m/s2 from 1.2 m/s"}


def text(f, ref):
    o = f[ref]
    return "".join(chr(c) for c in o[()].ravel())


def main():
    f = h5py.File(stdlib.raw_dir(DS) / "Normalized.mat", "r")
    N = f["Normalized"]
    ck = stdlib.Checks()
    rows, people, curves = [], [], []
    for ab in sorted(N.keys()):
        det = {}
        a = N[ab]["ParticipantDetails"][()]
        for j in range(a.shape[1]):
            name = text(f, a[0, j])
            o = f[a[1, j]]
            det[name] = text(f, a[1, j]) if o.dtype == np.uint16 else float(o[()].ravel()[0])
        code = f"{DS}_S{len(people) + 1:04d}"
        people.append(dict(person=code, sex="F" if det["Sex"] == 1 else "M", age=det["Age"], mass_kg=det["Bodymass"],
                           height_m=det["Height"] / 1000, left_leg_length_m=det["LeftLegLength"] / 1000,
                           right_leg_length_m=det["RightLegLength"] / 1000))
        k = 0
        W = N[ab]["Walk"]
        for sp in sorted(W.keys()):
            for inc in sorted(W[sp].keys()):
                g = W[sp][inc]
                sd = g["events"]["StrideDetails"][()]  # 4 x n
                n = sd.shape[1]
                hip = g["jointAngles"]["HipAngles"][()]  # n x 3 x 150
                knee = g["jointAngles"]["KneeAngles"][()]
                ank = g["jointAngles"]["AnkleAngles"][()]
                if n == 1 and hip.ndim == 2:  # HDF5 drops the stride axis when a task holds one stride
                    hip, knee, ank = hip[None], knee[None], ank[None]
                assert hip.shape == (n, 3, 150) and sd.shape[0] == 4 and set(np.unique(sd[3])) <= {1.0, 2.0}
                steady = sp in STEADY
                for i in range(n):
                    k += 1
                    left = sd[3, i] == 1
                    dur = sd[2, i] / 100.0
                    rows.append(dict(
                        dataset=DS, study=DS, experiment="", group="", subject=code, subject_excluded_by_authors=False,
                        age=det["Age"], sex=people[-1]["sex"], handedness="", population="healthy", setting="lab",
                        device="Bertec instrumented treadmill, Vicon motion capture", session=1, block=f"{sp}/{inc}",
                        trial=k, phase="other", time_s="", task="walking", unit="stride",
                        condition=(f"{STEADY[sp]:.1f} m/s steady" if steady else ACC[sp]) + f", incline {INCL[inc]} deg",
                        pert_type="none", pert_value=0.0, pert_unit="", feedback="normal", excluded_by_authors=False,
                        walking_context="treadmill", surface_incline_deg=INCL[inc],
                        belt_speed_fast_mps=STEADY.get(sp, ""), belt_speed_slow_mps=STEADY.get(sp, ""),
                        stride_start_s=sd[0, i] / 100.0, stride_time_s=dur,
                        stride_time_left_s=dur if left else "", stride_time_right_s="" if left else dur))
                    curves.append([code, k, "left" if left else "right"] + list(np.round(
                        np.concatenate([hip[i, 0], knee[i, 0], ank[i, 0]]), 3)))
    allc = [c[1] for c in stdlib.columns_for(MODULE)]
    df = pd.DataFrame(rows).reindex(columns=allc)
    df = df.fillna('')
    pp = pd.DataFrame(people)

    names = [f"{j}_{p:03d}" for j in ("hip_flexion_deg", "knee_flexion_deg", "ankle_dorsiflexion_deg") for p in range(1, 151)]
    cdf = pd.DataFrame(curves, columns=["subject", "trial", "leg"] + names)
    cdf.insert(0, "dataset", DS)
    extra = [(f"stride_curves_sagittal.csv", cdf.to_csv(index=False, float_format="%.5g", lineterminator="\r\n"))]
    readme_cols = ("subject, trial (as in the main table), leg (left or right: the foot whose heel strikes begin the stride), "
                   "then 150 points per joint over the stride (0 to 100 percent), hip then knee then ankle, sagittal-plane angle in degrees")

    # internal checks
    ck.internal("10 participants, none lost", df["subject"].nunique() == 10)
    ck.internal("every stride has a duration between 0.3 and 10 s (the longest are start-from-rest acceleration strides)", bool(((df.stride_time_s > 0.3) & (df.stride_time_s < 10)).all()))
    ck.internal("curve table has one row per stride", len(cdf) == len(df))
    ck.internal("curve values are finite", bool(np.isfinite(cdf[names].to_numpy(float)).all()))
    # published (data descriptor, Methods > Participants)
    ck.published("number of participants", len(pp), 10, 0, "Methods, Participants")
    ck.published("number of women", (pp.sex == "F").sum(), 5, 0, "Methods, Participants")
    ck.published("mean age (years)", pp.age.mean(), 30.4, 0.05, "Methods, Participants")
    ck.published("SD of age (years)", pp.age.std(ddof=0), 14.9, 0.05, "Methods, Participants")
    ck.published("mean body mass (kg)", pp.mass_kg.mean(), 74.6, 0.05, "Methods, Participants")
    ck.published("SD of body mass (kg)", pp.mass_kg.std(ddof=0), 9.7, 0.05, "Methods, Participants")
    ck.published("mean height (m)", pp.height_m.mean(), 1.73, 0.005, "Methods, Participants")
    for c in ck.items:
        print(c)

    pcols = [("sex", "category", "F, M", "Sex from the deposit's participant details (1 female, 2 male)."),
             ("age", "number", "years", "Age in years."), ("mass_kg", "number", "kg", "Body mass."),
             ("height_m", "number", "m", "Standing height (deposit gives mm)."),
             ("left_leg_length_m", "number", "m", "Left leg length (deposit gives mm)."),
             ("right_leg_length_m", "number", "m", "Right leg length (deposit gives mm).")]
    meta = dict(
        study=DS,
        paper=dict(authors="Reznick E, Embry KR, Neuman R, Bolivar-Nieto E, Fey NP, Gregg RD", year=2021,
                   title="Lower-limb kinematics and kinetics during continuously varying human locomotion",
                   venue="Sci Data 8:282", doi="10.1038/s41597-021-01057-9"),
        lab="Gregg (University of Michigan)",
        data_url="https://doi.org/10.6084/m9.figshare.c.5175254",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Kept the treadmill walking strides of the deposit's Normalized.mat (steady speeds 0.8, 1.0 and 1.2 m/s and "
                 "constant accelerations and decelerations, each at five slopes) with the authors' stride start, duration "
                 "and side, and the sagittal hip, knee and ankle angle curves in a second table; participant names are "
                 "replaced by codes. Running, stairs, sit-to-stand, markers, forces, moments and powers, and the "
                 "frontal and transverse angles are not included."),
        how=["One row is one stride (heel strike to the next heel strike of the same foot, left or right) of one participant "
             "walking on an instrumented treadmill, in the deposit's order: tasks in alphabetical order of speed code and "
             "slope code, strides in recording order within each task. Treadmill slopes are in degrees, positive uphill. "
             "stride_start_s is the stride's first frame at 100 Hz divided by 100; the deposit does not say what that frame "
             "counts from, so it orders strides within a task but is not a time since the session began.",
             "Steady tasks have belt_speed_fast_mps and belt_speed_slow_mps both set to the nominal speed (both legs on one "
             "belt; the authors scaled commanded speeds by leg length, see their Methods). Acceleration and deceleration "
             "tasks leave the belt speeds empty and give the rate in the condition column. stride_time_left_s or "
             "stride_time_right_s repeats stride_time_s for the side of the stride.",
             "The authors removed unsatisfactory strides (gaps, discontinuities) before sharing, so excluded_by_authors is "
             "false on every row. stride_curves_sagittal.csv holds, for the same strides, " + readme_cols + ". Axis 1 of "
             "the deposit's angle arrays is taken as the sagittal plane, as in the authors' example script; hip, knee and "
             "ankle angles are the Vicon Plug-in Gait relative joint angles, so their sign convention is the authors'."],
        unit_note="stride",
        differences=["The paper's standard deviations (age 14.9, mass 9.7) are population SDs (divided by n), which is how they were reproduced.", "The data paper's Usage Notes say AB04 performed 59 strides walking at 1.0 m/s on the -10 degree slope; the "
                     "deposit holds 53 strides for that task (and 59 for 1.2 m/s at -10 degrees), so that number was not used "
                     "as a check."],
    )
    stdlib.release(DS, MODULE, df, meta, ck, participants=pp, participant_columns=pcols, extra_files=extra)
    print(len(df), "rows")


main()
