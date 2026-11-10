"""Ferber, Brett, Fukuchi, Hettinga & Osis (2024) Sci Data 11:1232: Running Injury Clinic treadmill walking and running
dataset (1,798 people). Only the walking metadata table is converted: one row per walking recording with the self-selected
walking speed and the person's age, sex and injury status. The 3D marker data and derived gait variables sit in
ric_data.zip (22.7 GB, over the 5 GB limit) and are not converted.

Raw files: raw_extra/Ferber2024_SciData/{walk_data_meta.csv,README.txt} (Figshare+ 10.25452/figshare.plus.24255795).
Run from the site folder: python3 -I tools/std/convert/Ferber2024_SciData.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Ferber2024_SciData"
MODULE = "walking"


def main():
    raw = stdlib.raw_dir(DS)
    ck = stdlib.Checks()
    d = pd.read_csv(raw / "walk_data_meta.csv")
    n0 = len(d)
    d["_ord"] = np.arange(n0)
    d["session"] = d.groupby("sub_id").datestring.rank(method="first").astype(int)
    age = d.age.where((d.age >= 5) & (d.age <= 100))
    n_bad_age = int(d.age.notna().sum() - age.notna().sum())
    sex = d.Gender.map({"Female": "F", "Male": "M"}).fillna("")
    inj = d.InjDefn.fillna("")

    def pop(r):
        if r.InjDefn == "No injury":
            return "no injury"
        if isinstance(r.InjDefn, str):
            j = r.InjJoint if isinstance(r.InjJoint, str) and r.InjJoint.lower() not in ("no injury", "null", "n/a") else "joint not stated"
            return f"injured ({j.lower()})"
        return "injury status not stated"

    popu = d.apply(pop, axis=1)
    hand = ""  # DominantLeg is leg dominance, not handedness: not mapped
    out = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment="", group=np.where(d.Level.isna(), "", d.Level.str.lower()),
        subject=[f"{DS}_{int(s)}" for s in d.sub_id], subject_excluded_by_authors=False, age=age, sex=sex, handedness=hand,
        population=popu, setting="clinic", device="Motorized treadmill; Vicon motion capture", session=d.session,
        block="walking", trial=d.session, phase="other", time_s=np.nan, task="walking", unit="recording",
        condition="self-selected walking speed", pert_type="none", pert_value=0.0, pert_unit="", feedback="normal",
        excluded_by_authors=False, walking_speed_mps=d.speed_w))
    # trial = order of the recording within the person (same as session: one recording per session)
    ck.internal("2088 walking recordings, 1686 people", (n0, d.sub_id.nunique()) == (2088, 1686), f"{n0}, {d.sub_id.nunique()}")
    ck.internal("every recording has a walking speed", d.speed_w.notna().all())
    ck.internal("recordings of one person are numbered 1..k", (d.groupby("sub_id").session.max() == d.groupby("sub_id").size()).all())
    ck.published("people with walking data (396 walking only + 1290 both)", d.sub_id.nunique(), 396 + 1290, 0,
                 "Data Records: 396 subjects have only walking data, 112 only running, 1290 both")
    ck.published("mean walking speed lies within the Table 1 group means 1.11-1.30 m/s (1 = yes)",
                 float(1.11 <= d.speed_w.mean() <= 1.30), 1, 0, "Table 1: group mean speeds 1.11 to 1.30 m/s")
    n_out = int(((d.speed_w < 0.4) | (d.speed_w > 1.9)).sum())
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = np.nan
    meta = dict(
        study=DS,
        paper=dict(authors="Ferber R, Brett A, Fukuchi RK, Hettinga B, Osis ST", year=2024,
                   title="A Biomechanical Dataset of 1,798 Healthy and Injured Subjects During Treadmill Walking and Running",
                   venue="Sci Data 11(1):1232", doi="10.1038/s41597-024-04011-7"),
        lab="Ferber (University of Calgary, Running Injury Clinic)",
        data_url="https://doi.org/10.25452/figshare.plus.24255795",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Only walk_data_meta.csv was converted (the marker data are 22.7 GB). Dropped the subject number, the date "
                 "and the file name, replaced the subject number with a participant code, numbered each person's recordings by "
                 "date order (session) and left out height, weight, race history and the free-text injury fields."),
        how=["One row is one walking recording (about 60 s on a treadmill at a self-selected speed) of one person at one "
             "clinic visit; walking_speed_mps is the authors' calculated walking speed in m/s. sex is the recorded gender, "
             "population is no injury or the injured joint (InjDefn/InjJoint), and group is the self-reported activity level "
             "(recreational or competitive).",
             "The 3D marker trajectories and the authors' derived gait variables are in ric_data.zip (22.7 GB) and are not "
             "included here.",
             f"{n_bad_age} age values outside 5-100 years (255 and similar) are left empty; {n_out} recordings have speeds outside the "
             "0.4-1.9 m/s protocol range and are kept as shared."],
        unit_note="recording (one walking bout of about 60 s)",
        differences=["The paper reports group means only (Table 1), so the one overall check is that the mean speed lies in "
                     "their range; the paper's own abstract-level count of people with walking data (1,686) is reproduced exactly.",
                     "The article text is CC BY-NC-ND 4.0 but the Figshare+ record and the README state CC BY 4.0 for the data, "
                     "which is the licence used here."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        print(("  P " if c["kind"] == "published" else "  I ") + c["what"][:80], c.get("computed", ""), c.get("reported", ""), c.get("detail", "")[:150])


if __name__ == "__main__":
    main()
