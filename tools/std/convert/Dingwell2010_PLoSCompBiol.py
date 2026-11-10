"""Dingwell, John & Cusumano (2010) PLoS Comput Biol 6(7):e1000856: treadmill walking at five speeds, stride length, time
and speed for every stride of 17 young and 17 older healthy adults, one row per stride.

Raw files: raw_extra/Dingwell2010_PLoSCompBiol/{Young,Older}_Healthy_CSV_Files.zip and the two README_for_*_CSV_Files.txt
(Zenodo record 5015731, a copy of the Dryad deposit 10.5061/dryad.sk55m, CC0). The MATLAB zips hold the same numbers.
Each subject's CSV has 10 trials x 4 columns (Ln, Tn, Sn, Dn) under trial names SPD<1-5>TR<1-2>; a trial that is missing or
was discarded holds a single row of 9999s (README: do not use). The paper's Methods give the five speeds as 80, 90, 100, 110
and 120% of the preferred walking speed; the README names them 'PWS and PWS +/-10% and +/-20%' without saying SPD1 is the
slowest, which the stride speeds below confirm.
Run from the site folder: python3 -I tools/std/convert/Dingwell2010_PLoSCompBiol.py"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Dingwell2010_PLoSCompBiol"
MODULE = "walking"
PCT = {1: 80, 2: 90, 3: 100, 4: 110, 5: 120}
GROUPS = {"YH": ("Young Healthy", "healthy young adults"), "OH": ("Older Healthy", "healthy older adults")}


def main():
    raw = stdlib.raw_dir(DS)
    ck = stdlib.Checks()
    frames, missing, per_trial = [], [], []
    for pre, zname in (("YH", "Young_Healthy_CSV_Files.zip"), ("OH", "Older_Healthy_CSV_Files.zip")):
        zf = zipfile.ZipFile(raw / zname)
        for fn in sorted(n for n in zf.namelist() if n.endswith(".csv")):
            sid = re.fullmatch(rf"({pre}\d\d)\.csv", fn).group(1)
            df0 = pd.read_csv(zf.open(fn), header=None, skiprows=2)
            names = pd.read_csv(zf.open(fn), header=None, nrows=1).iloc[0]
            assert df0.shape[1] == 40
            n = 0
            for k in range(10):
                tn = names.iloc[4 * k]
                m = re.fullmatch(r"SPD(\d)TR(\d)", tn)
                spd, tr = int(m.group(1)), int(m.group(2))
                X = df0.iloc[:, 4 * k:4 * k + 4].dropna(how="all").to_numpy(float)
                if (X == 9999).all():
                    assert len(X) == 1
                    missing.append((sid, tn))
                    continue
                assert not (X == 9999).any() and np.isfinite(X).all()
                per_trial.append((pre, sid, tn, len(X)))
                for i, (L, T, S, D) in enumerate(X):
                    n += 1
                    frames.append(dict(
                        dataset=DS, study=DS, experiment="", group=GROUPS[pre][0], subject=f"{DS}_{sid}",
                        subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population=GROUPS[pre][1],
                        setting="lab", device="Motorized treadmill; Vicon motion capture", session=1, block=tn,
                        trial=0, phase="other", time_s=np.nan, task="walking", unit="stride",
                        condition=f"{PCT[spd]}% of preferred walking speed", pert_type="none", pert_value=0.0,
                        pert_unit="", feedback="normal", excluded_by_authors=False,
                        walking_trial=(spd - 1) * 2 + tr, stride_time_s=T, stride_length_m=L, walking_speed_mps=S,
                        net_displacement_m=D, _spd=spd, _pre=pre))
    df = pd.DataFrame(frames)
    df["trial"] = df.groupby("subject", sort=False).cumcount() + 1
    pt = pd.DataFrame(per_trial, columns=["pre", "sid", "trial", "n"])

    # ---- internal checks
    nsub = df.groupby("_pre").subject.nunique().to_dict()
    ck.internal("17 young and 17 older participants (README)", nsub == {"OH": 17, "YH": 17}, str(nsub))
    ck.internal("every row has a finite length, time, speed and displacement and no 9999 is left",
                np.isfinite(df[["stride_time_s", "stride_length_m", "walking_speed_mps", "net_displacement_m"]]).all().all())
    err = (df.stride_length_m / df.stride_time_s - df.walking_speed_mps).abs()
    ck.internal("stride speed = stride length / stride time (README Col 3) on every stride, to 1e-6", err.max() < 1e-6,
                f"max |diff| {err.max():.2e}")
    ms = df.groupby(["_pre", "_spd"]).walking_speed_mps.mean().unstack()
    ck.internal("mean stride speed rises from SPD1 to SPD5 in both groups (SPD1 = 80% of preferred, SPD5 = 120%)",
                bool((ms.diff(axis=1).iloc[:, 1:] > 0).all().all()), ms.round(3).to_dict("index").__repr__())
    ck.internal("missing/discarded trials (a single row of 9999) are not rows: " + str(len(missing)) + " trials",
                len(pt) + len(missing) == 340, f"{len(pt)} kept + {len(missing)} missing = 340")

    # ---- published numbers (paper Methods; young group)
    y = pt[pt.pre == "YH"]
    ck.published("young participants", y.sid.nunique(), 17, 0, "Methods, Subjects")
    ck.published("young trials collected (17 x 10 = 170)", len(y) + sum(s.startswith("YH") for s, _ in missing), 170, 0, "Methods, Experimental protocol")
    ck.published("young trials retained (4 of 170 discarded: 166)", len(y), 166, 0, "Methods, Data analysis")
    ck.published("young trials discarded as % of 170 (2.35%)", 100 * sum(s.startswith("YH") for s, _ in missing) / 170, 2.35, 0.005, "Methods, Data analysis")
    ck.published("young mean strides per trial (272)", y.n.mean(), 272, 0.5, "Methods, Data analysis")
    ck.published("young SD of strides per trial (25)", y.n.std(ddof=1), 25, 0.5, "Methods, Data analysis")
    ck.published("young fewest strides in a trial (213)", y.n.min(), 213, 0, "Methods, Data analysis")
    ck.published("young most strides in a trial (334)", y.n.max(), 334, 0, "Methods, Data analysis")

    out = df.drop(columns=["_spd", "_pre"])
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = np.nan  # columns of the module this deposit does not fill
    meta = dict(
        study=DS,
        paper=dict(authors="Dingwell JB, John J, Cusumano JP", year=2010,
                   title="Do Humans Optimally Exploit Redundancy to Control Step Variability in Walking?",
                   venue="PLoS Comput Biol 6(7):e1000856", doi="10.1371/journal.pcbi.1000856"),
        lab="Dingwell (Penn State / UT Austin)",
        data_url="https://doi.org/10.5061/dryad.sk55m",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the CSV versions (the MATLAB files hold the same numbers) from the Zenodo copy of the Dryad deposit, "
                 "made one row per stride, replaced the file names YH01-OH17 with participant codes and dropped the "
                 "placeholder rows of 9999 that mark missing trials."),
        how=["One row is one stride of one participant's 5-minute treadmill trial, in file order: stride_length_m, "
             "stride_time_s, walking_speed_mps (stride length / stride time) and net_displacement_m (the authors' Ln, Tn, "
             "Sn and Dn; metres and seconds).",
             "Each participant walked two trials at each of five speeds, 80, 90, 100, 110 and 120% of their own preferred "
             "walking speed (condition); block is the authors' trial name SPD<speed>TR<repeat> and walking_trial numbers the "
             "trials 1-10 in the file's column order, which is not necessarily the order walked.",
             "The deposit shares no age, sex or preferred speed per participant (only group summaries in the README: young "
             "23.3 +- 2.66 years, 12 men and 5 women; older 72.3 +- 6.10 years, 11 men and 6 women), so those cells are "
             "empty. The README says that trials that were missing or discarded hold one row of 9999s and must not be used; "
             "they are not rows here (6 of 340 trials in all)."],
        unit_note="stride (one heel contact to the next of the same foot)",
        differences=["The paper's 272 +- 25 strides per trial (range 213-334) describes the 17 young participants, which "
                     "the young rows reproduce. The older group was recorded in the same study but belongs to the "
                     "deposit's other papers; its trial counts are not published in this paper."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        print(("  P " if c["kind"] == "published" else "  I ") + c["what"][:80], c.get("computed", ""), c.get("reported", ""), c.get("detail", "")[:150])


if __name__ == "__main__":
    main()
