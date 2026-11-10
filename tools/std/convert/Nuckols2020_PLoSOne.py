"""Nuckols et al. (2020) PLoS ONE: joint-level mechanics of walking (1.25 m/s) and running (2.25 m/s) up and downhill, 8 adults.

Raw files: raw_extra/Nuckols2020_PLoSOne/Nuckols_PLoS_UpDown_Summary_Repo.mat (Zenodo 3986714, the Dryad deposit's copy, CC0)
and Nuckols_PLoS_MatStructureInfo.txt (read as text). Summary.<gait>.<grade>.<joint>.<metric>.In is subjects x 101 points of the
stride (heel strike to heel strike of the right foot, time-normalized by the authors); Pow.InAvgPwr is subjects x 9 work and
power values. One row is one participant in one gait and grade; the 101-point curves of 3 joints x 4 metrics are its samples file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Nuckols2020_PLoSOne"
MODULE = "timeseries"
GRADES = {"walk": [("level", 0.0), ("up15", 8.53), ("up10", 5.71), ("down10", -5.71), ("down15", -8.53)],
          "run": [("level", 0.0), ("up10", 5.71), ("up5", 2.86), ("down5", -2.86), ("down10", -5.71)]}
JOINTS = ["ANK", "KNEE", "HIP"]
METRICS = [("Ang", "deg"), ("Mom", "N*m/kg"), ("Pow", "W/kg"), ("Vel", "rad/s")]
WORK = [("PosWork", "J/kg"), ("NegWork", "J/kg"), ("NetWork", "J/kg"), ("TotWork", "J/kg"),
        ("PosPow", "W/kg"), ("NegPow", "W/kg"), ("NetPow", "W/kg"), ("TotPow", "W/kg")]


def main():
    S = sio.loadmat(stdlib.raw_dir(DS) / "Nuckols_PLoS_UpDown_Summary_Repo.mat")["Summary"]
    rows, files, ck = [], [], stdlib.Checks()
    missing = []
    pos = {}
    n_subj = None
    # subjects outermost so that the codes assigned by release() (order of first appearance) are S0001..S0008 in array order
    n_subj = S["walk"][0, 0]["level"][0, 0]["ANK"][0, 0]["Ang"][0, 0]["In"][0, 0].shape[0]
    rec = 0
    for si in range(n_subj):
        code = f"{DS}_S{si + 1:04d}"
        rec_i = 0
        for gait, grades in GRADES.items():
            for gname, deg in grades:
                G = S[gait][0, 0][gname][0, 0]
                cols, names, units = [], [], []
                for j in JOINTS:
                    for m, u in METRICS:
                        cols.append(G[j][0, 0][m][0, 0]["In"][0, 0][si, :])
                        names.append(f"{j}_{m}")
                        units.append(u)
                curves = np.array(cols).T
                assert curves.shape == (101, 12)
                if np.isnan(curves).all():  # the authors left this participant's condition empty
                    missing.append((si + 1, gait, gname))
                    continue
                assert not np.isnan(curves).any()
                info = []
                for j in JOINTS:
                    a = G[j][0, 0]["Pow"][0, 0]["InAvgPwr"][0, 0][si, :]
                    for (w, _), v in zip(WORK, a[:8]):
                        info.append(f"{j}_{w}={float(v):.8g}")
                    pos[(gait, gname, j, si)] = float(a[4])
                rec_i += 1
                fname = f"samples/{code}_{gait}_{gname}.csv"
                df = pd.DataFrame(curves, columns=names)
                df.insert(0, "stride_percent", np.arange(101))
                files.append((fname, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n").encode()))
                rows.append(dict(
                    dataset=DS, study="Nuckols 2020 PLoS ONE", experiment="walking and running on grades", group=None,
                    subject=f"S{si + 1}", person=f"S{si + 1}", subject_excluded_by_authors=False, age=None, sex=None,
                    handedness=None, population="healthy", setting="lab", device="treadmill, 8-camera motion capture",
                    session=1, block=f"{gait} {gname}", trial=rec_i, phase="other", time_s=None, task="walking",
                    unit="recording", condition=f"{gait}, grade {deg:+.2f} deg", pert_type="none" if deg == 0 else "treadmill_grade",
                    pert_value=deg, pert_unit="deg", feedback="normal", excluded_by_authors=False,
                    recording=rec_i, samples_file=fname, signal_kind="joint angle, moment, power and angular velocity (right limb), one normalized stride",
                    channels=";".join(names), channel_units=";".join(units), sample_rate_hz=None, n_samples=101, duration_s=None,
                    events=None, processing="inverse dynamics (Visual3D), stride time-normalized to 101 points by the authors; averaged over strides",
                    trial_info=f"gait={gait};grade_label={gname};" + ";".join(info)))
    ck.internal("8 participants x 10 gait/grade conditions minus empty ones", len(rows) + len(missing) == 80, f"{len(rows)} rows, empty: {missing}")
    ck.internal("each samples file 101 x 13", all(len(b.splitlines()) == 102 for _, b in files))

    def total(g, gr):
        return np.nanmean([sum(pos[(g, gr, j, s)] for j in JOINTS) for s in range(n_subj)])

    def pct(g, gr, j):
        return np.nanmean([100 * pos[(g, gr, j, s)] / sum(pos[(g, gr, k, s)] for k in JOINTS) for s in range(n_subj)])
    ck.published("participants", n_subj, 8, 0, "Methods, Participants")
    ck.published("level walking average positive limb power (W/kg)", total("walk", "level"), 1.02, 0.005, "Results, Positive power")
    ck.published("level running average positive limb power (W/kg)", total("run", "level"), 3.66, 0.005, "Results, Positive power")
    ck.published("8.53 deg walking positive limb power (W/kg)", total("walk", "up15"), 2.60, 0.005, "Results, Positive power")
    for j, v in zip(JOINTS, (44, 19, 37)):
        ck.published(f"level walking {j} share of positive power (%)", pct("walk", "level", j), v, 0.5, "Table 2")
    for j, v in zip(JOINTS, (29, 19, 52)):
        ck.published(f"8.53 deg walking {j} share of positive power (%)", pct("walk", "up15", j), v, 0.5, "Table 2")
    meta = dict(
        study=DS,
        paper=dict(authors="Nuckols RW, Takahashi KZ, Farris DJ, Mizrachi S, Riemer R, Sawicki GS",
                   year=2020, title="Mechanics of walking and running up and downhill: A joint-level perspective to guide design of lower-limb exoskeletons",
                   venue="PLoS ONE 15(8):e0231996", doi="10.1371/journal.pone.0231996"),
        lab="Sawicki (Georgia Tech)", data_url="https://doi.org/10.5061/dryad.ns1rn8pqc (copy: https://zenodo.org/records/3986714)",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Each participant-gait-grade became one recording row; the 12 joint curves (ankle, knee, hip x angle, moment, power, "
                 "velocity; 101 points per stride) went into one samples CSV per recording with a stride_percent column (0-100) in "
                 "place of time_s, the work and power values went into trial_info, participants were coded in array order, and the "
                 "grades the authors name as percent slope (15, 10, 5) are given in degrees as the paper does."),
        how=["One row is one participant walking (1.25 m/s) or running (2.25 m/s) on one treadmill grade; 8 adults, 5 grades per gait "
             "(walking -8.53, -5.71, 0, 5.71, 8.53 deg; running -5.71, -2.86, 0, 2.86, 5.71 deg). The samples file holds the "
             "mean-over-strides curves of one stride (heel strike to heel strike of the right foot) for the ankle, knee and hip: "
             "angle, moment, power and angular velocity, 101 points each. trial_info holds, per joint, the stride-average positive, "
             "negative, net and total work and power from the authors' InAvgPwr matrix.",
             "Units: power W/kg and work J/kg are stated in the paper. The structure file does not state the units of angle, moment "
             "or velocity or any sign convention; deg, N*m/kg and rad/s are inferred from the value ranges (ankle angle -90 to -48, "
             "moments 0-2.6, velocities within +-10) and the paper's use of Visual3D, and signs are as the authors shared them.",
             "Participants have no identifiers in the file: they are numbered by array position, assumed to be the same participant "
             "in every grade and gait. Column 9 of InAvgPwr ('StrideAvg?' in the authors' note) is not documented and is left out."],
        unit_note="one participant on one gait and grade",
        differences=[f"The authors left {len(missing)} participant-condition entries empty (all NaN); no row was written for them: {missing} (participant number in array order, gait, grade).", "Age, sex and mass are given only as group means in the paper, so the participant columns are empty."])
    stdlib.release(DS, MODULE, rows, meta, ck, extra_files=files)


if __name__ == "__main__":
    main()
