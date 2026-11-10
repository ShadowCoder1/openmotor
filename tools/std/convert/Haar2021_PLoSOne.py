"""Haar, Sundar & Faisal (2021), PLoS ONE 16(1):e0245717: embodied virtual-reality pool, 10 people x 300 shots, one row per
shot in the 'game' module, with the 54 joint-angle velocity channels around each shot's peak as a CSV inside the ZIP.

Run from the site folder: python3 -I tools/std/convert/Haar2021_PLoSOne.py
Raw file (untrusted data, read only) in raw_extra/Haar2021_PLoSOne/poolVRdata.mat, from https://doi.org/10.6084/m9.figshare.13553606 (CC BY 4.0):
  vrBallErr (10 x 300), vrBallScs (10 x 300), jointVelocity (10 x 61 frames x 300 x 54), jointsLabels (54)
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Haar2021_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
m = sio.loadmat(RAW / "poolVRdata.mat", squeeze_me=True)
err, scs, vel = m["vrBallErr"], m["vrBallScs"], m["jointVelocity"]
labels = [str(x) for x in m["jointsLabels"]]
nS, nT = err.shape
checks.internal("shapes: 10 x 300 outcomes, 10 x 61 x 300 x 54 velocities", vel.shape == (10, 61, 300, 54) and scs.shape == err.shape == (10, 300) and len(labels) == 54)
checks.internal("success is 0/1", set(np.unique(scs)) <= {0, 1})
tvec = np.round(np.arange(-30, 31) / 60.0, 6)
CH = ";".join(labels)

rows, files = [], []
n_nan_err = n_empty_vel = 0
for s in range(nS):
    for t in range(nT):
        e = err[s, t]
        n_nan_err += bool(np.isnan(e))
        v = vel[s, :, t, :]
        sf = ""
        if not np.isnan(v).all():
            sf = f"samples/S{s + 1:02d}_trial{t + 1:03d}.csv"
            df = pd.DataFrame(v, columns=labels)
            df.insert(0, "time_s", tvec)
            files.append((sf, df.to_csv(index=False, lineterminator="\r\n", float_format="%.8g").encode()))
        else:
            n_empty_vel += 1
        rows.append(dict(
            dataset=ID, study=ID, experiment="VR pool", group="", subject=f"P{s + 1:02d}", subject_excluded_by_authors=False,
            age=None, sex="", handedness="right", population="healthy", setting="lab",
            device="Xsens inertial suit (60 Hz) with a virtual-reality pool task", session=1,
            block=f"set {t // 50 + 1}, block {t // 25 + 1}", trial=t + 1, phase="learning", time_s=None,
            task="sports and games", unit="trial", condition="", pert_type="none", pert_value=0, pert_unit="",
            feedback="normal", excluded_by_authors=False, game="Pool (virtual reality)", attempt=t + 1,
            success=bool(scs[s, t]), outcome_error=None if np.isnan(e) else float(e), outcome_error_unit="deg",
            samples_file=sf, channels=CH if sf else "", channel_units="unspecified (joint-angle velocity)" if sf else "",
            sample_rate_hz=60 if sf else None))
checks.internal("315 shots have no angular error in the deposit", n_nan_err == 315, str(n_nan_err))
checks.internal("samples file only for shots with velocity values", len(files) + n_empty_vel == nS * nT, f"{len(files)} files, {n_empty_vel} empty")

checks.published("participants", nS, 10, 0, "Methods (10 right-handed volunteers)")
checks.published("trials per participant", nT, 300, 0, "Methods (300 repeated trials)")
checks.published("trials in blocks of 25 (12 blocks)", nT // 25, 12, 0, "Methods (6 sets of 50, each two blocks of 25)")
checks.published("joint angles x degrees of freedom: 3 per joint", len(labels) / 3, 18, 0, "Methods (3 DoF joint angles, Xsens)")

b = np.nanmean(np.abs(err).reshape(10, 12, 25), axis=2)
d = b[:, 0] - b[:, -1]
sb = scs.astype(float).reshape(10, 12, 25).mean(2)
ds = sb[:, -1] - sb[:, 0]
differences = [
    f"The paper reports that mean directional error fell by 8.3 +/- 2.47 degrees and success rose by 6.2 +/- 1.78 percent over the session. "
    f"Taking block 1 (shots 1-25) against block 12 (shots 276-300) of the deposit, absolute error falls by {d.mean():.2f} degrees "
    f"(SD of the paired differences {d.std(ddof=1):.2f}) and success rises by {100 * ds.mean():.1f} points (SEM {100 * ds.std(ddof=1) / np.sqrt(10):.2f}). "
    "The authors do not say which blocks or which averaging they used, so these were not used as checks.",
    "The paper describes angular error as an absolute difference, but the deposit's vrBallErr has negative values (883 of 2,685), "
    "so it is kept as the authors shared it (signed; the sign convention is not stated). 315 shots have no error value and are kept empty.",
    "The deposit's jointVelocity has 61 frames per shot, described by the authors as 60 Hz data around the peak velocity; time_s is given as "
    "-0.5 to +0.5 s relative to that peak (inferred from 61 frames at 60 Hz). Units of velocity are not stated anywhere; they are left "
    "unspecified. 7,374 values are missing (NaN) and are written empty.",
    "Age (24 +/- 2 years) and sex (4 women, 6 men) are given only as group figures in the paper, not per participant, so age and sex are empty.",
]
meta = dict(
    paper=dict(authors="Haar S, Sundar G, Faisal AA", year=2021, title="Embodied virtual reality for the study of real-world motor learning",
               venue="PLoS ONE 16(1):e0245717", doi="10.1371/journal.pone.0245717"),
    lab="Faisal (Imperial College London)", data_url="https://doi.org/10.6084/m9.figshare.13553606", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Rows flattened to one per shot in the authors' order; the 61 x 54 velocity block of each shot written to its own CSV; "
            "participant codes replaced.",
    how=["One row is one pool shot in virtual reality: 300 shots per person, in order, in 6 sets of 50 (the paper analyses blocks of 25).",
         "success is whether the target ball fell in the pocket (authors' vrBallScs); outcome_error is the angular error of the target ball "
         "from the centre of the pocket in degrees (authors' vrBallErr, signed as shared).",
         "samples_file holds 61 frames at 60 Hz around the shot's peak arm velocity: one column per joint-angle velocity "
         "(18 joints x flexion, abduction, rotation), in the authors' order and units."],
    unit_note="shot", differences=differences)
stdlib.release(ID, "game", rows, meta, checks, extra_files=files)
