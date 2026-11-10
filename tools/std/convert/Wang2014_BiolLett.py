"""Wang & Srinivasan (2014) Biol Lett 10(9):20140405: 10 adults walking on a treadmill for 5 minutes at each of 3 speeds, with
pelvis and heel marker positions at 100 Hz (Vicon). The deposit has only the position traces, so the format is the timeseries
module, one row per recording.

Raw files: raw_extra/Wang2014_BiolLett/WangSrinivasanBiologyLetters_DryadDataUpload.zip (Subject<1-10>Trial<1-3>.txt) and its
README (Zenodo record 4999877, a copy of the Dryad deposit 10.5061/dryad.5kh00, CC0).
Each file: first line names the 9 columns (Torso.X Torso.Y Torso.Z Lfoot.X ... Rfoot.Z), then one line per 0.01 s sample, in
metres (X sideways, Y forward, Z vertical, lab frame).
Run from the site folder: python3 -I tools/std/convert/Wang2014_BiolLett.py"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wang2014_BiolLett"
MODULE = "timeseries"
P = "WangSrinivasanBiologyLetters_DryadDataUpload/"
CH = ["Torso.X", "Torso.Y", "Torso.Z", "Lfoot.X", "Lfoot.Y", "Lfoot.Z", "Rfoot.X", "Rfoot.Y", "Rfoot.Z"]
SPEEDS = {True: (1.0, 1.2, 1.4), False: (0.98, 1.25, 1.43)}  # subjects 1-5, subjects 6-10 (README)


def main():
    ck = stdlib.Checks()
    zf = zipfile.ZipFile(stdlib.raw_dir(DS) / "WangSrinivasanBiologyLetters_DryadDataUpload.zip")
    rows, extra, strides, dts = [], [], [], []
    for s in range(1, 11):
        for t in (1, 2, 3):
            lines = zf.read(P + f"Subject{s}Trial{t}.txt").decode("ascii").splitlines()
            assert lines[0].split() == CH, lines[0]
            A = np.array([[float(x) for x in l.split()] for l in lines[1:]])
            assert A.shape[1] == 9 and np.isfinite(A).all()
            n = len(A)
            tcol = np.round(np.arange(n) * 0.01, 2)
            df = pd.DataFrame(A, columns=CH)
            df.insert(0, "time_s", tcol)
            buf = io.StringIO()
            df.to_csv(buf, index=False, float_format="%.8g", lineterminator="\r\n")
            arc = f"samples/Subject{s:02d}Trial{t}.csv"
            extra.append((arc, buf.getvalue().encode("ascii")))
            # heel strike = the forward-most position of the heel marker (paper Methods); one per stride
            for col in (4, 7):
                pk, _ = find_peaks(A[:, col], distance=60)
                if col == 4:
                    strides.append((s, t, len(pk)))
            v = SPEEDS[s <= 5][t - 1]
            rows.append(dict(
                dataset=DS, study=DS, experiment="", group="", subject=f"S{s:02d}", subject_excluded_by_authors=False,
                age=np.nan, sex="", handedness="", population="healthy adults", setting="lab",
                device="Motorized treadmill (" + ("0.92 m" if s <= 5 else "0.51 m") + " wide); Vicon T20 motion capture, 100 Hz",
                session=1, block="treadmill walking", trial=t, phase="test", time_s=np.nan, task="walking", unit="recording",
                condition=f"treadmill {v:g} m/s", pert_type="none", pert_value=0.0, pert_unit="", feedback="normal",
                excluded_by_authors=False, recording=t, samples_file=arc,
                signal_kind="3D marker positions (pelvis point and both heel markers)", channels=";".join(CH),
                channel_units=";".join(["m"] * 9), sample_rate_hz=100, n_samples=n, duration_s=round(n * 0.01, 2),
                events="", processing="none stated; position traces as shared (pelvis point = weighted sum of 3 markers)"))
    out = pd.DataFrame(rows)
    names_ = [c[1] for c in stdlib.columns_for(MODULE)]
    for c in names_:
        if c not in out.columns:
            out[c] = np.nan
    out = out[names_]
    st = pd.DataFrame(strides, columns=["s", "t", "n"])

    ck.internal("10 participants x 3 trials = 30 recordings, each with the 9 named columns and no missing value",
                len(out) == 30 and out.subject.nunique() == 10)
    ck.internal("every sample step is 0.01 s (README, 100 Hz) and every trial lasts about 5 minutes except one shorter trial",
                bool(((out.duration_s > 270) & (out.duration_s < 310)).sum() == 29), f"{out.duration_s.min()}-{out.duration_s.max()} s")
    ck.published("participants", out.subject.nunique(), 10, 0, "Methods, Subjects")
    ck.published("recordings (10 participants x 3 speeds)", len(out), 30, 0, "Methods")
    ck.published("trial length, median over recordings (5 min = 300 s)", out.duration_s.median(), 300, 1, "Methods (5 minutes at each speed)")
    ck.published("sampling rate (Hz)", out.sample_rate_hz.median(), 100, 0, "Methods / README")
    ck.published("strides per trial, mean (about 265; heel-marker forward peaks, left foot)", st.n.mean(), 265, 4, "Methods, data collection")
    ck.published("strides per trial, SD (36)", st.n.std(ddof=1), 36, 4, "Methods, data collection")

    meta = dict(
        study=DS,
        paper=dict(authors="Wang Y, Srinivasan M", year=2014,
                   title="Stepping in the direction of the fall: the next foot placement can be predicted from current upper body state in steady-state walking",
                   venue="Biol Lett 10(9):20140405", doi="10.1098/rsbl.2014.0405"),
        lab="Srinivasan (Ohio State University)",
        data_url="https://doi.org/10.5061/dryad.5kh00",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the 30 whitespace-separated text files from the Zenodo copy of the Dryad deposit, wrote each as a "
                 "comma-separated CSV with a time_s column (sample number x 0.01 s) and the authors' values unchanged, and "
                 "replaced the subject numbers with codes. No age, sex or other per-participant values are shared (README gives "
                 "group means only)."),
        how=["One row is one treadmill trial of one participant (about 5 minutes); its samples file has time_s and nine "
             "channels at 100 Hz: the pelvis point (weighted sum of three pelvis markers, near the centre of mass) and the "
             "left and right heel markers, each as X (sideways), Y (forward) and Z (vertical) in the lab frame, in metres.",
             "Each participant walked three trials at increasing treadmill speeds: 1.0, 1.2 and 1.4 m/s for participants "
             "1-5 and 0.98, 1.25 and 1.43 m/s for participants 6-10 (README); the treadmill was 0.92 m wide for the first "
             "five and 0.51 m for the others (paper).",
             "No gait events are shared, so events is empty; the strides-per-trial check counts forward peaks of the left "
             "heel marker (heel strike, per the paper) in a script of ours, not in the authors' data."],
        unit_note="recording (one 5-minute treadmill trial)",
        differences=["The paper says about 265 strides per speed (SD 36); counting heel-marker forward peaks gives a mean close to that and a similar SD. One recording (participant 9, trial 1) is only 134 s long."],
    )
    stdlib.release(DS, MODULE, out, meta, ck, extra_files=extra)
    for c in ck.items:
        print(("  P " if c["kind"] == "published" else "  I ") + c["what"][:80], c.get("computed", ""), c.get("reported", ""), c.get("detail", "")[:150])


if __name__ == "__main__":
    main()
