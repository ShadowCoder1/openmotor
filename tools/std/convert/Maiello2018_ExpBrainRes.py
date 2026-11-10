"""Maiello, Kwon & Bex (2018) Exp Brain Res 236(3):691-709: three-dimensional binocular eye-hand coordination in normal vision and
with simulated visual impairment (Gaussian blur of one or both eyes' images).
5 observers; E1 horizontal pursuit of a target moved by the observer's finger, E2 replay of the E1 target motion with the hand still,
E3 and E4 the same in depth (vergence). Deposit: Zenodo 1100971 (CC BY 4.0), Data_and_Analyses.zip: one MATLAB workspace file per
observer and experiment (S00E1.mat ...) plus the authors' analysis scripts and the experiment scripts (ExperimentalScripts.zip), which were
read as text only. The EyeLink .edf/.asc raw files are not converted (binary eye-tracker format, same recordings as the .mat gaze).
Nothing from the deposit is executed.
Run from the site folder: python3 -I tools/std/convert/Maiello2018_ExpBrainRes.py"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Maiello2018_ExpBrainRes"
MODULE = "timeseries"
P = "Data_and_Analyses/Data/"
BLUR = {1: 0.0, 2: 6.25, 3: 12.5, 4: 25.0, 5: 50.0, 6: 100.0, 7: 200.0}      # arcmin (Methods: Stimuli)
EXP = {1: ("horizontal pursuit, finger", "finger", "horizontal"), 2: ("horizontal pursuit, replay", "replay", "horizontal"),
       3: ("vergence pursuit, finger", "finger", "vergence"), 4: ("vergence pursuit, replay", "replay", "vergence")}
z = zipfile.ZipFile(stdlib.raw_dir(ID) / "data.zip")
subs = sorted({n.split("/")[-1][:3] for n in z.namelist() if n.startswith(P) and n.endswith(".mat")})
assert subs == ["S00", "S01", "S02", "S03", "S04"], subs

rows, extra = [], []
durations = {1: {}, 2: {}, 3: {}, 4: {}}
dts, n_nan_gaze = [], 0
cond_counts = {}
dim_ok = True
for si, sub in enumerate(subs, start=1):
    code = f"{ID}_S{si:04d}"
    trial = 0
    for e in (1, 2, 3, 4):
        m = sio.loadmat(io.BytesIO(z.read(f"{P}{sub}E{e}.mat")))
        order = m["Order"].ravel().astype(int)           # order[p-1] = condition index shown at presentation position p
        pl, pr = m["pyramydL"].ravel().astype(int), m["pyramydR"].ravel().astype(int)
        n = len(order)
        assert n == (70 if e <= 2 else 60) and sorted(order) == list(range(1, n + 1))
        drift = m["CenterEyeDrift"]
        label, track, kind = EXP[e]
        for p in range(1, n + 1):
            k = order[p - 1]                              # 1-based index into the cell arrays
            G = np.asarray(m["Gaze"][0][k - 1], float)
            F = np.asarray(m["Finger"][0][k - 1], float)
            S = np.asarray(m["TrueStimLoc"][0][k - 1], float).ravel()
            T = np.asarray(m["Time"][0][k - 1], float).ravel()
            if not (G.shape[0] == F.shape[0] == len(S) == len(T) and G.shape[1] == 3 and F.shape[1] == 3):
                dim_ok = False
            t = T - T[0]
            dts.append(float(np.median(np.diff(T))))
            durations[e].setdefault(sub, []).append(float(t[-1]))
            n_nan_gaze += int(np.isnan(G).any(axis=1).sum())
            lvl_l, lvl_r = -pl[k - 1], -pr[k - 1]
            bino = lvl_l == lvl_r
            blur = BLUR[lvl_l]
            cname = f"{'binocular' if bino else 'monocular (left eye)'} blur {blur:g} arcmin" if blur > 0 else "no blur"
            cond_counts[(sub, e, bino, blur)] = cond_counts.get((sub, e, bino, blur), 0) + 1
            trial += 1
            body = pd.DataFrame({"time_s": t, "gaze_x_cm": G[:, 0], "gaze_y_cm": G[:, 1], "gaze_depth_cm": G[:, 2], "finger_x_cm": F[:, 0],
                                 "finger_y_cm": F[:, 1], "finger_z_cm": F[:, 2], "target_cm": S})
            buf = io.StringIO()
            body.to_csv(buf, index=False, float_format="%.9g", lineterminator="\r\n")
            fname = f"samples/{code}_E{e}_trial{p:03d}.csv"
            extra.append((fname, buf.getvalue().encode()))
            rate = 1.0 / np.mean(np.diff(t))
            rows.append(dict(
                dataset=ID, study="Maiello et al. 2018 Exp Brain Res", experiment=label, group="", subject=code, person=code,
                subject_excluded_by_authors=False, age=np.nan, sex="", handedness="right", population="healthy adults (normal or corrected vision)",
                setting="lab", device="EyeLink 1000 (binocular), Leap Motion finger tracker, stereo display with shutter glasses", session=e,
                block=label, trial=trial, phase="other", time_s=np.nan, task="tracking", unit="recording", condition=cname,
                pert_type="gaussian_blur" if blur > 0 else "none", pert_value=blur, pert_unit="arcmin (both eyes if binocular, left eye only if monocular)",
                feedback="normal", excluded_by_authors=False, recording=trial, samples_file=fname,
                signal_kind="binocular gaze, index finger and target position",
                channels="gaze_x_cm;gaze_y_cm;gaze_depth_cm;finger_x_cm;finger_y_cm;finger_z_cm;target_cm", channel_units="cm;cm;cm;cm;cm;cm;cm",
                sample_rate_hz=round(float(rate), 3), n_samples=len(t), duration_s=float(t[-1]), events="",
                processing="gaze: binocular fixation point from the two eyes' EyeLink positions, drift-corrected per trial with the median of the first 250 ms (CenterEyeDrift); finger: Savitzky-Golay smoothed over 19 samples with gaps filled (authors, online)",
                trial_info=f"tracking={track};movement={kind};blur_level_left={lvl_l};blur_level_right={lvl_r};presentation_position={p};"
                           f"eye_drift_px_xL={drift[k - 1, 0]:.6g};eye_drift_px_xR={drift[k - 1, 1]:.6g};eye_drift_px_yL={drift[k - 1, 2]:.6g};eye_drift_px_yR={drift[k - 1, 3]:.6g}"))
df = pd.DataFrame(rows)

ck = stdlib.Checks()
ck.published("observers", df.person.nunique(), 5, 0, "Methods, Participants: 5 observers")
ck.published("trials per observer (70 + 70 + 60 + 60)", len(df) / 5, 260, 0, "Methods, Procedure: sessions of 70, 70, 60 and 60 trials")
ck.published("horizontal pursuit blur levels", len({v for (_, e, b, v) in cond_counts if e == 1}), 7, 0, "Methods, Stimuli: 0 to 200 arcmin, 7 levels")
ck.published("vergence blur levels", len({v for (_, e, b, v) in cond_counts if e == 3}), 6, 0, "Methods, Stimuli: first six levels, up to 100 arcmin")
ck.published("trials per blur level x binocular/monocular condition per observer and experiment (median; blur > 0)", np.median([v for (_, _, _, b), v in cond_counts.items() if b > 0]), 5, 0, "Methods, Procedure: 5 trials per condition")
ck.published("largest blur (arcmin)", df.pert_value.max(), 200, 0, "Methods, Stimuli: 200 arcmin")
for e in (1, 3):
    tt = np.mean([np.mean(v) for v in durations[e].values()])
    print("mean duration across observers, E%d" % e, tt)
ck.internal("gaze, finger, target and time arrays have the same length in every trial", dim_ok)
ck.internal("recordings = 5 x 260", len(df) == 1300 and df.groupby("person").size().eq(260).all())
ck.internal("blur conditions balanced: 5 trials each (binocular blur>0, monocular blur>0, and the no-blur condition counted twice) per experiment",
            all((v == 5 and b > 0) or (v == 10 and b == 0) for (_, _, _, b), v in cond_counts.items()), str(sorted(set(cond_counts.values()))))
ck.internal("median sample interval is about 16.7 ms (60 samples/s)", abs(1 / np.median(dts) - 60) < 1, f"{1 / np.median(dts):.2f} Hz")
meta = dict(
    paper=dict(authors="Maiello G, Kwon M, Bex PJ", year=2018, title="Three-dimensional binocular eye-hand coordination in normal vision and with simulated visual impairment",
               venue="Exp Brain Res 236(3):691-709", doi="10.1007/s00221-017-5160-8"),
    lab="Bex (Northeastern University)", data_url="https://doi.org/10.5281/zenodo.1100971", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Per observer and experiment, the cell arrays Gaze, Finger, TrueStimLoc and Time of the authors' MATLAB workspace file were written as one CSV per trial "
             "(time_s = the authors' clock minus its first value; values unchanged). Rows follow the order in which the trials were presented (the authors' Order variable); "
             "the arrays are stored by condition index. Blur level numbers were turned into arcmin with the paper's list. The authors' EyeLink .edf/.asc raw files and analysis outputs were not carried."),
    how=["One row is one trial: a Gabor target moved along a bar (E1, E2: left-right within about +/-4 cm = 4 deg, viewed from 57 cm) or in depth (E3, E4: from the screen toward the observer up to 8 cm) while the observer tracked it with gaze. In E1 and E3 the target followed the observer's index finger (tracked by a Leap Motion); "
         "in E2 and E4 it replayed the finger motion recorded in the matching E1 or E3 trial while the hand stayed still, so finger_* in E2 and E4 repeats E1's and E3's.",
         "Gaussian blur of 0 to 200 arcmin (E1, E2) or 0 to 100 arcmin (E3, E4) was applied to both eyes' images (binocular) or only the left eye's (monocular); condition says which, pert_value is the blur size, and trial_info has the level numbers. "
         "The no-blur condition appears under both binocular and monocular codings in the authors' design (10 trials per experiment), which are the same stimulus.",
         "Columns: gaze x, y and depth (cm; depth relative to the screen, negative toward the observer) from the two eyes' EyeLink positions, finger x, y, z (cm, Leap Motion coordinates transformed by the authors' calibration) and the target's location (cm: horizontal position in E1/E2, distance from the observer in E3/E4). "
         "Missing finger or gaze samples are empty (NaN in the source). Samples are one per display frame, about 60 per second."],
    unit_note="trial (about 3 s of binocular gaze, finger and target position)",
    differences=["The paper states gaze and finger data were sampled at 120 Hz; the deposited arrays have one sample per about 16.7 ms (60 per second), so sample_rate_hz is the rate measured from the clock.",
                 "Only the five observers' MATLAB workspaces are converted; the deposit's binary EyeLink .edf files (and one .asc) hold the same recordings and are not read. No age or sex per observer is in the deposit (the paper reports a mean age of 29 +/- 6 years and 3 men)."])
stdlib.release(ID, MODULE, df, meta, ck, extra_files=extra)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))
