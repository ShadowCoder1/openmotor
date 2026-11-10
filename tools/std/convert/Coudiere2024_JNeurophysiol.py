"""Coudiere & Danion (2024) J Neurophysiol 131(4):652-667: eye-hand coordination from discrete to continuous hand movements.
28 participants tracked an unpredictable target whose position was updated at 1.5-240 Hz with eyes and joystick (EYE-HAND),
the joystick with the eyes fixed (HAND) or the eyes with the hand fixed (EYE), for 4 target patterns.
Deposit: OSF s67g5 (CC0), 'DATA SET.zip': one .dat file per signal and trial (10000 samples at 1000 Hz), plus the authors' MATLAB
scripts, which were read as text only (nothing from the deposit is executed). The paper's full text is behind a paywall and not in
Europe PMC, so the checks come from its abstract and the deposit's Data Set Description.
Run from the site folder: python3 -I tools/std/convert/Coudiere2024_JNeurophysiol.py"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Coudiere2024_JNeurophysiol"
MODULE = "timeseries"
P = "DATA SET/"
FREQ = [("1v5hz", 1.5), ("2hz", 2.0), ("2v5hz", 2.5), ("3hz", 3.0), ("3v5hz", 3.5), ("4hz", 4.0), ("8hz", 8.0), ("FULL", 240.0)]
CONDS = [("EYE-HAND", "eyes and hand track"), ("HAND", "hand tracks, eyes fixed"), ("EYE", "eyes track, hand fixed")]
CHANNELS = ["TargetX", "TargetY", "HandX", "HandY", "EyeX", "EyeY", "PupilDia"]
N = 10000

z = zipfile.ZipFile(stdlib.raw_dir(ID) / "data.zip")
names = {i.filename[len(P):]: i for i in z.infolist() if i.filename.startswith(P) and not i.is_dir()}
subs = sorted({k.split("/")[0] for k in names if re.fullmatch(r"S\d+/.*", k)}, key=lambda s: int(s[1:]))
assert len(subs) == 28


def vec(sub, fn):
    t = z.read(names[f"{sub}/{fn}"]).split()
    return np.array(t, dtype=float)


TMP = stdlib.raw_dir(ID) / '_tmp'
TMP.mkdir(exist_ok=True)
rows, extra = [], []
n_unique_ok = {}
n_trials_bad = 0
dim_ok = True
for si, sub in enumerate(subs, start=1):
    code = f"{ID}_S{si:04d}"
    calib = {}
    for cond, _ in CONDS:
        c = np.array(z.read(names[f"{sub}/Calibmatrix_{cond}.dat"]).split(), float).reshape(4, 2)
        calib[cond] = c
    trial = 0
    for cond, cdesc in CONDS:
        for ftag, fhz in FREQ:
            for pat in (1, 2, 3, 4):
                trial += 1
                stem = f"Pattern{pat}_{ftag}_{cond}.dat"
                cols = {ch: vec(sub, f"{ch}_{stem}") for ch in CHANNELS}
                if any(len(v) != N for v in cols.values()):
                    dim_ok = False
                blink_idx = vec(sub, f"Blink_{stem}").astype(int)
                sac_idx = vec(sub, f"SacXY_{stem}").astype(int)
                blink = np.zeros(N, int)
                blink[blink_idx[blink_idx > 0] - 1] = 1
                sac = np.zeros(N, int)
                sac[sac_idx[sac_idx > 0] - 1] = 1
                tx = cols["TargetX"]
                n_changes = int((np.diff(tx[1000:]) != 0).sum())
                n_unique_ok.setdefault(fhz, []).append(len(np.unique(tx)))
                df = pd.DataFrame({"time_s": np.arange(N) / 1000.0, **cols, "blink": blink, "saccade": sac})
                fname = f"samples/{code}_trial{trial:03d}.csv"
                df.to_csv(TMP / f"{code}_trial{trial:03d}.csv", index=False, float_format="%.8g", lineterminator="\r\n")
                extra.append((fname, TMP / f"{code}_trial{trial:03d}.csv"))
                cm = calib[cond]
                rows.append(dict(
                    dataset=ID, study="Coudiere & Danion 2024 J Neurophysiol", experiment="", group="", subject=code, person=code,
                    subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population="healthy adults", setting="lab",
                    device="joystick and eye tracker (1000 Hz), visual target on a screen", session=1, block=f"{cond}_Pattern{pat}",
                    trial=trial, phase="other", time_s=np.nan, task="tracking", unit="recording", condition=f"{cond}: {cdesc}",
                    pert_type="target_update_rate", pert_value=fhz, pert_unit="Hz", feedback="normal", excluded_by_authors=False,
                    recording=trial, samples_file=fname, signal_kind="target, joystick cursor and gaze position (x, y), pupil size, blink and saccade flags",
                    channels=";".join(CHANNELS + ["blink", "saccade"]),
                    channel_units=";".join(["authors' screen units (not stated)"] * 6 + ["a.u. (0 = eye closed, 10 = open)", "0/1 flag", "0/1 flag"]),
                    sample_rate_hz=1000.0, n_samples=N, duration_s=N / 1000.0, events="",
                    processing="none stated (raw files as shared; the authors' scripts apply the calibration gain and offset to gaze and a 25 Hz low-pass filter, which are not applied here)",
                    trial_info=f"pattern={pat};target_update_rate_hz={fhz:g};gaze_calibration_gain_x={cm[0, 0]:.8g};gaze_calibration_gain_y={cm[0, 1]:.8g};"
                               f"gaze_calibration_offset_x={cm[1, 0]:.8g};gaze_calibration_offset_y={cm[1, 1]:.8g};target_position_updates_after_1s={n_changes}"))
df = pd.DataFrame(rows)

ck = stdlib.Checks()
ck.published("participants", df.person.nunique(), 28, 0, "Abstract: twenty-eight participants; Data Set Description: 28 folders")
ck.published("lowest target update rate (Hz)", df.pert_value.min(), 1.5, 0, "Abstract: refresh rate from 1.5 to 240 Hz")
ck.published("highest target update rate (Hz)", df.pert_value.max(), 240, 0, "Abstract: refresh rate from 1.5 to 240 Hz")
ck.published("update rates tested", df.pert_value.nunique(), 8, 0, "Data Set Description: 8 refreshing rates")
ck.published("target patterns", df.block.str.extract(r"Pattern(\d)")[0].nunique(), 4, 0, "Data Set Description: 4 target patterns")
ck.published("tracking conditions (eyes and hand, hand only, eyes only)", df.condition.str.split(":").str[0].nunique(), 3, 0, "Abstract and Data Set Description")
ck.published("recordings per participant (3 x 8 x 4)", len(df) / 28, 96, 0, "Data Set Description: 8 rates x 4 patterns x 3 conditions")
ck.published("sampling rate (Hz)", df.sample_rate_hz.iloc[0], 1000, 0, "Data Set Description scripts: sf = 1000")
ck.internal("every signal file has 10000 samples", dim_ok)
ck.internal("recordings = 28 x 96", len(df) == 28 * 96 and df.groupby("person").size().eq(96).all())
ck.internal("target positions repeat in steps at the low rates (distinct values in 10 s far below 10000) and change on almost every sample at 240 Hz",
            max(np.median(n_unique_ok[f]) for f in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0)) < 100 and np.median(n_unique_ok[240.0]) > 4000,
            str({k: float(np.median(v)) for k, v in n_unique_ok.items()}))
meta = dict(
    paper=dict(authors="Coudiere A, Danion FR", year=2024, title="Eye-hand coordination all the way: from discrete to continuous hand movements",
               venue="J Neurophysiol 131(4):652-667", doi="10.1152/jn.00314.2023"),
    lab="Danion (CNRS, Universite de Poitiers)", data_url="https://osf.io/s67g5/", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Each trial's signals (one .dat file per signal: target x, y, hand x, y, gaze x, y, pupil size; blink and saccade files hold sample numbers) were combined into one CSV "
             "of 10000 rows with time_s (sample number / 1000), the seven signals as shared (8 significant digits), and blink and saccade columns that are 1 on the samples the authors' files list "
             "(sample numbers start at 1 in their files). Gaze calibration (gain, offset) is not applied; it is in trial_info. The authors' own analysis drops the first second."),
    how=["One row is one 10-s trial at 1000 Hz. The target jumped between positions of one of 4 unpredictable patterns, updated at 1.5, 2, 2.5, 3, 3.5, 4, 8 or 240 Hz (the last is the screen rate, so it moves continuously). "
         "In EYE-HAND trials participants followed it with gaze and a joystick cursor, in HAND trials with the joystick only (eyes fixed), in EYE trials with gaze only (hand fixed). "
         "28 participants x 8 rates x 4 patterns x 3 conditions = 96 trials each.",
         "Target, hand (joystick cursor) and gaze positions are in the same coordinates (x horizontal, y vertical). The deposit does not state their unit, so channel_units says so; the authors' gaze calibration gains are near 1. "
         "trial counts the recordings in the order EYE-HAND, HAND, EYE, then rate, then pattern; it is not the order the participant did them, which is not in the deposit.",
         "Calibmatrix rows 3-4 are not documented and are not used. No age, sex or handedness is in the deposit."],
    unit_note="trial (10 s of target, joystick cursor and gaze positions at 1000 Hz)",
    differences=["The paper's full text could not be read (paywalled, not in Europe PMC), so none of its tracking error, lag or fixation/pursuit numbers are reproduced; checks are against the abstract and the deposit's description.",
                 "The unit of position is not stated in the deposit (the target and gaze lie within about +/-10 units, so degrees of visual angle or cm are likely); treat values as the authors' screen units."])
stdlib.release(ID, MODULE, df, meta, ck, no_published_reason=None, extra_files=extra)
import shutil
shutil.rmtree(TMP, ignore_errors=True)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))
