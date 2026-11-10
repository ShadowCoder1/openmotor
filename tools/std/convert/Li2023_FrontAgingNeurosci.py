"""Li, Liu, Xie, Ji & Chen (2023) Front Aging Neurosci 15:1147079: age-related decline of online visuomotor adaptation.
52 right-handed adults (29 aged 60-73, 23 aged 19-27) kept a cursor inside a target that moved as a sum of five sines (60 s per trial,
horizontal hand movements on a digitizer) with 4 baseline trials, 12 trials with a 200 ms cursor delay and 3 post-test trials; gaze
was recorded at 1000 Hz. Deposit: OSF tq8ye (no license stated), 'Age related deteriorations of online visuomotor adaptation.rar'
(subject<id>.mat per participant, info.mat and the authors' per-trial summaries). The RAR is read with 7-Zip (data only, one member at a
time); nothing from the deposit is executed.
Run from the site folder: python3 -I tools/std/convert/Li2023_FrontAgingNeurosci.py"""
import glob
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Li2023_FrontAgingNeurosci"
MODULE = "timeseries"
RD = stdlib.raw_dir(ID)
RAR = str(RD / "data.rar")
SZ = r"C:\Program Files\7-Zip\7z.exe"
TMP = RD / "_x"
SUB = "Age related deteriorations of online visuomotor adaptation"
N_TRIAL = 19          # formal trials: 4 baseline, 12 adaptation, 3 post-test (4 practice trials are not in the files)
SAMP = 60000


def extract(*members):
    TMP.mkdir(exist_ok=True)
    p = subprocess.run([SZ, "x", RAR, f"-o{TMP}", "-y", *[f"{SUB}\\{m}" for m in members]], capture_output=True, text=True)
    assert "Everything is Ok" in p.stdout, p.stdout[-400:]
    return [TMP / SUB / m for m in members]


def cleanup(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


small = ["info.mat", "RMSE target-cursor.mat", "Target-cursor lag.mat", "Gaze-cursor lag.mat", "Gaze-target lag.mat", "Random_error.mat", "Saccade_glis.mat"]
fp = extract(*small)
M = {Path(f).name: sio.loadmat(f) for f in fp}


def only(name):
    ks = [k for k in M[name] if not k.startswith("__")]
    return M[name][ks[0]]


info = M["info.mat"]["info"]
rmse, tclag, gclag, gtlag, rand = only("RMSE target-cursor.mat"), only("Target-cursor lag.mat"), only("Gaze-cursor lag.mat"), only("Gaze-target lag.mat"), only("Random_error.mat")
sg = M["Saccade_glis.mat"]
assert info.shape == (52, 3) and rmse.shape == (52, N_TRIAL) and tclag.shape == (52, N_TRIAL) and gclag.shape == (52, N_TRIAL)
ids = info[:, 0].astype(int)
ages = info[:, 1]
sexcode = info[:, 2]
older = ages >= 60
PHASE = lambda t: ("baseline" if t <= 4 else "learning" if t <= 16 else "aftereffect")
BLOCK = lambda t: ("baseline" if t <= 4 else "adaptation (200 ms cursor delay)" if t <= 16 else "post-test")


def load_subject(sid):
    f = extract(f"subject{sid}.mat")[0]
    s = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["subject"]
    return s, f


def trial_arrays(s, t):
    """formal trial number t (1-19) is element t + 3 of the struct (struct trials 1-4 are the empty practice trials)."""
    e = s[t + 3]
    assert int(e.Trial) == t + 4
    E = np.asarray(e.EyeTotal, float)
    if E.ndim != 2:                      # two trials have an empty EyeTotal in the deposit: gaze columns are left empty
        E = np.full((SAMP, 4), np.nan)
    return np.asarray(e.Hand_Center, float), E, np.asarray(e.EyeTotal).ndim != 2, np.asarray(e.EyeTotal).shape[-1] if np.asarray(e.EyeTotal).ndim == 2 else 0


n_samples = {}
bad_shape = 0
n_no_eye = 0
nine_col = set()
dt_ok = True
no_eye_trials = set()
sigs = {}
for si, sid in enumerate(ids, start=1):
    s, f = load_subject(sid)
    assert len(s) == 23
    r = []
    for t in range(1, N_TRIAL + 1):
        H, E, no_eye, ncol = trial_arrays(s, t)
        if H.shape != (SAMP, 5) or E.shape[0] != SAMP:
            bad_shape += 1
        n_no_eye += int(no_eye)
        if no_eye:
            no_eye_trials.add((si, t))
        if ncol == 9:
            nine_col.add(si)
        dt_ok &= bool(np.allclose(np.diff(H[:, 0]), 0.001, atol=1e-6))
        n_samples[(si, t)] = H.shape[0]
        d = 200 if 5 <= t <= 16 else 0
        cur = np.concatenate([np.full(d, np.nan), H[:SAMP - d, 1]]) if d else H[:, 1]
        r.append(np.sqrt(np.nanmean((cur[2000:] - H[2000:, 3]) ** 2)))     # px RMSE of the (delayed) cursor to the target after the first 2 s
    sigs[si] = np.array(r)
    del s
    cleanup([f])

# The authors' summary matrices (52 x 19) are NOT in the order of info.mat. Each participant is matched to the matrix row whose RMSE profile over the
# 19 trials follows the participant's own cursor-to-target RMSE (hand delayed 200 ms in trials 5-16), one-to-one.
from scipy.optimize import linear_sum_assignment
C = np.array([[np.corrcoef(sigs[si], rmse[j])[0, 1] for j in range(52)] for si in range(1, 53)])
ri, ci = linear_sum_assignment(-C)
row_of = {si: int(ci[si - 1]) for si in range(1, 53)}
srt = np.sort(C, axis=1)
slopes = [float((sigs[si] * rmse[row_of[si]]).sum() / (rmse[row_of[si]] ** 2).sum()) for si in range(1, 53)]
print("match: min corr", C[ri, ci].min(), "min margin", (srt[:, -1] - srt[:, -2]).min(), "px per deg median", np.median(slopes))
assert C[ri, ci].min() > 0.99 and (srt[:, -1] - srt[:, -2]).min() > 0.02 and (C.argmax(1) == ci).all()

rows = []
for si, sid in enumerate(ids, start=1):
    code = f"{ID}_S{si:04d}"
    i = si - 1
    j = row_of[si]
    for t in range(1, N_TRIAL + 1):
        fname = f"samples/{code}_trial{t:02d}.csv"
        delay = 200 if 5 <= t <= 16 else 0
        no_eye = (si, t) in no_eye_trials
        rows.append(dict(
            dataset=ID, study="Li et al. 2023 Front Aging Neurosci", experiment="", group="older adults" if older[i] else "young adults", subject=code, person=code,
            subject_excluded_by_authors=False, age=float(ages[i]), sex="F" if sexcode[i] == 1 else "M", handedness="right",
            population="healthy older adults" if older[i] else "healthy young adults", setting="lab",
            device="Wacom Intuos digitizer (horizontal slider, right hand) and EyeLink Portable Duo (left eye, 1000 Hz), 27-inch 60 Hz monitor at 50 cm",
            session=1, block=BLOCK(t), trial=t, phase=PHASE(t), time_s=np.nan, task="tracking", unit="recording",
            condition="cursor delayed 200 ms" if delay else "no delay", pert_type="delay" if delay else "none", pert_value=delay, pert_unit="ms" if delay else "",
            feedback="altered" if delay else "normal", excluded_by_authors=False, recording=t, samples_file=fname,
            signal_kind="gaze position and pupil area (left eye), hand position and target position along the horizontal axis (screen pixels)",
            channels="eyelink_ms;gaze_x_px;gaze_y_px;pupil_area_px;hand_x_px;hand_y_px;target_x_px;target_y_px",
            channel_units="ms;px;px;px (area);px;px;px;px", sample_rate_hz=1000.0, n_samples=n_samples[(si, t)], duration_s=n_samples[(si, t)] / 1000.0,
            events="analysis_window_starts=2", processing="none stated; blink samples as shared (NaN or 0); hand and target resampled to 1 kHz by the authors (digitizer 40 Hz)",
            trial_info=f"rmse_target_cursor_deg={rmse[j, t - 1]:.8g};target_cursor_lag_ms={float(tclag[j, t - 1]):g};gaze_cursor_lag_ms={float(gclag[j, t - 1]):g};"
                       f"gaze_target_lag_ms={float(gtlag[j, t - 1]):g};random_error_deg={rand[j, t - 1]:.8g};saccades={sg['Sacc'][j, t - 1]:g};glissades={sg['Glis'][j, t - 1]:g};"
                       f"saccades_plus_glissades={sg['Saccglis'][j, t - 1]:g}" + (";gaze_data=missing in deposit" if no_eye else "")))
df = pd.DataFrame(rows)

ck = stdlib.Checks()
ck.published("older participants", int(older.sum()), 29, 0, "Methods, Participants: older group n = 29")
ck.published("young participants", int((~older).sum()), 23, 0, "Methods, Participants: young group n = 23")
ck.published("mean age, older group (years)", ages[older].mean(), 65.97, 0.005, "Participants: 65.97 y (SD 3.8)")
ck.published("SD of age, older group", ages[older].std(ddof=1), 3.8, 0.05, "Participants")
ck.published("mean age, young group (years)", ages[~older].mean(), 22.93, 0.005, "Participants: 22.93 y (SD 1.97)")
ck.published("SD of age, young group", ages[~older].std(ddof=1), 1.97, 0.005, "Participants")
ck.published("women in older group", int(sexcode[older].sum()), 13, 0, "Participants: 13 women")
ck.published("women in young group", int(sexcode[~older].sum()), 11, 0, "Participants: 11 women")
ck.published("formal trials per participant (4 baseline + 12 adaptation + 3 post-test)", len(df) / 52, 19, 0, "Methods, Procedure")
ck.published("trial length (s)", df.duration_s.iloc[0], 60, 0, "Methods: each formal trial 60 s")
ck.published("adaptation trials per participant", int((df.pert_value == 200).sum() / 52), 12, 0, "Methods: 12 adaptation trials with a 200 ms delay")
base = np.arange(4)
ck.published("baseline target-cursor RMSE, older (deg)", rmse[[row_of[k + 1] for k in range(52) if older[k]]][:, base].mean(), 1.55, 0.005, "Results, tracking error: older 1.55 +/- 0.08")
ck.published("baseline target-cursor RMSE, young (deg)", rmse[[row_of[k + 1] for k in range(52) if not older[k]]][:, base].mean(), 1.16, 0.005, "Results, tracking error: young 1.16 +/- 0.05")
ck.published("baseline target-cursor lag, older (ms)", tclag[[row_of[k + 1] for k in range(52) if older[k]]][:, base].mean(), 89.466, 0.0005, "Results, lag: older 89.466")
ck.published("baseline target-cursor lag, young (ms)", tclag[[row_of[k + 1] for k in range(52) if not older[k]]][:, base].mean(), 43.261, 0.0005, "Results, lag: young 43.261")
ck.internal("every trial has 60000 samples at 1 ms steps in hand and eye arrays", bad_shape == 0 and dt_ok, f"{bad_shape} bad shapes")
ck.internal("each participant's own cursor-to-target pixel RMSE profile matches one row of the authors' RMSE matrix at r > 0.99 with a clear margin, one-to-one (the matrices are ordered older group first, unlike info.mat)", True, f"min r {C[ri, ci].min():.4f}")
ck.internal("authors' RMSE (deg) = pixel RMSE of the cursor (hand delayed 200 ms in trials 5-16) to the target after 2 s, divided by 15 pixels per unit (median slope within 0.5 percent)", abs(np.median(slopes) - 15) < 0.075, f"median slope {np.median(slopes):.3f}")
ck.internal("recordings = 52 x 19", len(df) == 52 * 19)
ck.internal("two trials (two participants) have an empty EyeTotal in the deposit; one participant's EyeTotal has 9 instead of 10 columns (first four columns read the same)", n_no_eye == 2 and len(nine_col) == 1, f"{n_no_eye} empty, nine-column participants {sorted(nine_col)}")
meta = dict(
    paper=dict(authors="Li N, Liu J, Xie Y, Ji W, Chen Z", year=2023, title="Age-related decline of online visuomotor adaptation: a combined effect of deteriorations of motor anticipation and execution",
               venue="Front Aging Neurosci 15:1147079", doi="10.3389/fnagi.2023.1147079"),
    lab="Chen (East China Normal University)", data_url="https://osf.io/tq8ye/", license="none stated", license_url="",
    changes=("For each participant and formal trial, the Hand_Center (time in s, hand x, hand y, target x, target y; screen pixels) and EyeTotal (EyeLink time in ms, gaze x, gaze y, pupil area; "
             "the other columns of EyeTotal are undocumented and not carried) arrays were written as one CSV of 60000 rows; values as shared (7-8 significant digits for hand and target). "
             "The deposit's Cursor array (the hand series from 2 s on), EyeTotal1/EyeTotal2 (variants of EyeTotal), EyeSaccade1 (a saccade table whose row counts do not match the authors' saccade counts) and the video were not carried. "
             "The authors' per-trial summaries (RMSE, lags, random error, saccade counts) are in trial_info. Age and sex come from info.mat; the group is older (age 60 or more) or young."),
    how=["One row is one 60-s trial of continuous visuomotor tracking: a cursor (1 deg) had to be kept inside a moving target circle (1.4 deg) by moving the right hand horizontally on a digitizer. The target followed the sum of five sine waves (0.09-0.495 Hz). "
         "Trials 1-4 are baseline, 5-16 adaptation with the cursor delayed 200 ms behind the hand, 17-19 post-test (no delay); four earlier practice trials are not in the files. Samples are 1000 Hz.",
         "Hand position (hand_x_px, hand_y_px) is the digitizer position mapped to screen pixels; the displayed cursor in delay trials was this position 200 ms later (inferred: in delay trials the hand leads the target by about 110-150 ms, the baseline hand lags it by 40-60 ms, matching the authors' lag values after adding the delay), "
         "so the delayed cursor is not stored. hand_y and target_y are constant near the screen centre (the task is one-dimensional). Gaze (gaze_x_px, gaze_y_px) and pupil area are the left eye's EyeLink samples; during blinks x and pupil are empty and y is 0 for most participants (the coding of blink samples differs between participants as shared: NaN or 0). Two trials of two participants have no gaze data in the deposit (gaze columns empty; trial_info says so). "
         "Column meanings were read from the OSF description, the paper and the match between these series and the authors' lags; they are inferred, not documented column by column.",
         "Conversion of pixels to degrees is not stated in the deposit (the paper gives 1 deg cursor and 12.9 deg for 387 px). Their summary RMSE equals the pixel RMSE of the delayed cursor divided by 15 (see Known differences). Practice trials are not in the deposit."],
    unit_note="trial (60 s of gaze, pupil, hand and target position at 1000 Hz)",
    differences=["The authors' summary matrices (RMSE, lags, saccade counts; 52 x 19) are ordered with the 29 older participants first and the 23 young second, not in the order of info.mat. Each participant was matched to a matrix row by the profile of cursor-to-target pixel RMSE over the 19 trials (r above 0.99, one-to-one); the baseline means by group then equal the paper's.",
                 "The authors' RMSE (called degrees) equals the pixel RMSE of the cursor to the target divided by 15 pixels per unit (median slope 15.01); the paper's 12.9 deg for 387 px implies about 30 pixels per degree, so the unit of the deposited RMSE could not be reconciled with the paper's pixel size.",
                 "The paper says 56 volunteers were tested and 4 excluded for noisy or missing eye data; the 4 are not in the deposit, so all 52 participants have no exclusion flag. The 4 practice trials are not in the deposit."])

def extra_gen():
    for si, sid in enumerate(ids, start=1):
        code = f"{ID}_S{si:04d}"
        s, f = load_subject(sid)
        for t in range(1, N_TRIAL + 1):
            H, E, _, _ = trial_arrays(s, t)
            out = pd.DataFrame({"time_s": np.round(H[:, 0], 3), "eyelink_ms": E[:, 0], "gaze_x_px": E[:, 1], "gaze_y_px": E[:, 2], "pupil_area_px": E[:, 3],
                                "hand_x_px": H[:, 1], "hand_y_px": H[:, 2], "target_x_px": H[:, 3], "target_y_px": H[:, 4]})
            buf = io.StringIO()
            out.to_csv(buf, index=False, float_format="%.7g", lineterminator="\r\n")
            yield f"samples/{code}_trial{t:02d}.csv", buf.getvalue().encode()
        del s
        cleanup([f])


stdlib.release(ID, MODULE, df, meta, ck, extra_files=extra_gen())
shutil.rmtree(TMP, ignore_errors=True)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))
