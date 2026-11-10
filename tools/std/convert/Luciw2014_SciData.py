"""Luciw, Jarocka & Edin 2014 (Sci Data), WAY-EEG-GAL: 12 right-handed adults grasped, lifted, held and replaced an instrumented object
whose weight (165, 330, 660 g) and surface (sandpaper, suede, silk) changed between lifts; EEG, EMG, hand position and forces were recorded.
Figshare collection 988376 holds one ~0.8-1 GB ZIP per participant (about 10 GB in all: raw EEG, EMG, kinematics and forces per series).
Only the small P<n>_AllLifts.mat table of each ZIP (the authors' per-lift event times, force measures, kinematics and durations) is converted;
the raw signals are too large."""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Luciw2014_SciData"
MODULE = "grasp_lift"
W = {1: 165, 2: 330, 4: 660}
SURF = {1: "sandpaper", 2: "suede", 3: "silk"}
BT = {1: "weight series", 2: "friction series", 3: "mixed series"}
COLS = ["series", "lift", "block_type", "weight_g", "surface", "prev_weight_g", "prev_surface", "start_time_s", "led_on_s", "led_off_s",
        "t_ind_touch_s", "t_thumb_touch_s", "t_first_digit_touch_s", "t_both_digit_touch_s", "t_ind_start_load_s", "t_thumb_start_load_s",
        "t_both_start_load_s", "t_lift_off_s", "t_replace_s", "t_ind_release_s", "t_thumb_release_s", "t_both_released_s",
        "grip_force_max_n", "load_force_max_n", "grip_force_rate_max_n_s", "load_force_rate_max_n_s", "t_grip_force_max_s", "t_load_force_max_s",
        "t_grip_force_rate_max_s", "t_load_force_rate_max_s", "grip_force_hold_n", "load_force_hold_n", "t_hand_start_s", "t_hand_stop_s",
        "t_peak_vel_reach_s", "t_peak_vel_retract_s", "grip_aperture_max_cm", "t_grip_aperture_max_s", "dur_reach_s", "dur_preload_s",
        "dur_load_phase_s", "dur_release_s"]
SRC = ["Run", "Lift", "BlockType", "CurW", "CurS", "PrevW", "PrevS", "StartTime", "LEDOn", "LEDOff", "tIndTouch", "tThumbTouch",
       "tFirstDigitTouch", "tBothDigitTouch", "tIndStartLoadPhase", "tThuStartLoadPhase", "tBothStartLoadPhase", "tLiftOff", "tReplace",
       "tIndRelease", "tThuRelease", "tBothReleased", "GF_Max", "LF_Max", "dGF_Max", "dLF_Max", "tGF_Max", "tLF_Max", "tdGF_Max", "tdLF_Max",
       "GF_Hold", "LF_Hold", "tHandStart", "tHandStop", "tPeakVelHandReach", "tPeakVelHandRetract", "GripAparture_Max", "tGripAparture_Max",
       "Dur_Reach", "Dur_Preload", "Dur_LoadPhase", "Dur_Release"]


def main():
    ck = stdlib.Checks()
    rows = []
    stats = {}
    for p in range(1, 13):
        m = sio.loadmat(stdlib.raw_dir(DS) / f"P{p}_AllLifts.mat", squeeze_me=True, struct_as_record=False)["P"]
        A, names = m.AllLifts, [str(x) for x in m.ColNames]
        assert A.shape[1] == len(names) == 43 and set(A[:, 0]) == {float(p)}
        ix = {n: i for i, n in enumerate(names)}
        for r in range(len(A)):
            a = A[r]
            d = dict(dataset=DS, study=DS, experiment="", group="", subject=f"P{p}", person=f"P{p}", subject_excluded_by_authors=False,
                     age="", sex="", handedness="right", population="healthy", setting="lab",
                     device="Instrumented grasp-and-lift object with force/torque sensors, 3D hand tracking, 32-channel EEG and EMG",
                     session=1, block=f"series {int(a[ix['Run']])}", trial=r + 1, phase="test", time_s=float(a[ix['StartTime']]),
                     task="grasping and lifting", unit="trial", condition=f"{W[int(a[ix['CurW']])]} g, {SURF[int(a[ix['CurS']])]}",
                     pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False)
            for c, s in zip(COLS, SRC):
                v = a[ix[s]]
                if c == "block_type": v = BT[int(v)]
                elif c == "weight_g": v = W[int(v)]
                elif c == "surface": v = SURF[int(v)]
                elif c == "prev_weight_g": v = "" if np.isnan(v) else W[int(v)]
                elif c == "prev_surface": v = "" if np.isnan(v) else SURF[int(v)]
                elif c in ("series", "lift"): v = int(v)
                elif np.isnan(v): v = ""
                d[c] = v
            rows.append(d)
        runs = {}
        for r in A:
            runs.setdefault((int(r[ix['Run']]), int(r[ix['BlockType']])), []).append(r)
        stats[p] = runs
    n = len(rows)
    ck.internal("294 lifts for each of 12 participants", n == 12 * 294 and all(sum(len(v) for v in s.values()) == 294 for s in stats.values()))
    ck.internal("9 series per participant", all(len(s) == 9 for s in stats.values()))
    ck.internal("lift numbers run 1..n within each series", all([int(x[2]) for x in v] == list(range(1, len(v) + 1)) for s in stats.values() for v in s.values()))
    ck.published("participants (12)", len({r["subject"] for r in rows}), 12, 0, "Methods, Participants")
    w_lens = {len(v) for s in stats.values() for (run, bt), v in s.items() if bt in (1, 2)}
    m_lens = {len(v) for s in stats.values() for (run, bt), v in s.items() if bt == 3}
    ck.published("lifts per weight or friction series (34)", float(np.mean(list(w_lens))) if len(w_lens) == 1 else -1, 34, 0, "Methods, Experimental paradigm")
    ck.published("lifts per mixed series (28)", float(np.mean(list(m_lens))) if len(m_lens) == 1 else -1, 28, 0, "Methods, Experimental paradigm")
    ck.published("mixed series per participant (2)", np.mean([sum(bt == 3 for (_, bt) in s) for s in stats.values()]), 2, 0, "Methods, Experimental paradigm")
    ck.published("friction series per participant (2)", np.mean([sum(bt == 2 for (_, bt) in s) for s in stats.values()]), 2, 0, "Methods, Experimental paradigm")
    # weight changes in a weight series: the paper says 12 unpredictable changes among 34 lifts
    ch = [sum(1 for x in v if not np.isnan(x[5]) and x[3] != x[5]) for s in stats.values() for (_, bt), v in s.items() if bt == 1]
    chm = [sum(1 for x in v if not np.isnan(x[5]) and (x[3] != x[5] or x[4] != x[6])) for s in stats.values() for (_, bt), v in s.items() if bt == 3]
    ck.published("mean unexpected changes per mixed series (11)", np.mean(chm), 11, 0.5, "Methods, Experimental paradigm (11 unexpected changes)")
    print("weight changes per weight series", sorted(set(ch)), "mixed changes", sorted({sum(1 for x in v if not np.isnan(x[5]) and (x[3] != x[5] or x[4] != x[6])) for s in stats.values() for (_, bt), v in s.items() if bt == 3}))
    ck.published("mean weight changes per weight series (12)", np.mean(ch), 12, 0.5, "Methods, Experimental paradigm (12 unpredictable weight changes)")
    meta = dict(
        study=DS,
        paper=dict(authors="Luciw MD, Jarocka E, Edin BB", year=2014,
                   title="Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction",
                   venue="Sci Data 1:140047", doi="10.1038/sdata.2014.47"),
        lab="Edin (Umea University)", data_url="https://doi.org/10.6084/m9.figshare.c.988376",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Only the per-lift table P<n>_AllLifts.mat of each participant's ZIP was converted (pulled out of the ZIP with range requests); the code numbers for weight, "
                 "surface and series type were replaced by their meanings (165/330/660 g; sandpaper/suede/silk; weight/friction/mixed series), columns were renamed to snake_case with units, "
                 "empty (NaN) cells left empty, and the participant number became a code. The raw EEG, EMG, hand-position and force/torque signals (about 10 GB over the 12 ZIPs) are not included."),
        how=["One row is one lift (a grasp, lift, hold, replace and release of the instrumented object with thumb and index finger) by one participant, in the order of the deposit's table (series 1-9, lifts within each series).",
             "The values are the authors' own: event times in seconds from the start of the lift window (start_time_s is from the start of the series), forces in newtons, grip aperture in cm, durations in seconds.",
             "Series types: weight series change the weight (165, 330 or 660 g) between lifts, friction series change the surface, mixed series change both. time_s repeats start_time_s."],
        unit_note="lift (grasp-and-lift trial)",
        differences=["The paper counts 328 lifts per participant (3,936 in all). The deposit's table has 294 lifts per participant: 5 weight series and 2 friction series of 34 lifts and 2 mixed series of 28 lifts, "
                     "one 34-lift series fewer than the paper's six weight series, two friction series and two mixed series (the practice series and the friction-estimation lifts are not in the table either). "
                     "The converted file therefore has 3,528 rows.",
                     "Zenodo-style checks do not apply; the collection DOI is 10.6084/m9.figshare.c.988376 (Figshare, CC0, one article per participant)."],
    )
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
