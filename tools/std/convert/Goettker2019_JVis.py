"""Goettker, Braun & Gegenfurtner (2019) J Vis 19(7):2: dynamic combination of position and motion information when tracking
moving targets.

Raw file: raw_extra/Goettker2019_JVis/Data_Contrast.zip (Zenodo 3244152). Layout: Data_Contrast/<contrast: High Contrast | Low
Contrast | Isoluminance>/<Step-Ramp | Control (1) - position & motion | Control (2) - Static Targets>/e<set>v<participant>b<block>/
{e#v#b#.log, trial<N>.dat}. A .log has one row per planned trial (tab separated parameters), a .dat holds one trial's gaze samples
(time ms, x px, y px, EyeLink flags; 1000 Hz). The folders named 'Old Data' (earlier recordings of a few blocks) are not converted.
The ReadMe.pdf files (read as text) document the columns. Nothing in the deposit is run. Module: timeseries (one row per trial, its gaze
samples as CSV inside the ZIP).

Run from the site folder: python3 -I tools/std/convert/Goettker2019_JVis.py
"""
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Goettker2019_JVis"
RAW = stdlib.raw_dir(DS)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
PAT = re.compile(r"Data_Contrast/(High Contrast|Low Contrast|Isoluminance)/(Step-Ramp|Control \((\d)\)[^/]*)/(Old Data/)?"
                 r"e(\d)v(\d+)b(\d)/(trial(\d+)\.dat|e\dv\d+b\d\.log)$")
CONTR = {"High Contrast": "high luminance contrast", "Low Contrast": "low luminance contrast", "Isoluminance": "isoluminant (chromatic only)"}
CORDER = {"High Contrast": 0, "Low Contrast": 1, "Isoluminance": 2}
KIND = {"SR": "step-ramp (main experiment)", "C1": "control 1: position and motion components separately",
        "C2": "control 2: static targets"}
SPEEDS = {1: "10, 15 or 20 deg/s", 2: "2.5, 5 or 7.5 deg/s"}


def f(x):
    x = float(x)
    return f"{x:g}"


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "Data_Contrast.zip")
    logs, dats = {}, {}
    old = 0
    for n in z.namelist():
        m = PAT.match(n)
        if not m:
            continue
        if m[4]:
            old += n.endswith(".dat")
            continue
        kind = "SR" if m[2] == "Step-Ramp" else f"C{m[3]}"
        key = (kind, m[1], int(m[5]), int(m[6]), int(m[7]))  # kind, contrast, set, participant, block
        if n.endswith(".log"):
            logs[key] = n
        else:
            dats[key + (int(m[9]),)] = n
    inits = sorted({k[3] for k in logs})
    code = {v: f"P{i + 1:02d}" for i, v in enumerate(inits)}

    rows, files = [], []
    planned = defaultdict(int)
    nsamp_total = 0
    missing_dat = []
    dts = []
    seq = defaultdict(int)
    for key in sorted(logs, key=lambda k: (k[0], CORDER[k[1]], k[3], k[2], k[4])):
        kind, contrast, eset, v, blk = key
        lg = [l.split() for l in z.read(logs[key]).decode("latin1").strip().splitlines()]
        static = kind == "C2"
        ncol = 34 if static else 33
        assert all(len(r) == ncol for r in lg), (key, {len(r) for r in lg})
        assert [int(r[3]) for r in lg] == list(range(1, len(lg) + 1)), key
        planned[(kind, v)] += len(lg)
        for r in lg:
            t = int(r[3])
            dn = dats.get(key + (t,))
            if dn is None:
                missing_dat.append((kind, contrast, v, blk, t))
                continue
            a = np.array(z.read(dn).split(), dtype=float).reshape(-1, 4)
            nsamp_total += len(a)
            tm = a[:, 0]
            dts.append(float(np.median(np.diff(tm))))
            fl = a[:, 3].astype(int)
            marks = (tm[(fl & 4) > 0] - tm[0]) / 1000.0
            body = pd.DataFrame({"time_s": (tm - tm[0]) / 1000.0, "gaze_x_px": a[:, 1], "gaze_y_px": a[:, 2], "eyelink_flags": fl})
            sname = f"{kind.lower()}_{contrast.split()[0].lower()[:4]}_{code[v]}_e{eset}b{blk}_t{t:03d}.csv"
            body.to_csv(TMP / sname, index=False, float_format="%.6g")
            files.append((f"samples/{sname}", str(TMP / sname)))
            sk = (kind, contrast, v)
            seq[sk] += 1
            off = 1 if static else 0  # control 2's log has one more column (11), which shifts the later ones
            x0, y0, x1, y1, xe, ye = (r[22 + off], r[25 + off], r[28 + off], r[30 + off], r[29 + off], r[31 + off])
            info = (f"start_direction_deg={f(r[7])};end_direction_deg={f(r[8])};target_speed_deg_s={f(r[9])};"
                    + (f"target_position_after_step_deg={f(r[10])};" if static else "")
                    + f"target_x_start_px={f(x0)};target_y_start_px={f(y0)};target_x_after_step_px={f(x1)};target_y_after_step_px={f(y1)};"
                    f"target_x_end_px={f(xe)};target_y_end_px={f(ye)};speed_set={eset}")
            rows.append(dict(
                dataset=DS, study=DS, experiment=KIND[kind], group="", subject=f"{kind}_{contrast.split()[0]}_{code[v]}", person=code[v],
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device="EyeLink 1000 Plus (right eye, 1000 Hz), CRT-type monitor 1920x1080 px, 69.8 cm wide, 120 Hz, 90 cm viewing distance",
                session=eset, block=f"set {eset} block {blk}", trial=seq[sk], phase="test", time_s="",
                task="eye movements", unit="recording", condition=f"{CONTR[contrast]}; target speeds {SPEEDS[eset] if not static else 'static end positions of ' + SPEEDS[eset]}",
                pert_type="intrasaccadic_target_step" if not static else "none", pert_value=10 if not static else 0, pert_unit="deg" if not static else "",
                feedback="normal", excluded_by_authors=False,
                recording=seq[sk], samples_file=f"samples/{sname}",
                signal_kind="gaze position of the right eye (screen pixels) during one step-ramp or control trial",
                channels="gaze_x_px;gaze_y_px;eyelink_flags", channel_units="px;px;EyeLink flags (bit 1 saccade, bit 2 blink, bit 3 experiment marker)",
                sample_rate_hz=1000, n_samples=len(a), duration_s=float(tm[-1] - tm[0]) / 1000.0,
                events="marker_samples_s=" + "|".join(f"{x:g}" for x in marks),
                processing="none stated (raw gaze samples with the tracker's event flags)", trial_info=info))

    sr = sorted({k[3] for k in logs if k[0] == "SR"})
    c1 = sorted({k[3] for k in logs if k[0] == "C1"})
    c2 = sorted({k[3] for k in logs if k[0] == "C2"})
    ck.published("participants, main step-ramp experiment", len(sr), 11, 0, "Methods, Participants (eleven volunteers in the main and first control experiment)")
    ck.published("participants, control experiment 1", len(c1), 11, 0, "Methods, Participants (eleven volunteers in the main and first control experiment)")
    ck.published("planned trials per participant, main experiment (2 speed sets x 3 contrasts x 4 blocks x 120)",
                 float(np.mean([planned[("SR", v)] for v in sr])), 2880, 0, "Methods, Experimental setup and condition (a total of 2,880 trials)")
    ck.published("planned trials per participant, control experiment 1 (3 contrasts x 2 speed sets x 2 blocks x 120)",
                 float(np.mean([planned[("C1", v)] for v in c1])), 1440, 0, "Methods, Control Experiment 1 (a total of 1,440 trials)")
    ck.published("trials per participant in each contrast condition of control experiment 2 (3 blocks x 120)",
                 float(np.mean([planned[("C2", v)] for v in c2])) / 2, 360, 0,
                 "Methods, Control Experiment 2 (three blocks in each condition, 120 trials per block, 720 trials per participant for both conditions)")
    ck.published("sampling interval (ms)", float(np.median(dts)), 1.0, 0.05, "Methods, Experimental setup (EyeLink 1000 Plus, 1000 Hz)")
    ck.internal("one row and one samples file per converted trial", len(rows) == len(files) and len(rows) > 0, f"{len(rows)} trials")
    ck.internal("trials of the log without a gaze file are few", len(missing_dat) <= 5, str(missing_dat))

    meta = dict(
        study=DS,
        paper=dict(authors="Goettker A, Braun DI, Gegenfurtner KR", year=2019,
                   title="Dynamic combination of position and motion information when tracking moving targets", venue="J Vis 19(7):2",
                   doi="10.1167/19.7.2"),
        lab="Gegenfurtner (University of Giessen)",
        data_url="https://doi.org/10.5281/zenodo.3244152",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Each trial's gaze file (.dat) became one CSV with time in seconds from the first sample (the authors' time in ms is the sample "
                 "counter of the session), x and y in pixels and the EyeLink flags. Trial parameters of the log files are in trial_info. "
                 "The folders named 'Old Data' were left out. Participant numbers were replaced by codes."),
        unit_note="trial (recording) of one participant in one contrast condition",
        how=["One row is one trial of one participant. The participant fixated a point, then a Gaussian target appeared 10 deg above or below the "
             "screen centre, jumped to the centre after a random 1-1.5 s and moved at constant speed horizontally (80% of trials) or vertically "
             "(20%), in each of four directions. The target is of high luminance contrast, low luminance contrast or isoluminant "
             "(chromatic contrast only); every participant did all three conditions on different days. experiment is the main step-ramp "
             "experiment, control 1 (the position step and the ramp separated by a 1 s delay) or control 2 (static targets at the "
             "positions where saccades had landed). subject is one participant in one experiment and contrast; person is the participant "
             "number of the deposit, assumed to be the same person in all three experiments.",
             "Gaze samples (time_s, gaze_x_px, gaze_y_px, eyelink_flags) are in screen pixels of a 1920x1080 monitor, 69.8 cm wide at 90 cm, "
             "origin and axes as saved (the authors note that the y coordinate always needs to be flipped, so y increases downward on the screen). "
             "Flags: bit 1 saccade, bit 2 blink, bit 3 experiment marker; events lists the times of the marker samples. The authors' control 2 "
             "readme says the fourth marker is the target step and the fifth the end of presentation; the same order is assumed for the other "
             "experiments (inferred, not stated for them).",
             "trial_info: start_direction_deg (90 = the target started above the centre, 270 = below), end_direction_deg (direction of the ramp: 0 right, 90 up, "
             "180 left, 270 down), target_speed_deg_s, target start, after-step and end positions in pixels, speed_set (1 = 10, 15, 20 deg/s; 2 = 2.5, 5, 7.5 deg/s) "
             "(the authors' readme gives log column 24 for the target's start y, but that column is 0 in every row; the start y is taken from column 26, which holds "
             "+-437.92 px (10 deg) with the sign matching start_direction_deg, an inference; control 2's log has one more column than its readme counts for, "
             "so its columns 29-32 are read one place later), and, in control 2, target_position_after_step_deg (log column 11, described by the authors as the position of the target after the step in the "
             "respective direction; unit not stated, degrees assumed). Other log columns are undocumented and not included. Session is the speed set "
             "(set 1 was always run before set 2); the order of the three contrasts across days is not given. Trial counts each participant's "
             "trials in order of set, block and trial within a contrast condition."],
        differences=["The paper says six volunteers took part in control experiment 2; the deposit has eight participant numbers in that experiment "
                     "(each with 360 trials per contrast, as the paper's trial count implies), so all are rows.",
                     f"{len(missing_dat)} planned trials have no gaze file in the deposit (Low Contrast step-ramp, one participant, block 3), so they are not rows.",
                     "The paper excluded trials during analysis (for example with blinks or anticipatory pursuit); the deposit has no exclusion flags, so "
                     "excluded_by_authors is False throughout. The EyeLink flags mark blinks and saccades.",
                     "The paper's saccade latencies, landing errors and pursuit values are not in the deposit as numbers (it holds raw gaze only), so none of them was "
                     "reproduced; the published checks are counts of participants, trials and the sampling rate."],
    )
    stdlib.release(DS, "timeseries", rows, meta, ck, extra_files=files)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
    print("old-data dat files skipped:", old, "samples:", nsamp_total, "missing:", missing_dat)


if __name__ == "__main__":
    main()
