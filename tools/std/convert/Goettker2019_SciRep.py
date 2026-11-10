"""Goettker, Brenner, Gegenfurtner & de la Malla (2019) Sci Rep 9:5395: corrective saccades influence velocity judgments and
interception.

Raw file: raw_extra/Goettker2019_SciRep/data.zip (Zenodo 2566255, Sacc_Intercept.zip): Interception/action_<ini>_<block>.mat and
Judgment/judge_<ini>_<block>.mat, one MATLAB table per participant and block of 252 trials with a sample every 2 ms, plus a
ReadMe.txt in each folder. File names carry participant initials; they are replaced by codes in order of the sorted initials and
never written out. Nothing in the deposit is run. Module: timeseries (one row per trial, its gaze and finger samples as a CSV).

Run from the site folder: python3 -I tools/std/convert/Goettker2019_SciRep.py
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Goettker2019_SciRep"
RAW = stdlib.raw_dir(DS)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
GAZE = ["LeftGazeX", "LeftGazeY", "RightGazeX", "RightGazeY"]
HAND = ["x", "y", "z"]
CHNAME = {"LeftGazeX": "left_gaze_x", "LeftGazeY": "left_gaze_y", "RightGazeX": "right_gaze_x", "RightGazeY": "right_gaze_y",
          "x": "finger_x", "y": "finger_y", "z": "finger_z"}


def first(g, c):
    v = g[c].dropna()
    return v.iloc[0] if len(v) else float("nan")


def fnum(x):
    return "" if x != x else f"{x:g}"


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "data.zip")
    members = sorted(n for n in z.namelist() if n.endswith(".mat"))
    pat = re.compile(r"(Interception/action|Judgment/judge)_([a-z]+)_(\d)\.mat")
    parsed = []
    for n in members:
        m = pat.search(n)
        assert m, n
        parsed.append((n, "interception" if m[1].startswith("Inter") else "velocity judgment", m[2], int(m[3])))
    inits = sorted({p[2] for p in parsed})
    code = {i: f"P{k + 1:02d}" for k, i in enumerate(inits)}
    ck.published("participants", len(inits), 11, 0, "Methods, Participants (11)")
    ck.internal("each participant has two blocks of each task", all(sum(1 for p in parsed if p[2] == i and p[1] == t) == 2
                                                                    for i in inits for t in ("interception", "velocity judgment")))

    rows, files = [], []
    seq_len = {}
    n_cond = []
    dts = []
    notap = 0
    hits = misses = 0
    total_samples = 0
    for n, task, ini, blk in sorted(parsed, key=lambda p: (p[1], p[2], p[3])):
        m = sio.loadmat(io.BytesIO(z.read(n)), squeeze_me=True, struct_as_record=False)
        df = pd.DataFrame({k: v for k, v in m.items() if isinstance(v, np.ndarray) and v.ndim == 1})
        act = task == "interception"
        if act:
            for h in HAND:
                df[h] = pd.to_numeric(df[h].astype(str), errors="coerce")
        df["condition"] = df["condition"].astype(str)
        total_samples += len(df)
        chans = GAZE + (HAND if act else [])
        units = ["cm"] * 4 + (["mm"] * 3 if act else [])
        # trial order: contiguous runs of the trial number within the file; the file holds 252 trials, each split in two 'parts'
        # (part 1 before target movement onset, part 2 after, where Time restarts at 0)
        trial_arr = df["trial"].to_numpy()
        ok_contig = bool((np.diff(trial_arr) >= 0).all())
        assert ok_contig
        starts = np.flatnonzero(np.r_[True, np.diff(trial_arr) != 0])
        ends = np.r_[starts[1:], len(df)]
        assert len(starts) == 252, (n, len(starts))
        seqkey = (task, ini)
        for s, e in zip(starts, ends):
            g = df.iloc[s:e]
            t = int(g["trial"].iloc[0])
            seqn = (blk - 1) * 252 + t
            tv = g["Time"].to_numpy()
            dts.append(float(np.median(np.diff(tv))))
            reset = np.flatnonzero(np.diff(tv) < 0)
            per = np.zeros(len(g), dtype=int)
            if len(reset):
                per[reset[0] + 1:] = 1
            body = pd.DataFrame({"time_s": tv})
            for c in chans:
                body[c] = g[c].to_numpy()
            body["after_target_onset"] = per
            fname = f"{'int' if act else 'jud'}_{code[ini]}_b{blk}_t{t:03d}.csv"
            body.to_csv(TMP / fname, index=False, float_format="%.6g")
            files.append((f"samples/{fname}", str(TMP / fname)))
            cond = first(g, "condition")
            mm = re.fullmatch(r"pt(\d+)_vel(\d+)_ex(\d+)_(left|right)", cond)
            assert mm, cond
            info = (f"presentation_ms={mm[1]};target_speed_cm_s={mm[2]};crossing_time_ms={mm[3]};direction={mm[4]};"
                    f"target_x_cm={fnum(float(first(g, 'target_x')))};target_velocity_cm_s={fnum(float(first(g, 'target_velocity')))}")
            if act:
                succ = str(first(g, "success"))
                hits += succ == "hit"
                misses += succ == "miss"
                tx = float(first(g, "tap_x"))
                notap += tx != tx
                info += (f";success={succ};tap_x_cm={fnum(tx)};tap_y_cm={fnum(float(first(g, 'tap_y')))};"
                         f"target_at_tap_x_cm={fnum(float(first(g, 'target_at_tap_x')))};"
                         f"target_at_tap_y_cm={fnum(float(first(g, 'target_at_tap_y')))};time_to_tap_s={fnum(float(first(g, 'time_to_tap')))}")
            else:
                info += f";answer={'faster' if first(g, 'answer') == 2 else ('slower' if first(g, 'answer') == 1 else '')}"
            rows.append(dict(
                dataset=DS, study=DS, experiment=task, group="", subject=f"{task}_{code[ini]}", person=code[ini],
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device=("EyeLink II head-mounted (500 Hz, both eyes), Optotrak 3020 finger marker (250 Hz), rear-projection screen 1.25 x 1.00 m"
                        if act else "EyeLink II head-mounted (500 Hz, both eyes), rear-projection screen 1.25 x 1.00 m, keyboard"),
                session=blk, block=f"block {blk}", trial=seqn, phase="test", time_s="",
                task="eye movements", unit="recording", condition=cond,
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=seqn, samples_file=f"samples/{fname}",
                signal_kind=("gaze position (both eyes) and index finger position while pursuing a moving target and tapping it"
                             if act else "gaze position (both eyes) while pursuing a moving target"),
                channels=";".join([CHNAME[c] for c in chans] + ["after_target_onset"]),
                channel_units=";".join(units + ["0 = before target movement onset, 1 = after"]),
                sample_rate_hz=500, n_samples=len(g), duration_s=float(tv[-1] if not len(reset) else tv[-1] + tv[reset[0]]),
                events=("target_movement_onset=0 (time_s restarts at 0 there)" if len(reset) else ""),
                processing="none stated (gaze and finger as saved by the authors; drift correction and filtering were applied only in their analysis)",
                trial_info=info))
            seq_len.setdefault(seqkey, set()).add(seqn)
        n_cond.append(df.groupby("trial")["condition"].first().value_counts().to_numpy())
    ck.published("trials per participant and task (2 blocks of 252)", float(np.mean([len(v) for v in seq_len.values()])), 504, 0,
                 "Methods, Trial counts (252 trials per block, 2 blocks; 504 per task)")
    ck.published("trials per block (2 TX x 3 durations x 3 velocities x 2 directions x 7 repetitions)",
                 float(np.mean([len([r for r in rows if r['subject'] == s and r['block'] == 'block 1']) for s in {r['subject'] for r in rows}])),
                 2 * 3 * 3 * 2 * 7, 0, "Methods, Design (252 trials per block)")
    ck.published("conditions per block (each repeated 7 times)", float(np.mean([len(c) for c in n_cond])), 36, 0,
                 "Methods, Design (2 x 3 x 3 x 2 conditions, 7 repetitions each)")
    ck.internal("each condition appears 7 times per block in every file", all(set(c.tolist()) == {7} for c in n_cond))
    ck.published("gaze sampling interval (ms)", float(np.median(dts)) * 1000, 2.0, 0.01, "Methods, Apparatus (Eyelink II at 500 Hz)")
    ck.internal("interception trials: hit plus miss equals all but the trials with no recorded success flag",
                hits + misses >= 5540, f"{hits} hits, {misses} misses")
    ck.internal("one row per trial: 11 participants x 2 tasks x 504 trials", len(rows) == 11 * 2 * 504 and len(files) == len(rows))

    meta = dict(
        study=DS,
        paper=dict(authors="Goettker A, Brenner E, Gegenfurtner KR, de la Malla C", year=2019,
                   title="Corrective saccades influence velocity judgments and interception", venue="Sci Rep 9:5395",
                   doi="10.1038/s41598-019-41857-z"),
        lab="Gegenfurtner (Giessen) / Brenner & de la Malla (VU Amsterdam)",
        data_url="https://doi.org/10.5281/zenodo.2566255",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("The authors' MATLAB tables (one sample every 2 ms for a whole block of 252 trials) were cut into one CSV per trial with a time column "
                 "and a flag for the period after target movement onset; trial-level values (condition, success, tap position, answer, "
                 "target position) are in trial_info. Participant file names (initials) were replaced by codes."),
        unit_note="trial (recording) of one participant",
        how=["One row is one trial of one participant in one of two tasks: interception (a moving target is tapped on a screen, finger "
             "position recorded) or velocity judgment (is the target faster or slower than a reference; no hand movement). Each task has 2 "
             "blocks of 252 trials (2 target crossing times x 3 presentation durations x 3 target speeds x 2 directions x 7 repetitions); "
             "trial and recording number the trials 1-504 in the order of the blocks (block 1 first as in the file names).",
             "The samples file has time_s, gaze position of the left and right eye (screen cm; the paper gives gaze positions in cm on the screen), "
             "the finger position (interception only, x y z as in the authors' file; mm assumed from the Optotrak system, the paper "
             "states no unit for the raw values) and after_target_onset. The authors' Time column restarts at 0 when the target starts moving: "
             "the first part of a trial (fixation and waiting) has finger values empty and the second part runs from 0 to the response. "
             "time_s is the authors' value and is not made continuous.",
             "trial_info: presentation_ms (600, 650 or 700), target_speed_cm_s (13, 15, 17), crossing_time_ms (100 or 240; the authors' 'ex'), "
             "direction, target_x_cm (initial target position from the authors' file, cm), target_velocity_cm_s (signed), and for "
             "interception success (hit or miss), tap position in cm, target position at the tap and time_to_tap_s; for velocity judgment the answer "
             "(faster or slower)."],
        differences=["The paper analysed 10 of the 11 judgment participants (one had incomplete tracking) and removed trials with anticipatory pursuit, blinks, "
                     "large position errors or taps more than 3 cm from the target (4951 of 5554 interception trials, 4729 of 5040 judgment trials kept); "
                     "the deposit has no per-trial exclusion flags and does not say which participant was dropped, so all 5544 trials per task are rows "
                     "and excluded_by_authors is False throughout.",
                     "The paper gives 5554 interception trials, which is not a multiple of 504; the deposit has 11 x 504 = 5544.",
                     f"The paper reports 502 taps missed online of which 141 were recovered offline; the deposit has {notap} interception trials with no tap recorded."],
    )
    stdlib.release(DS, "timeseries", rows, meta, ck, extra_files=files)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
