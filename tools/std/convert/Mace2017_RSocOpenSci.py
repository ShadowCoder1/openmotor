"""Mace, Rinne, Liardon, Uhomoibhi, Bentley & Burdet (2017) R Soc open sci 4:160961: 66 healthy adults tracking a moving target by
squeezing an elastic or a rigid handgrip. 34 did the feedforward task (path visible ahead, game spaceWay: Sin2Chirp and PRBS
trajectories), 32 the feedback task (path hidden, game StarGame: OneC and OneB trajectories). One row per recording (about 2 min at 50 Hz);
samples (handgrip force, target and player position, game score) are CSV files inside the ZIP.
Raw file: raw_extra/Mace2017_RSocOpenSci/mace.zip (Zenodo 5024391 = Dryad 10.5061/dryad.m68q0, CC0). Read as text only.
Run from the site folder: python3 -I tools/std/convert/Mace2017_RSocOpenSci.py"""
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mace2017_RSocOpenSci"
MODULE = "timeseries"
ROOT = "MMace CVR study data/"
FF_COLS = [("Frame", "frame", "frame"), ("Time", "time_s", "s"), ("RefPath_X", "ref_path_x", "game units"), ("RefPath_Y", "ref_path_y", "game units"),
           ("PlayerPath_X", "player_path_x", "game units"), ("PlayerPath_Y", "player_path_y", "game units"),
           ("Force", "force", "device units"), ("Score", "score", "points")]
FB_COLS = [("Frame", "frame", "frame"), ("Time", "time_s", "s"), ("Target", "target", "game units"), ("Player", "player", "game units"),
           ("Force", "force", "device units"), ("DwellTime", "dwell_time", "device units"), ("Coins", "coins", "count"), ("Rewards", "rewards", "count")]
DEVICE = {"C": "elastic (compliant) handgrip", "R": "rigid handgrip"}


def key_of(name):
    m = re.match(r"(CVR|PRC)0*(\d+)([CR]{2})", name)
    return (m[1], int(m[2]), m[3])


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(ID) / "mace.zip")
    # questionnaire
    q = {}
    for sh, exp in (("Feedforward task", "feedforward"), ("Feedback task", "feedback")):
        x = pd.read_excel(io.BytesIO(z.read(ROOT + "CVR_questionnaire_data.xlsx")), sheet_name=sh, header=None, skiprows=3)
        x = x[x[0].notna()]
        for _, r in x.iterrows():
            k = key_of(str(r[0]))
            assert k not in q
            q[k] = dict(exp=exp, order=r[1], age=float(r[2]), sex=r[3], hand=float(r[4]), q=[r[i] for i in range(5, 17)], gaming=r[17], pref=r[18])
            assert q[k]["order"] == k[2]
    # files
    files = {}
    for i in z.infolist():
        m = re.match(rf"{re.escape(ROOT)}(Feedback|Feedforward) task/([^/]+)/([^/]+)/(.+)_(\d{{6}})_(\d{{3,4}})\.txt$", i.filename)
        if i.filename.endswith(".txt"):
            assert m, i.filename
            hhmm = int(m[6])
            files.setdefault(key_of(m[2]), []).append((m[5][4:6] + m[5][2:4] + m[5][0:2], hhmm, m[3], i.filename, m[1]))
    assert set(files) == set(q), (set(files) ^ set(q))
    tmp = Path("tmp_" + ID)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rows, extra, order_log = [], [], []
    for k in sorted(files, key=lambda k: (q[k]["exp"], k[1], k[0])):
        info = q[k]
        fl = sorted(files[k])  # date (yymmdd), time of day
        second_task = "StarGame_OneB" if info["exp"] == "feedback" else None
        dev_idx, seq, seen = 0, 0, set()
        counts = {}
        if info["exp"] == "feedforward":
            first_task = fl[0][2]
        for (d, t, folder, fn, kind) in fl:
            if info["exp"] == "feedforward":
                if folder in seen:
                    dev_idx += 1
                    seen = set()
            if dev_idx == 0 and not (info["exp"] == "feedback" and seen == {"__done"}):
                pass
            dev_i = dev_idx
            seen.add(folder)
            if info["exp"] == "feedback" and folder == second_task:
                dev_idx += 1
            dev_letter = k[2][dev_i] if dev_i < 2 else "?"
            counts[(dev_i, folder)] = counts.get((dev_i, folder), 0) + 1
            attempt = counts[(dev_i, folder)]
            lines = z.read(fn).decode("latin1").splitlines()
            hi = [j for j, l in enumerate(lines) if l.strip()][0]
            if not lines[hi].startswith("Frame"):
                raise AssertionError((fn, "no header", lines[hi][:60]))
            head = lines[hi].split()
            cols = FF_COLS if info["exp"] == "feedforward" else FB_COLS
            assert [h.split("[")[0] for h in head[:8]] == [c[0] for c in cols], (fn, head[:8])
            arr = np.array([l.split()[:8] for l in lines[hi + 1:] if l.strip()], float)
            seq += 1
            fname = f"r{len(rows) + 1:03d}.csv"
            with open(tmp / fname, "w", newline="") as fh:
                fh.write("time_s," + ",".join(c[1] for c in cols if c[1] != "time_s") + "\r\n")
                for r in arr:
                    fh.write(f"{r[1]:.6g}," + ",".join(f"{v:.8g}" for j, v in enumerate(r) if j != 1) + "\r\n")
            extra.append((f"samples/{fname}", str(tmp / fname)))
            names = [c[1] for c in cols if c[1] != "time_s"]
            units = [c[2] for c in cols if c[1] != "time_s"]
            rows.append(dict(
                dataset=ID, study="Mace et al. 2017 R Soc open sci", experiment=info["exp"], group=k[2], subject=f"{k[0]}{k[1]}{k[2]}", person=f"{k[0]}{k[1]}{k[2]}",
                subject_excluded_by_authors=False, age=info["age"], sex=info["sex"], population="healthy", setting="lab",
                device=DEVICE.get(dev_letter, "?") + " with force sensor", session=1, block=folder, trial=seq, phase="test", time_s=np.nan,
                task="grasping and lifting", unit="recording", condition=("elastic" if dev_letter == "C" else "rigid") + " handgrip",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=seq, samples_file=f"samples/{fname}",
                signal_kind="isometric handgrip force, with target and player position in a visuomotor tracking game",
                channels=";".join(names), channel_units=";".join(units), sample_rate_hz=50, n_samples=len(arr), duration_s=float(arr[:, 1].max()),
                events="", processing="none beyond what the authors' game software logged (force: 4th-order Butterworth low-pass per the paper)",
                trial_info=f"device_block={dev_i + 1};attempt_in_block={attempt};file_time={d}_{t:04d}"))
            order_log.append((info["exp"], folder, dev_i + 1, attempt))
    df = pd.DataFrame(rows)
    df["handedness"] = ""
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"])
    pp = pd.DataFrame([dict(person=f"{k[0]}{k[1]}{k[2]}", experiment=v["exp"], first_device="elastic" if k[2][0] == "C" else "rigid",
                            age=v["age"], sex=v["sex"], handedness_score=v["hand"], gaming_q=v["gaming"], preferred_device_q=v["pref"],
                            **{f"{d}_q{j + 1}": v["q"][6 * di + j] for di, d in enumerate(["first", "second"]) for j in range(6)})
                       for k, v in q.items()])
    ck = stdlib.Checks()
    ff, fb = pp[pp.experiment == "feedforward"], pp[pp.experiment == "feedback"]
    S = "Methods (Participants) and Table 1"
    ck.published("participants, feedforward task", len(ff), 34, 0, S)
    ck.published("participants, feedback task", len(fb), 32, 0, S)
    ck.published("participants in total", len(pp), 66, 0, "Abstract")
    ck.published("mean age feedforward (years)", ff.age.mean(), 43.0, 0.1, S)
    ck.published("SD age feedforward (years)", ff.age.std(), 17.7, 0.1, S)
    ck.published("mean age feedback (years)", fb.age.mean(), 37.8, 0.1, S)
    ck.published("SD age feedback (years)", fb.age.std(), 15.0, 0.1, S)
    ck.published("mean age all (years)", pp.age.mean(), 40.4, 0.1, S)
    ck.published("women, feedforward", (ff.sex == "F").sum(), 15, 0, S)
    ck.published("women, feedback", (fb.sex == "F").sum(), 18, 0, S)
    ck.published("mean handedness score feedforward", ff.handedness_score.mean(), 85.5, 0.1, S)
    # preference question: positive = second device tried preferred
    soft = ((pp.first_device == "rigid") & (pp.preferred_device_q > 0)) | ((pp.first_device == "elastic") & (pp.preferred_device_q < 0))
    rig = ((pp.first_device == "elastic") & (pp.preferred_device_q > 0)) | ((pp.first_device == "rigid") & (pp.preferred_device_q < 0))
    ck.published("percent preferring the elastic handgrip", 100 * soft.mean(), 62.1, 0.1, "Results (user experience)")
    ck.published("percent preferring the rigid handgrip", 100 * rig.mean(), 21.2, 0.1, "Results (user experience)")
    ck.published("rigid-first participants preferring elastic", (soft & (pp.first_device == "rigid")).sum(), 28, 0, "Results (order effects)")
    ck.internal("every participant has recordings for both devices", df.groupby("subject").trial_info.apply(
        lambda s: s.str.contains("device_block=1").any() and s.str.contains("device_block=2").any()).all())
    print(df[~df.duration_s.between(119, 121)][["subject", "block", "duration_s", "trial_info"]].to_string())
    ck.internal("all but one recording last 119-121 s (one lasts 222 s)", (~df.duration_s.between(119, 121)).sum() == 1, f"{df.duration_s.min()}-{df.duration_s.max()}")
    ck.internal("265 recordings, one sample file each", len(df) == 265 and len(extra) == 265, str(len(df)))
    meta = dict(
        paper=dict(authors="Mace M, Rinne P, Liardon JL, Uhomoibhi C, Bentley P, Burdet E", year=2017,
                   title="Elasticity improves handgrip performance and user experience during visuomotor control",
                   venue="R Soc open sci 4(2):160961", doi="10.1098/rsos.160961"),
        lab="Burdet (Imperial College London)", data_url="https://doi.org/10.5061/dryad.m68q0",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Converted the game log text files into one CSV per recording with the authors' values unchanged (frame, time, target and "
                 "player positions, handgrip force, score or coin counts). The ten obstacle/distractor position columns of each game log "
                 "(meteors_X/Y, distractors_X/Y) were left out. Participant names were replaced by codes; the questionnaire free-text "
                 "comments were left out. Age, sex, handedness score and questionnaire answers come from the deposit's spreadsheet."),
        how=["One row is one recording of about 2 minutes (6000 frames at 50 Hz) in which a participant squeezed an elastic or a rigid handgrip "
             "to move a cursor along a displayed path. Feedforward task (game spaceWay): the path is visible ahead; the two trajectories are "
             "Sin2Chirp and PRBS (block). Feedback task (game StarGame): the path is hidden and a target is followed; the two trajectories are "
             "folders OneC and OneB (block).",
             "force is the raw sensor reading in device units (the deposit gives no unit; the paper states 0.0288 N per bit, which does not "
             "reproduce Newtons from these values, so no conversion was made). Positions are in game units. Feedforward logs begin with "
             "duplicated padding rows (Frame 0) and repeat the last row; all logged rows are kept.",
             "The device used in a recording was not labelled in the files. It was derived from the group letters in the participant code "
             "(CR = elastic then rigid, RC = rigid then elastic) and the recording clock times (two recordings per device, in order). "
             "This reproduces the paper's preference counts (see checks). trial_info gives device_block (1 or 2) and attempt_in_block. "
             "In the feedback task OneC was always recorded first in a device block, as the paper describes for the sin-to-chirp trajectory; "
             "OneB second (harmonic series); this pairing is inferred from timing, not documented."],
        unit_note="recording (one 2-minute game trial)",
        differences=["The questionnaire has one preference answer per participant; positive values mean the second device tried was preferred "
                     "(inferred, as it reproduces 62.1% and 21.2% in the paper). One feedback participant has 5 recordings (an extra first attempt in one trajectory; attempt_in_block=2 marks the later one).",
                     "One feedback recording (the first participant of the PRC series, first device) lasts 222 s instead of 120 s; it is kept as logged.",
                     "The paper's minimum moving error (MME) was not recomputed."])
    part_cols = [("experiment", "text", "", "feedforward or feedback task."), ("first_device", "text", "elastic, rigid", "Device used first."),
                 ("age", "number", "years", "Age."), ("sex", "text", "F, M", "Sex."),
                 ("handedness_score", "number", "-100 to 100", "Handedness inventory score as in the deposit (Edinburgh scale)."),
                 ("gaming_q", "number", "", "Gaming-frequency answer (QG in the paper)."),
                 ("preferred_device_q", "number", "-2 to 2", "Preference question: positive = second device used preferred, negative = first, 0 = none (inferred)."),
                 ] + [(f"{d}_q{j + 1}", "number", "-2 to 2", f"Five-point answer to question {j + 1} for the {d} device (paper: Q1 performance, Q2 enjoyment, Q3 discomfort, Q4 improvement, Q5 fatigue, Q6 control; scale signs not documented).") for d in ("first", "second") for j in range(6)]
    try:
        stdlib.release(ID, MODULE, df, meta, ck, participants=pp, participant_columns=part_cols, extra_files=extra)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))
    print(pd.Series(order_log).value_counts().head(12))


main()
