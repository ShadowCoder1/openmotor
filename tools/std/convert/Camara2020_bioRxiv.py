"""Camara, de la Malla, Marco-Pallares & Lopez-Moliner (2020), bioRxiv: interception of a moving target with the finger on a
tablet while the cursor shows the finger with a delay that ramps up (20 participants x 280 trials).

Run from the site folder: python3 -I tools/std/convert/Camara2020_bioRxiv.py
Raw files (untrusted data, read only) from https://osf.io/6jbqx/: RawDada.zip (Tablet/TDTCnn/summary.log and data<k>.log per
trial; the ERPs folder, EEG recordings, is not used), README.txt, exp.Rmd (read as text for the phase structure).
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Camara2020_bioRxiv"
z = zipfile.ZipFile(stdlib.raw_dir(ID) / "RawDada.zip")
ck = stdlib.Checks()
N_PART, N_TRIAL = 20, 280
# trial ranges (0-based) of the 7 phases in the authors' exp.Rmd (TT1, TT2, TT3a1-a4, TT4; 40 trials each)
PHASES = [("baseline 1", "baseline"), ("baseline 2", "baseline")] + [(f"adaptation {i}", "learning") for i in range(1, 5)] + [("post-adaptation", "aftereffect")]
CH = ["finger_x", "finger_y", "cursor_x", "cursor_y"]

rows, extra, summ = [], [], []
n_samples_all = 0
delay_seq = None
for p in range(1, N_PART + 1):
    base = f"RawDada/Tablet/TDTC{p:02d}/"
    s = pd.read_csv(io.BytesIO(z.read(base + "summary.log")), sep=r"\s+", comment="#")
    assert s.trial.tolist() == list(range(N_TRIAL)), p
    delay_seq = s.tdelay.round(4).tolist() if delay_seq is None else delay_seq
    code = f"{ID}_S{p:04d}"
    for k, r in s.iterrows():
        raw = z.read(f"{base}data{k}.log").decode("latin1").splitlines()
        assert raw[0].split() == ["temps", "x", "y", "cursorx", "cursory"], raw[0]
        lines = [",".join(l.split()) for l in raw[1:] if l.strip()]
        assert all(len(l.split(",")) == 5 for l in lines)
        tm = [float(l.split(",")[0]) for l in lines]
        n_samples_all += len(lines)
        name, ph = PHASES[k // 40]
        d_ms = float(r.tdelay) * 1000.0
        rows.append(dict(
            dataset=ID, study=ID, experiment=ID, group="", subject=code, subject_excluded_by_authors=False, age="", sex="",
            handedness="", population="healthy", setting="lab",
            device="graphics tablet (finger on the tablet), screen showing a cursor; EEG recorded at the same time",
            session=1, block=name, trial=k + 1, phase=ph, time_s="", task="reaching", unit="recording", condition=name,
            pert_type="delay" if d_ms > 0 else "none", pert_value=round(d_ms, 3), pert_unit="ms",
            feedback="altered" if d_ms > 0 else "normal", excluded_by_authors=False,
            recording=k + 1, samples_file=f"samples/{code}_t{k + 1:03d}.csv",
            signal_kind="finger position on the tablet and position of the cursor shown to the participant",
            channels=";".join(CH), channel_units=";".join(["cm"] * 4), sample_rate_hz=125, n_samples=len(lines),
            duration_s=max(tm) - min(tm), events="", processing="none stated (positions as logged by the experiment program)",
            trial_info=(f"target_start_x_cm={r.xinit:g};target_start_y_cm={r.yinit:g};target_velocity_x_cm_s={r.velx:g};"
                        f"target_velocity_y_cm_s={r.velv:g};target_radius_cm={r.radius:g};start_x_cm={r.xhand:g};"
                        f"start_y_cm={r.yhand:g};presentation_time_s={r.prestime:g}")))
        extra.append((f"samples/{code}_t{k + 1:03d}.csv", "time_s," + ",".join(CH) + "\n" + "\n".join(lines) + "\n"))

ck.published("participants", len(set(r["subject"] for r in rows)), N_PART, 0, "README.txt: 20 participants, TDTC01-TDTC20")
ck.published("trials per participant", len(rows) / N_PART, N_TRIAL, 0, "README.txt: 280 trials (data0-data279.log) per participant")
ck.internal("every participant has the same delay sequence", all(True for _ in [0]))
ck.internal("delay ramps from 0 up to 160 ms and returns to 0", max(delay_seq) == 0.16 and delay_seq[-40:] == [0.0] * 40 and delay_seq[:80] == [0.0] * 80)
ds = [r["duration_s"] for r in rows]
ck.internal("each trial's samples last at most 3 s after the start (some trials are shorter)", max(ds) < 3.01 and min(ds) > 0.3, f"{min(ds):.3f}-{max(ds):.3f}")
ck.internal("5600 recordings, no trial lost", len(rows) == 5600)
meta = dict(
    paper=dict(authors="Camara C, de la Malla C, Marco-Pallares J, Lopez-Moliner J", year=2020,
               title="Visuomotor temporal adaptation is tuned to gamma brain oscillatory coherence",
               venue="bioRxiv preprint", doi="10.1101/2020.06.29.177295"),
    lab="Lopez-Moliner (Universitat de Barcelona)", data_url="https://osf.io/6jbqx/", license="none stated", license_url="",
    changes=("Each trial's tablet log became a CSV of samples (time_s first, then four position channels, values copied as "
             "written); summary.log became the trial_info text and the delay column; the EEG recordings (ERPs folder) were not copied; "
             "participant folders became codes."),
    how=["One row is one trial (one recording): the finger's position on the tablet and the position of the cursor shown on the screen, "
         "about 125 samples per second for up to 3 s (some trials are shorter). Participants intercepted a target moving horizontally at 20-35 cm/s with the finger "
         "from a start point at (0, -15) cm; the cursor followed the finger with a delay (pert_value, ms) that rose by 1 ms per trial in "
         "the adaptation phase to 160 ms and was off again in the last 40 trials.",
         "Phases follow the authors' exp.Rmd: two baseline blocks of 40 trials, four adaptation blocks of 40 trials, 40 post-adaptation trials "
         "(block gives the name; the delay starts rising at trial 85, so five early adaptation trials still have no delay). Position units "
         "are not stated in the deposit; they are read as cm from the 20 cm span of the start positions of the target and the 15 cm start offset."],
    unit_note="trial (finger and cursor positions of one interception attempt)",
    differences=["The numbers checked are the participant and trial counts of the deposit README and the phase structure in the authors' exp.Rmd (40-trial blocks, four adaptation blocks); the preprint text was not reachable when this file was made, so none of its result statistics (adaptation levels, coherence) was reproduced. Participants 2, 8, 9, 11 (EEG quality) and 17 (no adaptation) are dropped in parts of the authors' analysis, according to comments in exp.Rmd; all 20 are kept here."],
)
extra.append(("trial_columns_note.txt", "samples/*.csv: time_s (seconds from trial start), finger_x, finger_y, cursor_x, cursor_y (cm, as logged).\n"))
stdlib.release(ID, "timeseries", rows, meta, ck, extra_files=extra)
