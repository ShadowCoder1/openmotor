"""Delgado-Teran et al. (2025), Sensors 25(6):1895: right-ankle IMU windows (2 s, 60 Hz) of 21 people with Parkinson's disease and daily
freezing of gait (FoG), recorded in semi-free living, with window labels. One row per participant and session (morning / afternoon).

Run from the site folder: python3 -I tools/std/convert/DelgadoTeran2025_Sensors.py
Raw files (untrusted data, read only, loaded without pickles) in raw_extra/DelgadoTeran2025_Sensors/ from
https://doi.org/10.4121/40e06061-f441-43b5-9235-006829206509.v1 : PD0XX_AllActivities_FOG.npz (xTensor, yTensor, Metadata), README.txt.
The Walking-Turning files are, per the paper, a subset of All-Activities and are not used.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "DelgadoTeran2025_Sensors"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

ACT = {"Sitting": 1, "Standing": 2, "Walking": 3, "Small": 4, "Making": 5}      # Small = small steps, Making = making turns (first word of the authors' label)
FOGT = {"Shuffling": 1, "Trembling": 2, "Akinesia": 3}
TYPE = {"Mono-Type": 1, "Multi-Type": 2}
CH = ["window", "activity_code", "fog_type_code", "type_code", "fog", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CU = "window number;activity code;FoG type code;mono/multi code;0/1;g;g;g;deg/s;deg/s;deg/s"
SESS = {"Morning": "morning, off medication (levodopa withdrawn)", "Afternoon": "afternoon, on medication"}
files = sorted(RAW.glob("PD*_AllActivities_FOG.npz"))
rows, extra = [], []
tot_w = tot_fog = 0
first_n = None
for rank, f in enumerate(files, 1):
    pn = int(re.match(r"PD(\d+)_", f.name).group(1))
    z = np.load(f, allow_pickle=False)
    X, y, M = z["xTensor"], z["yTensor"], z["Metadata"]
    assert X.shape[1:] == (120, 6) and len(X) == len(y) == len(M)
    if pn == 1:
        first_n = len(y)
    parsed, ids = [], set()
    for s_ in M:
        p = s_.split(" | ")
        m = re.match(r"\d{4}-\d{2}-\d{2}_(PD\d+)\[(\w+)\]", p[0])
        ids.add(m.group(1))
        w, t = re.match(r"(\w+)\((.*)\)", p[2]).groups()
        parsed.append((m.group(2), w, t, p[3] if len(p) == 5 else "FoG"))
    assert len(ids) == 1
    sess = np.array([q[0] for q in parsed])
    for sname in [s_ for s_ in ("Morning", "Afternoon") if s_ in set(sess)]:
        idx = np.where(sess == sname)[0]
        n = len(idx)
        lab = np.array([parsed[i][3] == "FoG" for i in idx])
        assert (lab == (y[idx] == 1)).all()
        df = pd.DataFrame({
            "time_s": np.tile(np.arange(120) / 60.0, n), "window": np.repeat(np.arange(1, n + 1), 120),
            "activity_code": np.repeat([ACT.get(parsed[i][1], 0) for i in idx], 120),
            "fog_type_code": np.repeat([FOGT.get(parsed[i][1], 0) for i in idx], 120),
            "type_code": np.repeat([TYPE[parsed[i][2]] for i in idx], 120), "fog": np.repeat(y[idx], 120)})
        for j, c in enumerate(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]):
            df[c] = X[idx, :, j].ravel()
        fn = f"samples/{ID}_S{rank:04d}_{sname.lower()}.csv"
        extra.append((fn, df[["time_s"] + CH].to_csv(index=False, float_format="%.8g", lineterminator="\r\n").encode()))
        nf = int((y[idx] == 1).sum())
        tot_w += n
        tot_fog += nf
        rows.append(dict(
            dataset=ID, study=ID, experiment=ID, group="", subject=str(rank), person=str(rank), subject_excluded_by_authors=False,
            age="", sex="", handedness="", population="Parkinson's disease with daily freezing of gait", setting="lab",
            device="Movisens Move 4 IMU on the right ankle, 60 Hz (resampled by the authors)",
            session=1 if sname == "Morning" else 2, block=sname.lower(), trial=1 if sname == "Morning" else 2, phase="test", time_s="",
            task="walking", unit="recording", condition=SESS[sname], pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=False, recording=1 if sname == "Morning" else 2, samples_file=fn,
            signal_kind="right-ankle accelerometer and gyroscope in 2 s windows (120 samples) with activity and FoG labels",
            channels=";".join(CH), channel_units=CU, sample_rate_hz=60, n_samples=n * 120, duration_s=n * 2.0, events="",
            processing="3rd-order zero-phase Butterworth band-pass 0.3-15 Hz, resampled to 60 Hz, cut into 2 s windows (50% overlap for non-FoG, 87.5% for FoG) by the authors",
            trial_info=f"windows={n};fog_windows={nf}"))
df = pd.DataFrame(rows)

P = checks.published
P("participants analysed", df.person.nunique(), 21, 0, "Results (24 recruited, 3 without FoG excluded)")
P("window length in samples (2 s at 60 Hz)", 120, 2 * 60, 0, "Methods 2.4, Preprocessing")
P("windows in the first participant's All-Activities file", first_n, 9096, 0, "dataset README (xTensor shape)")
checks.internal("42 recordings: 21 participants x morning and afternoon", len(df) == 42, str(len(df)))
checks.internal("every window kept", df.n_samples.sum() == tot_w * 120, str(tot_w))
checks.internal("FoG windows are the label-1 windows", tot_fog == 57387, str(tot_fog))

meta = dict(
    paper=dict(authors="Delgado-Teran JD, Hilbrants K, Mahmutovic D, Silva de Lima AL, van Wezel RJA, Heida T", year=2025,
               title="Ankle Sensor-Based Detection of Freezing of Gait in Parkinson's Disease in Semi-Free Living Environments",
               venue="Sensors 25(6):1895", doi="10.3390/s25061895"),
    lab="Heida (University of Twente)", data_url="https://doi.org/10.4121/40e06061-f441-43b5-9235-006829206509.v1",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("The authors' NumPy tensors (windows x 120 x 6) were written as one samples CSV per participant and session, windows stacked in the authors' order. "
             "Dates and participant numbers in the authors' window labels were dropped; session (morning or afternoon) and the activity and FoG labels were kept as codes. "
             "Only the All-Activities files are used: the Walking-Turning files are, per the paper, a subset of them."),
    how=["One row is one participant's morning or afternoon session. Per the paper, the morning started after at least 12 h without levodopa (off) and the afternoon after medication (on); "
         "this mapping of morning/afternoon to off/on is read from the Methods, the deposit labels only say Morning and Afternoon.",
         "The samples (samples/<subject>_<session>.csv) are the authors' 2 s windows laid end to end: time_s is the time within the window (0 to 0.983 s) and window numbers the windows in the "
         "authors' order. Windows overlap (50% for non-FoG, 87.5% for FoG windows), so the same samples appear in neighbouring windows; the file order is not necessarily time order, "
         "and duration_s is windows x 2 s, not elapsed time. Channels are right-ankle accelerometer (acc_x..z, assumed g) and gyroscope (gyro_x..z, assumed deg/s); the README gives no units, so these are inferred.",
         "fog is the window label (1 if at least 25% of samples were in an annotated FoG episode). activity_code is the first word of the authors' activity label "
         "(1 sitting, 2 standing, 3 walking, 4 small steps, 5 making turns; 0 for FoG windows, which carry a FoG type instead); fog_type_code is 1 shuffling, 2 trembling, 3 akinesia for FoG windows, 0 otherwise; "
         "type_code is the authors' Mono-Type (1) or Multi-Type (2) tag, which the deposit does not explain. The word reading of these labels is inferred from the paper's activity list."],
    differences=["Numbers in the paper that the windows do not reproduce: 811 FoG episodes (the windows give 1,088 separate runs of FoG windows, and the windows do not mark episodes), 67.78 h of All-Activities "
                 "and 3.61 h of FoG (the 257,243 windows cover about 59.5 h of unique time at 1 s per non-FoG and 0.25 s per FoG window, and 4.0 h for FoG), and a FoG-window share of 22.1% "
                 "(22.3% here). No participant, age or clinical scores are in the deposit."],
    unit_note="recording (one participant's morning or afternoon session)",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
