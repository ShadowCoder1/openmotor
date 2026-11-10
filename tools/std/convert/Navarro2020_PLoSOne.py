"""Navarro et al. (2020), PLoS ONE 15(11):e0242818: eye-hand coordination in steering. 18 drivers drove a bend track in a driving
simulator with regular steering (manual) and with automated steering, wearing a head-mounted eye tracker. Per participant and drive
the deposit has an eye-tracker file (50 Hz gaze) and a simulator log (about 60 Hz, steering wheel in column 9). One row per file.

Run from the site folder: python3 -I tools/std/convert/Navarro2020_PLoSOne.py
Raw files (untrusted data, read only) in raw_extra/Navarro2020_PLoSOne/ from https://osf.io/sbg65/: subjects.zip,
Sauto_VisuDrive.txt, sync_data.csv, columns_VisuDrive.csv, readme.txt.
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

ID = "Navarro2020_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
STUDY = "Navarro et al. 2020, PLoS ONE"
DEVICE = ("fixed-base driving simulator (three 22-inch screens, Logitech G27 steering wheel and pedals, automatic gearbox) "
          "with a head-mounted SMI iView X eye tracker (50 Hz)")

SIMCOLS = ["speed_self", "lane_dist_self", "x_self", "y_self", "z_self", "gas_self", "brake_self", "steering_self", "speed_leading",
           "lane_dist_leading", "x_leading", "y_leading", "z_leading", "dist_self_leading", "highway", "speed_control", "roadworks",
           "town", "cross", "ambulance", "pedestrian", "red_light", "bends", "follow_the_leading", "accident"]
# columns_VisuDrive.csv names 26 columns: time then these 25
EYE_CH = ["gaze_x", "gaze_y"]

z = zipfile.ZipFile(RAW / "subjects.zip")
names = sorted(n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith("/") and "DS_Store" not in n)
folders = sorted({n.split("/")[0] for n in names})
assert len(folders) == 18

sync = {}
for l in (RAW / "sync_data.csv").read_text(encoding="utf-8-sig").splitlines()[1:]:
    k, v = l.split(";")
    sync[k.strip()] = float(v)

HEADER = (RAW / "columns_VisuDrive.csv").read_text(encoding="utf-8-sig").strip().split(";")
assert len(HEADER) == 26 and HEADER[0].strip() == "time"


def sim_df(b):
    t = pd.read_csv(io.BytesIO(b), sep="\t", header=None, decimal=",", encoding="latin1", dtype=str, usecols=range(26))
    tm = t[0].str.extract(r"^(\d+):(\d+):(\d+\.\d+)$").astype(float)
    ts = tm[0] * 3600 + tm[1] * 60 + tm[2]
    out = t.iloc[:, 1:].apply(lambda c: c.str.replace(",", ".").astype(float))
    out.columns = SIMCOLS
    out.insert(0, "time_s", ts.values)
    return out


def eye_df(b):
    e = pd.read_csv(io.BytesIO(b), sep="\t")
    assert (e["Type"] == "SMP").all()
    out = pd.DataFrame({"time_s": (e["Time"].values - e["Time"].values[0]) / 1e6, "gaze_x": e["R POR X [px]"].values,
                        "gaze_y": e["R POR Y [px]"].values})
    return out, e["Time"].values[0] / 1e6


auto_sim = sim_df((RAW / "Sauto_VisuDrive.txt").read_bytes())
auto_csv = None

rows, extra = [], []
eye_dt, sim_dt = [], []
n_pers = 0
for pi, fo in enumerate(folders, start=1):
    num = int(fo[1:])
    code = f"{ID}_S{pi:04d}"
    recs = []
    own_sim = sim_df(z.read(f"{fo}/S{num}_VisuDrive.txt"))
    for cond, ename, key in [("regular steering (manual)", "manuel", "manuel"), ("automated steering", "automatique", "auto")]:
        eye, t0 = eye_df(z.read(f"{fo}/S{num}_{ename} Samples.txt"))
        off = sync[f"S{num}_{key}"]
        sim = own_sim if key == "manuel" else auto_sim
        recs.append(("eye", cond, key, eye, off))
        recs.append(("sim", cond, key, sim, off))
    for ri, (kind, cond, key, df, off) in enumerate(recs, start=1):
        fn = f"samples/{code}_{'eye' if kind == 'eye' else 'simulator'}_{'manual' if key == 'manuel' else 'automated'}.csv"
        extra.append((fn, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
        t = df["time_s"].values
        dtm = float(np.median(np.diff(t)))
        (eye_dt if kind == "eye" else sim_dt).append(dtm)
        if kind == "eye":
            chan, units = EYE_CH, ["px", "px"]
            sk = "gaze position (right eye point of regard in the scene-camera image); 0,0 marks samples with no gaze"
            proc = "none (raw tracker samples; the Type, Trial and empty Aux1 columns, constant in every file, were dropped; time_s counts from the first sample)"
        else:
            chan, units = SIMCOLS, ["unit not documented"] * len(SIMCOLS)
            sk = ("driving simulator log: speed, lane position, positions, pedal and steering wheel values of the car and of the leading car, event flags"
                  + ("" if key == "manuel" else " (the same automated drive log for every participant)"))
            proc = "none (raw simulator log; decimal commas became decimal points and the hh:mm:ss.mmm clock became seconds; four empty trailing columns dropped)"
        rows.append(dict(
            dataset=ID, study=STUDY, experiment="Regular versus automated steering", group="", subject=str(pi), person=str(pi),
            subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab", device=DEVICE,
            session=1, block=cond, trial=ri, phase="test", time_s="", task="tracking", unit="recording", condition=cond,
            pert_type="none" if key == "manuel" else "automated_steering", pert_value=0 if key == "manuel" else 1,
            pert_unit="" if key == "manuel" else "on", feedback="normal", excluded_by_authors=False,
            recording=ri, samples_file=fn, signal_kind=sk, channels=";".join(chan), channel_units=";".join(units),
            sample_rate_hz=round((len(t) - 1) / float(t[-1] - t[0]), 2), n_samples=len(df), duration_s=float(t[-1] - t[0]), events="", processing=proc,
            trial_info=f"red_light_onset_in_eye_file_s={off}"))
D = pd.DataFrame(rows)

P = checks.published
P("participants", D.person.nunique(), 18, 0, "2.1 Participants (eighteen participants)")
P("drives per participant (regular and automated)", D.groupby("person").condition.nunique().mean(), 2, 0, "2.3 Procedure (two drives)")
P("eye-tracker sample rate (Hz)", 1 / np.median(eye_dt), 50, 0.5, "2.2.2 Eye movement tracker (50 Hz)")
checks.internal("4 rows per participant, 72 rows", len(D) == 72 and (D.groupby("person").size() == 4).all())
checks.internal("every row has a samples file", len(extra) == 72 and D.n_samples.min() > 40000)
checks.internal("sync offsets found for 36 drives", len(sync) == 36)

meta = dict(
    paper=dict(authors="Navarro J, Hernout E, Osiurak F, Reynaud E", year=2020,
               title="On the nature of eye-hand coordination in natural steering behavior", venue="PLoS ONE 15(11):e0242818",
               doi="10.1371/journal.pone.0242818"),
    lab="Navarro / Reynaud (Universite Lumiere Lyon 2)",
    data_url="https://osf.io/sbg65/", license="none stated (the paper is CC BY 4.0)", license_url="",
    changes=("Each eye-tracker file and each simulator log of the 18 participants became one samples CSV, four per participant (eye and simulator, regular and automated "
             "drive). Participant folder numbers were replaced by codes. Eye files: time_s counts seconds from the file's first sample (the tracker's microsecond clock); "
             "constant Type, Trial and empty Aux1 columns were dropped. Simulator logs: the hh:mm:ss.mmm clock became seconds, decimal commas became points, and the "
             "four empty trailing columns were dropped. The automated-steering simulator log (Sauto_VisuDrive.txt) is one file for all participants and is copied into each "
             "participant's rows. Age and sex are not shared per participant (paper: 8 females, 10 males, 18 to 29 years). The synchronisation value of sync_data.csv is in trial_info."),
    how=["One row is one file of one participant's drive on a 2650 m track of 10 bends in a driving simulator: either the right-eye gaze samples of the head-mounted tracker "
         "(50 Hz, gaze_x and gaze_y in pixels of the scene image, 0,0 where no gaze was found) or the simulator log (about 60 Hz). In the regular (manual) drive the participant steered; "
         "in the automated drive the car followed the road with the wheel and hands still, and the log's steering column holds the wheel angle the road would have required.",
         "Simulator log columns follow the authors' columns_VisuDrive.csv; units are not documented (the steering wheel value, column 9, ranges about -1 to 1 so it appears normalised; "
         "speed is probably km/h, an inference). Eye and simulator clocks are different: red_light_onset_in_eye_file_s is the authors' value from sync_data.csv, the time in the "
         "eye-tracker file at which the red-light flag (simulator column 23) first switches from 0 to 1; whether it counts from the eye file's first sample is inferred.",
         "The order in which each participant drove the two conditions was counterbalanced but is not shared, so recordings are listed regular first, then automated."],
    differences=["The paper says the steering signal was originally sampled at 60 Hz; the simulator logs have a median interval of about 17 ms (58.8 Hz, intervals alternate between 16 and 17 ms) and an average rate of about 60 Hz, "
                 "so this was not used as a check. sample_rate_hz is the average rate (samples per second of recording).",
                 "The paper's correlation and lag results rely on preprocessing (outlier removal, normalisation, saccade removal) not available in the deposit, so only the design "
                 "numbers (participants, drives, sample rates) are checked.",
                 "The paper says the track is 2650 m; the simulator logs cover a longer drive (about 16.7 km of positions in about 16 to 18 minutes), so they probably include more "
                 "road than the ten bends."],
    unit_note="recording (one eye-tracker file or one simulator log)",
)
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
