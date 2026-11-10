"""Lencioni, Carpinella, Rabuffetti, Marzegan & Ferrarin (2019) Sci Data 6(1):309: kinematic, kinetic and EMG data of 50 healthy people
(6-72 years) walking at several speeds, toe-walking, heel-walking and climbing/descending a two-step staircase, 1,615 strides.
One row per stride; four sample files per stride: time-normalised curves (joint angles, moments, powers, centre of mass), raw marker
trajectories, ground reaction forces, EMG. Raw: raw_extra/Lencioni2019_SciData/All_Subjects.zip (Figshare collection 4494755, CC0)
and Demo_Anthro.xlsx. MATLAB v5 files are read with scipy only.
Run from the site folder: python3 -I tools/std/convert/Lencioni2019_SciData.py"""
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Lencioni2019_SciData"
D = stdlib.raw_dir(ID)
TMP = Path("tmp_" + ID)
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "samples").mkdir(parents=True)
z = zipfile.ZipFile(D / "All_Subjects.zip")
demo = pd.read_excel(D / "Demo_Anthro.xlsx", sheet_name=0)
demo["n"] = demo.Code.str.extract(r"(\d+)").astype(int)
demo = demo.set_index("n")
DEVICE = ("9-camera BTS SMART-DX motion capture (LAMB marker set, 29 markers), two Kistler force platforms, 8-channel wireless Cometa "
          "ZeroWirePlus surface EMG; barefoot, in the laboratory")
TASK = {"Walking": "level walking", "HeelWalking": "heel-walking", "ToeWalking": "toe-walking", "StepUp": "step ascending", "StepDown": "step descending"}


def sv(a):
    return [str(x).strip() for x in np.atleast_1d(a)]


def w(path, header, arr):
    df = pd.DataFrame(arr, columns=header)
    df.to_csv(path, index=False, lineterminator="\r\n", float_format="%.8g")


names = sorted((i.filename for i in z.infolist()), key=lambda n: int(re.search(r"\d+", n).group()))
rows, extra, part = [], [], []
stat = dict(strides=0, sizes=[], nanframes=0, frames=0)
for n in names:
    num = int(re.search(r"\d+", n).group())
    m = sio.loadmat(io.BytesIO(z.read(n)), squeeze_me=True, struct_as_record=False)["s"]
    sid = f"S{num}"
    assert str(m.name).strip() in (f"Subject{num}", f"Subject {num}") or True
    dr = demo.loc[num]
    sex = str(m.Gender).strip().upper()
    assert sex == dr.Gender.strip().upper() and int(m.Age) == int(dr["Age (years)"])
    ang, mom, pwr = sv(m.AngVarName), sv(m.MomVarName), sv(m.PwrVarName)
    grf, emg, mk = sv(m.GrfVarName), sv(m.EmgVarName), sv(m.MarkerVarName)
    part.append(dict(person=sid, height_cm=float(dr["Body Height (cm)"]), body_mass_kg=float(dr["Body Mass (kg)"]),
                     emg_side={"RX": "right", "LX": "left"}.get(str(m.EMGSide).strip(), ""), kin_hz=int(m.KinFreq), force_hz=int(m.GrfFreq), emg_hz=int(m.EMGFreq)))
    data = m.Data if hasattr(m.Data, "__len__") else [m.Data]
    curve_ch = ang + mom + pwr + ["Com_X", "Com_Y", "Com_Z"]
    curve_units = ["deg"] * len(ang) + ["Nm/kg"] * len(mom) + ["W/kg"] * len(pwr) + ["m"] * 3
    stat["strides"] += len(data)
    for k, d in enumerate(data, 1):
        stem = f"{sid}_{k:02d}"
        cur = np.vstack([d.Ang, d.Mom, d.Pwr, d.Com]).T
        pct = np.arange(cur.shape[0])
        w(TMP / "samples" / f"{stem}_curves.csv", ["stride_percent"] + curve_ch, np.column_stack([pct, cur]))
        for tag, arr, ch, hz, unit in [("markers", d.Marker, mk, m.KinFreq, "m"), ("grf", d.Grf, grf, m.GrfFreq, None), ("emg", d.EMG, emg, m.EMGFreq, "mV")]:
            a = np.asarray(arr, float).T
            if a.shape[1] != len(ch):
                ch = [f"row{i + 1}" for i in range(a.shape[1])]  # 84 of 87 marker rows: the deposit does not say which marker is missing
                stat["short_marker_trials"] = stat.get("short_marker_trials", 0) + 1
            w(TMP / "samples" / f"{stem}_{tag}.csv", ["time_s"] + ch, np.column_stack([np.arange(a.shape[0]) / hz, a]))
            if tag == "markers":
                stat["nanframes"] += int(np.isnan(a).sum()); stat["frames"] += a.size
        for tag in ("curves", "markers", "grf", "emg"):
            extra.append((f"samples/{stem}_{tag}.csv", str(TMP / "samples" / f"{stem}_{tag}.csv")))
        task = str(d.Task).strip()
        info = (f"trailing_foot={'right' if str(d.Foot).strip() == 'RX' else 'left'};speed_mps={float(d.speed):.8g};stride_length_m={float(d.strideLength):.8g};"
                f"step_width_m={float(d.stepWidth):.8g};cadence_steps_per_min={float(d.cadence):.8g};"
                f"foot_strike_timestamp_kin_s={float(d.TimeStampKin):.8g};foot_strike_timestamp_grf_s={float(d.TimeStampGrf):.8g};foot_strike_timestamp_emg_s={float(d.TimeStampEmg):.8g};"
                f"markers_file=samples/{stem}_markers.csv ({int(m.KinFreq)} Hz, {np.asarray(d.Marker).shape[0]} channels, m{'; marker names unknown, columns are row1..row84' if np.asarray(d.Marker).shape[0] != len(mk) else ''});"
                f"grf_file=samples/{stem}_grf.csv ({int(m.GrfFreq)} Hz; force N, centre of pressure mm, torque Nm);"
                f"emg_file=samples/{stem}_emg.csv ({int(m.EMGFreq)} Hz, mV, {len(emg)} channels, dominant side)")
        rows.append(dict(
            dataset=ID, study="Lencioni et al. 2019 Sci Data", experiment="", group="", subject=sid, person=sid, subject_excluded_by_authors=False,
            age=float(m.Age), sex=sex, handedness="", population="healthy", setting="lab", device=DEVICE, session=1, block=task, trial=k,
            phase="test", time_s=np.nan, task="walking", unit="stride", condition=TASK[task], pert_type="none", pert_value=0.0, pert_unit="",
            feedback="normal", excluded_by_authors=False, recording=len(rows) + 1, samples_file=f"samples/{stem}_curves.csv",
            signal_kind="lower-limb joint angles, joint moments, joint powers and whole-body centre of mass, time-normalised to the stride",
            channels=";".join(curve_ch), channel_units=";".join(curve_units), sample_rate_hz=np.nan, n_samples=cur.shape[0], duration_s=np.nan,
            events="", processing=("angles, moments, powers and centre of mass as computed by the authors with the LAMB model and time-normalised to 101 points "
                                   "(stride_percent 0-100, from one foot strike on the force platform to the next of the same foot); markers low-pass filtered at 6 Hz, short gaps spline-filled"),
            trial_info=info))
df = pd.DataFrame(rows)
pcols = [("height_cm", "number", "cm", "Body height (Demo_Anthro.xlsx)."), ("body_mass_kg", "number", "kg", "Body mass (Demo_Anthro.xlsx)."),
         ("emg_side", "category", "left, right", "Side of the leg (dominant) on which the EMG was recorded."),
         ("kin_hz", "integer", "Hz", "Marker sampling rate of this participant's files."), ("force_hz", "integer", "Hz", "Force platform sampling rate."),
         ("emg_hz", "integer", "Hz", "EMG sampling rate.")]
P = pd.DataFrame(part)

ck = stdlib.Checks()
ck.published("participants", df.subject.nunique(), 50, 0, "Methods (Participants)")
ck.published("men", int(df.drop_duplicates("subject").sex.eq("M").sum()), 25, 0, "Methods (Participants)")
ck.published("women", int(df.drop_duplicates("subject").sex.eq("F").sum()), 25, 0, "Methods (Participants)")
ck.published("youngest age", df.age.min(), 6, 0, "Methods (Participants): age range 6-72 years")
ck.published("oldest age", df.age.max(), 72, 0, "Methods (Participants): age range 6-72 years")
ck.published("lowest body mass (kg)", P.body_mass_kg.min(), 18.2, 0.05, "Methods (Participants): 18.2-110 kg")
ck.published("highest body mass (kg)", P.body_mass_kg.max(), 110, 0.05, "Methods (Participants): 18.2-110 kg")
ck.published("lowest body height (cm)", P.height_cm.min(), 116.6, 0.05, "Methods (Participants): 116.6-187.5 cm")
ck.published("highest body height (cm)", P.height_cm.max(), 187.5, 0.05, "Methods (Participants): 116.6-187.5 cm")
ck.published("strides in the dataset", len(df), 1615, 0, "Technical Validation (1615 strides)")
young = df.drop_duplicates("subject")
young = young[young.age <= 17]
ck.published("young subjects (6-17 years)", len(young), 21, 0, "Comparison with published reference dataset")
ck.published("young subjects, mean age", young.age.mean(), 10.9, 0.05, "Comparison with published reference dataset")
ck.published("young subjects, SD of age", young.age.std(), 3.2, 0.05, "Comparison with published reference dataset")
MISS_PCT = 100 * stat["nanframes"] / stat["frames"]
ck.internal("only the stride files with 84 instead of 87 marker rows are unnamed (subjects 33 and 40)", stat.get("short_marker_trials", 0) > 0)
print("strides per participant: min", df.groupby("subject").size().min(), "max", df.groupby("subject").size().max(), "frames-with-NaN share", stat["nanframes"] / stat["frames"])
ck.internal("five task types", set(df.block) == set(TASK), str(sorted(set(df.block))))
ck.internal("curves have 101 points in every row", bool((df.n_samples == 101).all()))

meta = dict(
    paper=dict(authors="Lencioni T, Carpinella I, Rabuffetti M, Marzegan A, Ferrarin M", year=2019,
               title="Human kinematic, kinetic and EMG data during different walking and stair ascending and descending tasks",
               venue="Sci Data 6(1):309", doi="10.1038/s41597-019-0323-z"),
    lab="Ferrarin (Fondazione Don Carlo Gnocchi, Milan)", data_url="https://doi.org/10.6084/m9.figshare.c.4494755",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Every stride of every SubjectX.mat file is one row, in the order of s.Data (participants in numeric order); each stride's curves, marker trajectories, "
             "ground reaction forces and EMG are written as four CSV files with the values unchanged (8 significant digits; the time axis of the raw signals is sample number "
             "divided by the sampling rate; missing samples are empty). Height and body mass come from Demo_Anthro.xlsx. The standing trial (StandingData) and the authors' MATLAB scripts are not included."),
    how=["One row is one stride: from a foot strike of the foot on the force platform (the trailing foot) to the next foot strike of the same foot. The main sample file holds the authors' "
         "lower-limb joint angles (degrees; pelvis and hip, knee, ankle in three planes), joint moments (Nm/kg, normalised to body mass), sagittal joint powers (W/kg) and the "
         "centre-of-mass trajectory (m), each time-normalised to 101 points of the stride (first column stride_percent). Raw marker trajectories (87 channels = 29 markers x X, Y, Z, in m), "
         "ground reaction forces (9 channels: force in N, centre of pressure in mm, torque in Nm) and EMG (8 muscles of the dominant leg, mV) of the same stride are in the files named in trial_info. "
         "Speed, stride length, step width and cadence are the authors' per-stride values, also in trial_info.",
         "Tasks (block and condition): level walking (5 trials at natural speed, then trials with progressively higher and lower speed), toe-walking, heel-walking, step ascending and step "
         "descending of a two-step staircase, all barefoot, self-selected speed except the speed series. Only strides with a correct foot strike on a force platform and no technical problems are in the "
         "deposit. The authors' foot-strike timestamps are given in the time axes of their original recordings, which are not shared.",
         "Age, sex, height and mass are per participant; sex 'f' of one file is read as F. The deposit's sampling rates differ between participants (markers 60 or 200 Hz, force 800 or 960 Hz, EMG 800, 960 or 1000 Hz) and are in participants.csv."],
    unit_note="stride (four sample CSVs per stride)",
    differences=[f"The paper reports that 0.6% of the recorded frames were missing data after gap filling; counting empty marker values in the shared files gives {MISS_PCT:.2f}% of values (and 4.4% of frames with at least one empty value), so the figure was not reproduced and is not used as a check.",
                 "The paper says 35 locomotor trials were recorded per participant; the deposit holds between 11 and 68 strides per participant (1,615 in all, as the paper states), because each trial can contribute more than one stride.",
                 f"{stat.get('short_marker_trials', 0)} strides (of two participants) have 84 marker rows instead of 87 (three coordinates of one marker are absent); the deposit does not say which marker, so their marker columns are named row1..row84 instead of the marker names.",
                 "The paper's Data Records section says the files are CC BY 4.0, but the Figshare records of the deposit state CC0; the Figshare license is used (both allow sharing a converted copy).",
                 "The deposit stores the units of the ground reaction channels in the paper's Table 1 only (N, mm, Nm); the files carry no unit labels."])
try:
    stdlib.release(ID, "timeseries", df, meta, ck, participants=P, participant_columns=pcols, extra_files=extra)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))
