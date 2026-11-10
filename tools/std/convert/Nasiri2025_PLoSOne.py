"""Nasiri et al. (2025), PLoS ONE 20(1):e0292334: healthy young adults walking on a treadmill with an Indego Explorer lower-limb exoskeleton
under a Virtual Energy Regulator (VER) controller, a passive exoskeleton and without exoskeleton. One row per recording: the exoskeleton's own
signals (200 Hz) or the Vicon marker positions (100 Hz) of one trial.

Run from the site folder: python3 -I tools/std/convert/Nasiri2025_PLoSOne.py
Raw files (untrusted data, read only) in raw_extra/Nasiri2025_PLoSOne/ from https://doi.org/10.6084/m9.figshare.24910953 (twelve SubN zips).
Each zip was reduced by a probe script before this converter runs (the zips hold 4.3 GB, mostly 2000 Hz EMG, 1000 Hz force-plate and 200 Hz
IMU files that are not converted): the exoskeleton .mat files and the Vicon marker .csv files were rewritten to reduced/<sub>_<kind>_<cond>.csv
(columns renamed, values printed to 8 significant digits, nothing else changed) with one line each in reduced/manifest.jsonl.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Nasiri2025_PLoSOne"
RAW = stdlib.raw_dir(ID)
RED = RAW / "reduced"
checks = stdlib.Checks()
man = [json.loads(l) for l in open(RED / "manifest.jsonl")]
M = {(r["sub"], r["kind"], r["cond"]): r for r in man}
assert len(M) == len(man), "duplicate manifest lines"

SPEED = {"v1": 0.60, "v2": 0.75, "v3": 0.85}
EXP1 = "Experiment 1: virtual energy regulator (VER) performance"
EXP2 = "Experiment 2: VER versus phase-based path controller"
DEV_EXO = ("Indego Explorer exoskeleton (active hips and knees, locked passive ankles; joint angle, velocity and motor torque at 200 Hz) "
           "on a Bertec split-belt instrumented treadmill")
DEV_MK = "Vicon motion capture (8 Vero 2.2 cameras, 16 Plug-in Gait markers, 100 Hz) on a Bertec treadmill"


def unit_of(ch):
    if ch.endswith(("_x", "_y", "_z")) and ch.split("_")[0] in ("LASI", "RASI", "LPSI", "RPSI", "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
                                                               "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE"):
        return "mm"
    p = ch[:2]
    if ch == "ctrl_en":
        return "0 passive, 1 VER torque on"
    if ch == "ctrl_typ":
        return "0 passive, 1 VER, 2 phase-based path, 3 state-dependent disturbance"
    if ch.startswith("Pos_"):
        return "deg"
    if ch.startswith("Vel_"):
        return "deg/s"
    if ch.startswith("Tor_"):
        return "Nm"
    if ch.startswith(("Ene_", "Vir_")):
        return "virtual energy (unitless)"
    if ch.startswith("Phase_"):
        return "rad"
    if p in ("QH", "QK"):
        return "deg"
    if p in ("WH", "WK"):
        return "deg/s"
    if p in ("FH", "FK"):
        return "Nm"
    if p in ("EH", "EK"):
        return "virtual energy (unitless)"
    if p == "GY":
        return "gyroscope, unit not stated"
    if p == "AC":
        return "accelerometer, unit not stated"
    raise ValueError(ch)


rows, extra = [], []
seq = {}
for (sub, kind, cond), r in sorted(M.items(), key=lambda kv: (kv[0][0], {"v1": 1, "v2": 2, "v3": 3, "noexo": 4, "compare": 1}[kv[0][2]], kv[0][1])):
    exp2 = cond == "compare"
    seq[sub] = seq.get(sub, 0) + 1
    code = f"{ID}_S{sub:04d}"
    fn = f"samples/{code}_{kind}_{cond}.csv"
    extra.append((fn, RED / f"{r['name']}.csv"))
    ch = r["channels"]
    n = r["n"]
    rate = r["rate"]
    if kind == "exo":
        ev = []
        key = "ctrl_typ" if exp2 else "ctrl_en"
        ev.append(f"{key}_{r['ctrl_first'][1]}_from=0")
        for t, v in r["ctrl_changes"]:
            ev.append(f"{key}_{v}_from={t:.4f}")
        sig = ("exoskeleton joint angles, velocities, motor torques and virtual energies of hips and knees, with controller state"
               + ("" if exp2 else " and thigh gyroscope and accelerometer"))
        proc = "none (as recorded by the exoskeleton)"
        device = DEV_EXO
        cnd = (f"exoskeleton, controller comparison (VER, phase-based path controller, disturbance, passive), 0.8 m/s" if exp2 else
               f"exoskeleton walking at {SPEED[cond]} m/s: passive, then VER active, then passive")
        pt, pv, pu = ("exoskeleton_controller", "", "") 
        fb = "normal"
    else:
        ev = []
        sig = "3D positions of 16 Plug-in Gait markers on the lower body and pelvis"
        proc = "none (Vicon export; frames where a marker was not tracked are empty)"
        device = DEV_MK
        if cond == "noexo":
            cnd = "natural walking without exoskeleton, 0.6 then 0.75 then 0.85 m/s, two minutes each"
            pt, pv, pu = "none", 0, ""
        else:
            cnd = f"exoskeleton walking at {SPEED[cond]} m/s: passive, then VER active, then passive"
            pt, pv, pu = "exoskeleton_controller", "", ""
        fb = "normal"
    rows.append(dict(
        dataset=ID, study=ID, experiment=EXP2 if sub >= 10 else EXP1, group="", subject=str(sub), person=str(sub),
        subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy young adults", setting="lab",
        device=device, session=1, block=("compare" if exp2 else cond), trial=seq[sub], phase="test", time_s="", task="walking", unit="recording",
        condition=cnd, pert_type=pt, pert_value=pv, pert_unit=pu, feedback=fb, excluded_by_authors=False,
        recording=seq[sub], samples_file=fn, signal_kind=sig, channels=";".join(ch), channel_units=";".join(unit_of(c) for c in ch),
        sample_rate_hz=rate, n_samples=n, duration_s=n / rate, events=";".join(ev), processing=proc,
        trial_info=(f"trial={'exo_compare' if exp2 else ('natural' if cond == 'noexo' else 'exo_' + cond)}")))
D = pd.DataFrame(rows)

P = checks.published
exo1 = [r for r in man if r["kind"] == "exo" and r["cond"] in SPEED]
exo2 = [r for r in man if r["kind"] == "exo" and r["cond"] == "compare"]
nat = [r for r in man if r["kind"] == "marker" and r["cond"] == "noexo"]
mk1 = [r for r in man if r["kind"] == "marker" and r["cond"] in SPEED]
P("participants", D.person.nunique(), 12, 0, "Methods 3.1 (9 in Experiment 1, 3 in Experiment 2); figshare description sub1-12")
P("participants in Experiment 1", D[D.experiment == EXP1].person.nunique(), 9, 0, "Methods 3.1.1")
P("participants in Experiment 2", len({r["sub"] for r in exo2}), 3, 0, "Methods 3.1.2")
P("exoskeleton sampling rate (Hz)", np.mean([r["rate"] for r in man if r["kind"] == "exo"]), 200, 0, "Methods 3.2: exoskeleton measures at 200 Hz")
P("marker sampling rate (Hz)", np.mean([r["rate"] for r in man if r["kind"] == "marker"]), 100, 0, "figshare description: marker data 100 Hz")
P("Active VER duration (s), mean over exoskeleton trials", np.mean([r["ctrl_changes"][1][0] - r["ctrl_changes"][0][0] for r in exo1]), 120, 0.5, "Methods 3.1.1: active VER for two minutes")
P("Passive-before duration (s), mean over participants 1 and 3-7", np.mean([r["ctrl_changes"][0][0] for r in exo1 if r["sub"] not in (2, 8)]), 120, 0.5, "Methods 3.1.1: passive for two minutes")
P("exoskeleton trials per participant with exoskeleton files", np.mean(list(pd.Series([r["sub"] for r in exo1]).value_counts())), 3, 0, "Methods 3.1.1: three exoskeleton trials (0.60, 0.75, 0.85 m/s)")
checks.internal("all manifest files present as reduced CSV", all((RED / f"{r['name']}.csv").exists() for r in man))
checks.internal("recording counts as probed: exo 24 + 3, marker 21 + 7", len(exo1) == 24 and len(exo2) == 3 and len(mk1) == 21 and len(nat) == 7, f"{len(exo1)} {len(exo2)} {len(mk1)} {len(nat)}")
checks.internal("sample intervals regular in all exo files", all(abs(r["dt_min"] - 0.005) < 1e-6 and abs(r["dt_max"] - 0.005) < 1e-6 for r in man if r["kind"] == "exo"))
checks.internal("each Experiment 1 exo trial: ctrl_en is only 0/1 and switches on once", all(r["ctrl_values"] == [0, 1] and len(r["ctrl_changes"]) == 2 for r in exo1))
checks.internal("rows", len(D) == len(man), str(len(D)))

meta = dict(
    paper=dict(authors="Nasiri R, Dinovitzer H, Manohara N, Arami A", year=2025,
               title="Coordinated human-exoskeleton locomotion emerges from regulating virtual energy",
               venue="PLoS ONE 20(1):e0292334", doi="10.1371/journal.pone.0292334"),
    lab="Arami (University of Waterloo)", data_url="https://doi.org/10.6084/m9.figshare.24910953", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Only the exoskeleton recordings (sub_N_exo_v1..v3.mat, and sub_N_exo_compare.mat for participants 10-12) and the Vicon marker files "
             "(sub_N_marker_v1..v3.csv, sub_N_noexo_marker.csv) were converted, each into one samples CSV with time_s as first column. The 16-channel EMG (2000 Hz), "
             "force-plate (1000 Hz) and 16-sensor IMU (200 Hz) files in the deposit are not included because together they are several GB; they remain at the data link. "
             "Marker files were rewritten without their four header lines (marker names are in the column names, positions in mm), with time_s = (frame - 1) / 100. "
             "Exoskeleton variables keep the authors' names (QHL, QKR, ...); the two-dimensional gyroscope and accelerometer arrays are split into _x, _y, _z columns. "
             "Values are printed to 8 significant digits."),
    how=["One row is one recording of one participant: an exoskeleton file (200 Hz) or a Vicon marker file (100 Hz). In Experiment 1 (participants 1-9) each of three "
         "exoskeleton trials at 0.60, 0.75 and 0.85 m/s lasts about five minutes: the exoskeleton is passive for the first two minutes, applies VER torque for the next two "
         "(ctrl_en = 1) and is passive for the last stretch; a fourth recording is natural walking without the exoskeleton, three speeds of two minutes each in one file (markers only). "
         "In Experiment 2 (participants 10-12) one exoskeleton file per participant covers the whole session at 0.8 m/s, with ctrl_typ marking the controller in force "
         "(0 passive, 1 VER, 2 phase-based path controller, 3 state-dependent disturbance).",
         "Exoskeleton channels: Q = joint angle (deg), W = joint velocity (deg/s), F = motor torque (Nm), E = virtual energy, with H/K for hip/knee and L/R for left/right; "
         "GY and AC = thigh gyroscope and accelerometer (x, y, z) of the left (L) and right (R) thigh, units not stated in the deposit. Experiment 2 files use Pos, Vel, Tor, "
         "Ene (virtual energy), Vir (desired virtual energy) and Phase (limit-cycle phase, in rad here since values run from -pi to pi), per right/left hip/knee (hr, hl, kr, kl).",
         "events lists when the controller state changes: ctrl_en_1_from=120 means VER torque is on from 120 s into the recording. The deposit gives no age, sex or body size per participant; "
         "group values are in the paper."],
    differences=["Not every participant has every file converted. Participants 2 and 8 have exoskeleton files only (their markers are inside 230 MB emg_marker_imu .mat files, not converted); participant 9 has marker files only (its three exoskeleton .mat files are raw Simulink workspace dumps named sub9_exo_v2, v3 and v4 without the controller-state variable, not documented, so not converted); participants 1 and 3-7 have both. Natural-walking marker files run 360-452 s rather than the six minutes the protocol implies, so no duration check is made on them. For participants 2 and 8 the passive-before stretch is 180 s, not two minutes (the active stretch is 120 s). Participant 4's third marker file lacks one frame number (time_s is computed from the frame number, so it jumps by 0.02 s there). The paper says each exoskeleton trial lasts five minutes with the final passive stretch lasting one minute; the exoskeleton files run 290-295 s and the last passive stretch "
                 "about 50-55 s, so the five-minute figure is not used as a check. The paper says Experiment 2 had two consecutive trials, one per controller; each exo_compare file holds "
                 "two blocks 400 s apart in which VER, the phase-based path controller and a disturbance each appear once, so the controller order within them is not what the paper "
                 "describes. The paper's participant age, mass and height are not in the deposit, so they could not be checked."],
    unit_note="recording (one participant and trial, exoskeleton or marker file)",
)
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
