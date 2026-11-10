"""Brantley, Luu, Nakagome, Zhu & Contreras-Vidal (2018), Sci Data 5:180133: full-body kinematics (17 Xsens IMUs) and leg EMG
of 10 healthy adults walking a course of level ground, ramp and stairs (forward and back). One row per recording (one
-kin.mat or -emg.mat file of the deposit); the samples are in a CSV inside the ZIP.

Raw files: figshare 10.6084/m9.figshare.5616109.v5 (CC BY 4.0), 104 -kin.mat and 103 trial -emg.mat files (EEG,
impedance, channel location and emgsensitivity files are not used). Each raw file was fetched alone onto the PC, reduced by
a probe-style script (reading it with scipy.io.loadmat, writing a samples CSV and a one-line JSON into
om_std/tmp_samples/<id>/) and deleted at once, because the motor files together are 5.5 GB. This converter then assembles
the rows from those JSON files. Nothing from the deposit was executed.

Run from the site folder: python3 -I tools/std/convert/Brantley2018_SciData.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Brantley2018_SciData"
TMP = stdlib.HERE / "tmp_samples" / ID
checks = stdlib.Checks()

recs = [json.load(open(p)) for p in sorted(TMP.glob("*.json"))]
recs.sort(key=lambda r: (r["subject"], r["trial"], r["kind"]))
kin = [r for r in recs if r["kind"] == "kin"]
emg = [r for r in recs if r["kind"] == "emg"]

rows, extra = [], []
count = {}
for r in recs:
    s, tr, kind = r["subject"], r["trial"], r["kind"]
    count[s] = count.get(s, 0) + 1
    code = f"{ID}_S{s:04d}"  # subjects appear in numerical order, so stdlib's codes S0001.. follow the deposit's numbers
    fn = f"samples/{ID}_S{s:04d}_T{tr:02d}_{kind}.csv"
    extra.append((fn, str(TMP / r["file"])))
    baseline = tr == 0
    if kind == "kin":
        sk = "full-body segment and joint kinematics from 17 Xsens MVN inertial sensors"
        proc = ("Xsens MVN Studio output as shared (joint angles are Euler angles in the sequence ZXY); only the 22 joint angles, the "
                "pelvis and both feet positions are kept here; no filtering or resampling by us")
        ti = f"course_completions_in_file={r['numTrials']};sensors={r['n_sensors']};joints={r['n_joints']}"
        dev = "Xsens MVN, 17 wireless IMUs"
        cond = "walking the gait course (level ground, ramp, stairs; forward and back)"
    else:
        sk = "surface EMG of 6 muscles of each leg"
        proc = ("as shared (mV), then low-pass filtered (8th-order Chebyshev type I, zero phase) and kept at every 2nd sample, "
                "1000 Hz to 500 Hz; nothing else")
        ti = f"original_sample_rate_hz={int(r['fs_raw'])}"
        dev = "Biometrics SX230 bipolar EMG sensors on a DataLOG MWX8"
        cond = ("quiet standing baseline, eyes open (no motion capture)" if baseline
                else "walking the gait course (level ground, ramp, stairs; forward and back)")
    rows.append(dict(
        dataset=ID, study=ID, experiment="walking on level ground, ramp and stairs", group="", subject=str(s), person=str(s),
        subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy adults (18-31 years)",
        setting="lab", device=dev, session=1, block=f"T{tr:02d}", trial=tr + 1, phase="baseline" if baseline else "test",
        time_s="", task="walking" if not baseline else "standing", unit="recording", condition=cond, pert_type="none",
        pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=count[s], samples_file=fn, signal_kind=sk, channels=";".join(r["channels"]),
        channel_units=";".join(r["units"]), sample_rate_hz=r["fs"], n_samples=r["n_out"], duration_s=r["n_out"] / r["fs"],
        events="", processing=proc, trial_info=ti))
D = pd.DataFrame(rows)

P = checks.published
src = "Methods, Participants"
P("participants", D.person.nunique(), 10, 0, src)
P("IMUs per participant", np.unique([r["n_sensors"] for r in kin])[0] if len(set(r["n_sensors"] for r in kin)) == 1 else -1, 17, 0,
  "Methods, Instrumentation & Data Collection (17 wireless IMUs)")
P("joints per kinematics file", np.unique([r["n_joints"] for r in kin])[0] if len(set(r["n_joints"] for r in kin)) == 1 else -1, 22, 0,
  "Table 5 (kin.setup.jointlabel: 22 x 1)")
P("kinematics files with a sampling rate other than 30 or 60 Hz", sum(1 for r in kin if r["fs"] not in (30.0, 60.0)), 0, 0,
  "Methods (recorded at either 30 or 60 Hz); Table 5")
P("EMG channels per recording (6 muscles x 2 legs)", len(emg[0]["channels"]), 12, 0, "Methods (six sites bilaterally); Table 3")
P("EMG original sampling rate (Hz)", np.mean([r["fs_raw"] for r in emg]), 1000, 0, "Methods (recorded at 1000 Hz)")
P("kinematics files per participant (blocks of trials; paper: approximately ten)", len(kin) / 10, 10, 0.5,
  "Methods, Experimental Protocol (ten blocks of two trials)")
P("gait-course completions per participant (mean)", sum(r["numTrials"] for r in kin) / 10, 20, 0.5,
  "Methods, Experimental Protocol (an average of twenty trials)")

checks.internal("every kinematics file has 22 x 3 joint angles + 3 x 3 positions = 75 channels", all(len(r["channels"]) == 75 for r in kin))
checks.internal("all 10 participants have both kinds of file", set(r["subject"] for r in kin) == set(r["subject"] for r in emg) == set(range(1, 11)))
checks.internal("baseline recordings (T00) exist only as EMG", all(r["kind"] == "emg" for r in recs if r["trial"] == 0))
checks.internal("EMG has no NaN", all(r["nan"] == 0 for r in emg))
checks.internal("sample counts add up", int(D.n_samples.sum()) == sum(r["n_out"] for r in recs))
checks.internal("row count = JSON count", len(D) == len(recs) == 217 and len(kin) == 104 and len(emg) == 113, str(len(recs)))

meta = dict(
    paper=dict(authors="Brantley JA, Luu TP, Nakagome S, Zhu F, Contreras-Vidal JL", year=2018,
               title="Full body mobile brain-body imaging data during unconstrained locomotion on stairs, ramps, and level ground",
               venue="Sci Data 5:180133", doi="10.1038/sdata.2018.133"),
    lab="Contreras-Vidal (University of Houston)", data_url="https://doi.org/10.6084/m9.figshare.5616109.v5",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Only the kinematics (-kin.mat) and EMG (-emg.mat) files were converted; the EEG, impedance, electrode location and "
             "EMG sensitivity files are not included. From each kinematics file the 22 joint angles (3 Euler angles each, degrees), "
             "the global position of the pelvis and of both feet (m) were written to a samples CSV at the original rate (30 or 60 Hz); "
             "velocities, accelerations, orientations and the raw sensor signals of the deposit are left out. EMG was low-pass filtered "
             "and reduced from 1000 Hz to 500 Hz to keep the file small; column order is left leg then right leg. "
             "Participants are numbered in the order of the deposit's subject numbers."),
    how=["One row is one recording file of one participant: either the 17-IMU full-body kinematics of one block of the walking course "
         "(level ground, ramp, stairs, in the forward direction and back; one or two completions per file, see trial_info) or the leg EMG "
         "recorded at the same time. Trial T00 is a quiet-standing baseline (eyes open) with EMG only (the paper says one minute; the recordings last 78-131 s). There are no stride or "
         "stair/ramp markers in the deposit and none were derived.",
         "Kinematics samples (samples/*_kin.csv): time_s, then for each of the 22 joint angles of Xsens MVN (jL5S1 ... jLeftBallFoot) its three Euler "
         "components <joint>_x, _y, _z (sequence ZXY, degrees; Table 5 of the paper), then pos_<segment>_x/y/z for Pelvis, RightFoot and LeftFoot "
         "(m, global frame). Sample rate is in sample_rate_hz (30 or 60).",
         "EMG samples (samples/*_emg.csv): time_s, then left_TA, left_GAST, left_VL, left_RF, left_SEM, left_BFL, right_TA ... right_BFL "
         "(tibialis anterior, medial gastrocnemius, vastus lateralis, rectus femoris, semitendinosus, biceps femoris longus), in mV as the authors "
         "give them, at 500 Hz. The authors aligned the systems with a hardware pulse before sharing, but the kinematics and EMG files of the same trial do not always last equally long (the kinematics file of some trials, mostly of participant 1, is 6-40 s shorter), so the alignment of the two files is not documented beyond the paper's statement.",
         "Sex and age of single participants are not in the deposit (five men and five women aged 18-31 overall), so those columns are empty."],
    differences=["EMG is downsampled to 500 Hz (original 1000 Hz).",
                 "The paper says the subjects completed 'an average of twenty trials in approximately ten blocks'; the check uses the numTrials "
                 "field (course completions per file, 1 or 2) summed over a participant's kinematics files."],
    unit_note="recording (one participant, one trial file, kinematics or EMG)",
)
stdlib.release(ID, "timeseries", D, meta, checks, no_published_reason=None, extra_files=extra)
