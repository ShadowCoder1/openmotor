"""Angelidou, Chambers, Hobbs, Karakasis & Artemiadis (2025) Sci Data 12:84: walking on a variable-stiffness treadmill,
20 people x 3 speeds x 4 surface stiffnesses, one row per gait cycle (left heel strike to left heel strike).

Raw files (Figshare 10.6084/m9.figshare.27180288, CC BY 4.0): SUB01.mat ... SUB20.mat, MATLAB v7.3 (HDF5) structs of 2.7-3.4 GB.
They were read lazily with h5py by Angelidou2025_SciData_jobs.py (one file at a time, deleted after reduction); this converter
turns the reduced json files (tmp_samples/Angelidou2025_SciData/SUBxx.json: the authors' gaitEvents, per-condition cycle
counts, and time-normalised mean/SD joint-angle and force curves) into the OpenMotor walking format.
Run from the PC's om_std folder after the job script has finished: python3 -I tools/std/convert/Angelidou2025_SciData.py
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Angelidou2025_SciData"
TMP = Path("tmp_samples") / DS
SPEEDS = [("speed0p6", 0.6), ("speed0p8", 0.8), ("speed1p0", 1.0)]
STIFFS = [("stiff1000", 1000), ("stiff25", 25), ("stiff80", 80), ("stiff40", 40)]  # order of the protocol (README)
CH = ["LHip", "LKnee", "LAnkle", "RHip", "RKnee", "RAnkle", "GRF_L", "GRF_R"]
CH_UNIT = {c: "deg" for c in CH[:6]} | {"GRF_L": "raw force-mat signal", "GRF_R": "raw force-mat signal"}
# Table 1 of the paper: id, sex, age, height m, body mass kg, leg length mm, resting HR bpm, preferred speed rigid / compliant m/s
T1 = """01 M 26 1.78 75.5 840 68 1 0.9
02 F 27 1.65 70.8 840 58 0.975 0.875
03 F 28 1.63 64.9 805 65 0.95 0.95
04 F 29 1.73 68 840 58 0.95 0.85
05 M 36 1.77 85.6 780 60 1.05 0.9
06 M 19 1.77 72.5 860 65 0.875 0.8
07 F 37 1.63 64.6 810 75 0.9 0.95
08 M 27 1.89 86.8 930 75 1 1
09 M 26 1.80 76 920 50 0.95 0.95
10 M 28 1.74 78 780 67 0.95 0.9
11 F 24 1.68 58.9 870 63 0.95 0.9
12 F 29 1.70 69.4 830 67 0.95 0.9
13 F 21 1.68 60.1 810 61 1 0.9
14 M 25 1.75 60 860 65 0.95 0.9
15 M 25 1.70 55.3 860 47 1.05 1.1
16 F 32 1.75 89 900 75 0.95 0.8
17 F 30 1.75 59 880 84 0.9 0.875
18 M 24 1.65 81 800 78 0.9 0.8
19 F 22 1.65 81.2 830 78 1 0.95
20 M 26 1.78 82.9 930 72 0.95 0.9"""
PEOPLE = {}
for ln in T1.splitlines():
    i, sx, ag, h, m, ll, hr, ps_r, ps_c = ln.split()
    PEOPLE[int(i)] = dict(sex=sx, age=float(ag), height_m=float(h), mass_kg=float(m), leg_mm=float(ll), hr=float(hr),
                          pref_rigid=float(ps_r), pref_compliant=float(ps_c))

# SUB01 gait-cycle counts the README lists as differing from 150 (Known Dataset Limitations)
README_SUB01 = {("speed0p8", "stiff25"): 148, ("speed1p0", "stiff80"): 148, ("speed0p8", "stiff40"): 149,
                ("speed0p8", "stiff80"): 149, ("speed1p0", "stiff25"): 149, ("speed0p6", "stiff40"): 149,
                ("speed1p0", "stiff40"): 154}


def main():
    ck = stdlib.Checks()
    rows, extra, parts = [], [], []
    counts, durations, stances, order_ok = {}, [], [], True
    n_right_same = 0
    SUBS = sorted(int(q.stem[3:]) for q in TMP.glob('SUB*.json'))
    for sub in SUBS:
        d = json.load(open(TMP / f"SUB{sub:02d}.json"))
        code = f"{DS}_X{sub:02d}"  # replaced by stdlib in order of first appearance (SUB01 -> S0001 ...)
        pp = PEOPLE[sub]
        parts.append(dict(person=code, height_m=pp["height_m"], body_mass_kg=pp["mass_kg"], leg_length_mm=pp["leg_mm"],
                          resting_heart_rate_bpm=pp["hr"], preferred_speed_rigid_mps=pp["pref_rigid"],
                          preferred_speed_compliant_mps=pp["pref_compliant"]))
        trial = 0
        scode = f"{DS}_S{sub:04d}"
        for sp, spv in SPEEDS:
            r = d["speeds"][sp]
            L = np.array(r["lhs"])
            LT = np.array(r["lto"])
            n_right_same += bool(r["right_equals_left"])
            durations.append(r["stiff"]["stiff1000"]["t_first"] * 0 + r["stiff"]["stiff40"]["t_last"])
            for st, stv in STIFFS:
                s = r["stiff"][st]
                n, j0 = s["n"], s["j0"]
                counts[(sub, sp, st)] = n - 150 if st == "stiff1000" else n
                for k in range(n):
                    j = j0 + k
                    trial += 1
                    warm = st == "stiff1000" and k < 150
                    if j > 0:
                        prev = L[j - 1]
                        stride = L[j] - prev
                        stance = LT[j] - prev
                        swing = L[j] - LT[j]
                        stances.append(stance / stride)
                    else:
                        prev = stride = stance = swing = np.nan
                    rows.append(dict(
                        dataset=DS, study=DS, experiment="", group="", subject=scode, person=scode,
                        subject_excluded_by_authors=False, age=pp["age"], sex=pp["sex"], handedness="", population="healthy",
                        setting="lab", device="Variable Stiffness Treadmill 2 (instrumented split-belt, belts tied), Vicon, force mats",
                        session=1, block=f"{spv} m/s, {stv} kN/m" + (", force-mat warm-up" if warm else ""), trial=trial,
                        phase="practice" if warm else ("baseline" if st == "stiff1000" else "other"), time_s=prev if j > 0 else "",
                        task="walking", unit="stride", condition=f"{spv} m/s, {stv} kN/m",
                        pert_type="none" if st == "stiff1000" else "surface_stiffness",
                        pert_value=0 if st == "stiff1000" else stv, pert_unit="" if st == "stiff1000" else "kN/m",
                        feedback="normal", excluded_by_authors=False,
                        walking_context="treadmill", surface_stiffness_kn_per_m=stv, belt_speed_fast_mps=spv,
                        belt_speed_slow_mps=spv, walking_speed_mps=spv, fast_leg="unknown",
                        stride_start_s=prev if j > 0 else "", stride_time_s=stride, stride_time_left_s=stride,
                        stance_time_left_s=stance, swing_time_left_s=swing,
                        stance_pct_left=100 * stance / stride if j > 0 else "", swing_pct_left=100 * swing / stride if j > 0 else "",
                    ))
                # mean/SD curves of the cycles kept (stiff1000: the 150 after the warm-up)
                buf = io.StringIO()
                w = csv.writer(buf, lineterminator="\r\n")
                w.writerow(["percent_gait_cycle"] + [f"{kind}_{c}" for c in CH for kind in ("mean", "sd")])
                for q in range(100):
                    w.writerow([f"{100 * q / 99:.4g}"] + [f"{s[kk][c][q]:.6g}" for c in CH for kk in ("mean", "sd")])
                extra.append((f"samples/{DS}_S{sub:04d}_{sp.replace('speed', 'speed_')}_{st}_mean_gait_cycle.csv", buf.getvalue()))
    df = pd.DataFrame(rows)

    # ---------- checks
    p = [PEOPLE[i] for i in SUBS]
    one = df.drop_duplicates("subject")
    ck.published("number of participants (files with data)", len(one), 20, 0, "Methods, Participants")
    ck.published("number of men", (one.sex == "M").sum(), 10, 0, "Methods, Participants (10 males and 10 females)")
    ck.published("mean age, years (27.1 +- 4.5)", one.age.mean(), 27.1, 0.05 + 1e-9, "Methods, Participants")
    ck.published("SD of age, years", one.age.std(ddof=1), 4.5, 0.05, "Methods, Participants")
    ck.published("mean height, cm (172.4 +- 6.8)", 100 * np.mean([x["height_m"] for x in p]), 172.4, 0.05, "Methods, Participants")
    for (sp, st), v in README_SUB01.items():
        ck.published(f"SUB01 gait cycles at {sp} {st} (README: not 150)", counts[(1, sp, st)], v, 0, "deposit README, Known Dataset Limitations")
    ck.published("SUB01 conditions with exactly 150 cycles", sum(counts[(1, sp, st)] == 150 for sp, _ in SPEEDS for st, _ in STIFFS),
                 12 - len(README_SUB01), 0, "deposit README, Known Dataset Limitations (7 of 12 differ)")
    ck.published("conditions per participant (3 speeds x 4 stiffness levels)", len({(sp, st) for (s, sp, st) in counts if s == 1}), 12, 0,
                 "Methods, Experimental Procedure")
    off = {k: v for k, v in counts.items() if abs(v - 150) >= 10}
    ck.internal("every condition has 150 cycles within +-10 except the few listed under Known differences (README: 'always less than +-10')",
                len(off) <= 6 and all(k[0] == 6 for k in off), f"{len(off)} conditions off by 10 or more: {sorted(off.items())}")
    ck.internal("all but those conditions: 90% of the 240 conditions hold 148-152 cycles",
                np.mean([abs(v - 150) <= 2 for v in counts.values()]) > 0.9, f"{np.mean([abs(v - 150) <= 2 for v in counts.values()]):.3f}")
    ck.internal("240 conditions (20 people x 12)", len(counts) == 12 * len(SUBS), str(len(counts)))
    ck.internal("trial durations 12-20 min (paper: approximately 12-20 minutes), min/max s", 600 < min(durations) and max(durations) < 1300,
                f"{min(durations):.0f}-{max(durations):.0f}")
    ck.internal("stance is 55-80 % of stride on all strides (authors' left heel strike and toe-off times)",
                0.55 < np.percentile(stances, 0.1) and np.percentile(stances, 99.9) < 0.80,
                f"0.1-99.9 percentiles {np.percentile(stances, 0.1):.3f}-{np.percentile(stances, 99.9):.3f}")
    ck.internal("right-leg event times in the deposit are identical to the left-leg ones in all 60 trials (so only left-leg timing is released)",
                n_right_same == 3 * len(SUBS), str(n_right_same))
    ck.internal("no participant lost", SUBS == list(range(1, 21)), str(SUBS))

    people_df = pd.DataFrame(parts)
    pcols = [("height_m", "number", "m", "Body height (Table 1 of the paper)."),
             ("body_mass_kg", "number", "kg", "Body mass (Table 1)."),
             ("leg_length_mm", "number", "mm", "Leg length (Table 1)."),
             ("resting_heart_rate_bpm", "number", "bpm", "Resting heart rate measured in the first break between trials (Table 1)."),
             ("preferred_speed_rigid_mps", "number", "m/s", "Self-reported comfortable treadmill speed on the rigid surface (Table 1)."),
             ("preferred_speed_compliant_mps", "number", "m/s", "Self-reported comfortable speed on the most compliant setting tested (Table 1).")]
    # participants person codes: map to the same order stdlib uses
    people_df["person"] = [f"{DS}_S{i:04d}" for i in SUBS]

    meta = dict(
        study=DS,
        paper=dict(authors="Angelidou C, Chambers V, Hobbs B, Karakasis C, Artemiadis P", year=2025,
                   title="Kinematics, kinetics, and muscle activations during human locomotion over compliant terrains",
                   venue="Sci Data 12:84", doi="10.1038/s41597-025-04433-x"),
        lab="Artemiadis (University of Delaware)",
        data_url="https://doi.org/10.6084/m9.figshare.27180288",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Read the 20 participant files (MATLAB v7.3, 2.7-3.4 GB each, about 60 GB) lazily and kept, per gait cycle, "
                 "the authors' left heel-strike and toe-off times turned into stride, stance and swing times, plus per condition "
                 "the mean and SD over gait cycles of the sagittal hip, knee and ankle angles (both legs) and the two raw "
                 "force-mat signals on a 0-100 % gait-cycle axis; the 2 kHz EMG and IMU, the markers, joint kinetics, "
                 "centre of mass and pressure, heart rate and the per-cycle curves are not carried over (the deposit remains "
                 "the source for them); participant codes replace the authors' numbers."),
        how=["One row is one gait cycle (left heel strike to the next left heel strike) of one participant at one belt speed "
             "(0.6, 0.8 or 1.0 m/s) and one surface stiffness (1000 = rigid, then 25, 80, 40 kN/m, the order of the protocol); "
             "each trial (speed) held 150 cycles per stiffness, and the rigid block holds 300 because the first 150 cycles are the "
             "force-mat warm-up (block says so, phase practice). Rows are listed by speed (0.6, 0.8, 1.0; the order in which the "
             "speeds were walked is not given for 0.6 and 1.0) and within a speed in protocol order.",
             "The deposit stores gait events as one list per speed, aligned with the gait cycles, in which the event with index j "
             "closes cycle j. Stride time is the interval between consecutive left heel strikes, stance time runs from the previous "
             "left heel strike to the left toe-off of the cycle and swing time from that toe-off to the closing heel strike, all in "
             "seconds, and the percentages are of the stride. The first cycle of each trial has no earlier heel strike, so its times "
             "are empty. The deposit's right-leg heel-strike and toe-off lists are identical to the left-leg ones in every trial, so "
             "no right-leg timing is released. Belts were tied: both belt-speed columns hold the commanded speed.",
             "Samples are the mean and SD, over the gait cycles of one speed and stiffness (for the rigid block the 150 cycles "
             "after warm-up), of each channel resampled by linear interpolation to 100 points from the first to the last sample of "
             "the cycle, as the authors' example script does. Angles are the first (sagittal, x) component of the Plug-in Gait "
             "output in degrees; the force columns are the deposit's raw electrical force-mat signals (left, right), "
             "which the authors say drift and need their processGRF scaling to become newtons.",
             "Age and sex come from Table 1 of the paper (identical order to the SUBxx files); stiffness is as commanded (the "
             "paper's measured values are 89, 41 and 27 kN/m for the 80, 40 and 25 kN/m settings)."],
        unit_note="gait cycle (left heel strike to left heel strike); mean curves per condition are in samples/",
        differences=["The README says conditions differ from 150 gait cycles by less than 10, but one participant (the sixth file) has 135, 135, 139 and 124 and 122 cycles "
                     "in five conditions (the rigid block 124 and 122 after the 150 warm-up cycles); all other conditions are within 9 of 150. All cycles are kept.",
                     "The Methods give mean body mass 68.4 +- 12.2 kg and height SD 6.8 cm, but the paper's own Table 1 (used here for age, sex and the participants file) averages 72.0 +- 10.5 kg and 172.4 +- 6.7 cm; mean age (27.05, printed 27.1) and mean height (172.4) agree, so mass is not used as a check.",
                     "The deposit's README (and the figshare description) lists the speeds as 0.8, 1 and 1.2 m/s, but the files "
                     "hold speed0p6, speed0p8 and speed1p0, matching the paper's 0.6, 0.8 and 1.0 m/s; the paper's values are used.",
                     "The right-leg gait-event lists in the deposit are copies of the left-leg ones, so right-leg timing is not released."],
    )
    stdlib.release(DS, "walking", rows, meta, ck, participants=people_df, participant_columns=pcols, extra_files=extra)
    for c in ck.items:
        print(("  [published] " if c["kind"] == "published" else "  [internal] ") + c["what"][:90],
              (f"{c['computed']:.4g} vs {c['reported']:.4g}" if c["kind"] == "published" else c.get("detail", "")))


if __name__ == "__main__":
    main()
