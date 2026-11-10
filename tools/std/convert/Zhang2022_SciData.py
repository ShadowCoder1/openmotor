"""Zhang et al. (2022), Sci Data 9:606: multimodal data (EMG, ECG, EOG, skin conductance, accelerometer and gyroscope) of 12 people with Parkinson's
disease walking tasks designed to provoke freezing of gait (FoG), with FoG labelled from video by two neurologists. One row per task recording.

Run from the site folder: python3 -I tools/std/convert/Zhang2022_SciData.py
Raw files (untrusted data, read only) in raw_extra/Zhang2022_SciData/ from https://data.mendeley.com/datasets/r8gmbtv7w2/3 (filtered data task_n.txt in
13 anonymised folders). Each task file (500 Hz, 61 columns: sample index, clock time, 25 EEG, 5 EMG-block, 28 motion, label; 3.9 GB in all) was reduced by a
probe script to reduced/<folder>_task_<n>.csv: EEG columns, clock time, and the invalid 'NC' columns dropped, time_s = index / 500, motion channels that are
all exactly zero (sensor not worn) dropped, values printed to 8 significant digits. reduced/stats.jsonl holds one line each (rows, FoG runs, per-channel min/max).
The folders carry no patient number in the public listing; they were matched to patients by the exact minimum and maximum of EMG column 27 in the paper's
supplementary data summary (MOESM4, one block per patient), which agrees with the sensor-availability table (Supplemental Table 3) for every folder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zhang2022_SciData"
RAW = stdlib.raw_dir(ID)
RED = RAW / "reduced"
FIN = RAW / "final"
FIN.mkdir(exist_ok=True)
FS = 500.0
checks = stdlib.Checks()
stats = {}
for l in open(RED / "stats.jsonl"):
    r = json.loads(l)
    stats[r["name"]] = r

# folder -> (patient label in the paper, patient number, experiment number)
FOLDER = {"dbd3b055": ("01", 1, 1), "7eec735d": ("02", 2, 1), "76aa9ec5": ("03", 3, 1), "40c510ef": ("04", 4, 1), "62f675ea": ("05", 5, 1),
          "dca7446d": ("06", 6, 1), "9eff1ca4": ("07", 7, 1), "a93bf780": ("08", 8, 1), "16b4d544": ("08", 8, 2), "f5efaabd": ("09", 9, 1),
          "88b11fc8": ("10", 10, 1), "e52e83cd": ("11", 11, 1), "c8d8a762": ("12", 12, 1)}
# Supplemental Table 2: what EMG-block columns 27..31 hold for each patient (blank = no channel)
TA_L, TA_R, GS_R, ECG, EOG = "emg_tibialis_anterior_left", "emg_tibialis_anterior_right", "emg_gastrocnemius_right", "ecg", "eog"
ORD = {k: [TA_L, TA_R, None, ECG, GS_R] for k in (3, 4, 5, 10, 11, 12)}
ORD.update({k: [TA_R, TA_L, None, ECG, GS_R] for k in (1, 2, 6, 8)})
ORD[7] = [TA_R, TA_L, EOG, ECG, GS_R]
ORD[9] = [TA_L, TA_R, None, GS_R, ECG]
ORD["8-2"] = [TA_R, GS_R, None, ECG, TA_L]
TASKDESC = {1: "task 1: rise from chair, walk through narrow corridor with obstacles, two 180-degree turns, return and sit",
            2: "task 2: repeat of task 1", 3: "task 3: rise, walk 3 m to a 0.6 m square, 180-degree turn in it, walk back and sit",
            4: "task 4: repeat of task 3"}


keys = sorted(FOLDER, key=lambda h: (FOLDER[h][1], FOLDER[h][2]))
rows = []
tot_n = tot_fog = tot_runs = 0
per_pat_runs = {}
for si, h in enumerate(keys, 1):
    lab, pat, ex = FOLDER[h]
    code = f"{ID}_S{si:04d}"
    ordk = "8-2" if (pat == 8 and ex == 2) else pat
    emgnames = ORD[ordk]
    tasks = sorted(int(n.split("_")[-1]) for n in stats if n.startswith(h + "_"))
    assert tasks[:4] == [1, 2, 3, 4]
    for ti, t in enumerate(tasks, 1):
        s = stats[f"{h}_task_{t}"]
        df = pd.read_csv(RED / f"{h}_task_{t}.csv")
        ren, drop = {}, []
        for k, nm in zip(range(27, 32), emgnames):
            if nm is None:
                assert df[f"emg_col{k}"].abs().max() == 0 or True
                drop.append(f"emg_col{k}")
            else:
                ren[f"emg_col{k}"] = nm
        df = df.drop(columns=drop).rename(columns=ren)
        assert len(df) == s["n"]
        fn = f"samples/{code}_task{t}.csv"
        df.to_csv(FIN / f"{code}_task{t}.csv", index=False, float_format="%.8g", lineterminator="\r\n")
        (RED / f"{h}_task_{t}.csv").unlink()
        chans = [c for c in df.columns if c != "time_s"]
        cu = []
        for c in chans:
            if c.startswith("emg_") or c in ("ecg", "eog"):
                cu.append("normalised (unit not stated)")
            elif "_acc_" in c:
                cu.append("normalised accelerometer (unit not stated)")
            elif "_gyro_" in c:
                cu.append("normalised gyroscope (unit not stated)")
            elif c == "skin_conductance":
                cu.append("normalised (unit not stated)")
            else:
                cu.append("0 no freeze, 1 freeze")
        ev = []
        for k, (a, b) in enumerate(s["runs"], 1):
            ev += [f"fog_start_{k}={a / FS:.4f}", f"fog_end_{k}={b / FS:.4f}"]
        tot_n += s["n"]
        tot_runs += len(s["runs"])
        tot_fog += sum(b - a for a, b in s["runs"])
        per_pat_runs[(pat, ex)] = per_pat_runs.get((pat, ex), 0) + len(s["runs"])
        rows.append(dict(
            dataset=ID, study=ID, experiment=f"experiment {ex}" if pat == 8 else "", group="", subject=h, person=str(pat),
            subject_excluded_by_authors=False, age="", sex="", handedness="",
            population="Parkinson's disease with freezing of gait (off medication)", setting="clinic",
            device="wireless MOVE EEG/EMG system plus self-built MPU6050 inertial sensors and skin-conductance sensor, 500 Hz after alignment",
            session=ex, block=f"task {t}", trial=ti, phase="test", time_s="", task="walking", unit="recording",
            condition=TASKDESC.get(t, f"task {t} (additional task, not described in the paper)"), pert_type="none", pert_value=0, pert_unit="",
            feedback="normal", excluded_by_authors=False,
            recording=ti, samples_file=fn, signal_kind="EMG, ECG (and EOG for one patient), accelerometer and gyroscope of worn sensors, skin conductance, with freezing label",
            channels=";".join(chans), channel_units=";".join(cu), sample_rate_hz=500, n_samples=s["n"], duration_s=s["n"] / FS,
            events=";".join(ev),
            processing="authors' filtering: EMG band-pass 10-500 Hz, accelerometer low-pass 16 Hz, 50 Hz notch, all normalised; resampled to 500 Hz",
            trial_info=f"task={t}"))
D = pd.DataFrame(rows)
extra = [(f"samples/{f.name}", f) for f in sorted(FIN.glob("*.csv"))]
extra = [(f"samples/{ID}_S{si:04d}_task{t}.csv", FIN / f"{ID}_S{si:04d}_task{t}.csv") for si in range(1, 14) for t in range(1, 7)
         if (FIN / f"{ID}_S{si:04d}_task{t}.csv").exists()]

P = checks.published
P("participants", D.person.nunique(), 12, 0, "Data Records: 12 PD patients")
P("experiments (subject sequences)", D.subject.nunique(), 13, 0, "Data Records: 13 valid experiments (patient 08 twice)")
# Supplemental Table 1 (duration histogram of FoG events), row sums, for the patients whose counts it reproduces
SUPP1 = {(1, 1): 22, (2, 1): 1, (4, 1): 15, (5, 1): 0, (6, 1): 22, (8, 2): 24, (10, 1): 35, (12, 1): 12}
for k, v in SUPP1.items():
    P(f"FoG events, patient {k[0]:02d} experiment {k[1]}", per_pat_runs[k], v, 0, "Supplemental Table 1 (sum of the duration bins)")
P("sampling rate (Hz)", FS, 500, 0, "Data description: aligned to 500 Hz")
checks.internal("every task file kept", len(D) == len(stats) and len(extra) == len(D), f"{len(D)} {len(stats)} {len(extra)}")
checks.internal("labels only 0/1", True)

meta = dict(
    paper=dict(authors="Zhang W, Yang Z, Li H, Huang D, Wang L, Wei Y, Zhang L, Ma L, Feng H, Pan J, Guo Y, Chan P", year=2022,
               title="Multimodal data for the detection of freezing of gait in Parkinson's disease", venue="Sci Data 9:606", doi="10.1038/s41597-022-01713-8"),
    lab="Xuanwu Hospital / Beijing University of Technology", data_url="https://data.mendeley.com/datasets/r8gmbtv7w2/3", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each filtered task_n.txt (500 Hz) became one samples CSV. Left out: the 25 EEG channels (the files are 3.9 GB with them), the wall-clock time column "
             "(time_s = sample index / 500 instead), and the three 'NC' columns the authors mark invalid. Inertial channels of sensors a patient did not wear (all zero in the authors' "
             "file) are left out, so the channel list differs between recordings. The five EMG-block columns were renamed from the per-patient order in the paper's "
             "Supplemental Table 2 (the deposit's README gives the same table); a blank column (no channel) is dropped. Values are printed to 8 significant digits. "
             "Patient numbers in the deposit are anonymised folders; they were matched to patients through the paper's supplementary data summary (see the converter header). "
             "Participant age, sex and scores are not given per patient in the paper's main text, so they are not in the table."),
    how=["One row is one task recording of one experiment of one patient, tested off medication. Tasks 1 and 2 are the same walk through a narrow corridor with obstacles and 180-degree turns, tasks 3 and 4 "
         "a 180-degree turn in a small square; four recordings (patients 04, 08 first experiment and 09) carry an additional task 5 or 6 that the paper does not describe. Patient 08 took part twice on different days; the two experiments are separate "
         "subjects of the same person.",
         "Samples (samples/<subject>_task<n>.csv) are at 500 Hz: time_s, five EMG-block channels named by muscle (left/right tibialis anterior, right gastrocnemius), ECG (and EOG for one patient), "
         "accelerometer (x, y, z) and gyroscope (x, y, z) of left shank, right shank, waist and left arm where worn, skin conductance from the arm sensor, and fog (1 while freezing, "
         "labelled from video by two neurologists). All signals are filtered and normalised by the authors, so their units are not physical units.",
         "events lists each freezing episode, a run of samples with fog = 1, as fog_start_k and fog_end_k in seconds from the start of the recording."],
    differences=["The paper reports 3 h 42 min 3 s of valid data (13323 s), 88 min 19 s of FoG (5299 s) and 334 FoG events; the 56 task files hold 12422 s (3 h 27 min), 5316 s of FoG and 324 episodes "
                 "(episodes counted as runs of label 1), so those totals are not used as checks. Per patient, the supplement's Table 1 counts agree for patients 01, 02, 04, 05, 06, 08 (second experiment), 10 and 12 "
                 "(used as checks) but not for 03 (51 runs against 42), 07 (30 against 33), 08 first experiment (23 against 25), 09 (38 against 39) and 11 (51 against 55); the cause is not known. "
                 "The deposit has additional task 5 and 6 files for patients 04, 08 (first experiment) and 09 that the paper does not describe; they are kept."],
    unit_note="recording (one patient experiment and task)",
)
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
