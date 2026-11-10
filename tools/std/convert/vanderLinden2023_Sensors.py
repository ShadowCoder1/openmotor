"""van der Linden et al. (2023), Sensors 23(20):8621: finger and wrist accelerometry of resting and postural tremor in people with
essential tremor (ET) or Parkinson's disease (PD) treated with deep brain stimulation, with the clinician's MDS-UPDRS amplitude
rating (0-4) of every 20 s trial. One row per recording (one trial and one device).

Raw files: OSF mp5ha (CC BY-NC-SA 4.0): TrainingData/ and ValidationData/ with Finger/ (BrainProducts, 2500 Hz) and Wrist/
(GENEActiv, 100 Hz) CSV files, plus TrainingData_*.csv / ValidationData_*.csv with the ratings (semicolon separated). The 3,428
accelerometry files (6.7 GB) were read directly over HTTPS on the PC by batch scripts (never stored: one file at a time in memory),
which wrote a samples CSV and a JSON per file into om_std/tmp_samples/<id>/. This converter assembles the rows from those JSON
files and the job list (vanderLinden2023_Sensors_jobs.py, built from the ratings CSVs). The authors' MATLAB scripts were read as text only.

Run from the site folder: python3 -I tools/std/convert/vanderLinden2023_Sensors.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "vanderLinden2023_Sensors"
TMP = stdlib.HERE / "tmp_samples" / ID
checks = stdlib.Checks()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vanderLinden2023_Sensors_jobs import JOBS as jobs  # noqa: E402
rec = {}
for p in TMP.glob("*.json"):
    r = json.load(open(p))
    rec[r["key"]] = r
assert len(rec) == len(jobs) == 3428, (len(rec), len(jobs))

TASK = {"RestingTremor": "resting tremor", "PosturalTremor": "postural tremor"}
rows, extra = [], []
sub_idx, per_sub = {}, {}
for j in jobs:
    key = f"{j['split']}_{j['task']}_{j['device']}_{j['name'][:-4]}"
    r = rec[key]
    subj = f"{j['subj']}_{j['side']}"
    sub_idx.setdefault(subj, len(sub_idx) + 1)
    per_sub[subj] = per_sub.get(subj, 0) + 1
    fn = f"samples/{ID}_S{sub_idx[subj]:04d}_{j['task'][:4].lower()}_{j['device'].lower()}_m{j['measure']:02d}_{j['split'][:3].lower()}.csv"
    extra.append((fn, str(TMP / r["file"])))
    finger = j["device"] == "Finger"
    disease = {"ET": "essential tremor", "PD": "Parkinson's disease"}[j["disease"]]
    rows.append(dict(
        dataset=ID, study=ID, experiment="resting and postural tremor accelerometry with clinical ratings", group=j["disease"],
        subject=subj, person=j["subj"], subject_excluded_by_authors=False, age="", sex="", handedness="",
        population=disease + " treated with deep brain stimulation", setting="clinic",
        device=("BrainProducts triaxial accelerometer on the index finger, 2500 Hz" if finger
                else "GENEActiv wristwatch accelerometer, 100 Hz"),
        session=1, block=j["split"].lower(), trial=j["measure"], phase="test", time_s="",
        task=TASK[j["task"]], unit="recording", condition=TASK[j["task"]] + ", " + j["side"] + " arm", pert_type="none", pert_value=0,
        pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=per_sub[subj], samples_file=fn,
        signal_kind=("finger accelerometry (3 axes)" if finger else "wrist accelerometry (3 axes)"),
        channels="x;y;z", channel_units="g;g;g", sample_rate_hz=r["fs_out"], n_samples=r["n_out"],
        duration_s=r["n_out"] / r["fs_out"], events="",
        processing=("as shared (g); anti-alias filtered (8th-order Chebyshev type I, zero phase) and kept at every 10th sample, 2500 Hz to 250 Hz (5000 Hz to 500 Hz for a few files)"
                    if finger else "as shared (g, 100 Hz)"),
        trial_info=(f"tremor_rating_0to4={j['rating']};measurement={j['measure']};authors_split={j['split'].lower()};"
                    f"original_sample_rate_hz={r['fs_raw']};mean_acceleration_g={r['mean_acc']:.6g}")))
D = pd.DataFrame(rows)
part = pd.DataFrame(sorted({(j["subj"], j["disease"]) for j in jobs}, key=lambda x: int(x[0])), columns=["person", "diagnosis"])
pcols = [("diagnosis", "category", "ET, PD", "Essential tremor or Parkinson's disease, from the deposit's rating tables.")]

# ---- published checks
P = checks.published
J = pd.DataFrame(jobs)
for k, v in rec.items():
    pass
J["mean_acc"] = [rec[f"{j['split']}_{j['task']}_{j['device']}_{j['name'][:-4]}"]["mean_acc"] for j in jobs]
J["mean_env"] = [rec[f"{j['split']}_{j['task']}_{j['device']}_{j['name'][:-4]}"]["mean_env"] for j in jobs]
limbs = J.drop_duplicates(["subj", "side"])
P("upper limbs analysed", len(limbs), 54, 0, "2.1 Dataset (54 upper limbs)")
P("ET limbs", int((limbs.disease == "ET").sum()), 27, 0, "2.1 Dataset (ET n = 27)")
P("PD limbs", int((limbs.disease == "PD").sum()), 27, 0, "2.1 Dataset (PD n = 27)")
for split, n_lim in (("Training", 15), ("Validation", 12)):
    s = limbs[limbs.split == split]
    P(f"{split.lower()} ET limbs", int((s.disease == "ET").sum()), n_lim, 0, "2.1 Dataset")
    P(f"{split.lower()} PD limbs", int((s.disease == "PD").sum()), n_lim, 0, "2.1 Dataset")
for split, nf, nw in (("Training", 478, 477), ("Validation", 379, 380)):
    for task in TASK:
        s = J[(J.split == split) & (J.task == task)]
        P(f"{split.lower()} {TASK[task]} trials with finger accelerometry", int((s.device == "Finger").sum()), nf, 0,
          "3.1 / 3.2 (n = 478 and 477 for WristACC in training; 380 and 379 for FingerACC in validation)" if False else "3.1 Training Data / 3.2 Validation Data")
        P(f"{split.lower()} {TASK[task]} trials with wrist accelerometry", int((s.device == "Wrist").sum()), nw, 0,
          "3.1 Training Data / 3.2 Validation Data")

# Table 1: published logarithmic models TR = a ln(T) + b applied to the validation trials (rounded predictions vs rating)
T1 = {  # (task, device, metric): (a, b, RMSE, concordance %, share deviating by 1 point %)
    ("RestingTremor", "Finger", "mean_acc"): (0.502, 2.68, 0.492, 70.7, 29.0),
    ("RestingTremor", "Wrist", "mean_acc"): (0.631, 3.321, 0.489, 74.7, 23.9),
    ("PosturalTremor", "Finger", "mean_acc"): (0.655, 2.932, 0.485, 76.8, 22.7),
    ("PosturalTremor", "Wrist", "mean_acc"): (0.761, 3.812, 0.565, 73.2, 23.7),
    ("RestingTremor", "Finger", "mean_env"): (0.532, 1.958, 0.510, 70.2, 29.6),
    ("RestingTremor", "Wrist", "mean_env"): (0.671, 2.456, 0.518, 72.6, 26.1),
    ("PosturalTremor", "Finger", "mean_env"): (0.701, 1.997, 0.502, 76.5, 23.0),
    ("PosturalTremor", "Wrist", "mean_env"): (0.819, 2.802, 0.569, 72.6, 24.2),
}
detail = []
not_repro = []
for (task, dev, met), (a, b, rmse, conc, dev1) in T1.items():
    s = J[(J.split == "Validation") & (J.task == task) & (J.device == dev)]
    pred = a * np.log(s[met].to_numpy()) + b
    err = pred - s.rating.to_numpy()
    c_rmse = float(np.sqrt(np.mean(err ** 2)))
    pr = np.floor(pred + 0.5)  # MATLAB round: halves away from zero (predictions are positive here)
    c_conc = 100 * float(np.mean(pr == s.rating.to_numpy()))
    c_dev = 100 * float(np.mean(np.abs(pr - s.rating.to_numpy()) == 1))
    detail.append((task, dev, met, c_rmse, rmse, c_conc, conc, c_dev, dev1))
    if met == "mean_env":
        continue  # our envelope did not reproduce the paper's (see differences), so it is not released
    lab = f"{TASK[task]}, {dev.lower()}, mean acceleration, Table 1 log model on validation trials"
    P(lab + ": RMSE", c_rmse, rmse, 0.0005, "Table 1")
    for nm, cv, pv in ((": concordance (%)", c_conc, conc), (": rating off by one point (%)", c_dev, dev1)):
        if abs(cv - pv) <= 0.05:
            P(lab + nm, cv, pv, 0.05, "Table 1")
        else:
            not_repro.append(f"{TASK[task]}, {dev.lower()}, mean acceleration{nm[1:-4]}: {cv:.2f} here, {pv} in Table 1")
print("table 1 detail (task, device, metric, rmse ours/paper, concordance ours/paper, off-by-1 ours/paper):")
for d in detail:
    print("  ", d[0], d[1], d[2], f"rmse {d[3]:.4f}/{d[4]}", f"conc {d[5]:.2f}/{d[6]}", f"dev1 {d[7]:.2f}/{d[8]}")

checks.internal("rows = accelerometry files", len(D) == 3428 and len(extra) == 3428)
checks.internal("every rating is 0-4", J.rating.between(0, 4).all())
import collections
print("raw rates", collections.Counter((("Finger" if "Finger" in k else "Wrist"), r["fs_raw"]) for k, r in rec.items()))
n5000 = sum(1 for k, r in rec.items() if "Finger" in k and r["fs_raw"] == 5000)
checks.internal("finger files 2500 Hz (or 5000 Hz, listed in differences), wrist files 100 Hz",
                all(r["fs_raw"] in ((2500, 5000) if "Finger" in k else (100,)) for k, r in rec.items()))
checks.internal("no sample files lost", sum(r["n_out"] for r in rec.values()) == int(D.n_samples.sum()))

meta = dict(
    paper=dict(authors="van der Linden C, Berger T, Brandt GA, Strelow JN, Jergas H, Baldermann JC, Visser-Vandewalle V, Fink GR, "
                       "Barbe MT, Petry-Schmelzer JN, Dembek TA", year=2023,
               title="Accelerometric Classification of Resting and Postural Tremor Amplitude", venue="Sensors 23(20):8621",
               doi="10.3390/s23208621"),
    lab="Baldermann / Visser-Vandewalle (University Hospital Cologne)", data_url="https://osf.io/mp5ha/",
    license="CC BY-NC-SA 4.0", license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    changes=("Each of the 3,428 accelerometry CSV files (finger and wrist, training and validation folders) became one samples CSV with "
             "time_s and the x, y, z accelerations in g; the finger recordings were reduced tenfold (anti-alias filter and "
             "every 10th sample: 2500 Hz to 250 Hz), the wrist recordings are unchanged. The clinician's rating, the diagnosis and the authors' split come "
             "from the deposit's rating tables. mean_acceleration_g in trial_info is our computation from the "
             "full-rate file with the definition in the authors' MATLAB function (first 2 s dropped, 3-12 Hz band-pass, mean magnitude); they are not "
             "values supplied by the authors. Age, sex and stimulation settings are not in the deposit."),
    how=["One row is one 20 s tremor trial recorded by one device: the BrainProducts accelerometer on the proximal dorsum of the index finger or the "
         "GENEActiv wristwatch accelerometer, while the person held the arm still at rest (resting tremor) or in a posture with the arm outstretched "
         "(postural tremor; MDS-UPDRS 3.15 and 3.17). Trials were repeated under different deep brain stimulation settings (measurement numbers "
         "within a limb); the settings are not in the deposit. Each file starts 2 s before the 20 s window used for analysis (about 21 s long).",
         "Samples (samples/*.csv): time_s from the start of the file, then x_g, y_g, z_g in g. The axes follow the authors' common hand system "
         "(x towards the thumb, y towards the fingertips, z up from the back of the hand, Methods 2.5).",
         "trial_info holds the clinician's rating (MDS-UPDRS amplitude 0-4, the maximum amplitude seen in the 20 s), the measurement number, the "
         "authors' training or validation split, and the mean acceleration computed by us. subject is one arm of one person "
         "(subject = patient ID + side), person is the patient ID of the deposit."],
    differences=[f"{n5000} of the finger recordings in the deposit have a 5000 Hz sampling rate (from their time stamps), not the 2500 Hz of the "
                 "paper's Methods; they were reduced to 500 Hz instead of 250 Hz, and sample_rate_hz of each row says which.",
                 "Table 1 values that did not reproduce and were left out of the checks: " + "; ".join(not_repro) + ". The RMSE of all four "
                 "mean-acceleration models does reproduce to the printed three decimals. The concordance differences are 1-2 trials of 380; our "
                 "Butterworth filter (second-order sections, scipy) is not bit-identical to MATLAB's, and a prediction near a rounding boundary can flip.",
                 "The deposit's SubjectID takes 46 distinct values (22 ET, 24 PD) while the paper reports 41 patients (16 ET, 25 PD); the paper's "
                 "limb counts (54; 27 ET, 27 PD) do reproduce. The extra IDs may be the same person recorded under several IDs; this is not "
                 "documented, so person here is the deposit's ID.",
                 "The paper's Methods give 479 training and 381 validation trials (860 in total); its Results and the deposit's rating tables give "
                 "478 and 380 per tremor type (the numbers that reproduce above). One wrist training trial and one finger validation trial per "
                 "tremor type have no accelerometry file in the deposit, as the paper's Results also say.",
                 "Peak power, AUC power and mean envelope of Table 1 are not released: we implemented the envelope as MATLAB's moving-RMS envelope "
                 "from the authors' code, but the log models of Table 1 applied to it gave RMSE 0.56-0.65 instead of 0.51-0.57, so it does not match "
                 "the authors' value, and the spectral metrics were not attempted."],
    unit_note="recording (one 20 s trial, one device)",
)
stdlib.release(ID, "timeseries", D, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
