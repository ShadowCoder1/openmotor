"""Payne & Ting (2020) Gait & Posture 80:324-330: 20 healthy young adults given 48 backward support-surface translations (three
magnitudes, step or do-not-step instruction) while EEG, EMG, head and GSR signals were recorded; plus a beam-walking balance score.
Deposit: Dryad doi:10.5061/dryad.qz612jm9x (CC0), taken from its identical copy on Zenodo record 3898501 (the Dryad download needs a token).
Raw: raw_extra/Payne2020_GaitPosture/dryad_combined_with_beta.mat (MATLAB v5, plain arrays).
Run from the site folder: python3 -I tools/std/convert/Payne2020_GaitPosture.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Payne2020_GaitPosture"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
m = sio.loadmat(raw / "dryad_combined_with_beta.mat")
F = m["Flags"].astype(int)
S = m["Subject_data"]
t_e = m["time_erps"].ravel()
t_m = m["time_emg"].ravel()
t_g = m["gsr_time"].ravel()
assert np.allclose(t_e[100:], t_m) and abs(t_e[100] + 0.4) < 1e-9
EPOCH = ["C1", "C2", "CPz", "Cz", "FCz", "Fz", "Oz", "Pz", "Hx", "Hy", "Hz", "Y", "MG", "TA"]
UNIT = {"MG": "unspecified (EMG, as shared)", "TA": "unspecified (EMG, as shared)", "Y": "arbitrary (platform accelerometer, units meaningless per the authors)",
        "Hx": "unspecified", "Hy": "unspecified", "Hz": "unspecified"}
units = [UNIT.get(c, "microvolt (EEG)") for c in EPOCH]
MAG = {1: "small", 2: "medium", 3: "large"}
DEVICE = ("custom perturbation platform (Factory Automation Systems); 32-channel ActiCAP EEG and vertical EOG on a Brain Products ActiCHamp "
          "(1000 Hz); EMG; Vicon motion capture (not in the deposit)")
subs = [int(s) for s in S[:, 0]]
assert sorted(set(F[:, 0])) == sorted(subs)
age = {int(r[0]): r[1] for r in S}
rows, files = [], []
trial_n = {}


def fmt(a):
    return ",".join("" if not np.isfinite(x) else format(x, ".7g") for x in a)


for i in range(len(F)):
    sid = int(F[i, 0])
    trial_n[sid] = trial_n.get(sid, 0) + 1
    k = trial_n[sid]
    stepped = int(F[i, 4])
    base = dict(dataset=DS, study="Payne & Ting 2020 Gait & Posture", experiment="", group="", subject=f"S{sid:02d}", subject_excluded_by_authors=False,
                age=float(age[sid]), population="healthy young adults", setting="lab", device=DEVICE, session=1, block="", trial=k, phase="test",
                task="balance and posture", unit="recording",
                condition=f"{MAG[F[i, 3]]} perturbation, instructed to {'step' if F[i, 1] else 'resist stepping'}",
                pert_type="support_surface_translation_backward", pert_value=float(F[i, 3]), pert_unit="magnitude level (1 small, 2 medium, 3 large)",
                feedback="normal", excluded_by_authors=False)
    ev = "perturbation_onset=0"
    info = (f"instruction_step={int(F[i, 1])};platform_direction_deg={int(F[i, 2])};stepped_by_1000ms={stepped};"
            f"stepped_by_2000ms={int(F[i, 5])}")
    fn = f"samples/P{sid:02d}_trial{k:03d}.csv"
    rows.append({**base, "recording": k, "samples_file": fn,
                 "signal_kind": "perturbation-aligned epoch: scalp EEG (C1, C2, CPz, Cz, FCz, Fz, Oz, Pz), head signals (Hx, Hy, Hz), platform accelerometer (Y), EMG (MG, TA)",
                 "channels": ";".join(EPOCH), "channel_units": ";".join(units), "sample_rate_hz": 1000, "n_samples": len(t_e),
                 "duration_s": float(t_e[-1] - t_e[0]), "events": ev,
                 "processing": ("EEG: 1 Hz high-pass, 25 Hz low-pass, re-referenced to mastoids, blink/eye artefact subtracted, baseline-corrected (-150 to -50 ms) "
                                "(Cz as described in the paper; the other channels are as shared); EMG epoch starts at -0.4 s, earlier samples are empty"),
                 "trial_info": info})
    files.append(("epoch", fn, i))
    g = m["GSR"][i]
    if np.isfinite(g).any():
        fg = f"samples/P{sid:02d}_trial{k:03d}_gsr.csv"
        rows.append({**base, "recording": k, "samples_file": fg, "signal_kind": "GSR (galvanic skin response), named GSR in the deposit",
                     "channels": "GSR", "channel_units": "unspecified", "sample_rate_hz": 1000, "n_samples": len(t_g),
                     "duration_s": float(t_g[-1] - t_g[0]), "events": ev, "processing": "as shared; samples that are empty in the deposit stay empty",
                     "trial_info": info})
        files.append(("gsr", fg, i))
df = pd.DataFrame(rows).reindex(columns=[c[1] for c in stdlib.columns_for("timeseries") if c[1] != "person"])


def extra():
    for kind, fn, i in files:
        if kind == "epoch":
            cols = []
            for c in EPOCH:
                v = m[c][i]
                if c in ("MG", "TA"):
                    v = np.concatenate([np.full(100, np.nan), v])
                cols.append(v)
            a = np.column_stack(cols)
            yield fn, "time_s," + ",".join(EPOCH) + "\n" + "".join(f"{t_e[j]:.3f}," + fmt(a[j]) + "\n" for j in range(len(t_e)))
        else:
            g = m["GSR"][i]
            yield fn, "time_s,GSR\n" + "".join(f"{t_g[j]:.3f}," + fmt([g[j]]) + "\n" for j in range(len(t_g)))


pcols = [("height_cm", "number", "cm", "Height (Subject_data in the deposit)."),
         ("normalized_beam_distance", "number", "fraction, 1 = end of the beam reached on all 6 trials", "Balance ability from the beam-walking task."),
         ("peak_com_cm", "number", "cm", "Peak centre-of-mass displacement along the axis of platform motion, averaged over non-stepping responses to small perturbations, 1-2 s after perturbation (empty for the participant excluded for a missing marker)."),
         ("peak_trunk_angle_deg", "number", "degrees", "Peak trunk angle from vertical, averaged over non-stepping responses to large perturbations, 0.5-1 s after perturbation (empty where the deposit has no value).")]
nn = lambda v: None if np.isnan(v) else float(v)  # noqa: E731
part = [dict(person=f"S{int(r[0]):02d}", height_cm=float(r[2]), normalized_beam_distance=float(r[3]), peak_com_cm=nn(r[4]), peak_trunk_angle_deg=nn(r[5])) for r in S]

# checks
n1 = np.array([m["Cz"][F[:, 0] == s].mean(0)[(t_e >= 0.1) & (t_e <= 0.2)].min() for s in subs])
cnt = {s: int((F[:, 0] == s).sum()) for s in subs}
ck.published("participants", len(subs), 20, 0, "Abstract / Methods")
ck.published("mean age, years", S[:, 1].mean(), 26, 0.5, "Methods: Participants (26 years, SD 5)")
ck.published("SD of age, years", S[:, 1].std(ddof=1), 5, 0.5, "Methods: Participants")
ck.published("mean height, cm", S[:, 2].mean(), 168, 0.5, "Methods: Participants (168 cm, SD 8)")
ck.published("SD of height, cm", S[:, 2].std(ddof=1), 8, 0.5, "Methods: Participants")
ck.published("shortest participant, cm", S[:, 2].min(), 156, 0.5, "Methods: Participants (range 156-185 cm)")
ck.published("tallest participant, cm", S[:, 2].max(), 185, 0.5, "Methods: Participants (range 156-185 cm)")
ck.published("mean cortical N1 amplitude (magnitude of the Cz minimum 100-200 ms in each participant's average), microvolt", -n1.mean(), 54, 0.5, "Results (54 +/- 18 microvolt)")
ck.published("SD of N1 amplitude, microvolt", n1.std(ddof=1), 18, 0.5, "Results (54 +/- 18 microvolt)")
ck.published("participants with a centre-of-mass value (one excluded for a missing marker)", int((~np.isnan(S[:, 4])).sum()), 19, 0, "Methods: CoM position and trunk angle (N=19)")
ck.published("participants with exactly 48 perturbations", sum(v == 48 for v in cnt.values()), 19, 0, "Methods: Perturbations (48 each); one participant has 50 rows in the deposit")
ck.published("small perturbations per participant (one third of 48), participant with 48", int(((F[:, 0] == subs[0]) & (F[:, 3] == 1)).sum()), 16, 0, "Methods: Perturbations (evenly divided)")
ck.internal("all 962 trials kept, one epoch row each", int((df.signal_kind.str.startswith("perturbation-aligned")).sum()) == len(F) == 962)
ck.internal("every trial has 2501 epoch samples", bool((df[df.channels.str.contains("Cz")].n_samples == 2501).all()))
ck.internal("beam distances between 0 and 1", bool((S[:, 3] >= 0).all() and (S[:, 3] <= 1).all()))

meta = dict(
    paper=dict(authors="Payne AM, Ting LH", year=2020, title="Worse balance is associated with larger perturbation-evoked cortical responses in healthy young adults",
               venue="Gait & Posture 80:324-330", doi="10.1016/j.gaitpost.2020.06.018"),
    lab="Ting (Emory / Georgia Tech)", data_url="https://doi.org/10.5061/dryad.qz612jm9x", license="CC0 1.0",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Kept every trial of the deposit's combined MATLAB file (962 perturbation trials of 20 participants): the 8 EEG channels, 3 head channels, "
             "platform accelerometer and 2 EMG channels as one CSV per trial, and the GSR trace as a second CSV where it has any values, with a time "
             "column added and values written to 7 significant digits. The single-trial beta traces (200 values per trial, derived by the authors, units "
             "not described) are not included; participants are coded."),
    how=["One row is one trial: a backward translation of the support surface (direction 270 degrees in the deposit) at one of three magnitudes, "
         "with the participant instructed to step or to resist stepping. Time 0 is perturbation onset; epochs run from -0.5 to 2.0 s at 1000 Hz "
         "(EMG from -0.4 s; earlier samples are empty). A second row per trial holds the GSR trace (-1 to 20 s) where the deposit has values.",
         "The small perturbation was 7.7 cm for all participants; medium (12.6-15.0 cm) and large (18.4-21.9 cm) were scaled to height (Methods); "
         "pert_value gives the magnitude level 1-3. stepped_by_1000ms / stepped_by_2000ms in trial_info mark trials where a step was taken "
         "(vertical force below 5 N within 1000 / 2000 ms of platform onset); steps also occur on some do-not-step trials.",
         "The deposit does not describe the channels Hx, Hy, Hz (head signals; name only), the units of EMG and GSR, or the other EEG electrodes; "
         "they are given as shared with unit 'unspecified'. One participant has 50 trials instead of 48. Participant 14 does not exist in the "
         "deposit's numbering, so 20 participants remain."],
    unit_note="perturbation trial epoch or GSR trace (samples in samples/)",
    differences=["Peak trunk angle is empty for two participants in the deposit (paper: one participant excluded for a missing marker, N=19 for CoM and trunk angle); "
                 "peak CoM is empty for one."],
)
stdlib.release(DS, "timeseries", df, meta, ck, participants=part, participant_columns=pcols, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
