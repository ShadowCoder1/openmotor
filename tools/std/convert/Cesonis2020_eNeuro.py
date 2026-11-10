"""Cesonis & Franklin (2020), eNeuro: time-to-target and visuomotor feedback responses (retry of a 'too large' skip).
Raw file (untrusted data, read only, never loaded whole): raw_extra/Cesonis2020_eNeuro/TTT_subdata.mat (3.6 GB MATLAB v7.3 = HDF5,
from https://doi.org/10.6084/m9.figshare.11323289, CC BY 4.0). FrameData.<channel>[sub, cond] and TrialData.<field>[sub, cond] are object
references to (2000 frames x 880 trials) and (880) arrays (10 participants x 5 movement conditions); read one cell at a time.
MATLAB_timetotarget.zip (the authors' analysis code, read as text only) gives the codes: PertCond 1-11 = channel (probe) trials, 12-22 =
maintained jump in a null field; 1 and 12 no jump; MovementCondition 1 matched-cursor late-peak, 2 matched-cursor early-peak,
3 matched-hand early-peak, 4 matched-hand late-peak, 5 baseline.
One row per trial (44000), plus 50 sample files (participant x condition) with the trajectories at 100 Hz (every 10th 1 kHz frame)."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cesonis2020_eNeuro"
RAW = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "tmp_samples" / ID
TMP.mkdir(parents=True, exist_ok=True)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.columns_for("force_field_reaching")]

CONDS = {1: "matched-cursor late-peak", 2: "matched-cursor early-peak", 3: "matched-hand early-peak",
         4: "matched-hand late-peak", 5: "baseline"}
START_Y = -17.5  # the authors' start position (Fig3BC.m: (PY + 17.5) * 4)
f = h5py.File(RAW / "TTT_subdata.mat", "r")


def trial_arr(field, s, m):
    return np.array(f[f["TrialData"][field][s, m]]).ravel().astype(float)


def frame_arr(ch, s, m):
    return np.array(f[f["FrameData"][ch][s, m]]).T.astype(np.float64)  # (880 trials, 2000 frames)


rows, sample_files = [], []
dur_by = {}  # (sub, cond, pertcond) -> list of durations
n_valid_nan = 0
for s in range(10):
    trial_seq = 0
    for m in range(5):
        cond = m + 1
        T = {k: trial_arr(k, s, m) for k in ("CondNo", "HandSpeed", "MovementCondition", "PeakDC", "PeakDH", "PeakSpeed", "PertCond",
                                             "PertTime", "Success", "TrDur", "TrialNumber")}
        assert (T["MovementCondition"] == cond).all()
        assert list(T["TrialNumber"]) == list(range(1, 881))
        PX, PY, CX, CY = (frame_arr(k, s, m) for k in ("PX", "PY", "CX", "CY"))
        FTX, FTY, VY, TT, SG = (frame_arr(k, s, m) for k in ("FTX", "FTY", "VY", "TT", "SpeedGain"))
        sample_rows = []
        for t in range(880):
            trial_seq += 1
            pc = int(T["PertCond"][t])
            dur = T["TrDur"][t]
            tt = TT[t]
            n = int(np.sum(tt <= dur))  # frames of the movement (TT is NaN after the record ends)
            assert n > 50
            jump_trial = pc not in (1, 12)
            q = (pc - 1) % 11  # 0 = no jump, 1..10 = jump codes
            sign = 0 if q == 0 else (-2.0 if q % 2 == 1 else 2.0)  # even code (2,4,..) = -2 cm: measured from CX - PX after the jump
            loc = None if q == 0 else (q - 1) // 2 + 1
            pt = T["PertTime"][t]
            k = int(np.argmax(tt > pt))  # first frame after the jump time (the authors: find(TT > PertTime, 1))
            # feedback window 180-230 ms after the jump (authors: first frame with TT > PertTime + 0.18 to first with TT > PertTime + 0.23)
            a = np.searchsorted(tt[~np.isnan(tt)], pt + 0.180, side="right")
            b = np.searchsorted(tt[~np.isnan(tt)], pt + 0.230, side="right")
            fb = float(np.nanmean(FTX[t, a:b + 1])) if b >= a else np.nan
            base = np.nanmean(FTX[t, :50])
            i_pf = int(np.nanargmax(np.abs(FTX[t, :n] - base)))
            pf = float(FTX[t, i_pf] - base)
            vH = VY[t, k]
            vC = vH * SG[t, k]
            rows.append(dict(
                dataset=ID, study="Cesonis & Franklin 2020, eNeuro", experiment="visuomotor feedback responses", group="",
                subject=f"P{s + 1:02d}", subject_excluded_by_authors=False, age=None, sex="", handedness="right",
                population="healthy", setting="lab", device="vBOT planar robotic manipulandum, air sled, monitor and mirror",
                session=cond, block=f"block {t // 22 + 1}", trial=trial_seq, phase="test", time_s=None, task="reaching",
                unit="trial", condition=CONDS[cond],
                pert_type="cursor_jump" if jump_trial else "none", pert_value=sign, pert_unit="cm",
                feedback="normal" if (cond == 5 and not jump_trial) else "altered", excluded_by_authors=False,
                trial_type="channel" if pc <= 11 else "null", trial_in_condition=t + 1, block_in_condition=t // 22 + 1,
                movement_duration_s=dur, peak_speed_m_s=T["PeakSpeed"][t] / 100.0, speed_goal_met=bool(T["Success"][t]),
                target_direction_deg=90, hand="right",
                max_lateral_displacement_cm=float(np.nanmax(np.abs(PX[t, :n]))),
                max_cursor_lateral_displacement_cm=float(np.nanmax(np.abs(CX[t, :n]))),
                endpoint_x_cm=float(PX[t, n - 1]),
                cursor_jump_onset_s=pt, hand_distance_at_jump_cm=float(PY[t, k] - START_Y),
                jump_location_index=loc, hand_velocity_at_jump_cm_s=float(vH), cursor_velocity_at_jump_cm_s=float(vC),
                time_to_target_at_jump_s=dur - pt, feedback_force_x_180_230ms_n=fb, peak_force_x_n=pf,
                peak_hand_distance_fraction=T["PeakDH"][t], peak_cursor_distance_fraction=T["PeakDC"][t],
                reference_speed_modality="hand" if T["HandSpeed"][t] == 1 else "cursor"))
            idx = np.arange(0, n, 10)
            for i in idx:
                sample_rows.append((trial_seq, tt[i], PX[t, i], PY[t, i], CX[t, i], CY[t, i], FTX[t, i], FTY[t, i]))
            dur_by.setdefault((s, cond, pc), []).append(dur)
        # sample file for this participant and condition
        fname = f"samples/{ID}_S{s + 1:04d}_cond{cond}.csv"
        p = TMP / Path(fname).name
        with open(p, "w", newline="") as fh:
            fh.write("trial,time_s,hand_x_cm,hand_y_cm,cursor_x_cm,cursor_y_cm,force_x_n,force_y_n\r\n")
            for r in sample_rows:
                fh.write("%d,%.3f,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g\r\n" % r)
        sample_files.append((fname, str(p)))
        print("done participant", s + 1, "condition", cond, flush=True)

# ------------------------------------------------------------------ checks
S6 = "Results, movement durations (Fig. 6A and text)"
# maintained-jump trials (codes 13-22), mean per participant of the means per code, then across participants (authors' Fig6 code)
def cond_mean(cond):
    per_sub = []
    for s in range(10):
        per_sub.append(np.mean([np.mean(dur_by[(s, cond, pc)]) for pc in range(13, 23)]))
    return np.mean(per_sub) * 1000

checks.published("movement time, baseline condition, maintained-jump trials (ms)", cond_mean(5), 932, 0.5, S6)
checks.published("movement time, matched-cursor late-peak, maintained-jump trials (ms)", cond_mean(1), 1048, 0.5, S6)
checks.published("movement time, matched-cursor early-peak, maintained-jump trials (ms)", cond_mean(2), 1201, 0.5, S6)
S2 = "Methods, Participants and Experimental design"
checks.published("participants analysed", len({r["subject"] for r in rows}), 10, 0, S2)
checks.published("trials per session", len(rows) / 50, 880, 0, S2)
checks.published("probe (force channel) trials per session", sum(1 for r in rows if r["trial_type"] == "channel") / 50, 440, 0, S2)
checks.published("trial types (jump locations x 2 directions + no jump) per probe or maintained set",
                 len({(r["pert_value"], r["jump_location_index"]) for r in rows}), 11, 0, S2)
checks.internal("44000 rows", len(rows) == 44000, str(len(rows)))
checks.internal("every participant has 5 conditions x 880 trials", all(sum(1 for r in rows if r["subject"] == f"P{s + 1:02d}") == 4400 for s in range(10)))
checks.internal("jump size is 2 cm in absolute value on jump trials", all(abs(r["pert_value"]) == 2.0 for r in rows if r["pert_type"] == "cursor_jump"))
checks.internal("40 repetitions of every trial type per participant and condition", all(len(v) == 40 for v in dur_by.values()))
checks.internal("ideal-trial flag implies reference peak speed of 42-58 cm/s",
                all(41.9 <= r["peak_speed_m_s"] * 100 <= 58.1 for r in rows if r["speed_goal_met"]),
                "violations: %d" % sum(1 for r in rows if r["speed_goal_met"] and not (41.9 <= r["peak_speed_m_s"] * 100 <= 58.1)))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8")) if e["id"] == ID)["papers"][0]
meta = dict(
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Franklin (Technical University of Munich)", data_url="https://doi.org/10.6084/m9.figshare.11323289",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="the 3.6 GB MATLAB file of 1 kHz kinematics and forces was reduced to one row of per-trial measures (the authors' stored "
            "trial values plus measures computed from the stored trajectories as defined in columns.csv) and to 100 Hz trajectory "
            "files; participant codes replaced; cm/s converted to m/s for peak speed; the authors' analysis code and model files are not included.",
    unit_note="trial (44000 trials: 10 participants x 5 movement conditions x 880 trials)",
    how=["One row is one reach of one participant: 10 participants made 880 reaches in each of 5 sessions, one session per movement "
         "condition (baseline, matched-hand early and late peak, matched-cursor early and late peak). Half of the trials were "
         "probe trials in a mechanical force channel (trial_type channel) and half were reaches in a null field (null); on both "
         "kinds, the cursor jumped 2 cm sideways at one of five hand positions, or did not jump (pert_value 0).",
         "The deposit has the full 1 kHz records of hand and cursor position, velocity and lateral force for every trial, 3.6 GB. They "
         "are not released in full. The table holds the authors' trial values (movement time, peak speed, ideal-trial flag, peak "
         "distances, jump time) and measures computed from the records with the authors' definitions where they have them: "
         "the visuomotor feedback measure is the mean lateral force from 180 to 230 ms after the jump (the paper's measure, not "
         "baseline-corrected and not yet the left-minus-right difference), and the time to target is movement time minus jump time. "
         "Peak lateral displacement, end point and peak lateral force are standard measures that the paper does not tabulate: they "
         "are computed over the movement (from the first sample to the movement time TrDur).",
         "samples/ holds, for each participant and condition, the trajectories of all 880 trials at 100 Hz (every tenth sample "
         "of the 1 kHz record, no extra filtering; the authors had low-pass filtered at 40 Hz): trial (same numbering as the table's "
         "trial column), time since the start of the record, hand x and y, cursor x and y (cm), lateral and forward force on the "
         "handle (N). x is the lateral axis, y the reach direction; the start is at y = -17.5 cm and the target at y = +7.5 cm "
         "(25 cm reach). The deposit does not say whether positive x is left or right.",
         "The sign of pert_value is the sign of the cursor offset (cursor x minus hand x, about 100 ms after the jump): the even "
         "perturbation codes 2, 4, ... and 13, 15, ... are -2 cm, the odd ones +2 cm. Session is the movement condition number "
         "(the deposit does not give the order in which the five sessions were run), so the trial numbering of the CORE columns "
         "runs through condition 1 first, then 2 to 5, within each condition in the authors' trial order.",
         "Participants are the 10 analysed in the paper (a further participant, excluded by the authors for overlapping kinematic profiles, is not in the deposit)."],
    unit_note_extra="",
    differences=["Peak speed is the speed of the reference modality (cursor in the matched-cursor conditions), as the authors store it; the "
                 "meaning of the hand-speed flag and of PeakDH and PeakDC is inferred from the values (see columns.csv).",
                 "Movement times check against the paper's means for the maintained-jump trials of the baseline and matched-cursor conditions. "
                 "The paper's other numbers (feedback intensity ANOVA, onset latencies) need the left-minus-right difference per participant and the "
                 "authors' normalisation and are not reproduced here."])
meta.pop("unit_note_extra")
stdlib.release(ID, "force_field_reaching", rows, meta, checks, extra_files=sample_files)
