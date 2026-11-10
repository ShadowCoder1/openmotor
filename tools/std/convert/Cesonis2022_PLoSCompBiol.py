"""Cesonis & Franklin (2022), PLoS Comput Biol: task-dependent switching of feedback controllers. 14 adults make 25 cm forward reaches
with a vBOT robot, either to stop at the target ("stop") or to pass through it ("hit"), in blocked and mixed schedules; on channel
trials the target is shifted laterally by 2 cm for 250 ms. The deposit holds the 1144 channel (perturbation or zero-perturbation)
trials of each participant, 1 s at 1 kHz each. One row per trial, the trial's samples as a CSV inside the ZIP (module timeseries).

Run from the site folder: python3 -I tools/std/convert/Cesonis2022_PLoSCompBiol.py
Raw file (untrusted data, read only): raw_extra/Cesonis2022_PLoSCompBiol/Controller_switching.zip (Figshare 17113904): matlab_data1.mat
(FrameData and TrialData, one element per participant), triallist.mat (not used), and the authors' notebooks read as text only to
learn the variable meanings (never run).

Codes (inferred where the deposit has no readme): PertCond 1 = zero-magnitude channel trial, 2-11 = target shifts by onset position,
pairs (2,3) 4.2 cm, (4,5) 8.3, (6,7) 12.5, (8,9) 16.7, (10,11) 20.8 cm, even code = TX -2 cm, odd = TX +2 cm (analysis.ipynb;
checked on TX and on the cursor position at shift onset). TrialCond 1-4: 1 and 2 are the two blocked conditions (ActualTrials <= 832),
3 and 4 the mixed schedule; stop vs hit was decided from the terminal velocity at the end of the movement, which is low (about 14
cm/s) for codes 1 and 3 (stop) and high (about 34-36 cm/s) for 2 and 4 (hit), as in the paper (hit terminal velocity 38.5 cm/s).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cesonis2022_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
checks = stdlib.Checks()
CH = ["PX", "PY", "VX", "VY", "CX", "CY", "FX", "FY", "FTX", "FTY", "TX"]
UNITS = "cm;cm;cm/s;cm/s;cm;cm;N;N;N;N;cm"
FS = 1000.0
COND = {1: ("stop", "blocked"), 2: ("hit", "blocked"), 3: ("stop", "mixed"), 4: ("hit", "mixed")}
ONSET_CM = {2: 4.2, 3: 4.2, 4: 8.3, 5: 8.3, 6: 12.5, 7: 12.5, 8: 16.7, 9: 16.7, 10: 20.8, 11: 20.8}

d = sio.loadmat(RAW / "Controller_switching/matlab_data1.mat", squeeze_me=False)
FD, TD = d["FrameData"][0], d["TrialData"][0]
rows, files = [], []
shift_pos = {c: [] for c in ONSET_CM}
shift_len, shift_amp = [], []
cond_n = {}
term_ok = True
for p in range(len(TD)):
    f, t = FD[p], TD[p]
    A = {n: f[n].astype(float) for n in f.dtype.names}
    tc, pcn, at = (t[n].ravel().astype(int) for n in ("TrialCond", "PertCond", "ActualTrials"))
    pt, trd = t["PertTime"].ravel().astype(float), t["TrDur"].ravel().astype(float)
    assert A["PX"].shape == (1144, 1000) and (np.diff(at) > 0).all()
    tv = {}
    for i in range(1144):
        k = min(int(np.searchsorted(A["TT"][i], trd[i])), 999)
        tv.setdefault(tc[i], []).append(A["VY"][i, max(k - 5, 0)])
    mt = {c: float(np.mean(v)) for c, v in tv.items()}
    term_ok &= min(mt[2], mt[4]) > max(mt[1], mt[3]) + 10
    for i in range(1144):
        mov, sched = COND[tc[i]]
        fname = f"s{p + 1:02d}_t{i + 1:04d}.csv"
        path = TMP / fname
        tx = A["TX"][i]
        nz = np.where(np.abs(tx) > 1)[0]
        if len(nz):
            shift_pos[pcn[i]].append(A["PY"][i, nz[0]] - A["PY"][i, 0])
            shift_len.append(len(nz)); shift_amp.append(abs(tx[nz]).mean())
            ev = f"target_shift_on={A['TT'][i, nz[0]]:.3f};target_shift_off={A['TT'][i, nz[-1]]:.3f};"
        else:
            ev = ""
        ev += f"perturbation_onset={pt[i]:.4f};movement_end={trd[i]:.4f}"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\r\n")
            w.writerow(["time_s"] + CH)
            for j in range(1000):
                w.writerow([f"{A['TT'][i, j]:.3f}"] + [format(A[c][i, j], ".6g") for c in CH])
        files.append((f"samples/{fname}", str(path)))
        cond_n[(p, tc[i])] = cond_n.get((p, tc[i]), 0) + 1
        shift = float(tx[nz[0]]) if len(nz) else 0.0
        rows.append(dict(
            dataset=ID, study="Cesonis & Franklin 2022 PLoS Comput Biol", experiment=f"{ID}_{sched}_{mov}" if False else ID, group="",
            subject=f"S{p + 1:02d}", person=f"S{p + 1:02d}", subject_excluded_by_authors=False, age="", sex="", handedness="right",
            population="healthy", setting="lab", device="vBOT planar robot", session=1,
            block=f"{sched}_schedule", trial=i + 1, phase="test", time_s="", task="reaching", unit="recording",
            condition=f"{mov}_{sched}", pert_type="target_shift" if len(nz) else "none", pert_value=round(shift, 3) if len(nz) else 0,
            pert_unit="cm", feedback="normal", excluded_by_authors=False, recording=i + 1, samples_file=f"samples/{fname}",
            signal_kind="hand position and velocity, cursor position, robot forces and target shift during a forward reach in a force channel",
            channels=";".join(CH), channel_units=UNITS, sample_rate_hz=FS, n_samples=1000, duration_s=1.0, events=ev,
            processing="the authors' preprocessed traces, one second from the start of the recording; the filtering is described in the paper "
                       "as a 10th-order zero-phase Butterworth at 15 Hz for analysis (whether the stored traces are filtered is not stated)",
            trial_info=f"movement={mov};schedule={sched};trial_cond_code={tc[i]};pert_cond_code={pcn[i]};actual_trial_in_session={at[i]};"
                       f"shift_onset_cm_from_start={ONSET_CM.get(pcn[i], 'none')};field_code=2"))

n_p = len(TD)
checks.published("participants", n_p, 14, 0, "Methods, Participants (14)")
checks.internal("hit conditions have a clearly higher terminal velocity than stop conditions for every participant", term_ok)
per_cond = sorted({v for v in cond_n.values()})
checks.published("channel trials per movement type and schedule (26 blocks x 11 channel trials)", per_cond[0], 286, 0,
                 "Methods, Trial structure (each 16-trial block has 11 channel trials; 26 blocks per condition)")
checks.internal("all 4 conditions have the same number of stored trials for all participants", per_cond == [286])
checks.published("channel trials per participant", sum(1 for r in rows if r["subject"] == "S01"), 1144, 0,
                 "Methods, Trial structure (4 conditions x 26 blocks x 11 channel trials)")
checks.published("zero-magnitude channel trials per movement type and schedule (1 per block, 26 blocks)",
                 sum(1 for r in rows if r["subject"] == "S01" and r["pert_type"] == "none" and r["condition"] == "hit_blocked"), 26, 0,
                 "Methods, Perturbations (one zero-magnitude channel trial per block)")
checks.published("target shift size (cm)", float(np.mean(shift_amp)), 2.0, 0.01, "Methods, Perturbations (shifted 2.0 cm laterally)")
checks.published("target shift duration (ms)", float(np.mean(shift_len)), 250, 2, "Methods, Perturbations (for 250 ms)")
for c in sorted(ONSET_CM):
    checks.published(f"shift onset position, PertCond {c} (cm from the start)", float(np.mean(shift_pos[c])), ONSET_CM[c], 0.35,
                     "Methods, Perturbations (4.2, 8.3, 12.5, 16.7 and 20.8 cm)")
checks.internal("each of the 10 shift codes has 104 trials per participant... 26 per condition", all(
    sum(1 for r in rows if r["subject"] == "S01" and r["trial_info"].count(f"pert_cond_code={c};")) == 104 for c in range(1, 12)))
checks.internal("rows = 14 x 1144, one sample file each", len(rows) == 14 * 1144 and len(files) == len(rows))

paper = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Franklin (TU Munich)", data_url="https://doi.org/10.6084/m9.figshare.17113904",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="the authors' MATLAB structures (trials x samples) were written as one CSV per trial, with the time column from the authors' TT; "
            "values keep 6 significant digits; participant codes replaced.",
    unit_note="channel trial (recording) of one participant, in the order of the authors' trial numbers",
    how=[
        "One row is one channel trial of one of 14 adults doing 25 cm forward reaches with a vBOT robot, either stopping at the target "
        "(stop) or passing through it into a stopping zone (hit), in a blocked schedule (first 832 trials, one movement type per block) "
        "then a mixed schedule. Only channel trials are in the deposit (1144 of 1664 trials per participant: 11 of every 16); on them the "
        "target is shifted 2 cm sideways for 250 ms (10 codes: 5 onset positions x 2 sides) or not shifted (zero-magnitude, pert_value 0). "
        "Rows are in the authors' trial order; trial_info keeps the authors' trial number in the session.",
        "samples_file holds 1000 samples at 1 kHz from the start of the recording: hand position PX, PY and velocity VX, VY, cursor CX, CY, "
        "forces FX, FY and FTX, FTY, and TX, the lateral target offset (cm, N, cm/s; the shift is TX = -2 or +2). The authors' abbreviations were kept; "
        "the deposit has no readme, so they were read from the authors' analysis notebook (FTX is the force they analyse in the 180-230 ms window "
        "after shift onset) and from the value ranges: PY runs from about -17 cm at the start along the reach.",
        "Which condition is stop and which is hit was inferred from the terminal velocity (see the converter); even PertCond codes have TX = -2, "
        "odd +2 (left or right on the screen is not stated in the deposit, so no left/right label is given). Trial durations vary; the traces "
        "continue for 1 s from the start of the recording whatever the movement end (events gives movement_end).",
    ],
    differences=["The deposit holds only the channel trials; the 520 non-perturbed trials per participant are not in it.",
                 "Peak velocities and movement times of the paper (58.9 and 58.1 cm/s, 700 and 630 ms) are not reproduced from the stored channel "
                 "trials alone (they are means over all trials) and are not used as checks."],
)
stdlib.release(ID, "timeseries", rows, meta, checks, extra_files=files)
