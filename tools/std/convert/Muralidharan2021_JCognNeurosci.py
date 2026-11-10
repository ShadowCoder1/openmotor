"""Muralidharan & Aron (2021), J Cogn Neurosci: behaviour (reaction times, movement onset/end, EMG event times) of two
experiments in which a first movement is followed on 20% of trials by a cued second movement with either hand.

Run from the site folder: python3 -I tools/std/convert/Muralidharan2021_JCognNeurosci.py
Raw files (untrusted data, read only), from https://osf.io/qwb29/ (members of the split archives Expt1_Data_scripts_split and
Expt2_Data_scripts_split, extracted by HTTP range requests; the ~25 GB of EEG, ERSP and burst files were not downloaded):
  E1_s*__s*_behav_block14_EMG_times.mat (s1: block10)  MATLAB table trial_structure of Experiment 1, one per participant
  E2_s*__s*_behav_block_EEG.mat / _baseline_block_EEG.mat  MATLAB table trial_structure of the Main / Control task of Experiment 2
  E2_n_15_press_and_move_behavior.mat                 the authors' per-participant means (internal check)
The tables are MATLAB 'table' objects that scipy cannot load directly; read_table() below parses the object's stored columns
(variable names and data) from the file's function-workspace. Nothing in the files is executed.
"""
import glob
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Muralidharan2021_JCognNeurosci"
D = stdlib.raw_dir(ID)
ck = stdlib.Checks()


def read_table(f):
    m = sio.loadmat(str(f))
    ws = m["__function_workspace__"].tobytes()
    r = MatFile5Reader(io.BytesIO(ws[8:]), byte_order="<", struct_as_record=False, squeeze_me=False)
    r.mat_stream.seek(0)
    r.initialize_read()
    hdr, _ = r.read_var_header()
    arr = r.read_var_array(hdr, process=False)
    cell = arr[0, 0].MCOS[0][2]
    names = [re.sub(r"[\[\]' ]", "", str(x)) for x in cell[4][0][0].ravel()]
    cols = cell[5][0][0].ravel()
    return pd.DataFrame({n: np.asarray(c, dtype=float).ravel() for n, c in zip(names, cols)})


def sid(f):
    return int(re.search(r"_s(\d+)_", os.path.basename(f)).group(1))


def prctile(x, p):
    """MATLAB prctile (NaN ignored): linear interpolation between order statistics at p*n/100 + 0.5."""
    xs = np.sort(x[~np.isnan(x)])
    n = len(xs)
    return float(np.interp(p * n / 100 + 0.5, np.arange(1, n + 1), xs))


def ms(v):
    return np.where(np.isnan(v), np.nan, np.asarray(v, dtype=float) * 1000.0)


E1_DEV = "wrist flexion of the right hand, then thumb press on one of two hand-held buttons (surface EMG of forearm and thumb muscles; EEG)"
E2_DEV = "two joysticks held in both hands (trigger buttons and 2-D axis), screen cues (EEG recorded)"
frames = []


def base(n, subj, exp, device, block, trial, cond):
    return pd.DataFrame({
        "dataset": ID, "study": ID, "experiment": exp, "group": "", "subject": subj, "subject_excluded_by_authors": False,
        "age": "", "sex": "", "handedness": "right", "population": "healthy", "setting": "lab", "device": device,
        "session": 1, "block": [f"block {int(b)}" for b in block], "trial": trial, "phase": "other", "time_s": "",
        "task": "other", "unit": "trial", "condition": cond, "pert_type": "none", "pert_value": 0, "pert_unit": "",
        "feedback": "normal"}, index=range(n))


hand = {0: "none", 1: "right", 2: "left"}
# ---------------------------------------------------------------- Experiment 1
e1 = sorted(glob.glob(str(D / "E1_*EMG_times.mat")), key=sid)
e1_stats = []
for f in e1:
    s = sid(f)
    t = read_table(f)
    n = len(t)
    rt = t.RT.to_numpy()
    q1, q3 = prctile(rt, 25), prctile(rt, 75)
    iqr = q3 - q1
    outlier = (rt < q1 - 1.5 * iqr) | (rt > q3 + 1.5 * iqr)
    b = base(n, f"E1_s{s}", "Experiment 1: wrist flexion then cued thumb press", E1_DEV, t.block_no, np.arange(1, n + 1),
             "flex and move")
    mv = t.movement.astype(int).to_numpy()
    b["excluded_by_authors"] = outlier
    b["block_number"] = t.block_no.astype(int).to_numpy()
    b["trial_kind"] = np.where(mv > 0, "primary and secondary", "primary only")
    b["secondary_hand"] = [hand[x] for x in mv]
    b["secondary_outcome_code"] = np.where(mv > 0, t.correct, np.nan)
    b["secondary_rt_ms"] = ms(rt)
    b["primary_done"] = t.ifflexed.notna().to_numpy()
    for k, c in [("on", "ON"), ("peak", "PK"), ("dip", "DP"), ("off", "OF")]:
        b[f"emg_secondary_{k}_ms"] = ms(t[f"emg{c}_Move"].to_numpy())
        b[f"emg_primary_{k}_ms"] = ms(t[f"emg{c}_Flex"].to_numpy())
    frames.append(b)
    ok = (t.trial_type == 1).to_numpy() & (t.correct == 1).to_numpy() & t.ifflexed.notna().to_numpy()
    r_ = np.where(outlier, np.nan, rt)
    eon = np.where(outlier, np.nan, t.emgON_Move.to_numpy())
    e1_stats.append(dict(n=n, sec=int((mv > 0).sum()), pct=100 * outlier.sum() / (60 if n == 300 else 84),
                         L=np.nanmean(r_[ok & (mv == 2)]), R=np.nanmean(r_[ok & (mv == 1)]),
                         eL=np.nanmean(eon[ok & (mv == 2)]), eR=np.nanmean(eon[ok & (mv == 1)]),
                         blocks=int(t.block_no.max())))
st = pd.DataFrame(e1_stats)

# ---------------------------------------------------------------- Experiment 2
e2 = sorted(glob.glob(str(D / "E2_*__*behav_block_EEG.mat")), key=sid)
e2_stats = []
for f in e2:
    s = sid(f)
    parts = []
    for fn, kind, cond, off in [(f, "main", "Press and Move (main task)", 0),
                                (f.replace("behav_block_EEG", "behav_baseline_block_EEG"), "control", "Ready and Move (control task)", None)]:
        t = read_table(fn)
        n = len(t)
        mv = t.movement.astype(int).to_numpy()
        b = base(n, f"E2_s{s}", "Experiment 2: press then cued joystick movement", E2_DEV, t.block_no, 0, cond)
        b["block_number"] = t.block_no.astype(int).to_numpy()
        if kind == "main":
            b["trial_kind"] = np.where(mv > 0, "primary and secondary", "primary only")
        else:
            b["trial_kind"] = "secondary only"
        b["secondary_hand"] = [hand[x] for x in mv]
        b["secondary_outcome_code"] = t.correct.to_numpy()
        no, wo = t.RT_movt_end_NO.to_numpy(), t.RT_movt_end_WO.to_numpy()
        b["excluded_by_authors"] = np.isnan(no) & ~np.isnan(wo)
        b["secondary_rt_ms"] = ms(no)
        b["secondary_onset_ms"] = ms(t.RT_movt_onset_NO.to_numpy())
        b["secondary_peak_velocity"] = t.peak_velocity_NO.to_numpy()
        b["primary_done"] = (t.ifpressed.to_numpy() == 1) if kind == "main" else ""
        b["primary_rt_ms"] = ms(t.RTPress.to_numpy())
        b["_kind"] = kind
        b["_rtfull_ok"] = (~np.isnan(wo)).astype(int)
        parts.append(b)
        e2_stats.append(dict(s=s, kind=kind, n=n, sec=int((mv > 0).sum()), nR=int((mv == 1).sum()), nL=int((mv == 2).sum()),
                             end=[np.nanmean(no[(mv == m) & (t.correct == 1).to_numpy()]) for m in (2, 1)],
                             on=[np.nanmean(t.RT_movt_onset_NO.to_numpy()[(mv == m) & (t.correct == 1).to_numpy()]) for m in (2, 1)]))
    p = pd.concat(parts, ignore_index=True)
    p["trial"] = np.arange(1, len(p) + 1)
    frames.append(p.drop(columns=["_kind", "_rtfull_ok"]))

t = pd.concat(frames, ignore_index=True)
t["excluded_by_authors"] = t.excluded_by_authors.astype(bool)

# ---------------------------------------------------------------- checks
S = "Experiment 1 Methods/Results"
ck.published("Experiment 1: participants", len(st), 11, 0, "Experiment 1: Participants (11)")
ck.published("Experiment 1: trials per participant, 14 blocks x 30 (all but one participant)", (st.n == 420).sum(), 10, 0, "Experiment 1: Flex and Move Task (14 blocks of 30 trials, one participant 10 blocks)")
ck.published("Experiment 1: the one participant with 10 blocks has 300 trials", (st.n == 300).sum(), 1, 0, "Experiment 1: Flex and Move Task")
ck.published("Experiment 1: secondary-movement trials (20%) per full participant", st.sec[st.n == 420].iloc[0], 84, 0, "Experiment 1: 20% of 420 trials")
ck.published("Experiment 1: RT removed by the IQR rule, % of secondary trials (mean)", st.pct.mean(), 4.1, 0.25, "Experiment 1: Data Analysis (4.1 +/- 0.8%)")
ck.published("Experiment 1: mean button-press RT, right hand (ms)", st.R.mean() * 1000, 588, 1.5, "Results, Experiment 1 (588 +/- 41 ms)")
ck.published("Experiment 1: mean button-press RT, left hand (ms)", st.L.mean() * 1000, 569, 1.5, "Results, Experiment 1 (569 +/- 39 ms)")
d = (st.R - st.L) * 1000
ck.published("Experiment 1: RT difference right - left (ms)", d.mean(), 19, 1.5, "Results, Experiment 1 (19 +/- 6 ms)")
ck.published("Experiment 1: SEM of the RT difference (ms)", d.std(ddof=1) / np.sqrt(len(d)), 6, 0.6, "Results, Experiment 1 (+/- 6 ms)")
from scipy import stats  # noqa: E402
ck.published("Experiment 1: paired t of right vs left RT", stats.ttest_1samp(d, 0).statistic, 3.02, 0.1, "Results, Experiment 1 (t[10] = 3.02)")
ck.published("Experiment 1: EMG onset of the secondary press, right hand (ms)", st.eR.mean() * 1000, 441, 1.5, "Results, Experiment 1 (right = 441 ms)")
ck.published("Experiment 1: EMG onset of the secondary press, left hand (ms)", st.eL.mean() * 1000, 441, 1.5, "Results, Experiment 1 (left = 441 ms)")
e2s = pd.DataFrame(e2_stats)
m2 = e2s[e2s.kind == "main"]
c2 = e2s[e2s.kind == "control"]
ck.published("Experiment 2: participants", m2.s.nunique(), 15, 0, "Experiment 2: Participants (15)")
ck.published("Experiment 2: Main task trials per participant (20 blocks x 30)", m2.n.iloc[0], 600, 0, "Experiment 2: Press/Ready and Move Tasks (600)")
ck.internal("Experiment 2: every participant has 600 Main trials", bool((m2.n == 600).all()))
ck.published("Experiment 2: Main task secondary movements per participant", m2.sec.iloc[0], 120, 0, "Experiment 2 (120 trials, 60 Right and 60 Left)")
ck.published("Experiment 2: Main task right-hand secondary movements", m2.nR.iloc[0], 60, 0, "Experiment 2 (60 Right)")
ck.published("Experiment 2: Control task trials per participant (4 blocks x 24)", c2.n.iloc[0], 96, 0, "Experiment 2 (total of 96 trials)")
ck.published("Experiment 2: Control task right-hand movements", c2.nR.iloc[0], 48, 0, "Experiment 2 (48 Right and 48 Left)")
on = np.array(m2.on.tolist()) * 1000
en = np.array(m2.end.tolist()) * 1000
ck.published("Experiment 2 Main: movement onset, right hand (ms)", on[:, 1].mean(), 446, 1.5, "Results, Experiment 2 (Onset 446 +/- 16 ms)")
ck.published("Experiment 2 Main: movement onset, left hand (ms)", on[:, 0].mean(), 410, 1.5, "Results, Experiment 2 (Onset 410 +/- 12 ms)")
ck.published("Experiment 2 Main: movement end, right hand (ms)", en[:, 1].mean(), 539, 1.5, "Results, Experiment 2 (End 539 +/- 19 ms)")
ck.published("Experiment 2 Main: movement end, left hand (ms)", en[:, 0].mean(), 513, 1.5, "Results, Experiment 2 (End 513 +/- 16 ms)")
dd = on[:, 1] - on[:, 0]
ck.published("Experiment 2 Main: onset difference right - left (ms)", dd.mean(), 36, 1.5, "Results, Experiment 2 (Main Right-Left 36 +/- 8 ms)")
con = np.array(c2.on.tolist()) * 1000
ck.published("Experiment 2 Control: onset difference right - left (ms)", (con[:, 1] - con[:, 0]).mean(), 16, 1.5, "Results, Experiment 2 (Control Right-Left 16 +/- 6 ms)")
g = sio.loadmat(str(D / "E2_n_15_press_and_move_behavior.mat"))["RT_movt_end_allsubs"]
ck.internal("per-participant Main-task movement-end means match, within 1 ms, the authors' n_15_press_and_move_behavior file (left, right)",
            bool(np.allclose(g, en / 1000, atol=1e-3)), f"max diff {np.abs(g - en / 1000).max():.2e}")
ck.internal("Experiment 1 and 2 rows kept: 10 x 420 + 300 + 15 x 696", len(t) == 10 * 420 + 300 + 15 * 696, str(len(t)))
for ch in ck.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: {ch['computed']:.4f} vs {ch['reported']} ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED", ch["what"], ch.get("detail", ""))

names = [c[1] for c in stdlib.columns_for("cued_second_movement")]
for c in names:
    if c not in t.columns and c != "person":
        t[c] = np.nan if c.endswith(("_ms", "_velocity")) else ""
t["person"] = t["subject"]
t = t[[c for c in names]]
meta = dict(
    paper=dict(authors="Muralidharan V, Aron AR", year=2021,
               title="Behavioral Induction of a High Beta State in Sensorimotor Cortex Leads to Movement Slowing",
               venue="Journal of Cognitive Neuroscience 33(7):1311-1328", doi="10.1162/jocn_a_01717"),
    lab="Aron (UC San Diego)", data_url="https://osf.io/qwb29/", license="none stated", license_url="",
    changes=("Only the behavioural tables (trial_structure) of each participant were converted; the EEG, time-frequency and burst files "
             "(about 25 GB) and the joystick trajectory files were not. Times in seconds became milliseconds; block labels and "
             "participant codes were renamed; the constant columns response and mean_EMG (0 on every row) were dropped, and the "
             "three duplicate 'with outliers' (WO) Experiment 2 columns were used only to mark the authors' outlier removal."),
    how=["One row is one trial. Experiment 1 (11 participants): every trial the participant made a brisk right wrist flexion at a "
         "white square; on 20% of trials an arrow 2 s after the square asked for a thumb press with the left or right hand. Experiment 2 "
         "(15 other participants): a 'Press' cue asked for a right trigger press, and on 20% of trials a left or right joystick "
         "enlarged about 1 s later, asking for a center-out joystick movement; a Control task (Ready and Move, 96 trials, listed after the "
         "Main task) cued the joystick movement on every trial. secondary_rt_ms is the authors' RT (thumb press; joystick maximum deflection).",
         "excluded_by_authors marks secondary-movement trials whose RT the authors' outlier rule removes: in Experiment 1 RTs outside "
         "25th percentile - 1.5 IQR and 75th percentile + 1.5 IQR of the participant's RTs (recomputed here with MATLAB's percentile "
         "definition, as in analyse_BEH.m); in Experiment 2 the trials whose value is missing in the authors' outlier-removed columns "
         "(_NO) but present in the with-outliers columns (_WO). The Experiment 2 onset and peak-velocity columns are the outlier-removed "
         "values; Experiment 1 EMG times are the authors' values for all trials.",
         "The .mat tables are MATLAB table objects; the variable names and values were read from the file without MATLAB. "
         "Experiment 1 time origins of the EMG events and the meaning of correct code 3 (one participant, three trials) are not stated in the deposit."],
    unit_note="trial",
    differences=["Per-participant movement-end means recomputed from the rows differ from the means in the authors' n_15_press_and_move_behavior file by up to 0.64 ms (the deposit does not say which trials its means use); group means and differences match the paper to within about 1 ms.",
                 "Experiment 1 outlier removal is recomputed here from the authors' rule in analyse_BEH.m; the deposit stores no outlier flags for Experiment 1."],
)
stdlib.release(ID, "cued_second_movement", t.to_dict("records"), meta, ck)
