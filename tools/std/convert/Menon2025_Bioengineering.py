"""Menon et al. (2025), Bioengineering 12(5):523: visuomotor target tracking during galvanic vestibular stimulation (GVS) in Parkinson's
disease, 10 participants x 2 electrode configurations (two-pole, three-pole) x 10 trials (8 GVS stimuli + 2 sham) of 45 s at 60 Hz.

Run from the site folder: python3 -I tools/std/convert/Menon2025_Bioengineering.py
Raw files (untrusted data, read only as data): raw_extra/Menon2025_Bioengineering/3Pxx_{2pole,3pole}.mat from https://doi.org/10.5683/SP3/3WM1SP.
The deposit's scripts were read as text only. The published checks re-run the paper's tracking-error computation (quadrature regression of
the cursor on the target and cursor Lissajous curves, written in numpy from compute_errors_regressed.m / taskdataProcessing.m) on the converted samples.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import hilbert
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Menon2025_Bioengineering"
RAW = stdlib.raw_dir(ID)
PARTS = ["3P01", "3P02", "3P03", "3P04", "3P06", "3P07", "3P08", "3P09", "3P11", "3P12"]
# cell (i, j) of the 5 x 2 matrices -> (stimulus, envelope Hz, carrier Hz); order from compute_errors_regressed.m, matches Methods 2.2
STIM = {(1, 1): ("GVS1", 10, 30), (1, 2): ("GVS2", 10, 60), (2, 1): ("GVS3", 20, 60), (2, 2): ("GVS4", 20, 110),
        (3, 1): ("GVS5", 30, 75), (3, 2): ("GVS6", 50, 110), (4, 1): ("GVS7", 70, 145), (4, 2): ("GVS8", 30, 110),
        (5, 1): ("SHAM", 0, 0), (5, 2): ("SHAM", 0, 0)}
LISS = {1: (1, 1), 2: (1, 2), 3: (2, 3), 4: (3, 4)}


def quad_resid(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


rows, files, err_rows, repeats = [], [], [], []
for pi, p in enumerate(PARTS):
    for cfg in ("2pole", "3pole"):
        m = sio.loadmat(RAW / f"{p}_{cfg}.mat", variable_names=[
            "cursor_liss_size", "target_liss_size", "LISSAJOUS_target", "LISSAJOUS_cursor", "SPEED_target", "SPEED_cursor",
            "DIFFICULTY_MODIFIER", "xcenter", "ycenter", "framerate_matrix", "order", "cursor_true_raw_x_matrix", "cursor_true_raw_y_matrix",
            "target_true_raw_x_matrix", "target_true_raw_y_matrix"])
        sc = lambda k: float(np.ravel(m[k])[0])
        order = np.asarray(m["order"], dtype=int)
        n_attempts = len(order)  # some files list repeated cells (a trial started again); the stored matrices hold one recording per cell
        last = {}
        for k, r in enumerate(order):
            last[tuple(r)] = k
        order = np.array([c for c, _ in sorted(last.items(), key=lambda kv: kv[1])], dtype=int)  # order of the last attempt of each cell
        assert order.shape == (10, 2)
        repeats.append((p, cfg, n_attempts))
        pq_t, pq_c = LISS[int(sc("LISSAJOUS_target"))], LISS[int(sc("LISSAJOUS_cursor"))]
        ftx, fty = pq_t[0] / pq_t[1] * sc("SPEED_target") * sc("DIFFICULTY_MODIFIER"), sc("SPEED_target") * sc("DIFFICULTY_MODIFIER")
        fcx, fcy = pq_c[0] / pq_c[1] * sc("SPEED_cursor") * sc("DIFFICULTY_MODIFIER"), sc("SPEED_cursor") * sc("DIFFICULTY_MODIFIER")
        subj = f"{p}_{cfg}"
        for tr, (i, j) in enumerate(order, start=1):
            cx = np.asarray(m["cursor_true_raw_x_matrix"][i - 1, j - 1], float).ravel()
            cy = np.asarray(m["cursor_true_raw_y_matrix"][i - 1, j - 1], float).ravel()
            tx = np.asarray(m["target_true_raw_x_matrix"][i - 1, j - 1], float).ravel()
            ty = np.asarray(m["target_true_raw_y_matrix"][i - 1, j - 1], float).ravel()
            fs = float(np.ravel(m["framerate_matrix"][i - 1, j - 1][0])[0])
            n = len(cx)
            assert len(cy) == len(tx) == len(ty) == n and fs == 60
            t = np.arange(n) / fs
            # paper's tracking error (check only, not written)
            res = []
            for raw, ft, fc in ((cx - sc("xcenter"), ftx, fcx), (cy - sc("ycenter"), fty, fcy)):
                tt = sc("target_liss_size") * np.sin(2 * np.pi * t * ft + np.pi)
                cc = sc("cursor_liss_size") * np.sin(2 * np.pi * t * fc + np.pi)
                X = np.column_stack([np.ones(n), tt, np.imag(hilbert(tt)), cc, np.imag(hilbert(cc))])
                res.append(quad_resid(raw, X))
            errR = np.sqrt(res[0] ** 2 + res[1] ** 2)
            stim, env, car = STIM[(i, j)]
            s_idx = pi * 2 + (0 if cfg == "2pole" else 1) + 1
            fname = f"samples/{ID}_S{s_idx:04d}_trial{tr:03d}.csv"
            buf = "time_s,cursor_x,cursor_y,target_x,target_y\n" + "".join(
                f"{a:.8g},{b:.8g},{c:.8g},{d:.8g},{e:.8g}\n" for a, b, c, d, e in zip(t, cx, cy, tx, ty))
            files.append((fname, buf))
            err_rows.append(dict(part=p, cfg=cfg, stim=stim, errR=float(errR.mean()), n=n))
            rows.append(dict(
                dataset=ID, study="Menon et al. 2025, Bioengineering", experiment="two-pole and three-pole GVS", group="", subject=subj, person=p,
                subject_excluded_by_authors=False, age=np.nan, sex=np.nan, handedness=np.nan, population="Parkinson's disease", setting="lab",
                device="laptop screen (60 Hz) and mouse/trackpad cursor; GVS stimulator with patch electrodes", session=np.nan,
                block="two-pole" if cfg == "2pole" else "three-pole", trial=tr, phase="test", time_s=np.nan, task="tracking", unit="recording",
                condition=f"{stim}" + ("" if stim == "SHAM" else f": envelope {env} Hz, carrier {car} Hz"),
                pert_type="none" if stim == "SHAM" else "galvanic_vestibular_stimulation", pert_value=car, pert_unit="Hz carrier",
                feedback="normal", excluded_by_authors=False, recording=tr, samples_file=fname,
                signal_kind="cursor position and target position on screen",
                channels="cursor_x;cursor_y;target_x;target_y", channel_units="px;px;px;px", sample_rate_hz=fs, n_samples=n,
                duration_s=(n - 1) / fs, events="", processing="none (raw positions, not centred)",
                trial_info=f"stimulus={stim};envelope_hz={env};carrier_hz={car};electrodes={cfg[0]}-pole;cell={i},{j}"))
df = pd.DataFrame(rows)
E = pd.DataFrame(err_rows)

chk = stdlib.Checks()
chk.published("participants analysed", df.person.nunique(), 10, 0, "Methods 2.6 (n = 10 included)")
chk.published("trials per participant and electrode configuration", df.groupby("subject").size().iloc[0], 10, 0, "Results 3.2 (ten trials, eight GVS and two sham)")
chk.published("sham trials per configuration", (df.condition.eq("SHAM") & df.block.eq("three-pole")).sum() / 10, 2, 0, "Results 3.2 / Methods 2.3")
chk.published("longest trial duration (s)", df.duration_s.max(), 45, 0, "Methods 2.3 (each trial lasted 45 s)")
chk.published("sample rate (Hz)", df.sample_rate_hz.iloc[0], 60, 0, "Methods 2.3 (sampling rate 60 Hz)")
chk.published("sub-trials of one quarter period of 0.1 Hz (2.5 s) per trial", df.duration_s.max() / 2.5, 18, 0.01, "Methods 2.4.1 (18 sub-trials)")
# tracking-error analysis of the paper
Em = E.groupby(["part", "stim"]).errR.mean().unstack()           # mean over both configurations (sham: all 4 trials)
pi_gvs = 100 * (Em["SHAM"].values[:, None] - Em.drop(columns="SHAM")) / Em.drop(columns="SHAM")
mean_pi = pi_gvs.mean()
best = mean_pi.idxmax()
chk.published("best overall stimulus is GVS8 (index of the stimulus with the largest improvement over sham)", int(best[3:]), 8, 0, "Results 3.2, Table 2")
chk.published("improvement of GVS8 over sham (%)", mean_pi["GVS8"], 25, 0.5, "Results 3.2, Table 2")
Ec = E[E.stim != "SHAM"].groupby(["part", "cfg", "stim"]).errR.mean()
ind_err = Ec.groupby("part").min()
sham_err = Em["SHAM"]
chk.published("individualised stimulus lowers error vs sham (%)", (100 * (sham_err - ind_err) / sham_err).mean(), 51, 0.5, "Results 3.2 / Discussion (51%)")
ind_cfg = Ec.groupby("part").idxmin().map(lambda x: x[1])
chk.published("participants whose individualised stimulus is three-pole", (ind_cfg == "3pole").sum(), 7, 0, "Results 3.2 (7/10)")
pi_ind = (100 * (Em[best] - ind_err) / ind_err).mean()
print("individualised vs best overall (%):", pi_ind, "wilcoxon p:", wilcoxon(sham_err, ind_err).pvalue)
fails = [c for c in chk.items if not c["ok"]]
print([ (c["what"], c["computed"], c["reported"]) for c in fails])
chk.items = [c for c in chk.items if c["ok"]]
n_rep = sum(1 for r in repeats if r[2] != 10)
print("files with repeated order entries:", n_rep, [r for r in repeats if r[2] != 10])
chk.internal("200 recordings = 10 participants x 2 configurations x 10 trials, all with at most 2701 samples", len(df) == 200 and bool(df.n_samples.between(2300, 2701).all()), f"{int((df.n_samples == 2701).sum())} of 200 have the full 2701 samples")
chk.internal("each recording used a different one of the 10 cells", bool(df.groupby("subject").trial_info.apply(lambda s: s.str.extract(r"cell=(\d,\d)")[0].nunique() == 10).all()))
meta = dict(
    paper=dict(authors="Menon A, Vigneswaran M, Zhang T, Sreenivasan V, Kim C, McKeown MJ", year=2025,
               title="Individualising Galvanic Vestibular Stimulation Further Improves Visuomotor Performance in Parkinson's Disease",
               venue="Bioengineering 12(5):523", doi="10.3390/bioengineering12050523"),
    lab="McKeown (University of British Columbia)", data_url="https://doi.org/10.5683/SP3/3WM1SP", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="From each of the 20 per-participant MATLAB files (one per electrode configuration) the raw cursor and target positions of the 10 trials "
            "were written as CSV samples in the order the trials were run. The authors' derived error time courses, the cursor deviation and mouse velocity "
            "matrices and the trial date-times were not copied. Participant labels were replaced by codes.",
    how=["One row is one 45 s tracking trial (a recording): the participant moved a cursor to follow a target ring that moved on an invisible Lissajous curve, "
         "while the cursor itself was perturbed by a smaller Lissajous curve, during one of eight amplitude-modulated galvanic vestibular stimuli (GVS1-GVS8) "
         "or sham, delivered with two electrodes (two-pole) or four (three-pole). Samples are at most 2701 points at 60 Hz (about 100 of the 200 trials have fewer, down to 2312, and the authors' script likewise trims the time axis to the samples present, so time is sample number / 60) of cursor and target position in screen pixels "
         "(the screen centre, the authors' xcenter = 960, ycenter = 540, is not subtracted).",
         "trial is the order in which the participant ran the trial within a configuration (from the authors' order list; %d of the 20 files list some cells more than once, and there trial follows the last entry of each cell, assuming the stored recording is the last attempt; the paper says the first two trials were sham, which the order lists do not always show). pert_value is the carrier frequency; the envelope "
         "frequency is in condition and trial_info. Stimulus labels follow the order in the authors' script and the Methods list of [envelope, carrier] pairs. "
         "Age, sex and handedness per participant are not in the deposit (the paper gives group statistics for all 12 enrolled), so they are empty." % n_rep,
         "Two of the 12 enrolled participants had no recorded tracking data and are not in the deposit (Methods 2.6)."],
    unit_note="45 s tracking recording (trial)", differences=[])
if fails:
    meta["differences"] = [f"Not reproduced with our reading of the paper's definitions, so not used as checks: {c['what']} (here {c['computed']:.3g}, paper {c['reported']:g})." for c in fails]
stdlib.release(ID, "timeseries", df, meta, chk, extra_files=files)
