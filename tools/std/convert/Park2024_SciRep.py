"""Park, Kim & Kim (2024), Sci Rep 14:3731: continuous de novo motor learning with a data glove (fingers control a cursor), one row per trial.

Run from the site folder: python3 -I tools/std/convert/Park2024_SciRep.py
Raw files (untrusted data, read only) in raw_extra/Park2024_SciRep/, from https://github.com/psb629/GloveProject_GL (behav_data/):
  GL<NN>-fmri.mat   main session of each participant: allXY (cursor, 60 Hz, 240 frames per trial), allData (14 glove sensors), fb,
                    targetID, cnt_hit, LearnTrialStartTime (per trial). The calibration (-Cali, -preCali, -PreTrain) files and the
                    GL_beta folder are not used. Variable meanings come from the authors' scripts/genRegressors.m (read as text, never run)
                    and the paper (Methods, Main session); the deposit has no data dictionary.
"""
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Park2024_SciRep"
RAW = stdlib.raw_dir(ID)
OUTD = RAW / "_samples"
OUTD.mkdir(exist_ok=True)
checks = stdlib.Checks()

NS, FS, LEAD, K0, LAG = 240, 60.0, 12, 34540, 20
SENS = ["thumb_near", "thumb_far", "thumb_index", "index_near", "index_far", "index_middle", "middle_near", "middle_far",
        "middle_ring", "ring_near", "ring_far", "ring_little", "little_near", "little_far"]
CHN = ["cursor_x", "cursor_y"] + [f"glove_{s}" for s in SENS]
SEQ = {1: "13-3-25-21-13-25-3-21-25-13-21-3", 2: "13-23-5-1-13-5-23-1-5-13-1-23"}

files = sorted(glob.glob(str(RAW / "GL*-fmri.mat")))
rows, extra = [], []
perf = {}
full = []
for fpath in files:
    code = os.path.basename(fpath)[:4]
    m = sio.loadmat(fpath, struct_as_record=False, squeeze_me=True)
    xy, ad = m["allXY"], m["allData"].astype(np.float64)
    fb, tid, ch, st = (np.asarray(m[k]) for k in ("fb", "targetID", "cnt_hit", "LearnTrialStartTime"))
    n = xy.shape[1] // NS
    assert xy.shape[1] == n * NS and n % 145 == 0 and len(ch) == len(st) == LEAD + n, (code, xy.shape, len(ch))
    glove = ad[:, K0:]
    assert glove.shape[1] >= n * NS + LAG, (code, glove.shape)
    runs = n // 145
    complete = runs == 6
    full.append(complete)
    seqno = 1 if list(m["seq1"][:2]) == [13, 3] else 2
    assert tuple(int(x) for x in m["seq1"]) and (int(m["seq1"][1]) == 3 or int(m["seq1"][1]) == 23)
    perf[code] = np.full(870, np.nan)
    for t in range(n):
        r, i = divmod(t, 145)             # run (0-based) and trial in run (0 = lead-in trial)
        block = 12 * r + (i - 1) // 12 + 1 if i > 0 else 0
        k = int(tid[LEAD + t])
        row, col = (k - 1) // 5 + 1, (k - 1) % 5 + 1
        tx, ty = (col - 3) * 80, (3 - row) * 80
        # hit check (the authors' cnt_hit): cursor within +-40 px (half a cell) of the target centre
        c = xy[:, NS * t:NS * (t + 1)]
        hits = int(np.sum((np.abs(c[0] - tx) <= 40) & (np.abs(c[1] - ty) <= 40)))
        assert hits == int(ch[LEAD + t]), (code, t, hits, int(ch[LEAD + t]))
        g = glove[:, NS * t + LAG:NS * (t + 1) + LAG]
        df = pd.DataFrame(np.c_[np.arange(NS) / FS, c.T, g.T], columns=["time_s"] + CHN)
        fn = f"samples/{code}_run{r + 1}_trial{i:03d}.csv"
        p = OUTD / f"{code}_run{r + 1}_trial{i:03d}.csv"
        df.to_csv(p, index=False, float_format="%.8g", lineterminator="\r\n")
        extra.append((fn, str(p)))
        cursor_on = int(fb[LEAD + t]) == 1
        if i > 0:
            perf[code][t] = hits / NS
        rows.append(dict(
            dataset=ID, study=ID, experiment=ID, group=("sequence 1 (triangle)" if seqno == 1 else "sequence 2 (inverted triangle)"),
            subject=code, person=code, subject_excluded_by_authors=not complete, age="", sex="", handedness="right", population="healthy",
            setting="lab", device="5DT Data Glove 14 Ultra (MR-compatible, right hand) controlling a cursor on a 5 x 5 grid, in a 3 T MRI scanner",
            session=1, block=(f"run {r + 1}, lead-in trial" if i == 0 else f"block {block}"), trial=t + 1,
            phase=("learning" if r < 4 else "transfer"), time_s="", task="finger movements", unit="trial",
            condition="cursor visible" if cursor_on else "cursor hidden", pert_type="none", pert_value=0, pert_unit="",
            feedback="normal" if cursor_on else "masked", excluded_by_authors=(i == 0),
            recording=t + 1, samples_file=fn, signal_kind="cursor position and 14 data-glove sensor values (finger-to-cursor control)",
            channels=";".join(CHN), channel_units="px;px;" + ";".join(["normalised 0-1"] * 14), sample_rate_hz=FS, n_samples=NS,
            duration_s=NS / FS, events="",
            trial_info=f"run={r + 1};trial_in_run={i};block={block};target_cell={k};target_x_px={tx};target_y_px={ty};"
                       f"frames_on_target={hits};proportion_on_target={hits / NS:.6g};start_in_run_s={st[LEAD + t] / 1000:.3f}",
            processing="cursor as shown on screen (computed live by the authors from a weighted window of 20 glove samples); glove values are the sensors "
                       "normalised by the authors' calibration; not filtered"))
df = pd.DataFrame(rows)

# ---- published numbers (Results, Behavioral performance) from the success rate (proportion of frames on target) of the 24 complete participants
comp = [c for c, f in zip([os.path.basename(x)[:4] for x in files], full) if f]
checks.published("participants who completed all sessions", len(comp), 24, 0, "Methods, Participants")
checks.published("participants in the deposit (26 volunteers, 2 incomplete)", len(files), 26, 0, "Methods, Participants")
P = np.array([perf[c] for c in comp])                      # 24 x 870, NaN at lead-in trials
Pr = P.reshape(len(comp), 6, 145)[:, :, 1:].reshape(len(comp), 6 * 12, 12).mean(axis=2)   # block means: 72 blocks
bk = lambda a: np.array(a) - 1  # noqa: E731
on_odd = bk(range(1, 49, 2)); off_even = bk(range(2, 49, 2))
early_on = Pr[:, [b for b in on_odd if b < 24]].mean(1); late_on = Pr[:, [b for b in on_odd if b >= 24]].mean(1)
early_off = Pr[:, [b for b in off_even if b < 24]].mean(1); late_off = Pr[:, [b for b in off_even if b >= 24]].mean(1)


def t_paired(a, b=None):
    d = a if b is None else a - b
    return d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))


checks.published("cursor-off blocks, late vs early, paired t (blocks 2,4..24 vs 26,..48)", t_paired(late_off, early_off), 4.37, 0.05, "Results, Behavioral performance")
F_stage = t_paired((late_on + late_off) / 2, (early_on + early_off) / 2) ** 2
F_fb = t_paired((early_on + late_on) / 2, (early_off + late_off) / 2) ** 2
F_int = t_paired(late_on - early_on, late_off - early_off) ** 2
checks.published("two-way RM ANOVA, learning stage F(1,23)", F_stage, 70.98, 0.5, "Results, Behavioral performance")
checks.published("two-way RM ANOVA, feedback condition F(1,23)", F_fb, 227.1, 1.0, "Results, Behavioral performance")
checks.published("two-way RM ANOVA, interaction F(1,23)", F_int, 101.8, 0.5, "Results, Behavioral performance")
checks.internal("every participant of the 24 has 870 trials (6 runs x 145: a lead-in trial and 144 trials)", all(len(perf[c]) == 870 and not np.isnan(perf[c][1]) for c in comp))
checks.internal("hits recomputed from the cursor, the target cell (x = (column-3)*80 px, y = (3-row)*80 px) and a +-40 px box equal the authors' cnt_hit for every trial", True)
checks.internal("two incomplete participants (4 runs and 3 runs of the 6) are kept and marked excluded", int((~np.array(full)).sum()) == 2)

meta = dict(
    paper=dict(authors="Park S, Kim J, Kim S", year=2024, title="Corticostriatal activity related to performance during continuous de novo motor learning",
               venue="Sci Rep 14(1):3731", doi="10.1038/s41598-024-54176-9"),
    lab="Kim (Sungkyunkwan University)", data_url="https://github.com/psb629/GloveProject_GL",
    license="none stated", license_url="",
    changes=("Converted from MATLAB to one CSV of samples per trial plus the OpenMotor table. Only the main-session files (behav_data/GL<NN>-fmri.mat) are used; "
             "the calibration and pre-training files, the fMRI regressor betas, figures and results maps are not included. The 12 pre-session trials "
             "that have no cursor record, and the glove samples outside the task trials, are left out."),
    how=["One row is one 4-s trial of a de novo finger-to-cursor task in the MRI scanner: wearing a data glove on the right hand, the participant moved the "
         "cursor on a 5 x 5 grid to a target cell (a mapping learned from a calibration session) and kept it there; the target turned red while the cursor "
         "was on it. Each trial's samples (samples/<participant>_run<r>_trial<NNN>.csv, 60 Hz, 240 frames) hold time_s, the cursor position cursor_x and "
         "cursor_y in pixels (origin at the centre cell of the grid, x to the right, y up; one grid cell is 80 px, derived by checking that the authors' "
         "hit counts are reproduced exactly) and the 14 glove sensor values (0-1, thumb near and far, thumb-index, index near and far, index-middle, "
         "middle near and far, middle-ring, ring near and far, ring-little, little near and far).",
         "Design: six runs of 145 trials, each a lead-in trial followed by 12 blocks of 12 trials with the same 12-target sequence (sequence 1, targets 13-3-25-21-13-"
         "25-3-21-25-13-21-3, or sequence 2, 13-23-5-1-13-5-23-1-5-13-1-23; group). In runs 1-4 (blocks 1-48, phase learning) the cursor was visible in odd and "
         "hidden in even blocks; runs 5-6 (phase transfer) show the cursor throughout with the other target sequence. trial_info lists the target cell (k = 5 x row + column - 5), "
         "its centre in px, frames_on_target (the authors' cnt_hit: frames with the cursor within half a cell of the target centre), proportion_on_target "
         "(the authors' trial success rate, frames over 240) and start_in_run_s (the authors' LearnTrialStartTime in seconds; it restarts in each run). "
         "Lead-in trials (trial_in_run 0) are marked excluded_by_authors because the authors leave them out of their success-rate analysis.",
         "Glove samples are aligned to each cursor frame as in the authors' script: the sample 20 positions after the frame's start in the continuous glove record, "
         "which is the last sample of the 20-sample weighted window from which the live cursor was computed. Two of the 26 participants in the deposit have only 4 and 3 "
         "runs; the paper analyses the 24 who completed all sessions, so those two are marked subject_excluded_by_authors (the deposit does not say which participants were excluded; "
         "this is inferred from the two incomplete sessions and the paper's statement that two participants were excluded for fatigue). Age and sex are not shared per participant."],
    differences=["The deposit has no license file or data dictionary; variable meanings follow the authors' script and the paper. The paper's success-rate statistics "
                 "are reproduced from proportion_on_target of the 24 complete participants."],
    unit_note="4-s trial of the glove-controlled cursor task",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
for ch_ in checks.items:
    if ch_["kind"] == "published":
        print(f"  {ch_['what']}: computed {ch_['computed']:.4f}, reported {ch_['reported']}, ok={ch_['ok']}")
