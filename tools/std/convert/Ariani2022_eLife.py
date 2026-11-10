"""Ariani, Pruszynski & Diedrichsen (2022), eLife 11:e69517: single-finger planning, finger forces in 100 ms bins, one row per trial.

Run from the site folder: python3 -I tools/std/convert/Ariani2022_eLife.py
Raw file (untrusted data, read only) in raw_extra/Ariani2022_eLife/, from https://github.com/g14r/single-finger-planning:
  force_group.mat   (data/force_data-group-100ms.mat) tables pre, plan, exe: mean (and SD) force of each of the five fingers in 100 ms bins
                    for each trial of 22 participants; the code sfp_imana.m (read as text, never run) shows how the tables are used.
The deposit's other files (fMRI-derived ROI data, surfaces, PCM models) are not behavioural and are not used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Ariani2022_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

m = sio.loadmat(RAW / "force_group.mat", struct_as_record=False, squeeze_me=True)
PH = [("pre", "preT_bin", "pre", 0), ("plan", "prepTime_bin", "PF", 1), ("exe", "exeTime_bin", "EF", 2)]
KEY = ["SN", "BN", "TN"]
frames = {}
for name, binc, fc, code in PH:
    s = m[name]
    d = pd.DataFrame({f: getattr(s, f) for f in s._fieldnames}).rename(columns={binc: "bin", fc: "force", fc + "_sd": "sd"})
    d["rep"] = d.groupby(KEY + ["finger", "bin"]).cumcount()
    d["phase"] = code
    frames[name] = d
# order of recordings: first appearance in the 'pre' table
pre = frames["pre"]
order = pre.drop_duplicates(KEY + ["rep"])[KEY + ["rep", "exeType", "seqNum", "prepTime", "isError"]].reset_index(drop=True)
checks.internal("13 trials appear twice (same run and trial number) and are kept as separate recordings",
                (order.rep > 0).sum() == 13 and order.rep.max() == 1, str((order.rep > 0).sum()))
for name in ("plan", "exe"):
    o2 = frames[name].drop_duplicates(KEY + ["rep"])[KEY + ["rep", "exeType", "seqNum", "prepTime", "isError"]]
    mm = order.merge(o2, on=KEY + ["rep"], how="outer", indicator=True, suffixes=("", "_2"))
    checks.internal(f"{name} table holds the same recordings with the same go/no-go, sequence, planning time and error flag as pre",
                    (mm._merge == "both").all() and all((mm[c] == mm[c + "_2"]).all() for c in ("exeType", "seqNum", "prepTime", "isError")))
allf = pd.concat(frames.values(), ignore_index=True)
# single-finger sequences: the finger that presses is the one with the highest mean execution force on go trials
ex = allf[(allf.phase == 2) & (allf.exeType == 1)].groupby(["seqNum", "finger"]).force.mean().unstack().idxmax(axis=1)
checks.internal("sequences 1, 2, 3 are pressed with finger 1, 3, 5 (thumb, middle, little): highest mean execution force on go trials",
                ex[1] == 1 and ex[2] == 3 and ex[3] == 5, str(ex.to_dict()))

wide = allf.pivot_table(index=KEY + ["rep", "phase", "bin"], columns="finger", values=["force", "sd"], aggfunc="first")
checks.internal("every recording-phase-bin has all five fingers", wide.notna().all().all())
wide.columns = [f"{a}_f{b}" for a, b in wide.columns]
wide = wide[[f"force_f{i}" for i in range(1, 6)] + [f"sd_f{i}" for i in range(1, 6)]].reset_index()
wide["time_s"] = wide["bin"] / 10.0
wide = wide.sort_values(KEY + ["rep", "phase", "bin"])
groups = {k: g for k, g in wide.groupby(KEY + ["rep"], sort=False)}

SEQ = {1: "single-finger sequence, thumb (111,111)", 2: "single-finger sequence, middle finger (333,333)",
       3: "single-finger sequence, little finger (555,555)", 4: "multi-finger sequence 4", 5: "multi-finger sequence 5",
       6: "multi-finger sequence 6"}
CHN = ["phase", "bin"] + [f"force_f{i}" for i in range(1, 6)] + [f"force_sd_f{i}" for i in range(1, 6)]
rows, extra = [], []
cnt = {}
for _, r in order.iterrows():
    sn = int(r.SN)
    cnt[sn] = cnt.get(sn, 0) + 1
    g = groups[(r.SN, r.BN, r.TN, r.rep)]
    fn = f"samples/P{sn:02d}_run{int(r.BN):02d}_trial{int(r.TN):02d}" + (f"_b{int(r.rep) + 1}" if r.rep else "") + ".csv"
    out = g[["time_s", "phase", "bin"] + [f"force_f{i}" for i in range(1, 6)] + [f"sd_f{i}" for i in range(1, 6)]].copy()
    out.columns = ["time_s", "phase", "bin"] + CHN[2:]
    extra.append((fn, out.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
    go = "go" if r.exeType == 1 else "no-go"
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group="", subject=str(sn), person=str(sn), subject_excluded_by_authors=False,
        age="", sex="", handedness="right", population="healthy", setting="lab",
        device="MRI-compatible keyboard with force transducers under the fingertips of the right hand (Honeywell FS series)",
        session=1, block=f"run {int(r.BN)}", trial=cnt[sn], phase="test", time_s="", task="finger movements", unit="recording",
        condition=f"{go}, {SEQ[int(r.seqNum)]}, planning {int(r.prepTime) // 1000} s" + (", error trial" if r.isError else ""),
        pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=cnt[sn], samples_file=fn, signal_kind="finger force, mean and SD of five fingers in 100 ms bins (pre-cue, planning and execution phases)",
        channels=";".join(CHN), channel_units="phase code (0 pre, 1 plan, 2 exe);bin number;" + ";".join(["N"] * 10),
        sample_rate_hz=10, n_samples=len(g), duration_s=len(g) / 10.0, events=f"go_cue_in_planning_axis={int(r.prepTime) / 1000:g}",
        processing="forces low-pass filtered and averaged in 100 ms bins by the authors"))
df = pd.DataFrame(rows)

# ---- published numbers (Methods: participants, design)
checks.published("participants analysed", df.person.nunique(), 22, 0, "Methods, Participants")
tr = order
checks.published("proportion of go trials", (tr.exeType == 1).mean(), 0.6, 0.02, "Methods, Task design")
for pt, p in ((4000, 0.56), (6000, 0.30), (8000, 0.14)):
    checks.published(f"proportion of trials with {pt // 1000} s preparation", (tr.prepTime == pt).mean(), p, 0.02, "Methods, Task design")
checks.published("planning-phase bins in an 8 s trial", wide[(wide.phase == 1)].groupby(KEY + ["rep"]).size().max(), 80, 0, "Methods (8 s preparation, 100 ms bins)")
checks.internal("280 distinct trials per participant (279 for one), 28 per run of the 30 in the design",
                set(order.drop_duplicates(KEY).groupby("SN").size()) == {279, 280}, str(order.drop_duplicates(KEY).groupby("SN").size().value_counts().to_dict()))
checks.internal("rows = recordings in the 'pre' table", len(df) == len(order) == 6172)

meta = dict(
    paper=dict(authors="Ariani G, Pruszynski JA, Diedrichsen J", year=2022,
               title="Motor planning brings human primary somatosensory cortex into action-specific preparatory states",
               venue="eLife 11:e69517", doi="10.7554/eLife.69517"),
    lab="Diedrichsen (Western University)", data_url="https://github.com/g14r/single-finger-planning",
    license="GPL-3.0", license_url="https://www.gnu.org/licenses/gpl-3.0.html",
    changes=("Converted from MATLAB to one CSV of binned force samples per trial plus the OpenMotor table. Only data/force_data-group-100ms.mat "
             "is used (the fMRI-derived ROI data, cortical surfaces and PCM files are not behavioural). Forces are the authors' 100 ms bin means "
             "and SDs; the raw force traces are not in the deposit (one participant's raw trial table and video are not used)."),
    how=["One row is one trial of an instructed-delay finger task done in the MRI scanner with the right hand: a sequence cue (a single-finger sequence on "
         "the thumb, middle or little finger, or a multi-finger sequence) was shown, followed by 4, 6 or 8 s of preparation during which the fingers rested "
         "on the keys at 0.2 to 0.4 N, then a go cue (60 percent of trials) after which six presses were made, or no go (40 percent) with no movement. "
         "Its samples (samples/P<participant>_run<NN>_trial<NN>.csv) are the authors' 100 ms bins: for each of the five fingers (1 thumb, 2 index, 3 middle, "
         "4 ring, 5 little), the mean force in the bin (force_f1..5, N) and the SD of the force within the bin (force_sd_f1..5, N).",
         "Three phases are stacked in each file: phase 0 pre-cue baseline (bins -9 to 0), phase 1 preparation (bins 1 to 40, 60 or 80), phase 2 execution "
         "(bins 81 to 120, numbered after the longest preparation as in the authors' plotting code). time_s is the bin number divided by 10, the position on the "
         "authors' time axis; the authors' bin edges are not stated. The go cue falls at the end of the preparation bins (events). 'error trial' in "
         "condition is the authors' isError flag (timing, press or force-zone error); the paper includes all trials, so none is excluded here.",
         "The deposit holds 6,159 distinct trials of 22 participants (279 or 280 of the 300 run each) and 13 trial numbers that occur twice; both copies are kept. "
         "Participant age and sex are not shared per participant. Sequences 1, 2 and 3 are the thumb, middle and little finger (checked against the force data); sequences 4 to 6 are multi-finger and are not described further in the deposit."],
    differences=["The paper has 300 trials per participant (180 go, 120 no-go); the force deposit holds 279 or 280 distinct trials per participant (28 of the 30 per run; 6,172 recordings in all, "
                 "including 13 repeated trial numbers), so counts of trials do not match the design. The go and planning-time proportions do reproduce."],
    unit_note="trial (finger forces in 100 ms bins)",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
