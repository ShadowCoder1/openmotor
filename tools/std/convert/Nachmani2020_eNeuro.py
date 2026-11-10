"""Nachmani, Coutinho, Khan, Lefevre & Blohm (2020), eNeuro 7(1):ENEURO.0196-18.2019: predicted position error triggers catch-up saccades
during smooth pursuit (double step-ramp task, 15 participants, 100 blocks of 50 trials each). Data: Zenodo 3902832 (CC0 copy of the Dryad deposit),
LabeledS*.tar archives. Each block's .mat file holds D (raw eye traces, ~14 MB, not read), param (50 x 33 trial table), PE and RS (50 x 1501
position-error and retinal-slip series, -500..+1000 ms around the second target step). The tar members are read by HTTP range requests that skip D;
a helper (run on the PC beforehand) cached param/PE/RS of every block as small npz files in tmp_samples/<id>/cache. This converter reads that cache.

Run on the PC after the cache job has finished: python3 -I tools/std/pc.py run Nachmani2020_eNeuro
"""
import collections
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Nachmani2020_eNeuro"
TMPROOT = stdlib.HERE / "tmp_samples" / ID
CACHE = TMPROOT / "cache"
OUTD = TMPROOT / "out"
checks = stdlib.Checks()

files = sorted(glob.glob(str(CACHE / "*.npz")))
assert len(files) > 1400, len(files)
blocks = []
relabeled = []
for f in files:
    z = np.load(f)
    p = z["param"]
    folder_subj = int(os.path.basename(f).split("_")[0][len("Subject"):])  # tar folder = participant (17 files carry another participant's number inside)
    if folder_subj != int(p[0, 0]):
        relabeled.append(os.path.basename(f))
    blocks.append((folder_subj, int(p[0, 1]), os.path.basename(f), f))
    assert len({int(x) for x in p[:, 0]}) == 1 and len({int(x) for x in p[:, 1]}) == 1
blocks.sort()
subs = sorted({b[0] for b in blocks})
rank = {s: i + 1 for i, s in enumerate(subs)}
OUTD.mkdir(parents=True, exist_ok=True)

DEVICE = ("EyeLink 1000 video eye tracker (1000 Hz, horizontal and vertical gaze), ViewPixx 120 Hz screen, head on a chin rest; "
          "eye position low-pass filtered at 50 Hz (zero-phase) by the authors")


def num(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else v


rows, extra = [], []
seen_blocks = collections.Counter()
tr_count = collections.Counter()
unique_ids = set()
cond_by_block = collections.defaultdict(set)
t_ms = np.arange(-500, 1001)
for (s, bid, nm, f) in blocks:
    z = np.load(f)
    P, PE, RS = z["param"], z["PE"].astype(np.float64), z["RS"].astype(np.float64)
    si = rank[s]
    seen_blocks[(s, bid)] += 1
    dup = seen_blocks[(s, bid)] > 1
    sfx = "b" if dup else ""
    fn = f"samples/{ID}_S{si:04d}_block{bid:03d}{sfx}.csv"
    cols = {"t_ms": t_ms}
    for k in range(len(P)):
        cols[f"t{k + 1:02d}_PE_deg"] = PE[k]
        cols[f"t{k + 1:02d}_RS_deg_s"] = RS[k]
    pd.DataFrame(cols).to_csv(OUTD / os.path.basename(fn), index=False, float_format="%.4g", lineterminator="\r\n")
    extra.append((fn, str(OUTD / os.path.basename(fn))))
    for k in range(len(P)):
        r = P[k]
        tr_count[s] += 1
        unique_ids.add((s, bid, int(r[3])))
        sacc = r[11] > 0  # at least one saccade after the second target step
        sess = (bid - 10) // 10 + 1
        cond_by_block[(s, sess)].add(int(r[5]))
        v1, step, v2 = r[6], r[7], r[8]
        info = [f"trial_in_block={int(r[3])}", f"series_file={fn}", f"series_columns=t{k + 1:02d}_PE_deg|t{k + 1:02d}_RS_deg_s",
                f"flag_saccade_200_to_100ms_before_step={int(r[25])}", f"flag_saccade_within_100ms_before_step={int(r[26])}",
                f"flag_saccade_during_step={int(r[27])}", f"flag_trigger_time_short={int(r[28])}",
                f"flag_saccade_to_first_ramp={int(r[29])}", f"flag_double_saccade_check_1={int(r[30])}",
                f"flag_double_saccade_check_2={int(r[31])}", f"flag_col33_not_used={int(r[32])}", f"block_recorded_twice={dup}"]
        row = dict(
            dataset=ID, study="Nachmani et al. 2020, eNeuro", experiment="double step-ramp smooth pursuit", group="",
            subject=f"s{s:02d}", person=f"s{s:02d}", subject_excluded_by_authors=False, age="", sex="", handedness="",
            population="healthy", setting="lab", device=DEVICE, session=sess, block=str(bid), trial=tr_count[s], phase="test", time_s="",
            task="eye movements", unit="trial", condition="blurred target" if r[5] == 1 else "clear target",
            pert_type="second_target_step_and_ramp", pert_value=step, pert_unit="deg", feedback="normal",
            excluded_by_authors=bool(r[24] == 1),
            n_saccades=int(r[10]),
            pursuit_target_speed_deg_s=abs(v1), pursuit_target_direction="rightward" if v1 > 0 else "leftward",
            first_ramp_velocity_deg_s=v1, second_step_deg=step, second_ramp_velocity_deg_s=v2, velocity_step_deg_s=r[9],
            saccades_after_second_step=int(r[11]), saccades_before_second_step=int(r[12]),
            trial_info=";".join(info))
        if sacc:
            row.update(saccade_onset_ms=r[13], saccade_duration_ms=r[14], saccade_start_x_deg=r[15], saccade_end_x_deg=r[16],
                       saccade_peak_velocity_deg_s=r[17], eye_x_100ms_before_saccade_deg=r[18],
                       position_error_100ms_before_saccade_deg=r[19], retinal_slip_100ms_before_saccade_deg_s=r[20],
                       position_error_at_saccade_start_deg=r[21], position_error_at_saccade_end_deg=r[22],
                       position_error_at_second_step_deg=r[23])
        rows.append(row)

try:
    n_rows = len(rows)
    ub = collections.defaultdict(set)
    for (s, bid, t) in unique_ids:
        ub[s].add(bid)
    nblk = {s: len(v) for s, v in ub.items()}
    ntr_unique = len(unique_ids)
    print("participants", len(subs), "rows", n_rows, "unique trials", ntr_unique, "blocks per participant", sorted(nblk.values()))
    sess_clear = collections.defaultdict(int)
    alt_ok = all(len(v) == 1 for v in cond_by_block.values())
    for (s, sess), v in cond_by_block.items():
        if v == {0}:
            sess_clear[s] += 1
    bad_frac = float(np.mean([r["excluded_by_authors"] for r in rows]))
    v1s = {abs(r["first_ramp_velocity_deg_s"]) for r in rows}
    checks.published("participants", len(subs), 15, 0, "Methods, Sampling plan: 15 participants")
    checks.published("participants with 100 distinct blocks", sum(1 for v in nblk.values() if v == 100), 15, 0,
                     "Methods, Sampling plan: 10 sessions of 10 blocks each, 100 blocks per participant")
    checks.published("distinct trials in the deposit", ntr_unique, 75000, 0, "Methods, Sampling plan: 5000 trials per participant, 75,000 in all")
    checks.published("trials per block", collections.Counter(len(np.load(f)["param"]) for f in files).most_common(1)[0][0], 50, 0,
                     "Methods, Sampling plan: blocks of 50 trials")
    checks.published("participants with 5 clear-target sessions (alternating clear and blurred)", sum(1 for s in subs if sess_clear[s] == 5), 15, 0,
                     "Methods, Experimental procedure: ten alternating sessions of clear and blurred target")
    checks.published("distinct first-ramp speeds (10, 20, 30 deg/s)", len(v1s), 3, 0, "Methods, Experimental procedure: VS 10/20/30 deg/s")
    checks.published("largest second target step (deg)", max(abs(r["second_step_deg"]) for r in rows), 20, 0, "Methods, Experimental procedure: PS -20 to 20 deg")
    checks.published("all velocity steps within -50 to 50 deg/s (1 = yes)", int(max(abs(r["velocity_step_deg_s"]) for r in rows) <= 50), 1, 0,
                     "Methods, Experimental procedure: VS uniform from -50 to 50 deg/s")
    checks.internal("every row has a series file", all("series_file=samples/" in r["trial_info"] for r in rows))
    checks.internal("condition constant within runs of ten block labels (sessions)", alt_ok)
    checks.internal("flagged-bad share of trials is a few percent", 0.005 < bad_frac < 0.2, f"{bad_frac:.3f}")
    differences = [
        f"The paper says about 23% of the 75,000 trials were discarded (about 58,000 kept). The deposit's own manual good/bad label (param column 25) marks {100 * bad_frac:.1f}% of rows, and that is what excluded_by_authors holds. The rest of the 23% comes from further criteria of the authors' analysis code (blinks, saccades at the step, double saccades, saccades to the first ramp); the deposit's flag columns for those criteria are kept in trial_info, but no combination of them was fitted to the 23%, so the share is not reproduced here.",
        f"{len(relabeled)} block files ({len(relabeled) * 50} trials) sit in the archive folder of one participant but carry the number of another participant inside (and in their file names): 7 blocks in folder 8 are labelled 13 and 10 blocks in folder 15 are labelled 16. Participants 8 and 15 would otherwise lack exactly those blocks and participants 13 and 16 would have those block labels twice, so the folder is taken as the true participant; this is inferred, not stated by the authors.",
        "README of the deposit lists param column 25 as 'good or bad (0 or 1)'; the authors' extraction code stores good = 0, bad = 1, which is how it is read here.",
        "Column 29 ('trigger time less than 50 ms' in the README) is computed as trigger time < 80 ms in the authors' code; it is kept as flag_trigger_time_short. The README of the deposit marks column 33 'not used'; it is kept as flag_col33_not_used.",
        "Sessions are not labelled in the deposit. The block labels run 10 to 109; the condition (clear or blurred) is constant within each run of ten labels and alternates between runs, so session = (block label - 10) // 10 + 1 is inferred from that, and the order of blocks in time is assumed to follow the labels.",
        "Rows with no saccade after the second target step (smooth trials) have empty saccade columns; the deposit stores zeros there.",
    ]
    meta = dict(
        paper=dict(authors="Nachmani O, Coutinho J, Khan AZ, Lefevre P, Blohm G", year=2020,
                   title="Predicted position error triggers catch-up saccades during sustained smooth pursuit",
                   venue="eNeuro 7(1):ENEURO.0196-18.2019", doi="10.1523/eneuro.0196-18.2019"),
        lab="Blohm (Queen's University) / Lefevre (UCLouvain)", data_url="https://doi.org/10.5061/dryad.245j1p8 (Zenodo copy: https://zenodo.org/records/3902832)",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes="Only each trial's table row (param) and the position-error and retinal-slip series (PE, RS) were kept; the raw eye traces (D) and the EyeLink .edf files were not released. Participant numbers were replaced by codes; param columns were renamed to saccade-module columns; zeros stored for smooth trials were replaced by empty cells; series were rounded to 4 significant digits.",
        how=["One row is one trial of a double step-ramp pursuit task: after fixation the target stepped 2, 4 or 6 deg away from the screen centre and ramped back toward it at 10, 20 or 30 deg/s to induce pursuit; after 500-700 ms a second step and ramp (second step -20 to 20 deg, second ramp velocity -50 to 50 deg/s) occurred in a random direction, and participants tracked the target (clear or blurred). The saccade columns describe the first saccade after the second step (the catch-up saccade); trials with none are smooth trials and have those columns empty.",
             "Positions and velocities are horizontal, in deg and deg/s, positive rightward (the first ramp has the opposite sign of the target's start side, as the target ramps toward the centre). Position error is target minus eye. saccade_onset_ms is the trigger time: time from the second target step to saccade onset. Retinal slip is the second ramp velocity minus eye velocity. n_saccades is the number of saccades detected in the whole trial.",
             "Each block of 50 trials has a file samples/<file>.csv with t_ms (-500 to 1000 ms around the second target step, 1 kHz) and, for trial k of the block, the columns tkk_PE_deg (position error, deg) and tkk_RS_deg_s (retinal slip, deg/s); trial_info names the file and columns of the row.",
             "excluded_by_authors is the authors' manual good/bad label (bad = 1); the other criteria of their analysis are the flag_* values in trial_info (1 = flagged).",
             "block is the authors' block label (10 to 109); session is inferred (see differences); trial counts the participant's trials in order of block label."],
        unit_note="trial",
        differences=differences,
    )
    stdlib.release(ID, "saccade", rows, meta, checks, extra_files=extra)
finally:
    shutil.rmtree(OUTD, ignore_errors=True)
