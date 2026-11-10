"""Hussain & Cole (2015) PLoS ONE 10(6):e0129543: continuous isometric visuomotor tracking task (CIVTT), 26 young adults,
200 mg caffeine or placebo taken after day-1 practice, retest 24 h later.
Deposit: Dryad doi:10.5061/dryad.j91v7, read from its Zenodo copy (record 5008266, CC0): 'All Raw Data.zip' (one text file per
block: sample number, clock in ms, force in N; 30 trials of 6000 samples per block), the complex waveform (xlsx), the list of
trials to remove (docx, read as text), the processed data (xlsx; sex, age, caffeine intake, sleep, group) and the analysis script
(read as text only). Nothing from the deposit is executed.
Run from the site folder: python3 -I tools/std/convert/Hussain2015_PLoSONE.py"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hussain2015_PLoSONE"
MODULE = "timeseries"
RD = stdlib.raw_dir(ID)
ROOT = "Hussain&Cole2015 Raw Data/"
EPOCH, TRIAL_N, NTR = 6000, 4400, 30

# trials the authors list for removal (data collection errors): subject, day, block, trial
REMOVE = {(1, 1, 5, 26), (3, 2, 2, 7), (3, 2, 3, 13), (3, 2, 5, 26), (6, 2, 1, 12), (8, 1, 1, 17), (8, 1, 5, 17), (15, 2, 3, 10),
          (28, 2, 4, 30), (30, 1, 1, 16), (30, 1, 3, 8), (30, 1, 3, 21), (31, 1, 2, 14), (31, 1, 3, 4), (31, 1, 3, 16), (31, 1, 3, 23)}

# The docx says 'Subject 3: ... Day 2 Block 6 Trial 26' but there are only 5 blocks a day; Block 5 is the one whose deposited block mean
# (processed data) reproduces only with that trial removed, so it is read as Block 5 (see the README).
# the docx is read as text to confirm the list above
dz = zipfile.ZipFile(RD / "Hussain_Cole2015_listoftrialstoremove.docx")
txt = re.sub(r"<[^>]+>", " ", dz.read("word/document.xml").decode())
found = set()
for m in re.finditer(r"Subjects? (\d+):([^S]*?)(?=Subject|$)", re.sub(r"\s+", " ", txt)):
    s = int(m.group(1))
    for dm in re.finditer(r"Day (\d) Block (\d) Trial ([\d, and]+)", m.group(2)):
        for t in re.findall(r"\d+", dm.group(3)):
            found.add((s, int(dm.group(1)), int(dm.group(2)), int(t)))
found = {(a, b, 5 if (a, b, c, d) == (3, 2, 6, 26) else c, d) for a, b, c, d in found}
assert found == REMOVE, (sorted(found ^ REMOVE))

wb = openpyxl.load_workbook(RD / "Hussain_Cole2015_processeddata.xlsx", data_only=True)
prow = [r for r in wb.active.iter_rows(values_only=True)][1:]
part = {int(r[0][1:]): dict(group="caffeine" if r[1] == "C" else "placebo", sex=r[2], age=float(r[3]), caff=float(r[4]), sleep_h=float(r[5]),
                            sleep_q=float(r[6]), blk=[float(x) for x in r[9:19]], rate=[float(r[7]), float(r[8])]) for r in prow if r[0]}
assert len(part) == 26
wf = np.array([r[2] for r in openpyxl.load_workbook(RD / "Hussain_Cole2015_complexwaveform.xlsx", data_only=True)["Sheet1"].iter_rows(values_only=True)], float)
assert len(wf) == 6001
wf4 = wf[:TRIAL_N]
z = zipfile.ZipFile(RD / "raw.zip")

rows, extra, pp = [], [], []
rmse = {}
jump_log = []
n_jumps_epoch = []
n_dot = 0
n_samples_total = 0
for si, s in enumerate(sorted(part), start=1):
    p = part[s]
    sub = f"{ID}_S{si:04d}"
    trial = 0
    for day in (1, 2):
        for blk in range(1, 6):
            raw = z.read(f"{ROOT}s{s:02d}/s{s:02d}d{day}/s{s:02d}_day{day}_block{blk}.txt").decode().splitlines()
            assert len(raw) == NTR * EPOCH + 1, (s, day, blk, len(raw))
            tok = [l.split() for l in raw]
            idx = np.array([int(t[0]) for t in tok])
            assert (idx == np.arange(1, len(tok) + 1)).all()
            clock = np.array([float(t[1]) if t[1] != '.' else np.nan for t in tok])
            force = np.array([float(t[2]) if t[2] != '.' else np.nan for t in tok])
            n_dot += int(np.isnan(force).sum())
            for k in range(NTR):
                a, b = k * EPOCH, (k + 1) * EPOCH + (1 if k == NTR - 1 else 0)
                ck_ = clock[a:b]
                dd = np.diff(ck_)
                jumps = np.flatnonzero(dd[~np.isnan(dd)] != 1)
                jump_log.extend(int(j) for j in jumps)
                n_jumps_epoch.append(len(jumps))
                t_s = np.arange(b - a) / 1000.0
                body = "time_s,force_N\r\n" + "".join(f"{t:.3f},{'' if tok[a + i][2] == '.' else tok[a + i][2]}\r\n" for i, t in enumerate(t_s))
                gap_txt = ";clock_jump_after_sample=" + "|".join(str(int(j) + 1) for j in jumps) if len(jumps) else ""
                trial += 1
                fname = f"samples/{sub}_day{day}_block{blk}_trial{k + 1:02d}.csv"
                extra.append((fname, body.encode()))
                n_samples_total += b - a
                rm = float(np.sqrt(np.mean((force[a:a + TRIAL_N] - wf4) ** 2)))  # NaN for the one trial with missing samples
                rmse[(s, day, blk, k + 1)] = rm
                excl = (s, day, blk, k + 1) in REMOVE
                rows.append(dict(
                    dataset=ID, study="Hussain & Cole 2015 PLoS ONE", experiment="", group=p["group"], subject=sub, person=sub,
                    subject_excluded_by_authors=False, age=p["age"], sex=p["sex"], handedness="right", population="healthy young adults",
                    setting="lab", device="isometric force transducer (index finger abduction), cursor on a display", session=day,
                    block=f"day{day}_block{blk}", trial=trial, phase="learning" if day == 1 else "test", time_s=np.nan,
                    task="tracking", unit="recording", condition="", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                    excluded_by_authors=excl, recording=trial, samples_file=fname, signal_kind="isometric index-finger abduction force",
                    channels="force", channel_units="N", sample_rate_hz=1000.0, n_samples=b - a, duration_s=(b - a) / 1000.0,
                    events="trial_end=4.4", processing="none stated (force samples as shared)",
                    trial_info=f"trial_in_block={k + 1};tracking_window_samples=4400{gap_txt}"))
    pp.append(dict(person=sub, group=p["group"], sex=p["sex"], age=p["age"], weekly_caffeine_mg=p["caff"], sleep_hours=p["sleep_h"], sleep_quality=p["sleep_q"]))
df = pd.DataFrame(rows)

# target waveform as an extra file (the same for every trial)
wbody = "time_s,target_force_N\r\n" + "".join(f"{i / 1000:.3f},{v:g}\r\n" for i, v in enumerate(wf))
extra.append(("target_waveform.csv", wbody.encode()))

ck = stdlib.Checks()
pdf = pd.DataFrame(pp)
ck.published("participants", df.person.nunique(), 26, 0, "Methods: 26 subjects; ANOVA df 1,24")
ck.published("caffeine group size (6 M + 8 F)", (pdf.group == "caffeine").sum(), 14, 0, "Table 1 sex counts")
ck.published("placebo group size (6 M + 6 F)", (pdf.group == "placebo").sum(), 12, 0, "Table 1 sex counts")
ck.published("men in caffeine group", ((pdf.group == "caffeine") & (pdf.sex == "M")).sum(), 6, 0, "Table 1")
ck.published("men in placebo group", ((pdf.group == "placebo") & (pdf.sex == "M")).sum(), 6, 0, "Table 1")
for g, ref_age, ref_c, ref_q, ref_h in (("caffeine", 22.9, 160.8, 4.1, 7.2), ("placebo", 24.0, 97.5, 4.1, 7.3)):
    q = pdf[pdf.group == g]
    ck.published(f"mean age, {g} group", q.age.mean(), ref_age, 0.05, "Table 1")
    ck.published(f"mean weekly caffeine intake (mg), {g} group", q.weekly_caffeine_mg.mean(), ref_c, 0.05, "Table 1")
    ck.published(f"mean sleep quality, {g} group", q.sleep_quality.mean(), ref_q, 0.05, "Table 1")
    ck.published(f"mean sleep duration (h), {g} group", q.sleep_hours.mean(), ref_h, 0.05, "Table 1")
ck.published("trials per day (5 blocks x 30)", len(df) / 26 / 2, 150, 0, "Methods: 5 blocks of 30 trials per day")
ck.published("trial duration (s): tracking window samples / 1000 Hz", TRIAL_N / df.sample_rate_hz.iloc[0], 4.4, 0, "Methods: 4.4 s trials, force sampled at 1000 Hz")
ck.published("peak of the target waveform (N)", wf4.max(), 1.02, 0.005, "Methods: peak waveform height 12.5 cm = 1.02 N")

# block mean RMSE vs the deposit's processed data
dev = {}
for s in part:
    for day in (1, 2):
        for b in range(1, 6):
            v = np.nanmean([rmse[(s, day, b, k)] for k in range(1, 31) if (s, day, b, k) not in REMOVE])
            dev[(s, day, b)] = abs(v - part[s]["blk"][(day - 1) * 5 + b - 1])
off = sorted(k for k, v in dev.items() if v > 1e-6)
ck.internal("block mean RMSE (trial RMSE to the waveform over 4400 samples, authors' removed trials left out) equals the deposit's processed block means in 259 of 260 participant blocks "
            "(the exception, participant 30 day 1 block 1, differs by 4e-4; see README)", off == [(30, 1, 1)] and dev[(30, 1, 1)] < 1e-3, f"{len(off)} differ: {off}; max {max(dev.values()):.2e}")
ck.internal("the authors' clock steps 1 ms between samples except at one jump within the first few samples of most 6000-sample windows (the window grid is the authors' fixed one)", True, f"{sum(n_jumps_epoch)} jumps in {len(n_jumps_epoch)} windows, {sum(1 for j in jump_log if j < 20)} within the first 20 samples")
ck.internal("recordings = 26 participants x 300 trials", len(df) == 26 * 300 and df.groupby("person").size().eq(300).all())
ck.internal("16 trials flagged as removed by the authors", int(df.excluded_by_authors.sum()) == 16)
ck.internal("samples: 26 x 10 blocks x 180001", n_samples_total == 26 * 10 * 180001)
ck.internal("the only missing samples are 5990 placeholders (a dot) in one trial the authors list for removal (last trial of a block), left empty", n_dot == 5990 and bool(np.isnan(rmse[(28, 2, 4, 30)])), str(n_dot))

meta = dict(
    paper=dict(authors="Hussain SJ, Cole KJ", year=2015, title="No Enhancement of 24-Hour Visuomotor Skill Retention by Post-Practice Caffeine Administration",
               venue="PLoS ONE 10(6):e0129543", doi="10.1371/journal.pone.0129543"),
    lab="Cole (University of Iowa)", data_url="https://doi.org/10.5061/dryad.j91v7 (copy at https://zenodo.org/records/5008266)",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Samples the authors' file marks with a dot (5990 samples of one removed trial) are left empty. Each block text file (30 trials of 6000 samples, one extra final sample) was cut into one CSV per trial with time_s (the authors' ms clock, from 0) and "
             "force in N, values unchanged. The trial window the authors analyse is the first 4400 samples (4.4 s); the remaining 1600 samples are the rest period "
             "and are kept. The last trial of each block also carries the file's final sample (6001 samples). Sex, age, weekly caffeine intake and sleep come from the processed-data sheet."),
    how=["One row is one trial: the force an index finger pressed (isometric abduction) while a cursor on a display tracked a fixed complex waveform that scrolled "
         "left to right; samples are force in newtons at 1000 Hz. 26 young adults did 5 blocks of 30 trials on day 1, took 200 mg caffeine or a placebo, and did 5 more blocks 24 h later.",
         "events marks trial_end = 4.4 s: samples after it are the rest before the next trial (the authors score the first 4400 samples against the 4400-point target waveform in target_waveform.csv, at 1 ms spacing).",
         "excluded_by_authors marks the 16 trials the authors' list says to remove because of data collection errors. group is caffeine or placebo (given after the day-1 practice). "
         "phase is learning for day 1 and test for day 2 (the 24-hour retention session). Trial counts 1-300 across both days."],
    unit_note="trial (6 s of isometric finger force at 1000 Hz, tracking a fixed waveform)",
    differences=["The paper's exponential learning rates, R2 (0.94 day 1, 0.64 day 2) and ANOVA results are on 15 bins of 10 trials per day. Refitting a + b exp(-c n) to the block-trial RMSE did not give the deposited fit parameters or R2, so no learning-rate number is used as a check.",
                 "The authors' removal list names 'Subject 3: Day 2 Block 6 Trial 26'; there is no block 6, and the deposited block means reproduce only if that trial is removed from block 5, so it is read as block 5.",
                 "For one participant-block (participant 30, day 1, block 1) the deposited block mean RMSE (0.28482) is not reproduced from the raw force with the authors' removal list (0.28521); the other 259 block means match to 1e-6. No trial pair or single extra trial explains it, so nothing was changed.",
                 "In the authors' files the clock jumps (tens to thousands of ms) within the first few samples of most 6000-sample windows (6729 of 7800), so the window grid starts a few samples before the true trial start; time_s here is sample number / 1000 within the window, and trial_info lists the jump position.",
                 "The paper reports one data point removed as an outlier for the learning-rate analysis; the authors' deposited list of 16 removed trials is a different (data collection error) list and is what excluded_by_authors marks."])
pcols = [("group", "text", "caffeine or placebo", "Capsule taken after day-1 practice."), ("sex", "text", "", "Sex as reported."), ("age", "number", "years", "Age."),
         ("weekly_caffeine_mg", "number", "mg", "Average weekly caffeine consumption from the authors' paper-and-pencil survey."),
         ("sleep_hours", "number", "hours", "Sleep duration reported."), ("sleep_quality", "number", "1-5", "Sleep quality, 5-point scale.")]
stdlib.release(ID, MODULE, df, meta, ck, participants=pd.DataFrame(pp), participant_columns=pcols, extra_files=extra)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))
