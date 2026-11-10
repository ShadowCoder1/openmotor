"""Beuter, Titcombe, Richer, Gross & Guehl (2001) Thalamus Relat Syst 1(3):203-211: rest tremor velocity of the index finger (laser velocity
transducer on a reflective patch at the fingertip, 100 Hz, about 60 s per recording) in 16 people with Parkinson's disease who have
chronic deep brain stimulation (DBS), recorded with DBS on or off, medication on or off, and 15-60 min after DBS was stopped.
Deposit: PhysioNet tremordb 1.0.0 (ODC-By 1.0), tremordb.zip. Plain text, one velocity value per line.
Run from the site folder: python3 -I tools/std/convert/Beuter2001_ThalamusRelatSyst.py"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Beuter2001_ThalamusRelatSyst"
MODULE = "timeseries"
FS = 100.0
P = "effect-of-deep-brain-stimulation-on-parkinsonian-tremor-1.0.0/"
z = zipfile.ZipFile(stdlib.raw_dir(ID) / "tremordb.zip")
names = {n[len(P):]: n for n in z.namelist() if n.startswith(P)}
txt = lambda k: z.read(names[k]).decode("latin1").splitlines()

# subject_description.txt
subj = {}
order = []
for line in txt("subject_description.txt"):
    if line.startswith("#"):
        continue
    m = re.match(r"^([A-Za-z0-9/%() \-]+?)\s{2,}(\S.*)$", line)
    if not m:
        continue
    lab, rest = m.group(1).strip(), m.group(2).split()
    if lab == "SUBJECT":
        order = rest
        for s in rest:
            subj[s] = {}
    elif len(rest) == len(order):
        for s, v in zip(order, rest):
            subj[s][lab] = v
assert len(subj) == 16 and all("AGE (YRS)" in v for v in subj.values()), len(subj)

# file_description.txt: range, velocity unit, laser setting, rate, samples per file
fd = {}
for line in txt("file_description.txt"):
    t = line.split()
    if len(t) >= 7 and t[1].endswith((".let", ".rit")):
        fd[t[1]] = dict(range_v=t[2], unit=t[3], laser=t[4], rate=float(t[5]), samples=int(t[6]))

COND = {"en": ("DBS on", "medication on", 1, 0), "ef": ("DBS on", "medication off", 1, 0), "on": ("DBS off", "medication on", 0, 0),
        "of": ("DBS off", "medication off", 0, 0)}
rows, extra = [], []
files = sorted([k for k in names if k.endswith((".let", ".rit"))])
recs = []
for k in files:
    f = k.split("/")[-1]
    m = re.fullmatch(r"([gsv])(\d+)r(\d\d)?([eo])([nf])\.(let|rit)", f)
    assert m, f
    recs.append((int(m.group(2)), m.group(1), m.group(3), m.group(4) + m.group(5), m.group(6), k, f))
cond_order = {None: 0, "15": 1, "30": 2, "45": 3, "60": 4}
cat_order = {"en": 0, "ef": 1, "on": 2, "of": 3}
recs.sort(key=lambda r: (r[0], cond_order[r[2]] if r[2] else 0, cat_order[r[3]] if not r[2] else 3))
sub_n = {}
cat_counts = {}
mism = []
for num, tg, mins, cd, side, k, f in recs:
    sid = f"{tg}{num}"
    sd = subj[sid]
    v = np.array([float(x) for x in z.read(names[k]).decode("latin1").split()])
    d = fd[f]
    if d["samples"] != len(v):
        mism.append((f, d["samples"], len(v)))
    assert d["rate"] == FS
    dbs, med, dbs_on, _ = COND[cd]
    if mins:
        cname = f"{mins} min after DBS stopped, medication off"
        block = f"DBS stopped {mins} min"
    else:
        cname = f"{dbs}, {med}"
        block = cname
    sub_n[sid] = sub_n.get(sid, 0) + 1
    grp = "high amplitude tremor" if num <= 8 else "low amplitude tremor"
    cat_counts[(grp, f"r{mins or ''}{cd}")] = cat_counts.get((grp, f"r{mins or ''}{cd}"), 0) + 1
    fname = f"{f.replace('.', '_')}.csv"
    body = "time_s,velocity\r\n" + "".join(f"{i / FS:.2f},{x:.8g}\r\n" for i, x in enumerate(v))
    extra.append((f"samples/{fname}", body.encode()))
    rows.append(dict(
        dataset=ID, study="Beuter et al. 2001 Thalamus Relat Syst", experiment="", group=grp, subject=sid, person=sid, subject_excluded_by_authors=False,
        age=float(sd["AGE (YRS)"]), sex=sd["GENDER"], handedness="", population="Parkinson's disease with deep brain stimulation", setting="clinic",
        device="laser velocity transducer aimed at a reflective patch on the index fingertip (100 Hz)", session=1, block=block, trial=sub_n[sid],
        phase="other", time_s=np.nan, task="finger movements", unit="recording", condition=cname,
        pert_type="dbs_stopped" if not dbs_on else "none", pert_value=(int(mins) if mins else 0) if not dbs_on else 0,
        pert_unit="minutes since stimulator stopped" if not dbs_on else "", feedback="normal", excluded_by_authors=False,
        recording=sub_n[sid], samples_file=f"samples/{fname}", signal_kind="index finger rest tremor velocity",
        channels="velocity", channel_units=d["unit"], sample_rate_hz=FS, n_samples=len(v), duration_s=len(v) / FS, events="",
        processing="none stated (output of the velocity-transducing laser as shared)",
        trial_info=f"side_tested={'left' if side == 'let' else 'right'};dbs={'on' if dbs_on else 'off'};medication={'on' if cd[1] == 'n' else 'off'};"
                   f"laser_range_V={d['range_v']};laser_speed_setting_m_per_s={d['laser']}"))
df = pd.DataFrame(rows)

ck = stdlib.Checks()
ck.published("participants described", len(subj), 16, 0, "PhysioNet record page (Data Description)")
ck.internal("subjects with at least one recording (subject v3 has none in the deposit)", df.subject.nunique() == 15 and "v3" not in set(df.subject))
ck.published("recordings", len(df), 101, 0, "PhysioNet record page: 55 + 46 recordings")
ck.published("recordings of high amplitude tremor subjects", (df.group == "high amplitude tremor").sum(), 37, 0, "PhysioNet record page (Tremor Recordings)")
ck.published("recordings of low amplitude tremor subjects", (df.group == "low amplitude tremor").sum(), 64, 0, "PhysioNet record page (Tremor Recordings)")
ck.published("recordings with DBS on or off, medication on or off (not time-after-stop)", df.condition.isin(["DBS on, medication on", "DBS on, medication off", "DBS off, medication on", "DBS off, medication off"]).sum(), 55, 1, "PhysioNet record page (55, README says 54)")
ck.published("recordings after DBS was stopped (15-60 min)", df.condition.str.contains("after DBS").sum(), 46, 0, "PhysioNet record page")
ck.published("subjects with Vim / GPi / STN targets (Vim)", sum(1 for s in subj.values() if s["STIM TARGET"] == "Vim"), 3, 0, "PhysioNet record page")
ck.published("subjects with GPi target", sum(1 for s in subj.values() if s["STIM TARGET"] == "GPi"), 7, 0, "PhysioNet record page")
ck.published("subjects with STN target", sum(1 for s in subj.values() if s["STIM TARGET"] == "STN"), 6, 0, "PhysioNet record page")
ck.published("men", sum(1 for s in subj.values() if s["GENDER"] == "M"), 11, 0, "subject_description / PhysioNet page")
ck.published("HAT 'ren' recordings", cat_counts[("high amplitude tremor", "ren")], 5, 0, "PhysioNet record page (Tremor Recordings)")
ck.published("LAT 'r60of' recordings", cat_counts[("low amplitude tremor", "r60of")], 8, 0, "PhysioNet record page (Tremor Recordings)")
ck.internal("each file's sample count equals the SAMPLES column of file_description.txt", not mism, str(mism[:5]))
ck.internal("every recording in file_description.txt was converted", set(fd) == {r[6] for r in recs}, f"{len(fd)} vs {len(recs)}")
ck.internal("all samples finite", all(np.isfinite(r["n_samples"]) for r in rows) and all(b"nan" not in body.lower() for _, body in extra))
meta = dict(
    paper=dict(authors="Beuter A, Titcombe MS, Richer F, Gross C, Guehl D", year=2001,
               title="Effect of deep brain stimulation on amplitude and frequency characteristics of rest tremor in Parkinson's disease",
               venue="Thalamus & Related Systems 1(3):203-211", doi="10.1017/S1472928801000206"),
    lab="Beuter (Universite du Quebec a Montreal; Universite Montpellier I)", data_url="https://physionet.org/content/tremordb/1.0.0/",
    license="Open Data Commons Attribution License v1.0 (ODC-By 1.0)", license_url="https://opendatacommons.org/licenses/by/1-0/",
    changes=("Each recording (one text file with one velocity value per line, 100 Hz) was written as a CSV with a time_s column (sample number / 100) and the velocity, "
             "unchanged. Condition, side tested, laser range and laser speed setting come from the file names and file_description.txt; age and sex from "
             "subject_description.txt. The glossary images and checksums were not carried. Years of diagnosis and surgery were left out."),
    how=["One row is one recording of about 60 s of rest tremor of the index finger (left or right, side in trial_info) of one person with Parkinson's disease who has chronic "
         "high-frequency deep brain stimulation (targets Vim, GPi or STN). Conditions: DBS on or off crossed with L-dopa medication on or off (150 percent of the morning dose), "
         "and, with medication off, 15, 30, 45 and 60 minutes after the stimulator was stopped. Subjects 1-8 are the authors' high-amplitude-tremor group, 9-16 the low-amplitude group.",
         "The channel is finger velocity from a laser velocity transducer. Its unit is whatever file_description.txt lists for that file (mm/s or m/s; the file lists 'm/s' for the "
         "files recorded at the 0-1 m/s laser setting); channel_units carries it per recording. Do not compare amplitudes across recordings with different units or ranges without checking them.",
         "trial counts recordings of a participant in the order: DBS on/med on, on/off, off/on, off/off, then 15, 30, 45, 60 min after stopping; this is the file listing order, not the recording clock order."],
    unit_note="recording (about 60 s of index-finger rest tremor velocity at 100 Hz)",
    differences=["The paper (Thalamus & Related Systems, not open access) could not be read, so no tremor amplitude or frequency result of it is reproduced; checks are against the counts "
                 "the PhysioNet record page and file_description.txt state.",
                 "Subject v3 is described (and listed in the 16) but has no recording file in the deposit, so 15 subjects have rows and v3 is not in participants.csv.",
                 "The PhysioNet README says 54 recordings in the DBS x medication conditions and 8 STN subjects, its web page says 55 recordings and 6 STN; the data have 55 recordings, 6 STN subjects."])
part_cols = [(c, "text", "", d) for c, d in (("stim_target", "Stimulation target: Vim, GPi or STN."), ("laterality", "Bilateral (Bi) or unilateral (Uni) stimulation."),
             ("effective_freq_hz", "Frequency of effective stimulation, Hz."), ("ineffective_freq_hz", "Frequency of so-called ineffective stimulation, Hz."),
             ("intensity_v", "Stimulation intensity, V."), ("pulse_width_us", "Pulse width, microseconds."), ("mode", "Cont = continuous, Cycl = cyclic stimulation."),
             ("contacts", "Stimulating contacts in order from proximal to distal (- negative, + positive, . not stimulated)."),
             ("total_daily_medication_mg", "Total daily medication dose (morning, noon, evening), mg."), ("single_dose_150pct_mg", "Dose taken before the 'medication on' recordings (150 percent of the morning dose), mg."))]
pp = pd.DataFrame([dict(person=s, stim_target=v["STIM TARGET"], laterality=v["BI/UNI-LATERAL"], effective_freq_hz=v["EFF FREQ (Hz)"], ineffective_freq_hz=v["INEFF FREQ (Hz)"],
                        intensity_v=v["INTENSITY (V)"], pulse_width_us=v["PULSE WIDTH (mu sec)"], mode=v["MODE (CONT/CYCL)"], contacts=v["STIM CONTACTS"],
                        total_daily_medication_mg=v["TOT DAILY MED (mg)"], single_dose_150pct_mg=v["150% SINGLE DOSE (mg)"]) for s, v in subj.items() if s in set(df.subject)])
df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"] + ["person"]) if False else df
stdlib.release(ID, MODULE, df, meta, ck, participants=pp, participant_columns=part_cols, extra_files=extra)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))
