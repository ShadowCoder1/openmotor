"""Voisard et al. (2025) Sci Data 12:1674: clinical gait signals from four inertial sensors (head, lower back, both feet, 100 Hz) of 260 participants
(healthy, orthopedic and neurological cohorts) walking a short straight path, a U-turn and back; 1,356 trials, 'timeseries' module, one row per trial.

Raw file (untrusted data, read only): raw_extra/Voisard2025_SciData/dataset.zip from https://doi.org/10.6084/m9.figshare.28806086.v1 (CC BY 4.0).
Read from it: for every trial dataset/data/<group>/<cohort>/<subject>/<trial>/ the *_processed_data.txt (the four sensors' preprocessed signals) and the
*_meta.json (participant and trial metadata, gait events, U-turn boundaries). The deposit's Python files, the PNG plots and the four raw sensor files
per trial are not read or carried (see README). Nothing from the deposit is run.
Run from the site folder: python3 -I tools/std/convert/Voisard2025_SciData.py
"""
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Voisard2025_SciData"
MODULE = "timeseries"
RAW = stdlib.raw_dir(ID)
TMP = Path("tmp_" + ID)
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "samples").mkdir(parents=True)

z = zipfile.ZipFile(RAW / "dataset.zip")
names = {i.filename for i in z.infolist() if not i.is_dir()}
metas = sorted(n for n in names if n.endswith("_meta.json"))
assert len(metas) == 1356, len(metas)
M = []
for mp in metas:
    m = json.loads(z.read(mp))
    base = mp[: -len("_meta.json")]
    assert base + "_processed_data.txt" in names
    M.append((base, m))
# order: participant by first appearance sorted by cohort/subject, then session and trial
M.sort(key=lambda bm: (bm[1]["pathologyKey"], int(bm[1]["subject"].split("_")[1]), bm[1]["session"], bm[1]["trial"]))

CH_UNIT = {"Acc": "m/s2", "FreeAcc": "m/s2", "Gyr": "rad/s"}
HERE_COLS = None
POP = {"healthy subject": "healthy", "parkinson disease": "Parkinson's disease", "cerebrovascular accident": "stroke (cerebrovascular accident)",
       "toxic peripheral neuropathy": "chemotherapy-induced peripheral neuropathy", "radiation-induced leukoencephalopathy": "radiation-induced leukoencephalopathy",
       "knee osteoarthritis": "knee osteoarthritis", "hip osteoarthritis": "hip osteoarthritis", "anterior cruciate ligament injury": "anterior cruciate ligament injury"}
LAT = {"right": "right", "left": "left", "ambidextrous": "ambi"}
rows, seq, stat = [], {}, dict(nan_cells=0, counter_ok=True, ev_in_range=True, strides=[], n_samples=0, sizes=[])
for base, m in M:
    d = pd.read_csv(io.BytesIO(z.read(base + "_processed_data.txt")), sep="\t")
    freq = float(m["freq"])
    assert freq == 100.0
    cnt = d["PacketCounter"]
    stat["counter_ok"] &= bool(np.allclose(cnt.dropna().values, cnt.dropna().index.values, atol=1e-6))
    chan = [c for c in d.columns if c != "PacketCounter"]
    assert len(chan) == 36
    out = d[chan].copy()
    out.insert(0, "time_s", np.arange(len(d)) / freq)
    stat["nan_cells"] += int(out[chan].isna().sum().sum())
    subj = m["subject"]
    seq[subj] = seq.get(subj, 0) + 1
    k = seq[subj]
    fn = f"{base.split('/')[-1]}.csv"
    out.to_csv(TMP / "samples" / fn, index=False, lineterminator="\r\n", float_format="%.8g")
    ev = []
    for side in ("left", "right"):
        evs = m[f"{side}GaitEvents"]
        for j, (to, hs) in enumerate(evs, 1):
            ev += [(to, f"{side}_toe_off_{j}"), (hs, f"{side}_heel_strike_{j}")]
            stat["ev_in_range"] &= (0 <= to <= hs < len(d))
    ub = m["uturnBoundaries"]
    ev += [(ub[0], "uturn_start"), (ub[1], "uturn_end")]
    events = ";".join(f"{nm}={s / freq:.2f}" for s, nm in sorted(ev))
    # stride times from the heel strikes of each leg (end of one swing phase to the end of the next)
    for side in ("left", "right"):
        hs = [e[1] for e in m[f"{side}GaitEvents"]]
        stat["strides"].append((m["pathologyKey"], subj, base, np.diff(hs) / freq))
    info = [f"cohort={m['pathologyKey']}", f"protocol={m['protocol']}", f"sensor={m['sensor']}", f"height_m={m['height']}", f"weight_kg={m['weight']}",
            f"bmi={m['BMI']}", f"days_since_first_session={m['daysSinceFirstSession']}", f"trial_among_sessions={m['trial']}",
            f"visual_gait_assessment_0to4={m['visualGaitAssessment']}"]
    if m["TUG"] not in (None, "Not evaluated"):
        info.append(f"TUG_s={m['TUG']}")
    if m["evaluationScoreName"]:
        info.append(f"{m['evaluationScoreName'].replace(' ', '_')}={m['evaluationScoreValue']}")
    if m["clinicalDeficitSide"]:
        info.append(f"clinical_deficit_side={m['clinicalDeficitSide']}")
    info.append("sample_to_time=sample_index/100 (events are given in the authors' sample numbers = row index of the processed file)")
    rows.append(dict(
        dataset=ID, study="Voisard et al. 2025 Sci Data", experiment="", group=m["pathologyKey"], subject=subj, person=subj, subject_excluded_by_authors=False,
        age=m["age"], sex=m["gender"], handedness=LAT.get(m["laterality"], ""), population=POP[m["pathology"]], setting="clinic",
        device="4 inertial sensors (head, lower back L4/L5, dorsal face of each foot), 100 Hz: XSens MTw Awinda or Technoconcept I4 Motion", session=m["session"],
        block=f"visit {m['session']}", trial=k, phase="test", time_s=np.nan, task="walking", unit="recording",
        condition="walk at comfortable speed: straight path, U-turn, back", pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=k, samples_file=f"samples/{fn}", signal_kind="inertial sensors on head (HE), lower back (LB), left foot (LF), right foot (RF): acceleration, acceleration minus gravity, angular velocity",
        channels=";".join(chan), channel_units=";".join(CH_UNIT[c.split("_")[1]] for c in chan), sample_rate_hz=freq, n_samples=len(d), duration_s=len(d) / freq,
        events=events, processing="preprocessed by the authors: gaps under 1.5 s filled by linear interpolation, static acceleration (first 2 s or more) subtracted, gyroscope offset corrected, 8th-order Butterworth low-pass at 14 Hz (paper, Data processing); empty cells are missing samples",
        trial_info=";".join(info)))
    stat["sizes"].append(len(d))
df = pd.DataFrame(rows)
mm = pd.DataFrame([m for _, m in M])
pp = mm.drop_duplicates("subject")  # first record of each participant

ck = stdlib.Checks()
ck.published("participants", mm.subject.nunique(), 260, 0, "Abstract / Background (260 participants)")
ck.published("trials", len(mm), 1356, 0, "Abstract (1356 gait trials) and Data Records (7 files for each of the 1356 experiments)")
for key, ns, nt in (("HS", 73, 360), ("HOA", 15, 74), ("KOA", 18, 78), ("ACL", 11, 60), ("CVA", 49, 128), ("PD", 24, 160), ("CIPN", 19, 98), ("RIL", 51, 398)):
    ck.published(f"{key} participants", (pp.pathologyKey == key).sum(), ns, 0, "Technical Validation, Tables 6-8")
    ck.published(f"{key} trials", (mm.pathologyKey == key).sum(), nt, 0, "Technical Validation, Tables 6-8")
ck.published("healthy participants, males", ((pp.pathologyKey == "HS") & (pp.gender == "M")).sum(), 41, 0, "Methods Participants / Table 6")
ck.published("healthy participants, females", ((pp.pathologyKey == "HS") & (pp.gender == "F")).sum(), 32, 0, "Methods Participants / Table 6")
ck.published("orthopedic participants, males", ((pp.group == "ortho") & (pp.gender == "M")).sum(), 21, 0, "Methods Participants (21 males)")
ck.published("neurological participants, males", ((pp.group == "neuro") & (pp.gender == "M")).sum(), 92, 0, "Methods Participants (92 males, 51 females)")
ck.published("neurological participants, females", ((pp.group == "neuro") & (pp.gender == "F")).sum(), 51, 0, "Methods Participants (92 males, 51 females)")
ck.published("healthy trials per participant, median", mm[mm.pathologyKey == "HS"].groupby("subject").size().median(), 5, 0, "Table 6 (5 [1, 29])")
ck.published("healthy trials per participant, maximum", mm[mm.pathologyKey == "HS"].groupby("subject").size().max(), 29, 0, "Table 6 (5 [1, 29])")
ck.published("recordings with Technoconcept sensors", (mm.sensor == "TechnoConcept").sum(), 787, 0, "Table 9")
ck.published("recordings with XSens sensors", (mm.sensor == "MTw Awinda XSens").sum(), 569, 0, "Table 9")
visits = mm.drop_duplicates(["subject", "session"])
ck.published("visits with a TUG measurement", (visits.TUG != "Not evaluated").sum(), 127, 0, "Technical Validation (conducted during 127 visits (39%))")
ck.published("healthy participants right-handed", ((pp.pathologyKey == "HS") & (pp.laterality == "right")).sum(), 68, 0, "Table 6 (Laterality R/L 68/5)")
ck.published("healthy participants left-handed", ((pp.pathologyKey == "HS") & (pp.laterality == "left")).sum(), 5, 0, "Table 6 (Laterality R/L 68/5)")
ck.published("sampling rate (Hz)", mm.freq.mean(), 100, 0, "Methods (sampling rate of 100 Hz)")
ck.published("visits without a visual gait assessment", len(visits[visits.visualGaitAssessment == "Not evaluated"]), 13, 0,
             "Technical Validation (visual gait assessment not documented in 13 cases)")
hs_stride = np.mean([np.concatenate([s for c, su, b, s in stat["strides"] if c == "HS" and b == base]).mean() for base in {b for c, su, b, s in stat["strides"] if c == "HS"}])
ck.internal("every trial has a sample file with 37 columns (time_s + 36 channels)", len(df) == 1356)
ck.internal("event samples lie inside the recording and toe-off precedes heel strike", stat["ev_in_range"])
ck.internal("PacketCounter equals the row index (within 1e-6) wherever it is present", stat["counter_ok"])
ck.internal("total samples", sum(stat["sizes"]) == int(df.n_samples.sum()), f"{sum(stat['sizes'])} samples = {sum(stat['sizes']) / 100 / 3600:.2f} h")
ck.internal("missing sample cells are left empty", True, f"{stat['nan_cells']} empty cells")
dif = ["The paper reports 331 visits; counting distinct (participant, session number) pairs in the metadata gives %d, so that number is not used as a check (how the paper counted visits is not stated)." % len(visits),
       "Table 10 gives a mean stride time of 1.12 s (SD 0.09) for healthy participants; the mean over healthy trials of the mean time between consecutive heel strikes of the same leg in the authors' gait events is %.3f s, so it is not used as a check (the paper computes it with its own earlier method)." % hs_stride,
       "Table 6-8 give age, height, weight and BMI as mean (SD) without saying whether they are per participant or per trial; they are not used as checks.",
       "The deposit's processed files hold empty cells where a sensor had no data at the end of a recording (%d empty cells in all); they are kept empty." % stat["nan_cells"],
       "The deposit's four raw sensor files per trial (acceleration, angular velocity and magnetic field of each sensor before the authors' centring and filtering, about 1.9 GB of text) are not included, to keep the download small."]
meta = dict(
    paper=dict(authors="Voisard C, Barrois R, l'Escalopier N, Vayatis N, Vidal PP, Yelnik A, Ricard D, Oudre L", year=2025,
               title="A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts", venue="Scientific Data 12(1):1674",
               doi="10.1038/s41597-025-05959-w"),
    lab="Oudre / Vidal (Centre Borelli, Universite Paris Cite / ENS Paris-Saclay)", data_url="https://doi.org/10.6084/m9.figshare.28806086.v1", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("One row per trial (the deposit's 1,356 experiments), in cohort, participant, visit and trial order. The preprocessed signals of the four sensors (one tab-separated file per trial) are "
             "carried as a CSV of 36 channels with a time_s column (PacketCounter, which equals the sample number from 0, replaced by sample number / 100), values to 8 significant digits. "
             "Metadata (age, sex, laterality, cohort, sensor, protocol, height, weight, BMI, days since first session, TUG, visual gait assessment, clinical score and deficit side) come from "
             "each trial's JSON file; the authors' gait events and U-turn boundaries (sample numbers) are written in seconds in the events column. Participant labels are replaced by codes."),
    how=["One row is one trial: the participant walked at a comfortable speed along a straight path (5.5 to 10 m depending on the site), made a U-turn and walked back, wearing four inertial "
         "sensors (head, lower back, and the dorsal face of each foot). The samples file has the authors' preprocessed signals of the four sensors: acceleration (Acc, m/s2), acceleration minus gravity "
         "(FreeAcc, m/s2) and angular velocity (Gyr, rad/s) along the sensor axes, 100 Hz.",
         "events gives the authors' automatic gait events in seconds from the start of the recording: for each leg the toe-off and the heel strike that bound each swing phase (left_toe_off_1, "
         "left_heel_strike_1, ...) and the start and end of the U-turn.",
         "sex is the authors' gender field (M or F); handedness is their laterality field (ambidextrous written as ambi, empty when not recorded); group is the cohort code (HS healthy, HOA hip "
         "osteoarthritis, KOA knee osteoarthritis, ACL anterior cruciate ligament injury, CVA stroke, PD Parkinson's disease, CIPN chemotherapy-induced peripheral neuropathy, RIL "
         "radiation-induced leukoencephalopathy). Clinical scores (trial_info) are the cohort-specific score of the visit: WOMAC (hip and knee osteoarthritis), IKDC (anterior cruciate ligament), FMA-LE "
         "(stroke), UPDRS III (Parkinson's disease), TNSc (neuropathy), mSS (leukoencephalopathy). Age is the age at the trial's visit."],
    unit_note="trial (recording)", differences=dif)
rec = stdlib.release(ID, MODULE, df, meta, ck, extra_files=[(f"samples/{p.name}", str(p)) for p in sorted((TMP / "samples").iterdir())])
shutil.rmtree(TMP, ignore_errors=True)
