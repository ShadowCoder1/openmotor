"""Borzi et al. (2026), Sci Data 13:305: FoG-STAR, four-IMU recordings of 22 people with Parkinson's disease doing seven mobility tasks,
with freezing-of-gait (FoG) and activity labels. One row per recording (participant, session, task).

Run from the site folder: python3 -I tools/std/convert/Borzi2026_SciData.py
Raw files (untrusted data, read only) in raw_extra/Borzi2026_SciData/ from https://zenodo.org/records/17838806:
  sensor_data.csv, clinical_data.csv, README.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Borzi2026_SciData"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

d = pd.read_csv(RAW / "sensor_data.csv")
cl = pd.read_csv(RAW / "clinical_data.csv").rename(columns={"fes-i": "fes_i", "pdq8": "pdq_8"})
KEY = ["subjectID", "sessionID", "taskID"]
SENS = [c for c in d.columns if c.split("_")[0] in ("ankleL", "ankleR", "back", "wrist")]
LAB = ["activity", "fog", "fog_severity"]
assert len(SENS) == 24
# subject codes are assigned in order of first appearance in the file (participant 9 is recorded last)
first = d.subjectID.drop_duplicates().tolist()
assert sorted(first) == list(range(1, 23)), first
POS = {s: i + 1 for i, s in enumerate(first)}

TASK = {1: "timed up and go", 2: "stand still 1 min", 3: "walk back and forth 10 m", 4: "walk back and forth 10 m through a doorway",
        5: "walk back and forth 10 m carrying a glass of water", 6: "walk back and forth 10 m counting backward by 7",
        7: "360 degree turn"}
ACT = "1 walking, 2 sit, 3 stand, 4 sit-to-stand, 5 stand-to-sit, 6 turn right, 7 turn left, 0 not labelled"
UNITS = ["g"] * 3 + ["deg/s"] * 3
CH = SENS + LAB
CU = ";".join((UNITS * 4) + ["activity code", "0/1", "severity code 0-3"])

def runs(x):
    out, i, n = [], 0, len(x)
    while i < n:
        if x[i] != 0:
            j = i
            while j < n and x[j] == x[i]:
                j += 1
            out.append((i, j, x[i]))
            i = j
        else:
            i += 1
    return out

cli = cl.set_index("subjectID")
rows, extra = [], []
n_fog = 0
sev_dur = {1: [], 2: [], 3: []}
tr = {}
for (sid, ses, tk), g in d.groupby(KEY, sort=False):
    tr[sid] = tr.get(sid, 0) + 1
    code = f"{ID}_S{POS[sid]:04d}"
    fn = f"samples/{code}_session{ses}_task{tk}.csv"
    t0 = g.timestamp.iloc[0]
    out = g[["timestamp"] + CH].copy()
    out["timestamp"] = out.timestamp - t0
    out = out.rename(columns={"timestamp": "time_s"})
    extra.append((fn, out.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
    ev = []
    for k, (a, b, _) in enumerate(runs(g.fog.values), 1):
        ev += [f"fog_start_{k}={a / 60:.4f}", f"fog_end_{k}={b / 60:.4f}"]
        n_fog += 1
    for a, b, v in runs(g.fog_severity.values):
        sev_dur[v].append((b - a) / 60)
    c = cli.loc[sid]
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group="", subject=str(sid), person=str(sid), subject_excluded_by_authors=False,
        age=c.age, sex=c.gender, handedness="", population="Parkinson's disease with freezing of gait (off medication)",
        setting="clinic", device="four Nordic Thingy:52 IMUs (left ankle, right ankle, lower back, wrist), 60 Hz",
        session=int(ses), block=f"task {int(tk)}", trial=tr[sid], phase="test", time_s="", task="walking", unit="recording",
        condition=TASK[int(tk)], pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=tr[sid], samples_file=fn, signal_kind="accelerometer and gyroscope of four IMUs, with activity, FoG and FoG-severity labels",
        channels=";".join(CH), channel_units=CU, sample_rate_hz=60, n_samples=len(g), duration_s=len(g) / 60.0,
        events=";".join(ev),
        processing="IMU streams synchronised offline by the authors; activity and FoG labels from two neurologists' video annotation (10 fps)",
        trial_info=f"task_id={int(tk)};session_id={int(ses)}"))
df = pd.DataFrame(rows)

part = cl.copy()
part["person"] = part.subjectID.astype(str)
part = part.drop(columns=["subjectID", "age", "gender"])
pcols = [("disease_duration", "number", "years", "Years since PD diagnosis (empty for one participant)."),
         ("hoehn_yahr", "number", "stage", "Hoehn and Yahr stage (empty for one participant)."),
         ("updrs_iii", "number", "points", "MDS-UPDRS part III score."),
         ("fog_q", "number", "points", "New Freezing of Gait Questionnaire score."),
         ("moca", "number", "points", "Montreal Cognitive Assessment (empty for one participant)."),
         ("fes_i", "number", "points", "Falls Efficacy Scale International (empty for one participant)."),
         ("pdq_8", "number", "points", "Parkinson's Disease Questionnaire-8.")]
part = part.rename(columns={"h_y": "hoehn_yahr"})

# ---- published numbers (Data Records, Technical tables 3 and 9)
P = checks.published
P("participants", df.person.nunique(), 22, 0, "Abstract / Methods, Participants")
P("samples in sensor_data.csv", df.n_samples.sum(), 329027, 0, "Data Records, Sensor data")
P("FoG episodes annotated", n_fog, 101, 0, "Abstract")
P("sensor channels", len(SENS), 24, 0, "Data Records, Sensor data")
P("mean recording time per participant (min)", df.groupby("person").duration_s.sum().mean() / 60, 4.2, 0.05, "Data Overview, Fig 4a")
P("SD of recording time per participant (min)", df.groupby("person").duration_s.sum().std() / 60, 1.8, 0.05, "Data Overview, Fig 4a")
for v, nm, ev_, mean_, med, mn, mx, p95 in ((1, "shuffling", 33, 3.02, 2.40, 0.6, 9.9, 7.04), (2, "trembling", 40, 5.82, 4.90, 0.2, 24.1, 12.40),
                                         (3, "akinesia", 37, 19.98, 12.40, 1.8, 108.68, 71.65)):
    a = np.array(sev_dur[v])
    P(f"{nm}: events", len(a), ev_, 0, "Table 9")
    if v != 1:
        P(f"{nm}: mean duration (s)", a.mean(), mean_, 0.005, "Table 9")
    P(f"{nm}: median duration (s)", np.median(a), med, 0.005, "Table 9")
    P(f"{nm}: min duration (s)", a.min(), mn, 0.005, "Table 9")
    P(f"{nm}: max duration (s)", a.max(), mx, 0.005, "Table 9")
for nm, col, m_, s_ in (("disease duration", "disease_duration", 10.1, 6.8), ("Hoehn and Yahr", "h_y", 2.7, 0.8), ("FOG-Q", "fog_q", 18.3, 5.4),
                        ("MoCA", "moca", 21.3, 3.9), ("PDQ-8", "pdq_8", 14.4, 6.9)):
    P(f"{nm}: mean", cl[col].mean(), m_, 0.05, "Table 3")
    P(f"{nm}: SD", cl[col].std(), s_, 0.05, "Table 3")
P("UPDRS-III: SD", cl.updrs_iii.std(), 14.6, 0.05, "Table 3")
P("age: SD", cl.age.std(), 9.3, 0.05, "Table 3")
checks.internal("106 recordings", len(df) == 106, str(len(df)))
checks.internal("sample times are 1/60 s apart", all(abs(float(x) - 1 / 60) < 1e-6 for x in [d.timestamp.diff().dropna().min(), d.timestamp.diff().dropna().max()]))
checks.internal("no sample lost", len(d) == df.n_samples.sum())

meta = dict(
    paper=dict(authors="Borzì L, Demrozi F, Bacchin RA, Turetta C, Tebaldi M, Sigcha L, Zolfaghari S, Rinaldi D, Fazzina G, Balestro G, Picelli A, Pravadelli G, Olmo G, Tamburin S, Lopiano L, Artusi CA",
               year=2026, title="A multi-level annotated sensor dataset of gait freezing manifestations and severity in Parkinson's disease",
               venue="Sci Data 13:305", doi="10.1038/s41597-026-06645-1"),
    lab="Olmo (Politecnico di Torino) / Demrozi (University of Stavanger) / Tamburin (University of Verona)",
    data_url="https://zenodo.org/records/17838806", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("sensor_data.csv was split into one samples CSV per recording (participant, session, task); time starts at 0 in each recording "
             "(the deposit's timestamp runs on through the whole file). The clinical scores are in participants.csv; age and sex are in the main table. "
             "Nothing was filtered or resampled; the Python notebook and environment file in the deposit are not used."),
    how=["One row is one recording of one participant doing one of seven tasks (timed up and go, standing 1 min, three 10 m back-and-forth walks with a doorway, a glass of water or "
         "counting backward, and a 360 degree turn), after a 12 h levodopa washout. Some participants have a second or third session for one task when synchronisation failed; "
         "not every participant has every task.",
         "Samples (samples/<subject>_session<n>_task<n>.csv) are at 60 Hz: time_s, then accelerometer (g) and gyroscope (deg/s) x, y, z of the left ankle, right ankle, lower back and wrist, "
         "then the authors' labels from video: activity (" + ACT + "), fog (1 = freezing of gait) and fog_severity (0 none, 1 shuffling, 2 trembling in place, 3 akinesia). "
         "Sensor channels are empty where the authors' file has no value (up to a few percent of samples).",
         "events lists each freezing episode as fog_start_k and fog_end_k in seconds from the start of the recording, where an episode is a run of samples with fog = 1.",
         "The deposit's timestamp column is documented as milliseconds, but its values step by 0.016667, which is 1/60 s, so they are seconds; time_s here is in seconds."],
    differences=["The paper's Table 3 reports 13 men and 9 women; the deposit's clinical file has 12 M and 10 F. Mean age 70.9, UPDRS-III 38.9 and FES-I 36.9 (and the SD of FES-I, 26.9) are printed in the paper "
                 "more than 0.05 away from what the deposit's clinical file gives (70.96, 38.96, 36.95, SD 26.95), so those numbers are not used as checks. The mean duration of shuffling episodes is 3.03 s here against 3.02 s in Table 9."],
    unit_note="recording (one participant, session and task)",
)
stdlib.release(ID, "timeseries", df, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
