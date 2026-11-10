"""Skaramagkas, Andrikopoulos, Kefalopoulou & Polychronopoulos (2021), Signals 2(2):201-224: arm tremor accelerometer database.
37 people (2 essential tremor, 8 Parkinson's disease, 27 without a tremor diagnosis), four arm tasks, accelerometers (3 axes, 62.5 Hz) on the
index finger, thumb, metacarpal area and wrist of the strong hand. One row per recording (person x task x sensor position).
Deposit: https://zenodo.org/records/4772130 (CC BY 4.0): database.mat (cell of 37 structs holding MATLAB tables and string objects, read without
MATLAB with Skaramagkas2021_Signals_mcos.py) and readme.txt.
Run from the site folder: python3 -I tools/std/convert/Skaramagkas2021_Signals.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Skaramagkas2021_Signals_mcos import MatObjects  # noqa: E402

DS = "Skaramagkas2021_Signals"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
o = MatObjects(raw / "database.mat")
data = o.top["data"]
FS = 62.5  # Hz, readme


def sval(v):  # MATLAB string object -> list of str
    oid = int(v[0]["_ObjectMetadata"].ravel()[-2])
    sid = int(o.objects[oid][3])
    a = np.asarray(o.cells[int(o.saveobj[2 + 4 * (sid - 1) + 3]) + 2]).ravel().astype("<u8")
    nd = int(a[1])
    n = int(np.prod([int(x) for x in a[2:2 + nd]]))
    lens = a[2 + nd:2 + nd + n].astype(int)
    u = np.frombuffer(a[2 + nd + n:].tobytes(), dtype="<u2")
    out, p = [], 0
    for L in lens:
        out.append("".join(chr(c) for c in u[p:p + L]))
        p += L
    return out


def AGE(d):
    try:
        return float(d["age"])
    except (KeyError, ValueError, TypeError):
        return float("nan")


SEQ = ["rest", "postural", "free_motion", "motion_with_object"]
SEQ_TXT = {"rest": "rest (hand at resting pose)", "postural": "postural (hand held in postural position)",
           "free_motion": "free motion (hand touches the table, then the nose; oscillatory movement)",
           "motion_with_object": "motion while holding a bottle (same movement as free motion)"}
LOC = ["index", "thumb", "metacarpal", "wrist"]
POPN = {"Essential tremor": "essential tremor", "Parkinson's tremor": "Parkinson's disease", "No diagnosed tremor": "no diagnosed tremor"}
rows, files, part = [], [], []
for i in range(data.shape[1]):
    c = data[0, i]
    info = c["info"][0, 0]
    d = {}
    for n in info.dtype.names:
        v = info[n][0, 0]
        d[n] = sval(v)[0] if v.dtype.names else np.squeeze(v)
    pid = f"P{i + 1:02d}"
    part.append(dict(person=pid, age_years=AGE(d), hand_with_sensors=d.get("hand", ""), medication=d.get("med", ""),
                     tremor_years=d.get("tremor", ""),
                     disease_years=d.get("disease", ""), status=d["status"]))
    rec = c["recordings"][0, 0]
    k = 0
    for s in SEQ:
        rs = rec[s][0, 0]
        for l in LOC:
            k += 1
            fld = "Meatcarpal" if (l == "metacarpal" and "Meatcarpal" in rs.dtype.names) else l  # one person's rest field is misspelled in the deposit
            t = o.variable(rs[fld][0, 0])
            cols = [np.asarray(x, dtype=float).ravel() for x in t["data"]]
            n = len(cols[0])
            assert len(cols) == 3 and all(len(x) == n for x in cols) and t["nrows"].item() == n
            fn = f"samples/{DS}_{pid}_{s}_{l}.csv"
            rows.append(dict(
                dataset=DS, study="Skaramagkas et al. 2021 Signals", experiment="", group=POPN[d["status"]], subject=pid, person=pid,
                subject_excluded_by_authors=False, age=AGE(d), sex={"male": "M", "female": "F"}.get(d.get("sex"), ""), handedness="",
                time_s="", population=POPN[d["status"]], setting="clinic", device="accelerometers (3 axes, 62.5 Hz) on the arm of the strong hand",
                session=1, block=s, trial=k, phase="test", task="tremor recording", unit="recording",
                condition=f"{SEQ_TXT[s]}; sensor on {l}", pert_type="none", pert_value=0.0, pert_unit="", feedback="normal",
                excluded_by_authors=False, recording=k, samples_file=fn,
                signal_kind=f"accelerometer, three axes, sensor on the {l} of the {d.get('hand', 'strong')} (strong) hand",
                channels="x;y;z", channel_units="m/s^2", sample_rate_hz=FS, n_samples=n, duration_s=n / FS, events="",
                processing="none stated (as shared in the MATLAB database); the first sample of each recording is not in the table (see README)",
                trial_info=f"task={s};sensor_position={l};hand={d.get('hand', '')};medication={d.get('med', 'not applicable')}"))
            files.append((fn, np.column_stack(cols)))
df = pd.DataFrame(rows)


def extra():
    for fn, a in files:
        yield fn, "time_s,x,y,z\n" + "".join(f"{j / FS:.4f}," + ",".join(format(x, ".7g") for x in a[j]) + "\n" for j in range(len(a)))


pdf = pd.DataFrame(part)
st = pdf.status.value_counts()
ck.published("people with essential tremor", st["Essential tremor"], 2, 0, "readme.txt Database Structure (data{1}-data{2})")
ck.published("people with Parkinson's disease", st["Parkinson's tremor"], 8, 0, "readme.txt Database Structure (data{3}-data{10})")
ck.published("people with no diagnosed tremor", st["No diagnosed tremor"], 27, 0, "readme.txt Database Structure (data{11}-data{37})")
ck.published("people in the database", len(pdf), 37, 0, "readme.txt Database Structure (1x37 cell vector)")
ck.published("recordings per person (4 tasks x 4 sensor positions)", df.groupby("person").size().iloc[0], 16, 0, "readme.txt (four sequences, four positions)")
ck.internal("one row per person x task x position", len(df) == 37 * 16)
ck.internal("all recordings have samples", bool((df.n_samples > 100).all()))
ck.internal("sample files unique", df.samples_file.is_unique)

meta = dict(
    paper=dict(authors="Skaramagkas V, Andrikopoulos G, Kefalopoulou Z, Polychronopoulos P", year=2021,
               title="A Study on the Essential and Parkinson's Arm Tremor Classification", venue="Signals 2(2):201-224", doi="10.3390/signals2020016"),
    lab="Skaramagkas / Polychronopoulos (University of Patras)", data_url="https://zenodo.org/records/4772130", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept every recording of the deposit's MATLAB database (37 people x 4 tasks x 4 sensor positions) as one CSV of x, y, z acceleration with a "
             "time column added (time = sample index / 62.5 Hz, values to 7 significant digits); participants are coded. Nothing was filtered."),
    how=["One row is one accelerometer recording: one person, one task (rest, postural position, free motion touching the table then the nose, "
         "or the same motion holding a bottle) and one sensor position (index finger, thumb, metacarpal area or wrist) of the strong hand.",
         "Channels x, y, z are the three accelerometer axes in m/s^2 at 62.5 Hz (readme). participants.csv holds age, the hand that carried the sensors, "
         "medication state (ON or OFF, people with Parkinson's disease and essential tremor only), years since the tremor and the condition appeared, and the diagnosis.",
         "The group counts (2 essential tremor, 8 Parkinson's disease, 27 without tremor diagnosis) are from the deposit's readme; the paper's full text could not be "
         "opened (the publisher blocks automated access), so no value from the paper itself is checked."],
    unit_note="accelerometer recording (samples in samples/)",
    differences=["One participant's rest recording has its metacarpal field spelled 'Meatcarpal' in the deposit; it is read as the metacarpal sensor.",
                 "In the shared tables the first sample of each recording appears to have been read as the column names (the variable descriptions read 'Original "
                 "column heading: ...'), so each table starts at the second sample and the first sample is not included here."],
)
pcols = [("age_years", "number", "years", "Age."), ("hand_with_sensors", "text", "left/right", "Strong hand that carried the accelerometers."),
         ("medication", "text", "ON/OFF", "Medication state (ON: medication in the last 12 hours; OFF: none); empty for people without a diagnosis."),
         ("tremor_years", "text", "years (as written)", "Time since the tremor appeared, as written in the deposit (mostly years; some entries carry other units, e.g. months)."),
         ("disease_years", "text", "years (as written)", "Time since the disease or condition appeared, as written in the deposit (people with a diagnosis)."),
         ("status", "text", "", "Essential tremor, Parkinson's tremor or No diagnosed tremor.")]
stdlib.release(DS, "timeseries", df, meta, ck, participants=part, participant_columns=pcols, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
