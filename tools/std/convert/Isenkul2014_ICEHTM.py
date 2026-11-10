"""Isenkul, Sakar & Kursun (2014), ICEHTM-2014: spiral drawing and pen-hold tests on a Wacom Cintiq 12WX graphics tablet, 62 people with
Parkinson's disease and 15 healthy controls (UCI Machine Learning Repository dataset 395, doi 10.24432/C5Q01S).

Raw file (untrusted data, read only): raw_extra/Isenkul2014_ICEHTM/uci395.zip. Used: hw_dataset/control, hw_dataset/parkinson and
new_dataset/parkinson (.txt, semicolon-separated: X;Y;Z;Pressure;GripAngle;Timestamp;TestID, one file per person, the three tests concatenated)
and the two readme.txt files (read as text). NOT carried: hw_drawings/*.png (images of the drawings) and the original file names (they contain
what look like dates).
Run from the site folder: python3 -I tools/std/convert/Isenkul2014_ICEHTM.py
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Isenkul2014_ICEHTM"
RD = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RD / "SOURCES.json"))}
checks.internal("raw zip matches the downloaded hash", stdlib.sha256(RD / "uci395.zip") == src["uci395.zip"]["sha256"])
z = zipfile.ZipFile(RD / "uci395.zip")

TEST = {0: ("static spiral", "Static Spiral Test (SST): the participant retraces a spiral shown on the tablet"),
        1: ("dynamic spiral", "Dynamic Spiral Test (DST): the spiral blinks, so the participant must continue it from memory"),
        2: ("pen hold on a point", "Stability Test on Certain Point (STCP): hold the pen over a red point in the middle of the screen without touching the screen (the readme calls it 'Circular Motion Test, draw circles around the red point')")}
CH = "x;y;z;pressure;grip_angle"
UNITS = "tablet units (not documented);tablet units (not documented);tablet units (not documented);0-1023 scale;tablet units (not documented)"

names = sorted(i.filename for i in z.infolist() if i.filename.endswith(".txt") and "readme" not in i.filename.lower())
order = ([n for n in names if n.startswith("hw_dataset/control/")] + [n for n in names if n.startswith("hw_dataset/parkinson/")]
         + [n for n in names if n.startswith("new_dataset/parkinson/")])
rows, extra = [], []
n_samples = 0
nonmono = 0
persons = []
for pi, n in enumerate(order, start=1):
    folder = n.split("/")[0]
    pd_ = "/parkinson/" in n
    lines = [l for l in z.read(n).decode("latin1").splitlines() if l.strip()]
    d = pd.read_csv(io.StringIO("\n".join(lines)), sep=";", header=None, names=["x", "y", "z", "pressure", "grip_angle", "t", "test"])
    assert d.notna().all().all() and d.shape[1] == 7
    # contiguous runs of the same test id, in file order
    run = (d.test != d.test.shift()).cumsum()
    code = f"{'P' if pd_ else 'C'}{pi:03d}"
    persons.append(code)
    for k, (_, g) in enumerate(d.groupby(run, sort=True), start=1):
        t = int(g.test.iloc[0])
        tt = (g.t - g.t.iloc[0]).to_numpy(dtype=float) / 1000.0
        if not (np.diff(tt) >= 0).all():
            nonmono += 1
        dt = np.median(np.diff(tt)[np.diff(tt) > 0]) if len(tt) > 2 else np.nan
        out = pd.DataFrame({"time_s": tt, "x": g.x.to_numpy(), "y": g.y.to_numpy(), "z": g.z.to_numpy(),
                            "pressure": g.pressure.to_numpy(), "grip_angle": g.grip_angle.to_numpy()})
        fn = f"samples/{code}_t{t}_{k}.csv"
        extra.append((fn, out.to_csv(index=False, float_format="%.6g", lineterminator="\r\n")))
        n_samples += len(g)
        rows.append(dict(
            dataset=ID, study=ID, experiment=TEST[t][0], group="Parkinson's disease" if pd_ else "control", subject=code, person=code,
            subject_excluded_by_authors=False, age="", sex="", handedness="", population="Parkinson's disease" if pd_ else "healthy",
            setting="clinic", device="Wacom Cintiq 12WX pen display, custom C# recording software", session=1,
            block=TEST[t][0], trial=k, phase="test", time_s="", task="handwriting and drawing", unit="recording",
            condition=TEST[t][0], pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            recording=k, samples_file=fn, signal_kind="pen position (x, y, z), pen pressure and grip angle on a graphics tablet",
            channels=CH, channel_units=UNITS, sample_rate_hz=round(1.0 / dt, 1) if dt == dt and dt > 0 else "",
            n_samples=len(g), duration_s=round(float(tt[-1] - tt[0]), 3), events="",
            processing="none stated; time_s is the Timestamp column (ms) minus its first value in the recording",
            trial_info=f"deposit_folder={folder}; test_id={t}"))
df = pd.DataFrame(rows)

n_pd = sum(p.startswith("P") for p in persons)
n_c = sum(p.startswith("C") for p in persons)
SRCP = "UCI dataset 395 page (abstract)"
checks.published("people with Parkinson's disease", n_pd, 62, 0, SRCP)
checks.published("healthy controls", n_c, 15, 0, SRCP)
hw = [p for p in order if p.startswith("hw_dataset")]
checks.published("people with Parkinson's disease in the hw_dataset folder (the paper's 25 PWP, UCI dataset 358 page)", sum("/parkinson/" in p for p in hw), 25, 0, "UCI dataset 358 page, abstract")
checks.published("healthy controls in the hw_dataset folder (UCI dataset 358 page)", sum("/control/" in p for p in hw), 15, 0, "UCI dataset 358 page, abstract")
checks.internal("every sample of every file kept (524,743 lines)", n_samples == 524743, f"{n_samples}")
checks.internal("77 files = 77 people, all three test ids only 0, 1, 2", len(persons) == 77 and set(df.trial.unique()) <= set(range(1, 8)))
checks.internal("pressure inside the documented 0-1023 scale", all(float(e[1].split('\r\n')[1].split(',')[4]) <= 1023 for e in extra[:20]))
print("recordings", len(df), "non-monotonic time segments", nonmono)
print(df.groupby(["group", "experiment"]).size())

s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
p = s["papers"][0]
meta = dict(
    study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"],
    data_url="https://doi.org/10.24432/C5Q01S", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted the semicolon-separated text files into one CSV of samples per test and person (samples/...csv) plus the OpenMotor "
             "table, split each person's file at the changes of test id, set time_s to the Timestamp column in seconds from the first sample of "
             "the test, and replaced file names with codes. The images of the drawings (hw_drawings) were not carried."),
    how=["One row is one test of one person: the Static Spiral Test (retrace a spiral shown on the tablet), the Dynamic Spiral Test (the spiral "
         "blinks, so the person continues from memory) or the Stability Test on a Certain Point (hold the pen over a red point without touching "
         "the screen; the deposit's readme calls it a circular motion test). Samples are in samples/<person>_t<test id>_<n>.csv: time_s, x, y, z, "
         "pressure (0-1023, UCI page), grip_angle. The units of x, y, z and grip angle and the sampling are not documented; the Timestamp column is "
         "system time in ms (UCI page), and sample_rate_hz is derived from its median step.",
         "Each source file is one person (15 controls and 62 with Parkinson's disease: 25 in the folder hw_dataset, which holds the data of the "
         "paper, and 37 in new_dataset); the deposit holds no age, sex, handedness or disease severity. Not every person did all three tests; "
         "recordings are in file order and deposit_folder and test_id are in trial_info. In 14 recordings the Timestamp column is not in increasing order (kept as shared, so time_s decreases there)."],
    unit_note="test recording (pen samples on a graphics tablet)",
    differences=["The paper's analysis used the first 40 people (25 with Parkinson's disease and 15 controls, the hw_dataset folder, also published as UCI "
                 "dataset 358); dataset 395 adds 37 people with Parkinson's disease in new_dataset. The numbers in the paper itself could not be read "
                 "(PDF not accessible), so the checks use the counts the UCI pages give for the two datasets."],
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
