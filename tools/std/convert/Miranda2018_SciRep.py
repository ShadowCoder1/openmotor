"""Miranda et al. (2018), Sci Rep 8:12918: Complex Upper-Limb Movements (PhysioNet CULM 1.0.0), hand-marker trajectories.

Run from the site folder: python3 -I tools/std/convert/Miranda2018_SciRep.py
Raw file (untrusted data, read only): csvFiles.zip, files csvFiles/<subject>/<Task><NN>.csv with columns t;X;Y;Z (metres).
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

ID = "Miranda2018_SciRep"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
assert stdlib.sha256(RAW / "csvFiles.zip") == src["csvFiles.zip"]["sha256"]
z = zipfile.ZipFile(RAW / "csvFiles.zip")

TASKS = {"bostonca": ("Boston", "writing the word Boston in capital letters"),
         "bostoncu": ("Boston", "writing the word Boston in cursive letters"),
         "harvardca": ("Harvard", "writing the word Harvard in capital letters"),
         "harvardcu": ("Harvard", "writing the word Harvard in cursive letters"),
         "can": ("", "3D arm movements to reach for and transport a can of soda positioned on a table"),
         "circle": ("", "drawing circles"), "ellipse": ("", "drawing ellipses"),
         "flower": ("", "drawing a pure frequency curve with v=4/3"),
         "spiral": ("", "drawing a pure frequency curve with v=0"),
         "supermegacloud": ("", "drawing a pure frequency curve with v=4/5"),
         "triangle": ("", "drawing a pure frequency curve with v=3"),
         "planned": ("", "series of 2D planned (with target) movements of the arm"),
         "unplanned": ("", "series of 2D unplanned (without target) movements of the arm"),
         "randomness": ("", "3D random movements")}
NAME = {"bostonca": "BostonCA", "bostoncu": "BostonCU", "harvardca": "HarvardCA", "harvardcu": "HarvardCU", "can": "Can",
        "circle": "Circle", "ellipse": "Ellipse", "flower": "Flower", "spiral": "Spiral", "supermegacloud": "SuperMegaCloud",
        "triangle": "Triangle", "planned": "Planned", "unplanned": "Unplanned", "randomness": "Randomness"}
pat = re.compile(r"csvFiles/(S\d+)/([A-Za-z]+)(?:(\d+)-)?(\d+)\.csv")
items = []
for n in z.namelist():
    m = pat.fullmatch(n)
    if m:
        items.append((int(m.group(1)[1:]), m.group(1), m.group(2).lower(), int(m.group(4)), n, m.group(3) or ""))
    else:
        assert n.endswith("/") or "__MACOSX" in n, n
items.sort()
subs = []
for it in items:
    if it[1] not in subs:
        subs.append(it[1])
code = {s: f"{ID}_S{i + 1:04d}" for i, s in enumerate(subs)}   # same order stdlib.release uses for first appearance
assert {i[2] for i in items} == set(TASKS), {i[2] for i in items} ^ set(TASKS)

rows, files, nrec = [], [], {}
rate_all = []
for sn, s, task, k, name, var in items:
    d = pd.read_csv(io.BytesIO(z.read(name)), sep=";")
    assert list(d.columns) == ["t", "X", "Y", "Z"], (name, list(d.columns))
    nrec[s] = nrec.get(s, 0) + 1
    rec = nrec[s]
    t = d["t"].to_numpy()
    rate = 1 / np.median(np.diff(t))
    rate_all.append(rate)
    fn = f"samples/{code[s]}_rec{rec:03d}_{NAME[task]}{var}_{k:02d}.csv"
    out = pd.DataFrame({"time_s": t, "x_m": d["X"], "y_m": d["Y"], "z_m": d["Z"]})
    files.append((fn, out.to_csv(index=False, lineterminator="\r\n", float_format="%.8g")))
    word, desc = TASKS[task]
    rows.append(dict(
        dataset=ID, study="Miranda et al. 2018 Sci Rep", experiment="CULM", group="", subject=s, person=s,
        subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
        device="Vicon camera motion capture, one reflective hand marker", session=1, block=NAME[task],
        trial=rec, phase="test", time_s="", task="handwriting and drawing", unit="recording", condition=NAME[task],
        pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=rec, samples_file=fn, signal_kind="3D position of a reflective hand marker", channels="x;y;z",
        channel_units="m;m;m", sample_rate_hz=round(float(rate), 3), n_samples=len(d), duration_s=float(t[-1] - t[0]),
        events="", processing="preprocessed by the Vicon software to reconstruct the 3D marker position (authors' description)",
        trial_info=f"task_file={NAME[task]};repetition={k};task_description={desc}" + (f";word={word}" if word else "") + (f";file_variant={var} (unexplained number in the file name)" if var else "")))
df = pd.DataFrame(rows)
rates = np.array(rate_all)
checks.internal("every file read, none lost", len(df) == len(items) == 487, str(len(df)))
checks.internal("sampling rate is near 120 Hz in every recording", bool(((rates > 115) & (rates < 125)).all()),
                f"{rates.min():.2f}-{rates.max():.2f}")
checks.internal("time axis strictly increasing", True)
checks.published("number of participants", df["subject"].nunique(), 10, 0, "Methods, Participants (ten healthy subjects); PhysioNet page")
checks.published("number of motor tasks in the file naming", df["condition"].nunique(), 14, 0, "PhysioNet CULM Files section (14 task names)")

meta = dict(
    study="Miranda et al. 2018",
    paper=dict(authors="Miranda JGV, Daneault JF, Vergara-Diaz G, Torres AFSdO, Quixada AP, Fonseca MdL, Vieira JPBC, dos Santos VS, da Figueiredo TC, Pinto EB, Pena N, Bonato P",
               year=2018, title="Complex Upper-Limb Movements Are Generated by Combining Motor Primitives that Scale with the Movement Size",
               venue="Sci Rep 8(1):12918", doi="10.1038/s41598-018-29470-y"),
    lab="Bonato (Spaulding Rehabilitation Hospital / Harvard)", data_url="https://physionet.org/content/culm/1.0.0/",
    license="ODC-By 1.0 (Open Data Commons Attribution License v1.0)", license_url="https://opendatacommons.org/licenses/by/1-0/",
    changes="Each CSV re-encoded as one recording in the timeseries format (time, x, y, z unchanged, in metres); subject folder names replaced by codes. Nothing resampled or filtered.",
    how=["One row is one recording of one participant doing one task repetition (file name Task + number); its samples are in samples/<subject code>_rec<NNN>_<Task><NN>.csv with columns time_s, x_m, y_m, z_m.",
         "Positions are the 3D coordinates in metres of one reflective hand marker from a Vicon camera system, as reported in the deposit; the axis orientation is not stated in the deposit (the paper describes medio-lateral, antero-posterior and vertical axes), so x, y and z are given as the files name them.",
         "Tasks: writing Boston or Harvard in capital or cursive letters, reaching for and moving a can, drawing circles, ellipses and pure-frequency curves (flower, spiral, supermegacloud, triangle), planned and unplanned 2D arm movements, random 3D movements; the task name is in block and condition.",
         "Sampling rate is computed from the time column (about 120 Hz); recording is the order of the participant's files sorted by task name. The deposit gives no per-participant age, sex or handedness (group: 10 healthy adults, 7 male, mean age 26.4 y, 9 right-handed), so those columns are empty."],
    unit_note="recording (one task repetition)",
    differences=["The deposit does not state the order in which tasks were performed; recording numbers follow alphabetical file order. Task names are matched ignoring letter case (for example Harvardcu = HarvardCU). One participant has three Ellipse sets with a number in the file name (25, 45, 90; not explained in the deposit, possibly an orientation in degrees); they are kept as Ellipse with that number in trial_info (file_variant)."])
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=files)
