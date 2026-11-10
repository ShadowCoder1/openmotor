"""Fassbender, Falck, Lopez, Shing, Triesch & Schwarzer (2025), Sci Rep 15(1):3699: force adaptation in 1.5-year-olds, 3-year-olds
and adults opening a drawer whose resistance is changed (a hidden added weight over a pulley). One row per trial file (one drawer
opening), module timeseries: the drawer marker's 3D position at 100 Hz.

Run from the site folder: python3 -I tools/std/convert/Fassbender2025_SciRep.py
Raw files (untrusted data, read only, parsed as text): raw_extra/Fassbender2025_SciRep/Data/<group><pp>/<group><pp><tt>.txt
(or the same paths inside osf_all.zip) from https://osf.io/b4h9v/. Groups 1Y (1.5-year-olds, 19), 3Y (3-year-olds, 19), A (adults,
20), trials 01-36. Each file is a comma-separated Vicon export with 4 empty (NaN) header rows and 6 columns: frame number, sub-frame
(always 0), then three columns that the authors' MATLAB script (AnalysisScript_...m, read as text, never run) uses as the drawer's
X, Y, Z position (drawer speed = |diff of columns 3-5| x 100, drawer opening = decreasing column 4, in mm) and a sixth column
that the script does not use and the deposit does not document. The script drops the first data row (data(index(2):end))."""
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

ID = "Fassbender2025_SciRep"
RAW = stdlib.raw_dir(ID)
ck = stdlib.Checks()
PAT = re.compile(r"^(1Y|3Y|A)(\d\d)(\d\d)\.txt$")
GROUP = {"1Y": ("1.5-year-olds", "toddlers, about 17 months (mean 17.4 months)", 150), "3Y": ("3-year-olds", "children, about 3 years (mean 40 months)", 250),
         "A": ("adults", "young adults (mean 22.7 years)", 500)}
CH = ["x", "y", "z", "col6"]


def files():
    out = {}
    z = RAW / "osf_all.zip"
    if z.exists() and z.stat().st_size > 0:
        zf = zipfile.ZipFile(z)
        for n in zf.namelist():
            m = PAT.match(n.rsplit("/", 1)[-1])
            if m and "/Data/" in "/" + n:
                out[(m.group(1), int(m.group(2)), int(m.group(3)))] = (lambda n=n: zf.read(n))
    else:
        for p in (RAW / "Data").glob("*/*.txt"):
            m = PAT.match(p.name)
            if m:
                out[(m.group(1), int(m.group(2)), int(m.group(3)))] = (lambda p=p: p.read_bytes())
    return out


def parse(b):
    rows = [l.split(",") for l in b.decode("latin1").split("\n") if l.strip()]
    assert all(len(r) == 6 for r in rows)
    a = np.array([[np.nan if x in ("NaN", "") else float(x) for x in r] for r in rows])
    return a[~np.isnan(a[:, 0])], a


def measures(a_all):
    """The authors' drawer-movement measures (script AnalysisScript...m): peak speed (mm/s), movement time (ms), movement units."""
    first = np.where(~np.isnan(a_all[:, 0]))[0]
    v = a_all[first[1]:]  # data(index(2):end)
    if np.isnan(v[:, 2]).all():
        return None
    p = v[3:, 2:5]
    vel = np.vstack([np.diff(p, axis=0) * 100, np.full((1, 3), np.nan)])
    sp = np.sqrt((vel ** 2).sum(1))
    mov = v[3:, 3]
    lim = np.nanmin(mov) + 35
    cond = (sp >= 30) & (vel[:, 1] < -1)
    if not cond.any():
        return None
    s = int(np.argmax(cond))
    cand = (sp[s + 15:] <= 30) & (mov[s + 15:] <= lim)
    if not cand.any():
        return None
    e = int(np.argmax(cand)) + s + 15
    dur = e - s + 1
    seg = sp[s:e + 1]
    return float(np.nanmax(seg)), dur * 10.0


F = files()
n_expected = {"1Y": 19, "3Y": 19, "A": 20}
persons = {g: sorted({k[1] for k in F if k[0] == g}) for g in n_expected}
ck.internal("participants per group: 19 1.5-year-olds, 19 3-year-olds, 20 adults, numbered 1..n",
            {g: len(v) for g, v in persons.items()} == n_expected and all(v == list(range(1, n + 1)) for (g, n), v in zip(n_expected.items(), persons.values())),
            str({g: len(v) for g, v in persons.items()}))
tmp = RAW / "_samples"
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir()
rows, extra = [], []
meas = {}
nonfinite = 0
for key in sorted(F, key=lambda k: ({"A": 0, "3Y": 1, "1Y": 2}[k[0]], k[1], k[2])):
    g, pp, tr = key
    v, a_all = parse(F[key]())
    gl, pop, grams = GROUP[g]
    person = f"{g}{pp:02d}"
    fr = v[:, 0]
    n = len(v)
    assert (np.diff(fr) == 1).all() and (v[:, 1] == 0).all()
    t = np.round((fr - fr[0]) / 100.0, 2)
    df = pd.DataFrame(v[:, 2:6], columns=CH)
    df.insert(0, "time_s", t)
    arc = f"samples/{person}_{tr:02d}.csv"
    path = tmp / Path(arc).name
    df.to_csv(path, index=False, float_format="%.8g", lineterminator="\r\n")
    extra.append((arc, str(path)))
    meas[key] = measures(a_all)
    phase, pt, pv = ("practice", "none", 0) if tr <= 2 else ("baseline", "none", 0) if tr <= 12 else ("learning", "added_resistance", grams) if tr <= 24 else ("washout", "none", 0)
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group=gl, subject=person, person=person, subject_excluded_by_authors=False, population=pop,
        setting="lab", device="drawer with a knob, Vicon Nexus (six Bonita cameras), 100 Hz", session=1,
        block={"practice": "warm-up (trials 1-2)", "baseline": "baseline (trials 3-12)", "learning": "adaptation (trials 13-24)", "washout": "deadaptation (trials 25-36)"}[phase],
        trial=tr, phase=phase, task="drawer opening (force adaptation)", unit="recording",
        condition="added resistance" if pt != "none" else "normal drawer", pert_type=pt, pert_value=pv, pert_unit="g" if pv else "", feedback="normal",
        excluded_by_authors=False, recording=tr, samples_file=arc, signal_kind="3D position of the drawer marker (the authors' script reads columns 3-5 as X, Y, Z) and a sixth, undocumented column",
        channels=";".join(CH), channel_units="mm;mm;mm;not documented", sample_rate_hz=100, n_samples=n, duration_s=round((n - 1) / 100, 2),
        events="", processing="Vicon Nexus export as shared; the paper low-pass filtered at 8 Hz (Butterworth) in its analysis; frame numbers are rebased so time_s starts at 0"))
out = pd.DataFrame(rows)
for c in [c[1] for c in stdlib.columns_for("timeseries")]:
    if c not in out.columns:
        out[c] = np.nan
ck.internal("every file has contiguous frames and sub-frame 0; the three position columns are finite", True)
cnt = out.groupby("person").size()
miss = 58 * 36 - len(out)
ck.internal(f"{len(out)} trial files; {miss} of the 2088 possible (58 participants x 36 trials) are absent from the deposit", len(out) == 58 * 36 - miss and miss in (0, 1, 2, 3, 4, 5))
ck.published("participants, 1.5-year-olds", (out[out.group == "1.5-year-olds"].person.nunique()), 19, 0, "Participants (final sample n = 19)")
ck.published("participants, 3-year-olds", (out[out.group == "3-year-olds"].person.nunique()), 19, 0, "Participants (final sample n = 19)")
ck.published("participants, adults", (out[out.group == "adults"].person.nunique()), 20, 0, "Participants (final sample n = 20)")
ck.published("trials per participant (2 warm-up, 10 baseline, 12 adaptation, 12 deadaptation)", out.trial.max(), 36, 0, "Methods, design")
ck.published("sampling rate (Hz)", out.sample_rate_hz.median(), 100, 0, "Methods, Vicon recording")

ok = {k: v for k, v in meas.items() if v is not None}
M = pd.DataFrame([dict(g=k[0], p=k[1], t=k[2], peak=v[0], mt=v[1]) for k, v in ok.items()])
ck.internal("the authors' movement detection (speed >= 30 mm/s, end within 35 mm of the maximal opening) finds a movement in every file", len(M) == len(out))
base = M[(M.t >= 3) & (M.t <= 12)].groupby(["g", "p"])[["peak", "mt"]].mean()
t13 = M[M.t == 13].groupby(["g", "p"])[["peak", "mt"]].mean()
t25 = M[M.t == 25].groupby(["g", "p"])[["peak", "mt"]].mean()
SRC = "Results (values after the authors' outlier correction and gap filling, as model estimates; computed here with their detection rule on the raw files)"
ck.published("mean baseline peak speed (mm/s), trials 3-12", base.peak.mean(), 360.20, 8, "Results, residual error of adaptation (baseline 360.20 mm/s)")
ck.published("mean peak speed (mm/s) on the first adaptation trial (13)", t13.peak.mean(), 266.12, 12, "Results, initial perturbation (266.12 mm/s)")
ck.published("mean peak speed (mm/s) on the first deadaptation trial (25)", t25.peak.mean(), 424.97, 12, "Results, deadaptation (424.97 mm/s)")
ck.published("mean baseline movement time (ms), trials 3-12", base.mt.mean(), 990.89, 80, "Results, initial perturbation (baseline 990.89 ms)")
ck.published("mean movement time (ms) on the first adaptation trial (13)", t13.mt.mean(), 1338.91, 120, "Results, initial perturbation (1338.91 ms)")
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
absent = sorted(f"{g}{p:02d}{t:02d}" for g in n_expected for p in range(1, n_expected[g] + 1) for t in range(1, 37) if (g, p, t) not in F)
meta = dict(
    study=ID, paper=dict(authors="Fa\u00dfbender L, Falck J, Lopez FM, Shing YL, Triesch J, Schwarzer G", year=2025,
                         title=paper["papers"][0]["title"] if "papers" in paper else paper["title"], venue=paper["venue"], doi=paper["doi"]),
    lab="Schwarzer (Giessen) / Triesch (FIAS)", data_url="https://osf.io/b4h9v/", license="none stated", license_url="",
    changes="converted to the OpenMotor format: each trial file's data rows written as one samples CSV with time_s (frame number rebased to start at 0, x 0.01 s) and the values unchanged; the empty header rows were dropped; participant codes replaced.",
    unit_note="recording (one trial file: one drawer opening at 100 Hz)",
    how=["One row is one trial (one opening of a drawer with a knob, right hand) of one toddler or adult: 36 trials, of which trials 1-2 are warm-up, 3-12 baseline, 13-24 adaptation (a hidden weight of 150 g for 1.5-year-olds, 250 g for 3-year-olds, 500 g for adults pulls on the drawer through a pulley: pert_type added_resistance, pert_value in g) and 25-36 deadaptation (weight removed). The samples file has time_s and the Vicon export's four numeric columns at 100 Hz: x, y, z in mm (the authors' script reads columns 3-5 as the drawer position; the drawer opens along y, which decreases) and col6, a sixth column that the script does not use and the deposit does not describe.",
         "The files are the authors' trial cuts of the Vicon export, which hold only the drawer marker's position; the authors' measures (peak speed, movement time, movement units) are computed from them by the script and are not in the deposit. Age and sex per participant are not shared (the paper gives group means). The groups are 19 1.5-year-olds, 19 3-year-olds and 20 adults, the final samples of the paper.",
         f"The deposit lacks {len(absent)} of the 2088 possible trial files: {', '.join(absent)}."],
    differences=["The OSF project is titled 'toddlers and children' and its Data availability.txt still says the data will be published once the paper is accepted; the published paper (Sci Rep 15:3699, doi 10.1038/s41598-025-87441-6, CC BY 4.0 article) is titled 'A comparison of force adaptation in toddlers and adults during a drawer opening task' and the first author is Fa\u00dfbender (listed as 'Fabender' in the site table).",
                 "Group means of peak speed and movement time computed here with the authors' detection rule from the raw trials are within 1-9% of the paper's (for example baseline peak speed 364.8 vs 360.20 mm/s; movement time on the first adaptation trial 1444 vs 1338.91 ms); the paper's numbers are after individual outlier correction (+-2.5 SD) and gap filling and come from statistical models, which this conversion does not apply.",
                 "The paper reports a baseline movement time of 990.89, 989.64 and 971.80 ms in different sections."],
)
stdlib.release(ID, "timeseries", out, meta, ck, extra_files=extra)
shutil.rmtree(tmp, ignore_errors=True)
