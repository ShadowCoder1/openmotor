"""Horst, Lapuschkin, Samek, Muller & Schollhorn (2019) Sci Rep 9:2391: 57 healthy adults walking barefoot over a 10 m path at a self-selected speed,
about 20 gait trials each (1,142 gait recordings) plus static trials, full-body markers (Qualisys, 250 Hz) and two Kistler force plates (1000 Hz).
'timeseries' module: per recording one row for the marker trajectories and one row for the ground reaction forces.

Raw files (untrusted data, read only) in raw_extra/Horst2019_SciRep/, from Mendeley Data https://doi.org/10.17632/svx74xcrjr.3 (CC BY 4.0):
markers_tsv.zip (Gait_rawdata_tsv.zip of the deposit), f1_tsv.zip and f2_tsv.zip (force plate 1 and 2 text exports) and subject_info.csv.
The deposit's C3D copy of the same recordings (c3d.zip) is not carried.
Run from the site folder: python3 -I tools/std/convert/Horst2019_SciRep.py
"""
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Horst2019_SciRep"
MODULE = "timeseries"
RAW = stdlib.raw_dir(ID)
TMP = Path("tmp_" + ID)
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "samples").mkdir(parents=True)

si = pd.read_csv(RAW / "subject_info.csv", sep=";")
si.columns = ["sid", "female", "age", "mass", "size"]
si = si.set_index("sid")

zm = zipfile.ZipFile(RAW / "markers_tsv.zip")
zf1 = zipfile.ZipFile(RAW / "f1_tsv.zip")
zf2 = zipfile.ZipFile(RAW / "f2_tsv.zip")
mk = {i.filename[:-4]: i.filename for i in zm.infolist() if not i.is_dir()}
f1 = {i.filename[:-len("_f_1.tsv")]: i.filename for i in zf1.infolist() if not i.is_dir()}
f2 = {i.filename[:-len("_f_2.tsv")]: i.filename for i in zf2.infolist() if not i.is_dir()}
assert set(mk) == set(f1) == set(f2) and len(mk) == 1251, (len(mk), len(f1), len(f2))


DROPPED = []
SHORT = []


def split_tsv(raw, first_col, name=""):
    lines = raw.decode("latin1").split("\r\n")
    j = next(k for k, l in enumerate(lines) if l.startswith(first_col + "\t"))
    head = {}
    for l in lines[:j]:
        p = l.split("\t")
        head[p[0]] = p[1:]
    data = [l for l in lines[j + 1:] if l.strip()]
    width = max(set(len(l.split("\t")) for l in data), key=[len(l.split("\t")) for l in data].count)
    good = [l for l in data if len(l.split("\t")) == width]
    if len(good) != len(data):
        DROPPED.append((name, len(data) - len(good), [l[:60] for l in data if len(l.split("\t")) != width]))
    d = pd.read_csv(io.StringIO("\n".join([lines[j]] + good)), sep="\t", index_col=False)
    d = d.loc[:, [c for c in d.columns if not c.startswith("Unnamed")]]
    return head, d


def key(name):
    m = re.fullmatch(r"(S\d+)_(\d+)_(Static|Gait)", name)
    return m.group(1), int(m.group(2)), m.group(3)


order = sorted(mk, key=key)
rows, seq, stat = [], {}, dict(frames_ok=True, freq_ok=True, zeros=0, cells=0, n_gait=0, n_static=0, markers_gait=set(), markers_static=set(),
                              plates=set(), fz_ratio=[], rates=set(), per_subject={}, force_len_ok=True, time_ok=True)
FORCE_CH = ["Force_X", "Force_Y", "Force_Z", "Moment_X", "Moment_Y", "Moment_Z", "COP_X", "COP_Y", "COP_Z"]
for base in order:
    sid, no, kind = key(base)
    h, d = split_tsv(zm.read(mk[base]), "Frame", base + " markers")
    assert len(d) == int(h["NO_OF_FRAMES"][0]), base
    stat["freq_ok"] &= float(h["FREQUENCY"][0]) == 250.0
    mnames = h["MARKER_NAMES"]
    assert len(mnames) == int(h["NO_OF_MARKERS"][0]) and len(d.columns) == 2 + 3 * len(mnames)
    (stat["markers_gait"] if kind == "Gait" else stat["markers_static"]).add(len(mnames))
    mc = [c for c in d.columns if c not in ("Frame", "Time")]
    stat["zeros"] += int(((d[mc] == 0) | d[mc].isna()).sum().sum())
    stat["cells"] += d[mc].size
    mout = d[["Time"] + mc].rename(columns={"Time": "time_s"})
    assert mout.time_s.is_monotonic_increasing
    stat["time_ok"] &= bool(np.allclose(np.diff(mout.time_s.values), 0.004, atol=1e-6))
    plates = []
    for zz, mp, tag in ((zf1, f1, "P1"), (zf2, f2, "P2")):
        hh, dd = split_tsv(zz.read(mp[base]), "Sample", base + " " + tag)
        if len(dd) != int(hh["NO_OF_SAMPLES"][0]):
            SHORT.append((base, tag, len(dd), int(hh["NO_OF_SAMPLES"][0])))
        assert list(dd.columns) == ["Sample", "Time"] + FORCE_CH, (base, list(dd.columns))
        stat["force_len_ok"] &= float(hh["FREQUENCY"][0]) == 1000.0
        stat["plates"].add((tag, hh["FORCE_PLATE_MODEL"][0]))
        dd = dd.drop(columns="Sample")
        dd["k"] = np.rint(dd.Time * 1000).astype(int)
        dd = dd.set_index("k").rename(columns={c: f"{tag}_{c}" for c in FORCE_CH}).drop(columns="Time")
        plates.append(dd)
    fo = plates[0].join(plates[1], how="outer").sort_index()
    fo.insert(0, "time_s", fo.index.values / 1000.0)
    fcols = [c for c in fo.columns if c != "time_s"]
    seq[sid] = seq.get(sid, 0) + 1
    if kind == "Gait":
        stat["n_gait"] += 1
        stat["per_subject"][sid] = stat["per_subject"].get(sid, 0) + 1
        fz = max(fo["P1_Force_Z"].abs().max(), fo["P2_Force_Z"].abs().max())
        stat["fz_ratio"].append(fz / (si.loc[sid, "mass"] * 9.81))
    else:
        stat["n_static"] += 1
    info_kind = "gait trial" if kind == "Gait" else "static standing trial"
    s = si.loc[sid]
    common = dict(dataset=ID, study="Horst et al. 2019 Sci Rep", experiment="", group="", subject=sid, person=sid, subject_excluded_by_authors=False,
                  age=s["age"], sex="F" if s["female"] == 1 else "M", handedness="", population="healthy", setting="lab",
                  device="Qualisys motion capture (10 Oqus 310 cameras, 250 Hz) and two Kistler 9287CA force plates (1000 Hz), barefoot, 10 m path", session=1, block=info_kind,
                  phase="test" if kind == "Gait" else "other", time_s=np.nan, task="walking", unit="recording",
                  condition="barefoot overground walking at a self-selected speed" if kind == "Gait" else "standing still (static trial)",
                  pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False)
    ev = ""
    for sig, tab, nm in (("markers", mout, "marker"), ("forces", fo, "force")):
        fn = f"{base}_{sig}.csv"
        tab.to_csv(TMP / "samples" / fn, index=False, lineterminator="\r\n", float_format="%.8g")
        k = len([r for r in rows if r["subject"] == sid]) + 1
        if sig == "markers":
            ch, un, kindtxt, rate, n = mc, ["mm"] * len(mc), "3D marker positions (x, y, z per marker), laboratory frame", 250.0, len(mout)
            proc = ("as exported by the authors' Qualisys software: marker trajectories of one recording cut to the analysed stride window, time_s counts from the start of the "
                    "Qualisys measurement; a cell of 0 means the marker was not tracked in that frame (as exported); the paper's 12 Hz filter and normalization were applied "
                    "later in its analysis and are not in these data")
        else:
            ch, un, kindtxt, rate, n = fcols, ["N", "N", "N", "Nm", "Nm", "Nm", "mm", "mm", "mm"] * 2, "ground reaction force, moment and centre of pressure of force plate 1 (P1, front) and 2 (P2, rear)", 1000.0, len(fo)
            proc = ("as exported by the authors' Qualisys software (force plate text files); time_s counts from the start of the Qualisys measurement, the same clock as the marker file; "
                    "a plate with no samples at a time has empty cells; units are not written in the deposit's text files and are the export's usual N, Nm and mm")
        rows.append(dict(common, trial=k, recording=k, samples_file=f"samples/{fn}", signal_kind=kindtxt, channels=";".join(ch), channel_units=";".join(un),
                         sample_rate_hz=rate, n_samples=n, duration_s=float(tab.time_s.iloc[-1] - tab.time_s.iloc[0]), events="", processing=proc,
                         trial_info=f"recording_type={info_kind};authors_file_number={no};signal={nm}s;body_mass_kg={s['mass']};body_size_m={'' if np.isnan(s['size']) else s['size']}"))
df = pd.DataFrame(rows)
nsub = si.shape[0]
ck = stdlib.Checks()
ck.published("participants", si.shape[0], 57, 0, "Methods, Subjects (fifty-seven subjects)")
ck.published("female participants", int(si.female.sum()), 29, 0, "Methods, Subjects (29 female, 28 male)")
ck.published("male participants", int((si.female == 0).sum()), 28, 0, "Methods, Subjects (29 female, 28 male)")
ck.published("mean age (years)", si.age.mean(), 23.1, 0.05, "Methods, Subjects (23.1 +- 2.7 years)")
ck.published("SD of age (years)", si.age.std(ddof=1), 2.7, 0.05, "Methods, Subjects (23.1 +- 2.7 years)")
ck.published("mean body size (m)", si["size"].mean(), 1.74, 0.005, "Methods, Subjects (1.74 +- 0.10 m)")
ck.published("SD of body size (m)", si["size"].std(ddof=1), 0.10, 0.005, "Methods, Subjects (1.74 +- 0.10 m)")
ck.published("mean body mass (kg)", si.mass.mean(), 67.9, 0.05, "Methods, Subjects (67.9 +- 11.3 kg)")
ck.published("SD of body mass (kg)", si.mass.std(ddof=1), 11.3, 0.05, "Methods, Subjects (67.9 +- 11.3 kg)")
ck.published("gait recordings", stat["n_gait"], 1142, 0, "Mendeley Data description (raw data for each of the 1142 gait recordings)")
ck.published("markers in the static trials", list(stat["markers_static"])[0], 62, 0, "Methods (full-body marker set of 62 retro reflective markers)")
ck.published("marker sampling rate (Hz)", 250.0 if stat["freq_ok"] else 0, 250, 0, "Methods (sampling frequency of 250 Hz)")
ck.published("force plate sampling rate (Hz)", 1000.0 if stat["force_len_ok"] else 0, 1000, 0, "Methods (frequency of 1000 Hz)")
ck.internal("only the two force files of one static recording have a corrupted line, and every other file has all samples its header announces", len(DROPPED) == 2 and len(SHORT) == 2 and {x[0] for x in SHORT} == {"S23_0001_Static"}, f"{DROPPED}; {SHORT}")
ck.internal("every subject in subject_info has recordings and vice versa", set(df.subject) == set(si.index))
ck.internal("marker frame counts match the file headers, marker time steps are 4 ms", stat["frames_ok"] and stat["time_ok"])
ck.internal("gait files carry one marker count, static files another", len(stat["markers_gait"]) == 1 and len(stat["markers_static"]) == 1, f"{stat['markers_gait']} / {stat['markers_static']}")
ck.internal("force plates are two Kistler 9287CA", stat["plates"] == {("P1", "9287CA"), ("P2", "9287CA")})
fzr = np.array(stat["fz_ratio"])
ck.internal("peak vertical force of a gait recording is 0.8-1.6 body weights in 95% of recordings (supports the N unit)", float(np.mean((fzr > 0.8) & (fzr < 1.6))) > 0.95, f"median {np.median(fzr):.2f}, share {np.mean((fzr > 0.8) & (fzr < 1.6)):.3f}")
pc = pd.Series(stat["per_subject"])
dif = [f"The paper says 20 gait trials per participant; the deposit has between {pc.min()} and {pc.max()} gait recordings per participant (1,142 in all, {int((pc != 20).sum())} participants with a count other than 20).",
       f"The marker files of the gait trials hold {list(stat['markers_gait'])[0]} of the {list(stat['markers_static'])[0]} markers: the static trials also carry the knee and ankle (lateral and medial) markers, which the authors removed for walking.",
       f"{stat['zeros']} of {stat['cells']} marker coordinate cells ({100 * stat['zeros'] / stat['cells']:.2f}%) are 0 or empty (marker not tracked), as exported.",
       ("In the static recording of one participant both force plate files end with a corrupted line that holds only sample numbers and times for the last %d samples (no values); it is left out, so these two files have %d of the %d samples their headers announce." % (SHORT[0][3] - SHORT[0][2], SHORT[0][2], SHORT[0][3])) if SHORT else "",
       "One participant has no body size in subject_info.csv; it is left empty.",
       "Units of the force files (N, Nm, mm) are not written in the deposit's text files; they follow the export's usual units and the check above (peak vertical force near body weight)."]
parts = [dict(person=sid, body_mass_kg=r["mass"], body_size_m=r["size"]) for sid, r in si.iterrows()]
pcols = [("body_mass_kg", "number", "kg", "Body mass (authors' subject_info.csv)."), ("body_size_m", "number", "m", "Body size, i.e. height (authors' subject_info.csv); empty for one participant.")]
dif = [x for x in dif if x]
meta = dict(
    paper=dict(authors="Horst F, Lapuschkin S, Samek W, Muller KR, Schollhorn WI", year=2019,
               title="Explaining the unique nature of individual gait patterns with deep learning", venue="Scientific Reports 9(1):2391", doi="10.1038/s41598-019-38748-8"),
    lab="Schollhorn (Johannes Gutenberg-University Mainz)", data_url="https://doi.org/10.17632/svx74xcrjr.3", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each of the deposit's 1,251 recordings (1,142 gait, 109 static) became two rows: the marker trajectories (Qualisys tsv export, Frame column dropped, Time kept as time_s) and the "
             "ground reaction forces (the two force plate tsv exports joined on their common time stamps; Sample column dropped; columns prefixed P1_ and P2_). Values to 8 significant digits, "
             "nothing resampled or filtered. Participant labels replaced by codes; age, sex, body mass and body size from subject_info.csv. The C3D copy of the recordings is not carried."),
    how=["One row is one signal file of one recording: either the 3D positions (mm) of the full-body markers at 250 Hz, or the ground reaction forces (N), moments (Nm) and centre of pressure (mm) "
         "of the two force plates at 1000 Hz, while the participant walked barefoot along a 10 m path at a self-selected speed (gait recordings) or stood still (static recordings). "
         "Each gait recording holds the analysed stride (right foot heel strike to left toe off, per the paper) cut from a longer measurement; time_s is the time from the start of that measurement.",
         "Marker channel names are the authors' (including their spelling, for example WIRST); X, Y, Z are the Qualisys laboratory axes. In the force files Force, Moment and COP are given for each plate "
         "in the Qualisys export convention; P1 is the front plate ('vorne 1' in the files), P2 the rear one ('hinten 2').",
         "sex is F when the deposit's gender field is 1 (1 = female, 0 = male). The study measured only healthy young adults; handedness was not reported."],
    unit_note="signal file of one recording (markers or forces)", differences=dif)
rec = stdlib.release(ID, MODULE, df, meta, ck, participants=pd.DataFrame(parts), participant_columns=pcols,
                     extra_files=[(f"samples/{p.name}", str(p)) for p in sorted((TMP / "samples").iterdir())])
shutil.rmtree(TMP, ignore_errors=True)
