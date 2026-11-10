"""Schreiber & Moissenet (2019) Sci Data 6:111: 50 injury-free adults walking on a level walkway at five speed conditions (metronome-paced 0-0.4, 0.4-0.8 and
0.8-1.2 m/s, self-selected spontaneous and fast), 1,143 walking trials plus one static trial each. 'timeseries' module: per walking trial one row for the 52 marker
trajectories (100 Hz) and one row for the analog channels (8 EMG and the two force plates' force and moment channels, 1500 Hz); per static trial one marker row.

Raw file (untrusted data, read only): raw_extra/Schreiber2019_SciData/gait.zip, from https://doi.org/10.6084/m9.figshare.7734767 (CC BY 4.0), 1,193 C3D files read with
the small C3D reader in Schreiber2019_SciData_c3d.py (bytes only, nothing is executed). Age, sex, height, mass and leg lengths come from the deposit's PDF 'Anthropometric and demographic information of the participants' (transcribed into ANTHRO below, because the C3D SUBJECT parameters differ from it for some participants).
Run from the site folder: python3 -I tools/std/convert/Schreiber2019_SciData.py
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Schreiber2019_SciData_c3d import read_c3d  # noqa: E402

ID = "Schreiber2019_SciData"
MODULE = "timeseries"
RAW = stdlib.raw_dir(ID)
TMP = Path("tmp_" + ID)
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "samples").mkdir(parents=True)

z = zipfile.ZipFile(RAW / "gait.zip")
files = sorted(i.filename for i in z.infolist() if i.filename.endswith(".c3d"))
assert len(files) == 1193


ANTHRO = {  # (the PDF lists participant 2014016 as 2015016, a typo: the C3D folder is 2014016 and its weight and height match that row) transcribed from the deposit's 'Anthropometric and demographic information of the participants.pdf': id -> (gender M/W, age, height m, mass kg, right leg m, left leg m)
    "2014001": ("M", 31, 1.66, 67.0, 0.731, 0.735),
    "2014002": ("W", 48, 1.64, 65.4, 0.774, 0.77),
    "2014003": ("W", 28, 1.56, 50.0, 0.721, 0.72),
    "2014004": ("M", 23, 1.77, 72.5, 0.829, 0.848),
    "2014005": ("M", 25, 1.83, 73.5, 0.864, 0.892),
    "2014006": ("M", 23, 1.76, 73.0, 0.849, 0.856),
    "2014007": ("W", 44, 1.69, 65.0, 0.837, 0.838),
    "2014008": ("W", 30, 1.66, 57.1, 0.802, 0.793),
    "2014009": ("M", 57, 1.88, 86.0, 0.897, 0.89),
    "2014011": ("M", 59, 1.8, 63.4, 0.849, 0.854),
    "2014013": ("W", 26, 1.7, 61.3, 0.789, 0.784),
    "2014014": ("M", 29, 1.8, 92.0, 0.842, 0.847),
    "2014015": ("W", 22, 1.58, 67.0, 0.716, 0.708),
    "2014019": ("W", 26, 1.76, 73.8, 0.819, 0.828),
    "2014022": ("W", 48, 1.71, 59.8, 0.819, 0.826),
    "2014024": ("M", 33, 1.92, 87.5, 0.906, 0.906),
    "2014025": ("W", 31, 1.66, 80.5, 0.781, 0.788),
    "2014029": ("M", 38, 1.89, 89.9, 0.877, 0.885),
    "2014030": ("W", 62, 1.7, 60.7, 0.802, 0.806),
    "2014031": ("M", 21, 1.77, 67.2, 0.802, 0.822),
    "2014033": ("W", 24, 1.6, 63.5, 0.706, 0.714),
    "2014034": ("M", 21, 1.84, 89.6, 0.902, 0.899),
    "2014040": ("W", 19, 1.55, 56.5, 0.734, 0.715),
    "2014046": ("W", 40, 1.65, 61.8, 0.85, 0.853),
    "2014048": ("W", 40, 1.64, 61.5, 0.81, 0.816),
    "2014049": ("M", 32, 1.74, 72.2, 0.841, 0.835),
    "2014050": ("W", 28, 1.64, 61.9, 0.75, 0.758),
    "2014051": ("M", 25, 1.91, 88.0, 0.92, 0.915),
    "2014052": ("M", 25, 1.82, 79.5, 0.871, 0.858),
    "2014053": ("W", 21, 1.72, 62.8, 0.835, 0.822),
    "2015002": ("M", 39, 1.74, 74.0, 0.828, 0.838),
    "2015003": ("M", 52, 1.77, 87.2, 0.843, 0.847),
    "2015004": ("W", 35, 1.7, 62.0, 0.802, 0.809),
    "2015005": ("M", 48, 1.9, 89.4, 0.874, 0.877),
    "2015007": ("W", 63, 1.66, 60.2, 0.755, 0.752),
    "2015013": ("W", 58, 1.69, 73.0, 0.808, 0.808),
    "2015015": ("W", 50, 1.73, 68.0, 0.816, 0.829),
    "2014016": ("W", 46, 1.69, 76.0, 0.855, 0.829),
    "2015017": ("W", 41, 1.67, 60.5, 0.805, 0.806),
    "2015020": ("M", 43, 1.79, 95.0, 0.845, 0.846),
    "2015021": ("W", 30, 1.69, 58.0, 0.775, 0.788),
    "2015026": ("W", 64, 1.71, 51.5, 0.8, 0.814),
    "2015027": ("M", 51, 1.72, 65.5, 0.791, 0.788),
    "2015030": ("M", 24, 1.87, 86.0, 0.917, 0.917),
    "2015032": ("M", 26, 1.72, 50.8, 0.803, 0.817),
    "2015035": ("M", 38, 1.77, 81.5, 0.819, 0.839),
    "2015037": ("M", 42, 1.76, 66.1, 0.837, 0.85),
    "2015041": ("M", 31, 1.88, 74.8, 0.876, 0.857),
    "2015042": ("M", 67, 1.83, 98.0, 0.855, 0.874),
    "2015043": ("M", 21, 1.78, 74.0, 0.84, 0.833),
}
assert len(ANTHRO) == 50


def parse_name(fn):
    m = re.fullmatch(r".*/(\d{7})/\1_(C[1-5]_\d\d|ST)\.c3d", fn)
    assert m, fn
    pid, rest = m.group(1), m.group(2)
    return pid, ("static", 0) if rest == "ST" else (rest[:2], int(rest[3:]))


COND = {"C1": "metronome-paced walking, 0-0.4 m/s (C1)", "C2": "metronome-paced walking, 0.4-0.8 m/s (C2)", "C3": "metronome-paced walking, 0.8-1.2 m/s (C3)",
        "C4": "self-selected spontaneous speed (C4)", "C5": "self-selected fast speed, not running (C5)", "static": "standing still (static trial)"}
files.sort(key=lambda f: (parse_name(f)[0], parse_name(f)[1][0] == "static", parse_name(f)[1]))
subj = {}
for fn in files:
    pid, (c, n) = parse_name(fn)
    if c != "static" and pid not in subj:
        d = read_c3d(z.read(fn))
        v = dict(zip(d["params"]["SUBJECT:LABELS"], d["params"]["SUBJECT:VALUES"]))
        subj[pid] = v
assert len(subj) == 50
rows, seq = [], {}
stat = dict(n_walk=0, n_static=0, ev_ok=True, resid_neg=0, pts_zero=0, pts_cells=0, labels=set(), alabels=set(), rates=set(), per_cond={}, an_len_ok=True, ev_names=set(),
            frames_ok=True, npts={}, units=set(), nan=0, ratio=set())
for fn in files:
    pid, (cond, no) = parse_name(fn)
    d = read_c3d(z.read(fn))
    s = subj[pid]
    a = ANTHRO[pid]
    pts, nfr = d["points"], d["n_frames"]
    assert d["rate"] == 100.0 and pts.shape[1] <= 52
    stat["labels"].add(tuple(d["point_labels"]))
    stat["npts"][pts.shape[1]] = stat["npts"].get(pts.shape[1], 0) + 1
    stat["resid_neg"] += int((pts[:, :, 3] < 0).sum())
    xyz = pts[:, :, :3]
    stat["pts_zero"] += int((xyz == 0).all(axis=2).sum())
    stat["pts_cells"] += xyz.shape[0] * xyz.shape[1]
    stat["nan"] += int(np.isnan(xyz).sum())
    mcols = [f"{l}_{a}" for l in d["point_labels"] for a in "XYZ"]
    mtab = pd.DataFrame(xyz.reshape(nfr, -1), columns=mcols)
    mtab.insert(0, "time_s", np.arange(nfr) / 100.0)
    base = fn.split("/")[-1][:-4]
    seq[pid] = seq.get(pid, 0)
    common = dict(dataset=ID, study="Schreiber & Moissenet 2019 Sci Data", experiment="", group="", subject=pid, person=pid, subject_excluded_by_authors=False,
                  age=a[1], sex="M" if a[0] == "M" else "F", handedness="", population="healthy", setting="lab",
                  device="Qualisys 10-camera system (100 Hz), two AMTI OR6-5 force plates (1500 Hz), Noraxon wireless EMG (1500 Hz), barefoot, 10 m walkway", session=1,
                  block=cond if cond != "static" else "static", phase="test" if cond != "static" else "other", time_s=np.nan, task="walking", unit="recording",
                  condition=COND[cond], pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False)
    info = f"authors_file={base.split('_', 1)[1]};height_m={a[2]};weight_kg={a[3]};right_leg_length_m={a[4]};left_leg_length_m={a[5]}"
    ev_txt = ""
    if cond == "static":
        stat["n_static"] += 1
    else:
        stat["n_walk"] += 1
        stat["per_cond"][(pid, cond)] = stat["per_cond"].get((pid, cond), 0) + 1
        assert d["analog_rate"] == 1500.0 and d["ratio"] == 15
        stat["alabels"].add(tuple(d["analog_labels"]))
        stat["units"].add(tuple(d["params"]["ANALOG:UNITS"]))
        n_an = d["analog"].shape[0]
        stat["an_len_ok"] &= (n_an == nfr * 15)
        evp = d["params"]
        ev = []
        cnt = {}
        for ctx, lab, t in zip(evp["EVENT:CONTEXTS"], evp["EVENT:LABELS"], evp["EVENT:TIMES"][:, 1]):
            nm = f"{ctx.lower()}_{lab.lower().replace(' ', '_')}"
            cnt[nm] = cnt.get(nm, 0) + 1
            ev.append((float(t), nm, cnt[nm]))
        names = [f"{nm}_{i}" if cnt[nm] > 1 else nm for t, nm, i in ev]
        stat["ev_names"] |= {re.sub(r"_\d$", "", n) for n in names}
        stat["ev_ok"] &= all(0 <= t <= nfr / 100.0 for t, _, _ in ev)
        ev_txt = ";".join(f"{n}={t:.4f}".rstrip("0").rstrip(".") for (t, _, _), n in sorted(zip(ev, names)))
        atab = pd.DataFrame(d["analog"], columns=d["analog_labels"])
        atab.insert(0, "time_s", np.arange(n_an) / 1500.0)
        afn = f"{base}_analog.csv"
        atab.to_csv(TMP / "samples" / afn, index=False, lineterminator="\r\n", float_format="%.8g")
    seq[pid] += 1
    mfn = f"{base}_markers.csv"
    mtab.to_csv(TMP / "samples" / mfn, index=False, lineterminator="\r\n", float_format="%.8g")
    rows.append(dict(common, trial=seq[pid], recording=seq[pid], samples_file=f"samples/{mfn}", signal_kind="3D positions of 52 reflective markers (X, Y, Z each), laboratory frame",
                     channels=";".join(mcols), channel_units=";".join(["mm"] * len(mcols)), sample_rate_hz=100.0, n_samples=nfr, duration_s=(nfr - 1) / 100.0,
                     events=ev_txt, processing=("as shared by the authors: gaps interpolated by principal component analysis, then 4th-order Butterworth low-pass at 6 Hz; the fourth value of each C3D point (residual) is not carried"
                                                if cond != "static" else "as shared by the authors (single frame)"), trial_info=info + ";signal=markers"))
    if cond != "static":
        seq[pid] += 1
        rows.append(dict(common, trial=seq[pid], recording=seq[pid], samples_file=f"samples/{afn}",
                         signal_kind="analog channels: EMG of 8 muscles of the right leg and hip, and force and moment channels of force plates 1 and 2 (in each plate's own coordinate system)",
                         channels=";".join(d["analog_labels"]), channel_units=";".join(d["params"]["ANALOG:UNITS"]), sample_rate_hz=1500.0, n_samples=n_an, duration_s=(n_an - 1) / 1500.0,
                         events=ev_txt, processing=("as shared by the authors: EMG band-pass filtered 30-300 Hz (4th-order Butterworth); forces and moments low-pass filtered (2nd-order Butterworth, 15 Hz) "
                                                    "and set to zero below 5 N of vertical force; the C3D gives the Fx..Mz channels as 'fake analog channels for virtual force platform components'"),
                         trial_info=info + ";signal=analog"))
df = pd.DataFrame(rows)
P = pd.DataFrame(ANTHRO, index=["gender", "age", "height", "weight", "rleg", "lleg"]).T
Q = pd.DataFrame(subj).T
mis_g = int(sum((ANTHRO[k][0] == "M") != (Q.loc[k, "gender"] == 1) for k in ANTHRO))
mis_a = int(sum(ANTHRO[k][1] != Q.loc[k, "age"] for k in ANTHRO))
ck = stdlib.Checks()
ck.published("participants", len(P), 50, 0, "Abstract / Methods, Participants (50 adults)")
ck.published("women", int((P.gender == "W").sum()), 24, 0, "Methods, Participants (24 women and 26 men)")
ck.published("men", int((P.gender == "M").sum()), 26, 0, "Methods, Participants (24 women and 26 men)")
ck.published("mean age (years)", P.age.astype(float).mean(), 37.0, 0.05, "Methods, Participants (37.0 +- 13.6 years)")
ck.published("SD of age (years)", P.age.astype(float).std(ddof=1), 13.6, 0.05, "Methods, Participants (37.0 +- 13.6 years)")
ck.published("youngest age (years)", P.age.astype(float).min(), 19, 0, "Background (aged between 19 and 67 years)")
ck.published("oldest age (years)", P.age.astype(float).max(), 67, 0, "Background (aged between 19 and 67 years)")
ck.published("mean height (m)", P.height.astype(float).mean(), 1.74, 0.005, "Methods, Participants (1.74 +- 0.09 m)")
ck.published("SD of height (m)", P.height.astype(float).std(ddof=1), 0.09, 0.005, "Methods, Participants (1.74 +- 0.09 m)")
ck.published("SD of weight (kg)", P.weight.astype(float).std(ddof=1), 12.3, 0.05, "Methods, Participants (71.0 +- 12.3 kg)")
ck.published("walking trials", stat["n_walk"], 1143, 0, "Abstract / Data Records (a total of 1143 trials)")
ck.published("static trials (one per participant)", stat["n_static"], 50, 0, "Data Records (at least 3 trials per condition plus one static, for each of the 50 participants)")
ck.published("markers (most in a file)", max(len(l) for l in stat["labels"]), 52, 0, "Records / Table 1 (52 cutaneous reflective markers)")
ck.published("EMG probes", sum(1 for l in list(stat["alabels"])[0] if not re.match(r"[FM][xyz][12]$", l)), 8, 0, "Records (8 probes) / Table 2")
ck.published("walking speed conditions", len({c for _, c in stat["per_cond"]}), 5, 0, "Methods, Procedure (C1-C5)")
ck.internal("at least 3 trials per participant and condition", min(stat["per_cond"].values()) >= 3 and len(stat["per_cond"]) == 250, f"min {min(stat['per_cond'].values())}")
ck.internal("analog labels identical in every walking file; marker labels of every file are a subset of the 52", len(stat["alabels"]) == 1 and all(set(l) <= set(max(stat["labels"], key=len)) for l in stat["labels"]), str(stat["npts"]))
ck.internal("analog samples = 15 per marker frame in every walking file", stat["an_len_ok"])
ck.internal("event times lie inside the recording", stat["ev_ok"], str(sorted(stat["ev_names"])))
ck.internal("no NaN coordinates", stat["nan"] == 0)
parts = [dict(person=pid, height_m=r["height"], body_mass_kg=r["weight"], right_leg_length_m=r["rleg"], left_leg_length_m=r["lleg"]) for pid, r in P.iterrows()]
pcols = [("height_m", "number", "m", "Body height (the deposit's anthropometric table)."), ("body_mass_kg", "number", "kg", "Body weight (the deposit's anthropometric table)."),
         ("right_leg_length_m", "number", "m", "Right leg length (the deposit's anthropometric table)."), ("left_leg_length_m", "number", "m", "Left leg length (the deposit's anthropometric table).")]
dif = ["The deposit's anthropometric table lists one participant as 2015016; no such folder exists and the folder 2014016 (absent from the table) has the same height and weight, so the row is assigned to 2014016.",
       f"Age, sex, height, mass and leg lengths are those of the deposit's anthropometric table (a PDF), which reproduces the paper's 24 women, 26 men and age 37.0 +- 13.6 years. The SUBJECT parameters inside the C3D files disagree with that table for {mis_g} participant(s) in gender and {mis_a} in age and were not used.",
       f"The paper gives a mean mass of 71.0 kg; the deposit's table gives {P.weight.astype(float).mean():.1f} kg, so that number is not used as a check.",
       f"{sum(v for k, v in stat['npts'].items() if k < 52)} of the {len(files)} files (marker count by file: {dict(sorted(stat['npts'].items()))}) hold fewer than 52 markers: the authors left out markers that could not be tracked; each row lists its own markers in channels.",
       f"{stat['pts_zero']} of {stat['pts_cells']} marker positions are all zero ({100 * stat['pts_zero'] / stat['pts_cells']:.3f}%): markers the authors could not reconstruct; kept as shared.",
       "The paper counts 1143 trials as walking trials; the 50 static trials (one per participant) are carried as well, as one-frame marker recordings.",
       "The force and moment channels (Fx1...Mz2) are the C3D's analog channels, expressed in each force plate's coordinate system; the force plate structures that the authors describe (centre of pressure, force and moment in the laboratory frame) are not stored in the C3D files and are not reconstructed here."]
meta = dict(
    paper=dict(authors="Schreiber C, Moissenet F", year=2019, title="A multimodal dataset of human gait at different walking speeds established on injury-free adult participants",
               venue="Scientific Data 6(1):111", doi="10.1038/s41597-019-0124-4"),
    lab="Moissenet (Rehazenter, Luxembourg)", data_url="https://doi.org/10.6084/m9.figshare.7734767", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each C3D file was read as bytes and written as CSV: for a walking trial the 52 marker trajectories (X, Y, Z in mm; the residual value of each point dropped) and, separately, the 20 analog "
             "channels (EMG in V, forces in N and moments in N.mm); for a static trial the markers of its single frame. Values to 8 significant digits, nothing filtered or resampled "
             "(the authors' filtering is stated in processing). time_s is the sample number divided by the rate (100 Hz markers, 1500 Hz analog). The authors' foot strike and foot off events "
             "are in the events column in seconds. Age, sex, height, weight and leg lengths come from the deposit's anthropometric table (PDF), transcribed into the converter. Participant codes replace the deposit's numbers."),
    how=["One row is one signal file of one trial: the 3D marker positions (100 Hz) or the analog channels (1500 Hz: 8 EMG channels of the right side, and the force and moment channels of the "
         "two force plates) of one walk along a 10 m straight level walkway, barefoot. Each file was cropped by the authors to one right and one left gait cycle.",
         "condition is the walking speed condition: C1 to C3 were paced by a metronome to 0-0.4, 0.4-0.8 and 0.8-1.2 m/s (household, limited community and community ambulators), C4 was the "
         "participant's spontaneous speed and C5 a fast speed without running. Static rows are the five seconds of standing used for calibration, reduced to one frame by the authors.",
         "events lists the authors' manually checked foot strike and foot off events (threshold of 5 N on the vertical force, or marker trajectories when no force was available) as "
         "<side>_foot_strike<1|2> and <side>_foot_off (with _1, _2 when a side has two), in seconds from the start of the recording. sex is the table's gender (W woman, M man)."],
    unit_note="signal file of one trial (markers or analog channels)", differences=dif)
rec = stdlib.release(ID, MODULE, df, meta, ck, participants=pd.DataFrame(parts), participant_columns=pcols,
                     extra_files=[(f"samples/{p.name}", str(p)) for p in sorted((TMP / "samples").iterdir())])
shutil.rmtree(TMP, ignore_errors=True)
