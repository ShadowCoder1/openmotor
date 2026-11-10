"""van der Zee, Mundinger & Kuo (2022), Sci Data 9:704: Level 1 raw C3D files of 10 healthy adults walking on an instrumented
split-belt treadmill in 33 speed / step length / step frequency / step width conditions (plus one static standing trial each).
One row per recording stream: the 3D marker trajectories (120 Hz) and the two force plates' channels (1200 Hz) of a trial are
separate rows, each with its own samples CSV.

Run from the site folder: python3 -I tools/std/convert/vanderZee2022_SciData.py
Raw files (untrusted data, read only) in raw_extra/vanderZee2022_SciData/: level1_c3d.zip (figshare 19718500),
trial_look_up.xlsx (figshare 19714639). Paper text: Europe PMC PMC9669008.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
import vanderZee2022_SciData_c3d as c3d  # noqa: E402

ID = "vanderZee2022_SciData"
RAW = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "tmp_samples" / ID
checks = stdlib.Checks()

# ---- trial look-up table of the deposit (figshare 19714639)
import openpyxl  # noqa: E402
LOOK = {}
for r in openpyxl.load_workbook(RAW / "trial_look_up.xlsx", data_only=True).active.iter_rows(min_row=2, values_only=True):
    if r[0] is not None:
        LOOK[int(r[0])] = (float(r[1]), str(r[2]))
assert sorted(LOOK) == list(range(1, 34))

# ---- paper Table 2: targets as multiples of the participant's preferred step length s* and frequency f* (None = self-selected)
SL = {1: 1.00, 3: .56, 4: 1.00, 6: .72, 7: 1.00, 9: .88, 10: 1.28, 12: 1.00, 13: 1.44, 15: 1.00, 17: 1.43, 18: 1.25, 19: 1.11, 20: 1.00, 21: 1.00,
      22: 1.00, 23: .91, 24: .83, 25: .77, 26: 1.00, 27: 1.00, 28: 1.00, 29: 1.00, 30: 1.00, 31: 1.00, 33: 1.12}
SF = {1: .56, 3: 1.00, 4: .72, 6: 1.00, 7: .88, 9: 1.00, 10: 1.00, 12: 1.28, 13: 1.00, 15: 1.44, 17: .70, 18: .80, 19: .90, 20: 1.00, 21: 1.00,
      22: 1.00, 23: 1.10, 24: 1.20, 25: 1.30, 26: 1.00, 27: 1.00, 28: 1.00, 29: 1.00, 30: 1.00, 31: 1.12, 33: 1.00}
SW = {26: 0.0, 27: 0.1, 28: 0.2, 29: 0.3, 30: 0.4}

# ---- paper Table 1 (anthropometry; age, mass, height, leg length, foot width, preferred step length, preferred step frequency)
T1 = {1: (25, 81.8, 1.73, .89, .13, .71, 1.77), 2: (21, 57.3, 1.64, .94, .09, .60, 2.07), 3: (23, 97.5, 1.90, 1.04, .11, .71, 1.77),
      4: (21, 57.0, 1.73, .88, .10, .68, 1.83), 5: (21, 56.7, 1.63, .86, .09, .65, 1.93), 6: (24, 72.6, 1.83, .94, .10, .74, 1.70),
      7: (28, 86.2, 1.89, .99, .10, .74, 1.70), 8: (27, 88.6, 1.85, .99, .10, .74, 1.70), 9: (22, 77.0, 1.78, .94, .10, .68, 1.83),
      10: (23, 60.3, 1.60, .86, .10, .65, 1.93)}
T1_RAW = {1: 33, 2: 33, 3: 33, 4: 33, 5: 33, 6: 33, 7: 32, 8: 33, 9: 33, 10: 33}

zf = zipfile.ZipFile(RAW / "level1_c3d.zip")
FILES = []
for info in zf.infolist():
    m = re.fullmatch(r"Level 1 - C3D files/p(\d+)_c3dfiles/p\d+_(standing(?:_\d+)?|trial(\d+)(?:_(\d+))?)\.c3d", info.filename)
    if not m:
        assert info.filename.endswith("/"), info.filename
        continue
    p = int(m.group(1))
    if m.group(2).startswith("standing"):
        FILES.append((p, 0, 0, info.filename))
    else:
        FILES.append((p, int(m.group(3)), int(m.group(4) or 1), info.filename))
FILES.sort()
assert len(FILES) == 340

MK_FMT, V_FMT = "%.1f", "%.4f"
CAL_SETS = {}


def cal_of(params):
    c = np.array(params[("FORCE_PLATFORM", "CAL_MATRIX")], dtype=float)
    assert len(c) == 72
    key = tuple(np.round(c, 4))
    if key not in CAL_SETS:
        CAL_SETS[key] = (chr(ord("A") + len(CAL_SETS)), c, params)
    return CAL_SETS[key][0]


def load(fn):
    b = zf.read(fn)
    h, params, en, dec = c3d.parse(b)
    pts, an, n = c3d.data(b, h, params, en, dec)
    return h, params, pts, an, n


def tag(trial, rep):
    return "standing" if trial == 0 else f"trial{trial:02d}" + ("b" if rep > 1 else "")


# ---- pass 1: metadata of every recording stream
streams = []  # dict per row
labels_std = None
for p, trial, rep, fn in FILES:
    h, params, pts, an, n = load(fn)
    labels = params[("POINT", "LABELS")]
    if len(labels) == 35 and labels_std is None:
        labels_std = labels
    assert h["rate"] == 120.0
    has_mk = h["npts"] > 0 and not np.isnan(pts).all()
    base = dict(p=p, trial=trial, rep=rep, fn=fn)
    if has_mk:
        streams.append(dict(base, kind="markers", n=n, labels=labels, nmiss=float(np.isnan(pts[:, :, 0]).mean())))
    else:
        streams.append(dict(base, kind="nomarkers", n=n, labels=labels))
    if an is not None and an.shape[0]:
        assert params[("ANALOG", "RATE")] == [1200.0] and params[("ANALOG", "LABELS")][-1] == "SINE", params[("ANALOG", "LABELS")]
        streams.append(dict(base, kind="forces", n=an.shape[0], cal=cal_of(params)))
    else:
        streams.append(dict(base, kind="noforces", n=0))
assert len(CAL_SETS) <= 3, len(CAL_SETS)

FCH = [f"{pl}_{c}" for pl in ("plate1", "plate2") for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")] + ["sine"]
FUNITS = ["V"] * 13
person_n = {}
rows = []
extra_plan = []
rec_no = {}
trial_idx = {}
for s in streams:
    if s["kind"] in ("nomarkers", "noforces"):
        continue
    p, trial, rep = s["p"], s["trial"], s["rep"]
    code = f"{ID}_S{p:04d}"
    tg = tag(trial, rep)
    if (p, trial, rep) not in trial_idx:
        trial_idx[(p, trial, rep)] = sum(1 for k in trial_idx if k[0] == p) + 1
    rec_no[p] = rec_no.get(p, 0) + 1
    fn = f"samples/{code}_{tg}_{'markers' if s['kind'] == 'markers' else 'forces'}.csv"
    spd, cond = LOOK[trial] if trial else (None, "static standing")
    info = []
    if trial:
        info += [f"trial_number={trial}", f"belt_speed_m_s={spd:g}"]
        info += [f"step_length_x_pref={SL[trial]:g}" if trial in SL else "step_length=self-selected",
                 f"step_frequency_x_pref={SF[trial]:g}" if trial in SF else "step_frequency=self-selected",
                 f"step_width_m={SW[trial]:g}" if trial in SW else "step_width=self-selected"]
        if rep > 1:
            info += ["repeat_file=2"]
    else:
        info += ["standing_reference=1"]
    if s["kind"] == "markers":
        sk, ch, cu, rate = "3D marker positions", ";".join(f"{l}_{a}" for l in s["labels"] for a in "XYZ"), ";".join(["mm"] * (3 * len(s["labels"]))), 120
        proc = "none (raw Level 1 C3D; coordinates as stored in the C3D, in the laboratory frame); empty where the marker was not tracked"
    else:
        sk, ch, cu, rate = ("ground reaction force and moment channels of two force plates (one per belt) and a sine synchronisation channel",
                            ";".join(FCH), ";".join(FUNITS), 1200)
        proc = "none (raw Level 1 C3D; plate output voltages, not yet converted to N and Nmm: see calibration/force_platform_calibration.csv)"
        info += [f"calibration_set={s['cal']}"]
    rows.append(dict(
        dataset=ID, study=ID, experiment="treadmill walking at 33 speed, step length, step frequency and step width conditions",
        group="", subject=f"p{p}", person=f"p{p}", subject_excluded_by_authors=False, age=T1[p][0], sex="", handedness="", population="healthy",
        setting="lab",
        device="instrumented split-belt treadmill (two Bertec force plates, 1200 Hz) and Motion Analysis 3D motion capture (120 Hz)",
        session=1, block="standing reference" if trial == 0 else f"trial {trial}", trial=trial_idx[(p, trial, rep)],
        phase="baseline" if trial == 0 else "test", time_s="", task="walking", unit="recording", condition=cond,
        pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=rec_no[p], samples_file=fn, signal_kind=sk, channels=ch, channel_units=cu, sample_rate_hz=rate, n_samples=s["n"],
        duration_s=s["n"] / rate, events="", processing=proc, trial_info=";".join(info)))
    extra_plan.append((fn, s))
df = pd.DataFrame(rows)

# ---- published checks
P = checks.published
P("participants", df.person.nunique(), 10, 0, "Abstract (N = 10)")
P("distinct experimental trial conditions", df[df.condition != "static standing"].trial_info.str.extract(r"trial_number=(\d+)")[0].nunique(), 33, 0,
  "Abstract / Table 2 (33 trials)")
for p in range(1, 11):
    tr = df[(df.person == f"p{p}") & (df.condition != "static standing")].trial_info.str.extract(r"trial_number=(\d+)")[0].nunique()
    P(f"participant {p}: trials with a raw C3D file", tr, T1_RAW[p], 0, "Table 1 (# Raw files)")
P("total raw C3D trials (10 x 33 minus one missing file)", sum(T1_RAW[p] for p in range(1, 11)), 329, 0, "Table 1; Missing data (one C3D file missing)")
fr = df[(df.sample_rate_hz == 1200)]
mk = df[(df.sample_rate_hz == 120)]
P("force plate sampling rate (Hz)", fr.sample_rate_hz.median(), 1200, 0, "Instrumentation and data collection")
P("motion capture sampling rate (Hz)", mk.sample_rate_hz.median(), 120, 0, "Instrumentation and data collection")
P("typical trial length (s)", mk[mk.condition != "static standing"].duration_s.median(), 60, 0, "Experimental protocol / Abstract (60 s per trial)")
P("markers per participant 1 trial (Table 3 rows minus 2 virtual Helen Hayes points)", mk[mk.person == "p1"].channels.iloc[0].count(";") // 3 + 1, 35, 0,
  "Table 3 (37 rows, HHL and HHR virtual)")
P("markers of participant 9 (upper-body acromion, elbow and wrist markers absent)", mk[mk.person == "p9"].channels.iloc[0].count(";") // 3 + 1, 35 - 6, 0,
  "Missing data (subject 9 lacks acromion, elbow epicondyle and wrist markers)")
a = np.array([T1[p][0] for p in T1], float)
P("age mean (years)", a.mean(), 23.5, 0.05, "Participants (23.5 +- 2.5); Table 1 values")
P("age SD (years)", a.std(ddof=1), 2.5, 0.05, "Participants; Table 1 values")
m = np.array([T1[p][1] for p in T1], float)
P("body mass mean (kg)", m.mean(), 73.5, 0.05, "Participants (73.5 +- 15 kg); Table 1 values")
P("body mass SD (kg)", m.std(ddof=1), 15, 0.5, "Participants; Table 1 values")
hgt = np.array([T1[p][2] for p in T1], float)
P("height mean (m)", hgt.mean(), 1.76, 0.005, "Participants (1.76 +- 0.11 m); Table 1 values")
P("height SD (m)", hgt.std(ddof=1), 0.11, 0.005, "Participants; Table 1 values")
P("speeds 0.7 to 2.0 m/s: lowest", min(v[0] for v in LOOK.values()), 0.7, 0, "Abstract (0.7-2.0 m/s)")
P("speeds 0.7 to 2.0 m/s: highest", max(v[0] for v in LOOK.values()), 2.0, 0, "Abstract (0.7-2.0 m/s)")
P("step widths range from 0 to 0.4 m: highest", max(SW.values()), 0.4, 0, "Abstract (0-0.4 m)")

none_mk = [(s["p"], s["trial"]) for s in streams if s["kind"] == "nomarkers"]
none_fo = [(s["p"], s["trial"]) for s in streams if s["kind"] == "noforces"]
checks.internal("one participant-trial without usable markers (participant 6 trial 31, forces only)", none_mk == [(6, 31)], str(none_mk))
checks.internal("recordings without force channels listed", True, str(none_fo))
short = [(s["p"], s["trial"], s["n"]) for s in streams if s["kind"] == "markers" and s["n"] < 7200 and s["trial"]]
checks.internal("shortened trials listed", True, str(short))
checks.internal("participant 6 trial 21 stopped early (less than 60 s)", any(x[:2] == (6, 21) for x in short), str(short))
checks.internal("participant 7 trial 24 has no file", not any(f[0] == 7 and f[1] == 24 for f in FILES))
checks.internal("340 C3D files read", len(FILES) == 340)
checks.internal("every C3D file gives one marker or no-marker stream", sum(1 for s in streams if s["kind"] in ("markers", "nomarkers")) == 340)

# ---- calibration and geometry
buf = io.StringIO()
buf.write("calibration_set,plate,row_output,col_Fx_V,col_Fy_V,col_Fz_V,col_Mx_V,col_My_V,col_Mz_V\r\n")
GEO = io.StringIO()
GEO.write("calibration_set,plate,corner,x_mm,y_mm,z_mm\r\n")
for key, (name, c, params) in CAL_SETS.items():
    for pl in range(2):
        M = c[36 * pl:36 * (pl + 1)].reshape(6, 6, order="F")
        for i, out in enumerate(["Fx_N", "Fy_N", "Fz_N", "Mx_Nmm", "My_Nmm", "Mz_Nmm"]):
            buf.write(f"{name},{pl + 1},{out}," + ",".join(f"{v:.8g}" for v in M[i]) + "\r\n")
        assert len(params[("FORCE_PLATFORM", "CORNERS")]) == 24
        co = np.array(params[("FORCE_PLATFORM", "CORNERS")]).reshape(2, 4, 3)
        for k in range(4):
            GEO.write(f"{name},{pl + 1},{k + 1}," + ",".join(f"{v:.6g}" for v in co[pl][k]) + "\r\n")
checks.internal("number of calibration sets", len(CAL_SETS) in (1, 2, 3), str(len(CAL_SETS)))


def gen():
    TMP.mkdir(parents=True, exist_ok=True)
    cache = {}
    try:
        for fn, s in extra_plan:
            key = s["fn"]
            if key not in cache:
                cache = {key: load(key)}
            h, params, pts, an, n = cache[key]
            if s["kind"] == "markers":
                t = np.arange(n) / 120.0
                out = pd.DataFrame(pts.reshape(n, -1), columns=[f"{l}_{a}" for l in s["labels"] for a in "XYZ"])
                out.insert(0, "time_s", [f"{x:.5f}" for x in t])
                out.to_csv(TMP / "x.csv", index=False, float_format=MK_FMT, lineterminator="\r\n")
            else:
                t = np.arange(an.shape[0]) / 1200.0
                out = pd.DataFrame(an, columns=FCH)
                out.insert(0, "time_s", [f"{x:.5f}" for x in t])
                out.to_csv(TMP / "x.csv", index=False, float_format=V_FMT, lineterminator="\r\n")
            yield fn, str(TMP / "x.csv")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


part = pd.DataFrame([dict(person=f"p{p}", body_mass_kg=T1[p][1], height_m=T1[p][2], leg_length_m=T1[p][3], foot_width_m=T1[p][4],
                          preferred_step_length_m=T1[p][5], preferred_step_frequency_hz=T1[p][6]) for p in T1])
pcols = [("body_mass_kg", "number", "kg", "Body mass, from Table 1 of the paper."),
         ("height_m", "number", "m", "Height, from Table 1 of the paper."),
         ("leg_length_m", "number", "m", "Leg length (floor to greater trochanter), from Table 1 of the paper."),
         ("foot_width_m", "number", "m", "Foot width, from Table 1 of the paper."),
         ("preferred_step_length_m", "number", "m", "Preferred step length s* at 1.25 m/s, from Table 1 of the paper."),
         ("preferred_step_frequency_hz", "number", "Hz", "Preferred step frequency f* at 1.25 m/s, from Table 1 of the paper.")]

extra_static = [("calibration/force_platform_calibration.csv", buf.getvalue().encode()), ("calibration/force_platform_corners.csv", GEO.getvalue().encode())]


class Files:
    def __iter__(self):
        yield from extra_static
        yield from gen()


meta = dict(
    paper=dict(authors="van der Zee TJ, Mundinger EM, Kuo AD", year=2022,
               title="A biomechanics dataset of healthy human walking at various speeds, step lengths and step widths",
               venue="Sci Data 9(1):704", doi="10.1038/s41597-022-01817-1"),
    lab="Kuo (University of Calgary); data collected at the University of Michigan",
    data_url="https://doi.org/10.6084/m9.figshare.c.5897423", license="CC0",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Only the Level 1 raw C3D files (340 files) were converted; the Level 2 inverse dynamics (CMO) and Level 3 MATLAB files were not used. "
             "Each C3D file became up to two sample CSVs (marker trajectories at 120 Hz, force plate channels at 1200 Hz), named with the OpenMotor subject codes. "
             "Marker coordinates are rounded to 0.1 mm and plate voltages to 0.0001 V to keep the ZIP near 1 GB (the C3D integer resolution is 0.049 mm and 0.00244 V, "
             "so the voltages lose nothing, markers lose at most 0.05 mm); nothing was filtered, resampled or calibrated. Anthropometry (Table 1 of the paper) is in participants.csv and age in the main table."),
    how=["One row is one recording stream of one trial: the 35 (29 for one participant) reflective marker trajectories, or the 13 force platform channels. "
         "Every participant walked 33 conditions of 60 s on a split-belt treadmill (trial numbers 1 to 33, conditions in the deposit's trial look-up table and Table 2 of the paper) "
         "and has one static standing trial; the order in which trials were run was randomized per participant and is not recorded in the deposit, so trial counts the rows in trial-number order (standing first).",
         "Marker samples (samples/<subject>_trialNN_markers.csv, 120 Hz) hold X, Y, Z of each marker in mm, in the laboratory frame stored in the C3D (X_SCREEN +Y, Y_SCREEN +Z in the file header); empty cells are frames in which the marker was not tracked.",
         "Force samples (samples/<subject>_trialNN_forces.csv, 1200 Hz) hold the six channels of plate 1 and plate 2 (Fx, Fy, Fz, Mx, My, Mz) and a sine synchronisation channel, all in volts as stored in the C3D "
         "(the C3D labels them N and Nmm, but the file is a type 4 force platform whose output is converted with the calibration matrix: forces in N and moments in Nmm are the 6x6 matrix of the plate times the six voltages). "
         "calibration/force_platform_calibration.csv gives that matrix per plate (calibration_set in trial_info), arranged as row output (Fx_N .. Mz_Nmm) by column voltage (Fx .. Mz) after reading the C3D's 36 values in the C3D standard (column-major) order; "
         "calibration/force_platform_corners.csv gives the plate corners in mm. A 60 s trial then has a mean Fz of about -800 N summed over both plates (downward is negative); the plates were not tared to the participant's weight, so the sums differ from body weight by up to about 15 percent.",
         "Some recordings are shorter than 60 s or lack a stream, as described in the paper's Missing data section: participant 7 has no trial 24, participant 6 trial 31 has force data but no markers, participant 6 trial 21 stopped early, "
         "participant 9 has no acromion, elbow and wrist markers. Further files in the deposit that the paper does not describe: participant 8 has a second file for trial 28 (shown with the suffix b), participant 10 trial 21 and participant 3's standing trial have no force channels, "
         "and participant 3 trial 1 and participant 10 trial 27 are shorter than 60 s (n_samples gives each length)."],
    differences=["The paper's Table 1 lists 33 raw files for participant 8; the deposit has 34 files for participant 8 (a repeat file for trial 28 as well). The paper's inverse-dynamics outlier comment (participant 10) applies to Level 2/3 data, not to these raw files, so participant 10 is kept.",
                 "The deposit's license is CC0 1.0; the data descriptor article itself is CC BY 4.0.",
                 "Body mass from the force plates is not compared with Table 1 because the plates were not tared (see how this file was made)."],
    unit_note="recording stream (marker trajectories or force plate channels of one trial)",
)
stdlib.release(ID, "timeseries", df, meta, checks, participants=part, participant_columns=pcols, extra_files=Files())
