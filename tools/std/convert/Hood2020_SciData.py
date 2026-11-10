"""Hood, Ishmael, Gunnell, Foreman & Lenzi (2020) Sci Data 7:150: kinematics and kinetics of 18 people with an above-knee
amputation walking on an instrumented treadmill at five speeds (Figshare collection 10.6084/m9.figshare.c.4962305, CC BY 4.0).

Raw files (raw_extra/Hood2020_SciData/): the 18 per-subject MATLAB files TF##_DATA.mat (19-31 MB each) and
SubjectInformation.xlsx. Each TF##_DATA.mat sits inside that subject's 0.36-1.3 GB zip together with the Vicon workspace
(marker data, video-free C3D trials); only the MATLAB file was read out of each zip with HTTP range requests, so the 13 GB
of the deposit were not downloaded (the Vicon workspaces are not converted). The MATLAB files are plain structs (no classes).
Run from the site folder: python3 -I tools/std/convert/Hood2020_SciData.py"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Hood2020_SciData"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
DEVICE = ("Bertec dual-belt instrumented treadmill, 10-camera Vicon motion capture (67 markers), Visual 3D inverse dynamics; "
          "joint angles and moments of the prosthesis (ipsilateral) and sound (contralateral) leg")
JOINTS = ["ankle", "knee", "hip"]
VARS = [("position", "deg"), ("moment", "Nm/kg")]
CH = [f"{j}_{v}" for v, _ in VARS for j in JOINTS]
UNITS = [u for _, u in VARS for _ in JOINTS]
SIDE = {"ipsilateral": "prosthesis side (ipsilateral)", "contralateral": "sound side (contralateral)"}

# ---- subject information (SubjectInformation.xlsx, same as Table 1 of the paper)
raw_info = pd.read_excel(raw / "SubjectInformation.xlsx", header=1, engine="openpyxl")
raw_info = raw_info.loc[:, [c for c in raw_info.columns if not str(c).startswith("Unnamed")]]
raw_info.columns = [str(c).strip() for c in raw_info.columns]
raw_info = raw_info[raw_info.iloc[:, 0].astype(str).str.startswith("TF")].reset_index(drop=True)
info = {r["Subject Code"].strip(): r for _, r in raw_info.iterrows()}
subs = sorted(info, key=lambda s: int(s[2:]))
assert len(subs) == 18, subs

# Table 1 of the paper (age, sex, mass kg, height m, side, K-level), for the published checks
T1 = {"TF01": (26, "Male", 64.9, 1.78, "Right", "K3"), "TF02": (79, "Male", 126.1, 1.75, "Right", "K2"), "TF05": (72, "Male", 79.4, 1.65, "Left", "K2"),
      "TF06": (60, "Male", 86.6, 1.70, "Left", "K2"), "TF07": (49, "Male", 102.1, 1.91, "Left", "K3"), "TF08": (42, "Male", 95.3, 1.85, "Right", "K3"),
      "TF09": (65, "Male", 69.4, 1.70, "Left", "K2"), "TF10": (57, "Female", 58.5, 1.65, "Left", "K2"), "TF11": (51, "Male", 70.3, 1.68, "Right", "K3"),
      "TF12": (59, "Male", 99.8, 1.83, "Left", "K2"), "TF13": (61, "Male", 88.5, 1.88, "Left", "K3"), "TF14": (51, "Male", 108.9, 1.73, "Right", "K2"),
      "TF15": (23, "Female", 68.0, 1.75, "Right", "K3"), "TF16": (36, "Male", 100.2, 1.80, "Left", "K3"), "TF17": (38, "Male", 104.3, 1.91, "Left", "K3"),
      "TF18": (69, "Male", 129.3, 1.73, "Right", "K2"), "TF19": (30, "Female", 59.0, 1.60, "Left", "K3"), "TF20": (59, "Male", 120.2, 1.78, "Left", "K2")}


def clean(v):
    return "" if (isinstance(v, float) and np.isnan(v)) else str(v).strip()


rows, files, strides_total = [], [], 0
n_speeds, avg_ok, std_ok, nan_total, k_speed = {}, True, True, 0, {}
for si, sid in enumerate(subs, start=1):
    m = sio.loadmat(raw / f"{sid}_DATA.mat", squeeze_me=True, struct_as_record=False)[sid]
    r = info[sid]
    d = m.data
    speeds = list(d._fieldnames)
    n_speeds[sid] = len(speeds)
    k_speed[sid] = [float(s.replace("speed_", "").replace("p", ".")) for s in speeds]
    rec = 0
    for sp in speeds:
        spd = float(sp.replace("speed_", "").replace("p", "."))
        node = getattr(d, sp)
        for side in ("ipsilateral", "contralateral"):
            blocks = [getattr(getattr(getattr(node, side), j), v).raw for v, _ in VARS for j in JOINTS]
            avgs = [getattr(getattr(getattr(node, side), j), v).avg for v, _ in VARS for j in JOINTS]
            sds = [getattr(getattr(getattr(node, side), j), v).stdev for v, _ in VARS for j in JOINTS]
            ns = {b.shape for b in blocks}
            assert len(ns) == 1 and list(ns)[0][0] == 1001, ns
            n = list(ns)[0][1]
            nan_total += int(sum(np.isnan(b).sum() for b in blocks))
            for b, a, s in zip(blocks, avgs, sds):
                avg_ok &= bool(np.allclose(np.nanmean(b, axis=1), a, rtol=1e-4, atol=1e-6))
                std_ok &= bool(np.allclose(np.nanstd(b, axis=1, ddof=1), s, rtol=1e-3, atol=1e-6))
            for k in range(n):
                rec += 1
                strides_total += 1
                fn = f"samples/P{si:02d}_{sp.replace('speed_', 'v')}_{side[:5]}_{k + 1:03d}.csv"
                rows.append(dict(
                    dataset=DS, study=DS, experiment="", group=clean(r["K-Level"]), subject=sid, subject_excluded_by_authors=False,
                    age=float(r["Age"]), sex={"Male": "M", "Female": "F"}[clean(r["Gender"])], handedness="",
                    population="unilateral above-knee (transfemoral) amputation", setting="lab", device=DEVICE, session=1,
                    block=sp, trial=rec, phase="test", time_s="", task="walking", unit="stride",
                    condition=f"treadmill {spd:g} m/s, {SIDE[side]}", pert_type="none", pert_value=0.0, pert_unit="",
                    feedback="normal", excluded_by_authors=False, recording=rec, samples_file=fn,
                    signal_kind="joint angle (position) and joint moment of the ankle, knee and hip, one stride normalized in time",
                    channels=";".join(CH), channel_units=";".join(UNITS), sample_rate_hz="", n_samples=1001, duration_s="",
                    events="", processing=("Visual 3D inverse dynamics; marker data low-pass filtered at 6 Hz and force-plate data at 15 Hz; "
                                           "each stride (heel strike to heel strike) resampled to 1001 samples, 0-100% of the stride; moments in Nm/kg")))
                files.append((fn, [b[:, k] for b in blocks]))
    del m
df = pd.DataFrame(rows).reindex(columns=[c[1] for c in stdlib.columns_for("timeseries") if c[1] != "person"])


def extra():
    head = "gait_cycle_pct," + ",".join(CH) + "\n"
    pct = np.linspace(0, 100, 1001)
    for fn, cols in files:
        body = [head]
        a = np.column_stack([pct] + cols)
        body.extend(",".join(format(x, ".8g") for x in row) + "\n" for row in a)
        yield fn, "".join(body)


# ---- participants.csv
pcols = [("mass_kg", "number", "kg", "Body mass as reported in SubjectInformation.xlsx (includes the prosthesis, per the paper's Methods)."),
         ("height_m", "number", "m", "Height."), ("amputation_side", "category", "Left, Right", "Side of the amputation (the ipsilateral, prosthesis side)."),
         ("amputation_etiology", "text", "", "Reason for amputation, as in the deposit."),
         ("amputation_years_as_labelled", "number", "years", "The deposit's 'Age of Amputation (yrs)'; the values behave as years since amputation (the study required at least one year since amputation and the smallest value is 1), not as age at amputation."),
         ("k_level", "category", "K2, K3", "Medicare Functional Classification Level, which sets the speeds walked (K2 0.4-0.8 m/s, K3 0.6-1.4 m/s)."),
         ("knee_prosthesis", "text", "", "Prescribed prosthetic knee."), ("foot_prosthesis", "text", "", "Prescribed prosthetic foot."),
         ("socket_suspension", "text", "", "Socket suspension."), ("treadmill_training", "text", "", "Whether the participant had treadmill training before the session and the number of training sessions."),
         ("handrail_use", "text", "", "Whether handrails were used.")]
part = []
for sid in subs:
    r = info[sid]
    part.append(dict(person=sid, mass_kg=float(r["Mass (kg)"]), height_m=float(r["Height (m)"]), amputation_side=clean(r["Amputation Side"]),
                     amputation_etiology=clean(r["Reason for Amputation"]), amputation_years_as_labelled=float(r["Age of Amputation"]),
                     k_level=clean(r["K-Level"]), knee_prosthesis=clean(r["Knee Prosthesis"]), foot_prosthesis=clean(r["Foot Prosthesis"]),
                     socket_suspension=clean(r["Socket Suspension"]), treadmill_training=clean(r["Treadmill Training?"]), handrail_use=clean(r["Handrails?"])))

# ---- checks
ck.internal("every stride of every speed and side is kept: one row per column of each raw array", len(df) == strides_total, f"{strides_total} strides")
ck.internal("the authors' average curve equals the mean of the raw stride columns for every joint, variable, speed and side", avg_ok)
ck.internal("the authors' stdev curve equals the sample SD of the raw stride columns", std_ok)
ck.internal("no empty samples in the raw arrays", nan_total == 0, f"{nan_total} NaN")
ipsi = df.condition.str.contains("ipsi").sum()
ck.internal("equal numbers of ipsilateral and contralateral strides per speed", ipsi * 2 == len(df), f"{ipsi} each")
ck.published("participants", len(subs), 18, 0, "Methods, para 1")
ck.published("K3 participants (full community ambulators)", sum(clean(info[s]["K-Level"]) == "K3" for s in subs), 9, 0, "Background & Summary")
ck.published("K2 participants (limited community ambulators)", sum(clean(info[s]["K-Level"]) == "K2" for s in subs), 9, 0, "Background & Summary")
ck.published("women", sum(clean(info[s]["Gender"]) == "Female" for s in subs), 3, 0, "Table 1")
ck.published("sum of ages, years (Table 1)", sum(float(info[s]["Age"]) for s in subs), sum(v[0] for v in T1.values()), 0, "Table 1")
ck.published("sum of body masses, kg (Table 1)", sum(float(info[s]["Mass (kg)"]) for s in subs), sum(v[2] for v in T1.values()), 0.05, "Table 1")
ck.published("sum of heights, m (Table 1)", sum(float(info[s]["Height (m)"]) for s in subs), sum(v[3] for v in T1.values()), 0.005, "Table 1")
ck.published("participants with a left-side amputation", sum(clean(info[s]["Amputation Side"]) == "Left" for s in subs), sum(v[4] == "Left" for v in T1.values()), 0, "Table 1")
spd_ok = all(k_speed[s] == ([0.4, 0.5, 0.6, 0.7, 0.8] if clean(info[s]["K-Level"]) == "K2" else [0.6, 0.8, 1.0, 1.2, 1.4]) or s == "TF02" for s in subs)
ck.published("K2 participants walked at 0.4-0.8 m/s and K3 at 0.6-1.4 m/s (all but one subject complete)", int(spd_ok), 1, 0, "Abstract and Table 1 note")
ck.published("TF02 walked 4 of the 5 speeds (could not walk the last speed, 0.8 m/s)", n_speeds["TF02"], 4, 0, "Table 1 note")
ck.published("total walking speeds over subjects (17 x 5 + 4)", sum(n_speeds.values()), 89, 0, "Abstract and Table 1 note")

meta = dict(
    paper=dict(authors="Hood S, Ishmael MK, Gunnell A, Foreman KB, Lenzi T", year=2020,
               title="A kinematic and kinetic dataset of 18 above-knee amputees walking at various speeds",
               venue="Sci Data 7:150", doi="10.1038/s41597-020-0494-7"),
    lab="Lenzi (University of Utah)", data_url="https://doi.org/10.6084/m9.figshare.c.4962305",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept every stride of the deposit's processed MATLAB files (joint angle and moment curves of the ankle, knee and hip, "
             "both legs, all speeds), one CSV of 1001 samples per stride, with participants coded. The authors' average and SD curves "
             "(computable from the strides), the raw Vicon marker and force-plate trials (C3D) and the calibration trials are not included."),
    how=["One row is one stride of one leg at one treadmill speed; its samples file has one line per 0.1% of the stride (gait_cycle_pct, "
         "0 to 100, 1001 samples; the deposit gives the strides already normalized in time, so there is no time axis in seconds and "
         "sample_rate_hz and duration_s are empty) and six columns: ankle, knee and hip joint position (degrees) and moment (Nm/kg), "
         "for the prosthesis side (ipsilateral) or the sound side (contralateral), as named in condition.",
         "Rows run through the subject's speeds in the deposit's order (slowest first), the ipsilateral strides and then the contralateral "
         "strides of each speed, in the order of the deposit's columns. The strides of the two sides are not paired. The deposit does not "
         "state the joint axes or sign conventions beyond the labels 'position' and 'moment' (the plots label them Joint Position (deg) "
         "and Joint Moment (Nm/kg)), so values are exactly as shared.",
         "Group is the K-level: K2 walked at 0.4, 0.5, 0.6, 0.7 and 0.8 m/s and K3 at 0.6, 0.8, 1.0, 1.2 and 1.4 m/s on a dual-belt "
         "instrumented treadmill, five trials of at least ten strides per speed; one participant (K2) could not walk the last speed."],
    unit_note="stride of one leg (samples in samples/)",
    differences=["The paper reports no gait values, only participant and design numbers, so the published checks are the Table 1 participant "
                 "data, the K-level split and the speed design. The deposit's README says a manuscript was still in revision when the data "
                 "were posted; the paper is the Scientific Data article above."],
)
stdlib.release(DS, "timeseries", df, meta, ck, participants=part, participant_columns=pcols, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
