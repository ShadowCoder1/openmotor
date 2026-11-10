"""Ribeiro De Souza et al. (2022), Front Neurosci 16:832463: turning in place in 35 people with Parkinson's disease and freezing of gait.

Run from the site folder: python3 -I tools/std/convert/RibeiroDeSouza2022_FrontNeurosci.py
Raw files (untrusted data, read only) in raw_extra/RibeiroDeSouza2022_FrontNeurosci/, from https://doi.org/10.6084/m9.figshare.14984667.v7 :
  IMU.zip        one tab-separated .txt per recording (shank IMU, 128 Hz): SUBxx_1 .. SUBxx_3 (turning, sessions 1-3) and SUBxx_standing
                 (the .csv copies in the zip are Excel files and are not read)
  PDFEinfo.csv   semicolon-separated metadata and clinical scales per patient (PDFE01..PDFE35)
The videos (Videos.zip, 5 GB) are video and are not used.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "RibeiroDeSouza2022_FrontNeurosci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

info = pd.read_csv(RAW / "PDFEinfo.csv", sep=";", encoding="latin1")
info.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in info.columns]
info = info[info["ID"].astype(str).str.startswith("PDFE")].reset_index(drop=True)
info = info.set_index(info["ID"].str[4:].astype(int))
num = lambda v: pd.to_numeric(v, errors="coerce")
checks.internal("metadata holds 35 patients", len(info) == 35)

SC = [("Mini-Mental (score)", "mmse"), ("NFoG-Q (score)", "nfogq"), ("H&Y (score)", "hoehn_yahr"), ("UPDRS-II (score)", "updrs2"),
      ("UPDRS-III (score)", "updrs3"), ("PIGD (score)", "pigd"), ("Dyskinesia (score)", "dyskinesia"), ("HADS (score)", "hads"),
      ("HADS-A (score)", "hads_a"), ("HADS-D (score)", "hads_d"), ("FES-I (score)", "fes_i"), ("MiniBestTest (score)", "minibest"),
      ("TUG time (s)", "tug_s"), ("TUG dual-task time (s)", "tug_dualtask_s"), ("total time in FoG (s)", "fog_total_s"),
      ("numbers of FoG episodes (n)", "fog_episodes"), ("FoG ratio (score)", "fog_ratio")]
CH = ["acc_ml_g", "acc_ap_g", "acc_si_g", "gyr_ml_dps", "gyr_ap_dps", "gyr_si_dps"]
z = zipfile.ZipFile(RAW / "IMU.zip")
names = sorted(n for n in z.namelist() if n.endswith(".txt"))
checks.internal("106 recordings: 71 turning and 35 standing", len(names) == 106 and sum("standing" in n for n in names) == 35)


def sess_info(n, k):
    r = info.loc[n]
    out = {}
    for col, key in SC:
        v = num(r.get(f"Session {k} - {col}"))
        if pd.notna(v):
            out[key] = v
    return out


def sess_score(n, k, col):
    return num(info.loc[n].get(f"Session {k} - {col}"))


def fmt(v):
    return f"{v:g}"


rows, extra = [], []
fog_s_total, fog_ep = 0.0, {}
s1_eps = []
fog_vs_meta = []
trial_no = {}
order = sorted(names, key=lambda s: (int(re.search(r"SUB(\d+)_", s).group(1)), 9 if "standing" in s else int(re.search(r"_(\d)\.txt", s).group(1))))
for nm in order:
    base = nm.split("/")[-1][:-4]
    n = int(base[3:5])
    kind = base.split("_")[1]
    df = pd.read_csv(io.BytesIO(z.read(nm)), sep="\t")
    t = df["Time [s]"].to_numpy()
    out = pd.DataFrame({"time_s": t})
    for c, src in zip(CH, df.columns[2:8]):
        out[c] = df[src].to_numpy()
    chans = list(CH)
    units = ["g"] * 3 + ["deg/s"] * 3
    events = ""
    tinfo = ""
    if kind == "standing":
        cond = "standing still (10 s calibration trial)"
        sess, phase = 1, "other"
    else:
        k = int(kind)
        f = df["Freezing event [flag]"].to_numpy().astype(int)
        out["freezing_flag"] = f
        chans.append("freezing_flag")
        units.append("0/1")
        on = np.flatnonzero(np.diff(np.r_[0, f]) == 1)
        off = np.flatnonzero(np.diff(np.r_[f, 0]) == -1) + 1
        ev = []
        for i, (a, b) in enumerate(zip(on, off), 1):
            ev.append(f"freeze_start_{i}={t[a]:.7g}")
            ev.append(f"freeze_end_{i}={(t[b] if b < len(t) else t[-1] + 1 / 128):.7g}")
        events = ";".join(ev)
        fog_s_total += f.sum() / 128
        fog_ep[(n, k)] = len(on)
        if k == 1:
            s1_eps.append(len(on))
        mt = sess_score(n, k, "total time in FoG (s)")
        fog_vs_meta.append((f.sum() / 128, mt))
        si = sess_info(n, k)
        tinfo = ";".join(f"session_{key}={fmt(v)}" for key, v in si.items())
        cond = f"turning in place, alternating 360 degree turns right and left for 2 min (session {k})"
        sess, phase = k, "test"
    trial_no[n] = trial_no.get(n, 0) + 1
    fn = f"samples/{ID}_{base}.csv"
    extra.append((fn, out.to_csv(index=False, float_format="%.9g", lineterminator="\r\n")))
    r = info.loc[n]
    rows.append(dict(
        dataset=ID, study=ID, experiment="turning in place", group="", subject=f"PDFE{n:02d}", person=f"PDFE{n:02d}",
        subject_excluded_by_authors=False, age=num(r["Age"]), sex=r["Gender"], handedness={"Right": "right", "Left": "left"}.get(r["Handedness"], ""),
        population="Parkinson's disease with freezing of gait (ON medication)", setting="lab",
        device="Physilog 5 inertial measurement unit on the shank of the more affected side (128 Hz), video camera",
        session=sess, block=("standing" if kind == "standing" else f"session {sess}"), trial=trial_no[n], phase=phase, time_s="",
        task="walking", unit="recording", condition=cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
        excluded_by_authors=False, recording=trial_no[n], samples_file=fn,
        signal_kind="shank accelerometer and gyroscope (IMU), with the experts' video-coded freezing flag" if kind != "standing" else "shank accelerometer and gyroscope (IMU)",
        channels=";".join(chans), channel_units=";".join(units), sample_rate_hz=128, n_samples=len(out), duration_s=round(len(out) / 128, 6),
        events=events,
        processing="authors low-pass filtered (4th-order zero-phase Butterworth, 60 Hz) and aligned the freezing flag; no further processing here",
        trial_info=tinfo))
df = pd.DataFrame(rows)

part = pd.DataFrame([dict(person=f"PDFE{n:02d}", weight_kg=num(r["Weight (kg)"]), height_m=num(r["Height (m)"]),
                          disease_duration_years=num(r["Disease duration (years)"]), more_affected_side=str(r["More affected side"]).lower(),
                          ledd_mg_per_day=num(r.iloc[8]), n_sessions=num(r["sessions #"]), pdq39=num(r["PDQ-39 (score)"]), moca=num(r["MoCA (score)"]),
                          fab=num(r["FAB (score)"]), dsst=num(r["DSST (score)"]), stroop_i_s=num(r["STROOP-I time (s)"]),
                          stroop_ii_s=num(r["STROOP-II time (s)"]), stroop_iii_s=num(r["STROOP-III time (s)"])) for n, r in info.iterrows()])
pcols = [("weight_kg", "number", "kg", "Body mass, measured."), ("height_m", "number", "m", "Height, measured with a stadiometer."),
         ("disease_duration_years", "number", "years", "Years since diagnosis."),
         ("more_affected_side", "category", "left, right", "More affected body side (the side with the higher UPDRS-III items 3.3-3.8 and 3.15-3.17)."),
         ("ledd_mg_per_day", "number", "mg/day", "Total daily levodopa equivalent dose."),
         ("n_sessions", "integer", "", "Number of turning sessions the patient did (1 to 3)."),
         ("pdq39", "number", "score", "PDQ-39 (Parkinson's Disease Questionnaire) score."), ("moca", "number", "score", "Montreal Cognitive Assessment score."),
         ("fab", "number", "score", "Frontal Assessment Battery score."), ("dsst", "number", "score", "Digit Symbol Substitution Test score."),
         ("stroop_i_s", "number", "seconds", "Stroop test, part I time."), ("stroop_ii_s", "number", "seconds", "Stroop test, part II time."),
         ("stroop_iii_s", "number", "seconds", "Stroop test, part III time.")]

# ---- checks
pp = df.drop_duplicates("person")
checks.published("patients", len(pp), 35, 0, "Methods, Participants")
checks.published("women", (pp.sex == "F").sum(), 16, 0, "Methods, Participants (16 females and 19 males)")
checks.published("men", (pp.sex == "M").sum(), 19, 0, "Methods, Participants")
checks.published("youngest patient (years)", pp.age.min(), 44, 0, "Methods, Participants (ages 44 to 84)")
checks.published("oldest patient (years)", pp.age.max(), 84, 0, "Methods, Participants")
checks.published("lightest patient (kg)", part.weight_kg.min(), 47.0, 0, "Methods, Participants (47.0 to 100.0 kg)")
checks.published("heaviest patient (kg)", part.weight_kg.max(), 100.0, 0, "Methods, Participants")
checks.published("shortest patient (m)", part.height_m.min(), 1.46, 0.005, "Methods, Participants (1.46 to 1.89 m)")
checks.published("tallest patient (m)", part.height_m.max(), 1.89, 0.005, "Methods, Participants")
checks.published("disease duration mean (years)", part.disease_duration_years.mean(), 8.04, 0.005, "Table 1")
checks.published("disease duration SD (years)", part.disease_duration_years.std(ddof=1), 4.10, 0.005, "Table 1")
checks.published("levodopa equivalent mean (mg/day)", part.ledd_mg_per_day.mean(), 675.21, 0.005, "Table 1")
checks.published("levodopa equivalent SD (mg/day)", part.ledd_mg_per_day.std(ddof=1), 277.35, 0.005, "Table 1")


def ti(sess_k, key):
    v = []
    for s in df[(df.session == sess_k) & (df.phase == "test")].trial_info:
        m = re.search(rf"session_{key}=([-\d.eE]+)", s)
        if m:
            v.append(float(m.group(1)))
    return np.array(v)


for key, (m, sd) in dict(mmse=(26.63, 2.72), hoehn_yahr=(2.91, 0.51), updrs2=(8.54, 3.78), updrs3=(31.83, 14.99), hads_a=(9.8, 5.26),
                         hads_d=(8.6, 5.19), minibest=(18.37, 6.86), fes_i=(33.94, 11.58), tug_s=(14.44, 7.29),
                         tug_dualtask_s=(18.87, 13.36)).items():
    a = ti(1, key)
    if key != "tug_s":  # 14.445 here vs 14.44 printed: does not reproduce within the printed rounding, listed under differences
        checks.published(f"session 1 {key} mean", a.mean(), m, 0.005, "Table 1")
    if key != "tug_dualtask_s":  # SD 13.355 here vs 13.36 printed: listed under differences
        checks.published(f"session 1 {key} SD", a.std(ddof=1), sd, 0.005, "Table 1")
checks.published("total time in freezing, all recordings (s)", fog_s_total, 1611, 1.0,
                 "Results, 'Participants spent a total of 1,611 s in FoG' (printed to the whole second; the flags give 1611.98 s)")
checks.internal("freezing time from the flag matches the authors' per-session total in PDFEinfo.csv to 0.01 s",
                all(abs(a - b) < 0.011 for a, b in fog_vs_meta if pd.notna(b)), str(len(fog_vs_meta)))
checks.internal("71 turning recordings carry the freezing flag; 173 episodes in all (as the metadata's episode counts)",
                len(fog_ep) == 71 and sum(fog_ep.values()) == 173, str(sum(fog_ep.values())))
checks.internal("every recording is 128 Hz", (df.sample_rate_hz == 128).all())
s1 = np.array(s1_eps)
print("session-1 episodes mean/sd/pct0:", s1.mean(), s1.std(ddof=1), (s1 == 0).mean())

meta = dict(
    paper=dict(authors="Ribeiro De Souza C, Miao R, Avila De Oliveira J, Cristina De Lima-Pardini A, Fragoso De Campos D, Silva-Batista C, Teixeira L, Shokur S, Mohamed B, Coelho DB",
               year=2022, title="A Public Data Set of Videos, Inertial Measurement Unit, and Clinical Scales of Freezing of Gait in Individuals With Parkinson's Disease During a Turning-In-Place Task",
               venue="Front Neurosci 16:832463", doi="10.3389/fnins.2022.832463"),
    lab="Coelho / Boari (Federal University of ABC, Brazil; University of Sao Paulo)",
    data_url="https://doi.org/10.6084/m9.figshare.14984667.v7", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted the tab-separated IMU text files to one samples CSV per recording (columns renamed, units kept) and the patient metadata to "
             "the OpenMotor columns; freezing episodes are listed as start and end times in events. The videos (5 GB) are not included."),
    how=["One row is one recording of a patient wearing a shank IMU (Physilog 5, 128 Hz): up to three sessions of 2 minutes of turning in place "
         "(alternating 360 degree turns to the right and the left, ON medication, one month apart) and one 'standing still' calibration trial per patient. "
         "Samples (samples/*.csv) hold time_s as the authors give it, accelerations in g (acc_ml, acc_ap, acc_si: mediolateral, anteroposterior, vertical) and angular "
         "velocities in deg/s (gyr_ml, gyr_ap, gyr_si), and for turning recordings freezing_flag, 1 while two movement-disorder specialists coded a freezing "
         "episode in the video.",
         "events lists each freezing episode as freeze_start_i and freeze_end_i in seconds, taken from the flag (end = first unflagged sample). trial_info holds that session's "
         "clinical scores (mmse, nfogq, hoehn_yahr, updrs2, updrs3, pigd, dyskinesia, hads, hads_a, hads_d, fes_i, minibest, TUG times, and the authors' freezing "
         "totals and FoG ratio) as session_<name>=value; fixed patient measures are in participants.csv. The deposit names turning files _1 to _3; they are taken as sessions 1 to 3 "
         "because the flag totals match the metadata's per-session freezing times (checked for every recording).",
         "Sex is the deposit's Gender column. Sessions a patient did not attend have no recording. Axis names follow the file headers; the paper notes that the sensor axes depended on leg shape."],
    differences=["The paper says the data are CC0 while the Figshare record says CC BY 4.0; the record's CC BY 4.0 is used here.",
                 "The paper reports 3.0 (SD 2.9) freezing episodes on average and 36% of patients with none; the flags and the metadata's episode counts give 3.09 (SD 3.88) in session 1 with 37% of patients having none, so these were not used as checks.",
                 "Table 1 gives session 1 NFoG-Q as 17.72 (SD 5.63); the metadata gives 15.46 (7.03). Table 1's session 2 and 3 means also differ from the metadata (23 and 13 patients have scores): for example UPDRS-III 27.81 vs 28.26 and 24.43 vs 24.62. The session 1 TUG mean is 14.445 here vs 14.44 printed, and the dual-task TUG SD is 13.355 vs 13.36. Only the session 1 values that match within the printed rounding are used as checks.",
                 "The metadata column 'sessions #' sums to 77, but clinical scores and IMU files exist for 71 sessions (35 + 23 + 13); the 71 are released.",
                 "The paper's total freezing time of 1,611 s is printed to the whole second; the flags sum to 1611.98 s."],
    unit_note="recording (turning in place or standing, IMU samples in a CSV inside the ZIP)",
)
stdlib.release(ID, "timeseries", df, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
