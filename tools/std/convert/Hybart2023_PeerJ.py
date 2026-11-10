"""Hybart, Villancio-Wolter & Ferris (2023) PeerJ 11:e15775: walking with bilateral ankle exoskeletons (Dephy ExoBoots) under
proportional myoelectric control, on a treadmill and on an outdoor course, 12 young adults. One row per recording.

Raw files (Figshare 10.6084/m9.figshare.22572868, CC BY 4.0): EMG_EXO_Data.zip (one CSV per participant and condition, read from
the zip without extracting), Subject_Information.xlsx, Readme_Exo_EMG.docx (read as text). The deposit's metabolic data are in a
companion item and are not part of this release.

Run from the site folder: python3 -I tools/std/convert/Hybart2023_PeerJ.py
"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hybart2023_PeerJ"
RAW = stdlib.raw_dir(ID)
ck = stdlib.Checks()

NAMES = ["LTime", "LExoTime", "LSORaw", "LTARaw", "LSOFilt", "LSetCurrent", "LMotorCurrent", "LMotorVoltage", "LAnkleAngle",
         "LMotorAngle", "LLG", "LMG", "LAX", "LAY", "LAZ", "LGX", "LGY", "LGZ", "LFSR", "RTime", "RExoTime", "RSORaw", "RTARaw",
         "RSOFilt", "RSetCurrent", "RMotorCurrent", "RMotorVoltage", "RAnkleAngle", "RMotorAngle", "RLG", "RMG", "RAX", "RAY",
         "RAZ", "RGX", "RGY", "RGZ", "sync", "RFSR", "event"]
# (released channel name, deposit column, unit) after time_s; the right-side clock is released as r_time_s
CHAN = []
for side, S in (("left", "L"), ("right", "R")):
    p = side[0]
    CHAN += [(f"{p}_time_s", f"{S}Time", "s"), (f"{p}_exo_clock_ms", f"{S}ExoTime", "exoskeleton clock units as shared"),
             (f"{p}_soleus_raw", f"{S}SORaw", "as recorded (about 1.65 at rest; the readme gives no unit)"),
             (f"{p}_tibialis_raw", f"{S}TARaw", "as recorded (about 1.65 at rest; the readme gives no unit)"),
             (f"{p}_soleus_filtered", f"{S}SOFilt", "controller units (soleus EMG high-passed 50 Hz, rectified, low-passed 8 Hz)"),
             (f"{p}_set_current_ma", f"{S}SetCurrent", "mA"), (f"{p}_motor_current_ma", f"{S}MotorCurrent", "mA"),
             (f"{p}_motor_voltage_mv", f"{S}MotorVoltage", "mV"), (f"{p}_ankle_angle_cdeg", f"{S}AnkleAngle", "deg*100"),
             (f"{p}_motor_angle_ticks", f"{S}MotorAngle", "motor ticks"),
             (f"{p}_gastrocnemius_lateral_raw", f"{S}LG", "as recorded (about 1.65 at rest; the readme gives no unit)"),
             (f"{p}_gastrocnemius_medial_raw", f"{S}MG", "as recorded (about 1.65 at rest; the readme gives no unit)"),
             (f"{p}_acc_x", f"{S}AX", "raw sensor counts"), (f"{p}_acc_y", f"{S}AY", "raw sensor counts"),
             (f"{p}_acc_z", f"{S}AZ", "raw sensor counts"), (f"{p}_gyro_x", f"{S}GX", "raw sensor counts"),
             (f"{p}_gyro_y", f"{S}GY", "raw sensor counts"), (f"{p}_gyro_z", f"{S}GZ", "raw sensor counts"),
             (f"{p}_foot_switch", f"{S}FSR", "1 weight on the foot, 0 none")]
CHAN += [("sync", "sync", "square wave from the controller for synchronisation"), ("event", "event", "1 = step or error marked by the operator")]
TIME_COLS = {"l_time_s", "r_time_s"}


def classify(fname):
    """participant number, condition label, lap, extra tag from a file name"""
    base = fname.split("/")[-1]
    m = re.match(r"(?i)phh?d(\d+)", base)
    pid = int(m.group(1))
    low = base.lower()
    ts = re.search(r"_Data_(\d+\.\d+)\.csv$", base)
    if "noexoin" in low:
        cond, lap = "boots only, treadmill", 0
    elif "unpowin" in low:
        cond, lap = "exoskeleton unpowered, treadmill", 0
    elif re.search(r"(exopowin|powin|exoin)", low):
        cond, lap = "exoskeleton powered, treadmill", 0
    elif "noexoout" in low:
        cond, lap = "boots only, outdoor", int(re.search(r"noexoout_?(\d)", low).group(1))
    elif re.search(r"(exopowout|powout|exoout)", low):
        cond, lap = "exoskeleton powered, outdoor", int(re.search(r"out_?(\d)", low).group(1))
    else:
        raise ValueError(base)
    extra = ""
    if re.search(r"exoout1b", low):
        extra = "second file for lap 1 (restart)"
    if re.search(r"exopowout2_1", low):
        extra = "second file for lap 2 (repeat)"
    if re.search(r"noexoout2_data\.csv$", low) and pid == 17:
        extra = "second file for lap 2 (no timestamp in the file name)"
    return pid, cond, lap, extra, float(ts.group(1)) if ts else 0.0


def read_rec(z, name, usecols=None):
    with z.open(name) as f:
        d = pd.read_csv(f, index_col=False, skipinitialspace=True, on_bad_lines="skip")
    d.columns = [c.strip() for c in d.columns]
    return d


def main():
    z = zipfile.ZipFile(RAW / "EMG_EXO_Data.zip")
    files = [n for n in z.namelist() if n.lower().endswith(".csv") and "__MACOSX" not in n and not n.split("/")[-1].startswith("._") and "_GUI_" not in n]
    info = []
    for n in files:
        pid, cond, lap, extra, ts = classify(n)
        info.append(dict(name=n, pid=pid, cond=cond, lap=lap, extra=extra, ts=ts))
    info.sort(key=lambda r: (r["pid"], r["ts"] if r["ts"] else 1e18, r["name"]))
    sub = pd.read_excel(RAW / "Subject_Information.xlsx").dropna(subset=["Subject ID"])
    sub = sub[sub["Subject ID"].astype(str).str.startswith("PhD")]
    sub["pid"] = sub["Subject ID"].str.extract(r"(\d+)").astype(int)
    sub = sub.set_index("pid")
    pids = sorted({r["pid"] for r in info})
    assert pids == sorted(sub.index), (pids, list(sub.index))
    code = {p: i + 1 for i, p in enumerate(pids)}

    rows, ntr = [], {}
    bad_total = [0]
    n_total = 0
    for r in info:
        d = read_rec(z, r["name"])
        with z.open(r["name"]) as fh:
            r["bad"] = sum(1 for _ in fh) - 1 - len(d)
        d.columns = [c.strip() for c in d.columns]
        t = d.LTime.to_numpy() - d.LTime.iloc[0]
        ev = np.flatnonzero(d.event.fillna(0).to_numpy() == 1)
        # events are runs of 1; list the first sample of each run
        starts = [i for i in ev if i == 0 or d.event.iloc[i - 1] != 1]
        evs = ";".join(f"marked_{k + 1}={t[i]:.4f}" for k, i in enumerate(starts))
        n_total += len(d)
        bad_total[0] += r["bad"]
        s = r["pid"]
        ntr[s] = ntr.get(s, 0) + 1
        sex = sub.loc[s, "Sex"]
        lab = {"boots only, treadmill": "boots_treadmill", "exoskeleton unpowered, treadmill": "unpowered_treadmill",
               "exoskeleton powered, treadmill": "powered_treadmill", "boots only, outdoor": "boots_outdoor",
               "exoskeleton powered, outdoor": "powered_outdoor"}[r["cond"]]
        fn = f"samples/{ID}_S{code[s]:04d}_{lab}" + (f"_lap{r['lap']}" if r["lap"] else "") + \
             (f"_extra{ntr[s]}" if r["extra"] else "") + ".csv"
        r["fn"] = fn
        r["n"] = len(d)
        outdoor = "outdoor" in r["cond"]
        powered = "exoskeleton powered" in r["cond"]
        tinfo = f"walking_surface={'outdoor course' if outdoor else 'treadmill'}"
        if r["lap"]:
            tinfo += f";lap={r['lap']}"
        if r["extra"]:
            tinfo += f";note={r['extra']}"
        tinfo += f";self_selected_walking_speed_mps={sub.loc[s, 'walking speed (m/s)']:.2f}"
        rows.append(dict(
            dataset=ID, study=ID, experiment="", group="", subject=str(s), person=str(s), subject_excluded_by_authors=False,
            age=float(sub.loc[s, "age"]), sex=sex, handedness="right", population="healthy", setting="lab",
            device="Dephy ExoBoots (EB60) with soleus proportional myoelectric control, Coapt bipolar EMG, foot switches, onboard IMUs",
            session=1, block=r["cond"], trial=ntr[s], phase="test", time_s="", task="walking", unit="recording",
            condition=r["cond"] + (f", lap {r['lap']}" if r["lap"] else ""),
            pert_type="exoskeleton_assist" if powered else "none", pert_value=1 if powered else 0, pert_unit="on" if powered else "",
            feedback="normal", excluded_by_authors=False,
            recording=ntr[s], samples_file=fn,
            signal_kind="bilateral lower-leg EMG (soleus, tibialis anterior, gastrocnemii), exoskeleton motor and ankle-angle signals, IMU, foot switches",
            channels=";".join(c[0] for c in CHAN), channel_units=";".join(c[2] for c in CHAN),
            sample_rate_hz=round((len(d) - 1) / t[-1], 1), n_samples=len(d), duration_s=round(float(t[-1]), 3), events=evs,
            processing=("raw as logged by the controller, except the soleus_filtered channel (high-pass 50 Hz, rectified, low-pass 8 Hz, "
                        "computed in real time by the authors' controller); times are seconds from the first left-loop sample"),
            trial_info=tinfo))
    D = pd.DataFrame(rows)

    def samples():
        for r in info:
            d = read_rec(z, r["name"])
            assert len(d) == r["n"], (r["name"], len(d), r["n"])
            assert list(d.columns) == NAMES, list(d.columns)
            t0 = d.LTime.iloc[0]
            out = pd.DataFrame(index=d.index)
            out["time_s"] = (d.LTime - t0).map("{:.6f}".format)
            for nm, col, _ in CHAN:
                if nm in TIME_COLS:
                    out[nm] = (d[col] - t0).map("{:.6f}".format)
                else:
                    out[nm] = d[col]
            yield r["fn"], out.to_csv(index=False, float_format="%.6g", lineterminator="\r\n").encode()

    part = pd.DataFrame([dict(person=str(p), height_cm=sub.loc[p, "Height (cm)"], mass_kg=sub.loc[p, "weight (kg)"],
                              self_selected_speed_mps=sub.loc[p, "walking speed (m/s)"]) for p in pids])
    pcols = [("height_cm", "number", "cm", "Body height."), ("mass_kg", "number", "kg", "Body mass."),
             ("self_selected_speed_mps", "number", "m/s", "Self-selected walking speed set on the first training day and used on the treadmill and outdoors.")]

    ages = np.array([sub.loc[p, "age"] for p in pids], float)
    ht = np.array([sub.loc[p, "Height (cm)"] for p in pids], float) / 100
    ms = np.array([sub.loc[p, "weight (kg)"] for p in pids], float)
    P = ck.published
    P("participants", D.person.nunique(), 12, 0, "Methods, Participants")
    P("female participants", sum(sub.loc[p, "Sex"] == "F" for p in pids), 5, 0, "Methods, Participants")
    P("male participants", sum(sub.loc[p, "Sex"] == "M" for p in pids), 7, 0, "Methods, Participants")
    P("age, mean (22.6 years)", ages.mean(), 22.6, 0.05, "Methods, Participants")
    P("age, SD (8.66; population SD, as in the deposit's own std row)", ages.std(ddof=0), 8.66, 0.005, "Methods, Participants")
    P("age, minimum (18)", ages.min(), 18, 0, "Methods, Participants")
    P("age, maximum (50)", ages.max(), 50, 0, "Methods, Participants")
    P("age without the oldest, mean (20.1)", np.sort(ages)[:-1].mean(), 20.1, 0.05, "Methods, Participants")
    sd_wo = np.sort(ages)[:-1].std(ddof=0)  # paper: 2.1; does not reproduce, listed under differences
    P("height, mean (1.73 m)", ht.mean(), 1.73, 0.005, "Methods, Participants")
    P("height, SD (0.075 m)", ht.std(ddof=0), 0.075, 0.0005, "Methods, Participants")
    P("mass, mean (69.3 kg)", ms.mean(), 69.3, 0.05, "Methods, Participants")
    P("mass, SD (10.2 kg)", ms.std(ddof=0), 10.2, 0.05, "Methods, Participants")
    ck.internal("every CSV of the deposit is one recording (no file dropped)", len(D) == len(files) == 107, f"{len(D)} recordings")
    ck.internal("12 participants all have treadmill and outdoor recordings",
                D[D.block.str.contains("treadmill")].person.nunique() == 12 and D[D.block.str.contains("outdoor")].person.nunique() == 12)
    ck.internal("no outdoor condition has more than 3 laps per participant except the files marked as a second file",
                D[(D.block.str.contains("outdoor"))].groupby(["person", "block"]).size().max() <= 4)
    ck.internal("treadmill recordings last about 6 minutes (the readme) : median between 6 and 8 minutes (readme says 6, paper says 8)",
                360 < D[D.block.str.contains("treadmill")].duration_s.median() < 480, str(D[D.block.str.contains("treadmill")].duration_s.median()))

    meta = dict(
        study=ID,
        paper=dict(authors="Hybart R, Villancio-Wolter KS, Ferris DP", year=2023,
                   title="Metabolic cost of walking with electromechanical ankle exoskeletons under proportional myoelectric control on a treadmill and outdoors",
                   venue="PeerJ 11:e15775", doi="10.7717/peerj.15775"),
        lab="Ferris (University of Florida)", data_url="https://doi.org/10.6084/m9.figshare.22572868", license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Each of the deposit's CSV files (one per participant, condition and lap) became one samples CSV. The two controller clocks "
                 "(Unix times in the deposit) are released as seconds from the first sample, because absolute clock times encode the session "
                 "date; all other columns are the deposit's, renamed, with numbers written to 6 significant digits (the deposit's EMG values "
                 "carry 16 digits, well below the converter's resolution). The deposit's header has one fewer name than its rows have fields "
                 "(rows end with a comma); the names were applied to the first 40 fields. Height, mass, age, sex and speed come from "
                 "Subject_Information.xlsx. The metabolic data of the companion item are not included."),
        how=["One row is one recording of one participant walking with ExoBoots ankle exoskeletons: 6 minutes on a treadmill at the participant's "
             "self-selected speed with only the boots (boots only), with the exoskeletons worn but unpowered, or with proportional myoelectric "
             "assistance from the soleus EMG, or one 0.45 km lap of an outdoor course (three laps per condition, boots only or powered). "
             "File names in the deposit are inconsistent; 'exoin', 'powin' and 'exopowin' are read as powered treadmill, 'unpowin' and "
             "'exounpowin' as unpowered, 'exoout', 'powout' and 'exopowout' as powered outdoor laps. A few laps have a second file "
             "(a restart or a repeat), marked in trial_info.",
             "Samples (samples/<subject>_<condition>[_lap].csv) are the controller's loop samples, about 370 per second but not evenly spaced "
             "(the median step is 2.3 ms and gaps up to 0.2 s occur). time_s and l_time_s are the left loop clock, r_time_s the right loop "
             "clock, both from the first left sample. Then, for each leg, the exoskeleton clock, raw soleus and tibialis EMG, the filtered "
             "soleus signal the controller used, the commanded and measured motor current, motor voltage, ankle angle (degrees x 100), motor "
             "angle, raw lateral and medial gastrocnemius EMG, accelerometer and gyroscope counts and the foot switch, then the shared "
             "synchronisation square wave and the operator's event flag. events lists the first sample of each run of event = 1.",
             "pert_type exoskeleton_assist (value 1, unit on) marks the powered conditions."],
        unit_note="recording (one participant, condition and lap)",
        differences=[f"{bad_total[0]} malformed lines (wrong number of fields) in the deposit's CSV files were left out ({sum(1 for r in info if r['bad'])} files affected).",
                     f"Standard deviations of age, height and mass in the paper are population SDs (divided by n), as in the deposit's own std row. The paper's SD of age without the oldest participant (2.1) comes out {sd_wo:.2f} here and is not used as a check.",
                     "One 438-byte controller-settings log (PhD27 exoout2_GUI, gains and thresholds, 13 values per row under 12 header names) is not released because its columns cannot be matched to names; all 107 recordings are.",
                     "The paper describes 8-minute treadmill trials; the readme and the data (a median of about 6.6 minutes) are closer to the readme's 6.",
                     "The readme says the right-leg foot switch is a 'left foot switch'; the data have a 0/1 switch for each leg and are released as such.",
                     "Treadmill force-plate, respiratory (metabolic) and IMU data named in the paper are not part of this deposit."],
    )
    stdlib.release(ID, "timeseries", D, meta, ck, participants=part, participant_columns=pcols, extra_files=samples())
    for c in ck.items:
        print(("  P " if c["kind"] == "published" else "  I ") + c["what"][:80], c.get("computed", ""), c.get("reported", ""), c.get("detail", "")[:120])


if __name__ == "__main__":
    main()
