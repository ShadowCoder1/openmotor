"""Fukuchi, Fukuchi & Duarte (2018) PeerJ 6:e4640: public dataset of overground and treadmill walking kinematics and kinetics of
healthy young and older adults (figshare 10.6084/m9.figshare.5722711, version 2 with 51 participants).
One row per recording and signal type: the raw marker trajectories (150 Hz, mm) and the raw ground reaction forces (300 Hz) of every
walking trial, as the authors' tab-separated ASCII files. Static calibration trials and the processed ensemble averages (*ang, *knt)
are not converted.

Raw files (untrusted data, read with zipfile and csv text only): raw_extra/Fukuchi2018_PeerJ/WBDSascii.zip, WBDSinfo.csv
Run from the site folder: python3 -I tools/std/convert/Fukuchi2018_PeerJ.py
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

ID = "Fukuchi2018_PeerJ"
MODULE = "timeseries"
SRC = "Fukuchi et al. 2018, PeerJ"
COND = {"S": "slow (30% below comfortable)", "C": "comfortable", "F": "fast (30% above comfortable)"}
FROUDE = {1: 40, 2: 55, 3: 70, 4: 85, 5: 100, 6: 115, 7: 130, 8: 145}
GRF_UNITS = {"F": "N", "C": "mm", "T": "N*m (as stated in the paper; magnitudes look like N*mm)"}


def grf_unit(ch):
    return GRF_UNITS["F"] if ch.startswith("F") else GRF_UNITS["C"] if ch.startswith("COP") else GRF_UNITS["T"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    raw = stdlib.raw_dir(ID)
    info = pd.read_csv(raw / "WBDSinfo.csv")
    info = info[info.FileName.str.endswith(".txt")].set_index("FileName")
    z = zipfile.ZipFile(raw / "WBDSascii.zip")
    names = sorted(n for n in z.namelist() if re.search(r"WBDS\d+walk[OT]\d+[SCF]?(mkr|grf)\.txt$", n))
    tmp = Path("tmp_" + ID)
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp).mkdir()
    ck = stdlib.Checks()
    rows, extra, people, speeds = [], [], {}, []
    rate_m, rate_g, dur_t = [], [], []
    seq = {}
    for n in names:
        base = n.split("/")[-1]
        m = re.fullmatch(r"WBDS(\d+)walk([OT])(\d+)([SCF]?)(mkr|grf)\.txt", base)
        sid, env, k, sp, kind = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), m.group(5)
        r = info.loc[base]
        text = z.read(n).decode("ascii").splitlines()
        head = text[0].split("\t")
        body = [l.split("\t") for l in text[1:] if l.strip()]
        assert all(len(b) == len(head) for b in body), base
        if kind == "mkr":
            assert head[0] == "Time"
            t = [num(b[0]) for b in body]
            rate = (len(t) - 1) / t[-1]  # time stamps are rounded to 1 ms, so use the whole span
            rate_m.append(rate)
            chans = head[1:]
            units = ["mm"] * len(chans)
            kindtxt = "raw 3D marker trajectories (X, Y, Z of each marker)"
            samples = [[b[0]] + b[1:] for b in body]
            hz = 150.0
        else:
            frames = [int(b[0]) for b in body]
            assert frames == list(range(1, len(frames) + 1)), base
            hz = 300.0
            samples = [[f"{(f - 1) / hz:.6f}"] + b[1:] for f, b in zip(frames, body)]
            chans = head[1:]
            units = [grf_unit(c) for c in chans]
            kindtxt = "raw ground reaction forces, centre of pressure and free moment of each force plate (overground: 5 plates; treadmill: 2 belts)"
            t = [float(s[0]) for s in samples]
            rate_g.append(1 / np.median(np.diff(t)))
        dur = float(t[-1]) + (1 / hz if kind == "grf" else 0.0)
        if env == "T" and kind == "mkr":
            dur_t.append(t[-1])
        out = tmp / (base.replace(".txt", ".csv").replace("WBDS", "s"))
        with open(out, "w", newline="") as fh:
            fh.write("time_s," + ",".join(chans) + "\r\n")
            for s in samples:
                fh.write(",".join(s) + "\r\n")
        arc = f"samples/{out.name}"
        extra.append((arc, str(out)))
        spd = num(r["GaitSpeed(m/s)"])
        if env == "O":
            cond = f"overground, {COND[sp]} speed"
            speeds.append((sid, "O" + sp, spd))
        else:
            cond = f"treadmill, speed {k} of 8 ({FROUDE[k]}% of the Froude speed)"
            speeds.append((sid, "T%d" % k, spd))
        notes = [] if str(r["Notes"]).strip("- ") == "" else [re.sub(r"\s+", " ", str(r["Notes"]).strip())]
        if env == "T" and str(r["TreadHands"]) == "Yes":
            notes.append("participant held the treadmill handrails")
        seq[sid] = seq.get(sid, 0) + 1
        people[sid] = r
        rows.append(dict(
            dataset=ID, study=SRC, experiment="", group=r["AgeGroup"] + " adults", subject=f"s{sid:02d}", person=f"s{sid:02d}",
            subject_excluded_by_authors=False, age=float(r["Age"]), sex=r["Gender"], handedness={"R": "right", "L": "left"}[r["Dominance"]],
            population="healthy", setting="lab", device="12-camera motion capture (150 Hz), force plates (300 Hz), "
            + ("instrumented split-belt treadmill" if env == "T" else "walkway with force plates"),
            session=1, block="treadmill" if env == "T" else "overground", trial=seq[sid], phase="test", time_s=np.nan, task="walking",
            unit="recording", condition=cond + (", gait speed %.2f m/s" % spd if spd == spd else ""), pert_type="none", pert_value=0,
            pert_unit="", feedback="normal", excluded_by_authors=False, recording=seq[sid], samples_file=arc, signal_kind=kindtxt,
            channels=";".join(chans), channel_units=";".join(units), sample_rate_hz=hz, n_samples=len(samples), duration_s=round(dur, 4),
            events="", processing="none (raw ASCII export)" + ("; authors' note: " + "; ".join(notes) if notes else "")))
    df = pd.DataFrame(rows)
    sp = pd.DataFrame(speeds, columns=["sid", "cond", "v"]).dropna()
    mean_sp = sp.groupby(["sid", "cond"]).v.mean().unstack()

    pp = pd.DataFrame([dict(person=f"s{s:02d}", age_group=r["AgeGroup"], height_cm=r["Height"], mass_kg=r["Mass"],
                            leg_length_m=num(r["LegLength"])) for s, r in sorted(people.items())])
    pp["_age"] = [float(people[s]["Age"]) for s in sorted(people)]
    young = pp[pp.age_group == "Young"]
    ck.published("young adults", len(young), 24, 0, "Methods, Participants")
    ck.published("young adults, mean age (years)", young._age.mean(), 27.6, 0.1, "Table 1")
    ck.published("young adults, SD of age (years)", young._age.std(), 4.4, 0.1, "Table 1")
    ck.published("young adults, mean height (cm)", young.height_cm.mean(), 171.1, 0.1, "Table 1")
    ck.published("young adults, SD of height (cm)", young.height_cm.std(), 10.5, 0.1, "Table 1")
    ck.published("young adults, mean mass (kg)", young.mass_kg.mean(), 68.4, 0.1, "Table 1")
    ck.published("young adults, SD of mass (kg)", young.mass_kg.std(), 12.2, 0.1, "Table 1")
    ck.published("marker sampling rate (Hz)", np.median(rate_m), 150, 1, "Methods, Data collection")
    ck.published("ground reaction force sampling rate (Hz)", np.median(rate_g), 300, 1, "Methods, Data collection")
    ck.published("treadmill recording length (s; last 30 s of each 90 s trial recorded)", np.median(dur_t), 30, 0.1, "Methods, Treadmill trials")
    ck.published("treadmill speed 1 relative to speed 5 (40% / 100% of Froude speed)", (mean_sp["T1"] / mean_sp["T5"]).mean(), 0.40, 0.03, "Methods (40-145% of Froude speed)")
    ck.published("treadmill speed 8 relative to speed 5 (145% / 100%)", (mean_sp["T8"] / mean_sp["T5"]).mean(), 1.45, 0.05, "Methods")
    ck.published("overground slow speed relative to comfortable (30% slower)", (mean_sp["OS"] / mean_sp["OC"]).mean(), 0.70, 0.05, "Methods, Overground trials")
    ck.published("overground fast speed relative to comfortable (30% faster)", (mean_sp["OF"] / mean_sp["OC"]).mean(), 1.30, 0.05, "Methods, Overground trials")
    nm = (df.signal_kind.str.startswith("raw 3D")).sum()
    ng = (df.signal_kind.str.startswith("raw ground")).sum()
    ck.internal("every walking trial has a marker file and a force file", nm == ng, f"{nm} marker and {ng} force recordings")
    ck.internal("51 participants kept", df.person.nunique() == 51, str(df.person.nunique()))
    ck.internal("all sample files written", len(extra) == len(df))
    bad = df[(df.duration_s < 0.5) | (df.duration_s > 300)]
    ck.internal("recording lengths between 0.5 and 300 s", len(bad) == 0, f"{df.duration_s.min():.1f}-{df.duration_s.max():.1f} s")

    meta = dict(
        study=ID,
        paper=dict(authors="Fukuchi CA, Fukuchi RK, Duarte M", year=2018,
                   title="A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals",
                   venue="PeerJ 6:e4640", doi="10.7717/peerj.4640"),
        lab="Duarte (Federal University of ABC, Brazil)", data_url="https://doi.org/10.6084/m9.figshare.5722711",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Converted every raw walking trial (WBDS<nn>walk[O|T]<nn>[S|C|F]mkr.txt and grf.txt) from tab-separated text to comma-separated "
                 "CSV with the authors' values unchanged; the force files' frame number was turned into time_s = (frame - 1) / 300; "
                 "static calibration trials and the processed files (ang, knt: ensemble averages over 101 points) are left out; "
                 "age, sex, handedness and the authors' file notes were added from WBDSinfo.csv; dates of birth are not in the files."),
        how=["One row is one recording of one walking trial and one signal type: the marker trajectories (mm, 150 Hz, columns X, Y, Z "
             "of each marker) or the force plates (Fx, Fy, Fz in N, centre of pressure COPx, COPy, COPz in mm, free moment Ty; 300 Hz). "
             "A trial therefore has two rows, one per signal type.",
             "Participants walked barefoot overground at a comfortable, 30% slower and 30% faster speed, and on an instrumented treadmill at "
             "eight speeds (40 to 145% of their Froude speed, in random order); only the last 30 s of each 90 s treadmill trial "
             "were recorded. The gait speed in condition is the authors' value from WBDSinfo.csv (missing for some trials).",
             "Samples are raw: gaps in marker trajectories and force plate offsets are kept as the authors left them; their notes on "
             "individual files (a missing marker, a plate with an offset, handrail use) are in processing. The authors' paper "
             "gives the free moment Ty in N*m, but the magnitudes suggest N*mm."],
        unit_note="recording (one walking trial and one signal type: markers or force plates)",
        differences=["The paper describes 42 volunteers (24 young, 18 older); the current figshare version (2) holds 51 (24 young, 27 older). "
                     "The young group reproduces the paper's age, height and mass; the older-group numbers of the paper were not checked "
                     "because the group has grown.",
                     "The paper counts 6,916 metadata rows; the deposit's WBDSinfo.csv now has 7,203 (more participants)."])
    names_ = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df[names_]
    pc = [("age_group", "text", "", "Young or Older adults (the authors' grouping)."), ("height_cm", "number", "cm", "Height."),
          ("mass_kg", "number", "kg", "Body mass."), ("leg_length_m", "number", "m", "Leg length as given in WBDSinfo.csv (empty if missing).")]
    stdlib.release(ID, MODULE, df, meta, ck, participants=pp.drop(columns="_age"), participant_columns=pc, extra_files=extra)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
