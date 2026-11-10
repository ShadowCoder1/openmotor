"""Moreira, Figueiredo, Fonseca, Vilas-Boas & Santos (2021) Sci Data 8:103: lower-limb kinematics, kinetics and EMG of 16 healthy
adults walking at seven metronome-controlled speeds (1.0 to 4.0 km/h) over force platforms. One row per stride (left leg: one stride
per trial; right leg: two), with the authors' gait-cycle-normalized curves (1001 samples) as a CSV inside the ZIP.

Raw file: raw_extra/Moreira2021_SciData/ascii.zip (Figshare 10.6084/m9.figshare.13169240, "ASCII files.zip", CC0), read with zipfile and
text parsing only. Only the Processed_Data folders are converted (angles, ground reaction forces, torques normalized by body mass,
EMG envelope, stride times). The raw 10 m recordings (markers, CoP, FPM, raw EMG), the unnormalized torques and the filtered EMG are
left out to keep the ZIP small; the MAT file (item 13169237) holds the same data.
Run from the site folder: python3 -I tools/std/convert/Moreira2021_SciData.py"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Moreira2021_SciData"
MODULE = "timeseries"
SPEED = {"V1": 1.0, "V15": 1.5, "V2": 2.0, "V25": 2.5, "V3": 3.0, "V35": 3.5, "V4": 4.0}
# (folder, channel prefix, unit)
GROUPS = [("Angles", "angle", "deg"), ("GRF", "grf", "N"), ("Torques_Norm", "torque", "N*m/kg"), ("EMG envelope", "emg_envelope", "% MVC")]
EMGMAP = {"VL": "vastus_lateralis", "BF": "biceps_femoris", "TA": "tibialis_anterior", "GAL": "gastrocnemius_lateralis"}
ZROOT = "ASCII files/"
DEVICE = ("12-camera Qualisys motion capture (200 Hz), six embedded force platforms (200 Hz), Delsys Trigno EMG (2000 Hz); "
          "10 m walkway, Visual3D joint angles and torques")


def table(z, name):
    lines = [l for l in z.read(name).decode("ascii").splitlines() if l.strip()]
    head = lines[0].split("\t")
    body = [l.split("\t") for l in lines[1:]]
    assert all(len(b) == len(head) for b in body), name
    return head, body


def meta_of(z, p):
    d = {}
    for l in z.read(f"{ZROOT}Participant{p}/Metadata.txt").decode("ascii", "replace").splitlines():
        if ":" in l:
            k, v = l.split(":", 1)
            d[k.strip()] = v.strip()
    num = lambda s: float(re.match(r"[-\d.]+", s).group())  # noqa: E731
    return dict(age=num(d["Age"]), sex={"Male": "M", "Female": "F"}[d["Gender"]], height_m=num(d["Body Height"]),
                mass_kg=num(d["Body Mass"]), leg_length_m=num(d["Leg Length"]), foot_length_m=num(d["Foot Length"]))


def channel_name(prefix, col):
    parts = col.split("_")[1:]  # drop St<k>
    if prefix == "emg_envelope":
        return f"emg_envelope_{EMGMAP[parts[0]]}"
    if prefix == "grf":
        return f"grf_{parts[-1].lower()}"
    return f"{prefix}_{parts[0].lower()}_{parts[1].lower()}"


def main():
    raw = stdlib.raw_dir(ID)
    z = zipfile.ZipFile(raw / "ascii.zip")
    names = set(z.namelist())
    pids = sorted({int(m.group(1)) for n in names if (m := re.match(r"ASCII files/Participant(\d+)/Metadata.txt", n))})
    tmp = Path("tmp_" + ID)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rows, extra, people = [], [], {}
    unit_of = {}
    n_nostride = 0
    for p in pids:
        people[p] = meta_of(z, p)
        seq = 0
        for v in SPEED:
            base = f"{ZROOT}Participant{p}/Processed_Data/{v}/"
            for leg in ("R", "L"):
                sname = f"{base}{leg}/StrideTime/StrideTime.txt"
                if sname not in names:
                    continue
                sh, sb = table(z, sname)
                stime = {int(r[0]): {i: r[i] for i in range(1, len(r)) if r[i].strip()} for r in sb}
                trials = sorted({int(m.group(1)) for n in names if n.startswith(f"{base}{leg}/Angles/")
                                 and (m := re.search(r"/T(\d+)\.txt$", n))})
                assert set(trials) <= set(stime), (p, v, leg, trials, sorted(stime))
                for t in trials:
                    cols, data = {}, {}
                    for folder, prefix, unit in GROUPS:
                        fn = f"{base}{leg}/{folder}/T{t}.txt"
                        if fn not in names:
                            continue
                        head, body = table(z, fn)
                        assert len(body) == 1001, (fn, len(body))
                        for ci, h in enumerate(head):
                            m = re.match(r"St(\d+)_(.*)", h)
                            if not m:
                                continue
                            s = int(m.group(1))
                            cols.setdefault(s, []).append((channel_name(prefix, h), unit, [b[ci] for b in body]))
                    for s in sorted(cols):
                        st = stime[t].get(s)
                        assert st is not None, (p, v, leg, t, s)
                        st_f = float(st)
                        names_c = [c[0] for c in cols[s]]
                        assert len(set(names_c)) == len(names_c)
                        fname = f"p{p:02d}_{v}_{leg}_T{t}_S{s}.csv"
                        with open(tmp / fname, "w", newline="") as fh:
                            fh.write("time_s," + ",".join(names_c) + "\r\n")
                            for k in range(1001):
                                fh.write(f"{k * st_f / 1000:.5f}," + ",".join(c[2][k] for c in cols[s]) + "\r\n")
                        extra.append((f"samples/{fname}", str(tmp / fname)))
                        seq += 1
                        sk = {"deg": "joint angles", "N": "ground reaction forces", "N*m/kg": "joint torques", "% MVC": "EMG envelope"}
                        kinds = [sk[u] for u in dict.fromkeys(c[1] for c in cols[s])]
                        rows.append(dict(
                            dataset=ID, study=ID, experiment="", group="", subject=f"P{p:02d}", person=f"P{p:02d}",
                            subject_excluded_by_authors=False, age=people[p]["age"], sex=people[p]["sex"], population="healthy",
                            setting="lab", device=DEVICE, session=1, block=v, trial=seq, phase="test", time_s=np.nan, task="walking",
                            unit="stride", condition=f"{SPEED[v]:.1f} km/h (metronome-paced), {'right' if leg == 'R' else 'left'} leg, "
                            f"trial {t}, stride {s}", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                            excluded_by_authors=False, recording=seq, samples_file=f"samples/{fname}",
                            signal_kind=", ".join(kinds) + " over one gait cycle, 0-100% from heel strike to the next heel strike",
                            channels=";".join(names_c), channel_units=";".join(c[1] for c in cols[s]), sample_rate_hz=np.nan,
                            n_samples=1001, duration_s=st_f, events=f"heel_strike=0;next_heel_strike={st_f:g}",
                            processing=("gait-cycle normalized to 1001 samples (time_s here = sample/1000 x the authors' stride time); "
                                        "angles low-pass filtered at 6 Hz, torques divided by body mass, EMG band-pass filtered at 20-450 Hz "
                                        "and enveloped by RMS over 300 ms (% of maximal voluntary contraction), as the authors describe")))
    df = pd.DataFrame(rows)
    df["handedness"] = ""
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"])
    pp = pd.DataFrame([dict(person=f"P{p:02d}", **m) for p, m in people.items()])
    ck = stdlib.Checks()
    ck.published("participants", len(pp), 16, 0, "Methods (Participants)")
    ck.published("males", (pp.sex == "M").sum(), 8, 0, "Methods (Participants)")
    ck.published("females", (pp.sex == "F").sum(), 8, 0, "Methods (Participants)")
    ck.published("mean age (years)", pp.age.mean(), 23.8, 0.1, "Methods (Participants)")
    ck.published("SD of age (years, population SD)", pp.age.std(ddof=0), 2.02, 0.02, "Methods (Participants)")
    ck.published("mean body mass (kg)", pp.mass_kg.mean(), 67.5, 0.1, "Methods (Participants)")
    ck.published("SD of body mass (kg, population SD)", pp.mass_kg.std(ddof=0), 10.8, 0.1, "Methods (Participants)")
    ck.published("mean body height (m)", pp.height_m.mean(), 1.69, 0.01, "Methods (Participants)")
    ck.published("SD of body height (m, population SD)", pp.height_m.std(ddof=0), 0.108, 0.002, "Methods (Participants)")
    ck.published("minimum body height (m)", pp.height_m.min(), 1.51, 0, "Introduction (1.51 to 1.83 m)")
    ck.published("maximum body mass (kg)", pp.mass_kg.max(), 83.7, 0, "Introduction (52.0 to 83.7 kg)")
    ck.published("walking speeds", df.block.nunique(), 7, 0, "Methods (Experimental protocol)")
    nr = df.condition.str.contains("right leg").sum()
    nl = df.condition.str.contains("left leg").sum()
    ck.internal("right-leg strides are twice the left-leg strides (two right strides per trial, one left)", nr == 2 * nl, f"{nr} right, {nl} left")
    ck.internal("every stride has a sample file of 1001 samples", len(extra) == len(df) and (df.n_samples == 1001).all())
    ck.internal("stride times between 0.5 and 15 s", df.duration_s.between(0.5, 15).all(), f"{df.duration_s.min()}-{df.duration_s.max()}")
    ck.internal("slower speeds give longer strides (median stride time falls from 1.0 to 4.0 km/h)",
                df[df.block == "V1"].duration_s.median() > df[df.block == "V4"].duration_s.median())
    meta = dict(
        paper=dict(authors="Moreira L, Figueiredo J, Fonseca P, Vilas-Boas JP, Santos CP", year=2021,
                   title="Lower limb kinematic, kinetic, and EMG data from young healthy humans during walking at controlled speeds",
                   venue="Sci Data 8:103", doi="10.1038/s41597-021-00881-3"),
        lab="Santos (University of Minho)", data_url="https://doi.org/10.6084/m9.figshare.13169240",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Converted the Processed_Data text files (tab-separated, one gait cycle of 1001 samples per stride and channel group) into one "
                 "CSV per stride with the authors' values unchanged; time_s was added as sample/1000 x the stride time of "
                 "StrideTime.txt. Joint angles, ground reaction forces, torques normalized by body mass and the EMG envelope are kept; "
                 "the raw 10 m recordings, the unnormalized torques and the filtered EMG are not included. Age and sex come from each "
                 "participant's Metadata.txt; height, mass, leg and foot length are in participants.csv."),
        how=["One row is one stride of one leg, taken from the force-platform steps of a walking trial. Its samples file holds 1001 "
             "samples from heel strike (0%) to the next heel strike of the same leg (100%): hip, knee and ankle angles about X, Y, Z "
             "(deg; the right-leg rows also hold the pelvis angles), the three components of the ground reaction force (N), joint "
             "torques (N*m/kg of body mass) and, for the trials with usable EMG, the envelope of vastus lateralis, biceps femoris, "
             "tibialis anterior and gastrocnemius lateralis (% of maximal voluntary contraction) of that leg.",
             "Participants walked a 10 m walkway at seven speeds (1.0 to 4.0 km/h, set by a metronome), ten valid trials per speed; the "
             "first step on the force platforms was always taken with the right foot, so a trial gives two right strides and one left stride. "
             "block is the speed folder (V1 = 1.0 km/h, V15 = 1.5, ... V4 = 4.0). Trial numbers in condition are the authors' (EMG files miss "
             "a few trials). The order of rows is by speed, then leg (right first), trial and stride; the deposit does not give the order "
             "in which the trials were recorded.",
             "X, Y, Z are the sagittal, coronal and transverse components in the authors' Visual3D model; sample_rate_hz is empty because "
             "the curves are time-normalized."],
        unit_note="stride (one leg, one gait cycle)",
        differences=["The paper counts 1120 trials (1120 left-leg and 2240 right-leg strides). The ASCII deposit holds %d trials (%d left-leg "
                     "and %d right-leg strides); the missing trials are not in the files, and trial numbers in the deposit are renumbered 1..N "
                     "in the StrideTime tables." % (nl, nl, nr)])
    part_cols = [("height_m", "number", "m", "Body height."), ("mass_kg", "number", "kg", "Body mass."),
                 ("leg_length_m", "number", "m", "Leg length."), ("foot_length_m", "number", "m", "Foot length.")]
    participants = pp[["person", "height_m", "mass_kg", "leg_length_m", "foot_length_m"]]
    try:
        stdlib.release(ID, MODULE, df, meta, ck, participants=participants, participant_columns=part_cols, extra_files=extra)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))


main()
