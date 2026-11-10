"""Fukuchi, Fukuchi & Duarte (2017) PeerJ 5:e3298: a public dataset of running biomechanics (figshare 10.6084/m9.figshare.4543435,
39 runners, treadmill running at 2.5, 3.5 and 4.5 m/s).
One row per recording and signal type: the raw marker trajectories (150 Hz, mm) and the raw ground reaction forces (300 Hz) of every
30 s treadmill running trial, plus one row per runner for the authors' processed ensemble averages (101 points over the gait cycle).
Static calibration trials, the c3d files and the Visual 3D files are not converted.

Raw files (untrusted data, read as plain text only): raw_extra/Fukuchi2017_PeerJ/RBDSinfo.txt, RBDS<nnn>run T<25|35|45>markers.txt,
RBDS<nnn>runT<25|35|45>forces.txt, RBDS<nnn>processed.txt
Run from the site folder: python3 -I tools/std/convert/Fukuchi2017_PeerJ.py
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Fukuchi2017_PeerJ"
MODULE = "timeseries"
SRC = "Fukuchi et al. 2017, PeerJ"
SPEEDS = {"25": 2.5, "35": 3.5, "45": 4.5}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_txt(path):
    text = Path(path).read_text(encoding="latin-1").splitlines()
    head = text[0].rstrip("\r\n").split("\t")
    body = []
    for l in text[1:]:
        if not l.strip():
            continue
        b = l.rstrip("\r\n").split("\t")
        assert len(b) == len(head), (path, len(b), len(head))
        body.append(b)
    return head, body


def proc_unit(c):
    if "Ang" in c:
        return "deg"
    if "Mom" in c:
        return "N*m/kg"
    if "grf" in c:
        return "N/kg"
    return "W/kg"


def main():
    raw = stdlib.raw_dir(ID)
    info = pd.read_csv(raw / "RBDSinfo.txt", sep="\t", encoding="latin-1")
    info = info.drop_duplicates("Subject").set_index("Subject")
    tmp = Path("tmp_" + ID)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    ck = stdlib.Checks()
    files = {}
    for p in raw.glob("RBDS*.txt"):
        m = re.fullmatch(r"RBDS(\d+)(?:run(T\d+)(markers|forces)|(processed))\.txt", p.name)
        if m:
            sid = int(m.group(1))
            key = ("processed", None) if m.group(4) else (m.group(3), m.group(2)[1:])
            files[(sid, key)] = p
    sids = sorted({s for s, _ in files})
    rows, extra, people = [], [], {}
    n_mk_ch, n_fr_cols, n_fr_rows, last_t, n_pr_rows, n_pr_cols, rate_m = [], [], [], [], [], [], []
    for n, sid in enumerate(sids, 1):
        r = info.loc[sid]
        code = f"S{n:04d}"
        people[sid] = r
        seq = 0
        order = [("markers", s) for s in SPEEDS] + [("forces", s) for s in SPEEDS]
        order = sorted(order, key=lambda k: (k[1], k[0]))
        order = [k for k in order if (sid, k) in files] + ([("processed", None)] if (sid, ("processed", None)) in files else [])
        for key in order:
            kind, sp = key
            head, body = read_txt(files[(sid, key)])
            seq += 1
            if kind == "markers":
                assert head[0] == "Time"
                t = [num(b[0]) for b in body]
                chans, units, hz = head[1:], ["mm"] * (len(head) - 1), 150.0
                samples = [b for b in body]
                kt = "raw 3D marker trajectories (X, Y, Z of each marker)"
                n_mk_ch.append(len(chans) / 3)
                last_t.append(t[-1])
                rate_m.append((len(t) - 1) / t[-1])
                fname = f"samples/{ID}_{code}_T{sp}_markers.csv"
                tcol = "time_s"
            elif kind == "forces":
                frames = [int(b[0]) for b in body]
                assert frames == list(range(1, len(frames) + 1)), (sid, sp)
                hz = 300.0
                samples = [[f"{(f - 1) / hz:.6f}"] + b[1:] for f, b in zip(frames, body)]
                chans = head[1:]
                units = ["N", "N", "N", "mm", "mm", "mm", "N*m (as stated in the paper; magnitudes look like N*mm)"]
                assert chans == ["Fx", "Fy", "Fz", "COPx", "COPy", "COPz", "Ty"], chans
                kt = "raw ground reaction forces of the instrumented treadmill (left belt), centre of pressure and free moment"
                n_fr_cols.append(len(head))
                n_fr_rows.append(len(body))
                fname = f"samples/{ID}_{code}_T{sp}_forces.csv"
                tcol = "time_s"
            else:
                assert head[0] == "PercGcycle"
                samples = body
                chans = head[1:]
                units = [proc_unit(c) for c in chans]
                hz = np.nan
                kt = ("processed ensemble averages over the runner's footfalls at 2.5, 3.5 and 4.5 m/s, right and left leg, normalized to 101 "
                      "points of the gait cycle (3D hip, knee and ankle angles, joint moments, ground reaction force, joint powers; suffix 25, 35, 45 = speed)")
                n_pr_rows.append(len(body))
                n_pr_cols.append(len(head))
                fname = f"samples/{ID}_{code}_processed.csv"
                tcol = "gait_cycle_pct"
            out = tmp / Path(fname).name
            with open(out, "w", newline="") as fh:
                fh.write(tcol + "," + ",".join(chans) + "\r\n")
                for s in samples:
                    fh.write(",".join(s) + "\r\n")
            extra.append((fname, str(out)))
            if kind == "processed":
                cond, dur, nsamp = "treadmill running at 2.5, 3.5 and 4.5 m/s (processed ensemble averages)", np.nan, len(samples)
            else:
                cond = f"treadmill running at {SPEEDS[sp]} m/s"
                dur = round(float(samples[-1][0]) + (1 / hz if kind == "forces" else 0.0), 4)
                nsamp = len(samples)
            rows.append(dict(
                dataset=ID, study=SRC, experiment="", group="runners", subject=code, person=code, subject_excluded_by_authors=False,
                age=float(r["Age"]), sex=r["Gender"], handedness="", population="healthy (recreational to competitive runners)",
                setting="lab", device="12-camera motion capture (150 Hz), Bertec instrumented dual-belt treadmill (300 Hz)",
                session=1, block="treadmill running", trial=seq, phase="test", time_s=np.nan, task="running", unit="recording",
                condition=cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=seq, samples_file=fname, signal_kind=kt, channels=";".join(chans), channel_units=";".join(units),
                sample_rate_hz=hz, n_samples=nsamp, duration_s=dur, events="",
                trial_info=(f"n_markers={len(chans) // 3}" if kind == "markers" else ""),
                processing=("authors' time normalization and averaging (inverse dynamics in Visual 3D, 10 Hz low-pass filter)" if kind == "processed"
                            else "none (raw ASCII export)")))
    df = pd.DataFrame(rows)

    pp = pd.DataFrame([dict(person=f"S{n:04d}", height_cm=num(people[s]["Height"]), mass_kg=num(people[s]["Mass"]),
                            dominant_leg=people[s]["Dominance"], running_level=people[s]["Level"],
                            running_experience_months=num(people[s]["Experience"]), sessions_per_week=num(people[s]["SessionsPerWk"]),
                            weekly_volume_km=people[s]["Volume"], pace_min_per_km=num(people[s]["Pace"]))
                       for n, s in enumerate(sids, 1)])
    ck.published("marker sampling rate (Hz)", np.median(rate_m), 150, 1, "Methods, Equipment (rate of acquisition of kinematics)")
    ck.internal("markers per running marker file (most files)", np.median(n_mk_ch) == 32, f"median {np.median(n_mk_ch)} markers")
    ck.published("rows per force file", np.median(n_fr_rows), 9000, 0, "Results, Raw data (9,000 rows and eight columns)")
    ck.published("columns per force file (time plus 7 channels)", np.median(n_fr_cols), 8, 0, "Results, Raw data (eight columns)")
    ck.published("length of each running recording (s)", np.median(last_t), 30, 0.1, "Methods, Protocol step 6 (data recorded for 30 s)")
    ck.published("rows per processed file (101 gait-cycle points)", np.median(n_pr_rows), 101, 0, "Methods, Data processing (101 time points)")
    ck.published("columns per processed file (144 + gait-cycle percentage)", np.median(n_pr_cols), 145, 0, "Results, Processed data (101 rows and 145 columns)")
    ck.internal("39 runners kept", df.person.nunique() == len(sids) == 39, str(df.person.nunique()))
    ck.internal("every runner has 3.5 m/s marker trajectories",
                all(((df.person == p) & df.condition.str.endswith("3.5 m/s") & df.signal_kind.str.startswith("raw 3D")).any() for p in df.person.unique()))
    nm = int(df.signal_kind.str.startswith("raw 3D").sum())
    ng = int(df.signal_kind.str.startswith("raw ground").sum())
    ck.internal("every marker recording has a force recording", nm == ng, f"{nm} marker and {ng} force recordings")
    ck.internal("all sample files written", len(extra) == len(df))
    ck.internal("ages plausible", df.age.between(15, 70).all(), f"{df.age.min()}-{df.age.max()}")

    meta = dict(
        study=ID,
        paper=dict(authors="Fukuchi RK, Fukuchi CA, Duarte M", year=2017,
                   title="A public dataset of running biomechanics and the effects of running speed on lower extremity kinematics and kinetics",
                   venue="PeerJ 5:e3298", doi="10.7717/peerj.3298"),
        lab="Duarte (Federal University of ABC, Brazil)", data_url="https://doi.org/10.6084/m9.figshare.4543435",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Converted every raw 30 s treadmill running trial (RBDS<nnn>runT<25|35|45>markers.txt and forces.txt) and each runner's processed "
                 "file from tab-separated text to comma-separated CSV with the authors' values unchanged; the force files' frame number "
                 "was turned into time_s = (frame - 1) / 300; the c3d files, the 1 s static calibration trials and the Visual 3D model and "
                 "pipeline files are left out; age and sex were added from RBDSinfo.txt, and a few training and body measures are in participants.csv "
                 "(injury history, dates, shoe details and strength and flexibility tests are left out)."),
        how=["One row is one recording: the marker trajectories (mm, 150 Hz, columns X, Y, Z of each marker; blank cells are gaps), the "
             "force data of the left belt of the instrumented treadmill (Fx, Fy, Fz in N, centre of pressure COPx, COPy, COPz in mm, free "
             "moment Ty; 300 Hz), or a runner's processed file. A running trial at one speed therefore has two rows, and each runner has "
             "one more row for the processed file (101 points of the gait cycle at 2.5, 3.5 and 4.5 m/s, right and left leg; first column gait_cycle_pct).",
             "Runners ran for 30 s on the treadmill at 2.5, 3.5 and 4.5 m/s, in that order, after 3 min at each speed. Samples are raw: marker gaps and force offsets are "
             "kept as the authors left them. The authors state the free moment Ty in N*m, but the magnitudes suggest N*mm (coordinates are in mm); "
             "the unit is therefore given as the authors state it with this caveat. The force time column of the deposit holds frame numbers, so time_s is "
             "(frame - 1) / 300.",
             "Later-added runners differ from the paper's protocol in places: the number of markers per file varies (trial_info n_markers; 32 in runners 1 to 28 and 37 to 39, 48 or 84 in runners 29 to 36), one runner's recordings are shorter than 30 s, and some files have blank cells where a marker or a force sample is missing. Not every runner has all three speeds: 31 have 2.5 and 4.5 m/s files and all 39 have 3.5 m/s. Age and sex are from RBDSinfo.txt; the "
             "dominant leg is the leg the runner would use to kick a ball."],
        unit_note="recording (one 30 s running trial and one signal type, or a runner's processed file)",
        differences=["The paper describes 28 runners (Subject numbers 1 to 28); the current figshare version holds 39, so the paper's participant count and "
                     "demographic table do not apply to this release. Table 1 of the paper lists 48 markers, but 20 of them are anatomical markers removed after the standing calibration; the running files hold 32 markers (96 coordinate columns), so the table's marker count was not used as a check. The 2.5 and 4.5 m/s trials exist for only 31 of the 39 runners."])
    names_ = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df[names_]
    pc = [("height_cm", "number", "cm", "Height."), ("mass_kg", "number", "kg", "Body mass."),
          ("dominant_leg", "text", "R or L", "Leg the runner would use to kick a ball."),
          ("running_level", "text", "", "Self-assessed running level."),
          ("running_experience_months", "number", "months", "Months of regular running practice (at least three times a week)."),
          ("sessions_per_week", "number", "", "Running training sessions per week."),
          ("weekly_volume_km", "text", "km/week", "Weekly running volume, in the authors' ranges."),
          ("pace_min_per_km", "number", "min/km", "Average pace in the runner's shortest long-distance race.")]
    stdlib.release(ID, MODULE, df, meta, ck, participants=pp, participant_columns=pc, extra_files=extra)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
