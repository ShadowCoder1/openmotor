"""Vielemeyer, Staufenberg, Schreff, Rixen & Muller 2023 (R Soc Open Sci): ten adults walked along a 5 m walkway imitating the humanoid
robot LOLA (2 speeds x 2 postures x 2 mass distributions). Deposit (Figshare 20381559): kinematics.zip (Vicon marker exports, one
text file per trial), kinetics.zip (Kistler BioWare exports of four force plates, one .txt per trial; the .dat files are BioWare's
binary originals and are not read) and LOLA.zip (the robot's own sensor data, not human movement, not converted)."""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Vielemeyer2023_RSOS"
MODULE = "timeseries"
ORDER = ["normal", "vlola", "groucho_vlola", "groucho", "m_normal", "m_vlola", "m_groucho_vlola", "m_groucho"]
FORCE = ["Fx", "Fy", "Fz", "Ft", "Fx12", "Fx34", "Fy14", "Fy23", "Fz1", "Fz2", "Fz3", "Fz4", "Ax", "Ay"]
FORCE_RAW = ["Fx", "Fy", "Fz", "|Ft|", "Fx12", "Fx34", "Fy14", "Fy23", "Fz1", "Fz2", "Fz3", "Fz4", "Ax", "Ay"]
FORCE_U = ["N"] * 12 + ["m", "m"]
DEVICE = "5 m walkway with four Kistler force plates (250 Hz) and a 10-camera Vicon system (250 Hz)"
HZ = 250


def label(code):
    mass = code.startswith("m_")
    c = code[2:] if mass else code
    speed = "slow (~0.5 m/s)" if "vlola" in c else "preferred (~1.3 m/s)"
    post = "crouched" if "groucho" in c else "upright"
    return f"{speed} speed, {post} posture, {'added mass on legs' if mass else 'no added mass'}"


def main():
    ck = stdlib.Checks()
    zk = zipfile.ZipFile(stdlib.raw_dir(DS) / "kinematics.zip")
    zf = zipfile.ZipFile(stdlib.raw_dir(DS) / "kinetics.zip")
    out = Path("scratch") / "vie_samples"
    out.mkdir(parents=True, exist_ok=True)
    trials = {}
    for kind, zz in (("kin", zk), ("frc", zf)):
        for n in zz.namelist():
            if not n.endswith(".txt"):
                continue
            _, s, f = n.split("/")
            m = re.fullmatch(r"(.+?)[_ ](\d+)\.txt", f)
            code = m[1].replace("nomal", "normal")
            assert code in ORDER, f
            trials.setdefault((int(s), code, int(m[2])), {})[kind] = n
    assert all(set(v) == {"kin", "frc"} for v in trials.values())
    rows, extra, rec_no = [], [], {}
    kin_rates, frate_rates = {}, {}
    plates, nsamp, markers_seen, hz_ok = set(), [], set(), True
    keys = sorted(trials, key=lambda k: (k[0], ORDER.index(k[1]), k[2]))
    for (s, code, num) in keys:
        rec_no[s] = rec_no.get(s, 0)
        for kind in ("kin", "frc"):
            n = trials[(s, code, num)][kind]
            raw = (zk if kind == "kin" else zf).read(n).decode("utf-8-sig" if kind == "kin" else "latin1", errors="replace").replace("\r", "").split("\n")
            rec_no[s] += 1
            r = rec_no[s]
            fn = f"samples/S{s:02d}_{code}_{num:02d}_{'markers' if kind == 'kin' else 'forceplates'}.csv"
            lp = out / fn.split("/")[1]
            if kind == "kin":
                assert raw[0].strip("﻿") == "Trajectories"
                khz = int(raw[1])
                kin_rates.setdefault(s, set()).add(khz)
                names = [x for x in raw[2].split("\t")[2:] if x]
                assert len(names) == 16
                names = [x.split(":")[-1].strip() for x in names]  # drops the participant-specific marker-set prefix
                markers_seen.add(tuple(names))
                chans = [f"{nm}_{a}" for nm in names for a in "xyz"]
                with open(lp, "w", newline="") as fh:
                    fh.write("time_s," + ",".join(chans) + "\r\n")
                    cnt = 0
                    for l in raw[5:]:
                        if not l.strip():
                            continue
                        p = l.split("\t")
                        assert p[1] in ("0", ""), p[:3]
                        vals = (p + [""] * 50)[2:50]
                        fh.write(f"{(int(p[0]) - 1) / khz:g}," + ",".join(vals) + "\r\n")
                        cnt += 1
                nsamp.append(cnt)
                ev, units, kindname = "", ";".join(["mm"] * 48), "Vicon marker positions (16 markers x, y, z); blank where a marker was not visible"
                proc = "none stated for the exported trajectories (the paper filtered them with a 4th-order bidirectional Butterworth low-pass at 50 Hz in its own analysis)"
                dur = (cnt - 1) / khz
            else:
                h = raw[17].split("\t")
                dv = raw[1].split("\t")
                for p in range(4):
                    assert h[1 + 42 * p:15 + 42 * p] == FORCE_RAW, (n, p)
                    plates.add(dv[1 + 42 * p].split()[0])
                frate = float(raw[3].split("\t")[1])
                frate_rates.setdefault(s, set()).add(frate)
                hz_ok &= frate in (250, 300)
                chans = [f"plate{p + 1}_{c}" for p in range(4) for c in FORCE]
                idx = [1 + 42 * p + j for p in range(4) for j in range(14)]
                cnt = 0
                evs = []
                for p in range(4):
                    c0 = 1 + 42 * p
                    evs.append(f"plate{p + 1}_contact_start={float(raw[6].split(chr(9))[c0]):g}")
                    evs.append(f"plate{p + 1}_contact_end={float(raw[7].split(chr(9))[c0]):g}")
                with open(lp, "w", newline="") as fh:
                    fh.write("time_s," + ",".join(chans) + "\r\n")
                    for l in raw[19:]:
                        if not l.strip():
                            continue
                        p = l.rstrip("\t").split("\t")
                        fh.write(p[0] + "," + ",".join(p[i] for i in idx) + "\r\n")
                        cnt += 1
                assert cnt == int(raw[2].split("\t")[1]), (n, cnt)
                ev = ";".join(evs)
                units = ";".join(FORCE_U * 4)
                kindname = "force plate signals (4 plates: 3 force components, total force, 8 sensor sums, centre of pressure)"
                proc = "BioWare export; the paper filtered kinetic data with a 4th-order bidirectional Butterworth low-pass at 30 Hz in its own analysis"
                dur = (cnt - 1) / frate
            extra.append((fn, str(lp)))
            rows.append(dict(
                dataset=DS, study=DS, experiment="", group="", subject=f"s{s}", subject_excluded_by_authors=False, population="healthy",
                setting="lab", device=DEVICE, session=1, block=code, trial=r, phase="test", task="walking", unit="recording",
                condition=label(code), pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                age="", sex="", handedness="", time_s="", recording=r, samples_file=fn, signal_kind=kindname,
                channels=";".join(chans), channel_units=units, sample_rate_hz=(khz if kind == "kin" else int(frate)), n_samples=cnt, duration_s=dur, events=ev,
                processing=proc, trial_info=f"authors_trial_number={num};measurement={'kinematics' if kind == 'kin' else 'kinetics'}"))
    df = pd.DataFrame(rows)
    ck.internal("each of the 551 trials has a kinematics and a kinetics recording: 1,102 rows", len(df) == 1102 and len(keys) == 551)
    ck.internal("all kinetics files are 250 Hz and the 4 plates are the same devices in every file", hz_ok and len(plates) == 4,
                str(plates))
    ck.internal("the 16 markers have the same names and order in every kinematics file", len(markers_seen) == 1)
    ck.published("participants analysed", df.subject.nunique(), 10, 0, "Methods 2.1, Participants")
    ck.published("experimental conditions (2 speeds x 2 postures x 2 mass distributions)", df.block.nunique(), 8, 0, "Methods 2.2")
    ck.published("sampling rate of force plates, Hz", df[df.channels.str.startswith("plate")].sample_rate_hz.iloc[0], 250, 0, "Methods 2.2")
    ck.published("number of force plates", len(plates), 4, 0, "Methods 2.2")
    ck.published("number of markers (toe, malleolus, femoral epicondyle, trochanter, acromion, humeral epicondyle, ulnar styloid on both sides, L5, C7)",
                 len(next(iter(markers_seen))), 16, 0, "Methods 2.2")
    per = df[df.channels.str.startswith("plate")].groupby(["subject", "block"]).size()
    meta = dict(
        study=DS,
        paper=dict(authors="Vielemeyer J, Staufenberg NS, Schreff L, Rixen D, Muller R", year=2023,
                   title="Walking like a robot: do the ground reaction forces still intersect near one point when humans imitate a humanoid robot?",
                   venue="R Soc open sci 10(5):221473", doi="10.1098/rsos.221473"),
        lab="Muller (Bayreuth / Jena)", data_url="https://doi.org/10.6084/m9.figshare.20381559",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Wrote each trial's exported marker trajectories and force-plate signals to CSV files (values kept as exported, time in seconds "
                 "from the frame or sample number); for the force plates kept 14 of the 42 channels per plate; the participant folders became codes; "
                 "the file name typo 'nomal' was read as 'normal'. The robot's data (LOLA.zip) and BioWare's binary .dat files are not included."),
        how=["One row is one recording of one walking trial: each trial has a kinematics row (16 Vicon markers, x/y/z in mm, 250 Hz, time_s = (frame - 1)/rate, "
             "blank where a marker was not seen) and a kinetics row (four force plates at 250 Hz). The authors' trial number, shared by the two rows, is in trial_info. "
             "The deposit does not state how the two recordings are aligned in time beyond a common camera trigger, so the two time axes are given separately.",
             "Force-plate channels per plate: Fx, Fy, Fz (force components, N), Ft (total force magnitude), Fx12, Fx34, Fy14, Fy23, Fz1-Fz4 (sensor-pair sums, N) and "
             "Ax, Ay (centre of pressure, m); axes and signs are those of the BioWare export and the deposit does not describe them. The 28 further channels per plate in the export (centre-of-mass "
             "acceleration, velocity, position, power, momentum, work and force derivatives computed by BioWare for a nominal 650 N weight) are not included. Participants used plates 1-2 for the slow "
             "conditions and plates 3-4 for the preferred speed, so the other plates hold noise. events are BioWare's contact periods per plate in seconds.",
             "Conditions: 'normal' = preferred speed (~1.3 m/s) upright; 'vlola' = slow (~0.5 m/s) upright; 'groucho' = crouched posture; 'm_' = weights added to the legs. block is the deposit's condition name. "
             "Rows are ordered by participant, then condition in the order the paper describes (control, slow, slow crouched, preferred crouched, then the same four with mass), then trial number; the true "
             "trial order within the experiment is not in the deposit."],
        unit_note="walking trial (one kinematics or kinetics recording of a pass along the walkway)",
        differences=[f"The paper says cameras and force plates ran at 250 Hz; for one participant the kinematics files and the kinetics files state 300 Hz in their headers: the kinematics files of {sorted(s_ for s_,v in kin_rates.items() if v != {250})} (subject number of the deposit folder; the same participant's force plates also state 300 Hz) state a different rate and are kept with their stated rate. The paper says participants completed 5-10 trials per condition; the deposit holds {int(per.min())}-{int(per.max())} per participant and condition (551 trials in all), probably because it includes the unusable trials the authors did not analyse. All files are kept and none are marked as excluded because the deposit does not say which trials the authors analysed. The paper's analyses (spatio-temporal parameters, virtual pivot point) were not reproduced from the rows; its Table 1 is an image."],
    )
    stdlib.release(DS, MODULE, df, meta, ck, extra_files=extra)


if __name__ == "__main__":
    main()
