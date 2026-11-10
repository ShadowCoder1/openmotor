"""Schwarz et al. 2020 (Sensors): ten chronic stroke survivors wore eight IMUs (sternum, shoulder, upper arm, lower arm, hand, thumb,
index, middle finger) and fingertip force sensors while moving blocks (BL 108 g, BW 490 g, BH 1008 g) to four target positions, three
repetitions each, with each arm (adapted ARAT grasp-and-displace). Deposit (Zenodo 3930752): Data.zip with Subject1..10/Left|Right/
ARAT_BL|BW|BH.mat (IMU.data accelerometer 100 Hz, gyroscope 200 Hz, IMU.dataParams.Inds = movement start/end samples,
force per movement), plus calibration files that are not converted. Table 2 of the paper gives each participant's affected side."""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Schwarz2020_Sensors"
MODULE = "timeseries"
SEG = ["Sternum", "Shoulder", "Upper", "Lower", "HandIMU", "ThumbIMU", "IndexIMU", "MiddleIMU"]
SEGNAME = {"Sternum": "sternum", "Shoulder": "shoulder", "Upper": "upper_arm", "Lower": "lower_arm", "HandIMU": "hand",
           "ThumbIMU": "thumb", "IndexIMU": "index", "MiddleIMU": "middle"}
FORCE = [("ThumbForce", "thumb"), ("IndexForce", "index"), ("MiddleForce", "middle")]
WEIGHT = {"BL": ("BL", "big light block (108 g)"), "BW": ("BW", "big wooden block (490 g)"), "BH": ("BH", "big heavy block (1008 g)")}
# Table 2 of the paper (Schwarz et al. 2020): sex, age (y), months since stroke, affected side, Fugl-Meyer upper extremity /66
TAB2 = {1: ("M", 72, 73, "left", 35), 2: ("M", 73, 30, "right", 33), 3: ("M", 62, 181, "right", 33), 4: ("F", 61, 70, "left", 40),
        5: ("M", 57, 45, "left", 46), 6: ("F", 70, 21, "right", 34), 7: ("F", 70, 33, "left", 34), 8: ("M", 59, 9, "right", 32),
        9: ("F", 42, 20, "right", 32), 10: ("M", 42, 13, "left", 28)}
OUTD = Path("scratch") / "schwarz_samples"


def write(path, t, names, cols):
    df = pd.DataFrame(np.column_stack([t] + cols), columns=["time_s"] + names)
    df.to_csv(path, index=False, float_format="%.7g", lineterminator="\r\n")
    return len(df)


def main():
    ck = stdlib.Checks()
    OUTD.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(next(stdlib.raw_dir(DS).glob("*.zip")))
    files = sorted(n for n in z.namelist() if re.fullmatch(r"Data/Subject\d+/(Left|Right)/ARAT_(BL|BW|BH)\.mat", n))
    rows, extra, persons = [], [], []
    per_sub = {}
    movs_by_arm = {}
    dur = {}
    gmax = 0.0
    nmov, nmatch = [], [0, 0]
    for sub in range(1, 11):
        sex, age, months, aff, fma = TAB2[sub]
        sid = f"Subject{sub}"
        persons.append(dict(person=sid, affected_side=aff, fma_ue=fma, months_since_stroke=months))
        rec = 0
        mine = [n for n in files if f"/Subject{sub}/" in n]
        order = sorted(mine, key=lambda n: (0 if (("/Left/" in n) != (aff == "left")) else 1, ["BL", "BW", "BH"].index(n[-6:-4])))
        for n in order:
            side = "left" if "/Left/" in n else "right"
            w = n[-6:-4]
            arm = "affected arm" if side == aff else "non-affected arm"
            m = sio.loadmat(io.BytesIO(z.read(n)), squeeze_me=True, struct_as_record=False)
            I, P, f = m["IMU"], m["IMU"].dataParams, m["force"]
            assert int(P.fsAcc) == 100 and int(P.fsGyr) == 200
            inds = np.asarray(P.Inds).astype(int)
            assert len(inds) % 2 == 0
            pairs = inds.reshape(-1, 2)
            names_ev = []
            for k in range(1, len(pairs) + 1):
                names_ev += [f"interval{k}_start", f"interval{k}_end"]
            mv = pairs[1:] if len(pairs) >= 13 else pairs
            ev = ";".join(f"{k}={(v - 1) / 100:.2f}" for k, v in zip(names_ev, inds))
            movs_by_arm.setdefault((sub, arm), 0)
            movs_by_arm[(sub, arm)] += 1
            nmov.append((n[5:], len(pairs)))
            dur.setdefault((sub, arm), []).extend(((mv[:, 1] - mv[:, 0]) + 1).tolist())
            fi = []
            for pos in range(1, 5):
                for r in range(1, 4):
                    rr = getattr(getattr(f, f"m{pos}", None), f"rep{r}", None)
                    if rr is None:
                        nmatch[0] += 1
                        continue
                    k = (pos - 1) * 3 + r - 1
                    fl = len(rr.Index)
                    match = k < len(mv) and fl == mv[k, 1] - mv[k, 0] + 1 or bool(np.any((pairs[:, 1] - pairs[:, 0] + 1) == fl))
                    nmatch[0] += 1; nmatch[1] += int(match)
                    fi.append(f"force_position{pos}_rep{r}_bf={int(rr.bf)};force_position{pos}_rep{r}_ef={int(rr.ef)};force_position{pos}_rep{r}_n_samples={fl}")
            data = I.data
            def seg(sname):
                return getattr(data.ARM, sname) if sname in ("Lower", "Upper", "Shoulder", "Sternum") else getattr(data.GLOVE, sname)
            na = len(seg("Lower").Acc)
            accn, acc = [], []
            gyrn, gyr = [], []
            for s in SEG:
                a, g = np.asarray(seg(s).Acc, float), np.asarray(seg(s).Gyr, float)
                assert a.shape == (na, 3) and g.shape == (2 * na - 1, 3) or g.shape == (2 * na, 3) or abs(len(g) - 2 * na) <= 2, (a.shape, g.shape)
                accn += [f"{SEGNAME[s]}_acc_{ax}" for ax in "xyz"]; acc += [a[:, i] for i in range(3)]
                gyrn += [f"{SEGNAME[s]}_gyr_{ax}" for ax in "xyz"]; gyr += [g[:, i] for i in range(3)]
                gmax = max(gmax, np.nanmax(np.abs(g)))
            fn, fc = [], []
            for fk, nm in FORCE:
                v = np.asarray(getattr(data.GLOVE, fk).Force, float)
                assert len(v) == na
                fn.append(f"{nm}_fingertip_force"); fc.append(v)
            base = dict(dataset=DS, study=DS, experiment="", group=arm, subject=sid, person=sid, subject_excluded_by_authors=(sub == 2),
                        age=age, sex=sex, handedness="", population="chronic stroke", setting="lab",
                        device="8 inertial measurement units (accelerometer + gyroscope) on trunk, arm, hand and fingers, with fingertip force sensors",
                        session=1, block=f"{w} {WEIGHT[w][1]}", phase="test", time_s="", task="grasping and lifting", unit="recording",
                        condition=f"{arm} ({side}), {WEIGHT[w][1]}", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                        excluded_by_authors=(sub == 2),
                        processing="calibrated signals as shared in the deposit's Clean Data (the paper reports a 10 Hz low-pass filter in the analysis); raw sensor values (RawAcc, RawGyr) not included")
            # accelerometers + force at 100 Hz
            rec += 1
            fa = f"samples/{sid}_{side}_{w}_acc_force.csv"
            nn = write(OUTD / Path(fa).name, np.arange(na) / 100.0, accn + fn, acc + fc)
            extra.append((fa, str(OUTD / Path(fa).name)))
            rows.append({**base, "trial": rec, "recording": rec, "samples_file": fa,
                         "signal_kind": "accelerometer (8 IMUs) and fingertip force sensors",
                         "channels": ";".join(accn + fn), "channel_units": ";".join(["m/s^2"] * len(accn) + ["ADC counts"] * len(fn)),
                         "sample_rate_hz": 100, "n_samples": nn, "duration_s": round((nn - 1) / 100, 6), "events": ev,
                         "trial_info": ";".join(fi)})
            rec += 1
            ng = len(gyr[0])
            fg = f"samples/{sid}_{side}_{w}_gyro.csv"
            nn = write(OUTD / Path(fg).name, np.arange(ng) / 200.0, gyrn, gyr)
            extra.append((fg, str(OUTD / Path(fg).name)))
            rows.append({**base, "trial": rec, "recording": rec, "samples_file": fg, "signal_kind": "gyroscope (8 IMUs)",
                         "channels": ";".join(gyrn), "channel_units": ";".join(["rad/s"] * len(gyrn)),
                         "sample_rate_hz": 200, "n_samples": nn, "duration_s": round((nn - 1) / 200, 6), "events": ev, "trial_info": ""})
            per_sub.setdefault(sub, []).append((side, w))
    df = pd.DataFrame(rows)
    print("max |gyro|", gmax, "force/Inds length match", nmatch, [x for x in nmov if x[1] != 13])
    ck.internal("gyroscope magnitudes are plausible for rad/s", gmax < 60, str(gmax))
    ck.internal("ten participants", df.subject.nunique() == 10)
    ck.internal("subject 2 has only two block weights per arm (as in the paper)", all(len(set(w for _, w in per_sub[2])) == 2 for _ in [0]))
    # published: 324 data sets for affected arm and for non-affected arm over the nine analysed participants
    for arm in ("affected arm", "non-affected arm"):
        tot = 12 * sum(v for (s, a), v in movs_by_arm.items() if a == arm and s != 2)
        ck.published(f"blocks x 12 movements, nine analysed participants, {arm} (324)", tot, 324, 0, "Results, 36 trials per participant and arm (324 data sets); 27 blocks per arm")
    ck.published("participants recruited (10)", df.subject.nunique(), 10, 0, "Methods, Participants")
    ck.published("participants analysed (9)", df[~df.subject_excluded_by_authors].subject.nunique(), 9, 0, "Abstract/Results")
    ages = np.array([TAB2[s][1] for s in range(1, 11)], float)
    ck.published("mean age (60.8)", ages.mean(), 60.8, 0.05, "Table 2")
    ck.published("SD of age (11.4)", ages.std(ddof=1), 11.4, 0.05, "Table 2")
    # sanity: the affected arm is slower
    slower = np.mean([np.mean(dur[(s, "affected arm")]) > np.mean(dur[(s, "non-affected arm")]) for s in range(1, 11)])
    print("fraction of participants whose affected arm moves slower", slower, [ (s, round(np.mean(dur[(s,'affected arm')])), round(np.mean(dur[(s,'non-affected arm')]))) for s in range(1,11)])
    ck.internal("affected arm has longer movements in most participants (confirms the side assignment)", slower >= 0.7, str(slower))
    ck.internal("recordings", len(df) == 2 * len(files) and len(extra) == len(df), str(len(df)))
    meta = dict(
        study=DS,
        paper=dict(authors="Schwarz A, Bhagubai MMC, Wolterink G, Held JPO, Luft AR, Veltink PH", year=2020,
                   title="Assessment of Upper Limb Movement Impairments after Stroke Using Wearable Inertial Sensing",
                   venue="Sensors 20(17):4770", doi="10.3390/s20174770"),
        lab="Veltink (University of Twente) / Luft (University of Zurich)", data_url="https://doi.org/10.5281/zenodo.3930752",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("From each ARAT_*.mat file wrote the calibrated accelerometer signals of the eight IMUs and the three fingertip force signals (both 100 Hz) to one CSV and the "
                 "calibrated gyroscope signals (200 Hz) to a second CSV, same values (7 significant digits); movement start/end samples (IMU.dataParams.Inds) became event times in seconds and the "
                 "per-movement force start/end detections became trial_info; participant IDs became codes. Not included: raw sensor values (RawAcc, RawGyr), the sensor-synchronisation segments, rotation matrices, "
                 "the sensor-to-segment calibration files (s2s folders), and the Lower.SC vector."),
        how=["One row is one recording, a block of one weight (BL 108 g, BW 490 g, BH 1008 g) carried by one arm of one stroke survivor to 4 target positions x 3 repetitions: each recording has two rows, "
             "one for the accelerometers (24 channels, m/s^2) with the three fingertip force signals (ADC counts, arbitrary units) at 100 Hz, and one for the gyroscopes (24 channels, rad/s) at 200 Hz. "
             "Axes follow the deposit (x, y, z of each sensor in its own frame).",
             "events: the deposit's Inds (movement start and end samples) as interval<k>_start/end in seconds from the start of the recording (sample number at 100 Hz, minus 1). Usually there are 13 intervals: "
             "the first is a short sensor-synchronisation movement and intervals 2-13 are the 12 movements, position 1-4 times repetition 1-3 in order (the readme: 4 positions, each repeated 3 times); "
             "recordings with a different number of intervals are listed in Known differences. Positions 1-4 are, in the paper's order, table height at ipsilateral arm length, table height at abducted arm length, top shelf at ipsilateral length, top shelf at abducted length. "
             "trial_info gives the deposit's per-movement force start (bf) and end (ef) detections, samples counted as in the deposit's readme (500 ms removed at the beginning, plus 50 to align with the force data).",
             "Order: for each participant the non-affected arm comes first, then the affected arm (as the paper describes), blocks in the order BL, BW, BH (the order actually used was randomized and is not in the deposit). "
             "The group column says whether the arm was the affected or non-affected one; the affected side, Fugl-Meyer score and age and sex come from Table 2 of the paper (the deposit does not contain them), assuming SubjectN is participant SN; "
             "this assumption is supported by the affected arm having longer movements in most participants. Participant 2 was discarded by the authors (only two block weights, defective data) and is kept here and marked as excluded."],
        unit_note="block recording (about 4-6 minutes)",
        differences=[f"Most recordings have 13 intervals (one synchronisation movement and 12 movements); exceptions: {[x for x in nmov if x[1] != 13]} (file, number of intervals). Participant 2 has no BW recordings. The per-movement force records (m1-m4, rep1-3) match the lengths of the marked intervals (in the order above, or any interval) in {nmatch[1]} of {nmatch[0]} cases.",
                     "Zenodo lists the record as 'Kinematics of reach-to-grasp and displacement after stroke' with no related paper; the matching paper (data link in its text and references) is Schwarz et al. 2020, Sensors 20(17):4770, doi 10.3390/s20174770 (Crossref checked). The OpenMotor listing has this right.",
                     "The paper says 324 motion data sets; that is the count for one arm (9 participants x 36 movements), and the deposit holds 324 for each arm."],
    )
    pdf = pd.DataFrame(persons)
    pcols = [("affected_side", "category", "left, right", "Stroke-affected side (paper Table 2)."),
             ("fma_ue", "integer", "points of 66", "Fugl-Meyer upper extremity score (paper Table 2)."),
             ("months_since_stroke", "integer", "months", "Time since stroke (paper Table 2).")]
    stdlib.release(DS, MODULE, df, meta, ck, participants=pdf, participant_columns=pcols, extra_files=extra)


if __name__ == "__main__":
    main()
