"""Mico-Amigo et al. (2023), J NeuroEngineering Rehabil 20:78 (Mobilise-D): example real-world recordings, one participant from each of the six
technical-validation cohorts (congestive heart failure, COPD, healthy older adult, multiple sclerosis, Parkinson's disease, proximal femoral
fracture). Lower-back inertial sensor (McRoberts Dynaport MM+, 100 Hz) plus the INDIP reference system (pressure insoles, foot distance sensors).
One row per participant (a recording of 1.2 to 2.6 hours).

Run from the site folder: python3 -I tools/std/convert/MicoAmigo2023_JNeuroengRehabil.py
Source (untrusted data, read only): the deposit's single 1.3 GB zip at https://zenodo.org/records/7547125 (stored, not compressed). The disk is small,
so the converter reads each member (<cohort>/data.mat, <cohort>/infoForAlgo.mat) from the zip with HTTP range requests, checks its CRC-32 against the
zip's directory, parses it with scipy in memory and writes the samples CSV. raw_extra/<id>/SOURCES.json keeps the record of the zip.
"""
import io
import shutil
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "MicoAmigo2023_JNeuroengRehabil"
TMP = stdlib.HERE / "tmp_samples" / ID
TMP.mkdir(parents=True, exist_ok=True)
checks = stdlib.Checks()
U = "https://zenodo.org/api/records/7547125/files/Mobilise-D%20dataset_1-18-2023.zip/content"
ZSIZE = 1341286272
COHORTS = [("CHF", "congestive heart failure"), ("COPD", "chronic obstructive pulmonary disease"), ("HA", "healthy older adult"),
           ("MS", "multiple sclerosis"), ("PD", "Parkinson's disease"), ("PFF", "proximal femoral fracture")]


def rng(a, b):
    for t in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(U, headers={"Range": f"bytes={a}-{b}"}), timeout=600).read()
        except Exception as e:  # 429 and network drops: back off
            time.sleep(30 * (t + 1))
    raise RuntimeError("download failed")


tail = rng(ZSIZE - 70000, ZSIZE - 1)
i = tail.rfind(b"PK\x05\x06")
n_ent, cdsize, cdoff = struct.unpack("<HII", tail[i + 10:i + 20])
cd = rng(cdoff, cdoff + cdsize - 1)
ENT, p = {}, 0
while cd[p:p + 4] == b"PK\x01\x02":
    meth = struct.unpack("<H", cd[p + 10:p + 12])[0]
    crc = struct.unpack("<I", cd[p + 16:p + 20])[0]
    csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
    nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
    off = struct.unpack("<I", cd[p + 42:p + 46])[0]
    ENT[cd[p + 46:p + 46 + nl].decode()] = (off, csz, meth, crc)
    p += 46 + nl + el + cl
assert n_ent == len(ENT) == 13


def member(name):
    off, csz, meth, crc = ENT[name]
    assert meth == 0
    h = rng(off, off + 29)
    nl, el = struct.unpack("<HH", h[26:30])
    b = rng(off + 30 + nl + el, off + 30 + nl + el + csz - 1)
    assert len(b) == csz and (zlib.crc32(b) & 0xFFFFFFFF) == crc, name
    return b


ACC = ["acc_x_g", "acc_y_g", "acc_z_g"]
GYR = ["gyr_x_dps", "gyr_y_dps", "gyr_z_dps"]
MAG = ["mag_x", "mag_y", "mag_z"]
CH = (ACC + GYR + MAG + ["barometer"] + [f"left_insole_p{k:02d}" for k in range(1, 17)] + [f"right_insole_p{k:02d}" for k in range(1, 17)]
      + ["left_foot_distance_1", "left_foot_distance_2", "right_foot_distance_1", "right_foot_distance_2"])
UNITS = (["g"] * 3 + ["deg/s"] * 3 + ["uT (inferred)"] * 3 + ["hPa (inferred)"] + ["normalized pressure (0-1)"] * 32 + ["cm (inferred)"] * 4)

def load(c):
    """Parse one cohort's data.mat into a DataFrame (time_s + 46 channels); returns the table and (n, seconds, 10 ms spacing ok)."""
    d = sio.loadmat(io.BytesIO(member(f"{c}/data.mat")), squeeze_me=True, struct_as_record=False)["data"].TimeMeasure1.Recording4
    lb, st = d.SU.LowerBack, d.Standards
    assert lb.Fs.Acc == lb.Fs.Gyr == lb.Fs.Mag == lb.Fs.Bar == 100
    for s in (st.PressureInsoles_raw.LeftFoot, st.PressureInsoles_raw.RightFoot, st.DistanceModule_raw.LeftFoot, st.DistanceModule_raw.RightFoot):
        assert s.Fs == 100 and s.Presence == 1
    t = np.asarray(lb.Timestamp, dtype=float)
    A = np.column_stack([lb.Acc, lb.Gyr, lb.Mag, lb.Bar, st.PressureInsoles_raw.LeftFoot.NormalizedPressure,
                         st.PressureInsoles_raw.RightFoot.NormalizedPressure, st.DistanceModule_raw.LeftFoot.Distance,
                         st.DistanceModule_raw.RightFoot.Distance])
    assert A.shape == (len(t), len(CH)) and len(CH) == len(UNITS) == 46, (A.shape, len(CH))
    assert not np.isnan(A).any(), int(np.isnan(A).sum())
    ok = bool(np.all(abs(np.diff(t) - 0.01) < 1e-5))
    df = pd.DataFrame(A, columns=CH)
    df.insert(0, "time_s", np.round(t - t[0], 5))
    return df, (len(t), float(t[-1] - t[0]), ok)


class Samples:
    """Lazy extra_files for stdlib.release: writes one cohort's CSV just before release zips it and deletes it right after (the PC disk is small)."""

    def __iter__(self):
        for c, _ in COHORTS:
            out = TMP / f"{ID}_{c}.csv"
            df, _m = load(c)
            df.to_csv(out, index=False, float_format="%.6g", lineterminator="\r\n")
            del df
            yield f"samples/{ID}_{c}.csv", str(out)
            out.unlink()


rows, part = [], []
dur = {}
dts_ok = True
for c, pop in COHORTS:
    info = sio.loadmat(io.BytesIO(member(f"{c}/infoForAlgo.mat")), squeeze_me=True, struct_as_record=False)["infoForAlgo"].TimeMeasure1
    _df, (n, secs, ok) = load(c)
    del _df
    dts_ok = dts_ok and ok
    dur[c] = secs
    fn = f"samples/{ID}_{c}.csv"
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group=pop, subject=c, person=c, subject_excluded_by_authors=False,
        age="", sex="", handedness={"R": "right", "L": "left"}.get(str(info.Handedness), ""), population=pop, setting="home",
        device="McRoberts Dynaport MM+ on the lower back (100 Hz) and the INDIP reference system (pressure insoles, foot distance sensors)",
        session=1, block="Recording4", trial=1, phase="test", time_s="", task="walking", unit="recording",
        condition="real-world monitoring with a few requested tasks", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
        excluded_by_authors=False,
        recording=1, samples_file=fn, signal_kind="lower-back accelerometer, gyroscope, magnetometer and barometer; plantar pressure and foot distance sensors (reference system)",
        channels=";".join(CH), channel_units=";".join(UNITS), sample_rate_hz=100, n_samples=n, duration_s=round(secs, 3), events="",
        processing="raw sensor signals as shared (the deposit's *_raw structures); values rounded to 6 significant digits here",
        trial_info=f"walking_aid={int(info.WalkingAid_01)};sensor_attachment={info.SensorAttachment_SU};sensor_type={info.SensorType_SU};tasks=rise from a chair and walk to another room, make a drink in the kitchen, stairs, outdoor walking, inclined path (as far as possible)"))
    part.append(dict(person=c, height_cm=float(info.Height), weight_kg=float(info.Weight), sensor_height_cm=float(info.SensorHeight),
                     foot_size_cm=float(info.FootSize)))
D = pd.DataFrame(rows)
pcols = [("height_cm", "number", "cm", "Body height (infoForAlgo)."), ("weight_kg", "number", "kg", "Body weight (infoForAlgo)."),
         ("sensor_height_cm", "number", "cm", "Height of the lower-back sensor above the floor (infoForAlgo)."),
         ("foot_size_cm", "number", "cm", "Foot size (infoForAlgo), unit inferred.")]

P = checks.published
P("participants (one per cohort)", D.person.nunique(), 6, 0, "Zenodo record description and README: one participant from each of six cohorts")
P("cohorts (five disease cohorts plus healthy older adults)", D.group.nunique(), 6, 0, "Methods, Participants")
P("sampling frequency of the lower-back sensor (Hz)", float(D.sample_rate_hz.iloc[0]), 100, 0, "Methods (Dynaport MM+, 100 Hz)")
checks.internal("sample interval is 10 ms in every recording", dts_ok)
checks.internal("every recording is between 1 and 3 hours", all(3600 < v < 10800 for v in dur.values()), str({k: round(v / 3600, 2) for k, v in dur.items()}))

meta = dict(
    paper=dict(authors="Mico-Amigo ME, Bonci T, Paraschiv-Ionescu A, Ullrich M, Kirk C, Soltani A, Kuderle A, Gazit E, Salis F, Alcock L, Aminian K, Becker C, et al.",
               year=2023, title="Assessing real-world gait with digital technology? Validation, insights and recommendations from the Mobilise-D consortium",
               venue="J NeuroEngineering Rehabil 20(1):78", doi="10.1186/s12984-023-01198-5"),
    lab="Mobilise-D consortium (Del Din / Rochester, Newcastle University)",
    data_url="https://zenodo.org/records/7547125", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each cohort's MATLAB data.mat (Recording 4 only, as the authors provide) became one samples CSV: the lower-back accelerometer, gyroscope, magnetometer and barometer, "
             "the 16-cell pressure insoles of both feet and the two foot distance sensors of both feet, in the deposit's order; the Unix timestamp became time_s from the first sample, "
             "and values were rounded to 6 significant digits (the sensors resolve 1 mg and 70 mdps, coarser than that). The recording's start date and time zone are not copied. "
             "Height, weight, sensor height and foot size from infoForAlgo.mat are in participants.csv. Age and sex are not in the deposit."),
    how=["One row is one participant's whole real-world recording (1.2 to 2.6 hours at 100 Hz) while the participant went about normal activities and was asked to do a few tasks: rise from a chair and walk to another room, "
         "make a drink in the kitchen, stairs, outdoor walking and an inclined path where possible. The participant is the only person in their cohort (group).",
         "Samples (samples/*.csv): time_s, then lower-back acceleration (g, from the values near 1 at rest), angular velocity (deg/s, from the README's gyroscope range), magnetic field, barometric pressure, "
         "normalised pressure of the 16 insole cells of the left and right foot, and the two distance-sensor readings per foot. The README does not give the units of magnetometer, barometer and distance channels; "
         "uT, hPa and cm are inferred from their ranges and the Mobilise-D data standard, and are marked so in channel_units. The distance readings saturate near 255.",
         "The deposit does not include the gait-event or walking-bout annotations of the Mobilise-D reference system, so events is empty."],
    differences=["The Zenodo description speaks of 2.5 hours of real-world data per participant; the six recordings last 1.19, 1.93, 2.08, 2.29, 2.54 and 2.63 hours (mean 2.11). "
                 "The paper's validation statistics (gait sequences, initial contacts, cadence, stride length) are for the full 108-participant study, whose data are not in this record, so only the cohort and sampling checks apply."],
    unit_note="recording (one participant's whole real-world recording)",
)
stdlib.release(ID, "timeseries", D, meta, checks, participants=pd.DataFrame(part), participant_columns=pcols, extra_files=Samples())
shutil.rmtree(TMP, ignore_errors=True)
