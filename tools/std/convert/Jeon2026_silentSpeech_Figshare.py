"""Jeon, Sohn, Jung, Kim & Im (2026), figshare 33326679: surface EMG and inertial (IMU) recordings of 10 healthy participants silently
articulating 100 words, 5 sessions each (for a silent speech interface).

Raw files, raw_extra/Jeon2026_silentSpeech_Figshare/ (fetched with tools/std/fetch.py; CC BY 4.0):
  EMG_Accel_Silent_Speech_opendata.zip   S01..S10 / EMG / session_1..5 / raw_word<1..100>.mat   variable save_data_raw, 8192 x 6 (raw
        BioSemi ActiveTwo sEMG, 2048 Hz, 4 s, six electrodes; values with 4 decimals)
                                          S01..S10 / EMG / session_1..5 / word<1..100>.mat       variable save_data, 2048 x 6 (the authors'
        preprocessed sEMG: downsampled to 512 Hz, 60 Hz notch, 20-210 Hz band-pass; from the Figshare description)
                                          S01..S10 / IMU / session_1..5 / word<1..100>.mat       variable save_data, 5 x 6 x T int16 (five
        MPU-9250 sensors x six components x time; T is 374 to 547 though the description says about 400)
  Word_List.csv                          Number, Content (the word in Korean script), English (its meaning)
What the description says (Figshare record, read as text): 10 healthy participants (6 male, 4 female, 24.6 +- 2.50 years), 5 sessions of
100 words each articulated silently once, in random order, about 4 s per word; EMG from six electrodes over facial and neck regions at
2,048 Hz (BioSemi ActiveTwo); IMU at about 100 Hz with components x, y, z, roll, pitch, yaw. Nothing states units or sensor positions.
Two passes over the .mat files: the first reads every file for the table and the checks (about a minute), the second writes the sample
CSVs straight into the ZIP while stdlib.release runs (a generator, so no 4 GB of temporary text).
Run from the site folder: python3 -I tools/std/convert/Jeon2026_silentSpeech_Figshare.py
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Jeon2026_silentSpeech_Figshare"
SRC = "Jeon et al. 2026, figshare"
BASE = "EMG_Accel_Silent_Speech_opendata"
DEVICE = ("BioSemi ActiveTwo (six surface EMG electrodes, 2,048 Hz) and five MPU-9250 inertial sensors read by an ESP32-S3 (about "
          "100 Hz) on the face and under the chin")
EMG_CH = [f"emg{i}" for i in range(1, 7)]
IMU_CH = [f"imu{s}_{c}" for s in range(1, 6) for c in ("x", "y", "z", "roll", "pitch", "yaw")]
NOMINAL_IMU_HZ = 100.0
# key -> (file in the participant folder, variable, rate, channels, unit, signal_kind, processing, float format of the CSV)
SIGNALS = [
    ("emg_raw", "EMG/session_{s}/raw_word{w}.mat", "save_data_raw", 2048.0, EMG_CH, "not stated",
     "surface EMG, raw, six electrodes over facial and neck regions (BioSemi ActiveTwo)",
     "none (the raw recording as shared)", "%.10g"),
    ("emg", "EMG/session_{s}/word{w}.mat", "save_data", 512.0, EMG_CH, "not stated",
     "surface EMG, preprocessed by the authors, six electrodes over facial and neck regions",
     "as the authors describe: downsampled from 2,048 to 512 Hz, 60 Hz notch filter, 20-210 Hz band-pass filter (not repeated here)",
     "%.8g"),
    ("imu", "IMU/session_{s}/word{w}.mat", "save_data", NOMINAL_IMU_HZ, IMU_CH, "raw sensor counts (16-bit integers)",
     "inertial measurement units: five MPU-9250 sensors on facial and submental regions, six components each (x, y, z, roll, pitch, yaw)",
     "none (raw sensor counts as shared; the authors' truncation or interpolation to 400 samples and downsampling to 50 Hz were applied "
     "only to their model input)", "%.8g"),
]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def raise_priority():
    """Windows throttles a background ssh job to a crawl; ask for a normal-plus priority (this converter only runs on the PC)."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x8000)  # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:  # noqa: BLE001
        pass


def load(zf, s, sess, w, pat, var):
    mat = sio.loadmat(io.BytesIO(zf.read(f"{BASE}/S{s:02d}/" + pat.format(s=sess, w=w))))
    assert [k for k in mat if not k.startswith("__")] == [var], (s, sess, w, pat)
    return mat[var]


def main():
    raise_priority()
    raw = stdlib.raw_dir(ID)
    zf = zipfile.ZipFile(raw / "EMG_Accel_Silent_Speech_opendata.zip")
    words_bytes = (raw / "Word_List.csv").read_bytes()
    words = list(csv.DictReader(io.StringIO(words_bytes.decode("utf-8-sig"))))
    assert [int(w["Number"]) for w in words] == list(range(1, 101)), "word list is not 1..100"

    # ---- pass 1: the table and what the checks need -----------------------------------------------------------------------------
    rows, sig_of = [], []
    imu_n, nan_total, shape_ok, imu_extremes, imu_shape = [], 0, True, [0, 0], None
    for s in range(1, 11):
        rec = 0
        for sess in range(1, 6):
            for w in range(1, 101):
                wd = words[w - 1]
                for key, pat, var, rate, ch, unit, kind, proc, fmt in SIGNALS:
                    a = load(zf, s, sess, w, pat, var)
                    if key == "imu":
                        assert a.ndim == 3 and a.shape[:2] == (5, 6) and a.dtype == np.int16, (s, sess, w, a.shape, a.dtype)
                        imu_shape = a.shape[:2]
                        n = a.shape[2]
                        imu_n.append(n)
                        imu_extremes[0], imu_extremes[1] = min(imu_extremes[0], int(a.min())), max(imu_extremes[1], int(a.max()))
                    else:
                        nan_total += int(np.isnan(a).sum())
                        shape_ok &= a.shape == (int(round(4 * rate)), 6)
                        n = a.shape[0]
                    rec += 1
                    arc = f"samples/P{s:04d}_session{sess}_word{w:03d}_{key}.csv"  # P000s: the code stdlib.release gives participant s
                    sig_of.append(key)
                    rows.append(dict(
                        dataset=ID, study=SRC, experiment="", group="", subject=f"S{s:02d}", person=f"S{s:02d}",
                        subject_excluded_by_authors=False, age=None, sex=None, handedness=None, population="healthy", setting="lab",
                        device=DEVICE, session=sess, block="", trial=(sess - 1) * 100 + w, phase="other", time_s=None, task="speech",
                        unit="recording", condition=f"word {w}: {wd['English']} ({wd['Content']})", pert_type="none", pert_value=0,
                        pert_unit="", feedback="none", excluded_by_authors=False,
                        recording=rec, samples_file=arc, signal_kind=kind, channels=";".join(ch),
                        channel_units=";".join([unit] * len(ch)), sample_rate_hz=rate, n_samples=n, duration_s=n / rate, events="",
                        processing=proc))

    # ---- checks -----------------------------------------------------------------------------------------------------------------
    d = pd.DataFrame(rows)
    d["sig"] = sig_of
    ck = stdlib.Checks()
    R = "Figshare record description"
    emg, pre, im = d[d.sig == "emg_raw"], d[d.sig == "emg"], d[d.sig == "imu"]
    ck.published("participants", d.person.nunique(), 10, 0, f"{R}, Dataset Overview")
    ck.published("recording sessions per participant", int(d.groupby("person").session.nunique().min()), 5, 0, f"{R}, Experimental Protocol")
    ck.published("words per session (each articulated once)", int(emg.groupby(["person", "session"]).size().min()), 100, 0,
                 f"{R}, Experimental Protocol")
    ck.published("target words in the word list", len(words), 100, 0, f"{R}, Dataset Overview (Word_List.csv)")
    ck.published("silent articulations (10 participants x 5 sessions x 100 words)", len(emg), 5000, 0, R)
    ck.published("surface EMG electrodes", len(emg.channels.iloc[0].split(";")), 6, 0, f"{R}, Data Acquisition (sEMG)")
    ck.published("raw sEMG sampling rate (Hz)", float(emg.sample_rate_hz.iloc[0]), 2048, 0, f"{R}, Data Acquisition (sEMG)")
    ck.published("preprocessed sEMG sampling rate (Hz)", float(pre.sample_rate_hz.iloc[0]), 512, 0,
                 f"{R}, Data Acquisition (sEMG, downsampled from 2,048 to 512 Hz)")
    ck.published("duration of each raw sEMG recording (s)", float(emg.duration_s.mean()), 4, 0,
                 f"{R}, Data Acquisition (each trial is a 4-s silent articulation)")
    ck.published("duration of each preprocessed sEMG recording (s)", float(pre.duration_s.mean()), 4, 0, f"{R}, Data Acquisition")
    ck.published("IMU sensors per recording", imu_shape[0], 5, 0, f"{R}, Data Acquisition (IMU)")
    ck.published("IMU measurement components per sensor", imu_shape[1], 6, 0, f"{R}, Data Acquisition (IMU: x, y, z, roll, pitch, yaw)")
    ck.internal("15,000 rows: raw sEMG, preprocessed sEMG and IMU for each of the 5,000 articulations",
                len(d) == 15000 and d.groupby("sig").size().eq(5000).all() and d.groupby(["person", "session", "trial"]).size().eq(3).all())
    ck.internal("every sEMG file has the expected size (8,192 x 6 raw, 2,048 x 6 preprocessed) and no missing values", shape_ok and nan_total == 0,
                f"{nan_total} NaN")
    ck.internal("sample file names are unique", d.samples_file.is_unique)
    ck.internal("IMU counts stay within the 16-bit range", imu_extremes[0] >= -32768 and imu_extremes[1] <= 32767, str(imu_extremes))
    a = np.array(imu_n)
    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)
    print("rows", len(d), "| IMU samples per recording: min", int(a.min()), "median", int(np.median(a)), "max", int(a.max()),
          "| exactly 400:", int((a == 400).sum()))
    if not ck.passed():
        sys.exit("checks failed")

    # ---- pass 2: the sample CSVs, written into the ZIP one at a time ------------------------------------------------------------
    spec = {key: (pat, var, rate, ch, fmt) for key, pat, var, rate, ch, unit, kind, proc, fmt in SIGNALS}

    def samples():
        for r, key in zip(rows, sig_of):
            pat, var, rate, ch, fmt = spec[key]
            s = int(r["person"][1:])
            a_ = load(zf, s, r["session"], int(r["samples_file"].split("_word")[1][:3]), pat, var)
            if key == "imu":
                a_ = a_.reshape(30, -1).T  # (time, sensor-major x component)
            df = pd.DataFrame(a_, columns=ch)
            df.insert(0, "time_s", np.arange(a_.shape[0]) / rate)
            assert a_.shape[0] == r["n_samples"]
            yield r["samples_file"], df.to_csv(index=False, float_format=fmt, lineterminator="\r\n").encode("utf-8")
        yield "word_list.csv", words_bytes

    s_, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s_["lab"], data_url=s_["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Each .mat file became one row and one CSV in samples/ (time_s, then one column per channel); the raw sEMG values are "
                 "copied exactly, the preprocessed sEMG to 8 significant digits and the IMU counts as integers; time_s is the sample "
                 "index divided by the sampling rate; the IMU array (sensor x component x time) was written as 30 columns, sensor "
                 "by sensor; participant folder names were replaced by codes and the word list copied as word_list.csv."),
        how=[
            "One row is one sensor file of the deposit: for each of 10 participants, 5 sessions and 100 words (each silently "
            "articulated once per session, about 4 s), the raw sEMG of six electrodes (2,048 Hz, 8,192 samples), the authors' "
            "preprocessed sEMG (512 Hz, 2,048 samples) and the IMU (five sensors with x, y, z, roll, pitch and yaw each), so 15,000 "
            "rows and sample files. The words are in word_list.csv; condition gives the word's number, English meaning and Korean "
            "spelling.",
            "The order of the words within a session was random and is not in the deposit, so trial and recording follow the session "
            "and word number. The deposit states no units: sEMG values are kept as shared and the IMU values are the sensors' raw "
            "16-bit counts. Age and sex are not given per participant (6 men and 4 women, 24.6 +- 2.5 years as a group).",
            "The IMU files have 374 to 547 samples each (median 490) where the description says about 400 at about 100 Hz, and the "
            "files carry no time stamps; time_s of the IMU samples is therefore nominal (index divided by the authors' 100 Hz) and runs "
            "past 4 s, and the true IMU rate is probably closer to 120 Hz.",
            "The participants articulated the words silently, so there was no voice and no auditory feedback of it (feedback is none). "
            "Only the description on Figshare was available: no paper that goes with the dataset was found, so the checks compare the "
            "converted rows with the numbers in that description.",
        ],
        unit_note="sensor file of one silent articulation (raw sEMG, preprocessed sEMG or IMU)",
        differences=[
            f"The description gives about 400 IMU samples per 4-s recording at about 100 Hz; the files hold {int(a.min())} to {int(a.max())} "
            f"(median {int(np.median(a))}), and only {int((a == 400).sum())} of 5,000 have exactly 400.",
            "The description speaks of 100 English words; the word list gives each word in Korean script (column Content) with its "
            "English meaning (column English).",
        ],
    )
    rec_ = stdlib.release(ID, "timeseries", rows, meta, ck, extra_files=samples())

    # ---- after the release: the codes and file names in the ZIP agree with the table ---------------------------------------------
    with zipfile.ZipFile(stdlib.OUT / f"{ID}.zip") as z_:
        main_csv = list(csv.reader(io.TextIOWrapper(z_.open(f"{ID}/{ID}.csv"), encoding="utf-8", newline="")))
        members = {n[len(ID) + 1:] for n in z_.namelist()}
    hdr = main_csv[0]
    pi, fi = hdr.index("person"), hdr.index("samples_file")
    order = list(dict.fromkeys(r[pi] for r in main_csv[1:]))
    assert order == [f"{ID}_P{k:04d}" for k in range(1, 11)], "participant codes differ"
    assert all(r[fi].startswith(f"samples/P{int(r[pi][-4:]):04d}_") for r in main_csv[1:]), "participant codes in the sample file names differ"
    assert {r[fi] for r in main_csv[1:]} <= members and len(main_csv) - 1 == 15000 and "word_list.csv" in members, "a sample file is missing"
    print("verified: 15000 sample files in the ZIP, participant codes consistent")
    return rec_


if __name__ == "__main__":
    main()
