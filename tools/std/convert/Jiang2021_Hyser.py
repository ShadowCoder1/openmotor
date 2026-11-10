"""Jiang et al. (2021), IEEE TNSRE 29:1035-1046: Hyser dataset (PhysioNet hd-semg 2.0.0), finger-force recordings only.

Run from the site folder: python3 -I tools/std/convert/Jiang2021_Hyser.py
Raw files (untrusted data, read only; fetched and hash-checked against the deposit's SHA256SUMS.txt) are the 2,400 force recordings
(.dat + .hea) of the MVC, 1-DoF, N-DoF and random sub-datasets, stored flattened as <folder>__<subject folder>__<file>.
The 256-channel HD-sEMG (about 140 GB) and the pattern-recognition sub-dataset (EMG and gesture labels only) are not used.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Jiang2021_Hyser"
RAW = stdlib.raw_dir(ID)
FINGERS = ["thumb", "index", "middle", "ring", "little"]
checks = stdlib.Checks()


def read_rec(folder, subj, sess, stem):
    base = RAW / f"{folder}__subject{subj:02d}_session{sess}__{stem}"
    hea = base.with_suffix(".hea").read_text().splitlines()
    top = hea[0].split()
    assert top[0] == stem and int(top[1]) == 5, (stem, top)
    fs, n = float(top[2]), int(top[3])
    raw = np.frombuffer(base.with_suffix(".dat").read_bytes(), dtype="<i2")
    assert raw.size == 5 * n, (stem, raw.size, n)
    adc = raw.reshape(n, 5).astype(float)
    phys = np.empty_like(adc)
    for j in range(5):
        f = hea[1 + j].split()
        assert f[0] == stem + ".dat" and f[1] == "16" and f[-1] == FINGERS[j], (stem, f)
        m = re.fullmatch(r"([-0-9.eE+]+)\((-?\d+)\)/V", f[2])
        gain, baseline = float(m.group(1)), int(m.group(2))
        init, csum = int(f[5]), int(f[6])
        assert int(raw.reshape(n, 5)[0, j]) == init, (stem, j, "first sample differs from header")
        s = int(raw.reshape(n, 5)[:, j].astype(np.int64).sum()) & 0xFFFF
        s = s - 65536 if s >= 32768 else s
        assert s == csum, (stem, j, "checksum differs from header")
        phys[:, j] = (adc[:, j] - baseline) / gain       # volts, as the deposit's own loader does
    return fs, n, phys


rows, files = [], []
mvc_sign_ok = mvc_total = 0
per_session = {}
trial_counter = {}
for subj in range(1, 21):
    for sess in (1, 2):
        recs = []
        for i in range(1, 11):                    # MVC: odd = flexion, even = extension (toolbox load_mvc.m)
            finger = (i + 1) // 2
            d = "flexion" if i % 2 == 1 else "extension"
            recs.append(("mvc_dataset", f"mvc_force_finger{finger}_{d}", "mvc", f"MVC {FINGERS[finger - 1]} {d}",
                         f"finger={FINGERS[finger - 1]};direction={d}", finger, d))
        for finger in range(1, 6):
            for k in range(1, 4):
                recs.append(("1dof_dataset", f"1dof_force_finger{finger}_sample{k}", "1dof", f"1-DoF {FINGERS[finger - 1]}",
                             f"finger={FINGERS[finger - 1]};sample={k}", None, None))
        for u in range(1, 16):
            for k in range(1, 3):
                recs.append(("ndof_dataset", f"ndof_force_combination{u}_sample{k}", "ndof", f"N-DoF combination {u}",
                             f"combination={u};sample={k}", None, None))
        for k in range(1, 6):
            recs.append(("random_dataset", f"random_force_sample{k}", "random", "random finger combinations",
                         f"sample={k}", None, None))
        per_session[(subj, sess)] = len(recs)
        for folder, stem, exp, cond, info, finger, d in recs:
            fs, n, phys = read_rec(folder, subj, sess, stem)
            if exp == "mvc":
                mvc_total += 1
                x = phys[:, finger - 1]
                ext = x[np.argmax(np.abs(x))]
                mvc_sign_ok += (ext < 0) if d == "flexion" else (ext > 0)
            t = trial_counter.get(subj, 0) + 1
            trial_counter[subj] = t
            fn = f"samples/{ID}_S{subj:04d}_rec{t:04d}.csv"
            lines = ["time_s,thumb,index,middle,ring,little"]
            for r in range(n):
                lines.append(f"{r / fs:.2f}," + ",".join(f"{v:.8g}" for v in phys[r]))
            files.append((fn, "\r\n".join(lines) + "\r\n"))
            rows.append(dict(
                dataset=ID, study=ID, experiment=exp, group="", subject=f"s{subj:02d}", person=f"s{subj:02d}",
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device="five uniaxial fingertip force sensors with amplifiers (Huatran SAS and HSGA); 256-channel HD-sEMG "
                       "(Quattrocento) was recorded at the same time and is not included",
                session=sess, block=exp, trial=t, phase="test", time_s="", task="hand gestures and grasps", unit="recording",
                condition=cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=t, samples_file=fn, signal_kind="isometric fingertip forces of the five fingers (force-sensor amplifier output)",
                channels=";".join(FINGERS), channel_units=";".join(["V"] * 5), sample_rate_hz=fs, n_samples=n,
                duration_s=n / fs, events="",
                processing="low-pass filtered by the authors (8th-order Butterworth, 10 Hz); converted from the stored 16-bit "
                           "values to volts with the gain and baseline of the WFDB header",
                trial_info=info))

t = pd.DataFrame(rows)

# ---- published numbers: PhysioNet record page and readme (20 subjects, 2 sessions each = 40 folders, 100 Hz force), toolbox loaders (recordings per session)
checks.published("participants", t.person.nunique(), 20, 0, "PhysioNet record page, Abstract and Methods: 20 subjects")
checks.published("subject-session folders (20 subjects x 2 sessions)", t.groupby(["person", "session"]).ngroups, 40, 0,
                 "PhysioNet record page, Data Description: 40 folders subjectXX_sessionY per sub-dataset")
checks.published("force sampling rate (Hz), all recordings", t.sample_rate_hz.mean(), 100, 0,
                 "PhysioNet record page, Methods: force signals acquired at 100 Hz")
checks.published("force channels per recording (thumb, index, middle, ring, little)", len(FINGERS), 5, 0,
                 "PhysioNet record page, Data Description: force matrix is N x 5")
checks.published("MVC recordings per session (5 fingers x flexion/extension)", (t.experiment == "mvc").sum() / 40, 10, 0,
                 "PhysioNet record page: MVC files per finger and direction; toolbox load_mvc.m loops over 10")
checks.published("N-DoF recordings per session (15 combinations x 2 samples)", (t.experiment == "ndof").sum() / 40, 30, 0,
                 "PhysioNet record page: 15 finger combinations; toolbox load_ndof.m loops over 15 x 2")
checks.published("1-DoF recordings per session (5 fingers x 3 samples)", (t.experiment == "1dof").sum() / 40, 15, 0,
                 "Toolbox main_1dof.m (feature = cell(5,3)); PhysioNet record page: five fingers")
checks.published("random-task recordings per session", (t.experiment == "random").sum() / 40, 5, 0,
                 "Toolbox load_random.m loops over 5 samples")

# ---- internal checks
checks.internal("all 2,400 force recordings kept (40 sessions x 60)", len(t) == 2400 == len(files), str(len(t)))
checks.internal("every recording matches its header first sample and checksum", True)
checks.internal("every session has the same 60 recordings", set(per_session.values()) == {60})
frac = mvc_sign_ok / mvc_total
checks.internal("sign: peak of the pressed finger is negative in flexion and positive in extension MVC recordings (>= 85%)",
                frac >= 0.85, f"{frac:.3f} of {mvc_total}")
checks.internal("sample file names carry the release subject codes (S0001..S0020 in order)",
                all(r["samples_file"].split("_S")[-1][:4] == f"{int(r['subject'][1:]):04d}" for r in rows))
print("MVC sign agreement", frac)
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']}, reported {ch['reported']}, ok={ch['ok']}")

meta = dict(
    study="Jiang et al. 2021",
    paper=dict(authors="Jiang X, Liu X, Fan J, Ye X, Dai C, Clancy EA, Akay M, Chen W",
               title="Open Access Dataset, Toolbox and Benchmark Processing Results of High-Density Surface Electromyogram Recordings",
               venue="IEEE Trans Neural Syst Rehabil Eng 29:1035-1046", year=2021, doi="10.1109/tnsre.2021.3082551"),
    lab="Chen (Fudan University)", data_url="https://physionet.org/content/hd-semg/2.0.0/",
    license="ODC-By 1.0", license_url="https://opendatacommons.org/licenses/by/1-0/",
    changes=("Converted to the OpenMotor timeseries format from the finger-force recordings only (MVC, 1-DoF, N-DoF and random "
             "sub-datasets, 2,400 recordings). The 256-channel HD-sEMG (raw and preprocessed, about 140 GB) and the pattern-recognition "
             "sub-dataset (34 hand gestures; EMG and gesture labels only, no force) are not included. Force values are converted from "
             "the stored 16-bit integers to volts with the gain and baseline in each WFDB header, as the authors' own loader does; "
             "nothing was resampled or filtered."),
    how=[
        "One row is one force recording from one of 20 healthy adults, in one of two sessions on separate days: five columns of the "
        "recording's CSV (samples/...) hold the force of the thumb, index, middle, ring and little finger at 100 Hz, in volts at the "
        "force-sensor amplifier output. The deposit does not give a volts-to-newtons conversion (the sensors are rated for 200 N); "
        "the authors normalise force by each person's maximal voluntary contraction (the MVC recordings) before analysing it.",
        "Sign (from the authors' loader, not stated in the deposit text): negative force is finger flexion and positive force is "
        "extension, as in their normalisation code, which divides negative values by the flexion MVC. The released rows agree: the peak "
        "of the pressed finger is negative in the flexion MVC recordings and positive in the extension ones.",
        "Experiments: mvc = maximal voluntary contraction of one finger in one direction (10 recordings per session, 10 s); "
        "1dof = force-varying contraction of one finger tracking a target force trajectory (5 fingers x 3 samples, 25 s); "
        "ndof = prescribed contractions of one of 15 finger combinations (x 2 samples, 25 s); random = free random contraction of "
        "finger combinations without a prescribed trajectory (5 samples, 25 s). The deposit does not list which fingers make up "
        "each of the 15 combinations, so condition gives the combination number only. The target trajectories are not in the deposit.",
        "Rows are ordered by participant, session, then experiment (mvc, 1dof, ndof, random) and the deposit's file numbering; the "
        "deposit does not give the order in which they were recorded. The authors' low-pass filter (8th order, 10 Hz) is already in "
        "the force files. Age and sex are not given per participant (the deposit says 12 men and 8 women, aged 22 to 34).",
        "subject and person are the same: one participant, both sessions in one sequence.",
    ],
    unit_note="force recording (five fingers)",
    differences=[
        "The full text of the IEEE paper could not be read (the publisher's PDF is behind a bot check), so the checks use the "
        "PhysioNet record page, readme and the authors' toolbox code instead of numbers printed in the paper.",
    ],
)
stdlib.release(ID, "timeseries", t, meta, checks, extra_files=files)
