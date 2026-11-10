"""Wang & van den Bogert (2020) J Neurosci Methods 334:108580, data of the standing-balance experiment with long-duration random square
pulses of treadmill sway (Wang & van den Bogert 2020, Zenodo): 6 healthy adults (final protocol; the 2 pilot participants are not in the processed
files) x 4 five-minute trials: quiet standing, perturbed, perturbed, quiet standing. Per trial: 32 marker positions, force-plate forces/moments,
and joint angles, velocities, moments and contact forces of the hip, knee and ankle (100 Hz).
Deposit: Zenodo 3819630 (CC BY 4.0), Processed_Data.zip (the 2.9 GB raw data, with EMG and 1000 Hz analog channels, are not used).
Run from the site folder: python3 -I tools/std/convert/Wang2020_JNeurosciMethods.py"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wang2020_JNeurosciMethods"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
z = zipfile.ZipFile(raw / "Processed_Data.zip")
SUBJ = {3: ("M", 18, 79.12), 4: ("M", 27, 63.10), 5: ("M", 32, 70.56), 6: ("M", 35, 58.24), 7: ("M", 28, 68.75), 8: ("M", 27, 60.33)}  # dataset documentation, Table I
BLOCK = {1: "Q1", 2: "P1", 3: "P2", 4: "Q2"}


def rd(sub, kind, tr):
    return pd.read_csv(io.BytesIO(z.read(f"Processed_Data/Subj{sub:02d}/{kind}{tr:04d}.txt")), sep=r"\s+")


BASE = list(rd(3, "Mocap", 1).columns)
assert len(BASE) == 104 and BASE[:2] == ["TimeStamp", "FrameNumber"]
MARK = [c for c in BASE[2:] if c.startswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) and ".Pos" in c]
assert len(MARK) == 96
GRF = [c for c in BASE[2:] if c.startswith("FP1.")]
MOTION = ["HipA", "KneeA", "AnkleA", "HipV", "KneeV", "AnkleV", "HipM", "KneeM", "AnkleM", "HipFx", "HipFy", "KneeFx", "KneeFy", "AnkleFx", "AnkleFy"]
UNITS = {c: "m" for c in MARK}
UNITS.update({c: ("N" if ".For" in c else "N*m") for c in GRF})
UNITS.update({c: ("rad" if c.endswith("A") else "rad/s" if c.endswith("V") else "N*m" if c.endswith("M") else "N") for c in MOTION})
CH = MARK + GRF + MOTION
rows, files, fixes = [], [], []
mass_check, sway, quiet_sway = {}, [], []
for sub, (sex, age, mass_doc) in SUBJ.items():
    for tr in (1, 2, 3, 4):
        m, mo = rd(sub, "Mocap", tr), rd(sub, "Motion", tr)
        if list(m.columns) != BASE:  # one file (participant 8, trial 1) labels the markers M1..M32 in order
            assert len(m.columns) == 104 and m.columns[:2].tolist() == ["TimeStamp", "FrameNumber"] and m.columns[2].startswith("M1.")
            ref = rd(sub, "Mocap", 2)
            new = BASE[:2] + [c for c in BASE[2:]]
            mean_new = m.iloc[:, 2:98].mean().values
            dd = np.abs(mean_new - ref.iloc[:, 2:98].mean().values).reshape(32, 3).max(axis=1)
            assert dd.max() < 0.1 and dd[-5:].max() < 0.01, "marker order guess failed"  # body markers move a few cm between trials; the 5 treadmill markers do not
            m.columns = new
            fixes.append((sub, tr))
        assert len(m) == len(mo) and abs(mo.Time.iloc[-1] - (m.TimeStamp.iloc[-1] - m.TimeStamp.iloc[0])) < 0.02
        assert mo.columns.tolist() == ["Time"] + MOTION
        a = pd.concat([mo[["Time"]].rename(columns={"Time": "time_s"}), m[MARK + GRF].reset_index(drop=True), mo[MOTION].reset_index(drop=True)], axis=1)
        fn = f"samples/S{sub:02d}_trial{tr}.csv"
        files.append((fn, a))
        perturbed = tr in (2, 3)
        if tr == 4:
            mass_check[sub] = float(np.median(m["FP1.ForY"]) / 9.81)
        sw = m[[f"T{i}.PosX" for i in range(1, 6)]].mean(axis=1)
        (sway if perturbed else quiet_sway).append(float(sw.max() - sw.min()))
        rows.append(dict(
            dataset=DS, study="Wang & van den Bogert 2020 J Neurosci Methods", experiment="", group="", subject=f"Subj{sub:02d}", person=f"Subj{sub:02d}",
            subject_excluded_by_authors=False, age=float(age), sex=sex, handedness="", population="healthy adults", setting="lab",
            device="V-Gait treadmill (Motek) as sway platform with force plates; Motion Analysis Osprey cameras (100 Hz)", session=1, block=BLOCK[tr], trial=tr,
            time_s="", phase="baseline" if tr == 1 else "test", task="balance and posture", unit="recording",
            condition="perturbed standing (random square pulses of treadmill sway)" if perturbed else "quiet standing",
            pert_type="treadmill_sway_random_square_pulses" if perturbed else "none", pert_value=5.0 if perturbed else 0.0,
            pert_unit="cm (largest pulse amplitude; pulses of -5, -2.5, 0, 2.5, 5 cm lasting 0.25-1.5 s)", feedback="normal", excluded_by_authors=False,
            recording=tr, samples_file=fn,
            signal_kind="3D positions of 32 markers (27 on the body, 5 on the treadmill frame), force-plate forces and moments (inertia-compensated in perturbed trials), and hip, knee and ankle joint angles, angular velocities, moments and joint contact forces",
            channels=";".join(CH), channel_units=";".join(UNITS[c] for c in CH), sample_rate_hz=100, n_samples=len(a), duration_s=float(a.time_s.iloc[-1] - a.time_s.iloc[0]),
            events="", processing=("as shared: marker gaps filled with PCHIP interpolation; force-plate data of perturbed trials compensated for treadmill inertia; joint angles and "
                                   "moments from a 2D seven-link model, averaged over left and right, zeroed by the mean of the first 10 s of the trial"),
            trial_info=f"trial_label={BLOCK[tr]}"))
df = pd.DataFrame(rows)


def extra():
    for fn, a in files:
        yield fn, a.to_csv(index=False, float_format="%.7g", lineterminator="\n")


part = [dict(person=f"Subj{s:02d}", mass_kg_documentation=v[2], mass_kg_from_quiet_trial_4=mass_check[s]) for s, v in SUBJ.items()]
pcols = [("mass_kg_documentation", "number", "kg", "Body mass listed in the dataset documentation (Table I), computed by the authors from the vertical ground reaction force in quiet standing."),
         ("mass_kg_from_quiet_trial_4", "number", "kg", "Median vertical ground reaction force of the participant's last (quiet standing) trial divided by 9.81, as a check of the listed mass.")]

for s, v in SUBJ.items():
    ck.published(f"mass of participant {s}, kg (documentation Table I vs median vertical force of quiet trial 4)", mass_check[s], v[2], 0.3, "dataset documentation: Table I")
ck.published("participants in the processed data (final protocol, participants 3-8)", len(SUBJ), 6, 0, "dataset documentation: Protocol (the identification used the last six participants)")
ck.published("recordings (4 trials of 6 participants)", len(df), 24, 0, "dataset documentation: Protocol (four trials per participant)")
ck.published("markers per recording (27 body + 5 treadmill)", len(MARK) / 3, 32, 0, "dataset documentation: Equipment (32 markers)")
ck.published("motion capture sampling rate, Hz", 1 / np.diff(files[0][1].time_s).mean(), 100, 0.5, "dataset documentation: Equipment (100 Hz)")
ck.published("treadmill sway range in perturbed trials (peak to peak), m", float(np.median(sway)), 0.10, 0.005, "dataset documentation: Perturbation Signal (square pulses between -5 and +5 cm)")
ck.internal("quiet trials show no treadmill sway (range under 5 mm)", max(quiet_sway) < 0.005, str(max(quiet_sway)))
ck.internal("24 sample files, same length for markers and joint data", len(files) == 24)

meta = dict(
    paper=dict(authors="Wang H, van den Bogert AJ", year=2020, title="Identification of the human postural control system through stochastic trajectory optimization",
               venue="Journal of Neuroscience Methods 334:108580", doi="10.1016/j.jneumeth.2020.108580"),
    lab="van den Bogert (Cleveland State University)", data_url="https://doi.org/10.5281/zenodo.3819630", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept the deposit's processed files of 6 participants x 4 trials: Mocap (32 marker positions and 6 force-plate channels) and Motion (joint angles, velocities, "
             "moments and contact forces) joined side by side in one CSV per trial at 100 Hz, with the Motion time column as time_s (values to 7 significant digits). The two "
             "timestamp columns of the Mocap files (an acquisition clock and a frame counter), the data-quality files, figures and the raw data (including EMG, the 1000 Hz analog "
             "channels and the treadmill and accelerometer records) are not included; in one file the markers carry the generic names M1-M32 and are renamed by position."),
    how=["One row is one five-minute standing trial of one participant (about 310 s are recorded). Trials 1 and 4 are quiet standing; trials 2 and 3 are standing while the "
         "treadmill sways forwards and backwards in random square pulses (amplitudes -5, -2.5, 0, 2.5, 5 cm, durations 0.25-1.5 s); all participants had the same pulse sequence. "
         "Marker coordinates are in metres (Y vertical, X the sway direction; the five treadmill markers T1-T5 show the actual sway).",
         "Joint angles are in radians and velocities in rad/s, inferred from their size; the documentation gives their sign convention (counterclockwise positive) but no units. "
         "Moments are in N*m and joint contact forces in N. Age and sex are from the documentation's participant table (participants 3-8).",
         "Participants 1 and 2 (the pilot participants of the protocol) have no processed files in the deposit and are not included."],
    unit_note="standing trial of five minutes (samples in samples/)",
    differences=["The vertical force of the first (quiet standing) trial of participant 3 is about zero in the deposit (force-plate data missing), so the mass check uses each participant's "
                 "last trial. The deposit's documentation is a thesis chapter, not the 2020 paper, which only cites the data."],
)
stdlib.release(DS, "timeseries", df, meta, ck, participants=part, participant_columns=pcols, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
