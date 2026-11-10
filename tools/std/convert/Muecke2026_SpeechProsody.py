"""Muecke, Pagel, Georgis, Taheri-Ardali & Roessig (2026), Speech Prosody 2026:574-578: 3D head movement of 12 native Persian speakers measured with
electromagnetic articulography (Carstens AG 501, three head sensors: both ears and the nose bridge, 250 Hz) while they said a target word under four focus conditions.

Raw files, raw_extra/Muecke2026_SpeechProsody/ (https://osf.io/cukwp/):
  persian_head_movement_data_target_raw.csv: one row per 4 ms sample, with speaker, recording id, focus, word, the start of the three syllables and the end of the third,
      time (ms) and x, y, z of the left ear, right ear and nose sensors; only the window around the target word is shared.
  (the 'smoothed' file of the deposit is a processed version of the same samples and is not converted; the analysis script persian_visual_prominence_analysis.Rmd was read as text)
Run from the site folder: python3 -I tools/std/convert/Muecke2026_SpeechProsody.py
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Muecke2026_SpeechProsody"
SRC = "Muecke et al. 2026, Speech Prosody 2026:574-578"
raw = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "scratch" / f"{ID}_samples"
CH = [f"{s}_{a}" for s in ("lear", "rear", "nose") for a in "xyz"]
NAMES = ["left_ear_x", "left_ear_y", "left_ear_z", "right_ear_x", "right_ear_y", "right_ear_z", "nose_x", "nose_y", "nose_z"]
ORDER = ["lear_x", "lear_y", "lear_z", "rear_x", "rear_y", "rear_z", "nose_x", "nose_y", "nose_z"]
UNIT = "mm (inferred)"
DEVICE = "Carstens AG 501 electromagnetic articulograph (sampled at 1250 Hz, shared at 250 Hz), sensors behind both ears and on the bridge of the nose; audio recorded at the same time (not shared)"
KIND = "EMA positions (x, y, z) of the left-ear, right-ear and nose-bridge head sensors around a Persian target word spoken in a question-answer game"

d = pd.read_csv(raw / "persian_head_movement_data_target_raw.csv")
assert list(d.columns) == ["speaker", "id", "focus", "word", "syll1_start", "syll2_start", "syll3_start", "syll3_end", "time"] + ORDER, list(d.columns)
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)

rows, extra = [], []
speakers = list(dict.fromkeys(d["speaker"]))
pcode = {s: f"P{i:04d}" for i, s in enumerate(speakers, start=1)}
cnt = {s: 0 for s in speakers}
pad_start, pad_end, steps, nan = [], [], set(), int(d.isna().sum().sum())
nrec = {}
for rid, g in d.groupby("id", sort=False):
    s = g["speaker"].iloc[0]
    assert g[["speaker", "focus", "word", "syll1_start", "syll2_start", "syll3_start", "syll3_end"]].nunique().max() == 1
    cnt[s] += 1
    k = cnt[s]
    t = g["time"].to_numpy() / 1000.0
    steps |= set(np.round(np.diff(g["time"].to_numpy()), 3).tolist())
    r0 = g.iloc[0]
    pad_start.append(r0["syll1_start"] - g["time"].min())
    pad_end.append(g["time"].max() - r0["syll3_end"])
    arc = f"samples/{pcode[s]}_rec{k:03d}.csv"
    with open(TMP / f"{pcode[s]}_rec{k:03d}.csv", "w", newline="") as fh:
        fh.write("time_s," + ",".join(NAMES) + "\r\n")
        np.savetxt(fh, np.column_stack([t, g[ORDER].to_numpy()]), fmt=["%.3f"] + ["%.7g"] * 9, delimiter=",", newline="\r\n")
    extra.append((arc, TMP / f"{pcode[s]}_rec{k:03d}.csv"))
    num = int(re.search(r"_(\d+)$", rid).group(1))
    ev = ";".join(f"{n}={r0[c] / 1000.0:.6f}" for n, c in (("syllable1_start", "syll1_start"), ("syllable2_start", "syll2_start"), ("syllable3_start", "syll3_start"), ("syllable3_end", "syll3_end")))
    dur = (g["time"].max() - g["time"].min()) / 1000.0 + 0.004
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group="", subject=s, person=s, subject_excluded_by_authors=False, age=None,
        sex=None, handedness=None, population="healthy adult, native Persian speakers living in Cologne", setting="lab", device=DEVICE,
        session=1, block=r0["word"], trial=k, phase="other", time_s=None, task="speech", unit="recording",
        condition=r0["focus"], pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=k, samples_file=arc, signal_kind=KIND, channels=";".join(NAMES), channel_units=";".join([UNIT] * 9),
        sample_rate_hz=250.0, n_samples=len(g), duration_s=dur, events=ev, trial_info=f"target_word={r0['word']};authors_recording_number={num}",
        processing="as released (raw file: the window from 100 ms before the target word's first syllable to 100 ms after its third syllable; sampled at 1250 Hz and down-sampled to 250 Hz by the authors)"))
df = pd.DataFrame(rows)

cells = d.drop_duplicates("id").groupby(["speaker", "focus", "word"]).size()
ck = stdlib.Checks()
ck.published("native Persian speakers", int(df["person"].nunique()), 12, 0, "Muecke et al. 2026, Sec. 2.1 (12 native speakers of Persian)")
ck.published("target sentences recorded (8 words x 4 focus conditions x 12 speakers x 3 repetitions) minus the 2 discarded", len(df), 1152 - 2, 0, "Muecke et al. 2026, Sec. 2.1 (1152 target sentences recorded, 2 utterances discarded)")
ck.published("target words", int(df["block"].nunique()), 8, 0, "Muecke et al. 2026, Sec. 2.2 (8 Persian-sounding pseudo-names)")
ck.published("focus conditions", int(df["condition"].nunique()), 4, 0, "Muecke et al. 2026, Abstract and Sec. 1.3 (background, broad, narrow and corrective focus)")
ck.published("sampling rate of the shared kinematic data (Hz)", float(np.round(1000.0 / np.median(list(steps)), 3)), 250, 0, "Muecke et al. 2026, Sec. 2.1 (sampled at 1250 Hz and downsampled to 250 Hz)")
ck.published("sensors tracking the head (behind both ears and on the bridge of the nose)", len(NAMES) // 3, 3, 0, "Muecke et al. 2026, Sec. 2.1 and Fig. 1")
ck.internal("no missing values; all time steps 4 ms within recordings", nan == 0 and steps == {4.0}, f"nan {nan}, steps {sorted(steps)}")
ck.internal("window runs from 100 ms (within one sample) before syllable 1 to 100 ms after the end of syllable 3 in every recording",
            np.abs(np.array(pad_start) - 100).max() <= 4 and np.abs(np.array(pad_end) - 100).max() <= 4, f"start {min(pad_start):.1f}-{max(pad_start):.1f}, end {min(pad_end):.1f}-{max(pad_end):.1f} ms")
ck.internal("3 repetitions per speaker, focus and word, except the 2 discarded", int((cells == 3).sum()) == 12 * 4 * 8 - 2 and int((cells == 2).sum()) == 2, f"cells with 3: {(cells == 3).sum()}, with 2: {(cells == 2).sum()}")
ck.internal("rows in the file equal samples in the sample files", len(d) == int(df["n_samples"].sum()), f"{len(d)} rows")
mean = d.groupby("id")[ORDER].mean()
ck.internal("axes: nose at larger x than both ears, nose above the ears (larger y) and left ear at larger z than the right ear in all recordings",
            bool(((mean["nose_x"] > mean["lear_x"]) & (mean["nose_x"] > mean["rear_x"]) & (mean["nose_y"] > mean["lear_y"]) & (mean["lear_z"] > mean["rear_z"])).all()), "")
print("recordings", len(df), "| samples", len(d), "| cells", dict(cells.value_counts()))
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)

meta = dict(
    paper=dict(authors="Muecke D, Pagel L, Georgis P, Taheri-Ardali M, Roessig S", year=2026,
               title="Visual prosody in Persian: Tracking 3D head movement patterns with electromagnetic articulography",
               venue="Speech Prosody 2026:574-578", doi="10.21437/speechprosody.2026-116"),
    lab="Muecke (University of Cologne)",
    data_url="https://osf.io/cukwp/",
    license="none stated", license_url="",
    changes=("MODIFIED from the authors' file: persian_head_movement_data_target_raw.csv was split into one CSV per recording in samples/ (time in ms divided by 1000 "
             "to give time_s, the 9 sensor columns renamed left_ear_x ... nose_z, values unchanged to 7 significant digits); the speaker, focus, word and syllable columns "
             "were moved to the table. The authors' 'smoothed' file was not converted."),
    how=["One row is one recording of a spoken target sentence: samples_file holds the x, y and z positions of three head sensors (left ear, right ear, nose bridge) at 250 Hz "
         "for the window around the target word (from 100 ms before its first syllable to 100 ms after its third syllable). condition is the focus condition "
         "(background, broad, narrow, corrective), block and target_word in trial_info are the target pseudo-word, and events lists the start of the three syllables and the end "
         "of the third in seconds on the time axis of the samples file (time_s is the time within the whole recording, so it does not start at 0).",
         "The deposit states no unit. Positions are labelled mm (inferred) because the sensor is a Carstens AG 501 and the ear-to-ear distance is about 130 units "
         "(plausible in mm). The authors label x as front-back, y as up-down and z as left-right; in the data the nose sensor lies at the larger x, the nose above the ears (larger y) "
         "and the left ear at the larger z in every recording, so x grows to the front, y upwards and z to the left (inferred). The authors' script flips the sign of z for their plots.",
         "Age and sex are not given per speaker (the paper reports 7 female and 5 male speakers aged 25 to 43). trial is the order of the speaker's recordings in the file; "
         "the authors' own recording number is in trial_info. The task was a question-answer game with a virtual avatar; the speech audio and the oral articulator sensors are not part of this deposit."],
    unit_note="EMA recording (one speaker saying one target sentence)",
    differences=["The paper's results (displacement and velocity of the nose sensor under the four focus conditions) were fitted with models on the authors' processed data and were not reproduced here; only the design counts and the sampling rate are checked."],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)
shutil.rmtree(TMP, ignore_errors=True)
