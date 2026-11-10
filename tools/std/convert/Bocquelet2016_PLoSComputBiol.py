"""Bocquelet et al. (2016), PLoS Comput Biol 12(11):e1005119: the BY2014 articulatory-acoustic dataset, electromagnetic articulography
(EMA, 9 coils, 100 Hz) of one male French speaker reading 672 short sentences and word lists.

Raw files, raw_extra/Bocquelet2016_PLoSComputBiol/ (https://doi.org/10.5281/zenodo.154083):
  BY2014.zip: BY2014/ema_ascii/by_phr_2_NNN_sync.ema (one text file per item: one line per 10 ms sample, 27 columns = x, y, z of 9 coils),
      ema_htk/ (the same data in HTK format, used only to cross-check the sample counts), wav/ (audio, not converted),
      transcription.txt (file name and text of each item), README.txt, license.txt
Run from the site folder: python3 -I tools/std/convert/Bocquelet2016_PLoSComputBiol.py
"""
import re
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Bocquelet2016_PLoSComputBiol"
SRC = "Bocquelet et al. 2016, PLoS Comput Biol 12(11):e1005119"
raw = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "scratch" / f"{ID}_samples"
RATE = 100.0
COILS = ["right_lip_corner", "left_lip_corner", "lower_lip", "upper_lip", "jaw", "tongue_tip", "tongue_dorsum", "tongue_back", "velum"]
NAMES = [f"{c}_{a}" for c in COILS for a in "xyz"]
UNIT = "mm"
DEVICE = "electromagnetic articulograph (9 coils glued on the lips, lip corners, jaw, tongue and soft palate), audio recorded at the same time (not converted)"
KIND = ("EMA coil positions (x, y, z) of the right and left lip corner, lower lip, upper lip, jaw, tongue tip, tongue dorsum, tongue back "
        "and velum (soft palate) while one speaker said a sentence or word list")

z = zipfile.ZipFile(next(raw.glob("*.zip")))
names = sorted(n for n in z.namelist() if not n.startswith("__MACOSX"))
ema = [n for n in names if "/ema_ascii/" in n and n.endswith(".ema")]
trans = {}
for l in z.read("BY2014/transcription.txt").decode("utf-8", "replace").splitlines():
    if l.strip():
        k, _, t = l.partition("\t")
        trans[k.strip()] = t.strip()
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)

rows, extra = [], []
n_tot, htk_ok, bad, orient = 0, 0, 0, {"y": [], "x": []}
htk_period = set()
for k, n in enumerate(ema, start=1):
    stem = n.split("/")[-1][:-4]
    m = re.match(r"^by_phr_(\d+)_(\d+)_sync$", stem)
    d = np.loadtxt(z.open(n))
    assert d.ndim == 2 and d.shape[1] == 27, (n, d.shape)
    bad += int(np.isnan(d).sum())
    h = z.read(n.replace("ema_ascii", "ema_htk"))
    nsamp, period, size, kind = struct.unpack(">iihh", h[:12])
    htk_ok += int(nsamp == len(d) and size == 108 and len(h) == 12 + nsamp * 108)
    htk_period.add(period)
    t = np.arange(len(d)) / RATE
    arc = f"samples/P0001_rec{k:03d}.csv"
    with open(TMP / f"P0001_rec{k:03d}.csv", "w", newline="") as fh:
        fh.write("time_s," + ",".join(NAMES) + "\r\n")
        np.savetxt(fh, np.column_stack([t, d]), fmt=["%.2f"] + ["%.6g"] * 27, delimiter=",", newline="\r\n")
    extra.append((arc, TMP / f"P0001_rec{k:03d}.csv"))
    mean = d.mean(0)
    orient["y"].append(mean[NAMES.index("upper_lip_y")] > mean[NAMES.index("lower_lip_y")])
    orient["x"].append(mean[NAMES.index("tongue_tip_x")] > mean[NAMES.index("tongue_back_x")] > mean[NAMES.index("velum_x")])
    n_tot += len(d)
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group="", subject="BY", person="BY", subject_excluded_by_authors=False, age=None,
        sex="M", handedness=None, population="healthy adult, native French speaker", setting="lab", device=DEVICE,
        session=1, block=f"series {m.group(1)}, item {m.group(2)}", trial=k, phase="other", time_s=None, task="speech", unit="recording",
        condition=trans.get(stem, ""), pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=k, samples_file=arc, signal_kind=KIND, channels=";".join(NAMES), channel_units=";".join([UNIT] * 27),
        sample_rate_hz=RATE, n_samples=len(d), duration_s=len(d) / RATE, events="", trial_info="", processing="as released (the authors' paper says the 400 Hz recording was down-sampled to 100 Hz)"))
extra.append(("transcription.txt", z.read("BY2014/transcription.txt")))
df = pd.DataFrame(rows)

ck = stdlib.Checks()
ck.published("EMA sampling rate after down-sampling (Hz)", float(df["sample_rate_hz"].iloc[0]), 100, 0,
             "Bocquelet et al. 2016, Methods, 'Articulatory data acquisition and parameterization' (down-sampled from 400 Hz to 100 Hz)")
ck.published("coils tracked (nine 3D coils)", len(COILS), 9, 0, "Bocquelet et al. 2016, Methods ('Nine such 3D coils were glued on the tongue tip, dorsum, back, upper lip, lower lip, both lip corners, jaw and soft palate')")
ck.published("raw 3D parameters per sample (9 coils x 3 coordinates)", len(NAMES), 27, 0, "Bocquelet et al. 2016, Methods (the 27-parameter raw 3D parameterization)")
ck.published("speakers of the reference corpus", int(df["person"].nunique()), 1, 0, "Bocquelet et al. 2016, Methods ('a native French male speaker')")
ck.internal("every ASCII file has 27 columns, no missing values, and the HTK copy has the same number of samples (108-byte frames)",
            htk_ok == len(ema) and bad == 0, f"{htk_ok}/{len(ema)} files agree, nan {bad}")
ck.internal("one recording per ema_ascii file and a transcription for each", len(df) == 672 and (df["condition"] != "").all(), f"{len(df)} recordings")
ck.internal("upper lip above lower lip (larger y) in nearly every recording", np.mean(orient["y"]) > 0.99, f"{np.mean(orient['y']):.4f}")
ck.internal("x decreases from tongue tip to tongue back to velum in nearly every recording (x grows towards the front)", np.mean(orient["x"]) > 0.99, f"{np.mean(orient['x']):.4f}")
print("recordings", len(df), "samples", n_tot, "min", n_tot / RATE / 60, "| htk period values", htk_period)
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)

meta = dict(
    paper=dict(authors="Bocquelet F, Hueber T, Girin L, Savariaux C, Yvert B", year=2016,
               title="Real-time control of an articulatory-based speech synthesizer for brain computer interfaces",
               venue="PLoS Comput Biol 12(11):e1005119", doi="10.1371/journal.pcbi.1005119"),
    lab="Yvert (INSERM Grenoble) / Hueber (GIPSA-lab)",
    data_url="https://doi.org/10.5281/zenodo.154083",
    license="CC BY-SA 4.0", license_url="https://creativecommons.org/licenses/by-sa/4.0/",
    license_text=z.read("BY2014/license.txt").decode("utf-8", "replace"),
    changes=("MODIFIED from the authors' files: the ASCII EMA files (ema_ascii/) were copied to CSV, one CSV per item in samples/, with a time_s "
             "column added (sample number divided by 100 Hz) and the 27 columns named by coil and axis; values are unchanged (6 significant digits). "
             "The transcription file is included as transcription.txt. The audio (wav/) and the HTK copy of the EMA data were not converted."),
    how=["One row is one item the speaker said (a sentence or a list of vowels; condition is the authors' text of the item, block gives the "
         "series number and item number of the file name by_phr_<series>_<item>_sync). samples_file holds the 27 EMA channels at 100 Hz: x, y and z "
         "of the right lip corner, left lip corner, lower lip, upper lip, jaw, tongue tip, tongue dorsum, tongue back and velum, in the order of the "
         "authors' readme.",
         "The deposit states no unit. Positions are labelled mm because the paper reports mm-sized errors for these coils (calibration error about "
         "2.5 mm) and the values fit it (lip-to-lip distance about 10, tongue tip to tongue back about 30 units).",
         "Axes are not defined in the deposit and the paper's captions disagree. In the data x grows towards the front of the mouth (the tongue tip "
         "lies at a larger x than the tongue back and the velum in nearly every item) and y is upward (the upper lip lies above the lower lip). The "
         "sign of z was not inferred.",
         "trial is the item order of the file names. Age is not given; the speaker is one male native French speaker."],
    unit_note="EMA recording (one item said by one speaker)",
    differences=[
        "The paper says the reference database had 712 items of variable length and 'more than 45 minutes of speech after removing the periods of silence'; "
        f"the deposit has 672 EMA items with {n_tot / RATE / 60:.1f} minutes of recording including silences, so 40 items of the paper's database are not in "
        "the deposit and the duration claim cannot be checked.",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)
shutil.rmtree(TMP, ignore_errors=True)
