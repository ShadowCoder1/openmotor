"""Badin et al., PB2007 acoustic-articulatory speech dataset (Zenodo 6390598), used by Ben Youssef, Badin, Bailly & Heracleous (2009), Interspeech
2009:2255-2258: electromagnetic articulography (EMA, 6 coils in the midsagittal plane, 100 Hz) of one male French speaker saying 1109 items
(nonsense VCV sequences, French word pairs and sentences, per the paper), with a phonetic segmentation of each item.

Raw files, raw_extra/Badin_PB2007_Zenodo/ (https://doi.org/10.5281/zenodo.6390598):
  PB2007.zip: _seq/item_NNNN.seq (text: a header line with 12 parameter names PAR01-PAR12, then one line per 10 ms frame: '#' and 12 values),
      _lab/item_NNNN.lab (phone segmentation: start, end in units of 100 ns, label), _wav16/ (audio, not converted)
  README (data description and licence)
Run from the site folder: python3 -I tools/std/convert/Badin_PB2007_Zenodo.py
"""
import collections
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Badin_PB2007_Zenodo"
SRC = "Ben Youssef et al. 2009, Interspeech 2009 (PB2007 dataset)"
raw = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "scratch" / f"{ID}_samples"
RATE = 100.0
# README: PAR01-PAR12 = x of lower incisor (jaw), tongue tip, tongue dorsum, tongue back, lower lip, upper lip; then z of the same six
NAMES = ["jaw_x", "tongue_tip_x", "tongue_dorsum_x", "tongue_back_x", "lower_lip_x", "upper_lip_x",
         "jaw_z", "tongue_tip_z", "tongue_dorsum_z", "tongue_back_z", "lower_lip_z", "upper_lip_z"]
UNIT = "cm (inferred)"
DEVICE = "electromagnetic articulograph (6 coils: lower incisors, tongue tip, tongue middle, tongue back, upper lip, lower lip), audio recorded at the same time (not converted)"
KIND = "EMA coil positions (x and z in the midsagittal plane) of the jaw (lower incisors), tongue tip, tongue dorsum, tongue back, lower lip and upper lip while one speaker said an item"

z = zipfile.ZipFile(next(raw.glob("*.zip")))
names = sorted(n for n in z.namelist() if not n.startswith("__MACOSX"))
seqs = [n for n in names if n.startswith("_seq/") and n.endswith(".seq")]
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)

rows, extra = [], []
labcount = collections.Counter()
n_lab_files, lab_gap, n_tot, bad_fmt = 0, [], 0, 0
orient = {"x": [], "z": [], "z2": []}
ranges = []
for k, n in enumerate(seqs, start=1):
    stem = n.split("/")[-1][:-4]
    L = z.read(n).decode("latin-1").splitlines()
    hdr = L[0].split()
    assert hdr[0] == "12" and hdr[1:] == [f"PAR{i:02d}" for i in range(1, 13)], hdr
    body = [l.split() for l in L[1:] if l.strip()]
    if not all(len(b) == 13 and b[0] == "#" for b in body):
        bad_fmt += 1
    d = np.array([[float(v) for v in b[1:]] for b in body])
    assert d.shape[1] == 12
    n_tot += len(d)
    ranges.append((d.min(), d.max()))
    t = np.arange(len(d)) / RATE
    arc = f"samples/P0001_rec{k:04d}.csv"
    with open(TMP / f"P0001_rec{k:04d}.csv", "w", newline="") as fh:
        fh.write("time_s," + ",".join(NAMES) + "\r\n")
        np.savetxt(fh, np.column_stack([t, d]), fmt=["%.2f"] + ["%.6g"] * 12, delimiter=",", newline="\r\n")
    extra.append((arc, TMP / f"P0001_rec{k:04d}.csv"))
    lab = [l.split(None, 2) for l in z.read(f"_lab/{stem}.lab").decode("latin-1").splitlines() if l.strip()]
    n_lab_files += 1
    txt = "start_s,end_s,label\r\n"
    for a, b, c in lab:
        labcount[c.strip()] += 1
        txt += f"{int(a) / 1e7:.4f},{int(b) / 1e7:.4f},{c.strip()}\r\n"
    lab_gap.append(int(lab[-1][1]) / 1e5 - len(d))
    extra.append((f"alignments/P0001_rec{k:04d}.csv", txt.encode("utf-8")))
    m = d.mean(0)
    orient["x"].append(m[1] < m[2] < m[3])  # tip, dorsum, back: x grows towards the back
    orient["z"].append(m[10] > m[11])        # lower lip z greater than upper lip z
    orient["z2"].append(m[6] < m[10])        # jaw z smaller than lower lip z
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group="", subject="PB", person="PB", subject_excluded_by_authors=False, age=None,
        sex="M", handedness=None, population="healthy adult, native French speaker", setting="lab", device=DEVICE,
        session=1, block=stem.replace("item_", "item "), trial=k, phase="other", time_s=None, task="speech", unit="recording",
        condition="", pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=k, samples_file=arc, signal_kind=KIND, channels=";".join(NAMES), channel_units=";".join([UNIT] * 12),
        sample_rate_hz=RATE, n_samples=len(d), duration_s=len(d) / RATE, events="", trial_info="",
        processing="as released (the paper says the 500 Hz recording was down-sampled to 100 Hz)"))
extra.append(("README_authors.txt", z.read("README")))
df = pd.DataFrame(rows)

n_phone = sum(labcount.values())
a_count = labcount["a"]
ck = stdlib.Checks()
ck.published("items (utterances) of the corpus: 740 for training plus 369 for testing", len(df), 740 + 369, 0,
             "Ben Youssef et al. 2009, Sec. 4 (evaluation: two thirds, 740 utterances, for training and the remaining third, 369 utterances, for testing)")
ck.published("coils recorded (jaw, tongue tip, middle, back, upper lip, lower lip)", 6, 6, 0, "Ben Youssef et al. 2009, Sec. 3.2 (six coils are used)")
ck.published("speakers", int(df["person"].nunique()), 1, 0, "Ben Youssef et al. 2009, Sec. 3.1 (a single male French subject)")
ck.published("instances of the most frequent phone, /a/", a_count, 348, 0, "Ben Youssef et al. 2009, Sec. 3.3 (maximum number of instances per phoneme: 348, for /a/)")
ck.published("distinct segment labels (36 phonemes including the two pause labels)", len(labcount), 36, 0, "Ben Youssef et al. 2009, Sec. 3.1 (the 36 phonemes; _ and __ are the short and the long pause)")
ck.internal("/a/ is the most frequent non-pause label", max((c for c in labcount if c not in ("_", "__")), key=labcount.get) == "a", "")
ck.internal("every .seq line is '#' plus 12 values, a label file exists for each item, no missing values",
            bad_fmt == 0 and n_lab_files == len(seqs) == 1109, f"bad lines {bad_fmt}, {n_lab_files} label files")
ck.internal("the last label of an item ends within 3 frames of the end of its EMA signal", min(lab_gap) >= -0.5 and max(lab_gap) <= 3.5, f"{min(lab_gap):.2f} to {max(lab_gap):.2f} frames")
ck.internal("x increases from tongue tip to dorsum to back in nearly every item (x grows towards the back of the mouth)", np.mean(orient["x"]) > 0.95, f"{np.mean(orient['x']):.4f}")
ck.internal("lower-lip z larger than upper-lip z in nearly every item (z grows downwards)", np.mean(orient["z"]) > 0.95, f"{np.mean(orient['z']):.4f}")
ck.internal("jaw z smaller than lower-lip z in nearly every item", np.mean(orient["z2"]) > 0.95, f"{np.mean(orient['z2']):.4f}")
print("items", len(df), "samples", n_tot, "min", n_tot / RATE / 60, "| phones", n_phone, "| short pauses", labcount["_"], "| a", a_count, "| range", min(r[0] for r in ranges), max(r[1] for r in ranges))
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)

meta = dict(
    paper=dict(authors="Ben Youssef A, Badin P, Bailly G, Heracleous P", year=2009,
               title="Acoustic-to-articulatory inversion using speech recognition and trajectory formation based on phoneme hidden Markov models",
               venue="Interspeech 2009:2255-2258", doi="10.21437/interspeech.2009-640"),
    lab="GIPSA-lab (Grenoble)",
    data_url="https://doi.org/10.5281/zenodo.6390598",
    license="CC BY-SA (the licence is named without a version in the deposit's readme; the Zenodo record's metadata says CC BY 4.0)",
    license_url="https://zenodo.org/records/6390598",
    license_text=z.read("README").decode("utf-8", "replace"),
    changes=("MODIFIED from the authors' files: the .seq text files (12 parameters per 10 ms frame) were copied to CSV, one CSV per item in samples/, with a "
             "time_s column added (frame number divided by 100 Hz) and the parameters named by coil and axis; values are unchanged (6 significant digits). "
             "The authors' phone segmentation (.lab files, times in units of 100 ns) is in alignments/ as CSVs with times in seconds. The audio (_wav16/) was not "
             "converted. The authors' readme is included as README_authors.txt."),
    how=["One row is one item the speaker said (block is the authors' item number; the deposit gives no text of the items, which the paper describes as "
         "nonsense VCV sequences, French word pairs and sentences). samples_file holds the 12 EMA parameters at 100 Hz: the x and the z coordinate of the "
         "coils on the lower incisors (jaw), tongue tip, tongue dorsum (middle), tongue back, lower lip and upper lip, in the order of the authors' readme "
         "(PAR01-PAR12). alignments/ has the authors' phone labels for each item (start and end in seconds, label; __ and _ are long and short pauses).",
         "The deposit states no unit. Positions are labelled cm (inferred) because the paper puts the tongue coils 1.2, 4.2 and 7.3 cm from the tongue "
         "extremity (about 3 cm apart) and the values of x of tongue tip, dorsum and back are about 2 to 2.5 units apart.",
         "Axes are not defined in the deposit. In the data x grows towards the back of the mouth (tongue tip, dorsum and back at increasing x) and z "
         "grows downwards (the lower lip lies at a larger z than the upper lip), so both signs are inferred.",
         "The paper says the EMA was recorded at 500 Hz and down-sampled to 100 Hz; the readme gives 100 Hz, which is what the files hold. Age is not given."],
    unit_note="EMA recording (one item said by one speaker)",
    differences=[
        f"The paper counts 7352 phones and at least 18 short pauses ('minimum ... 18 for short pauses'); the label files hold {n_phone} labels in all "
        f"(pauses included) and {labcount['_']} short pauses (_), so these two counts do not reproduce and are not used as checks. The paper says about 12 minutes of speech; "
        f"the EMA signals are {n_tot / RATE / 60:.1f} minutes in all, silences included.",
        "The licence is CC BY-SA in the deposit's readme and CC BY 4.0 in the Zenodo record's metadata (and source/extra_studies.json lists CC BY-SA 4.0).",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)
shutil.rmtree(TMP, ignore_errors=True)
