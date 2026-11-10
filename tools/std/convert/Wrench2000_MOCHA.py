"""Wrench & Hardcastle (2000), 5th Seminar on Speech Production, Kloster Seeon: MOCHA-TIMIT, a multichannel articulatory database of
English speakers reading the 460 British-TIMIT sentences (electromagnetic articulography, electropalatography, laryngograph, audio).
This converter standardizes the electromagnetic articulography (EMA, Carstens articulograph, 10 coils at 500 Hz) of the three speakers
in the main folder of the deposit, with their phone labels; the audio, laryngograph and electropalatograph files and the seven speakers
of the deposit's unchecked/ folder (no readme, not verified by the authors) are not converted.

Raw files, raw_extra/Wrench2000_MOCHA/ (https://data.cstr.ed.ac.uk/mocha/):
  fsew0_v1.1.tar.gz (female), msak0_v1.1.tar.gz (male), maps0.tar.gz: per sentence <speaker>_NNN.ema (EST track file: ASCII header with
      the channel names, then for every sample 4-byte little-endian floats: time in s, a presence flag that is always 1, and the 20
      channels), .lab (phone labels: start, stop in s, label; fsew0 and msak0 only), .wav, .lar, .epg (not converted); maps0_palate.ema
  README_v1.2.txt, README.fsew0.v1.1, README.msak0, README.maps0, LICENCE.txt, mocha-timit.txt (the 460 sentences)
The files state no unit for the EMA positions. Run from the site folder: python3 -I tools/std/convert/Wrench2000_MOCHA.py
"""
import re
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wrench2000_MOCHA"
SRC = "Wrench and Hardcastle 2000, 5th Seminar on Speech Production"
raw = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "scratch" / f"{ID}_samples"
DEVICE = ("Carstens articulograph (10 coils, 500 Hz) in a sound-damped studio; microphone, laryngograph and Reading electropalatograph "
          "were recorded at the same time (not converted)")
KIND = ("EMA coil positions (x, y) of the upper incisor, lower incisor, upper lip, lower lip, tongue tip, tongue blade, tongue dorsum, "
        "velum and the bridge of the nose while reading sentences")
UNIT = "0.01 mm"
# speaker, tarball, sex (README_v1.2.txt 2.2; the maps0 readme gives none), notes by sentence number from the readmes
SPEAKERS = [
    ("fsew0", "fsew0_v1.1.tar.gz", "F",
     "as released (v1.1): the authors removed the jump in the tongue-blade and velum coil positions after those coils were re-attached "
     "(readme); no other processing is described",
     {125: "velum coil re-attached before this sentence", 284: "middle tongue (tongue blade) coil re-attached before this sentence"}),
    ("msak0", "msak0_v1.1.tar.gz", "M",
     "as released (v1.1): the authors removed the jump in the tongue-dorsum and velum coil positions after those coils were re-attached "
     "and compensated for sentences where coils were loose (readme); no other processing is described",
     {**{i: "rear tongue (dorsum) coil loose, corrected by the authors" for i in list(range(29, 40)) + list(range(291, 299))},
      40: "rear tongue coil re-attached", 118: "tongue tip coil signal corrupted, reason unknown (readme)", 251: "velar coil re-attached",
      299: "rear tongue coil re-attached", 426: "middle tongue coil re-attached"}),
    ("maps0", "maps0.tar.gz", None, "as released (no processing is described); the velum coil was not recorded for this speaker (readme)", {}),
]


def coil_of(ch):
    """Coil number (1-10) and axis of channel ch, in the order the readme gives: x of coils 1-5, y of coils 1-5, x of coils 6-10, y of coils 6-10."""
    return (ch % 5) + 1 + (5 if ch >= 10 else 0), "x" if (ch < 5 or 10 <= ch < 15) else "y"


def read_ema(b):
    i = b.find(b"EST_Header_End")
    hdr = b[:i + len("EST_Header_End")].decode("latin-1")
    body = b[i + len("EST_Header_End") + 1:]
    nfr = int(re.search(r"NumFrames (\d+)", hdr).group(1))
    nch = int(re.search(r"NumChannels (\d+)", hdr).group(1))
    assert re.search(r"ByteOrder (\S+)", hdr).group(1) == "01" and nch == 20
    names = dict((int(a), c) for a, c in re.findall(r"Channel_(\d+) (\S+)", hdr))
    names = [names[c] for c in range(nch)]
    need = nfr * (nch + 2) * 4
    assert need <= len(body) <= need + 1, (nfr, len(body), need)
    arr = np.frombuffer(body[:need], dtype="<f4").reshape(nfr, nch + 2)
    names = [f"unused_coil{coil_of(c)[0]}_{coil_of(c)[1]}" if n.startswith("*") else n for c, n in enumerate(names)]
    return names, arr


def read_lab(txt):
    rows = []
    for l in txt.splitlines():
        if l.strip():
            a, b_, lab = l.strip().split(None, 2)
            rows.append((a, b_, lab))
    return rows


sentences = {}
for l in (raw / "mocha-timit.txt").read_text(encoding="latin-1").splitlines():
    m = re.match(r"^(\d{3})\.\s+(.*\S)\s*$", l)
    if m:
        sentences[int(m.group(1))] = m.group(2)
assert sorted(sentences) == list(range(1, 461)), len(sentences)

if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
rows, extra = [], []
rates, lab_past_end, n_lab, bad = [], 0, 0, []
lab_gap, lagdev = [], {}
orient = {"lip_y": [], "tip_x": [], "order": []}
ui_abs, flags = [], set()
chan_count = {}
for si, (spk, tarball, sex, proc, notes) in enumerate(SPEAKERS, start=1):
    pcode = f"P{si:04d}"
    ema, lab, palate = {}, {}, None
    with tarfile.open(raw / tarball, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            mm = re.match(rf"^{spk}_(\d{{3}})\.(ema|lab)$", m.name)
            if mm:
                data = tf.extractfile(m).read()
                (ema if mm.group(2) == "ema" else lab)[int(mm.group(1))] = data
            elif m.name == f"{spk}_palate.ema":
                palate = tf.extractfile(m).read()
    nums = sorted(ema)
    k = 0
    items = [(n, ema[n]) for n in nums] + ([("palate", palate)] if palate is not None else [])
    for n, b in items:
        names, arr = read_ema(b)
        ix = {nm: i for i, nm in enumerate(names)}
        t, flag, v = arr[:, 0], arr[:, 1], arr[:, 2:]
        flags |= set(np.unique(flag).tolist())
        bad.append(int(np.isnan(arr).sum()))
        nfr = len(t)
        rate = (nfr - 1) / float(t[-1] - t[0])
        rates.append(rate)
        k += 1
        arc = f"samples/{pcode}_rec{k:03d}.csv"
        with open(TMP / f"{pcode}_rec{k:03d}.csv", "w", newline="") as fh:
            fh.write("time_s," + ",".join(names) + "\r\n")
            np.savetxt(fh, np.column_stack([t, v]), fmt="%.7g", delimiter=",", newline="\r\n")
        extra.append((arc, TMP / f"{pcode}_rec{k:03d}.csv"))
        if n != "palate" and n in lab:
            al = read_lab(lab[n].decode("latin-1"))
            n_lab += 1
            lab_past_end += int(float(al[-1][1]) > nfr / rate + 0.05)
            lab_gap.append(float(al[-1][1]) - nfr / rate)
            ap = v[:, ix["ul_y"]] - v[:, ix["ll_y"]]  # lip aperture: smallest in bilabial closures when the labels fit the EMA time axis
            for lag in (-0.12, 0.0, 0.12):
                vals = []
                for a, b_, lab_ in al:
                    if lab_ in ("p", "b", "m"):
                        s0, e0 = int(round((float(a) + lag) / 0.002)), int(round((float(b_) + lag) / 0.002))
                        if 0 <= s0 < e0 <= nfr:
                            vals.append(ap[s0:e0].mean() - ap.mean())
                if vals:
                    lagdev.setdefault((spk, lag), []).append(float(np.mean(vals)))
            txt = "start_s,end_s,label\r\n" + "".join(f"{a},{b_},{c}\r\n" for a, b_, c in al)
            extra.append((f"alignments/{pcode}_rec{k:03d}.csv", txt.encode("utf-8")))
        if n != "palate":
            mean = v.mean(0)
            orient["lip_y"].append(mean[ix["ul_y"]] > mean[ix["ll_y"]])
            orient["tip_x"].append(mean[ix["tt_x"]] > mean[ix["ul_x"]])
            if "v_x" in ix:
                orient["order"].append(mean[ix["tt_x"]] < mean[ix["tb_x"]] < mean[ix["td_x"]] < mean[ix["v_x"]])
            ui_abs.append(float(np.abs(mean[[ix["ui_x"], ix["ui_y"]]]).max()))
            used = [nm for nm in names if not nm.startswith("unused")]
            chan_count[spk] = max(chan_count.get(spk, 0), len([nm for nm in used if nm.split("_")[0] in ("ul", "ll", "li", "tt", "tb", "td", "v")]))
        note = notes.get(n) if n != "palate" else None
        p = proc + (f"; authors' note for this sentence: {note}" if note else "")
        if n == "palate":
            cond, block = "palate trace (a coil traced along the palate; the lower-lip channels hold it as palx and paly)", "palate trace"
            p = "as released: the authors' palate trace (readme: palate trace on channel 4 of the palate file)"
        else:
            cond, block = sentences[n], f"sentence {n:03d}"
        rows.append(dict(
            dataset=ID, study=SRC, experiment="", group="", subject=spk, person=spk, subject_excluded_by_authors=False, age=None,
            sex=sex, handedness=None, population="not stated", setting="lab", device=DEVICE,
            session=1, block=block, trial=k, phase="other", time_s=None, task="speech", unit="recording", condition=cond,
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            recording=k, samples_file=arc, signal_kind=KIND, channels=";".join(names), channel_units=";".join([UNIT] * len(names)),
            sample_rate_hz=round(rate, 6), n_samples=nfr, duration_s=nfr / rate, events="", processing=p))
extra.append(("sentences.txt", (raw / "mocha-timit.txt").read_bytes()))
df = pd.DataFrame(rows)

# ---- checks -------------------------------------------------------------------------------------------------------------
sent = df[df["block"].str.startswith("sentence")]
ck = stdlib.Checks()
ck.published("sentences read by the female speaker", int((sent["subject"] == "fsew0").sum()), 460, 0,
             "Abstract (each speaker reads up to 460 TIMIT sentences) and Wrench & Richmond 2000, Sec. 2.2 (all 460 sentences of the two analysed speakers)")
ck.published("sentences read by the male speaker", int((sent["subject"] == "msak0").sum()), 460, 0, "Abstract (up to 460 sentences); Wrench & Richmond 2000, Sec. 2.2")
ck.published("speakers with a documented sex: male", int(df.loc[df["sex"] == "M", "person"].nunique()), 1, 0, "Wrench & Richmond 2000, Sec. 2.1 (two speakers, one male and one female, were analysed)")
ck.published("speakers with a documented sex: female", int(df.loc[df["sex"] == "F", "person"].nunique()), 1, 0, "Wrench & Richmond 2000, Sec. 2.1")
ck.published("EMA sampling rate (Hz)", float(np.mean(rates)), 500, 0.5, "Wrench & Richmond 2000, Sec. 3.1 (EMA sampled at 500 Hz)")
ck.published("EMA coefficients of the articulators (x and y of upper lip, lower lip, jaw, 3 tongue sensors, velum), female speaker", chan_count["fsew0"], 14, 0,
             "Wrench & Richmond 2000, Sec. 3.1 (14 coefficients in total)")
ck.published("EMA coefficients of the articulators, male speaker", chan_count["msak0"], 14, 0, "Wrench & Richmond 2000, Sec. 3.1")
ck.internal("20 channels (coils 1-10, x and y) in every recording, every time step 2 ms, presence flag always 1, no missing values",
            flags == {1.0} and sum(bad) == 0 and all(abs(r - 500) < 0.5 for r in rates), f"flags {flags}, nan {sum(bad)}")
ck.internal("recordings: 460 sentences for the female and the male speaker, 459 for the third (sentence 93 is missing, as the deposit's readme for that speaker says) plus its palate trace",
            [int((sent["subject"] == s).sum()) for s in ("fsew0", "msak0", "maps0")] == [460, 460, 459] and (df["block"] == "palate trace").sum() == 1,
            f"{len(df)} recordings")
ck.internal("axes: upper lip above lower lip in nearly every recording (y up)", np.mean(orient["lip_y"]) > 0.99, f"{np.mean(orient['lip_y']):.4f}")
ck.internal("axes: tongue-tip coil further along x than the upper-lip coil in nearly every recording (x grows towards the back)",
            np.mean(orient["tip_x"]) > 0.99, f"{np.mean(orient['tip_x']):.4f}")
ck.internal("axes: mean x increases from tongue tip to blade to dorsum to velum in most recordings", np.mean(orient["order"]) > 0.9, f"{np.mean(orient['order']):.4f}")
ck.internal("the upper-incisor coil sits at the origin in every sentence (mean |x|, |y| below 1 mm)", max(ui_abs) < 100, f"largest {max(ui_abs):.1f} units")
dev = {k: float(np.mean(v)) for k, v in lagdev.items()}
ck.internal("labels fit the EMA time axis: in labelled bilabial closures (p, b, m) the lip aperture is below the sentence mean without a shift and "
            "above it when the labels are shifted by 0.12 s either way (both labelled speakers)",
            all(dev[(sp, 0.0)] < 0 < min(dev[(sp, -0.12)], dev[(sp, 0.12)]) for sp in ("fsew0", "msak0")),
            "; ".join(f"{name}: {dev[(sp, -0.12)] / 100:+.1f}, {dev[(sp, 0.0)] / 100:+.1f}, {dev[(sp, 0.12)] / 100:+.1f} mm" for sp, name in (("fsew0", "female"), ("msak0", "male"))))
ck.internal("every sentence has a sample file; the female and the male speaker have a label file for each sentence", n_lab == 920, f"{n_lab} label files")
print("recordings", len(df), "| label files", n_lab, "| rate", round(min(rates), 4), round(max(rates), 4), "| labels ending after the EMA by more than 50 ms:", lab_past_end,
      "| ui offset max", round(max(ui_abs), 1), "| samples MB", sum(p.stat().st_size for a, p in extra if isinstance(p, Path)) // 1_000_000)
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)

meta = dict(
    paper=dict(authors="Wrench AA, Hardcastle WJ", year=2000,
               title="A multichannel articulatory speech database and its application for automatic speech recognition",
               venue="Proc 5th Seminar on Speech Production: Models and Data, Kloster Seeon, pp 305-308", doi=""),
    lab="Wrench (Queen Margaret University College, Edinburgh) / CSTR",
    data_url="https://data.cstr.ed.ac.uk/mocha/",
    license=("under the MOCHA licence of Queen Margaret University College (1999; use, copy, modify and distribute for research, educational "
             "and individual use only, no commercial use without prior written permission)"),
    license_url="https://data.cstr.ed.ac.uk/mocha/LICENCE.txt",
    license_text=(raw / "LICENCE.txt").read_bytes().decode("latin-1"),
    changes=("MODIFIED from the authors' files: only the EMA files of the three speakers in the main folder of the deposit (a female speaker, a male speaker and a third whose sex is not given) were "
             "converted from the authors' binary track files to CSV (one CSV per sentence in samples/, copied to 7 significant digits, the "
             "presence flag that is always 1 left out, unused coil slots renamed unused_coil...), with the authors' phone labels of "
             "the female and the male speaker as alignments/ CSVs and their sentence list as sentences.txt. The audio, laryngograph and electropalatograph "
             "files, the HTK label file (the same labels in another unit) and the seven speakers of the unchecked/ folder were not "
             "converted. The original licence is in LICENSE.txt and the authors' names are kept."),
    how=["One row is one recording of the Carstens articulograph (EMA) while one speaker read one of the 460 MOCHA-TIMIT sentences "
         "(block is the sentence number, condition its text); the third speaker's palate trace is one more row. samples_file holds the 20 channels "
         "of the file at 500 Hz, with the channel names of the file's header: x and y of the coils on the upper incisor (ui), lower incisor "
         "(li, the jaw), upper lip (ul), lower lip (ll), tongue tip (tt), tongue blade (tb), tongue dorsum (td), velum (v) and the bridge "
         "of the nose (bn); slots without a coil are called unused_coil...",
         "The files state no unit. Positions are labelled 0.01 mm because the sensor distances are those of a human vocal tract only in "
         "hundredths of a millimetre (for example the upper and lower lip coils are about 25 mm apart for the female speaker and 28 mm for the male speaker). "
         "In nearly every sentence x grows towards the back of the mouth (the lips lie at negative x, the velum at the largest x) and y upwards, "
         "with the upper-incisor coil at (0, 0).",
         "alignments/ has the authors' phone labels (start and end in seconds, label) for each sentence of the female and the male speaker; the "
         "authors' readme says about 6% of the transcriptions in this release are wrong. The third speaker has no labels and no velum coil, and "
         "its sentence 93 is missing. Notes of the authors' readmes about single sentences (coil re-attached, loose or corrupted) are in "
         "processing.",
         "trial is the order of the recordings of a speaker (sentence order, so trial 93 of the third speaker is sentence 94). Age is not given; sex "
         "is from the deposit's readme for the female and the male speaker."],
    unit_note="EMA recording (one speaker reading one sentence; the third speaker's palate trace is one more)",
    differences=[
        f"In {lab_past_end} of the {n_lab} label files the last label ends more than 50 ms after the end of the EMA signal of the same sentence "
        f"(median {np.median(lab_gap):.2f} s later: the EMA recording is shorter than the audio the labels were made on). The label times fit the "
        f"EMA time axis: in the labelled bilabial closures (p, b, m) the lip aperture is {abs(dev[('fsew0', 0.0)]) / 100:.1f} mm (female speaker) and "
        f"{abs(dev[('msak0', 0.0)]) / 100:.1f} mm (male speaker) below the sentence mean when the labels are used as given, and above the mean when they are "
        "shifted by 0.12 s either way; the label times are copied as given.",
        "The deposit's readme says EMA files hold 'x-values of coils 1-5, y-values of coils 1-5, x-values of coils 6-10, y-values of coils "
        "6-10' without naming the coils; coil names are those of each file's header. The listed paper could not be read: the "
        "numbers checked above are from its abstract and from Wrench & Richmond (2000, 'Continuous speech recognition using articulatory data', "
        "CSTR, Edinburgh), whose authors describe the same database.",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)
shutil.rmtree(TMP, ignore_errors=True)
