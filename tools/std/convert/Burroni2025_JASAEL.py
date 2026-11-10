"""Burroni, Kawahara & Shaw (2025), JASA Express Letters 5(1):015201: tongue tip articulography of Tokyo Japanese mimetic words with
singleton and geminate consonants; 7 speakers (3 on an NDI Wave at 100 Hz, 4 on a Carstens AG501 at 1250 Hz), 21 words, 10 repetitions.

Raw file, raw_extra/Burroni2025_JASAEL/Combined.tar.gz (OSF 27nyz, 342 MB). Only Combined/Data/TNew.mat is read (the analysis table of
both systems pooled, 2880 tokens); the other members are the authors' MATLAB scripts, audio, figures and model logs. The MATLAB table is
decoded from the MAT-file's own object store with scipy (helper Burroni2025_JASAEL_mat.py); nothing from the deposit is executed.
Definitions: paper (JASA-EL, CC BY, institutional copy), Methods 2.1-2.4, and the authors' scripts and Models logs (read as text).
Not carried: the authors' COns, CTarg, CRel and COff columns (recording-time landmarks): in the AG501 rows they hold values that
cannot be times (negative, up to -19), so none is used; durations derived from the landmarks (timeToTarg, plateau, relDur) are carried.
Run from the site folder: python3 -I tools/std/convert/Burroni2025_JASAEL.py
"""
import csv
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Burroni2025_JASAEL_mat import MatTables  # noqa: E402

ID = "Burroni2025_JASAEL"
SRC = "Burroni et al. 2025 (JASA Express Letters 5(1):015201)"
DEV = {"ndi": "NDI Wave electromagnetic articulograph (100 Hz)", "ag": "Carstens AG501 electromagnetic articulograph (1250 Hz)"}


def load():
    raw = stdlib.raw_dir(ID)
    t = tarfile.open(raw / "Combined.tar.gz")
    for m in t:
        if m.name == "Combined/Data/TNew.mat":  # one named member only
            (raw / "TNew.mat").write_bytes(t.extractfile(m).read())
            break
    T = MatTables(raw / "TNew.mat").table("TNew")
    assert T["_nrows"] == 2880
    return T


def f1(T, k):
    return np.asarray(T[k], dtype=float).ravel()


def build(T, checks):
    n = T["_nrows"]
    sp = list(T["SP"])
    ndi = [s.startswith("subject_") for s in sp]
    checks.internal("the first 1260 rows are the three NDI Wave speakers and the rest the four AG501 speakers", all(ndi[:1260]) and not any(ndi[1260:]))
    assert all(ndi[:1260]) and not any(ndi[1260:])
    gem = list(T["geminate"])
    assert gem == list(T["Geminate"]) and all(gem[i] == ("Singleton" if i % 2 == 0 else "Geminate") for i in range(n))
    seq, rows = {}, []
    ms = lambda k: f1(T, k) * 1000  # noqa: E731
    clo, pla, rel, cdur, v1, vco, vct, vv = (ms(k) for k in ("timeToTarg", "plateau", "relDur", "CDur", "V1Dur", "VCOns", "VCTarg", "V1V2"))
    for i in range(n):
        seq[sp[i]] = seq.get(sp[i], 0) + 1
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Tokyo Japanese mimetic words, singleton and geminate consonants", "group": "",
            "subject": sp[i], "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEV["ndi" if ndi[i] else "ag"], "session": 1, "block": "", "trial": seq[sp[i]], "phase": "test",
            "time_s": "", "task": "speech", "unit": "utterance", "condition": gem[i].lower(), "pert_type": "none", "pert_value": 0,
            "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
            "word_pair": T["Pair"][i], "preceding_vowel": T["V1"][i], "gesture_signal": "TT_Z", "trajectory_id": i + 1,
            "closing_duration_ms": clo[i], "plateau_duration_ms": pla[i], "opening_duration_ms": rel[i],
            "target_consonant_duration_ms": cdur[i], "preceding_vowel_duration_ms": v1[i], "v1_to_clo_onset_ms": vco[i],
            "v1_to_clo_target_ms": vct[i], "v1_v2_lag_ms": vv[i],
            "constriction_max_mm": f1(T, "TTMax")[i], "horizontal_max_mm": f1(T, "TTXMax")[i], "closing_amplitude_mm": f1(T, "TTAmp")[i],
            "closing_peak_vel_cm_s": f1(T, "CloPeakVel")[i], "closing_stiffness_per_s": f1(T, "stiff")[i],
        })
    return rows, ndi


def write_traces(T, ndi, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["trajectory_id", "sample", "time_s", "tongue_tip_vertical_mm"])
        for i in range(T["_nrows"]):
            fs = 100.0 if ndi[i] else 1250.0
            for s, v in enumerate(np.asarray(T["TTTraj"][i], dtype=float).ravel()):
                w.writerow([i + 1, s, format(s / fs, ".8g"), format(float(v), ".8g")])


def trim(a, b):
    """The authors' supplementary figures drop values below the 1st or above the 99th percentile of either variable."""
    lo = lambda x: (x < np.percentile(x, 1)) | (x > np.percentile(x, 99))  # noqa: E731
    k = ~(lo(a) | lo(b))
    return a[k], b[k]


def checks_all(T, rows, ndi, checks):
    a = pd.DataFrame(rows)
    checks.published("tokens analysed (21 words x 2 x 10 repetitions x 7 speakers = 2940, minus 60 excluded)", len(a), 2880, 0, f"{SRC}, Methods 2.1 and 2.3")
    checks.published("speakers", a.subject.nunique(), 7, 0, f"{SRC}, Methods 2.1")
    checks.published("excluded tokens (30 pairs)", 2940 - len(a), 60, 0, f"{SRC}, Methods 2.3")
    checks.published("words", a.word_pair.nunique(), 21, 0, f"{SRC}, Methods 2.1")
    checks.published("speakers recorded on the NDI Wave (S1-S3)", len({s for s, d in zip(a.subject, ndi) if d}), 3, 0, f"{SRC}, Methods 2.2")
    checks.published("speakers recorded on the Carstens AG501 (S4-S7)", len({s for s, d in zip(a.subject, ndi) if not d}), 4, 0, f"{SRC}, Methods 2.2")
    x, y = trim(a.plateau_duration_ms.to_numpy(), a.target_consonant_duration_ms.to_numpy())
    checks.published("Pearson correlation of target consonant acoustic duration and plateau duration", float(pearsonr(x, y)[0]), 0.84, 0.005,
                     f"{SRC}, Results 3.1 (r = 0.84, values outside the 1st-99th percentile dropped, Methods 2.4)")
    x, y = trim(a.v1_to_clo_target_ms.to_numpy(), a.preceding_vowel_duration_ms.to_numpy())
    checks.published("Pearson correlation of time to target (V1 onset to consonant target) and preceding vowel duration", float(pearsonr(x, y)[0]), 0.70, 0.005,
                     f"{SRC}, Results 3.3 (r = 0.70)")
    c = a.condition.value_counts()
    checks.internal("equal numbers of singleton and geminate tokens", c["singleton"] == c["geminate"] == 1440)
    n_neg = int((a.plateau_duration_ms < 0).sum())
    checks.internal("closing and opening durations are not negative; the authors' own negative plateaus (target after release) are few and kept",
                    bool((a.closing_duration_ms >= 0).all() and (a.opening_duration_ms >= 0).all() and 0 < n_neg < 40
                         and (a.closing_amplitude_mm < 0).sum() <= 5), f"{n_neg} negative plateau durations")


def main():
    checks = stdlib.Checks()
    T = load()
    rows, ndi = build(T, checks)
    checks_all(T, rows, ndi, checks)
    for c in checks.items:
        print(("ok     " if c["ok"] else "FAILED "), c["what"], c.get("computed", ""), c.get("reported", ""))
    tpath = stdlib.raw_dir(ID) / "token_traces.csv"
    write_traces(T, ndi, tpath)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Decoded the authors' pooled analysis table (TNew.mat), converted durations and lags from seconds to milliseconds, wrote each "
                 "token's tongue tip vertical trace to samples/token_traces.csv with time from the sample number and the system's rate, and "
                 "replaced speaker labels with codes. The audio, the other sensors, the gesture-only traces and the authors' recording-time "
                 "landmark columns (COns, CTarg, CRel, COff) were left out."),
        how=[
            "One row is one spoken token of a Tokyo Japanese mimetic word (21 words, each said as a singleton and as an emphatic geminate "
            "in isolation, ten times) by one of seven speakers: 2,940 tokens recorded, 60 excluded by the authors (30 singleton/geminate "
            "pairs without clear gestural boundaries), 2,880 shared (1,440 each). The authors' order is kept: a singleton row, then its "
            "geminate partner. Speakers 1-3 were recorded with an NDI Wave (100 Hz) and the other four with a Carstens AG501 (1250 Hz); device says which.",
            "The consonantal gesture is the closing and release of the tongue tip, landmarked on its vertical movement (gesture_signal TT_Z). "
            "closing_duration_ms, plateau_duration_ms and opening_duration_ms are the lags onset-target, target-release and release-offset "
            "(20 per cent of peak velocity thresholds). constriction_max_mm is the highest vertical tongue tip position, closing_amplitude_mm the "
            "closing movement amplitude, closing_peak_vel_cm_s its peak velocity and closing_stiffness_per_s peak velocity (mm/s) over amplitude (mm).",
            "target_consonant_duration_ms, preceding_vowel_duration_ms and the v1_to_clo_*, v1_v2_lag_ms lags use the acoustic onsets of the "
            "preceding vowel, the consonant and the following vowel from the authors' manual segmentation. preceding_vowel is the authors' vowel label (the "
            "suffix del marks a vowel they flagged as deleted). word_pair is the authors' ASCII spelling of the word.",
            "samples/token_traces.csv holds each token's tongue tip vertical position (mm) over the authors' analysis window, keyed by "
            "trajectory_id (the row number); time_s starts at 0 at the first sample, at 100 Hz (NDI Wave speakers) or 1250 Hz (AG501 speakers). The "
            "position of the window relative to the word is not stated in the deposit. Sex and age are not given per speaker (the paper reports 3 female and 4 male).",
        ],
        unit_note="spoken token (one gesture)",
        differences=[
            "The paper's mixed-effects results (MATLAB fitlme with by-speaker and by-word slopes) were not refitted; the checks are counts and two "
            "Pearson correlations (r = 0.84 and 0.70 as printed) computed after the 1st-99th percentile trimming the paper describes.",
            "17 tokens have a negative plateau duration (a target landmark after the release landmark; 33 are not positive, 16 of them exactly 0) and 2 have a "
            "negative movement amplitude; they are the authors' values and are kept.",
            "The OSF project states no license (the article itself is CC BY 4.0). The authors' COns, CTarg, CRel and COff columns hold implausible "
            "values for the four AG501 speakers (negative, up to -19) and are not carried. The paper says tongue tip height was measured; the "
            "deposit's TT_Z trace was identical to the vertical component of the tongue tip in all AG501 rows.",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    return stdlib.release(ID, "articulography", rows, meta, checks, extra_files=[("samples/token_traces.csv", str(tpath))])


if __name__ == "__main__":
    main()
