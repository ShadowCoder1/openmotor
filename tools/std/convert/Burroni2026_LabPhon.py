"""Burroni, Bucar Shigemori, Kawahara & Shaw (2026), Laboratory Phonology 17(1):26525: electromagnetic articulography of Tokyo
Japanese mimetic words with a singleton or a geminate bilabial, alveolar, palatal or dorsal consonant; 6 speakers, 15 words, 10 repetitions,
Carstens AG501 at 1250 Hz.

Raw file, raw_extra/Burroni2026_LabPhon/SupplementaryMaterial.zip (OSF 4tk8q, 188 MB). Only SupplementaryMaterial/TableAll/TAllClean.mat is read
(the other members are the authors' MATLAB scripts, figures and model logs). The MATLAB table is decoded from the MAT-file's own object store with
scipy (helper Burroni2026_LabPhon_mat.py, adapted from the Italian-length helper); nothing from the deposit is executed.
Definitions: the paper (OSF preprint 6p4qy PDF, read as text), Methods 2.1-2.4, and the authors' SupplementaryMaterial.m (read as text).
The table has one row per token (1758 = 1800 - 42 excluded). The 5 trajectory columns are stored only on the singleton row of each singleton/geminate
pair, because both words were said in one trial; they are written once per trial to samples/trial_traces.csv.
Run from the site folder: python3 -I tools/std/convert/Burroni2026_LabPhon.py
"""
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Burroni2026_LabPhon_mat import MatTables  # noqa: E402

ID = "Burroni2026_LabPhon"
SRC = "Burroni et al. 2026 (Laboratory Phonology 17(1):26525)"
FS = 1250.0
DEVICE = "Carstens AG501 electromagnetic articulograph (1250 Hz), Sennheiser ME66 microphone"
SIGNAL = {"Bilabial": "LA", "Apical": "TT_Z", "Laminal": "TB_Z", "Dorsal": "TD_Z"}


def sc(v):
    v = np.asarray(v)
    while v.dtype == object:
        v = np.asarray(v.ravel()[0])
    return float(v.ravel()[0])


def col(T, n):
    c = T[n]
    if isinstance(c, np.ndarray) and c.dtype != object:
        return c.ravel()
    return c


def load(checks):
    raw = stdlib.raw_dir(ID)
    z = zipfile.ZipFile(raw / "SupplementaryMaterial.zip")
    (raw / "TAllClean.mat").write_bytes(z.read("SupplementaryMaterial/TableAll/TAllClean.mat"))  # one named member only
    T = MatTables(raw / "TAllClean.mat").table("T")
    assert T["_nrows"] == 1758
    return T


def build(T, checks):
    n = T["_nrows"]
    sp = [str(np.asarray(x).ravel()[0]) for x in T["SP"]]
    word = [str(np.asarray(x).ravel()[0]) for x in T["wordLabel"]]
    lm = T["lm"]
    traj = {k: T[k] for k in ("traj", "trajTMz", "trajTMy", "trajTDz", "trajTDy")}
    # pairs: rows 2k (singleton, holds the traces) and 2k+1 (geminate)
    for k in range(0, n, 2):
        a, b = k, k + 1
        assert (np.asarray(traj["traj"][a]).size > 0 and np.asarray(traj["traj"][b]).size == 0 and sp[a] == sp[b]
                and T["trialN"][a] == T["trialN"][b] and T["wordPair"][a] == T["wordPair"][b]
                and T["Geminate"][a] == "Singleton" and T["Geminate"][b] == "Geminate")
    checks.internal("rows come in singleton/geminate pairs of one trial (same speaker, trial number, word pair); traces are on the singleton row", True)
    # landmark indices are MATLAB (1-based) samples of the trial trace; the gesture trace is trace[onset-1:offset]
    off_ok = 0
    for i in range(n):
        t = np.asarray(traj["traj"][i - i % 2]).ravel()
        g = np.asarray(T["gestTraj"][i]).ravel()
        a, b = int(sc(lm[i]["onsIx"])), int(sc(lm[i]["offIx"]))
        off_ok += int(len(g) == b - a + 1 and np.array_equal(t[a - 1:b], g))
    checks.internal("every token's gesture trace equals the trial trace from landmark onset to offset (landmark indices are 1-based samples)", off_ok == n, f"{off_ok}/{n}")
    assert off_ok == n
    pair_no = {}
    for i in range(0, n, 2):
        pair_no[i] = i // 2 + 1
    ms = lambda ix: (sc(ix) - 1) / FS * 1000  # noqa: E731
    seq, rows = {}, []
    for i in range(n):
        seq[sp[i]] = seq.get(sp[i], 0) + 1
        l = lm[i]
        poa = str(T["POA"][i])
        gem = str(T["Geminate"][i])
        sec = lambda k: float(np.asarray(T[k][i]).ravel()[0]) * 1000  # noqa: E731
        num = lambda k: float(np.asarray(T[k][i]).ravel()[0])  # noqa: E731
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Tokyo Japanese mimetic words, singleton and geminate consonants at four places of articulation",
            "group": "", "subject": sp[i], "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE, "session": 1, "block": "", "trial": seq[sp[i]], "phase": "test", "time_s": "", "task": "speech",
            "unit": "utterance", "condition": gem.lower(), "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal",
            "excluded_by_authors": False,
            "stimulus": word[i], "target_sound": str(T["targetSegment"][i]), "place_of_articulation": poa, "word_pair": str(T["wordPair"][i]),
            "gesture_signal": SIGNAL[poa], "trial_trace_id": pair_no[i - i % 2], "trajectory_id": pair_no[i - i % 2],
            "trial_number_authors": int(num("trialN")),
            "gons_ms": ms(l["onsIx"]), "pvel_ms": ms(l["PeakVel1Ix"]), "nons_ms": ms(l["targIx"]), "maxc_ms": (num("CDMaxIx") - 1) / FS * 1000,
            "noffs_ms": ms(l["relIx"]), "pvel2_ms": ms(l["PeakVel2Ix"]), "goffs_ms": ms(l["offIx"]),
            "onset_position_mm": sc(l["onsVal"]), "target_position_mm": sc(l["targVal"]), "release_position_mm": sc(l["relVal"]),
            "offset_position_mm": sc(l["offVal"]),
            "closing_duration_ms": sec("cloDur"), "plateau_duration_ms": sec("platDur"), "opening_duration_ms": sec("relDur"),
            "constriction_max_mm": num("CDMax"), "constriction_max_z": num("CDMaxZ"),
            "closing_amplitude_mm": num("MovAmpl1"), "opening_amplitude_mm": num("MovAmpl2"),
            "closing_peak_vel_cm_s": num("PeakVel1"), "opening_peak_vel_cm_s": num("PeakVel2"), "closing_peak_vel_z": num("PeakVel1Z"),
            "closing_stiffness_per_s": num("K1"), "opening_stiffness_per_s": num("K2"),
            "signal_mean_mm": num("POAMu"), "signal_sd_mm": num("POAStd"), "velocity_mean_cm_s": num("POAVelMu"), "velocity_sd_cm_s": num("POAVelStd"),
            "preceding_vowel_duration_ms": sec("V1Dur"), "v2_duration_ms": sec("V2Dur"), "v1_to_clo_onset_ms": sec("V1CloLag"),
            "v1_to_clo_target_ms": sec("V1TargLag"), "v1_v2_lag_ms": sec("V1V2Dur"), "target_consonant_duration_ms": sec("CAcDur"),
        })
    return rows


def write_traces(T, path):
    n = T["_nrows"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["trajectory_id", "sample", "time_s", "target_articulator_signal_mm", "tongue_body_z_mm", "tongue_body_y_mm",
                    "tongue_dorsum_z_mm", "tongue_dorsum_y_mm"])
        for i in range(0, n, 2):
            ch = [np.asarray(T[k][i]).ravel() for k in ("traj", "trajTMz", "trajTMy", "trajTDz", "trajTDy")]
            assert len({len(c) for c in ch}) == 1
            for s in range(len(ch[0])):
                w.writerow([i // 2 + 1, s, format(s / FS, ".8g")] + [format(float(c[s]), ".8g") for c in ch])


def checks_all(T, rows, checks):
    a = pd.DataFrame(rows)
    checks.published("tokens analysed (15 words x 2 x 10 repetitions x 6 speakers = 1800, minus 42 excluded)", len(a), 1758, 0, f"{SRC}, Methods 2.2 and 2.4")
    checks.published("speakers", a.subject.nunique(), 6, 0, f"{SRC}, Methods 2.1")
    checks.published("excluded tokens (21 pairs)", 1800 - len(a), 42, 0, f"{SRC}, Methods 2.4")
    checks.published("words (singleton/geminate pairs)", a.word_pair.nunique(), 15, 0, f"{SRC}, Methods 2.2")
    from scipy.stats import spearmanr
    r1 = spearmanr(a.plateau_duration_ms, a.target_consonant_duration_ms, nan_policy="omit")[0]
    checks.published("Spearman correlation of plateau duration and target consonant acoustic duration", float(r1), 0.75, 0.005, f"{SRC}, Results 3.1 (rho = 0.75)")
    r2 = spearmanr(a.v1_to_clo_target_ms, a.preceding_vowel_duration_ms, nan_policy="omit")[0]
    checks.published("Spearman correlation of V1-to-target lag and preceding vowel duration", float(r2), 0.77, 0.005, f"{SRC}, Results 3.3 (rho = .77)")
    c = a.condition.value_counts()
    checks.internal("equal numbers of singleton and geminate tokens", c["singleton"] == c["geminate"] == 879)
    checks.internal("durations are non-negative and the plateau is the target-to-release lag",
                    bool((a.closing_duration_ms >= 0).all() and (a.plateau_duration_ms >= 0).all() and (a.opening_duration_ms >= 0).all()
                         and np.allclose(a.nons_ms + a.plateau_duration_ms, a.noffs_ms, atol=0.01)))


def main():
    checks = stdlib.Checks()
    T = load(checks)
    rows = build(T, checks)
    checks_all(T, rows, checks)
    for c in checks.items:
        print(("ok     " if c["ok"] else "FAILED "), c["what"], c.get("computed", ""), c.get("reported", ""))
    tpath = stdlib.raw_dir(ID) / "trial_traces.csv"
    write_traces(T, tpath)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Decoded the authors' MATLAB table (TAllClean.mat), converted durations and lags from seconds to milliseconds, converted the "
                 "authors' 1-based landmark sample indices to times in ms from the start of the trial trace, wrote the five articulator "
                 "traces of each trial once to samples/trial_traces.csv, and replaced speaker labels with codes. The audio, the "
                 "word and phone intervals (TGW, TGPh) and the duplicate gesture-only traces (gestTraj, gestTrajDer) were left out."),
        how=[
            "One row is one spoken token of a Tokyo Japanese mimetic word by one of six speakers: 15 words, each said as a singleton and as an "
            "emphatic geminate (condition) in isolation, ten times. The authors' 1,800 tokens minus 21 singleton/geminate pairs without clear gestural boundaries leave the "
            "1,758 rows shared (879 singleton, 879 geminate); rows are in the authors' order, a singleton row followed by its geminate partner.",
            "The consonantal gesture was landmarked on the vertical movement of the articulator that fits the place of articulation "
            "(gesture_signal): lip aperture (LA) for bilabials, tongue tip height (TT_Z) for apical (alveolar) nasals, tongue body height "
            "(TB_Z) for the laminal (palatal) nasal and tongue dorsum height (TD_Z) for dorsal (velar) stops. gons_ms, pvel_ms, nons_ms, "
            "noffs_ms, pvel2_ms and goffs_ms are the gestural onset, closing peak velocity, target (plateau onset), release, opening peak "
            "velocity and offset (20 per cent of peak velocity thresholds), maxc_ms the sample of maximum constriction, all in ms from the start of the trial trace "
            "(the first sample is time 0). closing, plateau and opening duration are the lags onset-target, target-release and release-offset.",
            "samples/trial_traces.csv holds, for each trial (a singleton and its geminate partner, trial_trace_id), the authors' target "
            "articulator signal (target_articulator_signal_mm, in mm) and the vertical and anterior-posterior tongue body and tongue dorsum "
            "positions (mm) at 1250 Hz; time_s starts at 0 at the start of the trial trace. Kinematic columns are the authors' own: movement amplitude "
            "and peak velocity are signed (negative for lip aperture closing), stiffness is peak velocity (mm/s) over amplitude (mm), z-scores are within "
            "articulator (signal_mean_mm, signal_sd_mm, velocity_mean_cm_s, velocity_sd_cm_s are their z-scoring constants).",
            "Acoustic values (preceding_vowel_duration_ms, the V1 lags, target_consonant_duration_ms) come from the authors' manual PRAAT "
            "segmentation. Word labels use an ASCII spelling of the IPA (for example J for the palatal nasal, S for the voiceless palato-alveolar "
            "fricative); target_sound uses the authors' symbol. Sex and age of the speakers are not given per speaker (the paper reports 2 male and "
            "4 female speakers).",
        ],
        unit_note="spoken token (one gesture)",
        differences=[
            "The paper's mixed-effects regression coefficients (MATLAB fitlme with random slopes) were not refitted; the checks are the token and "
            "speaker counts and two Spearman correlations.",
            "The deposit's V2Dur column equals V1Dur in every row, which the paper does not explain; both are kept as shared.",
            "The paper is under review as a preprint at the OSF project; the published article is in Laboratory Phonology 17(1), "
            "26525 (full text blocked by a bot check, so numbers were checked against the OSF preprint PDF).",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    return stdlib.release(ID, "articulography", rows, meta, checks, extra_files=[("samples/trial_traces.csv", str(tpath))])


if __name__ == "__main__":
    main()
