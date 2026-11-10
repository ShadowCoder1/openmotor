"""Burroni, Maspong, Hoole & Kirby (preprint, under review, 2026): Phonological control of time in speech: Italian consonantal
length contrasts as an articulatory task. 23 native Italian speakers said the carrier sentence 'Please say ___ twice' with
one of six pseudo-words (ipa, ippa, iba, ibba, ima, imma; singleton vs geminate bilabial consonant) at five self-selected
rates (very slow ... very fast), 12 repetitions each, with electromagnetic articulography (Carstens AG501, 1250 Hz).

Raw file, raw_extra/Burroni_ItalianLength_OSF/Supp.zip (OSF yh8tm, SupplementaryMaterialBurroniEtAl.zip; fetched with
tools/std/fetch.py). Only three members are read, by name, from the ZIP:
  Tables/TIntragestural.mat   MATLAB table 'T', one row per analysed token (7900): SP speaker, Gem (0 singleton, 1 geminate),
                              Word, cloDur / platDur / relDur (closure, plateau, release phase of the lip aperture gesture,
                              seconds), LAMin / LAMax / LAAmp (mm), LAPeakVel (cm/s), LAStiff (1/s), V1Dur ... RelV2 (lags,
                              seconds), segDur / vDur / wDur / uttDur (acoustic, seconds), wDurZ / uttDurZ, ratePhonSec,
                              Segment (p, b, m), rate (1-5), rateWord (Italian rate cue)
  Tables/trajC.mat            7900 x 1 cell: the lip aperture samples (mm) of each token between gestural onset and offset,
                              same row order as the table (checked below: min, max and length agree with the table)
  Meta/MetaT.mat              MATLAB table: SP1..SP23, Sex, Orig (N or CS: north or central-south of the Rimini-La Spezia line)
The tables are MATLAB table objects decoded from the MAT-file's own object store with scipy (helper
Burroni_ItalianLength_OSF_mat.py); nothing from the deposit is executed. Definitions: paper (preprint PDF on the same OSF
project, read as text), Methods 2.1-2.4, and the authors' b_statsAnalyses.m / c_MLScript.m (read as text; the speaker key of
MetaT is the one their script builds: SPk = Sk, the last one S24).
Run from the site folder: python3 -I tools/std/convert/Burroni_ItalianLength_OSF.py
"""
import csv
import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Burroni_ItalianLength_OSF_mat import MatTables  # noqa: E402

ID = "Burroni_ItalianLength_OSF"
SRC = "Burroni et al. preprint (OSF yh8tm)"
P = "SupplementaryMaterialBurroniEtAl/"
DEVICE = "Carstens AG501 electromagnetic articulograph (1250 Hz), Sennheiser ME66 microphone"
REGION = {"N": "north of the Rimini-La Spezia line", "CS": "central or south of the Rimini-La Spezia line"}
MS = {"cloDur": "closing_duration_ms", "platDur": "plateau_duration_ms", "relDur": "opening_duration_ms",
      "V1Dur": "v1_articulatory_duration_ms", "V1Clo": "v1_to_clo_onset_ms", "V1Targ": "v1_to_clo_target_ms",
      "CloV2": "clo_onset_to_v2_ms", "TargV2": "targ_to_v2_ms", "RelV2": "rel_to_v2_ms", "V2Dur": "v2_duration_ms",
      "wDur": "word_duration_ms", "segDur": "target_consonant_duration_ms", "vDur": "preceding_vowel_duration_ms",
      "uttDur": "utterance_duration_ms"}
SAME = {"LAMin": "la_min_mm", "LAMax": "la_max_mm", "LAAmp": "la_amplitude_mm", "LAPeakVel": "la_peak_velocity_cm_s",
        "LAStiff": "la_stiffness_per_s", "uttDurZ": "utterance_duration_z", "wDurZ": "word_duration_z",
        "ratePhonSec": "speech_rate_phones_s"}


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def nz(x):
    return None if pd.isna(x) else float(x)


def frame(t):
    return pd.DataFrame({n: (list(t[n]) if not isinstance(t[n], np.ndarray) else t[n].ravel()) for n in t["_names"]})


def load(checks):
    raw = stdlib.raw_dir(ID)
    z = zipfile.ZipFile(raw / "Supp.zip")
    for m in ("Tables/TIntragestural.mat", "Tables/trajC.mat", "Meta/MetaT.mat"):  # three named members only
        (raw / Path(m).name).write_bytes(z.read(P + m))
    T = frame(MatTables(raw / "TIntragestural.mat").table("T"))
    M = frame(MatTables(raw / "MetaT.mat").table("T"))
    traj = [np.asarray(c[0]).ravel().astype(float) for c in sio.loadmat(raw / "trajC.mat")["trajC"]]
    ok = (len(traj) == len(T) == 7900
          and np.array_equal(np.array([x.min() for x in traj]), T.LAMin.to_numpy())
          and np.array_equal(np.array([x.max() for x in traj]), T.LAMax.to_numpy())
          and float(np.nanmax(np.abs(np.array([len(x) for x in traj]) / 1250 - (T.cloDur + T.platDur + T.relDur)))) < 0.001)
    checks.internal("lip aperture samples (trajC.mat) agree row by row with the table: minimum = LAMin, maximum = LAMax, "
                    "samples / 1250 Hz = closure + plateau + release duration", bool(ok))
    assert ok
    return T, M, traj


def build(T, M, traj, checks):
    keys = [f"S{i + 1}" for i in range(len(M))]
    keys[-1] = "S24"  # as the authors' script b_statsAnalyses.m does (SPArr(end) = "S24")
    meta = {k: (r.Sex, r.Orig) for k, r in zip(keys, M.itertuples(index=False))}
    checks.internal("every speaker of the table is in the meta table and vice versa (23)", set(T.SP) == set(meta), str(sorted(set(T.SP) ^ set(meta))))
    assert set(T.SP) == set(meta)
    rows, seq = [], {}
    for i, r in enumerate(T.itertuples(index=False)):
        sex, orig = meta[r.SP]
        seq[r.SP] = seq.get(r.SP, 0) + 1
        row = {
            "dataset": ID, "study": ID, "experiment": "Italian singleton and geminate bilabials at five speaking rates",
            "group": REGION[orig], "subject": r.SP, "subject_excluded_by_authors": False, "age": "", "sex": sex,
            "handedness": "", "population": "healthy", "setting": "lab", "device": DEVICE, "session": 1, "block": r.rateWord,
            "trial": seq[r.SP], "phase": "test", "time_s": "", "task": "speech", "unit": "utterance",
            "condition": "geminate" if r.Gem == 1 else "singleton", "pert_type": "none", "pert_value": 0, "pert_unit": "",
            "feedback": "normal", "excluded_by_authors": False,
            "stimulus": r.Word, "target_sound": r.Segment, "trajectory_id": i + 1, "rate_level": int(r.rate), "rate_cue": r.rateWord,
        }
        for a, b in MS.items():
            v = nz(getattr(r, a))
            row[b] = None if v is None else v * 1000
        for a, b in SAME.items():
            row[b] = nz(getattr(r, a))
        rows.append(row)
    same = bool(((T.V1Dur == T.V1V2) | (T.V1Dur.isna() & T.V1V2.isna())).all())
    checks.internal("V1Dur and V1V2 are identical in every row (one column kept)", same)
    return rows, meta


def fit(df, y, form):
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[y]).copy()
    if y == "platDur_ms":
        d = d[d.platDur > 0]  # the authors drop plateau durations of 0
    d["one"] = 1
    vc = {}
    for g in ("SP", "Segment"):
        vc[g + "_i"] = f"0+C({g})"
        vc[g + "_gem"] = f"0+C({g}):Gem"
        vc[g + "_utt"] = f"0+C({g}):uttDurZ"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(f"{y} ~ {form}", d, groups=d["one"], vc_formula=vc).fit(reml=True, method=["lbfgs", "powell"]).fe_params


def checks_all(T, M, rows, meta, checks):
    a = pd.DataFrame(rows)
    checks.published("tokens analysed", len(a), 7900, 0, f"{SRC}, Methods 2.3 (7900 tokens entered the analyses)")
    checks.published("speakers", a.subject.nunique(), 23, 0, f"{SRC}, Methods 2.1")
    sx = a.drop_duplicates("subject").sex.value_counts()
    checks.published("male speakers", int(sx["M"]), 7, 0, f"{SRC}, Methods 2.1")
    checks.published("female speakers", int(sx["F"]), 16, 0, f"{SRC}, Methods 2.1")
    rg = a.drop_duplicates("subject").group.value_counts()
    checks.published("speakers from the central or south", int(rg[REGION["CS"]]), 12, 0, f"{SRC}, Methods 2.1")
    checks.published("speakers from the north", int(rg[REGION["N"]]), 11, 0, f"{SRC}, Methods 2.1")
    for g, rep in ((0, 0.63), (1, 0.9)):
        d = T[T.Gem == g]
        checks.published(f"correlation of plateau duration and acoustic consonant duration, {'geminates' if g else 'singletons'}",
                         d.platDur.corr(d.segDur), rep, 0.006, f"{SRC}, Results 3.2")
    D = T.copy()
    for c in ("cloDur", "platDur"):
        D[c + "_ms"] = D[c] * 1000
    for y, form, items in (
        ("cloDur_ms", "Gem+uttDurZ", [("Intercept", 102, 1.0), ("Gem", 8, 0.6), ("uttDurZ", 6, 1.6)]),
        ("platDur_ms", "Gem*uttDurZ", [("Intercept", 43, 1.0), ("Gem", 52, 1.0), ("uttDurZ", 15, 1.1), ("Gem:uttDurZ", 41, 1.5)]),
        ("LAMin", "Gem+uttDurZ", [("Intercept", 16.7, 0.05), ("Gem", -0.8, 0.05), ("uttDurZ", -0.3, 0.05)]),
        ("LAAmp", "Gem", [("Intercept", 10.3, 0.05), ("Gem", 1.1, 0.06)]),
    ):
        fe = fit(D, y, form)
        for k, rep, tol in items:
            checks.published(f"{y}: fixed effect {k}, mixed model with by-speaker and by-segment slopes",
                             float(fe[k]), rep, tol, f"{SRC}, Results 3.2-3.3")
    checks.internal("durations are non-negative; lip aperture 5-60 mm; rate 1-5 with 12 repetitions per word and rate at most",
                    bool((a.closing_duration_ms.dropna() >= 0).all() and a.la_min_mm.between(5, 60).all()
                         and a.la_max_mm.between(5, 60).all() and a.rate_level.between(1, 5).all()
                         and a.groupby(["subject", "stimulus", "rate_level"]).size().max() <= 12 + 3))


def write_traj(traj, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["trajectory_id", "sample", "time_s", "lip_aperture_mm"])
        for i, x in enumerate(traj):
            for k, v in enumerate(x):
                w.writerow([i + 1, k, format(k / 1250, ".8g"), format(v, ".8g")])


def main():
    checks = stdlib.Checks()
    T, M, traj = load(checks)
    rows, meta = build(T, M, traj, checks)
    checks_all(T, M, rows, meta, checks)
    for c in checks.items:
        print(("ok     " if c["ok"] else "FAILED "), c["what"], c.get("computed", ""), c.get("reported", ""))
    tpath = stdlib.raw_dir(ID) / "lip_aperture.csv"
    write_traj(traj, tpath)
    s, paper = paper_meta()
    meta_r = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="Academic Free License (AFL) 3.0",
        license_url="https://opensource.org/license/afl-3-0-php",
        changes=("Decoded the authors' MATLAB tables, merged the speaker table (sex and region of origin) into the token table, "
                 "converted durations and lags from seconds to milliseconds, kept V1Dur and left out its identical copy V1V2, "
                 "wrote the lip aperture samples of every token to samples/lip_aperture.csv, and replaced speaker labels with codes."),
        how=[
            "One row is one spoken token of the carrier sentence 'Please say ___ twice' by one of 23 Italian speakers, with the "
            "target word ipa, ippa, iba, ibba, ima or imma (singleton or geminate bilabial consonant), at one of five cued rates "
            "(block and rate_cue, from very slow to very fast); rows are in the authors' order. These are the 7,900 tokens the "
            "authors analysed (they recorded 8,280; 210 were lost to equipment failure and 170 more were excluded when their "
            "gestural landmarks could not be found). One token has the word 'immaq' in the authors' table, kept as it is.",
            "The consonantal gesture is the closing and opening of lip aperture (distance between the upper and lower lip "
            "sensors). closing_duration_ms, plateau_duration_ms and opening_duration_ms are the lags from gestural onset to target, "
            "target to release and release to offset, found by the authors' velocity-threshold routine (20% of peak velocity). la_*"
            " columns are the minimum and maximum lip aperture, the closing movement's amplitude and peak velocity, and "
            "stiffness (peak velocity / amplitude). The v1_* and *_to_v2_* columns are lags between the tongue-movement landmarks "
            "of the first vowel (/i/), the lip gesture and the second vowel (/a/); target_consonant_duration_ms, "
            "preceding_vowel_duration_ms, word_duration_ms and utterance_duration_ms are acoustic durations; z-scores and the "
            "phoneme rate are the authors' own.",
            "samples/lip_aperture.csv holds each token's lip aperture (mm) at 1250 Hz from gestural onset to offset, keyed by "
            "trajectory_id (the row number in the authors' table); time_s there starts at 0 at the gestural onset.",
            "sex and group (region of origin: north or central-south of the Rimini-La Spezia line) come from the authors' "
            "speaker table; age is not shared per speaker (the paper gives 22-42 years, mean 29.5).",
        ],
        unit_note="spoken token of one speaker (one gesture)",
        differences=[
            "The OSF project carries the Academic Free License 3.0; the paper is a preprint under review (the OSF project lists "
            "the authors as Burroni et al.; the DOI is not yet assigned).",
            "The paper's mixed models (MATLAB fitlme with diagonal random-effect covariances) are refitted here with Python's "
            "statsmodels; fixed effects agree within the tolerances above (for example plateau duration with utterance duration: "
            "14.0 ms here, 15 in the paper; closure duration with utterance duration: 7.5 ms here, 6 in the paper). V2Dur is not defined in the deposit or the paper, and 12 tokens lack V1 landmarks.",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    return stdlib.release(ID, "articulography", rows, meta_r, checks, extra_files=[("samples/lip_aperture.csv", str(tpath))])


if __name__ == "__main__":
    main()
