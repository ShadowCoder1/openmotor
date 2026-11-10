"""Barbier, Baum, Menard & Shiller (2020), J Acoust Soc Am 147(2):1163-1178: adaptation to a palatal prosthesis across eight VCV
syllables (9 talkers, 4 speech tests: 1 baseline without prosthesis, 2 right after insertion, 3 after 20 min of sentence practice with it,
4 after removal; 8 tokens per syllable and test).

Raw files, raw_extra/Barbier2020_JASA/ (OSF srcuf), read with the authors' scripts (kinematic_AMPLITUDE_analysis_PUB.m, LMER_analysis_PUB.R) as text:
  allDATA_kinematics_K.mat      allDATAk(test, VCV, talker).TBD / .TBL: 3 (V1, C, V2) x 2 (x, z) x 8 tokens, mm, head-centred frame
  allData_audioV_F1/F2_comp{1-4}.csv, allData_audioPTKS_comp{1-4}.csv   alldata + PHASE (1, 2 = first, second test of comparison comp),
      STIM (VCV 1-8), SUB, TRIAL (1-8): vowel F1, vowel F2 (Hz) and consonant spectral centroid (Hz). Comparison 1 = tests 1-2, 2 = tests 2-3,
      3 = tests 3-4, 4 = tests 1-4, so every test appears in more than one file; each test is taken once and the repeats are checked.
  the allData_TBD/TBL_both files are the authors' rotated, re-centred positions for the statistics and are not carried.
Run: python3 -I tools/std/convert/Barbier2020_JASA.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Barbier2020_JASA"
P = "Barbier et al. 2020, "
VCV = {1: "isi", 2: "asa", 3: "iti", 4: "ata", 5: "iki", 6: "aka", 7: "ipi", 8: "apa"}
COMP = {1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (1, 4)}
PHASE = {1: "baseline", 2: "test", 3: "test", 4: "aftereffect"}
BLOCK = {1: "Speech test 1 (no prosthesis)", 2: "Speech test 2 (prosthesis inserted)", 3: "Speech test 3 (after sentence practice)",
         4: "Speech test 4 (prosthesis removed)"}


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    ac = {}
    dup_ok = True
    n_dup = 0
    for key, fn in (("f1_hz", "audioV_F1"), ("f2_hz", "audioV_F2"), ("spectral_centroid_hz", "audioPTKS")):
        for comp in (1, 2, 3, 4):
            d = pd.read_csv(raw / f"allData_{fn}_comp{comp}.csv")
            for r in d.itertuples(index=False):
                t = COMP[comp][int(r.PHASE) - 1]
                k = (key, int(r.SUB), int(r.STIM), t, int(r.TRIAL))
                v = r.alldata
                if k in ac:
                    n_dup += 1
                    old = ac[k]
                    dup_ok &= (pd.isna(old) and pd.isna(v)) or (not pd.isna(old) and not pd.isna(v) and abs(old - v) < 1e-6)
                else:
                    ac[k] = v
    checks.internal("tests shared between comparison files carry identical values", bool(dup_ok), f"{n_dup} repeated values compared")
    A = sio.loadmat(raw / "allDATA_kinematics_K.mat", struct_as_record=False)["allDATAk"]
    rows = []
    cnt = {}
    for sub in range(1, 10):
        for t in (1, 2, 3, 4):
            for stim in range(1, 9):
                for tok in range(1, 9):
                    e = A[t - 1, stim - 1, sub - 1]
                    row = {
                        "dataset": ID, "study": ID, "experiment": "Palatal prosthesis, eight VCV syllables", "group": "",
                        "subject": f"T{sub}", "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
                        "population": "healthy", "setting": "lab", "device": "electromagnetic articulography (tongue blade and dorsum sensors), microphone",
                        "session": 1, "block": BLOCK[t], "phase": PHASE[t], "time_s": "", "task": "speech", "unit": "utterance",
                        "condition": VCV[stim], "pert_type": "palatal_prosthesis" if t in (2, 3) else "none",
                        "pert_value": 1.0 if t in (2, 3) else 0.0, "pert_unit": "prosthesis present (1) or absent (0)", "feedback": "altered" if t in (2, 3) else "normal",
                        "excluded_by_authors": False, "stimulus": VCV[stim],
                        "segment": "vowel (formants) and consonant (spectral centroid) of the VCV syllable",
                    }
                    for key in ("f1_hz", "f2_hz", "spectral_centroid_hz"):
                        v = ac[(key, sub, stim, t, tok)]
                        row[key] = "" if pd.isna(v) else float(v)
                    cnt[sub] = cnt.get(sub, 0) + 1
                    row["trial"] = cnt[sub]
                    rows.append(row)
                e = A[t - 1, stim - 1, sub - 1]
                arrs = {nm: np.asarray(getattr(e, sens), float).reshape(3, 2, -1) for sens, nm in (("TBD", "tbd"), ("TBL", "tbl"))}
                for tok in range(1, arrs["tbd"].shape[2] + 1):
                    row = {k: v for k, v in rows[-1].items() if k not in ("f1_hz", "f2_hz", "spectral_centroid_hz")}
                    row["segment"] = "tongue sensor positions at V1, C and V2 of the VCV syllable"
                    for nm, arr in arrs.items():
                        for pi, pos in enumerate(("v1", "c", "v2")):
                            for ai, ax in enumerate(("x", "z")):
                                v = arr[pi, ai, tok - 1] if tok - 1 < arr.shape[2] else np.nan
                                row[f"{nm}_{pos}_{ax}_mm"] = "" if np.isnan(v) else float(v)
                    cnt[sub] += 1
                    row["trial"] = cnt[sub]
                    rows.append(row)
    a = pd.DataFrame(rows)
    checks.published("talkers", a.subject.nunique(), 9, 0, P + "deposit scripts (allDATAk is 4 x 8 x 9: tests x VCV x talkers)")
    checks.published("VCV syllables", a.condition.nunique(), 8, 0, P + "deposit scripts (8 VCV conditions)")
    checks.published("speech tests", a.block.nunique(), 4, 0, P + "deposit scripts (4 speech tests)")
    checks.published("tokens per VCV and test", a[a.segment.str.startswith("vowel")].groupby(["subject", "block", "condition"]).size().iloc[0], 8, 0, P + "deposit scripts (8 tokens per condition)")
    checks.internal("2304 acoustic rows (9 x 4 x 8 x 8)", int((a.segment.str.startswith("vowel")).sum()) == 2304, str(len(a)))
    print("kinematic rows", int((~a.segment.str.startswith("vowel")).sum()))
    f1 = pd.to_numeric(a.f1_hz.replace("", np.nan)).dropna()
    checks.internal("F1 within 100-1300 Hz", bool(f1.between(100, 1300).all()), f"{f1.min():.0f}-{f1.max():.0f}, {int((~f1.between(100, 1300)).sum())} outside")
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    sx = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(sx["papers"][0])
    meta = dict(
        study=ID, paper={k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=sx["lab"], data_url=sx["data_url"],
        license="none stated", license_url="",
        changes=("Merged the authors' acoustic comparison files (each speech test appears in several, taken once) with the per-token tongue "
                 "sensor positions in their MATLAB file, and replaced talker numbers with codes."),
        how=[
            "One row is one token of /isi asa iti ata iki aka ipi apa/: an acoustic row (8 per syllable, test and talker) with vowel F1 and F2 "
            "and consonant spectral centroid (Hz, the authors' acoustic files), or a kinematic row with the positions in mm of the tongue dorsum (tbd) and blade (tbl) "
            "sensors at the first vowel, consonant and second vowel (x, z in the mid-sagittal plane of the head-centred frame).",
            "Speech test 1 is without the palatal prosthesis, test 2 right after insertion, test 3 after about 20 minutes of sentence practice with it, "
            "and test 4 after removal. Acoustic rows and kinematic rows are separate rows of the same cell: the authors do not state that token numbers in the two "
            "sources match (some cells have fewer than 8 kinematic tokens), so the two are not paired. "
            "Axis orientation (x forward, z up) is taken from the authors' script comments and not confirmed from the paper, which was not openly available.",
        ],
        unit_note="token of a VCV syllable",
        differences=["The authors left talker 7 out of the kinematic statistics as an outlier; those rows are kept and not marked. Their rotated and re-centred "
                     "kinematic files used for the statistics are not carried, and none of the published statistics were recomputed (full text not accessible)."],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
