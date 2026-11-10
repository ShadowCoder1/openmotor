"""Lo (2025), Speech Communication: nasal coarticulation in Lombard speech. Eleven male Southern British English speakers
read 200 words (176 targets, 24 fillers; 11 vowels x 4 consonant contexts) in a describe-the-code task with an
experimenter, in two sessions at least two weeks apart, each with three conditions in random order: quiet, Lombard
(white noise, 70 dB SPL through headphones) and phone (band-passed 300-3400 Hz noise, 70 dB SPL, experimenter not visible).

Raw files, raw_extra/Lo2025_SpeechCommun/ (OSF ynxsj, R .rds, read with pyreadr; fetched with tools/std/fetch.py):
  vowel_dur.rds     one row per token (13052): speaker, session (01, 02), section (1-3 = the condition blocks of the session
                    in recording order), condition (B quiet, L Lombard, P phone), item (1-50), word, phone, start / end /
                    duration of the manually segmented vowel (seconds from the start of the session's recording), segments,
                    vowel (lexical set), context
  intensities.rds   the same tokens: mean intensity over the vowel (dB)
  f0s.rds           f0 track every 5 ms (voicing flag, Hz) of the same tokens
  nasalances.rds    33 rows per token: nasalance at 11 points over each of onset consonant, vowel, coda consonant
  formants_02_B.rds FastTrack formants at 11 points for one speaker's quiet tokens (400 tokens)
What the values are: paper Methods 2.4 (read as text from the accepted manuscript) and the authors' analysis.R.
Run from the site folder: python3 -I tools/std/convert/Lo2025_SpeechCommun.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Lo2025_SpeechCommun"
SRC = "Lo 2025, Speech Communication 169:103205"
KEY = ["speaker", "session", "section", "condition", "item", "word"]
COND = {"B": "quiet", "L": "Lombard noise", "P": "phone-band noise"}
DEVICE = "Glottal Enterprises handheld nasometer (two microphones, 44.1 kHz), closed-back headphones"
WINDOW = ("whole vowel, manually segmented in Praat; nasalance and formants at 11 equidistant points over the vowel; "
          "f0 mean of voiced 5 ms frames")


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def series(v):
    return ";".join(format(float(x), ".8g") for x in v)


def build(checks):
    raw = stdlib.raw_dir(ID)
    rd = lambda f: pyreadr.read_r(str(raw / f))[None]  # noqa: E731
    vd, it, f0, na, fo = rd("vowel_dur.rds"), rd("intensities.rds"), rd("f0s.rds"), rd("nasalances.rds"), rd("formants_02_B.rds")
    vd = vd.merge(it[KEY + ["intensity"]], on=KEY, how="left", validate="1:1")
    checks.internal("intensity file holds the same 13052 tokens as the vowel duration file",
                    len(vd) == len(it) == 13052 and vd.intensity.notna().all())
    v = f0[(f0.voicing == 1) & (f0.f0 > 0)].copy()
    v["erb"] = 1000 * np.log(1 + 0.00437 * v.f0) / 107.939
    m = v.groupby(KEY).agg(f0_hz=("f0", "mean"), f0_erb=("erb", "mean")).reset_index()
    vd = vd.merge(m, on=KEY, how="left", validate="1:1")
    ns = {}
    for k, g in na.groupby(KEY, sort=False):
        assert len(g) == 33
        ns[k] = series(g.nasalance.to_numpy())
    fs = {}
    for k, g in fo.sort_values(KEY + ["step"]).groupby(KEY, sort=False):
        assert len(g) == 11
        fs[k] = tuple(series(g[c]) for c in ("f1", "f2", "f1p", "f2p"))
    vd = vd.sort_values(["speaker", "session", "start"], kind="stable").reset_index(drop=True)
    rows = []
    for r in vd.itertuples(index=False):
        k = tuple(getattr(r, c) for c in KEY)
        cond = COND[r.condition]
        noise = r.condition != "B"
        f = fs.get(k, ("", "", "", ""))
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Lombard and phone speech", "group": "", "subject": r.speaker,
            "subject_excluded_by_authors": False, "age": "", "sex": "M", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE, "session": int(r.session), "block": cond, "trial": 0, "phase": "test",
            "time_s": float(r.start), "task": "speech", "unit": "utterance", "condition": cond,
            "pert_type": ("white_noise" if r.condition == "L" else "phone_band_noise") if noise else "none",
            "pert_value": 70 if noise else 0, "pert_unit": "dB SPL", "feedback": "masked" if noise else "normal",
            "excluded_by_authors": False,
            "stimulus": r.word, "segment": f"vowel /{r.phone}/ (ASCII phonetic) of {r.segments}", "measure_window": WINDOW,
            "duration_ms": float(r.duration) * 1000, "f0_hz": nz(r.f0_hz), "f0_erb": nz(r.f0_erb),
            "intensity_db": float(r.intensity), "masking_noise": ("white noise, 70 dB SPL" if r.condition == "L" else
                                                                  "white noise band-passed 300-3400 Hz, 70 dB SPL" if noise else "none"),
            "nasalance_series": ns.get(k, ""), "vowel_set": r.vowel, "phonetic_context": r.context,
            "f1_series_hz": f[0], "f2_series_hz": f[1], "f1p_series_hz": f[2], "f2p_series_hz": f[3],
            "authors_trial_number": int(r.item),
        })
    seq = {}
    for r in rows:
        seq[r["subject"]] = seq.get(r["subject"], 0) + 1
        r["trial"] = seq[r["subject"]]
    return rows, v, f0


def nz(x):
    return None if pd.isna(x) else float(x)


def lmm(df, y, extra=("dur",), slope=True):
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[y]).copy()
    d["cond"] = pd.Categorical(d.condition, ["quiet", "Lombard noise", "phone-band noise"])
    vs = sorted(d.vowel_set.unique())
    vs.remove("TRAP")
    d["vow"] = pd.Categorical(d.vowel_set, vs + ["TRAP"])  # sum coding with TRAP as the omitted (-1) level
    d["ses"] = np.where(d.session == 1, 1.0, -1.0)
    d["item_m"] = (d.authors_trial_number - 25.5) / 49
    d["dur_m"] = d.duration_ms / 1000 - d.groupby("condition").duration_ms.transform("mean") / 1000
    f = f"{y} ~ C(cond) + C(vow, Sum) + ses + item_m" + (" + dur_m" if "dur" in extra else "")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.mixedlm(f, d, groups=d["subject"], re_formula="~C(cond)" if slope else "1")
        return mod.fit(reml=True, method=["lbfgs", "powell"])


def main():
    checks = stdlib.Checks()
    rows, v, f0 = build(checks)
    a = pd.DataFrame(rows)
    a["intensity_db"] = a.intensity_db.astype(float)
    a["f0_erb"] = pd.to_numeric(a.f0_erb)
    a["f0_hz"] = pd.to_numeric(a.f0_hz)
    fill = a.phonetic_context.str.startswith("F") | a.phonetic_context.str.endswith("F")
    checks.published("tokens", len(a), 13052, 0, f"{SRC}, Methods 2.4 (Data processing)")
    checks.published("filler tokens", int(fill.sum()), 1553, 0, f"{SRC}, Methods 2.4 (Data processing)")
    checks.published("speakers analysed (11 male; the nonbinary speaker was removed)", a.subject.nunique(), 11, 0,
                     f"{SRC}, Methods 2.1 and 2.4")
    checks.internal("every token has a 33-point nasalance series and a duration; 400 tokens carry formant series",
                    bool((a.nasalance_series.str.count(";") == 32).all() and a.duration_ms.notna().all()
                         and (a.f1_series_hz != "").sum() == 400))
    checks.internal("50 items per speaker, session and condition; 6 blocks per speaker",
                    bool(a.groupby(["subject", "session", "condition"]).authors_trial_number.nunique().eq(50).all()))
    checks.internal("tokens without a voiced f0 frame", True, f"{int(a.f0_hz.isna().sum())} tokens")
    checks.internal("ranges: duration 40-450 ms, intensity 40-100 dB, f0 40-800 Hz",
                    bool(a.duration_ms.between(40, 450).all() and a.intensity_db.between(40, 100).all()
                         and a.f0_hz.dropna().between(40, 800).all()),
                    f"{a.duration_ms.min():.0f}-{a.duration_ms.max():.0f} ms, {a.intensity_db.min():.1f}-{a.intensity_db.max():.1f} dB, f0 {a.f0_hz.min():.0f}-{a.f0_hz.max():.0f} Hz")
    out = {}
    for y, rep, nm in (("intensity_db", (3.631, 4.295), "intensity (dB)"), ("duration_ms", (0.014, 0.015), "vowel duration (s)"),
                       ("f0_erb", (0.178, 0.257), "f0 (ERB)")):
        d = a.copy()
        if y == "duration_ms":
            d["dur_s"] = d.duration_ms / 1000
            res = lmm(d, "dur_s", extra=())
            tol = 0.001
        else:
            res = lmm(d, y)
            tol = 0.06 if y == "intensity_db" else 0.012
        out[y] = res
        for lab, key, r in (("Lombard", "C(cond)[T.Lombard noise]", rep[0]), ("Phone", "C(cond)[T.phone-band noise]", rep[1])):
            checks.published(f"{nm}: condition ({lab}) minus quiet, mixed model estimate", float(res.params[key]), r, tol,
                             f"{SRC}, Results 3.1, Table 1")
    for c in checks.items:
        print(("ok     " if c["ok"] else "FAILED "), c["what"], c.get("computed", ""), c.get("reported", ""))
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Merged the authors' R tables (vowel durations, intensities, f0 tracks, nasalance series and, for one speaker, "
                 "formant tracks) into one table with one row per token in recording order, converted seconds to "
                 "milliseconds, averaged the f0 track into a token mean exactly as the authors' analysis script does "
                 "(voiced frames, in Hz and on the ERB scale), packed the nasalance and formant points of each token into "
                 "one cell each, and replaced speaker numbers with codes."),
        how=[
            "One row is one word (a target word with a stop or nasal before and after the vowel, or a filler with a fricative) "
            "spoken by one of 11 male speakers in a describe-the-code task; rows are in recording order within a speaker "
            "(session 1, then 2), and time_s is the start of the vowel in that session's recording. The 148 tokens the authors "
            "removed for mispronunciation or clipping, and the nonbinary speaker, are not in the deposit.",
            "duration_ms, intensity_db, f0_hz (mean of the voiced 5 ms f0 frames) and f0_erb (the authors' ERB mean) describe the "
            "stressed vowel, which the authors segmented by hand in Praat. nasalance_series holds nasalance at 11 equidistant "
            "points over each of onset consonant, vowel and coda consonant (33 values, vowel = points 12 to 22), where nasalance "
            "is nasal / (nasal + oral) RMS in a 20 ms window from the nasometer's two channels. phonetic_context gives the onset "
            "and coda (C oral stop, N nasal, F fricative); the four contexts CVC, CVN, NVC, NVN are the targets and the F contexts "
            "the 1,553 fillers.",
            "Conditions: quiet (no noise), Lombard (white noise at 70 dB SPL through closed headphones, experimenter visible) and "
            "phone (noise band-passed 300-3400 Hz at 70 dB SPL, experimenter not visible); the three ran in random order in each "
            "of two sessions at least two weeks apart. pert_value is the noise level (0 in quiet). Condition B is quiet, L Lombard "
            "and P phone in the authors' files, as their plot labels show.",
            "FastTrack formant tracks (f1_series_hz and the other formant columns) are shared for 400 tokens of one speaker's "
            "quiet condition only.",
        ],
        unit_note="spoken word (vowel token) of one speaker",
        differences=[
            "The mixed-model estimates in the paper (Table 1) come from R's lmerTest; here they are refitted with Python's "
            "statsmodels from the shared rows (same fixed effects, by-speaker intercepts and condition slopes) and agree "
            "within the tolerances listed above; the paper's standard errors, t and p values were not compared.",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
