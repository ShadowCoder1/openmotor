"""Shamei, Soskuthy, Stavness & Gick (2023), Sci Rep 13:8231: vowel formants in the speech of four Space Shuttle captains
before a flight and right after landing (post-flight tarmac interview), from publicly available NASA recordings.

Raw files, raw_extra/Shamei2023_SciRep/ (OSF mgyfd, fetched with tools/std/fetch.py):
  space_vowels.csv   one row per stressed-vowel token (799), 48 columns: speaker name, the word with its neighbours, the vowel,
      F1/F2/F3 and bandwidths at the extraction software's measurement point, F1 and F2 at 20, 35, 50, 65 and 80% of the vowel,
      midF1 and midF2 (the mean of the 35, 50 and 65% values: the variables of the authors' models), Condition (Preflight or
      Postflight), height and backness classes
  GroupMean2.csv     the authors' mean mel F1 and F2 per vowel and condition (20 cells)
  space_analysis_sci_rep.Rmd   the authors' analysis, read as text: mel(): emuR's mel, 1000 * log2(1 + f / 1000) (verified below
      against GroupMean2.csv); lmer models of midF1 and midF2 on Condition with height or backness; no row is excluded
Run from the site folder: python3 -I tools/std/convert/Shamei2023_SciRep.py
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Shamei2023_SciRep"
PAPER = "Shamei et al. 2023, Sci Rep"
IPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AY": "a", "EH": "ɛ", "EY": "e", "IH": "ɪ", "IY": "i", "OW": "o", "UW": "u"}
# Table 1 of the paper: tokens per vowel, Total Pre / Total Post columns
T1_VOWEL = {"AA": (21, 22), "AE": (30, 51), "AH": (35, 42), "AY": (29, 33), "EH": (48, 57), "EY": (43, 59), "IH": (69, 62),
            "IY": (37, 45), "OW": (29, 31), "UW": (37, 19)}
# Table 1: tokens per speaker (Pre, Post); the speakers are matched by these counts, no names are written here
T1_SPEAKER = [(115, 110), (99, 95), (69, 75), (95, 141)]
CONDITION = {"Preflight": dict(session=1, phase="baseline"), "Postflight": dict(session=2, phase="aftereffect")}
DEVICE = "publicly available NASA recordings (pre-flight interview; post-flight tarmac interview)"
WINDOW = "mean of the values at 35%, 50% and 65% of the vowel's duration"
# Personal names said in the interviews (colleagues, crew members, and the participants' own names, which are taken from the
# data) are replaced by [name] in the word columns of the main table and of authors_table.csv
OTHER_NAMES = {"BARRY", "BOB", "CABANA", "CHARLIE", "DOUG", "HURLEY", "NICOLE", "REX", "SANDY", "WILMORE"}


def mel(f):
    return 1000 * np.log2(1 + np.asarray(f, dtype=float) / 1000)


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def main():
    raw = stdlib.raw_dir(ID)
    d = pd.read_csv(raw / "space_vowels.csv", encoding="utf-8-sig")
    txt = pd.read_csv(raw / "space_vowels.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    gm = pd.read_csv(raw / "GroupMean2.csv", encoding="utf-8-sig", index_col=0)
    ck = stdlib.Checks()
    names = OTHER_NAMES | {str(n).upper() for n in d["name"].unique()}
    word_cols = [("pre_word", "pre_word_trans"), ("word", "word_trans"), ("fol_word", "fol_word_trans")]
    n_masked = 0
    for wc, tc in word_cols:
        hit = txt[wc].str.upper().isin(names)
        n_masked += int(hit.sum())
        txt.loc[hit, [wc, tc]] = "[name]"
    d["word_masked"] = txt["word"].values

    rows = []
    for r in d.itertuples(index=False):
        cond = CONDITION[r.Condition]
        rows.append({
            "dataset": ID, "study": ID, "experiment": "", "group": "", "subject": r.name, "subject_excluded_by_authors": False,
            "age": "", "sex": "M", "handedness": "", "population": "healthy astronauts", "setting": "", "device": DEVICE,
            "session": cond["session"], "block": r.Condition, "trial": "", "phase": cond["phase"], "time_s": float(r.beg),
            "task": "speech", "unit": "utterance", "condition": r.Condition, "pert_type": "none", "pert_value": 0, "pert_unit": "",
            "feedback": "normal", "excluded_by_authors": False,
            "stimulus": r.word_masked.lower() if r.word_masked != "[name]" else "[name]", "segment": f"stressed vowel /{IPA[r.vowel]}/ ({r.vowel})", "measure_window": WINDOW,
            "duration_ms": round(float(r.dur) * 1000, 6), "f1_hz": float(r.midF1), "f2_hz": float(r.midF2),
            "vowel_height": r.height, "vowel_backness": r.backness,
        })
    # trial: order of the row within the participant's sequence (both recordings), from 1
    n = {}
    for r in rows:
        n[r["subject"]] = n.get(r["subject"], 0) + 1
        r["trial"] = n[r["subject"]]
    df = pd.DataFrame(rows)

    # participant codes as stdlib.release will assign them (order of first appearance), for the extra table
    pcode = {}
    for r in rows:
        pcode.setdefault(r["subject"], f"{ID}_P{len(pcode) + 1:04d}")
    scode = {k: v.replace("_P", "_S") for k, v in pcode.items()}

    # the authors' table with all its columns, the speaker name replaced by the person code
    ex = txt.copy()
    ex.insert(0, "person", [pcode[v] for v in ex["name"]])
    ex = ex.drop(columns=["name"])
    for c in ex.columns:  # identifier patterns in the text columns (long numbers and two long English words look like codes)
        if pd.to_numeric(ex[c], errors="coerce").notna().sum() > 0.5 * len(ex):
            continue
        vals = ex[c].astype(str).replace({"ACCOMPLISHED": "", "ACCOMPLISHING": ""}, regex=False)
        for pn, pat in stdlib.ID_PATTERNS.items():
            assert not vals.str.contains(pat).any(), (c, pn)
    buf = io.StringIO()
    ex.to_csv(buf, index=False, lineterminator="\r\n")
    extra = [("authors_table.csv", buf.getvalue().encode("utf-8"))]

    # ---- internal checks ---------------------------------------------------------------------------------------------
    ck.internal("every row of space_vowels.csv kept, in the file's order", len(df) == len(d) == 799, f"{len(df)} rows")
    ck.internal("4 participants, none lost", df.subject.nunique() == 4)
    ck.internal("midF1 and midF2 are the means of the 35, 50 and 65% values on every row",
                bool(np.allclose(d.midF1, d[["F1_35", "F1_50", "F1_65"]].mean(axis=1), atol=1e-9)
                     and np.allclose(d.midF2, d[["F2_35", "F2_50", "F2_65"]].mean(axis=1), atol=1e-9)))
    ck.internal("dur equals end minus beg on every row", bool(np.allclose(d.dur, d.end - d.beg, atol=1e-9)))
    ck.internal("vowel time (beg) increases within each participant and recording",
                bool(all(g.beg.is_monotonic_increasing for _, g in d.groupby(["name", "Condition"]))))
    ck.internal("f1_hz and f2_hz present on every row and in range", bool(df.f1_hz.between(250, 900).all() and df.f2_hz.between(700, 2600).all()),
                f"F1 {df.f1_hz.min():.0f}-{df.f1_hz.max():.0f}, F2 {df.f2_hz.min():.0f}-{df.f2_hz.max():.0f}")
    ck.internal("personal names in the word columns are masked", n_masked > 0
                and not any(txt[wc].str.upper().isin(names).any() for wc, _ in word_cols), f"{n_masked} words replaced by [name]")
    ck.internal("height and backness classes are constant within each vowel", bool((d.groupby("vowel").height.nunique() == 1).all()
                                                                                   and (d.groupby("vowel").backness.nunique() == 1).all()))

    # ---- published numbers -------------------------------------------------------------------------------------------
    ck.published("astronauts", d.name.nunique(), 4, 0, f"{PAPER}, Materials and methods (four captains)")
    ck.published("vowel tokens, preflight", int((d.Condition == "Preflight").sum()), 378, 0, f"{PAPER}, Table 1 (sum of Total Pre)")
    ck.published("vowel tokens, postflight", int((d.Condition == "Postflight").sum()), 421, 0, f"{PAPER}, Table 1 (sum of Total Post)")
    for v, (pre, post) in T1_VOWEL.items():
        for cnd, rep in (("Preflight", pre), ("Postflight", post)):
            ck.published(f"tokens of /{IPA[v]}/, {cnd.lower()}", int(((d.vowel == v) & (d.Condition == cnd)).sum()), rep, 0,
                         f"{PAPER}, Table 1")
    got = sorted((int(((d.name == nm) & (d.Condition == "Preflight")).sum()), int(((d.name == nm) & (d.Condition == "Postflight")).sum()))
                 for nm in d.name.unique())
    for i, (g_, e_) in enumerate(zip(got, sorted(T1_SPEAKER))):
        ck.published(f"tokens of speaker {i + 1} (counts sorted), preflight", g_[0], e_[0], 0, f"{PAPER}, Table 1 (Pre columns)")
        ck.published(f"tokens of speaker {i + 1} (counts sorted), postflight", g_[1], e_[1], 0, f"{PAPER}, Table 1 (Post columns)")
    # the authors' group means (GroupMean2.csv): mean of mel(midF1) and mel(midF2) per vowel and condition
    d["vipa"] = d.vowel.map(IPA)
    m1 = pd.Series(mel(d.midF1)).groupby([d.Condition, d.vipa]).mean()
    m2 = pd.Series(mel(d.midF2)).groupby([d.Condition, d.vipa]).mean()
    key = gm.set_index(["Condition", "vowel"])
    worst = 0.0
    for (cnd, v) in m1.index:
        for nm, ser, col in (("F1", m1, "mean_mel_F1"), ("F2", m2, "mean_mel_F2")):
            rep = float(key.loc[(cnd, v), col])
            ck.published(f"mean mel {nm} of /{v}/, {cnd.lower()}", ser[(cnd, v)], rep, 0.001, "authors' deposit file GroupMean2.csv")
            worst = max(worst, abs(ser[(cnd, v)] - rep))
    ck.internal("the 20 vowel x condition cells of GroupMean2.csv are all reproduced (mel = 1000 log2(1 + f / 1000))",
                m1.shape[0] == 20 and len(gm) == 20 and worst < 0.001, f"largest difference {worst:.2e} mel")
    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)

    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Re-coded the authors' vowel table into the OpenMotor speech format (formant values, durations and times as shared; "
                 "duration in ms instead of seconds); speaker names replaced by codes; the authors' full table, with every column "
                 "they shared, is also in authors_table.csv."),
        how=[
            "One row is one stressed vowel token spoken by one of four male Space Shuttle captains in a public NASA interview, in "
            "the authors' order (the word is in lower case here, and personal names said in the interviews are replaced by [name]): the recording before the flight (Preflight, session 1, phase baseline) or right after landing "
            "(Postflight, session 2, phase aftereffect; the flights lasted 10 to 14 days in microgravity). No perturbation is in "
            "force during either recording, so pert_type is none; time_s is the vowel's start in its recording.",
            "f1_hz and f2_hz are the authors' midF1 and midF2: the mean of the formant values at 35%, 50% and 65% of the vowel, in Hz; "
            "the authors convert them to mel with emuR's formula, mel = 1000 * log2(1 + f / 1000) (checked against their group "
            "means here). vowel_height and vowel_backness are the authors' classes for their models.",
            "authors_table.csv holds the file as shared (formants and bandwidths at the extraction software's measurement point, "
            "F1 and F2 at 20, 35, 50, 65 and 80% of the vowel, the Lobanov-normalized values, neighbouring words and "
            "transcriptions, phonological context codes) with person in place of the speaker name and [name] for personal names. The paper names the recordings "
            "as public NASA material; nothing else is known about the settings.",
        ],
        unit_note="vowel token (utterance) of one astronaut",
        differences=[],
    )
    if not ck.passed():
        sys.exit("checks failed")
    rec = stdlib.release(ID, "speech", rows, meta, ck, extra_files=extra)
    # the extra table must carry the same person codes as the main table
    with zipfile.ZipFile(stdlib.OUT / f"{ID}.zip") as z:
        main_rows = list(csv.DictReader(io.TextIOWrapper(z.open(f"{ID}/{ID}.csv"), encoding="utf-8", newline="")))
        auth = list(csv.DictReader(io.TextIOWrapper(z.open(f"{ID}/authors_table.csv"), encoding="utf-8", newline="")))
    assert [r["person"] for r in main_rows] == [r["person"] for r in auth], "person codes of authors_table.csv do not match"
    assert [r["subject"] for r in main_rows] == [scode[v] for v in d.name], "subject codes do not match"
    import re
    with zipfile.ZipFile(stdlib.OUT / f"{ID}.zip") as z:
        everything = " ".join(z.read(nm).decode("utf-8", "replace") for nm in z.namelist())
    for nm in sorted(names):
        assert not re.search(r"\b" + re.escape(nm) + r"\b", everything, re.I), "a name is still in the ZIP"
    print("authors_table.csv person codes match the main table; no personal name left in the ZIP")
    return rec


if __name__ == "__main__":
    main()
