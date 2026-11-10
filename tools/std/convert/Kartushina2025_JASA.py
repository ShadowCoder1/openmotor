"""Kartushina & Martin (2025), J Acoust Soc Am 157(3):2117-2126: talker variability in visual-feedback articulatory training of the French
vowels /e/ and /E/ by Spanish-speaking women. Isolated-vowel repetition tests (Day 1 before training, Day 3 after, Day 15 maintenance,
familiar and unfamiliar voice) and word tests (CV words with /e/ and /E/ after six consonants, by one new talker).

Raw files, raw_extra/Kartushina2025_JASA/ (OSF my74a, fetched with tools/std/fetch.py), tab-separated, one row per produced token:
  Repetition_All.txt                           isolated vowels, Day 1 and Day 3 (6,612): SUBJECT, SESSION, CONDITION (familiar/unfamiliar voice),
                                               GROUP (High/Low talker variability), VOWEL (ai = /E/, e), TOKEN (1-18 for ai, 19-36 for e), F0-F3 (Hz),
                                               F1-F0, F2-F0, DS, DS_to_voice
  formants_distances_articul2_day_15.txt       the same for Day 15 (1,783; plus a SUBJECTID column that is not carried over)
  formants_distances_articul2_words_day{1,3,15}.txt, ..._day15_sub04.txt   words (1,686 + 1,630 + 934 + 36): CONTEXT instead of CONDITION/TOKEN,
                                               no DS (only DS_to_voice)
  compactness_repetition_clean.txt             the authors' vowel compactness per participant, day, vowel and voice (464 rows)
  analyses_articul2.Rmd, analyses_articul2_words.Rmd   the authors' analyses, read as text. They: drop Sub06 (left the study) and, for the vowel
      tests, Sub44 (no Day 3); set Sub46 and Sub48 to the Low group; remove tokens (vowels: DS missing, F1 >= 900, F2 >= 3000, F0 <= 100,
      F1 >= 800, F2 <= 1500, DS_to_voice >= 12; words: F1 >= 900, F2 >= 3000, F0 <= 100, F1 >= 850, DS_to_voice >= 8; the word script's F1 > 250
      and F2 > 1500 steps are overwritten by the next line and not in effect); model DS_to_voice (log) by session, group, voice and vowel.
      In the word script the corrections of five misspelled Day-15 participant labels are made after the three days were joined, so those
      five participants have no Day-3 partner in the authors' Day 3 vs Day 15 word model (N = 22 instead of 27).
Run from the site folder: python3 -I tools/std/convert/Kartushina2025_JASA.py
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kartushina2025_JASA"
PAPER = "Kartushina & Martin 2025, J Acoust Soc Am"
SCRIPT = "authors' analysis script (OSF Analyses/analyses_articul2.Rmd)"
GROUP = {"High": "high variability (multiple talkers)", "Low": "low variability (single talker)"}
VOWEL = {"ai": "/ɛ/", "e": "/e/"}
SESS = {"Day1": (1, "baseline", "baseline"), "Day3": (2, "test", "transfer"), "Day15": (3, "test", "transfer")}
FIX = {"Subect18": "Sub18", "Sub35_wrep": "Sub35", "Subect09": "Sub09", "Subect17": "Sub17", "Sub37_wordrep": "Sub37"}
WINDOW = "100 ms centered at the vowel midpoint (COLEA)"


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def main():
    raw = stdlib.raw_dir(ID)
    rd = lambda n: pd.read_csv(raw / n, sep="\t", na_values=["NaN"], skipinitialspace=True)
    ck = stdlib.Checks()

    parts = []
    for exp, names in (("vowels", ["Repetition_All.txt", "formants_distances_articul2_day_15.txt"]),
                       ("words", ["formants_distances_articul2_words_day1.txt", "formants_distances_articul2_words_day3.txt",
                                  "formants_distances_articul2_words_day15.txt", "formants_distances_articul2_words_day15_sub04.txt"])):
        for n in names:
            d = rd(n)
            d = d.drop(columns=[c for c in ("SUBJECTID",) if c in d.columns])
            d["exp"], d["src"] = exp, n
            parts.append(d)
    n_src = {p.src.iloc[0]: len(p) for p in parts}
    T = pd.concat(parts, ignore_index=True).rename(columns={"F1-F0": "F1mF0", "F2-F0": "F2mF0"})
    for c in ("TOKEN", "CONDITION", "CONTEXT", "DS"):
        if c not in T.columns:
            T[c] = np.nan
    T["raw_subject"] = T.SUBJECT
    T["person"] = T.SUBJECT.replace(FIX)
    T["GROUP"] = np.where(T.person.isin(["Sub46", "Sub48"]), "Low", T.GROUP)
    T["file_idx"] = np.arange(len(T))

    # ---- the authors' exclusions -----------------------------------------------------------------------------------------------
    v = T.exp == "vowels"
    ok_v = T.DS.notna() & (T.F1 < 900) & (T.F2 < 3000) & (T.F0 > 100) & (T.F1 < 800) & (T.F2 > 1500) & (T.DS_to_voice < 12)
    ok_w = (T.F1 < 900) & (T.F2 < 3000) & (T.F0 > 100) & (T.F1 < 850) & (T.DS_to_voice < 8)
    T["subj_excl"] = np.where(v, T.raw_subject.isin(["Sub06", "Sub44"]), T.raw_subject == "Sub06")
    T["row_ok"] = np.where(v, ok_v, ok_w)
    T["excl"] = T.subj_excl | ~T.row_ok

    # ---- rows, chronological by test day within each experiment, the authors' file order within a day ---------------------------
    T["exp_i"] = (T.exp == "words").astype(int)
    T["sess_i"] = T.SESSION.map(lambda s: SESS[s][0])
    T = T.sort_values(["exp_i", "sess_i", "file_idx"], kind="stable").reset_index(drop=True)
    rows, n_seq = [], {}
    for r in T.itertuples(index=False):
        is_v = r.exp == "vowels"
        sess, ph_v, ph_w = SESS[r.SESSION]
        lab = f"{r.person}|{r.exp}"
        n_seq[lab] = n_seq.get(lab, 0) + 1
        stim = f"isolated vowel {VOWEL[r.VOWEL]}" if is_v else f"CV word: {r.CONTEXT} + {VOWEL[r.VOWEL]}"
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Isolated vowels" if is_v else "Words", "group": GROUP[r.GROUP], "subject": lab,
            "person": r.person, "subject_excluded_by_authors": bool(r.subj_excl), "age": "", "sex": "F", "handedness": "",
            "population": "healthy", "setting": "lab", "device": "", "session": sess, "block": r.SESSION, "trial": n_seq[lab],
            "phase": ph_v if is_v else ph_w, "time_s": "", "task": "speech", "unit": "utterance",
            "condition": r.CONDITION if is_v else "new unfamiliar speaker", "pert_type": "none", "pert_value": 0, "pert_unit": "",
            "feedback": "normal", "excluded_by_authors": bool(r.excl),
            "stimulus": stim, "segment": VOWEL[r.VOWEL], "measure_window": WINDOW,
            "f0_hz": r.F0, "f1_hz": r.F1, "f2_hz": r.F2, "f3_hz": r.F3, "f1_minus_f0_hz": r.F1mF0, "f2_minus_f0_hz": r.F2mF0,
            "ds_to_voice": r.DS_to_voice, "ds": r.DS,
            "authors_trial_number": int(r.TOKEN) if is_v else "", "consonant_context": "" if is_v else r.CONTEXT,
        })
    pcode = {}
    for r in rows:
        pcode.setdefault(r["person"], f"{ID}_P{len(pcode) + 1:04d}")

    # ---- compactness.csv: the authors' per-participant compactness, with person codes ------------------------------------------------
    cm = rd("compactness_repetition_clean.txt")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["person", "group", "block", "voice", "vowel", "compactness"])
    for r in cm.itertuples(index=False):
        w.writerow([pcode[r.SUBJECT], GROUP[r.GROUP], r.SESSION, r.CONDITION, VOWEL[r.VOWEL], stdlib._fmt(float(r.COMPACTNESS), "number")])
    extra = [("compactness.csv", buf.getvalue().encode("utf-8"))]

    # ---- the authors' analysis sets, for the checks -------------------------------------------------------------------------------
    A = T[~T.subj_excl & T.row_ok]
    Av = A[A.exp == "vowels"]
    d1, d3, d15 = (Av[Av.SESSION == s] for s in ("Day1", "Day3", "Day15"))
    learning = pd.concat([d1, d3])
    retention = pd.concat([d3[d3.person.isin(d15.person)], d15])
    Aw = A[A.exp == "words"]  # the word script joins the days with the labels as spelled in the files
    w1, w3, w15 = (Aw[Aw.SESSION == s] for s in ("Day1", "Day3", "Day15"))
    lw = pd.concat([w1[w1.raw_subject.isin(w3.raw_subject)], w3[w3.raw_subject.isin(w1.raw_subject)]])
    rw = pd.concat([w3[w3.raw_subject.isin(w15.raw_subject)], w15[w15.raw_subject.isin(w3.raw_subject)]])

    # ---- internal checks ----------------------------------------------------------------------------------------------------------
    ck.internal("every row of the six data files kept", len(rows) == sum(n_src.values()) == 6612 + 1783 + 1686 + 1630 + 934 + 36,
                f"{len(rows)} rows")
    ck.internal("48 participants in all (Sub01-Sub48), none lost", len(pcode) == 48)
    ck.internal("each of the five misspelled Day-15 word labels is mapped to its participant",
                bool(set(FIX) <= set(T.raw_subject) and T.person[T.raw_subject.isin(list(FIX))].isin(list(FIX.values())).all()))
    two_groups = T.groupby("person").GROUP.nunique()
    ck.internal("each person has one group, except Sub18 whose Day-15 word rows (labelled Subect18) say Low where all other rows say High",
                set(two_groups[two_groups > 1].index) == {"Sub18"}
                and set(T[(T.person == "Sub18") & (T.GROUP == "Low")].raw_subject) == {"Subect18"})
    ck.internal("trial runs 1..n within each subject sequence",
                all(sorted(g) == list(range(1, len(g) + 1)) for g in pd.Series([r["trial"] for r in rows]).groupby([r["subject"] for r in rows]).agg(list)))
    ck.internal("f1_minus_f0_hz and f2_minus_f0_hz differ from f1_hz - f0_hz and f2_hz - f0_hz by at most 1 Hz",
                bool(((T.F1mF0 - (T.F1 - T.F0)).abs().dropna() <= 1).all() and ((T.F2mF0 - (T.F2 - T.F0)).abs().dropna() <= 1).all()))
    ck.internal("compactness: 464 rows of 46 participants, all in the main table", len(cm) == 464 and cm.SUBJECT.nunique() == 46
                and set(cm.SUBJECT) <= set(pcode))

    # ---- published numbers --------------------------------------------------------------------------------------------------------
    ck.published("isolated vowels, Day 1 vs Day 3: participants", learning.person.nunique(), 46, 0, f"{PAPER}, Results, Model 1 (N = 46)")
    ck.published("isolated vowels, Day 1 vs Day 3: tokens analysed", len(learning), 6370, 0, f"{PAPER}, Results, Model 1 (6370 observations)")
    ck.published("isolated vowels, Day 3 vs Day 15: participants", retention.person.nunique(), 25, 0, f"{PAPER}, Results, Model 3 (N = 25)")
    ck.published("isolated vowels, Day 3 vs Day 15: tokens analysed", len(retention), 3506, 0, f"{PAPER}, Results, Model 3 (3506 observations)")
    ck.published("participants per training group", int(learning.groupby("GROUP").person.nunique().min()), 23, 0, f"{PAPER}, Methods (n = 23 in each group)")
    ck.published("returned for Day 15, multiple-talker group", int(d15[d15.GROUP == "High"].person.nunique()), 13, 0, f"{PAPER}, Methods (Day 15)")
    ck.published("returned for Day 15, single-talker group", int(d15[d15.GROUP == "Low"].person.nunique()), 12, 0, f"{PAPER}, Methods (Day 15)")
    ck.published("word tests, Day 1 vs Day 3: participants", lw.raw_subject.nunique(), 45, 0, f"{PAPER}, Results, Model 2 (N = 45)")
    ck.published("word tests, Day 1 vs Day 3: tokens analysed", len(lw), 3055, 0, f"{PAPER}, Results, Model 2 (3055 observations)")
    ck.published("word tests, Day 3 vs Day 15: participants (labels as spelled in the files)", rw.raw_subject.nunique(), 22, 0,
                 f"{PAPER}, Results, Model 4 (N = 22)")
    ck.published("word tests, Day 3 vs Day 15: tokens analysed", len(rw), 1488, 0, f"{PAPER}, Results, Model 4 (1488 observations)")
    sm = lw.groupby(["raw_subject", "SESSION"]).DS_to_voice.mean().groupby("SESSION").agg(["mean", "std"])
    for s_, (m_r, sd_r) in (("Day1", (2.78, 0.62)), ("Day3", (2.47, 0.57))):
        ck.published(f"words, {s_}: mean distance score over participants", sm.loc[s_, "mean"], m_r, 0.015, f"{PAPER}, Results, Model 2")
        ck.published(f"words, {s_}: SD over participants", sm.loc[s_, "std"], sd_r, 0.015, f"{PAPER}, Results, Model 2")
    sr = rw.groupby(["raw_subject", "SESSION"]).DS_to_voice.mean().groupby("SESSION").agg(["mean", "std"])
    for s_, (m_r, sd_r) in (("Day3", (2.56, 0.55)), ("Day15", (2.77, 0.83))):
        ck.published(f"words, {s_} (retention sample): mean over participants", sr.loc[s_, "mean"], m_r, 0.015, f"{PAPER}, Results, Model 4")
        ck.published(f"words, {s_} (retention sample): SD over participants", sr.loc[s_, "std"], sd_r, 0.015, f"{PAPER}, Results, Model 4")
    # the descriptive table of Model 1: mean (SD) over participants of each participant's mean distance score per cell
    cell = learning.groupby(["person", "GROUP", "SESSION", "CONDITION", "VOWEL"]).DS_to_voice.mean().reset_index()
    tab = cell.groupby(["GROUP", "CONDITION", "VOWEL", "SESSION"]).DS_to_voice.agg(["mean", "std"])
    P1 = {("Low", "familiar", "e"): ((3.46, 0.98), (2.28, 0.97)), ("Low", "familiar", "ai"): ((4.47, 1.48), (3.19, 1.74)),
          ("High", "familiar", "e"): ((3.20, 1.61), (2.94, 1.58)), ("High", "familiar", "ai"): ((4.07, 1.91), (3.15, 1.42)),
          ("Low", "unfamiliar", "e"): ((2.61, 0.71), (1.92, 0.89)), ("Low", "unfamiliar", "ai"): ((3.53, 1.28), (2.44, 1.51)),
          ("High", "unfamiliar", "e"): ((2.42, 0.76), (1.92, 0.68)), ("High", "unfamiliar", "ai"): ((2.90, 1.12), (2.31, 0.83))}
    for (g, c, vw), (pre, post) in P1.items():
        for s_, (m_r, sd_r) in (("Day1", pre), ("Day3", post)):
            lab = f"{'single' if g == 'Low' else 'multiple'}-talker, {c} voice, {VOWEL[vw]}, {s_}"
            ck.published(f"{lab}: mean distance score", tab.loc[(g, c, vw, s_), "mean"], m_r, 0.015, f"{PAPER}, Results, Model 1 (descriptive table)")
            ck.published(f"{lab}: SD", tab.loc[(g, c, vw, s_), "std"], sd_r, 0.015, f"{PAPER}, Results, Model 1 (descriptive table)")
    fam = learning[learning.CONDITION == "familiar"].groupby(["person", "GROUP", "SESSION"]).DS_to_voice.mean().groupby(["GROUP", "SESSION"]).mean()
    for g, rep in (("Low", 31), ("High", 16)):
        ck.published(f"improvement in the familiar voice, {'single' if g == 'Low' else 'multiple'}-talker group (% of Day 1)",
                     100 * (fam[(g, "Day1")] - fam[(g, "Day3")]) / fam[(g, "Day1")], rep, 0.5, f"{PAPER}, Results, Model 1")
    # the authors' own counts and percentages in their script comments
    v_all = T[(T.exp == "vowels") & ~T.raw_subject.isin(["Sub06", "Sub44"])]
    ck.published("vowel tokens without a distance score removed", int(v_all.DS.isna().sum()), 7, 0, f"{SCRIPT}, 'Remove NAs in DS (n=7)'")
    ck.published("vowel tokens with F1 >= 900 removed (after the missing-DS step)", int((v_all.DS.notna() & ~(v_all.F1 < 900)).sum()), 52, 0,
                 f"{SCRIPT}, comment 'number of removed trials 52'")
    ck.published("share of isolated-vowel tokens excluded, %", 100 * (1 - Av.shape[0] / len(v_all)), 2.4, 0.1, f"{SCRIPT}, comment '~2.4% of data'")
    w_all = T[(T.exp == "words") & (T.raw_subject != "Sub06")]
    ck.published("share of word tokens excluded, %", 100 * (1 - len(Aw) / len(w_all)), 5, 0.3, "authors' analysis script (OSF Analyses/analyses_articul2_words.Rmd), comment '~5% of data'")
    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)

    s, paper = paper_meta()
    n_rows_ex = int(T.excl.sum())
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the six token files into one table (formants in Hz, distance scores as shared), day by day; participant "
                 "labels replaced by codes, the five misspelled Day-15 word labels mapped to their participants, the group of Sub46 and "
                 "Sub48 set to low variability as the authors' script does; the authors' vowel compactness is in compactness.csv; "
                 "the SUBJECTID column (labels typed by hand) is not carried over."),
        how=[
            "One row is one vowel token produced by one of 48 Spanish-speaking women trained on the French vowels /e/ and /E/ with "
            "visual feedback (low variability: one French talker; high: five talkers), either an isolated vowel repeated after a "
            "French talker (experiment Isolated vowels; voice familiar or unfamiliar; segment /ɛ/ is the authors' 'ai') or the "
            "vowel of a French CV word repeated after one new talker (experiment Words, consonant_context m, l, d, t, s or g). "
            "block is the test day (Day 1 before training, Day 3 after, Day 15 maintenance); no feedback was given at test.",
            "f0_hz, f1_hz, f2_hz and f3_hz are the authors' values in Hz, averaged over 100 ms centered at the vowel midpoint (the paper "
            "describes F0 to F2 only). ds_to_voice is the authors' distance score from the native French target vowel space (lower is "
            "closer), used in all their models; ds is a second distance score that only the isolated-vowel files have; "
            "f1_minus_f0_hz and f2_minus_f0_hz are the F0-adjusted axes of that space. authors_trial_number is their TOKEN (1-18 for "
            "/ɛ/, 19-36 for /e/ in each voice).",
            f"Rows are in the authors' file order within each test day, with the days in time order; within a day the files are not in "
            f"the order of production. excluded_by_authors marks the {n_rows_ex} rows their scripts remove (missing values and "
            "implausible F0, F1, F2 or distance scores) together with the rows of the participants they drop (Sub06, left the study; "
            "Sub44, no Day 3, isolated vowels only); the rest are their analysed tokens.",
            "compactness.csv has the authors' vowel compactness (the inverse of the spread of a participant's vowel tokens, computed "
            "from the cleaned isolated-vowel data) per participant, day, voice and vowel.",
        ],
        unit_note="vowel token (utterance) of one participant",
        differences=[
            "The authors' Day 3 vs Day 15 word model has 22 participants (1,488 tokens) although 27 took the Day-15 word test: five Day-15 "
            "labels are misspelled in the word file (their script corrects them only after the days are joined), so those participants "
            "have no Day-3 partner there. Here they are mapped to their participants, and the 22 is reproduced from the labels as spelled.",
            "The Day-15 word rows labelled Subect18 are attached to Sub18 (the authors' own correction) but carry the group Low, "
            "while every other row of Sub18 says High; the label is kept as shared.",
            "The counts of removed tokens in the comments of the authors' scripts are from an earlier version of the data: with the files "
            "shared, the vowel steps remove 7 (missing distance), 52, 63, 50, 10, 3 and 10 tokens (comments: 7, 52, 64, 2%, 26, 8, 35), "
            "although the analysis sets reproduce the paper's token counts exactly.",
            "The paper's descriptive table (Model 1) is matched to within 0.015 (the shared tokens give means up to 0.01 higher in some cells).",
        ],
    )
    if not ck.passed():
        sys.exit("checks failed")
    return stdlib.release(ID, "speech", rows, meta, ck, extra_files=extra)


if __name__ == "__main__":
    main()
