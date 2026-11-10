"""Belz, Rasskazova, Krivokapic & Mooshammer (2022), Language and Speech 66(1):3-34: phrase-final lengthening of tense and lax
vowels in German monosyllabic and disyllabic words, 10 speakers, acoustic durations of the syllable constituents and electromagnetic
articulography (EMA) of the closing and opening movement toward the last consonant of the stressed syllable.

Raw files, raw_extra/Belz2022_LangSpeech/ (Zenodo 4896137, CC BY 4.0, fetched with tools/std/fetch.py):
  mono.txt, di.txt          emuR segment lists, one row per syllable constituent: monosyllabic words O (onset), N (nucleus), C (coda),
      disyllabic words O1, N1, O2, N2; start/end in ms of the constituent's first segment, sylposdur = the authors' constituent duration
      (stop closure and aspiration together), sylseglabel its phonetic label, pausedur the following pause, position the phrasal
      position the speaker produced, bundle the authors' recording name (DP1_<word>_<iteration>_<running number>); the other emuR columns
      (db_uuid, session, item ids, sample positions) are bookkeeping and are not carried over
  mono.art.txt, di.art.txt  MVIEW labels of one gesture per token: SOURCE (recording name), TRAJ (TTPOS_XZ tongue tip, LA lip aperture),
      times (ms), x/z positions (mm) and speeds (cm/s) at GONS PVEL NONS MAXC NOFFS PVEL2 GOFFS, and the authors' derived platdur,
      opdur, closdur (ms), closdisp, opdisp (mm)
  meta.speech.dur.txt       per speaker: sprechzeit (speaking time, minutes), tokens, syllables, sprechgeschw (syllables per second)
  analysis.R                the authors' analysis, read as text (exclusion rules: opdisp > 1 & closdisp > 1, closdur > 15 for di)
What was checked in the data: DP1 recordings are sentences meant to put the word phrase-medially and DP2 phrase-finally (the tokens whose
produced position differs are the 10 + 6 of the paper's Table 2); closdur = NONS - GONS, platdur = NOFFS - NONS, opdur = GOFFS - NOFFS;
closdisp and opdisp are the distances between the positions at GONS and MAXC, and MAXC and GOFFS.
Run from the site folder: python3 -I tools/std/convert/Belz2022_LangSpeech.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Belz2022_LangSpeech"
SRC = "Belz et al. 2022, Language and Speech"
DEVICE = ("Carstens AG 501 electromagnetic articulograph (1250 Hz, downsampled to 250 Hz), shotgun microphone in front of the speaker")
LM = ["GONS", "PVEL", "NONS", "MAXC", "NOFFS", "PVEL2", "GOFFS"]
EXP = {"mono": "monosyllabic target words", "di": "disyllabic target words"}
RX = re.compile(r"^DP(?P<dp>\d)_(?P<w>[A-Za-z]+)_(?P<it>\d+X?)(?:_(?P<tag>[A-Za-z]+)_)?_(?P<n>\d+)$")
DOUBLED = re.compile(r"^(DP\d_[A-Za-z]+_\d+X?(?:_[A-Za-z]+_)?_\d+)\1$")
SEX = {"f": "F", "w": "F", "m": "M"}


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def fix_name(name):
    """One EMA recording name is written twice in a row in the deposit (DP1_Ruhm_3_0143DP1_Ruhm_3_0143)."""
    m = DOUBLED.match(name)
    return m.group(1) if m else name


def speaker(label):
    """Speaker label of the acoustic tables (f1..f5, m1..m5); the EMA tables write w for the women."""
    return "f" + label[1:] if label[0] == "w" else label


def read(raw, name):
    d = pd.read_csv(raw / name, sep="\t")  # R's write.table: the header has one name fewer than the rows, pandas uses the row names as index
    return d


def acoustic_tokens(tab, kind):
    out = {}
    for r in tab.to_dict("records"):
        key = (r["subject"], r["bundle"])
        t = out.setdefault(key, dict(kind=kind, subj=r["subject"], name=r["bundle"], position=r["position"], word=r["word"],
                                     iteration=str(r["iteration"]), tenseness=r["tenseness"], pause=float(r["pausedur"]), cons={}))
        for k, c in (("position", "position"), ("word", "word"), ("tenseness", "tenseness")):
            assert t[k] == r[c], (key, k)
        assert t["pause"] == float(r["pausedur"]) and t["iteration"] == str(r["iteration"]), key
        assert r["sylpos"] not in t["cons"], (key, r["sylpos"])
        t["cons"][r["sylpos"]] = dict(dur=float(r["sylposdur"]), label=r["sylseglabel"], start=float(r["start"]), end=float(r["end"]),
                                      closure=float(r["labelsdur"]), segment=r["labels"])
    return out


def ema_tokens(tab, kind):
    out = {}
    for r in tab.to_dict("records"):
        name = fix_name(r["SOURCE"])
        key = (speaker(r["subject"]), name)
        assert key not in out, key
        out[key] = dict(kind=kind, subj=key[0], name=name, position=r["position"], word=r["word"], tenseness=r["tenseness"],
                        sex=r["sex"], rec=r)
    return out


def build(raw):
    mono, di = read(raw, "mono.txt"), read(raw, "di.txt")
    ma, da = read(raw, "mono.art.txt"), read(raw, "di.art.txt")
    meta = read(raw, "meta.speech.dur.txt")
    assert mono.shape == (2352, 27) and di.shape == (1567, 27) and ma.shape == (784, 42) and da.shape == (393, 42), (mono.shape, di.shape, ma.shape, da.shape)
    n_art_rows = len(ma) + len(da)
    cons_label = {}
    for tab, sp in ((mono, "C"), (di, "O2")):
        for w, g in tab[tab.sylpos == sp].groupby("word"):
            assert g.labels.nunique() == 1
            cons_label[w] = g.labels.iloc[0]
    tokens = {}
    for kind, ac_tab, em_tab in (("mono", mono, ma), ("di", di, da)):
        ac, em = acoustic_tokens(ac_tab, kind), ema_tokens(em_tab, kind)
        for k in list(ac) + [k for k in em if k not in ac]:
            tokens[k] = dict(kind=kind, ac=ac.get(k), em=em.get(k))
    # recording-name parts and the running number of each token (its order in the speaker's recordings)
    nums = {}
    for k, t in tokens.items():
        m = RX.match(k[1])
        assert m, k
        t["dp"], t["it"], t["tag"], t["n"] = int(m["dp"]), m["it"], m["tag"], int(m["n"])
        nums.setdefault(k[0], []).append(t["n"])
    assert all(len(v) == len(set(v)) for v in nums.values()), "running numbers are not unique within a speaker"
    rank = {s: {n: i for i, n in enumerate(sorted(v), start=1)} for s, v in nums.items()}

    rows = []
    for k, t in tokens.items():
        ac, em, kind = t["ac"], t["em"], t["kind"]
        ref = ac if ac is not None else em
        pos, word, ten = ref["position"], ref["word"], ref["tenseness"]
        row = dict(
            dataset=ID, study=SRC, experiment=EXP[kind], group="", subject=k[0], person=k[0], subject_excluded_by_authors=False,
            age=None, sex=SEX[k[0][0]], handedness=None, population="healthy", setting="lab", device=DEVICE, session=1,
            block=f"iteration {t['it']}", trial=rank[k[0]][t["n"]], phase="other", time_s=None, task="speech", unit="utterance",
            condition=f"{ten} vowel, {pos}", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=False,
            stimulus=word, gesture_segment=cons_label[word],
            phrasal_position=pos, intended_phrasal_position={1: "phrase-medial", 2: "phrase-final"}[t["dp"]], vowel_tenseness=ten,
            recording_name=k[1])
        if ac is not None:
            c = ac["cons"]
            names = ("O", "N", "C") if kind == "mono" else ("O1", "N1", "O2", "N2")
            assert set(c) <= set(names) and len(c) >= len(names) - 1, (k, sorted(c))
            first, order = c[names[0]], [c[n] for n in names if n in c]
            if kind == "mono":
                row.update(onset_ms=c["O"]["dur"], nucleus_ms=c["N"]["dur"], coda_ms=c["C"]["dur"])
            else:
                row.update(onset_ms=c["O1"]["dur"], nucleus_ms=c["N1"]["dur"], onset2_ms=c["O2"]["dur"],
                           nucleus2_ms=c["N2"]["dur"] if "N2" in c else None)
            starts = [x["start"] for x in order]
            row.update(pause_after_ms=ac["pause"], word_start_ms=starts[0],
                       word_end_ms=max(x["start"] + x["dur"] for x in order),
                       segment_labels=";".join(x["label"] for x in order))
            t["gap"] = max(abs(order[i + 1]["start"] - (order[i]["start"] + order[i]["dur"])) for i in range(len(order) - 1))
        if em is not None:
            r = em["rec"]
            ex = bool(r["opdisp"] <= 1 or r["closdisp"] <= 1 or (kind == "di" and r["closdur"] <= 15))
            row.update(gesture_signal=r["TRAJ"], gesture_order=1, gesture_excluded_by_authors=ex,
                       closing_duration_ms=r["closdur"], plateau_duration_ms=r["platdur"], opening_duration_ms=r["opdur"],
                       closing_displacement_mm=r["closdisp"], opening_displacement_mm=r["opdisp"])
            for L in LM:
                lo = L.lower()
                row[f"{lo}_ms"] = r[f"{L}..ms."]
                row[f"{lo}_x_mm"] = r[f"{L}..X.mm."]
                row[f"{lo}_z_mm"] = r[f"{L}..Z.mm."]
                row[f"{lo}_v_cm_s"] = r[f"{L}..V.cm.sec."]
            row["excluded_by_authors"] = bool(ac is None and ex)
            row["_sex_em"] = SEX[r["sex"]]
        rows.append(row)
    return rows, tokens, meta, n_art_rows, (mono, di, ma, da)


def main():
    raw = stdlib.raw_dir(ID)
    rows, tokens, meta_tab, n_art_rows, (mono, di, ma, da) = build(raw)
    sex_ok = all(r.pop("_sex_em", r["sex"]) == r["sex"] for r in rows)
    df = pd.DataFrame(rows)
    rate = dict(zip(meta_tab.subject, meta_tab.sprechgeschw))
    mono_df, di_df = df[df.experiment == EXP["mono"]], df[df.experiment == EXP["di"]]
    ck = stdlib.Checks()
    S21, S24, S311, S312 = "Section 2.1", "Section 2.4", "Section 3.1.1 (Monosyllabic target words)", "Section 3.1.2 (Disyllabic target words)"

    # ---- published numbers ---------------------------------------------------------------------------------------------------
    pers = df.drop_duplicates("person")
    ck.published("speakers", len(pers), 10, 0, f"Abstract; {S21}")
    ck.published("male speakers", int((pers.sex == "M").sum()), 5, 0, S21)
    ck.published("female speakers", int((pers.sex == "F").sum()), 5, 0, S21)
    ck.published("monosyllabic syllable onsets in the acoustic analysis", int(mono_df.onset_ms.notna().sum()), 784, 0, S24)
    ck.published("monosyllabic syllable nuclei in the acoustic analysis", int(mono_df.nucleus_ms.notna().sum()), 784, 0, S24)
    ck.published("monosyllabic syllable codas in the acoustic analysis", int(mono_df.coda_ms.notna().sum()), 784, 0, S24)
    ck.published("disyllabic first-syllable onsets", int(di_df.onset_ms.notna().sum()), 392, 0, S24)
    ck.published("disyllabic second-syllable onsets", int(di_df.onset2_ms.notna().sum()), 392, 0, S24)
    ck.published("disyllabic first-syllable nuclei", int(di_df.nucleus_ms.notna().sum()), 392, 0, S24)
    ck.published("disyllabic second-syllable nuclei (schwa elided once in Miete)", int(di_df.nucleus2_ms.notna().sum()), 391, 0, S24)
    n_cons = int(sum(df[c].notna().sum() for c in ("onset_ms", "nucleus_ms", "coda_ms", "onset2_ms", "nucleus2_ms")))
    ck.published("acoustic data excluded (% of 2,400 + 1,600 constituents)", 100 * (1 - n_cons / 4000), 2.02, 0.01, S24)
    ac_rows = df[df.onset_ms.notna()]
    ck.published("Table 2: target words analysed, phrase-medial", int((ac_rows.phrasal_position == "phrase-medial").sum()), 582, 0, "Table 2")
    ck.published("Table 2: target words analysed, phrase-final", int((ac_rows.phrasal_position == "phrase-final").sum()), 594, 0, "Table 2")
    ck.published("Table 2: target words analysed, total", len(ac_rows), 1176, 0, "Table 2")
    ck.published("Table 2: unexpected boundaries (phrase-medial target produced phrase-final)",
                 int(((ac_rows.intended_phrasal_position == "phrase-medial") & (ac_rows.phrasal_position == "phrase-final")).sum()), 10, 0,
                 "Table 2, row c")
    ck.published("Table 2: no noticeable boundary (phrase-final target produced phrase-medial)",
                 int(((ac_rows.intended_phrasal_position == "phrase-final") & (ac_rows.phrasal_position == "phrase-medial")).sum()), 6, 0,
                 "Table 2, row b")
    em_rows = df[df.gesture_signal.notna()]
    em_mono, em_di = em_rows[em_rows.experiment == EXP["mono"]], em_rows[em_rows.experiment == EXP["di"]]
    ck.published("monosyllabic stimuli in the articulatory analysis", int((~em_mono.gesture_excluded_by_authors.astype(bool)).sum()), 748, 0, S24)
    ck.published("disyllabic stimuli in the articulatory analysis", int((~em_di.gesture_excluded_by_authors.astype(bool)).sum()), 385, 0, S24)
    ck.published("monosyllabic stimuli excluded for a displacement below 1 mm", int(em_mono.gesture_excluded_by_authors.astype(bool).sum()), 36, 0, S24)
    ck.published("articulatory data excluded (% of 800 + 400 stimuli)",
                 100 * (1 - int((~em_rows.gesture_excluded_by_authors.astype(bool)).sum()) / 1200), 5.58, 0.01, S24)
    # durations in the text (raw ms, phrase-final)
    fin_c = mono_df.loc[mono_df.phrasal_position == "phrase-final", "coda_ms"]
    fin_s = di_df.loc[di_df.phrasal_position == "phrase-final", "nucleus2_ms"].dropna()
    ck.published("mean coda duration in phrase-final monosyllabic words (ms)", fin_c.mean(), 136, 0.5, S312)
    ck.published("SD of coda duration in phrase-final monosyllabic words (ms)", fin_c.std(), 31, 0.5, S312)
    ck.published("mean schwa duration in phrase-final disyllabic words (ms)", fin_s.mean(), 149, 0.5, S312)
    ck.published("SD of schwa duration in phrase-final disyllabic words (ms)", fin_s.std(), 38.3, 0.05, S312)

    def mean_by(d, col, pos):
        return d.loc[d.phrasal_position == pos, col].mean()

    for t, rep in (("lax", 26), ("tense", 45)):
        d = mono_df[mono_df.vowel_tenseness == t]
        ck.published(f"{t} vowels in monosyllabic words: lengthening phrase-final vs phrase-medial (ms)",
                     mean_by(d, "nucleus_ms", "phrase-final") - mean_by(d, "nucleus_ms", "phrase-medial"), rep, 0.5, S311)
    for t, rep in (("tense", 45), ("lax", 41)):
        d = mono_df[mono_df.vowel_tenseness == t]
        ck.published(f"{t} vowels in monosyllabic words: proportional lengthening (%)",
                     100 * (mean_by(d, "nucleus_ms", "phrase-final") / mean_by(d, "nucleus_ms", "phrase-medial") - 1), rep, 0.5, S311)
    for t, rep in (("lax", 8), ("tense", 19)):
        d = di_df[di_df.vowel_tenseness == t]
        ck.published(f"{t} vowels in disyllabic words: lengthening phrase-final vs phrase-medial (ms)",
                     mean_by(d, "nucleus_ms", "phrase-final") - mean_by(d, "nucleus_ms", "phrase-medial"), rep, 0.5, S312)

    # proportional lengthening per word, from durations normalised by speaking rate as in the authors' code (rate x ms / 1000)
    def prop(col, words):
        d = df[df[col].notna() & df.stimulus.isin(words)].copy()
        d["norm"] = d.person.map(rate) * d[col] / 1000
        g = d.groupby(["stimulus", "phrasal_position"]).norm.mean().unstack()
        return 100 * (g["phrase-final"] / g["phrase-medial"] - 1)

    mono_words, di_words = sorted(mono_df.stimulus.unique()), sorted(di_df.stimulus.unique())
    pn, pc, pn1, pn2 = prop("nucleus_ms", mono_words), prop("coda_ms", mono_words), prop("nucleus_ms", di_words), prop("nucleus2_ms", di_words)
    ck.published("Bahn: lengthening of the vowel phrase-finally (%)", pn["Bahn"], 41, 0.55, "Section 3.1.1")
    ck.published("Stiel (tense /iː/): lengthening of the vowel (%)", pn["Stiel"], 38, 0.55, S311)
    ck.published("still (lax /ɪ/): lengthening of the vowel (%)", pn["still"], 43, 0.55, S311)
    ck.published("monosyllabic codas: smallest lengthening over the 8 words (%)", pc.min(), 75, 0.55, S311)
    ck.published("monosyllabic codas: largest lengthening over the 8 words (%)", pc.max(), 151, 0.55, S311)
    ck.published("Huete (tense /yː/): lengthening of the first vowel (%)", pn1["Huete"], 25, 0.55, S312)
    ck.published("Miete (tense /iː/): lengthening of the first vowel (%)", pn1["Miete"], 28, 0.55, S312)
    ck.published("Huette (lax /ʏ/): lengthening of the first vowel (%)", pn1["Huette"], 17, 0.55, S312)
    ck.published("Mitte (lax /ɪ/): lengthening of the first vowel (%)", pn1["Mitte"], 23, 0.55, S312)
    ck.published("Huette: lengthening of the schwa (%)", pn2["Huette"], 169, 0.55, S312)
    ck.published("Miete: lengthening of the schwa (%)", pn2["Miete"], 241, 0.55, S312)

    # ---- internal checks -------------------------------------------------------------------------------------------------------
    ck.internal("every acoustic constituent row of the authors' tables is in the table",
                n_cons == len(mono) + len(di), f"{n_cons} of {len(mono) + len(di)}")
    ck.internal("every gesture row of the authors' EMA tables is in the table", len(em_rows) == n_art_rows, f"{len(em_rows)} of {n_art_rows}")
    ck.internal("1,191 tokens: 1,162 with acoustic and EMA values, 14 acoustic only, 15 EMA only",
                len(df) == 1191 and int((df.onset_ms.notna() & df.gesture_signal.notna()).sum()) == 1162
                and int((df.onset_ms.notna() & df.gesture_signal.isna()).sum()) == 14 and int((df.onset_ms.isna() & df.gesture_signal.notna()).sum()) == 15,
                f"{len(df)} tokens; both {int((df.onset_ms.notna() & df.gesture_signal.notna()).sum())}")
    ck.internal("10 speakers, none lost", df.person.nunique() == 10 and set(df.person) == set(meta_tab.subject))
    ck.internal("speaker sex agrees between the acoustic labels and the EMA tables", sex_ok)
    ck.internal("trial ranks run 1..n without gaps for each speaker",
                all(sorted(g.trial) == list(range(1, len(g) + 1)) for _, g in df.groupby("person")))
    ck.internal("iterations 1 to 5 only (X marks repeated recordings)", set(df.block.str.replace("iteration ", "").str.replace("X", "")) == {"1", "2", "3", "4", "5"})
    d = em_rows
    ck.internal("closing, plateau and opening duration equal NONS-GONS, NOFFS-NONS and GOFFS-NOFFS",
                bool(np.allclose(d.closing_duration_ms, d.nons_ms - d.gons_ms, atol=1e-6) and np.allclose(d.plateau_duration_ms, d.noffs_ms - d.nons_ms, atol=1e-6)
                     and np.allclose(d.opening_duration_ms, d.goffs_ms - d.noffs_ms, atol=1e-6)))
    tt = d[d.gesture_signal == "TTPOS_XZ"]
    la = d[d.gesture_signal == "LA"]
    ck.internal("tongue tip displacements are the distances between the positions at GONS and MAXC, and at MAXC and GOFFS",
                bool(np.allclose(tt.closing_displacement_mm, np.hypot(tt.maxc_x_mm - tt.gons_x_mm, tt.maxc_z_mm - tt.gons_z_mm), atol=0.06)
                     and np.allclose(tt.opening_displacement_mm, np.hypot(tt.goffs_x_mm - tt.maxc_x_mm, tt.goffs_z_mm - tt.maxc_z_mm), atol=0.06)))
    ck.internal("lip aperture gestures are the words with /m/ (Ruhm, Rum) and have no z values; tongue tip gestures have both",
                set(la.stimulus) == {"Ruhm", "Rum"} and la.gons_z_mm.isna().all() and tt.gons_z_mm.notna().all() and not set(tt.stimulus) & {"Ruhm", "Rum"})
    lm = d[[f"{L.lower()}_ms" for L in LM]].to_numpy(float)
    share = float((np.diff(lm, axis=1) >= 0).all(axis=1).mean())
    ck.internal("landmark times in the order GONS <= PVEL <= NONS <= MAXC <= NOFFS <= PVEL2 <= GOFFS on at least 99% of the gestures",
                share > 0.99, f"{100 * share:.2f}%")
    gaps = [t["gap"] for t in tokens.values() if t["ac"] is not None]
    ck.internal("the constituents of a word follow each other without gaps (start + duration = next start) in at least 99% of the words",
                float(np.mean(np.array(gaps) < 0.01)) > 0.99, f"{100 * float(np.mean(np.array(gaps) < 0.01)):.2f}% (largest gap {max(gaps):.2f} ms)")
    on_axis = df[df.word_start_ms.notna() & df.maxc_ms.notna()]
    near = (on_axis.maxc_ms >= on_axis.word_start_ms - 100) & (on_axis.maxc_ms <= on_axis.word_end_ms + 150)
    ck.internal("acoustic and EMA times share one time axis: the gesture's maximum constriction lies between 100 ms before the word's start "
                "and 150 ms after its end on at least 97% of the tokens that have both",
                float(near.mean()) > 0.97, f"{100 * float(near.mean()):.2f}% of {len(on_axis)}")
    ck.internal("across words, MAXC follows the end of the word's last constituent (correlation over tokens of at least 0.99)",
                float(np.corrcoef(on_axis.maxc_ms, on_axis.word_end_ms)[0, 1]) > 0.99,
                f"r = {float(np.corrcoef(on_axis.maxc_ms, on_axis.word_end_ms)[0, 1]):.4f}")
    rec = df[df.intended_phrasal_position != df.phrasal_position]
    print("tokens whose produced position differs from the intended one:", len(rec), "| speakers:", rec.person.value_counts().to_dict(),
          "| with acoustic values:", int(rec.onset_ms.notna().sum()))
    print("tokens", len(df), "| by experiment", df.experiment.value_counts().to_dict(), "| rows with gesture", len(em_rows),
          "| excluded gestures", int(em_rows.gesture_excluded_by_authors.astype(bool).sum()), "| excluded rows", int(df.excluded_by_authors.sum()))
    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)

    s, paper = paper_meta()
    n_recat = len(rec)
    counts = "words per speaker: " + ", ".join(str(n) for n in sorted(rec.person.value_counts(), reverse=True))
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Merged the authors' two acoustic tables (one row per syllable constituent) and their two EMA tables (one row per "
                 "gesture) into one table with one row per spoken word, matching tokens on speaker and recording name (one recording "
                 "name that is written twice in a row in the EMA table was corrected); the constituent durations became columns; "
                 "the emuR bookkeeping columns were dropped; the women's labels (w in the EMA tables, f in the acoustic tables) were "
                 "made one; speaker labels were replaced by codes."),
        how=[
            "One row is one spoken target word of one of 10 German speakers: 784 monosyllabic and 392 disyllabic words with acoustic "
            "values, and the same words' articulatory gesture where the authors tracked one, plus 15 words that are only in their "
            "articulatory tables and 14 that are only in their acoustic tables. Rows follow the authors' acoustic tables; trial is the "
            "order of the running number in the recording name. Huete and Huette are the deposit's spelling of Hüte and Hütte.",
            "onset_ms, nucleus_ms and coda_ms (monosyllabic words) or onset_ms, nucleus_ms, onset2_ms and nucleus2_ms (disyllabic "
            "words) are the authors' syllable-constituent durations in ms, with stop closure and aspiration counted together. "
            "word_start_ms and word_end_ms put the word on the recording's time axis, which the articulatory landmark times share.",
            "The gesture is the movement toward the last consonant of the stressed syllable, labelled in MVIEW with a 20% velocity "
            "criterion: tongue tip position (gesture_signal TTPOS_XZ, for n, t and l) or lip aperture (LA, for m). Landmark times are in ms, "
            "positions in mm and speeds in cm/s, together with the authors' closing, plateau and opening durations and displacements.",
            f"phrasal_position is the position the authors judged the speaker to have produced; {n_recat} words differ from the "
            "position the sentence was meant to elicit and are listed under the position produced. gesture_excluded_by_authors marks the "
            "44 gestures their articulatory analysis dropped (closing or opening displacement of 1 mm or less, or in disyllabic words a "
            "closing duration of 15 ms or less); their values stay in the table.",
        ],
        unit_note="spoken target word of one speaker (acoustic constituent durations and, where tracked, the EMA gesture toward the last consonant)",
        differences=[
            "The deposit's acoustic files are 48 kHz recordings, where the paper says the acoustic data were recorded at 44.1 kHz.",
            "The paper's Tables 3, 5 and 6 give the number of observations that entered each mixed model (for example 2,342 and 1,541 "
            "durations, and 760, 748, 783 and 782 closing movements): these follow residual-based trimming in the authors' models, which was "
            "not re-run, so only the counts of the data themselves are checked here.",
            f"The paper says 16 utterances of one speaker had a different prosodic structure than targeted and were re-categorized; "
            f"in the deposit the {n_recat} words whose phrasal_position differs from the intended one (10 intended phrase-medial, 6 intended "
            f"phrase-final, as in the paper's Table 2) come from {rec.person.nunique()} speakers ({counts}).",
        ],
    )
    parts = [dict(person=r.subject, speaking_time_min=float(r.sprechzeit), words_spoken=int(r.tokens), syllables_spoken=int(r.syllables),
                  speech_rate_syll_per_s=float(r.sprechgeschw)) for r in meta_tab.itertuples(index=False)]
    pcols = [("speaking_time_min", "number", "minutes",
              "Speaking time of the speaker in the recording (the authors' sprechzeit; in minutes, because syllables divided by this time "
              "times 60 gives the authors' rate)."),
             ("words_spoken", "integer", "", "Number of words the speaker produced in that time (the authors' tokens column)."),
             ("syllables_spoken", "integer", "", "Number of syllables the speaker produced in that time (the authors' syllables column)."),
             ("speech_rate_syll_per_s", "number", "syllables per second",
              "Speaking rate of the speaker (the authors' sprechgeschw), which they used to normalise the durations.")]
    if not ck.passed():
        sys.exit("checks failed")
    return stdlib.release(ID, "articulography", rows, meta, ck, participants=parts, participant_columns=pcols)


if __name__ == "__main__":
    main()
