"""Shiller, Bobbitt & Lametti (2024), J Exp Psychol Gen: adaptation to altered vowel feedback during French or English sentences in
bilinguals, and transfer of the adaptation to French and English words (Experiment 1 L1-French/L2-English, Montreal; Experiment 2
L1-English/L2-French, Wolfville), 20 participants each, two sessions each.

Raw files, raw_extra/Shiller2024_JExpPsycholGen/ (OSF gvf32): bilingual_data_MTL.xlsx and bilingual_data_WV.xlsx, each with three sheets
(sentence level compensation per participant and language; transfer per participant, training language and transfer language;
transfer per participant, training language and word), and basic_data_analysis.m (the authors' plotting script, read as text).
Paper read: the accepted manuscript on PsyArXiv (10.31234/osf.io/5qjx6).
Run from the site folder: python3 -I tools/std/convert/Shiller2024_JExpPsycholGen.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Shiller2024_JExpPsycholGen"
SRC = "Shiller et al. 2024, Results"
LANG = {1: "French", 2: "English"}
WORDS = ["pis", "patte", "pâte", "toute", "sept", "pote", "eux", "eau", "pea", "pat", "pot", "toot", "set", "putt", "hood", "bid"]
SITES = [("MTL", "Experiment 1 (Montreal, L1 French, L2 English)", "bilingual_data_MTL.xlsx", "MTL_"),
         ("WV", "Experiment 2 (Wolfville, L1 English, L2 French)", "bilingual_data_WV.xlsx", "WV_")]


def build():
    raw = stdlib.raw_dir(ID)
    rows = []
    for code, exp, fn, sh in SITES:
        sent = pd.read_excel(raw / fn, sheet_name=f"{sh}sentence_level")
        tavg = pd.read_excel(raw / fn, sheet_name=f"{sh}transfer_averages")
        twrd = pd.read_excel(raw / fn, sheet_name=f"{sh}transfer_words")
        sent.columns = ["p", "lang", "v"]
        tavg.columns = ["p", "train", "transfer", "v"]
        twrd.columns = ["p", "train", "word", "v"]

        def base(p, **kw):
            d = {"dataset": ID, "study": ID, "experiment": exp, "group": "", "subject": f"{code}{int(p)}",
                 "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy bilingual adults",
                 "setting": "lab", "device": "head-mounted microphone and headphones", "session": 1, "trial": 0, "time_s": "",
                 "task": "speech", "unit": "participant summary", "excluded_by_authors": False, "statistic": "mean", "value_unit": "mel"}
            d.update(kw)
            return d

        for r in sent.itertuples(index=False):
            rows.append(base(r.p, block="S6 minus S2", phase="learning", condition=f"sentences in {LANG[r.lang]}",
                             pert_type="formant_shift", pert_value=70.0, pert_unit="mel", feedback="altered",
                             measure="sentence_compensation",
                             measure_description=("Change in produced F1/F2 from the baseline sentence block to the last altered-feedback "
                                                  "sentence block, projected on the direction that opposes the feedback shift "
                                                  "(positive = compensating); the sheet's column is labelled S5-S2, the paper says S6 minus S2"),
                             value=r.v))
        for r in tavg.itertuples(index=False):
            rows.append(base(r.p, block="T2 minus T1", phase="transfer", condition=f"trained on {LANG[r.train]} sentences; words in {LANG[r.transfer]}",
                             pert_type="none", pert_value=0.0, pert_unit="", feedback="masked", measure="transfer",
                             measure_description=("Change in produced F1/F2 of the transfer words from before to after the altered-feedback "
                                                  "sentences (noise-masked feedback), projected on the direction that opposes the feedback "
                                                  "shift, averaged over the 8 words of the language (positive = compensating)"),
                             value=r.v, n_trials=40))
        for r in twrd.itertuples(index=False):
            rows.append(base(r.p, block="T2 minus T1", phase="transfer",
                             condition=f"trained on {LANG[r.train]} sentences; word {WORDS[int(r.word) - 1]}",
                             pert_type="none", pert_value=0.0, pert_unit="", feedback="masked", measure="transfer_word",
                             measure_description=("Transfer (T2 minus T1, projected as for the transfer measure) for one word, over its 5 "
                                                  "repetitions per block"),
                             value=r.v, n_trials=5))
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1  # order in this table
    return a


def main():
    checks = stdlib.Checks()
    a = build()
    a["measure_description"] = a.measure_description + "."
    for code, exp, fn, sh in SITES:
        d = a[a.subject.str.startswith(code)]
        checks.published(f"participants, {exp.split(' (')[0]}", d.subject.nunique(), 20, 0, SRC + " / Methods, Participants")
    checks.published("participants in total", a.subject.nunique(), 40, 0, "Shiller et al. 2024, Methods: 'We recruited 40 bilinguals'")
    s = a[a.measure == "sentence_compensation"]
    t = a[a.measure == "transfer"]

    def vals(d, code, cond):
        return d[d.subject.str.startswith(code) & (d.condition == cond)].set_index("subject").value

    exp_means = {("MTL", "French"): 24.11, ("MTL", "English"): 22.95, ("WV", "French"): 20.55, ("WV", "English"): 25.37}
    exp_t = {("MTL", "French"): 9.98, ("MTL", "English"): 6.74, ("WV", "French"): 5.81, ("WV", "English"): 4.37}
    for (code, lg), m in exp_means.items():
        v = vals(s, code, f"sentences in {lg}")
        if (code, lg) != ("MTL", "English"):  # computed 22.958 vs printed 22.95: listed under differences, not checked
            checks.published(f"mean adaptation, {code} sentences in {lg} (mel)", v.mean(), m, 0.005, SRC + " (Experiment %d, adaptation)" % (1 if code == "MTL" else 2))
        checks.published(f"one-sample t of adaptation, {code} {lg}", stats.ttest_1samp(v, 0).statistic, exp_t[(code, lg)], 0.005, SRC)
    for code, tt in (("MTL", 0.3), ("WV", 0.71)):
        f = vals(s, code, "sentences in French")
        e = vals(s, code, "sentences in English")
        checks.published(f"paired t French vs English sentences, {code}", abs(stats.ttest_rel(f, e.loc[f.index]).statistic), tt, 0.05 if code == "MTL" else 0.005, SRC)
    tr = {("MTL", "French", "French"): 4.76, ("MTL", "French", "English"): 4.98, ("MTL", "English", "French"): 4.0, ("MTL", "English", "English"): 5.15,
          ("WV", "French", "French"): 2.17, ("WV", "French", "English"): 2.28, ("WV", "English", "French"): 4.93, ("WV", "English", "English"): 5.0}
    for (code, trn, trf), tv in tr.items():
        v = vals(t, code, f"trained on {trn} sentences; words in {trf}")
        checks.published(f"one-sample t of transfer, {code} train {trn} transfer {trf} (n = {len(v)})", stats.ttest_1samp(v, 0).statistic, tv,
                         0.05 if tv == 4.0 else 0.005, SRC)
    checks.internal("Wolfville has 19 participants with train-French transfer (one removed by the authors as an outlier)",
                    int(vals(t, "WV", "trained on French sentences; words in French").size) == 19
                    and int(vals(t, "WV", "trained on English sentences; words in French").size) == 20)
    checks.internal("rows: 80 sentence, 158 transfer and 1264 word values", len(s) == 80 and len(t) == 158 and (a.measure == "transfer_word").sum() == 624 + 640)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    src = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(src["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=src["lab"], data_url=src["data_url"], license="none stated", license_url="",
        changes=("Joined the six sheets of the authors' two workbooks (Montreal and Wolfville) into one long table, translated their "
                 "numeric language and word codes to names (from the authors' plotting script) and replaced participant numbers."),
        how=[
            "The authors shared only per-participant values, so each row is one participant's value of one measure: sentence_compensation "
            "(adaptation during French or English sentences), transfer (to French or English words after training in each language) "
            "and transfer_word (transfer to each of the 16 words), all in mel and projected on the direction that opposes the feedback "
            "shift, so positive values mean compensation (paper, Data Analysis). Each participant took part in two sessions, one training "
            "on French and one on English sentences, so every participant has values for both.",
            "The feedback shift was 49.5 mel down in F1 and 49.5 mel up in F2 (70 mel in all; ramped over the first 25 trials of "
            "the first altered-feedback block). Transfer words were produced with noise-masked feedback (feedback = masked, "
            "pert_type = none).",
            "Language codes (1 French, 2 English) and word codes follow the authors' script basic_data_analysis.m; words 1-8 are French "
            "and 9-16 English. The sentence sheet's column is labelled S5-S2 while the paper defines adaptation as S6 minus S2; the "
            "values reproduce the paper's means, so the label is kept and the paper's definition used.",
            "The Wolfville transfer sheets lack one participant's values for training on French (two averages and sixteen words): the paper "
            "says one participant's transfer data were removed as a 2.5 SD outlier. Age and sex are not shared.",
        ],
        unit_note="participant, measure and condition",
        differences=["The paper prints a mean adaptation of 22.95 mel for English sentences in Experiment 1; the rows give 22.958 (22.96 rounded), so that number is not used as a check (the other three means match to the printed two decimals).",
                     "The paper's effect sizes (Cohen's d) and F tests were not recomputed; means, one-sample t and paired t values were."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
