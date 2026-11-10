"""Oschkinat, Reinisch & Hoole (2026), Language and Speech: "Lexical Bias Effects do not Influence Responses to Real-Time
Temporal Auditory Feedback Perturbation". 61 German speakers said Staat or Stab (vowel compression setup) or Stadt or Stamm
(vowel stretching setup) 110 times each while a real-time system (Audapter with online status tracking) warped the timing of
the speech they heard in a baseline - ramp - hold - after-effect design. One row is one segment of one spoken word, with its
spoken duration and the duration the speaker heard.

Raw files, raw_extra/Oschkinat2026_LangSpeech/ (OSF rsytu, fetched with tools/std/fetch.py):
  0_Perturbation_data_Stadt_Staat.RData   data frame Perturbation.data (257752 rows x 18 columns), long format: one row per
                                          speaker x word x trial x segment (Sound) x channel (1 = spoken, 2 = heard,
                                          2_simpert = simulated perturbation of baseline and after-effect productions); the
                                          columns are described in the authors' script 1 (Subject, setup, word, lexical_condition,
                                          phase B/R/H/A, trialnumber, pert_trial, Sound, channel, PCF1_dur, OST_group, pcf_ok,
                                          Bmean_ms, dur_ms, absdiff_ms, age, gender, order_first) plus a comment object
  1_Data_preparation_and_exclusions.Rmd and .html   the authors' script 1 and its rendered output: demographics, exclusion
                                          rules, the preV grouping and Table 1 (read as text only)
  2_Analyses_Vowels_section_3.1.Rmd       the analysis script, read as text for its row filters
  Table_1.docx                            the authors' Table 1 output (perturbation magnitude and its SD per word and segment)
The exclusion flags (exclude, preV_group_25ms, exclude_trials) are created by script 1, not stored in the RData file; they are
rebuilt here with the same rules (checked against the counts and Table 1 that script 1 prints).
Run from the site folder: python3 -I tools/std/convert/Oschkinat2026_LangSpeech.py
"""
import html
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Oschkinat2026_LangSpeech"
SRC = "Oschkinat, Reinisch & Hoole 2026"
SRC1 = f"{SRC}, authors' script 1 output (paper section 2.5)"
SRC_T1 = f"{SRC}, Table 1 (perturbation magnitude)"
DEVICE = "Audapter real-time speech perturbation system with online status tracking (OST)"
PHASE_NAME = {"B": "Baseline", "R": "Ramp", "H": "Hold", "A": "After-effect"}
PHASE_STD = {"B": "baseline", "R": "learning", "H": "learning", "A": "aftereffect"}
SOUND_ORDER = ["preV", "Onset", "C1", "C2", "V", "Ccoda", "Cburst", "Coda"]
SOUND_TXT = {
    "preV": "preV (the final /\u0250/ of 'besser' before the word)", "Onset": "Onset (as named in the deposit, not described there)",
    "C1": "C1 (/\u0283/)", "C2": "C2 (/t/)", "V": "V (the vowel /a/ or /a:/)", "Ccoda": "Ccoda (closure of /t/ or /m/)",
    "Cburst": "Cburst (burst and aspiration of /t/)", "Coda": "Coda (closure plus burst and aspiration of /t/ or /m/)",
}
KEY = ["Subject", "word", "trialnumber", "Sound"]
BASE_CUT = 6  # script 1: "remove first 5 trials of baseline (15 baseline trials)"


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def rebuild_exclusions(d):
    """Script 1 rules: (a) a speaker x word is kept ('exclude' = no) when, over trials 6-95 and with ramp/hold trials only if
    pcf_ok = 1, every phase has at least B 9, R 20, H 20, A 10 trials; (b) Staat/Stab speakers whose mean hold-phase preV
    perturbation (heard - spoken) is above 25 ms are 'preV stretched'; (c) single ramp/hold trials with pcf_ok = 0 are bad."""
    cc = d[(d.trialnumber >= BASE_CUT) & ~d.trialnumber.between(96, 110)]
    cc = cc[(cc.phase.isin(["R", "H"]) & (cc.pcf_ok == 1)) | cc.phase.isin(["B", "A"])]
    cells = cc.groupby(["Subject", "setup", "lexical_condition", "phase"]).trialnumber.nunique().reset_index(name="n")
    cells = cells[cells.n >= cells.phase.map({"B": 9, "R": 20, "H": 20, "A": 10})]
    cnt = cells.groupby(["Subject", "setup", "lexical_condition"]).size()
    valid = set(cnt[cnt == 4].index)
    h = d[(d.phase == "H") & d.word.isin(["Staat", "Stab"]) & (d.Sound == "preV") & d.channel.isin(["1", "2"])]
    p = h.pivot_table(index=["Subject", "word", "trialnumber"], columns="channel", values="dur_ms", aggfunc="first")
    m = (p["2"] - p["1"]).groupby(level=["Subject", "word"]).mean()
    return valid, m


def table1_docx(raw):
    z = zipfile.ZipFile(raw / "Table_1.docx")
    x = z.read("word/document.xml").decode("utf-8", "replace")
    out = {}
    for r in re.findall(r"<w:tr[ >].*?</w:tr>", x, flags=re.S)[1:]:
        c = ["".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", t)) for t in re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S)]
        out[(c[0], c[1])] = (float(c[2]), float(c[3]))
    return out


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    d = pyreadr.read_r(str(raw / "0_Perturbation_data_Stadt_Staat.RData"))["Perturbation.data"]
    d["trialnumber"] = d.trialnumber.astype(int)
    d["Subject"] = d.Subject.astype(str)
    checks.internal("257752 long rows: 105056 spoken, 105056 heard and 47640 simulated-perturbation rows, 61 speakers, no duplicate "
                    "speaker x word x trial x segment x channel",
                    bool(len(d) == 257752 and d.channel.value_counts().to_dict() == {"1": 105056, "2": 105056, "2_simpert": 47640}
                         and d.Subject.nunique() == 61 and not d.duplicated(KEY + ["channel"]).any()))
    checks.internal("phases follow the trial number (baseline 1-20, ramp 21-50, hold 51-80, after-effect 81-110); simulated "
                    "perturbation rows exist only in baseline and after-effect",
                    bool(d.groupby("phase").trialnumber.agg(["min", "max"]).to_dict("index") ==
                         {"A": {"min": 81, "max": 110}, "B": {"min": 1, "max": 20}, "H": {"min": 51, "max": 80}, "R": {"min": 21, "max": 50}}
                         and set(d[d.channel == "2_simpert"].phase) == {"A", "B"}))
    valid, m = rebuild_exclusions(d)
    sp = d[d.channel == "1"].drop(columns=["channel", "absdiff_ms"]).rename(columns={"dur_ms": "spoken"})
    hd = d[d.channel == "2"][KEY + ["dur_ms"]].rename(columns={"dur_ms": "heard"})
    sm = d[d.channel == "2_simpert"][KEY + ["dur_ms"]].rename(columns={"dur_ms": "simulated"})
    a = sp.merge(hd, on=KEY, how="left", validate="1:1").merge(sm, on=KEY, how="left", validate="1:1")
    checks.internal("every spoken segment has a heard segment; heard equals spoken in baseline and after-effect",
                    bool(a.heard.notna().sum() == a.spoken.notna().sum()
                         and (a[a.phase.isin(["B", "A"])].heard - a[a.phase.isin(["B", "A"])].spoken).abs().max() == 0))
    a["_valid"] = [(s, st, lc) in valid for s, st, lc in zip(a.Subject, a.setup, a.lexical_condition)]
    a["_pv"] = [m.get((s, w), np.nan) for s, w in zip(a.Subject, a.word)]
    a["_stretched"] = a._pv > 25
    a["_seqexcl"] = ~a._valid | a._stretched
    a["_badtrial"] = a.phase.isin(["R", "H"]) & (a.pcf_ok == 0)
    # order: speaker, then the word tested first, trial, segment
    both = a.groupby("Subject").word.nunique()
    firstword_ok = a.drop_duplicates(["Subject", "word"]).assign(ok=lambda x: x.word == x.order_first).groupby("Subject").ok.any()
    checks.internal("the word named in order_first is one of the speaker's words for all 60 speakers that have two words",
                    bool(firstword_ok[both[both == 2].index].all()), f"{int(firstword_ok.sum())} of {len(firstword_ok)} speakers overall")
    a["_wo"] = (a.word != a.order_first).astype(int)
    a["_so"] = a.Sound.map({s: i for i, s in enumerate(SOUND_ORDER)})
    a = a.sort_values(["Subject", "_wo", "word", "trialnumber", "_so"], kind="stable").reset_index(drop=True)
    a["_utt"] = a.groupby("Subject", sort=False).apply(
        lambda g: pd.Series(pd.factorize(g.word + "|" + g.trialnumber.astype(str))[0] + 1, index=g.index)).reset_index(level=0, drop=True)
    comp = a.setup == "Vowel compression"
    mod = a.phase.isin(["R", "H"])
    out = pd.DataFrame({
        "dataset": ID, "study": ID, "experiment": a.setup, "group": a.OST_group, "subject": a.Subject + "_" + a.word,
        "person": a.Subject, "subject_excluded_by_authors": a._seqexcl, "age": a.age.astype(int),
        "sex": a.gender.map({"w": "F", "m": "M"}), "handedness": "", "population": "healthy", "setting": "lab", "device": DEVICE,
        "session": 1, "block": a.phase.map(PHASE_NAME), "trial": a._utt, "phase": a.phase.map(PHASE_STD), "time_s": np.nan,
        "task": "speech", "unit": "utterance", "condition": a.lexical_condition,
        "pert_type": np.where(mod, np.where(comp, "vowel_compression_timewarp", "vowel_stretching_timewarp"), "none"),
        "pert_value": np.where(mod, a.heard - a.spoken, 0.0), "pert_unit": np.where(mod, "ms", ""),
        "feedback": np.where(mod, "altered", "normal"),
        "excluded_by_authors": a._seqexcl | a._badtrial | (a.trialnumber < BASE_CUT) | a.trialnumber.between(96, 110),
        "stimulus": a.word, "segment": a.Sound.map(SOUND_TXT), "measure_window": "whole segment", "duration_ms": a.spoken,
        "duration_heard_ms": a.heard, "duration_simulated_ms": a.simulated, "baseline_duration_ms": a.Bmean_ms,
        "perturbation_check_ok": a.pcf_ok == 1, "perturbed_section_dur": a.PCF1_dur, "authors_trial_number": a.trialnumber,
    })
    out["time_s"] = ""
    names = [c[1] for c in stdlib.columns_for("speech")]
    for c in names:
        if c not in out.columns:
            out[c] = ""
    out = out[names]
    checks.internal("105056 rows = 61 speakers' segments; 121 speaker x word sequences (one speaker has one word); every utterance "
                    "has all 8 segments",
                    bool(len(out) == 105056 and out.person.nunique() == 61 and out.subject.nunique() == 121
                         and (a.groupby(["Subject", "word", "trialnumber"]).size() == 8).all()))
    checks.internal("bad-trial flag: ramp/hold utterances with the perturbation check failed (pcf_ok = 0) are flagged; baseline and "
                    "after-effect trials are never flagged for it",
                    bool(a[a._badtrial].phase.isin(["R", "H"]).all() and a._badtrial.sum() > 0))
    # ---- published numbers: script 1 output (paper section 2.5) and Table 1
    ppl = d.drop_duplicates("Subject")
    checks.published("tested participants", len(ppl), 61, 0, SRC1)
    for w, n in (("Staat", 38), ("Stab", 37), ("Stadt", 23), ("Stamm", 23)):
        checks.published(f"tested participants saying {w}", d[d.word == w].Subject.nunique(), n, 0, SRC1)
    checks.published("mean age of the tested participants (years)", ppl.age.mean(), 23.0, 0.05, SRC1)
    checks.published("tested participants: men", (ppl.gender == "m").sum(), 10, 0, SRC1)
    checks.published("tested participants: women", (ppl.gender == "w").sum(), 51, 0, SRC1)
    seq = a.drop_duplicates(["Subject", "word"])
    for w, kept, gone in (("Staat", 30, 8), ("Stab", 27, 10), ("Stadt", 22, 1), ("Stamm", 22, 1)):
        s = seq[seq.word == w]
        checks.published(f"{w}: speakers kept by the trial-count rule (exclude = no)", s._valid.sum(), kept, 0, SRC1)
        checks.published(f"{w}: speakers excluded by the trial-count rule (exclude = yes)", (~s._valid).sum(), gone, 0, SRC1)
    for w, ns, st in (("Staat", 20, 10), ("Stab", 18, 9)):
        s = seq[(seq.word == w) & seq._valid]
        checks.published(f"{w}: kept speakers with a non-stretched preV", (~s._stretched).sum(), ns, 0, SRC1)
        checks.published(f"{w}: kept speakers with a stretched preV (excluded from the analyses)", s._stretched.sum(), st, 0, SRC1)
    fin = seq[~seq._seqexcl]
    fp = fin.drop_duplicates("Subject")
    checks.published("participants in the final data", fp.Subject.nunique(), 46, 0, SRC1)
    checks.published("mean age of the final participants (years)", fp.age.mean(), 22.9, 0.05, SRC1)
    checks.published("final participants: men", (fp.gender == "m").sum(), 5, 0, SRC1)
    checks.published("final participants: women", (fp.gender == "w").sum(), 41, 0, SRC1)
    nw = fin.groupby(["Subject", "setup"]).word.nunique().reset_index()
    checks.published("final participants with both words, vowel compression", ((nw.setup == "Vowel compression") & (nw.word == 2)).sum(), 15, 0, SRC1)
    checks.published("final participants with both words, vowel stretching", ((nw.setup == "Vowel stretching") & (nw.word == 2)).sum(), 21, 0, SRC1)
    t1 = a[~a._seqexcl & (a.phase == "H") & (a.Sound == "V") | (~a._seqexcl & (a.phase == "H") & a.Sound.isin(["C1", "C2"]) & a.word.isin(["Staat", "Stab"]))
           | (~a._seqexcl & (a.phase == "H") & (a.Sound == "Ccoda") & a.word.isin(["Stadt", "Stamm"]))].copy()
    t1["pert"] = t1.heard - t1.spoken
    per = t1.groupby(["word", "Sound", "Subject"]).pert.mean().reset_index()
    tab = per.groupby(["word", "Sound"]).pert.agg(["mean", "std"])
    pub = table1_docx(raw)
    checks.internal("Table 1 has 10 word x segment rows", bool(len(pub) == 10 and set(pub) == set(tab.index)))
    for (w, snd), (sd, mag) in pub.items():
        checks.published(f"Table 1, {w} {snd}: perturbation magnitude in the hold phase (ms)", tab.loc[(w, snd), "mean"], mag, 6e-5, SRC_T1)
        checks.published(f"Table 1, {w} {snd}: SD across speakers (ms)", tab.loc[(w, snd), "std"], sd, 6e-5, SRC_T1)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s, paper = paper_meta()
    ppl_rows = [dict(person=r.Subject, first_word=r.order_first) for r in ppl.itertuples()]
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Took the authors' long table (speaker x word x trial x segment x channel), put the spoken, heard and simulated "
                 "durations of each segment side by side in one row, kept every segment row of the spoken channel, rebuilt the "
                 "authors' exclusion flags from their script, added the segment descriptions, the apparatus and the realized "
                 "perturbation (heard minus spoken duration, their definition), dropped the authors' absdiff_ms (duration minus "
                 "the baseline mean) because it can be recomputed from baseline_duration_ms, and replaced speaker codes with codes."),
        how=[
            "One row is one segment (segment: preV, C1, C2, V, Ccoda, Cburst, Coda, Onset, as the authors name them) of one spoken "
            "word of one speaker; the 8 segment rows of an utterance share trial (the utterance order within the speaker, the word "
            "tested first first) and authors_trial_number (the authors' trial 1-110 within the word). subject is one speaker saying one "
            "word (a speaker has two words, except one), person is the speaker. duration_ms is the spoken duration, "
            "duration_heard_ms the duration of the same segment in the feedback the speaker heard, duration_simulated_ms the "
            "authors' simulated perturbation of unperturbed productions (baseline and after-effect only).",
            "Vowel compression setup: Staat (lexical, perturbed towards Stadt) and Stab (non-lexical); vowel stretching setup: Stadt "
            "(lexical, towards Staat) and Stamm (non-lexical). Phases by trial: baseline 1-20, ramp 21-50, hold 51-80, after-effect "
            "81-110 (phase maps ramp and hold to learning). In ramp and hold trials the timing of the speech the speaker heard was "
            "warped (a stretching first half and a compression second half of the perturbed section); pert_value is the "
            "realized change of that segment in ms (heard minus spoken, the authors' pert_ms), 0 in baseline and after-effect.",
            "subject_excluded_by_authors marks the speaker x word sequences their script 1 leaves out (fewer than 9 baseline, "
            "20 ramp, 20 hold or 10 after-effect good trials over trials 6-95, or, for Staat and Stab, a mean hold-phase preV "
            "perturbation above 25 ms); excluded_by_authors adds the rows their analyses drop: ramp and hold trials whose "
            "perturbation check failed, the first five baseline trials and after-effect trials 96-110.",
        ],
        unit_note="segment of a spoken word (utterance) of one speaker",
        differences=[
            "The exclusion flags are not stored in the authors' data file; they were rebuilt from script 1 and reproduce the "
            "participant counts the script prints and all 20 numbers of Table 1.",
            "The authors' absdiff_ms is not copied (it is duration minus baseline mean).",
        ],
    )
    return stdlib.release(ID, "speech", out, meta, checks, participants=ppl_rows,
                          participant_columns=[("first_word", "text", "",
                                                "Word the speaker was tested on first (the authors' order_first: the order of testing "
                                                "conditions within the setup).")])


if __name__ == "__main__":
    main()
