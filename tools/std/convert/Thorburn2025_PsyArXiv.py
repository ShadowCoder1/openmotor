"""Thorburn, Zhou, Dick, Nozari & Holt (2025), PsyArXiv preprint: speech motor control is not sequestered from general auditory
processes. Online (Gorilla, Prolific) experiments in which a context (Exp 1: the syllables /al/ or /ar/; Exp 2: audiobook excerpts
filtered to a high or low spectrum; Exp 3: sequences of high or low sine tones) precedes an ambiguous /ga/-/da/ target syllable that
the participant repeats aloud and categorizes with a key press; pairing context and target enhances or diminishes the target's
perceptual distinctiveness. The production measure is the long-term average spectrum (LTAS) of the first 75 ms of each utterance.

Raw files, raw_extra/Thorburn2025_PsyArXiv/ (OSF mnj85, fetched with tools/std/fetch.py; no license stated):
  exp<e>_behavioral_data.csv   one row per trial of every participant who ran the experiment (Gorilla export): participant, trial
        number, key press Response (ga, da) and Reaction Time (ms), condition (Enhanced, Diminished, Practice), context and target
        stimulus, counterbalancing order (Spreadsheet), block
  exp<e>_ltas_raw.csv          the authors' analysis table (their notebook sampling_analysis_general.ipynb, read as text): the
        trials of the analysed participants whose production gave a spectrum, with the LTAS as columns 0-299 (zero-padded) and
        response_num (ga = 1); 'Unnamed: 0' is the row key of the behavioral file
  exp<e>_sampling_analysis_general.ipynb   the authors' exclusion rules, read as text: a participant who answers only ga or only da
        overall or in one condition is excluded, plus hand-picked participants (recording problems, insufficient data); trials with
        a key press slower than 3 s are dropped; trials without a spectrum drop out when the spectra are attached
  README*.txt                  the extraction pipeline: recordings of 2 s, last non-silent segment, voice onset time found with Dr.VOT,
        LTAS (Praat, 100 Hz bandwidth) over 0-75 ms from the voice onset or burst, amplitude-normalized to 70 dB SPL
Paper: the authors' preprint (PsyArXiv 6xyr2_v1, a docx on the project page), Methods of Experiments 1-3.
Run from the site folder: python3 -I tools/std/convert/Thorburn2025_PsyArXiv.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Thorburn2025_PsyArXiv"
SRC = "Thorburn et al. 2025, PsyArXiv"
RAW = stdlib.raw_dir(ID)
PID = "Participant Private ID"
# participants the authors' notebooks exclude by hand, besides the ga/da rule (Experiment 3's 10586022 is listed in the notebooks)
EXTRA = {
    1: {12112402, 11988874, 12304283, 11988700, 11940497, 12112237, 12304513},
    2: {12240188, 12120033, 12239360, 12120453, 12295636, 12335408, 12071404, 12240277},
    3: {10586022},
}
NAME = {1: "Experiment 1: syllable contexts", 2: "Experiment 2: audiobook contexts", 3: "Experiment 3: tone contexts"}
TARGET = {"dg4": "dg4 (/ga/-like)", "dg6": "dg6 (/da/-like)"}
DEVICE = "participant's own computer, headphones and microphone, Google Chrome on the Gorilla platform (2 s recordings)"
WINDOW = ("first 75 ms from the voice onset time reference point (voicing onset for a positive VOT, stop burst for a negative VOT, "
          "found with Dr.VOT), amplitude-normalized to 70 dB SPL")
SEGMENT = "initial consonant of the produced /ga/ or /da/ syllable"
LTAS_COLS = [str(i) for i in range(300)]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    p["venue"] = p["venue"] or "PsyArXiv preprint"  # blank in extra_studies.json
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def rule_excluded(B, resp, cond):
    """The notebooks' rule: a participant who gives only ga or only da overall or in the Enhanced or Diminished condition."""
    crit = B[B[cond] != "Practice"]
    ex = set()
    for p, g in crit.groupby(PID):
        for c in (None, "Enhanced", "Diminished"):
            gg = g if c is None else g[g[cond] == c]
            if (gg[resp] == "ga").sum() == 0 or (gg[resp] == "da").sum() == 0:
                ex.add(int(p))
    return ex


def ltas_strings(L):
    V = L[LTAS_COLS].to_numpy(dtype=float)
    out = {}
    lens = []
    for key, row in zip(L["Unnamed: 0"].astype(int), V):
        nz = np.nonzero(row)[0]
        n = int(nz.max()) + 1 if len(nz) else 0
        lens.append(n)
        out[key] = ";".join(format(x, ".8g") for x in row[:n])
    return out, pd.Series(lens)


def prep(e):
    B = pd.read_csv(RAW / f"exp{e}_behavioral_data.csv", low_memory=False)
    L = pd.read_csv(RAW / f"exp{e}_ltas_raw.csv", low_memory=False)
    B[PID] = B[PID].astype(int)
    L[PID] = L[PID].astype(int)
    if e == 1:
        cond, tgt_col = "condition", "ContextStim"
        B["ctx"] = B[tgt_col].str.split("_").str[0]
        B["tgt"] = B[tgt_col].str.split("_").str[1]
        B["exposure"] = B["ctx"]
        B["block_i"] = B["randomise_trials"]
        order_col = "Spreadsheet"
    elif e == 2:
        cond = "Spreadsheet: Condition"
        B["tgt"] = B["Spreadsheet: TestStim"].str.replace(".wav", "", regex=False)
        B["exposure"] = B["Spreadsheet: Filter"]
        B["block_i"] = B["Spreadsheet: Block"].where(B[cond] != "Practice")
        order_col = "Spreadsheet"
    else:
        cond = "condition"
        B["tgt"] = B["TestStim"].str.replace(".wav", "", regex=False)
        B["exposure"] = B["Tones"].str.replace(".wav", "", regex=False)
        B["block_i"] = B["randomise_trials"]
        order_col = "Spreadsheet"
    B["cond"] = B[cond]
    first_crit = 4 if e in (1, 2) else 1  # Experiments 1 and 2 start with three practice trials
    B["block_chron"] = ((B["Trial Number"] - first_crit) // 40 + 1).where(B["cond"] != "Practice")
    B["first"] = np.where(B[order_col].str.startswith("Enhance"), "Enhanced first", "Diminished first")
    ex = rule_excluded(B, "Response", "cond") | (EXTRA[e] & set(B[PID]))
    return B, L, ex


def main():
    ck = stdlib.Checks()
    rows, helper, summary = [], [], {}
    paper_n = {1: 55, 2: 54, 3: 57}
    for e in (1, 2, 3):
        B, L, ex = prep(e)
        strings, lens = ltas_strings(L)
        m = B.merge(L[["Unnamed: 0", PID, "response_num"]], on=["Unnamed: 0", PID], how="left", indicator=True)
        has = m["_merge"].eq("both").to_numpy()
        ck.internal(f"Exp {e}: every row of the authors' spectrum table matches a trial of the behavioral file",
                    int(has.sum()) == len(L) and L["Unnamed: 0"].is_unique, f"{len(L)} spectrum rows, {len(B)} trials")
        agree = (m.loc[has, "Response"].map({"ga": 1.0, "da": 0.0}) == m.loc[has, "response_num"]).all()
        ck.internal(f"Exp {e}: key press Response agrees with the authors' response_num (ga = 1) on every matched row", bool(agree))
        dups = B[B.duplicated([PID, "Trial Number"], keep=False)]
        rec = "Spreadsheet: RecordName" if e == 2 else "RecordName"
        ck.internal(f"Exp {e}: rows that share a trial number within a participant are exact repeats (same recording, response, time)",
                    len(dups) <= 4 and (dups.empty or bool(dups.groupby([PID, "Trial Number"]).agg(
                        a=(rec, "nunique"), b=("Response", "nunique"), c=("Reaction Time", "nunique")).max().max() <= 1)),
                    f"{len(dups)} rows")
        ck.internal(f"Exp {e}: no spectrum for an excluded participant, a practice trial or a key press of 3 s or more",
                    bool(not has[(B[PID].isin(ex) | (B.cond == "Practice") | (B["Reaction Time"] >= 3000)).to_numpy()].any()))
        ck.internal(f"Exp {e}: spectra have 80, 120, 221 or 240 values", set(lens.unique()) <= {80, 120, 221, 240},
                    str(sorted(lens.unique())))
        vals = L[LTAS_COLS].to_numpy(dtype=float)
        ck.internal(f"Exp {e}: spectrum levels within -60 to 50 dB", bool(vals[vals != 0].min() > -60 and vals.max() < 50),
                    f"{vals[vals != 0].min():.1f} to {vals.max():.1f}")
        B["trial_rank"] = B.groupby(PID)["Trial Number"].rank(method="first").astype(int)
        crit = B[B.cond != "Practice"]
        ck.internal(f"Exp {e}: chronological block (40 trials each) is 1-6 on every critical trial"
                    + (" and equals the authors' block label" if e == 2 else ""),
                    bool(crit.block_chron.between(1, 6).all() and (e != 2 or (crit.block_chron == crit.block_i).all())))
        for r, hs in zip(B.to_dict("records"), has):
            p = int(r[PID])
            subj = f"E{e}|{p}"
            practice = r["cond"] == "Practice"
            if practice:
                blk = "Practice"
            else:  # chronological 40-trial block; Experiments 1 and 3 number blocks by condition (1-3 Enhanced, 4-6 Diminished)
                blk = f"Block {int(r['block_chron'])} of 6" + (f" (authors' randomise_trials {int(r['block_i'])})" if e != 2 else "")
            s = strings.get(int(r["Unnamed: 0"])) if hs else None
            rows.append(dict(
                dataset=ID, study=SRC, experiment=NAME[e], group=r["first"], subject=subj, person=subj,
                subject_excluded_by_authors=p in ex, age="", sex="", handedness="", population="healthy", setting="online",
                device=DEVICE, session=1, block=blk, trial=int(r["trial_rank"]), phase="practice" if practice else "test",
                time_s="", task="speech", unit="trial", condition=r["cond"], pert_type="none", pert_value=0, pert_unit="",
                feedback="normal", excluded_by_authors=not hs,
                stimulus=TARGET[r["tgt"]], segment=SEGMENT if hs else "", measure_window=WINDOW if hs else "",
                perceptual_response=r["Response"], exposure_set=r["exposure"], authors_trial_number=int(r["Trial Number"]),
                categorization_rt_ms=float(r["Reaction Time"]), ltas_db=s if s is not None else ""))
            helper.append((e, p, hs, p in ex, practice, float(r["Reaction Time"])))
        summary[e] = dict(n_part=B[PID].nunique(), n_excl=len(ex), n_trials=len(B), n_ltas=len(L), n_ltas_part=L[PID].nunique())
    h = pd.DataFrame(helper, columns=["e", "p", "has", "excl", "practice", "rt"])

    for e in (1, 2, 3):
        d = h[h.e == e]
        an = d[~d.excl & ~d.practice]  # critical trials of the analysed participants
        n_an = d[~d.excl].p.nunique()
        ck.published(f"Experiment {e}: participants analysed", n_an, paper_n[e], 0, f"Thorburn et al. 2025 preprint, Experiment {e}, Methods, Participants")
        slow = (an.rt >= 3000).sum()
        design = 240 * n_an  # 2 x 120 critical trials per analysed participant (paper, Procedure); the file lacks a few in Experiment 2
        ck.published(f"Experiment {e}: critical trials with a key press of 3 s or more (%)", 100 * slow / design, {1: 1.1, 2: 2.5, 3: 3.5}[e], 0.05,
                     f"Thorburn et al. 2025 preprint, Experiment {e}, Analytic Approach")
        if e == 1:
            ck.published("Experiment 1: participants excluded", d[d.excl].p.nunique(), 27, 0,
                         "Thorburn et al. 2025 preprint, Experiment 1, Methods, Participants")
        if e == 3:
            ck.published("Experiment 3: participants excluded", d[d.excl].p.nunique(), 3, 0,
                         "Thorburn et al. 2025 preprint, Experiment 3, Methods, Participants")
        miss = 100 * (design - slow - an.has.sum()) / design
        summary[e].update(slow_pct=100 * slow / design, miss_pct=miss, n_an=n_an, crit=design, rows_crit=len(an), kept=int(an.has.sum()))
        if e in (1, 2):
            ck.published(f"Experiment {e}: critical trials lost to recording problems (%)", miss, {1: 2.2, 2: 0.9}[e], 0.05,
                         f"Thorburn et al. 2025 preprint, Experiment {e}, Analytic Approach")
    for x in ck.items:
        if not x["ok"]:
            print("FAILED:", x)
    print(json.dumps(summary, default=float))

    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Merged the authors' analysis tables (exp<e>_ltas_raw.csv: production spectra) onto the trials of their behavioral "
                 "files (exp<e>_behavioral_data.csv) with the key the two share, kept all trials of all participants, marked the "
                 "authors' exclusions from their notebooks, wrote each spectrum as one text value without its zero padding, and "
                 "replaced Gorilla participant numbers with codes."),
        how=[
            "One row is one trial of one online participant in Experiment 1 (syllable contexts), 2 (audiobook contexts) or 3 "
            "(tone contexts), in the order of the authors' behavioral files (two exact repeated rows of Experiment 2 are kept); "
            "trial counts the participant's trials in the "
            "order of the authors' trial number (authors_trial_number). On each trial a context (exposure_set: the syllable "
            "al or ar; the audiobook filter high or low; the tone sequence, BBH high and BBL low) preceded the target syllable "
            "(stimulus: the more /ga/-like dg4 or the more /da/-like dg6), which the participant repeated aloud and categorized "
            "as ga or da with a key press (perceptual_response; categorization_rt_ms is its response time).",
            "condition is the authors' pairing of context and target: Enhanced (the pairing expected to make the target more "
            "distinct) or Diminished, or Practice (the first three trials of Experiments 1 and 2); group is the order of the two "
            "conditions the participant was given; block is the chronological block of 40 trials (in Experiments 1 and 3 the "
            "authors' own block numbers go with the condition, 1-3 Enhanced and 4-6 Diminished, so they are shown beside it). "
            "In Experiments 1 and 2 the key press came after the production and reports "
            "the syllable produced, in Experiment 3 it came before the production; the authors analyse it as the perceptual "
            "categorization in all three.",
            "ltas_db is the production measure: the authors' long-term average spectrum of the first 75 ms of the utterance "
            "(measure_window), one level in dB per 100 Hz band from 0 Hz, separated by semicolons and without the authors' zero "
            "padding. It is empty where the authors have no spectrum for the trial, which is also what excluded_by_authors marks: "
            "practice trials, key presses of 3 s or more, trials whose recording gave no spectrum, and all trials of excluded "
            "participants (subject_excluded_by_authors).",
            "Participants are separate in the three experiments (each experiment has its own Gorilla participant numbers, "
            "replaced here by codes), so a person who took part in more than one experiment would have more than one code.",
        ],
        unit_note="trial of one online participant (key press and, where the authors have a spectrum, the production spectrum)",
        differences=[
            "Experiment 2: the paper reports 54 analysed and 16 excluded participants (70 in all); the shared behavioral file "
            "has 71 participants and the authors' notebook excludes 17, which leaves the paper's 54. The paper's 0.9% of trials "
            "lost to recording problems reproduces when the roughly 30 trials that are absent from the behavioral file (one "
            "participant stopped after 212 trials) are counted among the losses (121 of the 12,960 designed trials).",
            f"Experiment 3: the authors' file has a spectrum for only {summary[3]['kept']} of the {summary[3]['crit']} critical "
            f"trials of the {summary[3]['n_an']} analysed participants ({100 * summary[3]['kept'] / summary[3]['crit']:.0f}%), although the paper "
            "reports that only 0.2% of trials were lost to recording problems; the spectra are missing from about half of "
            "every participant's trials, in both conditions and all blocks.",
            f"One analysed participant in Experiment 1 and one in Experiment 3 have no spectrum on any trial, so the authors' "
            f"production analysis has {summary[1]['n_ltas_part']} and {summary[3]['n_ltas_part']} participants where the paper "
            "reports 55 and 57.",
            "The paper's cluster-mass statistics come from random subsamples of the spectra and a permutation test (permuco "
            "in R); they were not recomputed here.",
        ],
    )
    if not ck.passed():
        sys.exit("checks failed")
    return stdlib.release(ID, "speech", rows, meta, ck)


if __name__ == "__main__":
    main()
