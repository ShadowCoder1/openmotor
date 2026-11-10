"""Tremblay, Poulin, Martel-Sauvageau & Denis (2019), Experimental Gerontology 126:110695: age differences in a diadochokinetic
(syllable repetition) task with phonological and sequential complexity manipulated, 85 cognitively healthy adults aged 20 to 93.

Raw files, raw_extra/Tremblay2019_ExpGerontol/ (Borealis doi:10.5683/SP2/RNBELU, CC BY-NC 4.0, fetched with tools/std/fetch.py):
  Data.xlsx          sheet Data: one row per participant (85), 84 columns: SID, file names, Age, Sex (1 = female), Education (years),
                     Degree (1-7), Languages, MoCA, GDS, SRT, PTA_RE, PTA_LE, then for each condition C1..C6 the sequence-level and
                     syllable-level accuracy proportions (CX_Seq_accuracy ... CX_Nb_Syll_Voice, CX_Nb_Syll_Simpl for the complex
                     conditions) and CX_rate, CX_nPVI; sheet Legend: the authors' codebook (read here as text)
  README.docx        the authors' readme (read as text): the TextGrids hold the analysed trials of the .wav files, one interval per
                     trial on the "sequence" tier (text "<stimulus> c<condition> essai <number>"), and on the "syl" tier one point per
                     syllable at its vocalic intensity peak; -999 marks syllables left out of the analyses (noise, speech errors,
                     talk of the experimenter)
  <SID>-DDK-C1-C6.TextGrid   85 files, one per participant; the .wav files (13.5 GB) are not needed and were not downloaded
Run from the site folder: python3 -I tools/std/convert/Tremblay2019_ExpGerontol.py
"""
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tremblay2019_ExpGerontol"
SRC = "Tremblay et al. 2019, Experimental Gerontology"
COND = {1: "one simple syllable", 2: "one complex syllable", 3: "two simple syllables", 4: "two complex syllables",
        5: "three simple syllables", 6: "three complex syllables"}
SEQ = ("For a one-syllable trial the sequence is the syllable; for a longer one it is the whole string of syllables. The authors "
       "first took the mean over the three trials of each stimulus and then the mean of those means over the condition's stimuli.")
SYL = ("The authors first took the mean over the three trials of each stimulus and then the mean of those means over the "
       "condition's stimuli.")
# column suffix (the authors' name) -> (description from the authors' Legend sheet, statistic, unit)
MEAS = {
    "Seq_accuracy": ("Mean proportion of correctly produced sequences. " + SEQ, "mean", "proportion"),
    "Seq_Err": ("Mean proportion of incorrectly produced sequences. " + SEQ, "mean", "proportion"),
    "Err_insertionS": ("Mean proportion of sequences with at least one added syllable. " + SEQ, "mean", "proportion"),
    "Err_omissionS": ("Mean proportion of sequences with at least one omitted syllable. " + SEQ, "mean", "proportion"),
    "Err_invS": ("Mean proportion of sequences with inverted syllables. " + SEQ, "mean", "proportion"),
    "Err_phoneme": ("Mean proportion of sequences with at least one mistaken phoneme. " + SEQ, "mean", "proportion"),
    "Syll_accuracy": ("Mean proportion of correctly produced syllables. " + SYL, "mean", "proportion"),
    "Syll_Err": ("Mean proportion of incorrectly produced syllables (errors other than voicing errors and simplifications of "
                 "consonant clusters). " + SYL, "mean", "proportion"),
    "Nb_Syll_Voice": ("Mean proportion of syllables with voicing errors. " + SYL, "mean", "proportion"),
    "Nb_Syll_Simpl": ("Mean proportion of syllables with simplifications of consonant clusters. " + SYL, "mean", "proportion"),
    "rate": ("Articulation rate in this condition, from semi-automatic analysis of the recordings with the authors' custom scripts. "
             "The deposit does not state the unit.", "value", "unit not stated"),
    "nPVI": ("Stability of the articulation rate in this condition: normalized vocalic pairwise variability index (nPVI), "
             "computed by the authors' custom scripts; higher values mean a less steady rate.", "value", "nPVI index"),
}
PCOLS = [
    ("education_years", "integer", "years", "Number of years of education (the authors' Education)."),
    ("highest_degree", "integer", "1-7",
     "Highest diploma obtained (the authors' Degree): 1 postdoctoral, doctoral or master's degree; 2 bachelor's degree; "
     "3 collegial degree; 4 high school degree; 5 incomplete secondary V; 6 incomplete secondary III; 7 less than 6 years of education."),
    ("n_languages", "integer", "", "Number of spoken languages (the authors' Languages)."),
    ("moca", "integer", "score out of 30",
     "Score on the Montreal Cognitive Assessment; higher scores indicate better cognitive function (the authors' MoCA)."),
    ("gds", "integer", "score out of 30",
     "Score on the Geriatric Depression Scale: 0 to 9 normal, 10 to 19 depression, 20 to 30 severe depression (the authors' GDS)."),
    ("speech_reception_threshold_db_hl", "number", "dB HL", "Speech reception threshold (the authors' SRT). Empty for 2 participants."),
    ("pure_tone_average_right_db_hl", "number", "dB HL",
     "Pure tone average of the right ear over 500, 1000, 2000, 4000 and 6000 Hz (the authors' PTA_RE). The deposit gives it as a "
     "negative number (-61 to -4) that is more negative in older participants (correlation with age -0.73), so a more negative value "
     "means poorer hearing; it is kept as shared. Empty for 1 participant."),
    ("pure_tone_average_left_db_hl", "number", "dB HL",
     "Pure tone average of the left ear over 500, 1000, 2000, 4000 and 6000 Hz (the authors' PTA_LE), negative and more negative in "
     "older participants like the right ear (correlation with age -0.77); kept as shared. Empty for 1 participant."),
]
TG_COLUMNS = [
    ("textgrid_trials.csv", "person", "Participant code, as in the main table."),
    ("textgrid_trials.csv", "start_s", "Start of the trial interval on the TextGrid's sequence tier, in seconds from the start of the recording."),
    ("textgrid_trials.csv", "end_s", "End of the trial interval, in seconds."),
    ("textgrid_trials.csv", "label",
     "The authors' interval text, with leading and trailing spaces removed: stimulus syllables, then c1 to c6 for the condition, then "
     "essai (trial) and the trial number of that stimulus, for example 'pa ta c3 essai 2'. A few labels have typing slips that are kept "
     "as they are (a missing space, a missing trial number, pu for pou)."),
    ("textgrid_syllable_marks.csv", "person", "Participant code, as in the main table."),
    ("textgrid_syllable_marks.csv", "time_s", "Time of the point on the TextGrid's syl tier in seconds from the start of the recording: the "
                                               "vocalic intensity peak of one syllable."),
    ("textgrid_syllable_marks.csv", "mark",
     "The authors' mark of the point, with spaces removed. Empty: the syllable was included in the analyses. -999: the syllable was "
     "left out (noise, a speech error, or the experimenter talking to the participant). A few other marks are typing slips or notes "
     "(exp, tra, -99) and are kept as they are."),
]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def parse_textgrid(txt):
    """Praat long-format TextGrid read as text: [(kind, name, items)], items (xmin, xmax, text) or (time, mark)."""
    lines = txt.splitlines()
    tiers, cur, i = [], None, 0

    def val(l):
        return l.split("=", 1)[1].strip()

    while i < len(lines):
        l = lines[i].strip()
        if l.startswith("class ="):
            cur = [val(l).strip('"'), None, []]
            tiers.append(cur)
        elif l.startswith("name =") and cur is not None and cur[1] is None:
            cur[1] = val(l).strip('"')
        elif l.startswith("intervals [") and cur is not None:
            assert lines[i + 3].strip().startswith("text =")
            cur[2].append((float(val(lines[i + 1])), float(val(lines[i + 2])), val(lines[i + 3]).strip('"').strip()))
            i += 3
        elif l.startswith("points [") and cur is not None:
            assert lines[i + 1].strip().startswith("number =") and lines[i + 2].strip().startswith("mark =")
            cur[2].append((float(val(lines[i + 1])), val(lines[i + 2]).strip('"').strip()))
            i += 2
        i += 1
    return tiers


def to_csv_bytes(header, recs):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(header)
    w.writerows(recs)
    return buf.getvalue().encode("utf-8")


def main():
    raw = stdlib.raw_dir(ID)
    wb = pd.read_excel(raw / "Data.xlsx", sheet_name="Data", nrows=85)
    assert len(wb) == 85 and wb.SID.is_unique and list(wb.columns[:13]) == [
        "SID", "Audiofile", "TextGrid file", "Age", "Sex", "Education", "Degree", "Languages", "MoCA", "GDS", "SRT", "PTA_RE", "PTA_LE"]
    mcols = [c for c in wb.columns if re.match(r"^C[1-6]_(Seq_|Err_|Syll_|Nb_Syll|rate|nPVI)", c)]
    assert len(mcols) == 69 and list(wb.columns[-2:]) == ["Audio", "TextGrid"], (len(mcols), list(wb.columns[-2:]))

    code = {sid: f"{ID}_P{k:04d}" for k, sid in enumerate(wb.SID, start=1)}  # the codes stdlib.release gives, in order of first appearance
    rows, parts = [], []
    for r in wb.itertuples(index=False):
        base = dict(dataset=ID, study=SRC, experiment="", group="", subject=r.SID, person=r.SID, subject_excluded_by_authors=False,
                    age=int(r.Age), sex="F" if int(r.Sex) == 1 else "M", handedness=None, population="healthy", setting="lab",
                    device="audio recording of speech (one .wav file per participant)", session=1, block="", phase="test", time_s=None,
                    task="speech", unit="participant summary", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                    excluded_by_authors=False)
        d = r._asdict()
        k = 0
        for col in mcols:
            c, m = int(col[1]), col[3:]
            v = d[col]
            assert m in MEAS, m
            if pd.isna(v):
                continue
            k += 1
            desc, stat, unit = MEAS[m]
            rows.append({**base, "trial": k, "condition": f"C{c}: {COND[c]}", "measure": m, "measure_description": desc,
                         "statistic": stat, "value": float(v), "value_unit": unit, "n_trials": None})
        parts.append(dict(person=r.SID, education_years=int(r.Education), highest_degree=int(r.Degree), n_languages=int(r.Languages),
                          moca=int(r.MoCA), gds=int(r.GDS), speech_reception_threshold_db_hl=r.SRT, pure_tone_average_right_db_hl=r.PTA_RE,
                          pure_tone_average_left_db_hl=r.PTA_LE))

    # the TextGrids, copied as they are into two extra tables (trial intervals, syllable marks)
    trials, marks = [], []
    per_part = {}
    for r in wb.itertuples(index=False):
        tiers = parse_textgrid((raw / r[2]).read_text(encoding="utf-8", errors="replace"))
        assert [t[0] for t in tiers] == ["IntervalTier", "TextTier"] and tiers[1][1] == "syl", (r.SID, [(t[0], t[1]) for t in tiers])
        n_tr = 0
        for a, b, t in tiers[0][2]:
            if t:
                trials.append([code[r.SID], a, b, t])
                n_tr += 1
        for tt, mk in tiers[1][2]:
            marks.append([code[r.SID], tt, mk])
        per_part[r.SID] = (n_tr, len(tiers[1][2]))
    blob = " ".join(str(x[3]) for x in trials) + " " + " ".join(str(x[2]) for x in marks)
    assert not any(p.search(blob) for p in stdlib.ID_PATTERNS.values()) and not re.search(r"\bS\d{1,3}\b", blob), "identifier-like text in TextGrid labels"
    n_short = {sid: n for sid, (n, _) in per_part.items() if n != 72}
    fmt = lambda x: format(x, ".10g") if isinstance(x, float) else x  # noqa: E731
    extra = [("textgrid_trials.csv", to_csv_bytes(["person", "start_s", "end_s", "label"], [[a, fmt(b), fmt(c), d] for a, b, c, d in trials])),
             ("textgrid_syllable_marks.csv", to_csv_bytes(["person", "time_s", "mark"], [[a, fmt(b), c] for a, b, c in marks])),
             ("textgrid_columns.csv", to_csv_bytes(["file", "column", "description"], TG_COLUMNS))]

    # ---- checks ------------------------------------------------------------------------------------------------------------------
    ck = stdlib.Checks()
    ck.published("participants", len(wb), 85, 0, "Abstract")
    ck.published("youngest participant (years)", int(wb.Age.min()), 20, 0, "Abstract (20-93 years)")
    ck.published("oldest participant (years)", int(wb.Age.max()), 93, 0, "Abstract (20-93 years)")
    n_cells = int(wb[mcols].notna().sum().sum())
    ck.internal("one row per participant, condition and measure for every filled cell of the workbook", len(rows) == n_cells,
                f"{len(rows)} rows from {n_cells} cells")
    ck.internal("85 participants, none lost", len({r['subject'] for r in rows}) == 85 and len(parts) == 85)
    ck.internal("sex is coded 0 or 1 and age is a whole number of years", set(wb.Sex) <= {0, 1} and (wb.Age == wb.Age.round()).all())
    for c in range(1, 7):
        p = f"C{c}_"
        ck.internal(f"C{c}: sequence accuracy plus sequence errors = 1 and syllable accuracy plus syllable errors = 1 (within 0.002)",
                    bool(np.allclose(wb[p + "Seq_accuracy"] + wb[p + "Seq_Err"], 1, atol=2e-3)
                         and np.allclose(wb[p + "Syll_accuracy"] + wb[p + "Syll_Err"], 1, atol=2e-3)))
    prop = [c for c in mcols if c[3:] not in ("rate", "nPVI")]
    ck.internal("all proportions lie between 0 and 1; rates are positive; nPVI is between 0 and 100",
                bool(((wb[prop].stack() >= 0) & (wb[prop].stack() <= 1)).all()
                     and (wb[[c for c in mcols if c.endswith("_rate")]].stack() > 0).all()
                     and ((wb[[c for c in mcols if c.endswith("_nPVI")]].stack() >= 0) & (wb[[c for c in mcols if c.endswith("_nPVI")]].stack() <= 100)).all()))
    r_pta = [float(np.corrcoef(wb.Age[wb[c].notna()], wb[c].dropna())[0, 1]) for c in ("PTA_RE", "PTA_LE")]
    ck.internal("hearing thresholds (PTA) are all negative and fall with age as the deposit's sign convention implies (r about -0.7)",
                bool((wb.PTA_RE.dropna() < 0).all() and (wb.PTA_LE.dropna() < 0).all() and max(r_pta) < -0.6),
                f"r with age {r_pta[0]:.2f} (right), {r_pta[1]:.2f} (left)")
    r_npvi = [float(wb[f"C{c}_nPVI"].corr(wb.Age)) for c in range(1, 7)]
    ck.internal("abstract: rate stability declines with age, so the instability index nPVI correlates positively with age in at least "
                "5 of the 6 conditions (raw correlations; the paper controls for cognition, hearing and depression)",
                sum(r > 0 for r in r_npvi) >= 5, f"nPVI r = {[round(r, 2) for r in r_npvi]}")
    ck.internal("85 TextGrids read, each with a trial tier and a syllable tier", len(per_part) == 85)
    ck.internal("most TextGrids hold 72 labelled trials (24 stimuli x 3)", sum(1 for n, _ in per_part.values() if n == 72) >= 70,
                f"{sum(1 for n, _ in per_part.values() if n == 72)} of 85 have 72; others: {sorted(n_short.values())}")
    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)
    print("rows", len(rows), "| trials", len(trials), "| marks", len(marks), "| TextGrids without 72 trials:", sorted(n_short.values()),
          "| marks coded -999:", sum(1 for m in marks if m[2] == "-999"), "| other marks:", sorted({m[2] for m in marks} - {"", "-999"}))

    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY-NC 4.0",
        license_url="http://creativecommons.org/licenses/by-nc/4.0",
        changes=("Stacked the workbook's values into long form (one row per participant, condition and measure, in the order of the "
                 "authors' columns), moved age and sex into the standard columns and the education, cognitive, depression and hearing "
                 "scores into participants.csv, dropped the workbook's file-name columns (they contain participant numbers), copied the "
                 "85 TextGrids into two extra tables (trial intervals and syllable marks, spaces at the start and end of labels removed), "
                 "and replaced participant numbers with codes."),
        how=[
            "One row is one measure of one participant in one of the six conditions of a syllable-repetition (diadochokinetic) task: "
            "one, two or three syllables, each simple (for example pa, ta ka) or complex (for example pra, tra kra). The deposit has one "
            "value per participant, condition and measure; each is the authors' mean over the condition's stimuli (three trials each).",
            "Accuracy measures are proportions of sequences or syllables produced correctly or with a given error, from the authors' "
            "transcription of the recordings; rate is the articulation rate (the deposit gives no unit) and nPVI the authors' index of how "
            "unsteady the rate is, both from their scripts. Age and sex are in the standard columns; education, MoCA, GDS and hearing "
            "scores are in participants.csv.",
            "textgrid_trials.csv has one row per trial interval of each participant's TextGrid (start, end and the authors' label) and "
            "textgrid_syllable_marks.csv one row per syllable point (vocalic intensity peak time and the authors' mark, -999 for syllables "
            "they left out); textgrid_columns.csv describes both. They are the authors' segmentation, copied without re-analysis.",
            "The paper is not open access: only its abstract (85 adults aged 20 to 93) was available for checking, so the workbook values "
            "themselves were not compared with the paper's tables.",
        ],
        unit_note="measure of one participant in one condition (long form)",
        differences=[
            f"Most TextGrids hold the 72 labelled trials of the design (24 stimuli x 3 trials); {len(n_short)} hold fewer "
            f"({', '.join(str(n) for n in sorted(n_short.values()))}) and a few trial labels have typing slips (kept as they are).",
        ],
    )
    if not ck.passed():
        sys.exit("checks failed")
    rec = stdlib.release(ID, "summary", rows, meta, ck, participants=parts, participant_columns=PCOLS, extra_files=extra)
    # the participant codes written into the extra tables must be the ones stdlib gave in the main table
    with zipfile.ZipFile(stdlib.OUT / f"{ID}.zip") as z:
        main_csv = list(csv.reader(io.TextIOWrapper(z.open(f"{ID}/{ID}.csv"), encoding="utf-8", newline="")))
    pi = main_csv[0].index("person")
    order = list(dict.fromkeys(r[pi] for r in main_csv[1:]))
    assert order == [code[s_] for s_ in wb.SID], "participant codes in the extra tables differ from the main table"
    return rec


if __name__ == "__main__":
    main()
