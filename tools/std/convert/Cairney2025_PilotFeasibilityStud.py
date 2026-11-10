"""Cairney, Crampin, Campbell & Cleland (2025), Pilot Feasib Stud 11(1):61: "Quantitative results of SonoSpeech Cleft Pilot: a
mixed-methods pilot randomised control trial of ultrasound visual biofeedback versus standard intervention for children with cleft
palate +/- cleft lip". 19 children said target word lists in assessments (baseline, 1 week and 1 month after) and in 6 weekly
therapy sessions (ultrasound visual biofeedback or articulation therapy); blinded listeners scored the target consonant of each word.
One row is one word said.

Raw files, raw_extra/Cairney2025_PilotFeasibilityStud/ (OSF gkd3h, node licence CC-By Attribution 4.0 International):
  PCC_digitised_tx_20241203-anon-OSF.csv  the authors' per-word scores (11139 rows; the 3 Dec 2024 version after a spreadsheet error was found)
  PCC_wrangling_clean-OSF-241209.R        their script (read as text): row filters, the repeat of two-target rows, Table 5 and the appendix table
  Table5_Appendix_PCC_per_participant_per_group.csv  their mean PCC per child and session (used as a check)
Not converted: questionnaires and standardised scores (Scores_..., Fidelity, ESQ, Qualtrics), recording file names, transcriber initials/codes.
Run from the site folder: python3 -I tools/std/convert/Cairney2025_PilotFeasibilityStud.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cairney2025_PilotFeasibilityStud"
SRC = "Cairney et al. 2025, Pilot Feasib Stud"
BAD = ["SS_Ax1_14_F_04_Error_specific_trimmed.WAV", "SS_Ax2_14_F_04_Error_specific_trimmed.wav", "SS_Ax3_14_F_04_Error_specific_td_trimmed.wav",
       "SS_Ax1_11_M_07_Error_specific_trimmed.WAV", "SS_Ax1_15_M_07_Error_specific_PCC_trimmed.wav", "SS_Ax2_15_M_07_Error_specific_wrong_trimmed.wav",
       "SS_Ax1_07_M_08_Error_specific_trimmed.wav", "SS_Ax2_07_M_08_Error_specific_trimmed.wav", "SS_Ax3_07_M_08_Error_specific_trimmed.wav",
       "SS_Tx_18_M_11session6.wav"]  # recordings of the wrong target list, removed by the authors' script
R = stdlib.raw_dir(ID)
ck = stdlib.Checks()
d = pd.read_csv(R / "PCC_digitised_tx_20241203-anon-OSF.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
ck.internal("11139 rows in file order (the order column runs 1..11139), 19 children, 157 recordings",
            len(d) == 11139 and (d.order.astype(int).values == np.arange(1, len(d) + 1)).all() and d.participant.nunique() == 19 and d.filename.nunique() == 157)
SESS = {("Ax", 1): 1, **{("Tx", i): 1 + i for i in range(1, 7)}, ("Ax", 2): 8, ("Ax", 3): 9}  # baseline, therapy 1-6, 1 week, 1 month
PHASE = {("Ax", 1): "baseline", ("Ax", 2): "test", ("Ax", 3): "test"}


def num(s):
    s = s.strip()
    return int(s) if s in ("0", "1", "2") else None


reason = []
for r in d.itertuples(index=False):
    q = []
    if r.word_type == "#N/A":
        q.append("word type not defined (#N/A in the authors' table)")
    if r.score_glottal_correct_added == "#N/A":
        q.append("score #N/A")
    if r.score_binomial == "#VALUE!":
        q.append("binomial score is a spreadsheet error (#VALUE!)")
    if r.filename in BAD:
        q.append("recording of the wrong target word list (removed by the authors' script)")
    reason.append("; ".join(q))
d["reason"] = reason
ctr = {}
rows = []
for r, why in zip(d.itertuples(index=False), d.reason):
    st, sn = r.session_type, int(r.session_number)
    p = r.participant
    ctr[p] = ctr.get(p, 0) + 1
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group=r.treatment, subject=p, person=p, subject_excluded_by_authors=False, age="", sex="",
        handedness="", population="children with cleft palate with or without cleft lip", setting="clinic",
        device="audio recordings of the child saying a word list targeting one consonant (assessments at the university or online; therapy sessions in a hospital clinic)",
        session=SESS[(st, sn)], block=f"{st}{sn}", trial=ctr[p], phase=PHASE.get((st, sn), "learning"), time_s="", task="speech", unit="utterance",
        condition="" if r.word_type == "#N/A" else r.word_type, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
        excluded_by_authors=bool(why), exclusion_reason=why, stimulus=r.word, segment=f"target consonant /{r.target}/",
        authors_trial_number=int(r.word_no), targets_correct=num(r.score_glottal_correct_added), targets_correct_strict=num(r.__getattribute__("_" + str(d.columns.get_loc("score_glottal-incorrect")))) if False else None,
        target_correct_binary=num(r.score_binomial), targets_possible=r.max_score.replace("\xa0", " ").strip()))
out = pd.DataFrame(rows)
out["targets_correct_strict"] = [num(v) for v in d["score_glottal-incorrect"]]
# the authors' own pipeline (their script): filters, rows with two correct targets counted twice
k = d[(d.word_type != "#N/A") & (d.score_glottal_correct_added != "#N/A") & (d.score_binomial != "#VALUE!") & ~d.filename.isin(BAD)].copy()
k["b"] = pd.to_numeric(k.score_binomial, errors="coerce")
k["g"] = pd.to_numeric(k.score_glottal_correct_added, errors="coerce")
k2 = pd.concat([k[k.g != 2], k[k.g == 2].loc[k[k.g == 2].index.repeat(2)]])
ck.internal("rows left after the authors' filters: 7436 (the rest are flagged as excluded)", len(k) == 7436 and int((~out.excluded_by_authors).sum()) >= 7436 - 20,
            f"{len(k)} kept by the script, {int((~out.excluded_by_authors).sum())} not flagged")
app = pd.read_csv(R / "Table5_Appendix_PCC_per_participant_per_group.csv", dtype={"participant": str})
mine = (k2.groupby(["treatment", "participant", "session_type", "session_number"]).b.mean().mul(100).round(2).reset_index())
mine["participant"] = mine.participant.astype(int).astype(str)
app["participant"] = app.participant.astype(int).astype(str)
app["session_number"] = app.session_number.astype(int)
mine["session_number"] = mine.session_number.astype(int)
mm = app.merge(mine, on=["treatment", "participant", "session_type", "session_number"], how="outer", indicator=True)
ck.internal("the authors' appendix table (mean PCC per child, arm and session) is reproduced from the converted rows",
            bool((mm._merge == "both").all() and np.allclose(mm.mean1, mm.b, atol=0.011)), f"{len(app)} cells")
P = "Cairney et al. 2025"
ppl = out.drop_duplicates("person")
ck.published("children randomised and analysed", len(ppl), 19, 0, f"{P}, Results / Table 5 and Exploratory (19 participants)")
ck.published("children in the articulation therapy arm", (ppl.group == "articulation").sum(), 8, 0, f"{P}, Abstract (eight to articulation therapy)")
ck.published("children in the ultrasound arm", (ppl.group == "ultrasound").sum(), 11, 0, f"{P}, Abstract (eleven to ultrasound)")
T5 = {  # Table 5, candidate primary outcome: (arm, session type, number): (mean, SD, n)
    ("articulation", "Ax", 1): (13, 33, 6), ("ultrasound", "Ax", 1): (14, 35, 9),
    ("articulation", "Tx", 1): (16, 37, 8), ("ultrasound", "Tx", 1): (19, 39, 9), ("articulation", "Tx", 2): (21, 41, 6), ("ultrasound", "Tx", 2): (25, 43, 10),
    ("articulation", "Tx", 3): (27, 44, 8), ("ultrasound", "Tx", 3): (14, 35, 11), ("articulation", "Tx", 4): (26, 44, 8), ("ultrasound", "Tx", 4): (14, 35, 11),
    ("articulation", "Tx", 5): (26, 44, 7), ("ultrasound", "Tx", 5): (29, 46, 8), ("articulation", "Tx", 6): (12, 32, 6), ("ultrasound", "Tx", 6): (21, 41, 8),
    ("articulation", "Ax", 2): (43, 50, 6), ("ultrasound", "Ax", 2): (22, 41, 10), ("articulation", "Ax", 3): (39, 49, 7), ("ultrasound", "Ax", 3): (37, 48, 9)}
LBL = {("Ax", 1): "baseline", ("Ax", 2): "1 week after", ("Ax", 3): "1 month after", **{("Tx", i): f"therapy session {i}" for i in range(1, 7)}}
for (arm, st, sn), (mu, sd, n) in T5.items():
    g = k2[(k2.treatment == arm) & (k2.session_type == st) & (k2.session_number == str(sn))]
    w = f"{arm}, {LBL[(st, sn)]}"
    S = f"{P}, Table 5, percentage targeted consonants correct"
    ck.published(f"PCC mean (%), {w}", 100 * g.b.mean(), mu, 0.5, S)
    ck.published(f"PCC SD (%), {w}", 100 * g.b.std(), sd, 0.5, S)
    ck.published(f"children analysed, {w}", g.participant.nunique(), n, 0, S)
for c in ck.items:
    if not c["ok"]:
        print("FAILED:", c)
pp = d.drop_duplicates("participant")
people = [dict(person=r.participant, age_group=r.age_group, therapy_target=r.target) for r in pp.itertuples()]
PCOLS = [("age_group", "category", "four-seven, eight-eleven, twelve-sixteen", "Age stratum used for randomisation (4;6-7;11, 8;0-11;11, 12;0-16;0 years); exact ages are not shared."),
         ("therapy_target", "category", "t, s, k", "Target consonant of the child's word list and therapy.")]
meta = dict(
    paper=dict(authors="Cairney M, Crampin L, Campbell L, Cleland J", year=2025,
               title="Quantitative results of SonoSpeech Cleft Pilot: a mixed-methods pilot randomised control trial of ultrasound visual biofeedback versus standard intervention for children with cleft palate +/- cleft lip",
               venue="Pilot Feasib Stud 11(1):61", doi="10.1186/s40814-025-01640-6"),
    lab="Cleland (University of Strathclyde)", data_url="https://osf.io/gkd3h/", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept all 11139 rows of the authors' per-word table in order and mapped its scores to the speech format. Rows the authors' script "
             "removes (word type #N/A, spreadsheet errors in the binomial score, ten recordings of the wrong word list) are kept and flagged. "
             "Left out: recording file names, the transcribers' initials and codes, the batch labels and the questionnaires. Child numbers are replaced by codes. "
             "The authors' script repeats rows with two correct targets for their binomial model; the repeat is not done here."),
    how=["One row is one word said by a child from the list that targets the child's consonant (t, s or k), scored by a blinded listener. "
         "session numbers the sessions in time order: 1 baseline assessment, 2-7 the six weekly therapy sessions, 8 the assessment 1 week after "
         "therapy and 9 the one 1 month after; block gives the authors' labels (Ax = assessment, Tx = therapy session). group is the therapy arm "
         "(ultrasound visual biofeedback or articulation therapy). condition is the authors' word type (treated or untreated word).",
         "targets_correct counts the targets correct with word-medial glottal /t/ marked correct (the authors' re-scoring); targets_correct_strict is the "
         "listener's original score; target_correct_binary is the 0/1 score used for the percentage of target consonants correct (PCC). "
         "trial counts the child's rows in the authors' table order, which follows the order of transcription batches, not time; "
         "authors_trial_number is the position of the word in its recording. Ages are not shared (age_group in participants.csv)."],
    unit_note="word said by a child, with the blinded listener's score of its target consonant",
    differences=["Table 5 of the paper is reproduced for all 18 cells (means, SDs and numbers of children). The rows of the deposit's older versions "
                 "differ (a spreadsheet error found on 3 December 2024 affected 39 recordings); this is the corrected version."],
)
if not ck.passed():
    sys.exit("checks failed")
stdlib.release(ID, "speech", out.to_dict("records"), meta, ck, participants=people, participant_columns=PCOLS)
