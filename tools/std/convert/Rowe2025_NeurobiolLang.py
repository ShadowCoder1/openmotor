"""Rowe, Miller, Belisle, Perrachione & Guenther (2025), Neurobiology of Language 6:NOL.a.18: speech motor learning of pseudowords with
non-native consonant clusters after anodal HD-tDCS of the left posterior inferior frontal sulcus, the right cerebellum, or sham.

Raw file, raw_extra/Rowe2025_NeurobiolLang/Data_NOL2025.zip (OSF jvdea, no license stated; the paper is CC BY 4.0):
  TDSUB###_Experiment_Results.csv  108 files, one per participant (numbers 1-109 except 14), 160 test-phase trials each, columns Subject
     (participant_set_Test_trial_stimulus), TrainedNovel, Trial, Stimulus, ReactionTime (s), Accuracy (1 accurate, 0 error, empty =
     not scorable), SeqAccuracy, OnsetAccuracy, OffsetAccuracy, OnsetErrorType, OffsetErrorType (numeric codes, no codebook),
     Duration (s), Comments (the scorer's transcription of what was said, or a note such as 'no response').
The deposit does not say which stimulation condition each participant had, so group stays empty. Not carried: SeqAccuracy and the two
error-type code columns (meaning not documented in the deposit or the paper's main text).
Run from the site folder: python3 -I tools/std/convert/Rowe2025_NeurobiolLang.py
"""
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

ID = "Rowe2025_NeurobiolLang"
SRC = "Rowe et al. 2025, Neurobiology of Language"
RAW = stdlib.raw_dir(ID)
DEV = "microphone recordings of cued pseudoword repetition in a sound booth (GO tone after the stimulus)"


def blank(v):
    return "" if pd.isna(v) else v


def main():
    checks = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "Data_NOL2025.zip")
    parts = []
    for n in sorted(z.namelist()):
        m = re.fullmatch(r"TDSUB(\d+)_Experiment_Results\.csv", n)
        assert m, n
        df = pd.read_csv(io.BytesIO(z.read(n)))
        df["sid"] = int(m.group(1))
        df["itemset"] = int(re.search(r"_(\d)_Test_", df.Subject.iloc[0]).group(1))
        parts.append(df)
    a = pd.concat(parts, ignore_index=True)
    rows = []
    for r in a.itertuples(index=False):
        code = f"P{r.sid:03d}"
        rows.append(dict(
            dataset=ID, study=SRC, experiment=f"item set {r.itemset}", group="", subject=code, person=code,
            subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab", device=DEV,
            session=1, block=f"test block {(int(r.Trial) - 1) // 20 + 1}", trial=int(r.Trial), phase="test", time_s="", task="speech",
            unit="utterance", condition=r.TrainedNovel, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=False, stimulus=r.Stimulus.lower(), segment="whole pseudoword", duration_ms=r.Duration * 1000,
            reaction_time_ms=r.ReactionTime * 1000, response_transcription=blank(r.Comments).strip(),
            word_correct=blank(r.Accuracy) if pd.isna(r.Accuracy) else bool(int(r.Accuracy)),
            onset_cluster_correct="" if pd.isna(r.OnsetAccuracy) else bool(int(r.OnsetAccuracy)),
            offset_cluster_correct="" if pd.isna(r.OffsetAccuracy) else bool(int(r.OffsetAccuracy))))
    b = a.dropna(subset=["Accuracy"])
    g = a.groupby("sid")
    # published checks
    checks.published("test trials per participant (80 trained + 80 novel)", g.size().mean(), 160, 0, f"{SRC}, Methods (Testing)")
    checks.published("trained trials per participant", g.TrainedNovel.apply(lambda s: (s == "trained").sum()).mean(), 80, 0, f"{SRC}, Methods (Testing)")
    checks.published("repetitions of each trained stimulus", a[a.TrainedNovel == "trained"].groupby(["sid", "Stimulus"]).size().mean(), 20, 0,
                     f"{SRC}, Methods (Testing)")
    checks.published("repetitions of each novel stimulus", a[a.TrainedNovel == "novel"].groupby(["sid", "Stimulus"]).size().mean(), 8, 0,
                     f"{SRC}, Methods (Testing)")
    checks.published("number of trained stimuli", a[a.TrainedNovel == "trained"].groupby("sid").Stimulus.nunique().mean(), 4, 0, f"{SRC}, Methods (Training)")
    checks.published("number of novel stimuli", a[a.TrainedNovel == "novel"].groupby("sid").Stimulus.nunique().mean(), 10, 0, f"{SRC}, Methods (Testing)")
    # Table 3: the three conditions have 35 participants each, so the pooled mean is the mean of the three condition means
    checks.published("accuracy, trained stimuli (mean of the three Table 3 condition means 0.42, 0.35, 0.27)",
                     b[b.TrainedNovel == "trained"].Accuracy.mean(), (0.42 + 0.35 + 0.27) / 3, 0.01, f"{SRC}, Table 3")
    checks.published("accuracy, novel stimuli (0.37, 0.30, 0.29)", b[b.TrainedNovel == "novel"].Accuracy.mean(), (0.37 + 0.30 + 0.29) / 3, 0.01, f"{SRC}, Table 3")
    checks.published("duration, trained stimuli, s (0.64, 0.65, 0.62)", a[a.TrainedNovel == "trained"].Duration.mean(), (0.64 + 0.65 + 0.62) / 3, 0.01, f"{SRC}, Table 3")
    checks.published("duration, novel stimuli, s (0.65, 0.67, 0.64)", a[a.TrainedNovel == "novel"].Duration.mean(), (0.65 + 0.67 + 0.64) / 3, 0.01, f"{SRC}, Table 3")
    checks.internal("108 participants, 17,280 trials", a.sid.nunique() == 108 and len(a) == 17280)
    checks.internal("an accurate word has accurate onset and offset clusters", bool(((b.Accuracy == 0) | ((b.OnsetAccuracy == 1) & (b.OffsetAccuracy == 1))).all()))
    checks.internal("trial numbers 1-160 in each file, in order", bool(g.Trial.apply(lambda s: list(s) == list(range(1, 161))).all()))
    checks.internal("each participant has one item set", bool((g.itemset.nunique() == 1).all()))
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the 108 per-participant files in order, converted duration and reaction time from seconds to ms, turned the "
                 "0/1 accuracy columns into true/false (empty where the authors' column is empty), recoded participant numbers as codes, and "
                 "left out SeqAccuracy and the numeric error-type columns, whose meaning the deposit does not give."),
        how=["One row is one test-phase utterance: a participant repeated a pseudoword with a non-native consonant cluster after a GO tone "
             "(160 per participant: 80 of the 4 trained pseudowords, 20 each, and 80 of 10 novel pseudowords, 8 each, in random order, "
             "in two runs of four 20-trial blocks; block is derived from the trial number by that plan). condition is trained or novel. "
             "No stimulation or feedback perturbation is applied in these rows; the stimulation was given earlier, during training.",
             "word_correct is the authors' Accuracy (a blinded rater scored a production 1 if it contained none of the listed errors such as "
             "deletion, insertion, substitution, wrong order or vowel epenthesis); onset_cluster_correct and offset_cluster_correct are their "
             "OnsetAccuracy and OffsetAccuracy (meaning taken from the column names). duration_ms is the whole-word duration from "
             "intensity thresholds; the paper analysed it only for productions with no error or with only an inserted vowel, but the "
             "deposit gives it for every trial. reaction_time_ms is the deposit's ReactionTime (the paper does not analyse or define it; "
             "negative values occur mostly on trials with no speech). response_transcription is the scorer's note of what was said.",
             "experiment is the item-set counterbalancing number 1 or 2 from the deposit's trial labels: the two sets swap which pseudowords "
             "are trained. The participant's stimulation condition (LpIFS, RCbm or sham, 35 each in the paper) is not in the deposit, so group "
             "is empty and the trained-versus-novel comparison by stimulation cannot be reproduced from these rows."],
        unit_note="utterance", differences=[
            "The paper analysed N = 105 participants (35 per stimulation condition); the deposit holds 108 participants (numbers 1 to 109 "
            "without 14) and does not say which three were left out, so no row is marked excluded.",
            "Table 3 means are given per stimulation condition; only the pooled means (mean of the three condition means) can be compared, "
            "within 0.01."])
    stdlib.release(ID, "speech", rows, meta, checks)


main()
