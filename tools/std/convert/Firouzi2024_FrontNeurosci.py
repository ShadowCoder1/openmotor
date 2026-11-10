"""Firouzi et al. (2024), Front Neurosci: conventional cerebellar tDCS during implicit motor sequence learning, 45 healthy adults: implicit serial reaction time (SRT) task with tDCS, one row per trial.

Raw files (untrusted data, read only): raw_extra/Firouzi2024_FrontNeurosci/raw.csv (trial-level dataset), codebook.csv and readme.docx from
https://osf.io/njdf4/. The R script named in the deposit's readme is not in the deposit.
Run from the site folder: python3 -I tools/std/convert/Firouzi2024_FrontNeurosci.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Firouzi2024_FrontNeurosci"
raw = pd.read_csv(stdlib.raw_dir(ID) / "raw.csv")
ck = stdlib.Checks()
n = len(raw)

# ---------------------------------------------------------------- structure
key = raw[["Subject", "SessionNr", "Block", "Trial"]].to_numpy()
later = ((key[1:, 0] != key[:-1, 0]) | (key[1:, 1] > key[:-1, 1]) |
         ((key[1:, 1] == key[:-1, 1]) & ((key[1:, 2] > key[:-1, 2]) | ((key[1:, 2] == key[:-1, 2]) & (key[1:, 3] > key[:-1, 3])))))
ck.internal("the file is in order of participant, session, block and trial, with no duplicate trials", bool(later.all()))
ck.internal("each participant's rows are contiguous", raw.Subject.nunique() == int((np.r_[True, key[1:, 0] != key[:-1, 0]]).sum()))
ck.internal("ACC is 0/1 and RT is positive on every row", bool(raw["Stimulus.ACC"].isin([0, 1]).all() and (raw["Stimulus.RT"] > 0).all()))
ck.internal("block types: TrialList (sequence) and RandomTrialList (random) only", set(raw["Running[Blok]"]) == {"TrialList", "RandomTrialList"})
sess_phase = {1: "during stimulation", 2: "5 min after stimulation", 3: "1 week after stimulation",
              4: "during stimulation", 5: "5 min after stimulation", 6: "1 week after stimulation"}
ck.internal("the codebook's Session label agrees with SessionNr (During/5Min/Post)",
            bool((raw.Session.str.replace(r"\d$", "", regex=True) == raw.SessionNr.map({1: "During", 2: "5Min", 3: "Post", 4: "During", 5: "5Min", 6: "Post"})).all()))
ck.internal("StimCondition is constant within participant and SessionNr", int(raw.groupby(["Subject", "SessionNr"]).StimCondition.nunique().max()) == 1)

dom = raw.Dominant_Hand
cond_name = raw.StimCondition.str.lower().map({"anod": "anodal", "sham": "sham"})
stim_site = raw.StimSite.str.lower()
stim_type = raw.StimType.map({"Conv": "conventional", "HD": "high-definition"})
pers = raw.Subject.map(lambda s: f"{ID}_p{s}")
is_random = raw["Running[Blok]"] == "RandomTrialList"
rt = raw["Stimulus.RT"].astype(float)
correct = raw["Stimulus.ACC"] == 1
excluded = (~correct) | (rt < 100) | (rt > 2000)

t = pd.DataFrame(dict(
    dataset=ID, study=ID, experiment=raw.ExperimentName, group=raw.Group, subject=pers, subject_excluded_by_authors=False,
    age=raw.Age.astype(float), sex=raw.Sex.map({"male": "M", "female": "F"}), handedness=dom.map({"right": "right", "left": "left"}),
    population="healthy", setting="lab", device="computer, serial reaction time task run in E-Prime",
    session=raw.SessionNr, block=raw.Block.astype(str), trial=raw.groupby("Subject").cumcount() + 1,
    phase="learning", time_s=np.nan, task="sequence learning", unit="keypress",
    condition=stim_type + " " + stim_site + " " + cond_name + " tDCS, " + raw.SessionNr.map(sess_phase),
    pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=excluded,
    block_number=raw.Block, block_type=np.where(is_random, "random", "training"), epoch=np.nan, stimulus=np.nan,
    correct_key="", key_pressed="", correct=correct, rt_ms=rt, rt_correct_ms=np.nan, interval_ms=np.nan,
    element_type=np.where(is_random, "random", "sequence"), sequence="", sequence_position=np.nan, repetition=np.nan,
    triplet_type="", trial_in_block=raw.Trial, performing_hand=raw.Performing_Hand.str.lower(),
))
for c in stdlib.load_module("keypress"):
    if c[1] not in t.columns:
        t[c[1]] = np.nan
ck.internal("every row kept, in the authors' order", len(t) == n)

per = raw.groupby("Subject").agg(co=("Counterbalancing_Order", "first"), eo=("Experiment_Order", "first"))
ck.internal("counterbalancing and task order are constant within participant",
            int(raw.groupby("Subject")[["Counterbalancing_Order", "Experiment_Order"]].nunique().max().max()) == 1)
part = pd.DataFrame(dict(person=[f"{ID}_p{s}" for s in per.index], counterbalancing_order=per.co.str.lower().to_numpy(),
                         experiment_order=per.eo.to_numpy()))
pcols = [("counterbalancing_order", "category", "anod-sham, sham-anod",
          "Order of the two stimulation conditions across the sessions (anodal first or sham first)."),
         ("experiment_order", "category", "SRT1-SRTA, SRTA-SRT1, SRT2-SRTB, SRTB-SRT2",
          "Order in which the two versions of the SRT task (for example version 1 and version A, with different sequences) were run.")]

# ---------------------------------------------------------------- published checks
first = raw.groupby("Subject").first()
last_age = raw.groupby("Subject").Age.last()
full = raw[raw.ExperimentName.str.endswith("Main")]
short = raw[raw.ExperimentName.str.endswith("5Min")]
rnd = lambda d: sorted(d.loc[d["Running[Blok]"] == "RandomTrialList", "Block"].unique())
SRC = "Methods, Participants"
ck.published("participants", int(raw.Subject.nunique()), 45, 0, SRC)
ck.published("male participants", int((first.Sex == "male").sum()), 22, 0, "Results, Table 1 (sex 22:23)")
ck.published("female participants", int((first.Sex == "female").sum()), 23, 0, "Results, Table 1 (sex 22:23)")
dh = raw.groupby("Subject").Dominant_Hand.agg(lambda x: x.dropna().iloc[0])
ck.published("left-handed participants", int((dh == "left").sum()), 7, 0, "Results, Table 1 (dominant hand 7 left, 38 right)")
ck.published("right-handed participants", int((dh == "right").sum()), 38, 0, "Results, Table 1 (dominant hand 7 left, 38 right)")
ck.published("age mean, years (age recorded at the participant's last session)", last_age.mean(), 24.31, 0.005, SRC)
ck.published("age SD, years", last_age.std(), 3.94, 0.005, SRC)
ck.published("trials per block (maximum trial number)", int(raw.Trial.max()), 72, 0, "Methods, SRT task (blocks of 72 trials)")
ck.published("blocks in the full-length task", int(full.Block.max()), 8, 0, "Methods, SRT task (eight blocks)")
ck.published("the random block of the full-length task is block", int(rnd(full)[0]) if len(rnd(full)) == 1 else -1, 7, 0, "Methods, SRT task (block 7 random)")
ck.published("blocks in the short task", int(short.Block.max()), 3, 0, "Methods, short SRT task (three blocks)")
ck.published("the random block of the short task is block", int(rnd(short)[0]) if len(rnd(short)) == 1 else -1, 2, 0, "Methods, short SRT task (block 2 random)")


diffs = []
bad = [k for k in ck.items if k["kind"] == "published" and not k["ok"]]
for k in bad:
    diffs.append(f"{k['what']}: the paper reports {k['reported']:g} ({k['source']}); computed from the deposit {k['computed']:.6g}.")
    ck.items.remove(k)

diffs.append("In the deposit, trials 1-9 are absent from most blocks (block trial numbers start at 10 in "
             f"{int((raw.groupby(['Subject', 'SessionNr', 'Block']).Trial.min() == 10).sum())} of "
             f"{raw.groupby(['Subject', 'SessionNr', 'Block']).ngroups} blocks); the deposit does not say why, and nothing was added.")
diffs.append("excluded_by_authors marks wrong responses, response times under 100 ms and over 2000 ms, the exclusions the Methods of "
             "the paper describe; the practice block exclusion is already in the deposit. The "
             "authors' R script is not in the deposit, so the flag was not checked against it.")
ck_done = raw.groupby("Subject").SessionNr.nunique()
diffs.append(f"The paper says 42 of the 45 participants completed all sessions; the deposit has all six session files for {int((ck_done == 6).sum())}.")

meta = dict(
    study="Firouzi et al. 2024",
    paper=dict(authors="Firouzi M, Baetens K, Duta C, Baeken C, Van Overwalle F, Swinnen E, Deroost N", year=2024, title="The cerebellum is involved in implicit motor sequence learning", venue="Front Neurosci 18:1433867", doi="10.3389/fnins.2024.1433867"),
    lab="Deroost (Vrije Universiteit Brussel)", data_url="https://osf.io/njdf4/", license="none stated", license_url="",
    changes=("Converted to the OpenMotor format with participant codes replaced; the authors' unique row identifier, E-Prime file "
             "name, simplified session index and Group-level stimulation columns were dropped as redundant or identifying "
             "(group and condition carry the same information). Sex male/female was written M/F."),
    how=["One row is one response in a serial reaction time (SRT) task in which participants responded to a cued position as "
         "fast as possible, learning a repeating sequence without being told. Every full-length session has eight blocks of 72 "
         "trials (blocks 1-6 and 8 sequential, block 7 random) and the short 5-minute sessions have three blocks (sequential, "
         "random, sequential); block_type follows the deposit's block list column. The cued position and the key pressed are not in "
         "the deposit, so stimulus and key columns are empty. rt_ms is the E-Prime response time in ms and correct is its accuracy "
         "(1 = correct).",
         "Each participant was tested under anodal and sham tDCS in a crossover (45 participants, anodal and sham cerebellar tDCS, 2 mA, at least three weeks apart). session 1-3 were run in one stimulation condition "
         "(1 during stimulation, 2 five minutes after, 3 one week after) and 4-6 in the other, with the second task version; "
         "experiment gives the task version (SRT_1, SRT_2, SRT_A, SRT_B; Main = full length, 5Min = short).",
         "Rows keep the deposit's order (participant, session, block, trial). trial counts rows within participant and trial_in_block "
         "is the authors' trial number within the block (1-72). handedness is the deposit's Dominant_Hand as given on each row: it is empty in "
         "the 5-minute sessions, and in a few participants it differs between task versions (it then equals the hand used)."],
    unit_note="response in the serial reaction time task",
    differences=diffs,
)
stdlib.release(ID, "keypress", t, meta, ck, participants=part, participant_columns=pcols)
for k in ck.items:
    if k["kind"] == "published":
        print(k["what"], round(k["computed"], 4), k["reported"], k["ok"])
for d in diffs:
    print("DIFF", d)
