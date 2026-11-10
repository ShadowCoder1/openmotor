"""Murphy, Nozari & Holt (2023), Psychon Bull Rev 31(3):1193-1205: 87 online participants passively hear sequences of beer and
pier sounds whose F0 and voice onset time are correlated as in English (Canonical blocks) or the opposite way (Reverse blocks),
then categorize an ambiguous beer-pier test word and hear it again and repeat it aloud. The repetition is the speech production:
its F0 and voice onset time (VOT) are measured.

Raw files, raw_extra/Murphy2023_PsychonBullRev/ (OSF cwg4d, fetched with tools/std/fetch.py):
  Repetition_Production_F0s_VOTs.xlsx   19,801 repetitions: subject, filename (platform identifiers, not copied), trial_num (1-248),
        response (beer or pier categorization just before), condition (Canonical, Reverse), test_cue_f0 (LowF0 and HighF0 = ambiguous
        test words with F0 180 and 250 Hz; Beer and Pier = unambiguous test words), block_num (1-6), block_trial_index, prod_f0
        (F0 of the first 40 ms of voicing, log-transformed and z-scored by the authors), item (exposure set), type (POS_VOT or
        NEG_VOT), vot_dur (ms), sex, Gender, age, vot_dur.z (VOT z-scored within participant)
  README.txt and the R scripts               what the authors' analyses do (read as text)
Run from the site folder: python3 -I tools/std/convert/Murphy2023_PsychonBullRev.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Murphy2023_PsychonBullRev"
SRC = "Murphy, Nozari & Holt 2023, Psychon Bull Rev"
RAW = stdlib.raw_dir(ID)
DEVICE = "online experiment on the Gorilla platform in the Chrome browser, participant's own headphones and microphone"
STIM = {"LowF0": "ambiguous beer-pier test word (VOT 20 ms), low F0 (180 Hz)", "HighF0": "ambiguous beer-pier test word (VOT 20 ms), high F0 (250 Hz)",
        "Beer": "unambiguous beer test word (VOT 0 ms, F0 180 Hz)", "Pier": "unambiguous pier test word (VOT 40 ms, F0 250 Hz)"}

m = pd.read_excel(RAW / "Repetition_Production_F0s_VOTs.xlsx")
m["_i"] = np.arange(len(m))
m["_o"] = m.subject.map({s: i for i, s in enumerate(dict.fromkeys(m.subject))})
m = m.sort_values(["_o", "trial_num", "_i"], kind="stable").reset_index(drop=True)

rows = []
for r in m.itertuples(index=False):
    rev = r.condition == "Reverse"
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group="", subject=r.subject, subject_excluded_by_authors=False, age=r.age,
        sex={"Female": "F", "Male": "M"}[r.sex], handedness="", population="healthy", setting="online", device=DEVICE, session=1,
        block=f"block {r.block_num}", trial=r.trial_num, phase="transfer", time_s="", task="speech", unit="utterance",
        condition=r.condition, pert_type="reversed_f0_vot_exposure" if rev else "none", pert_value="" if rev else 0, pert_unit="",
        feedback="normal", excluded_by_authors=False,
        stimulus=STIM[r.test_cue_f0], segment="word-initial stop consonant of beer or pier",
        measure_window="F0 over the first 40 ms of voicing; VOT from the release burst to voicing onset",
        perceptual_response=r.response, f0_z=r.prod_f0, vot_ms=r.vot_dur, exposure_set=r.item))
# vot_dur.z has a dot in its name (not usable as an attribute): fill it from the frame in the same order
for x, z in zip(rows, m["vot_dur.z"]):
    x["vot_z"] = z

gender = m.drop_duplicates("subject").set_index("subject").Gender
parts = [dict(person=s, gender=g) for s, g in gender.items()]
pcols = [("gender", "text", "", "Gender the participant reported (Female, Male or Non-Binary). The sex column of the main table is the sex "
                                 "assigned at birth, the variable the authors analysed.")]

# ---- checks
P = json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8"))
S = next(e for e in P if e["id"] == ID)
df = pd.DataFrame(rows)
ck = stdlib.Checks()
per = df.drop_duplicates("subject")
ck.published("participants analysed (90 recruited, 3 removed for poor audio)", df.subject.nunique(), 87, 0, "Method, Participants (ninety recruited; three removed)")
ck.published("female participants (45 recruited, 2 removed)", (per.sex == "F").sum(), 43, 0, "Method, Participants (45 females; two female participants removed)")
ck.published("male participants (45 recruited, 1 removed)", (per.sex == "M").sum(), 44, 0, "Method, Participants (three removed, two of them female)")
ck.published("mean age (the paper's value is for the 90 recruited)", per.age.mean(), 28.6, 0.1, "Method, Participants (Mage = 28.6, SD = 6)")
ck.published("SD of age", per.age.std(), 6, 0.1, "Method, Participants (Mage = 28.6, SD = 6)")
ck.published("test trials per participant (most)", df.groupby("subject").trial.max().max(), 248, 0, "Method, Procedure (248 test trials in six blocks)")
ck.published("blocks", df.block.nunique(), 6, 0, "Method, Procedure (six blocks)")
last =m[m.condition == "Canonical"].groupby("block_num").block_trial_index.max()
ck.published("canonical test trials (blocks of 42, 41 and 41)", last.sum(), 124, 0, "Method, Procedure (124 canonical test trials)")
ck.published("canonical blocks with 41 trials", (last == 41).sum(), 2, 0, "Method, Procedure (two of the three canonical blocks had 41 trials)")
amb = df[df.stimulus.str.startswith("ambiguous")].groupby("subject").trial.nunique().max()
una = df[df.stimulus.str.startswith("unambiguous")].groupby("subject").trial.nunique().max()
ck.published("ambiguous test trials per participant (most)", amb, 200, 0, "Method, Procedure (200 ambiguous test trials)")
ck.published("unambiguous test trials per participant (most)", una, 48, 0, "Method, Procedure (48 unambiguous test trials)")
ck.internal("rows = rows of the shared sheet", len(df) == len(m) == 19801)
ck.internal("type equals the sign of vot_dur", bool(((m["type"].str.strip() == "NEG_VOT") == (m.vot_dur < 0)).all()))
ck.internal("VOT z-scores have mean 0 and SD 1 within each participant", bool(np.allclose(m.groupby("subject")["vot_dur.z"].mean(), 0, atol=1e-9)
                                                                           and np.allclose(m.groupby("subject")["vot_dur.z"].std(), 1, atol=1e-6)))
dup = int(m.duplicated(["subject", "trial_num"]).sum())
ck.internal("only 3 repeated (participant, trial) pairs", dup == 3, f"{dup} rows repeat a trial number and a file name (kept)")

meta = dict(
    paper=dict(authors=S["papers"][0]["authors"], year=S["papers"][0]["year"], title=S["papers"][0]["title"],
               venue=S["papers"][0]["venue"], doi=S["papers"][0]["doi"]),
    lab="Holt (Carnegie Mellon / UT Austin) / Nozari", data_url=S["data_url"], license="none stated", license_url="",
    changes=("Rows are ordered by participant and trial number (the sheet is not in trial order). The file name column (it holds platform "
             "identifiers) was not copied; type (POS_VOT or NEG_VOT) was not copied because it only repeats the sign of vot_dur; Gender is "
             "in participants.csv; item is exposure_set; sex Female and Male are F and M. Values are otherwise as the authors' sheet."),
    how=["One row is one repetition of the ambiguous or unambiguous beer-pier test word that the participant said aloud after "
         "categorizing it; perceptual_response is the category chosen just before.",
         "condition is the exposure regularity of the block (Canonical: the English F0-VOT relation; Reverse: the opposite), played "
         "passively before each test trial; blocks alternate, starting with Canonical. stimulus gives the test word that was repeated.",
         "f0_z is the authors' z-scored log F0 of the first 40 ms of voicing (the Hz values are not shared); vot_ms is the produced VOT in "
         "milliseconds (negative when voicing started before the burst) and vot_z the same z-scored within participant. trial is the "
         "authors' trial number (1-248, with gaps where a repetition was dropped)."],
    unit_note="repetition of a test word",
    differences=["The paper's mean age (28.6, SD 6) describes the 90 recruited participants; the 87 analysed participants give "
                 f"{per.age.mean():.2f} (SD {per.age.std():.2f}).",
                 "Three pairs of rows share a participant, trial number and file name but have different F0 values; both rows of each "
                 "pair are kept as shared. The paper's mixed-model results were not refitted."])
stdlib.release(ID, "speech", rows, meta, ck, participants=parts, participant_columns=pcols)
