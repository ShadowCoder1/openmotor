"""Pinsard et al. (2019), eLife: MVPA task of the sleep-consolidation study, 18 healthy adults, 4 five-element sequences (2 trained the
previous days = consolidated, 2 new) x 16 blocks, performed with the left hand in the scanner. Raw (untrusted, read only):
raw_extra/Pinsard2019_eLife/mvpa_task_behavioral_data.csv (OSF epj2v). Run: python3 -I tools/std/convert/Pinsard2019_eLife.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Pinsard2019_eLife"
d = pd.read_csv(stdlib.raw_dir(ID) / "mvpa_task_behavioral_data.csv")
d = d.drop(columns=["Unnamed: 0"])
base = dict(dataset=ID, study="Pinsard et al. 2019 eLife", experiment="MVPA task (sequence practice in the scanner)", subject_excluded_by_authors=False,
            age=None, sex="", handedness="right", population="healthy", setting="lab", device="MR-compatible response box, left hand",
            session=3, phase="test", time_s=None, task="sequence learning", unit="participant summary", pert_type="none", pert_value=0,
            pert_unit="", feedback="normal", excluded_by_authors=False, statistic="value", n_trials=None)
rows = []
for i, r in enumerate(d.itertuples(index=False)):
    common = dict(base, group="", subject=f"P{r.subjects}", person=f"P{r.subjects}",
                  block=f"{r.sequences} block {r.blocks}", trial=None,
                  condition="consolidated sequence" if r.seq_consolidated else "new sequence")
    if pd.notna(r.mean_seq_duration):
        rows.append(dict(common, measure="mean_correct_sequence_duration",
                         measure_description="Mean duration of the correctly performed sequences in the block (authors' mean_seq_duration)",
                         statistic="mean", value=r.mean_seq_duration, value_unit="s"))
    rows.append(dict(common, measure="correct_sequences", measure_description="Number of correctly performed sequences in the block (authors' num_correct_seq)",
                     statistic="count", value=r.num_correct_seq, value_unit="sequences"))
# trial = running order of the row within the participant
cnt = {}
for r in rows:
    cnt[r["subject"]] = cnt.get(r["subject"], 0) + 1
    r["trial"] = cnt[r["subject"]]
checks = stdlib.Checks()
checks.published("participants analysed", d.subjects.nunique(), 18, 0, "Methods, Participants (group of 18 retained)")
checks.published("blocks per sequence", d.groupby(["subjects", "sequences"]).blocks.nunique().max(), 16, 0, "Methods, MVPA task (16 blocks of each sequence)")
checks.published("blocks per participant", d.groupby("subjects").size().max(), 64, 0, "Methods, MVPA task (total of 64 blocks)")
checks.published("sequences", d.sequences.nunique(), 4, 0, "Methods, MVPA task (four sequences)")
checks.internal("1152 block rows", len(d) == 1152)
checks.internal("2 consolidated and 2 new sequences", d.groupby("sequences").seq_consolidated.first().sum() == 2)
paper = dict(authors="Pinsard B, Boutin A, Gabitov E, Lungu O, Benali H, Doyon J", year=2019,
             title="Consolidation alters motor sequence-specific distributed representations", venue="eLife 8:e39324", doi="10.7554/elife.39324")
meta = dict(paper=paper, lab="Doyon (McGill University)", data_url="https://osf.io/epj2v/", license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            changes="One row per block and measure (block duration and number of correct sequences become two rows); the authors' unnamed index column was dropped; 6 blocks with no correct sequence have no duration and only the count row.",
            how=["One row is one measure of one block of the MVPA task for one participant: the mean duration of the correctly performed sequences (seconds) or the number of "
                 "correctly performed sequences. Each participant did 16 blocks of each of 4 five-element sequences (TSeq1, TSeq2 trained earlier and so called consolidated, "
                 "NewSeq1, NewSeq2 untrained), 64 blocks in all, in the scanner with the left hand.",
                 "Blocks are numbered 1 to 16 within each sequence as in the deposit; the deposit has no information on the order in which the 64 blocks were run, so trial is the row order of the file.",
                 "Age, sex and training-day data are not in the deposit file; the paper reports 18 participants (14 females, 25 +/- 6.2 y)."],
            unit_note="participant, block and measure")
stdlib.release(ID, "summary", rows, meta, checks)
