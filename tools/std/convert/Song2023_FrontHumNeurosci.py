"""Song, Shin, Kim & Jeong (2023), Front Hum Neurosci 17:1221944: multi-context visuomotor adaptation (rotation +90, rotation -90, mirror), erased dots per trial.

Run from the site folder: python3 -I tools/std/convert/Song2023_FrontHumNeurosci.py
Raw files (untrusted data, read only) in raw_extra/Song2023_FrontHumNeurosci/, from https://openneuro.org/datasets/ds004562 (BIDS):
  ev_NN.tsv   sub-NN/ses-01pre/beh/sub-NN_ses-01pre_task-adaptation_events.tsv (day-1 session outside the scanner)
  participants.tsv   age and sex
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Song2023_FrontHumNeurosci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
part = pd.read_csv(RAW / "participants.tsv", sep="\t").set_index("participant_id")
CTX = {"rotation+90": ("rotation", 90.0), "rotation-90": ("rotation", -90.0), "mirror": ("mirror", 1.0)}
rows = []
nblk = []
for n in range(1, 24):
    t = pd.read_csv(RAW / f"ev_{n:02d}.tsv", sep="\t", na_values=["n/a"])
    pid = f"sub-{n:02d}"
    er = t[t.trial_type == "erasing"].reset_index(drop=True)
    nblk.append(er.groupby("session_number").size().tolist())
    for k, r in er.iterrows():
        pt, pv = CTX[r.session_type]
        rows.append(dict(
            dataset=ID, study=ID, experiment="session 1 (day 1, outside the scanner)", group="", subject=str(n), person=str(n),
            subject_excluded_by_authors=(n == 16), age=int(part.loc[pid, "age"]), sex=part.loc[pid, "sex"], handedness="right",
            population="healthy", setting="lab", device="computer mouse (cursor on screen), pseudo-MRI environment",
            session=1, block=f"block {int(r.session_number)} ({r.session_type})", trial=k + 1, phase="learning", time_s=float(r.onset),
            task="tracking", unit="participant summary", condition=f"{r.session_type}; erase {r.stimulus_direction}",
            pert_type=pt, pert_value=pv, pert_unit="deg" if pt == "rotation" else "mirror reversal of horizontal cursor movement",
            feedback="altered", excluded_by_authors=(n == 16),
            measure="number_ErasedDot", measure_description="Number of the seven line dots erased by the cursor in this trial (max 3.5 s)",
            statistic="value", value=int(r.number_ErasedDot), value_unit="dots", n_trials=1, session_context="day-1 familiarization session"))
df = pd.DataFrame(rows)

p = part.copy()
checks.published("participants recruited", len(p), 23, 0, "Methods, Subjects and ethics statement")
checks.published("female participants", (p.sex == "F").sum(), 9, 0, "Methods, Subjects and ethics statement")
checks.published("mean age (years)", p.age.mean(), 22.9, 0.05, "Methods, Subjects and ethics statement")
checks.published("SD of age (years)", p.age.std(ddof=1), 1.8, 0.05, "Methods, Subjects and ethics statement")
checks.published("erasing trials per block", df.groupby(["person", "block"]).size().max(), 17, 0, "Methods, Task description")
checks.published("blocks per context repeated 5 times", df[df.person == "1"].groupby("pert_value").block.nunique().min(), 5, 0, "Methods, Task description")
checks.published("blocks per session", df[df.person == "1"].block.nunique(), 15, 0, "Methods, Task description")
checks.internal("23 participants x 255 erasing trials", len(df) == 23 * 255 and df.person.nunique() == 23, str(len(df)))
checks.internal("dots erased between 0 and 7", df.value.between(0, 7).all(), str((df.value.min(), df.value.max())))

meta = dict(
    paper=dict(authors="Song Y, Shin W, Kim P, Jeong J", year=2023,
               title="Neural representations for multi-context visuomotor adaptation and the impact of common representation on multi-task performance: a multivariate decoding approach",
               venue="Front Hum Neurosci 17:1221944", doi="10.3389/fnhum.2023.1221944"),
    lab="Jeong (KAIST)", data_url="https://openneuro.org/datasets/ds004562", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted from BIDS events TSV to the OpenMotor table. Only the day-1 behavioural events (ses-01pre) are in the deposit's BIDS tree; the fMRI images and the raw "
             "MATLAB task files in derivatives are not used. The 'erasing' rows are kept as trials; the start_session, score and end_session rows (block markers, 15 each per participant) are "
             "task events and are not rows."),
    how=["One row is one erasing trial of a mouse-cursor task: a line of seven dots had to be erased in sequence (leftward, rightward, upward or downward) within 3.5 s while the cursor "
         "was rotated 90 deg clockwise (rotation+90), 90 deg counter-clockwise (rotation-90) or mirrored horizontally. Each of the 15 blocks had 17 trials and one context, each context five times, in random order.",
         "value is the number of dots erased in the trial (0 to 7), the only trial outcome the deposit shares; time_s is the trial onset from the start of the session. Rotation sizes are signed as in the authors' events description "
         "(rotation+90 is clockwise); pert_value 1 for mirror is a code, not a size. The deposit has no cursor traces.",
         "Participant 16 did not complete the MRI session (arm numbness) and is excluded by the authors; their day-1 rows are kept and marked excluded."],
    differences=[],
    unit_note="trial (dots erased)",
)
stdlib.release(ID, "summary", df, meta, checks, participants=None)
