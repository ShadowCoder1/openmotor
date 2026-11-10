"""Bonstrup et al. (2019), Curr Biol, sequential finger tapping (SFTT) with MEG: OpenNeuro ds006502, one row per key press.

Run from the site folder: python3 -I tools/std/convert/Bonstrup2019_CurrBiol.py
Raw file (untrusted data, read only with zipfile) in raw_extra/Bonstrup2019_CurrBiol/, from
https://github.com/OpenNeuroDatasets/ds006502 (tag 1.0.0, the GitHub mirror of OpenNeuro ds006502, CC0):
  ds006502-1.0.0.zip   only its text files are read: README, participants.tsv/json and the *_events.tsv of every run
                       (event_type StartPractice/StopPractice/StartRest/StopRest markers and one KeyDown per key press)
The MEG, EMG and eye-tracking recordings (pointer files in the mirror) are not converted.
The checks use the numbers in the deposit's own README (the dataset description): the two papers named on OpenMotor
describe other cohorts (27 and 30 analysed participants, one retest session at most).
"""
import csv
import io
import json
import re
import statistics as st
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Bonstrup2019_CurrBiol"
RAW = stdlib.raw_dir(ID)
ZIP = "ds006502-1.0.0.zip"
checks = stdlib.Checks()

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("the archive matches the downloaded hash", stdlib.sha256(RAW / ZIP) == src[ZIP]["sha256"])
z = zipfile.ZipFile(RAW / ZIP)
names = [i.filename for i in z.infolist()]
pre = names[0].split("/")[0] + "/"


def rd(p):
    return z.read(pre + p).decode("utf-8")


def table(p):
    return list(csv.reader(io.StringIO(rd(p)), delimiter="\t"))


def num(x):
    return None if x in ("", "n/a", "NaN") else float(x)


# ---------------------------------------------------------------- participants
pt = table("participants.tsv")
ph = pt[0]
part = {}
for r in pt[1:]:
    d = dict(zip(ph, r))
    part[d["participant_id"]] = dict(age=num(d["age"]), sex=d["sex"], hand=d["handedness"].lower(),
                                     i12=num(d["meg1_to_meg2_interval"]), i13=num(d["meg1_to_meg3_interval"]))
pids = sorted(part, key=lambda s: int(s.split("-")[1]))
assert len(pids) == len(part)

CONTROL = {2: "21342 42431 34231 14342 32431 14231 32421 23142 42314".split(),
           3: "21342 31214 12434 41321 23241 31432 23134 12134 43241".split()}  # README, MEG2 and MEG3 control scans
LABEL = {"training": ("training", "learning"), "retest": ("test", "test"), "control": ("test", "transfer")}
FINGER = {"1": "Little", "2": "Ring", "3": "Middle", "4": "Index"}

# ---------------------------------------------------------------- the events files
ev_files = defaultdict(list)  # participant -> [(session, run, task, path)]
for n in names:
    m = re.fullmatch(re.escape(pre) + r"(sub-\d+)/ses-meg(\d)/meg/sub-\d+_ses-meg\d_task-(\w+)_run-(\d+)_events\.tsv", n)
    if m:
        ev_files[m.group(1)].append((int(m.group(2)), int(m.group(4)), m.group(3), n[len(pre):]))
checks.internal("every participant has events files and they are the 143 runs of the deposit",
                set(ev_files) == set(pids) and sum(len(v) for v in ev_files.values()) == 143)

NAMES = [c[1] for c in stdlib.columns_for("keypress") if c[1] != "person"]
rows = []
n_sub_trial = Counter()
trials_seen = defaultdict(set)      # (pid, session, task) -> trial numbers with a StartPractice marker
trials_pressed = defaultdict(set)   # (pid, session, task) -> trial numbers with at least one key press
bad = Counter()
n_nulltime = Counter()
disorder = []
seqs = defaultdict(set)
ctrl_set = {}
sess_of = defaultdict(set)
for pid in pids:
    p = part[pid]
    row_no = 0
    for ses, run, task, path in sorted(ev_files[pid]):
        sess_of[pid].add(ses)
        t = table(path)
        h = t[0]
        start, last_press, last_onset = {}, {}, -1e9
        # control trials are numbered after the retest trials of the same session
        block_base = max(trials_seen[(pid, ses, "retest")] or [0]) if task == "control" else 0
        label, phase = LABEL[task]
        for r in t[1:]:
            d = dict(zip(h, r))
            onset, tr = float(d["onset"]), int(d["trial_number"])
            if d["event_type"] == "StartPractice":
                start[tr] = onset
                trials_seen[(pid, ses, task)].add(tr)
                seqs[(ses, task)].add(d["target_sequence"])
                ctrl_set.setdefault((pid, ses, task), set()).add(d["target_sequence"])
                continue
            if d["event_type"] != "KeyDown":
                continue
            if tr not in start or d["keypress_id"] not in FINGER or d["finger_label"] != FINGER[d["keypress_id"]]:
                bad["key press outside its trial or key/finger mismatch"] += 1
            # a press with onset -120 s and no sample index has no usable time in the deposit (a placeholder)
            nulltime = onset == -120.0 and d["onset_sample_idx"] == "NaN"
            if nulltime:
                n_nulltime[(pid, ses, task)] += 1
            else:
                if onset < last_onset:
                    disorder.append((pid, ses, task, tr, round(last_onset - onset, 4)))
                last_onset = onset
            trials_pressed[(pid, ses, task)].add(tr)
            prev = last_press.get(tr)
            interval = (onset - prev) * 1000 if (prev is not None and not nulltime and onset >= prev) else None
            last_press[tr] = None if nulltime else onset
            row_no += 1
            rows.append(dict(
                dataset=ID, study=ID, experiment="MEG sequence learning study", group="", subject=pid,
                subject_excluded_by_authors=False, age=p["age"], sex=p["sex"], handedness=p["hand"],
                population="healthy", setting="lab",
                device="four-key response pad (Cedrus LS-Line), left hand, E-Prime 2, seated in a MEG scanner room",
                session=ses, block=f"{task} trial {tr}", trial=row_no, phase=phase, time_s=None if nulltime else onset,
                task="sequence learning",
                unit="keypress", condition="trained sequence" if task != "control" else "control sequence",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                block_number=block_base + tr, block_type=label, key_pressed=d["keypress_id"], interval_ms=interval,
                sequence=d["target_sequence"], time_in_block_ms=None if nulltime else (onset - start[tr]) * 1000))
checks.internal("every key press lies in a practice trial and matches its finger label (1 little, 2 ring, 3 middle, "
                "4 index)", not bad, str(dict(bad)))
checks.internal("key presses are in time order within each run, except one pair (in a MEG2 control trial 4) that the "
                "deposit lists 0.2 s out of order: kept in the authors' order, interval_ms left empty for the later-listed press",
                len(disorder) == 1 and disorder[0][:4] == ("sub-13", 2, "control", 4),
                f"{len(disorder)} pair, {disorder[0][4] if disorder else ''} s out of order")
A_SET, B_SET = set(CONTROL[2]), set(CONTROL[3])
pat = Counter((ses, "README MEG2 list" if v == A_SET else "README MEG3 list" if v == B_SET else "other")
              for (pid, ses, task), v in ctrl_set.items() if task == "control")
checks.internal("trained-sequence runs all show 41324; the control runs show the 9 untrained sequences of the README's MEG2 "
                "list (all 29 MEG2 runs) or MEG3 list (26 of the 27 MEG3 runs; one participant's MEG3 run used the MEG2 list)",
                seqs[(1, "training")] == {"41324"} and seqs[(2, "retest")] == {"41324"} and seqs[(3, "retest")] == {"41324"}
                and dict(pat) == {(2, "README MEG2 list"): 29, (3, "README MEG3 list"): 26, (3, "README MEG2 list"): 1}, str(dict(pat)))
checks.internal("key presses without a usable time (onset -120 s and no sample index): kept, time columns empty",
                sum(n_nulltime.values()) == 82 and all(k[2] == "control" for k in n_nulltime),
                f"{sum(n_nulltime.values())} presses in {len(n_nulltime)} control runs")
n_ses = Counter(len(v) for v in sess_of.values())
checks.internal("sessions: 31 participants have MEG1, 29 have MEG2 and 27 have MEG3 (events files)",
                (sum(1 for v in sess_of.values() if 1 in v), sum(1 for v in sess_of.values() if 2 in v),
                 sum(1 for v in sess_of.values() if 3 in v)) == (31, 29, 27), str(dict(n_ses)))
kp = Counter(r["key_pressed"] for r in rows)
checks.internal("key presses by key", sum(kp.values()) == len(rows) == 67485, str(dict(sorted(kp.items()))))

# ---------------------------------------------------------------- published numbers (README of the deposit)
first_seen = list(dict.fromkeys(r["subject"] for r in rows))
checks.internal("participants appear in the order of the deposit's labels, so the public codes S0001... follow them",
                first_seen == pids)
pub = lambda pid: f"{ID}_S{pids.index(pid) + 1:04d}"  # noqa: E731  (the code stdlib.release gives this participant)
SRC = "OpenNeuro ds006502 README (Subjects; Missing data)"
ages = [part[p]["age"] for p in pids]
checks.published("participants", len(set(r["subject"] for r in rows)), 31, 0, SRC)
checks.published("female participants", sum(part[p]["sex"] == "F" for p in pids), 21, 0, SRC)
checks.published("mean age (years)", st.mean(ages), 26.14, 0.005, SRC)
checks.published("SD of age (years)", st.stdev(ages), 4.17, 0.005, SRC)
h12 = [part[p]["i12"] * 24 for p in pids if part[p]["i12"] is not None]
d13 = [part[p]["i13"] for p in pids if part[p]["i13"] is not None]
checks.published("MEG1 to MEG2 interval, mean (hours)", st.mean(h12), 23.44, 0.01, SRC + ", MEG session design")
checks.published("MEG1 to MEG2 interval, SD (hours)", st.stdev(h12), 1.32, 0.01, SRC + ", MEG session design")
checks.published("MEG1 to MEG3 interval, mean (days)", st.mean(d13), 29.45, 0.01, SRC + ", MEG session design")
checks.published("MEG1 to MEG3 interval, SD (days)", st.stdev(d13), 6.77, 0.01, SRC + ", MEG session design")
mx = lambda ses, task: max(max(trials_seen[(p, ses, task)]) for p in pids if trials_seen[(p, ses, task)])  # noqa: E731
checks.published("practice trials in the MEG1 training run", mx(1, "training"), 36, 0, SRC + ", MEG session design")
checks.internal("MEG2 retest runs hold 9 practice trials, as the MEG3 ones (the README text says 36, which a 3-minute scan "
                "of 10-s trials with 10-s rests cannot hold)", mx(2, "retest") == 9 and min(
                    max(trials_seen[(p, 2, "retest")]) for p in pids if trials_seen[(p, 2, "retest")]) == 9)
checks.published("practice trials in the MEG3 retest run", mx(3, "retest"), 9, 0, SRC + ", MEG session design")
checks.published("control sequences (one trial each) in the MEG2 control run", mx(2, "control"), 9, 0, SRC + ", MEG session design")
empty = lambda p, ses, task: sorted(trials_seen[(p, ses, task)] - trials_pressed[(p, ses, task)])  # noqa: E731
checks.published(f"{pub('sub-01')}: MEG1 training trials without a key press (trial 1 only)", len(empty("sub-01", 1, "training")), 1, 0,
                 SRC + ", Missing data")
checks.published(f"{pub('sub-23')}: MEG1 training trials without a key press (trial 1 only)", len(empty("sub-23", 1, "training")), 1, 0,
                 SRC + ", Missing data")
checks.published(f"{pub('sub-23')}: MEG2 retest trials without a key press (trials 1 and 2)", len(empty("sub-23", 2, "retest")), 2, 0,
                 SRC + ", Missing data")
checks.internal(f"{pub('sub-10')}: the README says the MEG recording stopped after trial 34, but the events file still holds key "
                "presses for trials 35 and 36", max(trials_seen[("sub-10", 1, "training")]) == 36
                and {35, 36} <= trials_pressed[("sub-10", 1, "training")])
others = sum(len(empty(p, s, t)) for p in pids for s in (1, 2, 3) for t in ("training", "retest", "control")
             if (p, s, t) not in {("sub-01", 1, "training"), ("sub-23", 1, "training"), ("sub-23", 2, "retest")})
checks.internal("no other practice trial lacks key presses", others == 0, f"{others} other trials")
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']} " + (f"computed {c['computed']:.4f} reported {c['reported']}" if c["kind"] == "published" else c.get("detail", "")))

# ---------------------------------------------------------------- release
meta = dict(
    study="Bonstrup et al. 2019",
    paper=dict(authors="Bonstrup M, Iturrate I, Thompson R, Cruciani G, Censor N, Cohen LG", year=2019,
               title="A Rapid Form of Offline Consolidation in Skill Learning", venue="Curr Biol 29(8):1346-1351.e4",
               doi="10.1016/j.cub.2019.02.049"),
    lab="Cohen (NINDS, NIH)",
    data_url="https://openneuro.org/datasets/ds006502",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced: the KeyDown rows of the 143 events.tsv files "
             "became one row per key press, and the practice-trial markers were used only to number the trials and to give "
             "time_in_block_ms. The MEG, EMG and eye-tracking recordings, the scans files and the resting-state runs were "
             "not converted, and no key press was marked correct or incorrect."),
    how=[
        "One row is one key press of the sequential finger tapping task: participants typed the displayed 5-item sequence "
        "(41324) as fast and accurately as they could with the non-dominant left hand on a four-key pad, in 10-s practice "
        "trials separated by 10-s rests; MEG1 is the training session (36 trials), MEG2 (about 24 h later) and MEG3 (about "
        "30 days later) are retests of the trained sequence (9 trials each) followed by one trial each of 9 untrained "
        "control sequences (condition control sequence, block_number continuing after the 9 retest trials).",
        "key_pressed is the authors' keypress_id (1 little, 2 ring, 3 middle, 4 index finger, as their finger_label says) "
        "and sequence is the sequence displayed in that trial; time_s is the authors' onset in seconds from the start of "
        "the run's recording, time_in_block_ms is the press time minus the trial's StartPractice marker, and interval_ms is "
        "the time since the previous press in the same trial (empty for the first press).",
        "Whether a press belongs to a correct 5-item sequence is not coded by the authors in these files (their analysis "
        "finds correct sequences in any circular shift of 41324), so correct, sequence_position and repetition are empty; "
        "a trial without any press has no row (the README names three such trials), and 82 presses in the control runs of "
        "8 participants have the placeholder onset -120 s and no sample index in the deposit, so their time_s, "
        "time_in_block_ms and the interval_ms after them are empty (one press that the deposit lists 0.2 s out of time "
        "order also has no interval_ms).",
        "age, sex and handedness come from participants.tsv; the two retest intervals are in participants.csv in days.",
    ],
    unit_note="key press in a 10-s practice trial of the sequential finger tapping task",
    differences=["The README's MEG session design says the MEG2 retest has 36 practice trials; the files hold 9 (as in MEG3), "
                 "which fits its 3-minute scan. The README says one participant's MEG1 recording stopped after trial 34; the events file "
                 "still has key presses for trials 35 and 36 (the behavioural log was not part of the MEG recording). One "
                 "participant's MEG3 control run used the MEG2 control sequences.",
                 "The two papers named on OpenMotor (Bonstrup 2019, 27 participants analysed, one retest the next day; Buch "
                 "2021, 30 participants, one session) describe other cohorts than this deposit (31 participants, retests "
                 "about 24 h and 30 days later), and the deposit's README cites no paper; the numbers checked here are "
                 "therefore the README's, not those of the papers."],
)
participants = [dict(person=p, meg1_to_meg2_interval_days=part[p]["i12"], meg1_to_meg3_interval_days=part[p]["i13"]) for p in pids]
pcols = [("meg1_to_meg2_interval_days", "number", "days", "Time between the MEG1 and MEG2 sessions, as in participants.tsv. Empty if no MEG2."),
         ("meg1_to_meg3_interval_days", "number", "days", "Time between the MEG1 and MEG3 sessions, as in participants.tsv. Empty if no MEG3.")]
stdlib.release(ID, "keypress", rows, meta, checks, participants=participants, participant_columns=pcols)
