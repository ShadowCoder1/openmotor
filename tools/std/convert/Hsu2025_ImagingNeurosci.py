"""Hsu et al. (2025), Imaging Neuroscience: dose-response of tDCS on motor sequence learning, one row per key press.

Run from the site folder: python3 -I tools/std/convert/Hsu2025_ImagingNeurosci.py
Raw files (untrusted data, read only) in raw_extra/Hsu2025_ImagingNeurosci/, from https://osf.io/gyfae/ (GPL-3.0):
  BehavioralData.zip   one MATLAB v7.3 file per participant, session and sequence (Subject<id>_<session>_S<sequence>.mat): the key log of
                       36 trials (log.trial1..36, a string matrix of key letter and time stamp per press), the sequence, and in the
                       session-1 files the three skin-sensation ratings. Read with h5py: MATLAB string objects are stored as
                       uint64 arrays under #subsystem#/MCOS (version, ndims, dims, characters per string, UTF-16 characters);
                       every file is validated while it is read.
  tdcs_groups.csv      typing speed (WPM) and group of the 120 analysed participants (gn 3 = 0 mA, 1 = 4 mA, 2 = 6 mA, from
                       grouporder in the authors' combineData.m)
  demographics.csv     participant, age, sex (no header)
Task, from runFTT.m and get_FTT.m (read as text only): keypad digits 1-4 are the keys v, b, n, m; sequences 1/2/3 = 41324, 23142,
34213; 36 trials of 10 s with 10 s breaks; the authors score each trial by splitting the key presses into consecutive groups of
five and comparing the k-th press of a group with the k-th element of the sequence (a group with five matches is a correct
sequence, NCS). The same position rule gives correct_key and correct here.
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

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Hsu2025_ImagingNeurosci_license import LICENSE_TEXT  # noqa: E402

ID = "Hsu2025_ImagingNeurosci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
SRC = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for fn in ("BehavioralData.zip", "demographics.csv", "tdcs_groups.csv"):
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == SRC[fn]["sha256"])

KEY = {"1": "v", "2": "b", "3": "n", "4": "m"}
SECTIONS = {  # (session, sequence file) -> authors' label, hand, sequence, phase, block type, description
    (1, 1): ("L:S1 + tDCS", "left", "41324", "learning", "training", "left hand, sequence 41324, during tDCS"),
    (2, 1): ("L:S1", "left", "41324", "test", "test", "left hand, sequence 41324 again, 60 min after tDCS"),
    (2, 2): ("R:S2", "right", "23142", "transfer", "test", "right hand, new sequence 23142, 60 min after tDCS"),
    (2, 3): ("L:S3", "left", "34213", "transfer", "test", "left hand, new sequence 34213, 60 min after tDCS"),
}
DOSE = {3: 0, 1: 4, 2: 6}  # group number -> mA (grouporder = [3,1,2] in combineData.m: groups for 0, 4, 6 mA)
NTRIAL = 36

# ---------------------------------------------------------------- participants
groups = {}
with open(RAW / "tdcs_groups.csv", newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        groups[int(r["subjectID"])] = (float(r["speed"]), int(r["gn"]))
demo = {}
with open(RAW / "demographics.csv", newline="", encoding="utf-8-sig") as f:
    for r in csv.reader(f):
        if r:
            demo[int(r[0])] = (float(r[1]), r[2].strip())
checks.internal("tdcs_groups.csv: 120 analysed participants, 40 in each of the three groups (Methods: N = 40 per group)",
                len(groups) == 120 and Counter(g for _, g in groups.values()) == {1: 40, 2: 40, 3: 40}, str(Counter(g for _, g in groups.values())))


# ---------------------------------------------------------------- MATLAB v7.3 reading (string objects)
def decode_strings(words):
    w = np.asarray(words, dtype=np.uint64).reshape(-1)
    assert int(w[0]) == 1
    nd = int(w[1])
    dims = [int(x) for x in w[2:2 + nd]]
    n = int(np.prod(dims))
    counts = [int(x) for x in w[2 + nd:2 + nd + n]]
    chars = w[2 + nd + n:].tobytes().decode("utf-16-le")
    assert sum(counts) <= len(chars) < sum(counts) + 4
    out, p = [], 0
    for c in counts:
        out.append(chars[p:p + c])
        p += c
    return np.array(out, dtype=object).reshape(dims, order="F")


def string_object(f, cells, ds):
    a = np.array(ds).reshape(-1)
    assert a.dtype == np.uint32 and a.size == 6 and int(a[0]) == 0xDD000000, a
    return decode_strings(cells[int(a[4]) + 1])


def chars_of(ds):
    return "".join(chr(int(c)) for c in np.array(ds).reshape(-1))


z = zipfile.ZipFile(RAW / "BehavioralData.zip")
files = {}
stray = []
for i in z.infolist():
    m = re.fullmatch(r"Subject(\d+)_(\d)_S(\d)\.mat", i.filename)
    if m and int(m.group(2)) in (1, 2) and int(m.group(3)) in (1, 2, 3):
        files[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = i.filename
    else:
        stray.append((i.filename, i.file_size))
subjects = sorted({k[0] for k in files})
checks.internal("129 participants, each with the four task files (session 1 sequence 1; session 2 sequences 1, 2, 3)",
                len(subjects) == 129 and all(all((s, a, b) in files for a, b in SECTIONS) for s in subjects)
                and len(files) == 516, f"{len(files)} files; not read: {stray}")
checks.internal("every analysed participant (tdcs_groups.csv) has behaviour files; the others are the 9 the paper replaced",
                set(groups) <= set(subjects) and len(set(subjects) - set(groups)) == 9, f"{len(set(subjects) - set(groups))} not analysed")

presses = []          # (subject, section key, trial, index in trial, key, time_s, previous time_s or None)
ratings = {}
n_empty = Counter()
trial_ncs = defaultdict(list)   # (subject, section) -> NCS per trial (36)
bad = Counter()
no_log = []
odd_keys = []
for s in subjects:
    for sec in SECTIONS:
        f = h5py.File(io.BytesIO(z.read(files[(s, *sec)])), "r")
        if "log" not in f or "sequence" not in f:   # the authors' get_FTT.m also handles a file without a key log
            no_log.append((s, sec))
            continue
        cells = [np.array(f[r]) for r in np.array(f["#subsystem#"]["MCOS"]).reshape(-1)]
        digits = "".join(string_object(f, cells, f["sequence"]).reshape(-1))
        if digits != SECTIONS[sec][2]:
            bad["sequence differs from the one the file name says"] += 1
        seq = "".join(KEY[d] for d in digits)
        if sec == (1, 1):
            ratings[s] = [float(x) for x in np.array(f["rating"]).reshape(-1)]
        lg = f["log"]
        assert sorted(lg.keys()) == sorted(f"trial{k}" for k in range(1, NTRIAL + 1)), sorted(lg.keys())[:3]
        for k in range(1, NTRIAL + 1):
            a = np.array(lg[f"trial{k}"])
            if a.dtype == np.uint32 and a.size == 6 and int(a.reshape(-1)[0]) == 0xDD000000:
                arr = string_object(f, cells, lg[f"trial{k}"])
                assert arr.ndim == 2 and arr.shape[1] == 2, arr.shape
                keys = [str(x) for x in arr[:, 0]]
                times = [float(x) for x in arr[:, 1]]
                for kk in keys:
                    if kk not in "vbnm":
                        odd_keys.append((s, sec, k, kk))
                if not all(t2 >= t1 for t1, t2 in zip(times, times[1:])):
                    bad[f"times decrease: subject {s} {sec} trial {k}"] += 1
                if times[-1] - times[0] >= 10.6:
                    bad[f"trial longer than 10.6 s: subject {s} {sec} trial {k} ({times[-1] - times[0]:.2f} s)"] += 1
            else:
                keys, times = [], []
                n_empty[(sec, k)] += 1
            for j, (kk, t) in enumerate(zip(keys, times)):
                presses.append((s, sec, k, j, kk, t, times[j - 1] if j else None, seq))
            ncs = sum(1 for j in range(len(keys) // 5) if "".join(keys[5 * j:5 * j + 5]) == seq)
            trial_ncs[(s, sec)].append(ncs)
checks.internal("all files with a key log read with the expected sequence, non-decreasing time stamps and trials shorter than 10.6 s",
                not bad, f"{dict(bad)}; {len(no_log)} files without a key log {no_log[:12]}; {sum(n_empty.values())} trials without a key press"
                f" (trial 1 missing in {sum(v for (sec, k), v in n_empty.items() if k == 1)} files: the authors note a task-script bug); "
                f"{len(odd_keys)} presses of other keys {sorted(set(o[3] for o in odd_keys))} (subject, section, trial: {odd_keys[:3]})")
checks.internal(f"{len(presses)} key presses read", len(presses) > 300000)

# ---------------------------------------------------------------- published numbers
SRC_P = "Imaging Neuroscience 2025, "
checks.published("participants analysed", len(groups), 120, 0, SRC_P + "Abstract (120 participants)")
nd = [d for d in demo.values()]
checks.published("recruited participants who have demographic data", len(nd), 130, 0, SRC_P + "Methods (a total of 130 participants)")
checks.published("number of women among the 130", sum(1 for a, sx in nd if sx == "F"), 73, 0, SRC_P + "Methods (73 female and 57 male)")
checks.published("mean age of the 130 (years)", st.mean(a for a, _ in nd), 23.3, 0.05, SRC_P + "Methods (23.3 +- 5.86 years)")
checks.published("SD of age of the 130 (years)", st.stdev(a for a, _ in nd), 5.86, 0.01, SRC_P + "Methods")
sp = [v[0] for v in groups.values()]
checks.published("mean typing speed (WPM)", st.mean(sp), 33.5, 0.05, SRC_P + "Results 4.2.2 (33.5 +- 1.1 WPM)")
checks.published("SEM of typing speed (WPM)", st.stdev(sp) / len(sp) ** 0.5, 1.1, 0.05, SRC_P + "Results 4.2.2")


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5


ana = sorted(groups)
ncs1 = {s: trial_ncs[(s, (1, 1))] for s in ana}
overall = [st.mean(ncs1[s]) for s in ana]
r_first1 = pearson([groups[s][0] for s in ana], [ncs1[s][0] for s in ana])
r_first10 = pearson([groups[s][0] for s in ana], [st.mean(ncs1[s][:10]) for s in ana])
r_gain1 = pearson([groups[s][0] for s in ana], [st.mean(ncs1[s][-10:]) - ncs1[s][0] for s in ana])
r_gain10 = pearson([groups[s][0] for s in ana], [st.mean(ncs1[s][-10:]) - st.mean(ncs1[s][:10]) for s in ana])
checks.published("correlation of typing speed with overall performance (mean NCS over the 36 trials of L:S1 + tDCS)",
                 pearson([groups[s][0] for s in ana], overall), 0.569, 0.01, SRC_P + "Results 4.2.2, r(118) = 0.569")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    else:
        print(f"  [{'ok' if ch['ok'] else 'FAILED'}] {ch['what']} {ch.get('detail', '')}")

# ---------------------------------------------------------------- the OpenMotor table
NAMES = [c[1] for c in stdlib.columns_for("keypress") if c[1] != "person"]
order = {sec: n for n, sec in enumerate(SECTIONS)}
presses.sort(key=lambda p: (p[0], order[p[1]], p[2], p[3]))
run = Counter()


class Rows:
    def __len__(self):
        return len(presses)

    def __iter__(self):
        run.clear()
        for s, sec, k, j, kk, t, tprev, seq in presses:
            label, hand, sdig, phase, btype, desc = SECTIONS[sec]
            run[s] += 1
            gn = groups[s][1] if s in groups else None
            dose = DOSE[gn] if gn else None
            row = {n: None for n in NAMES}
            row.update(
                dataset=ID, study="Hsu et al. 2025 Imaging Neurosci", experiment="tDCS dose-response, finger sequence tapping",
                group=(f"{dose} mA" + (" (sham)" if dose == 0 else "")) if gn else None, subject=f"Subject{s}",
                subject_excluded_by_authors=s not in groups, age=demo[s][0], sex=demo[s][1], handedness="right",
                setting="lab", device="computer keyboard, four keys (v b n m for keypad digits 1-4)",
                session=sec[0], block=desc, trial=run[s], phase=phase, time_s=t, task="finger tapping and timing", unit="keypress",
                condition=label, pert_type="tDCS" if sec == (1, 1) else "none",
                pert_value=(dose if gn else None) if sec == (1, 1) else 0, pert_unit="mA" if sec == (1, 1) else None, feedback="normal",
                block_number=k, block_type=btype, correct_key=seq[j % 5], key_pressed=kk, correct=kk == seq[j % 5],
                interval_ms=round((t - tprev) * 1000, 3) if tprev is not None else None, element_type="sequence",
                sequence=sdig, sequence_position=j % 5 + 1, repetition=j // 5 + 1,
            )
            yield row


pcols = [
    ("typing_speed_wpm", "number", "words per minute", "Typing speed in the baseline typing test used to balance the groups (tdcs_groups.csv); empty for the 9 participants who were not analysed."),
    ("tdcs_intensity_ma", "number", "mA", "Intensity of the tDCS group (0 = sham), from the authors' group order; empty for participants outside the analysed groups."),
    ("sensation_rating_start", "number", "authors' scale", "Skin-sensation rating at the beginning of stimulation (session-1 file, first of three ratings)."),
    ("sensation_rating_middle", "number", "authors' scale", "Skin-sensation rating in the middle of stimulation."),
    ("sensation_rating_end", "number", "authors' scale", "Skin-sensation rating at the end of stimulation."),
]
participants = []
for s in subjects:
    r = ratings[s]
    participants.append(dict(person=f"Subject{s}", typing_speed_wpm=groups[s][0] if s in groups else None,
                             tdcs_intensity_ma=DOSE[groups[s][1]] if s in groups else None,
                             sensation_rating_start=r[0], sensation_rating_middle=r[1], sensation_rating_end=r[2]))
meta = dict(
    study="Hsu et al. 2025",
    paper=dict(authors="Hsu G, Jafari ZH, Ahmed A, Edwards DJ, Cohen LG, Parra LC", year=2025,
               title="Dose-response of tDCS effects on motor learning and cortical excitability: A preregistered study",
               venue="Imaging Neuroscience 3:imag_a_00431", doi="10.1162/imag_a_00431"),
    lab="Parra (City College of New York)",
    data_url="https://osf.io/gyfae/",
    license="GNU General Public License (GPL) 3.0", license_url="https://www.gnu.org/licenses/gpl-3.0.txt", license_text=LICENSE_TEXT,
    changes=("Converted to the OpenMotor format with participant codes replaced: the key logs of the four task files of each participant "
             "(MATLAB string objects) were unpacked into one row per key press, with the authors' scoring rule written out per press, "
             "intervals between presses added as arithmetic, and the group, typing speed, age, sex and sensation ratings joined from "
             "the deposit's own files; the MEP, TEP and EEG data, the stray file Subject10_10_S2.mat and the participant without "
             "behaviour files were not converted."),
    how=[
        "One row is one key press in a finger sequence-tapping task: the participant repeatedly typed a five-element sequence on "
        "four keys (keypad digits 1-4 are the keys v, b, n, m) in 36 trials of 10 s with 10-s breaks, four times: with the left "
        "(non-dominant) hand on sequence 41324 while receiving tDCS (4 mA, 6 mA or sham, session 1; condition L:S1 + tDCS), then "
        "60 min later without stimulation on the same sequence (L:S1), on a new sequence with the right hand (R:S2, 23142) and on "
        "another new sequence with the left hand (L:S3, 34213). Hands follow the paper's Methods text; the legend of its Figure 2 writes R:S1 for the repeat.",
        "time_s is the authors' time stamp of the press in seconds (their clock runs on through the 36 trials of one task file); interval_ms is the "
        "time since the previous press in the same trial; block_number is the trial (1-36). correct_key, correct, sequence_position and "
        "repetition follow the authors' scoring in get_FTT.m: the presses of a trial are cut into consecutive groups of five and the k-th "
        "press of a group is compared with the k-th element of the sequence (five matches make one correct sequence, NCS, their main outcome). "
        "Presses left over at the end of a trial (fewer than five) are given the same position rule. Trials without any press have no rows.",
        "subject_excluded_by_authors marks the 9 participants whose behaviour files are in the deposit but who are not in the analysed "
        "groups of tdcs_groups.csv (the paper replaced 9 participants after quality control). The deposit does not name the intensity of "
        "each group: group labels come from grouporder = [3,1,2] in the authors' combineData.m (group numbers for 0, 4 and 6 mA). "
        "Handedness is right for everyone because left-handed people were excluded.",
    ],
    unit_note="key press (finger sequence-tapping task, 36 trials per task file)",
    differences=[
        f"The paper's correlations of typing speed with baseline performance (r = .369) and performance gain (r = .421; Figure 11b-c) were not "
        f"reproduced because the paper does not define the windows: with the first trial the baseline correlation is {r_first1:.2f}, with the mean of the first 10 trials "
        f"{r_first10:.2f}; the gain (mean of the last 10 trials minus the first trial) gives {r_gain1:.2f}, minus the mean of the first 10 trials {r_gain10:.2f}. "
        "The correlation with overall performance (mean NCS over the 36 trials, r = .569) reproduces.",
        "Subject 69 has no key log in its R:S2 file (the authors' get_FTT.m treats a missing log as missing data), and " + (
            "one press is" if len(odd_keys) == 1 else f"{len(odd_keys)} presses are") + " of the space bar instead of one of the four keys "
        "(subject 59, session 1, trial 6); it is kept, with correct = False.",
        "The deposit's stray file Subject10_10_S2.mat (17 kB, session number 10) was not converted; each participant has exactly the four standard files without it.",
        "The Figure 2 legend of the paper writes R:S1 for the repeat of the first task, while the Methods text (and the file naming) give L:S1.",
    ],
)
stdlib.release(ID, "keypress", Rows(), meta, checks, participants=participants, participant_columns=pcols)
