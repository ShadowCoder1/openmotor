"""Wessel et al. (2023), Nat Neurosci: striatal theta-burst tTIS and sequential finger tapping (SFTT), per-block performance values.

Run from the site folder: python3 -I tools/std/convert/Wessel2023_NatNeurosci.py
Raw file (untrusted data, read only with zipfile) in raw_extra/Wessel2023_NatNeurosci/, from https://zenodo.org/records/14965889:
  behav_members.zip   the 434 behavioural members (data/Behav/**, Code/behav/*.ipynb, readme.txt) of the deposit's 124 GB Zenodo.zip,
                      read with HTTP range requests on the PC (SOURCES.json says so); nothing else of the zip was downloaded.
Used here:
  data/Behav/young_exp1/variables_per_block.csv   the authors' per-repetition table of Experiment 1 (14 participants, 2 sessions:
                                                  baseline repetition 0 and training repetitions 1-60, each 30 s)
  data/Behav/{young,older}_exp2/raw/<code>_<session>_<B|Train|Train_R|Post|FU1|FU2>_<A|B>.txt   one text file per participant,
                                                  session and block type with, for each trial, the lines "N keypresses",
                                                  "Maximum complete sequences M", "X correct sequences" and "Successrate S %"
  data/Behav/{young,older}_exp2/number_of_seq.csv, randomization/cond_*.csv   the authors' correct sequences per block (checks only)
Not converted: the Experiment 1 raw Presentation logs (one Response line per key press) and the lines of nine intervals per
sequence repetition in the Experiment 2 files, which the deposit does not document.
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

ID = "Wessel2023_NatNeurosci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
ZIP = "behav_members.zip"
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("the behavioural members match the downloaded hash", stdlib.sha256(RAW / ZIP) == src[ZIP]["sha256"])
z = zipfile.ZipFile(RAW / ZIP)
BASE = "Zenodo/data/Behav/"


def text(p):
    return z.read(BASE + p).decode("utf-8")


def table(p):
    return list(csv.DictReader(io.StringIO(text(p))))


STIM = {"A": "iTBS", "B": "HF control"}   # checked below against the authors' tables
DEVICE = "key-press response device, sequential finger tapping task"
rows = []
row_no = defaultdict(int)


def add(exp, group, pop, hand, subj, ses, block, phase, cond, measure, desc, stat, value, unit, n_trials=None):
    row_no[(exp, subj)] += 1
    rows.append(dict(
        dataset=ID, study=ID, experiment=exp, group=group, subject=f"{exp[-1]}_{subj}", subject_excluded_by_authors=False, age="",
        sex="", handedness=hand, population=pop, setting="lab", device=DEVICE, session=ses, block=block,
        trial=row_no[(exp, subj)], phase=phase, time_s="", task="sequence learning", unit="participant summary",
        condition=cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        measure=measure, measure_description=desc, statistic=stat, value=value, value_unit=unit, n_trials=n_trials))


# ---------------------------------------------------------------- Experiment 1 (authors' per-repetition table)
M1 = [("key_presses", "num_of_keys", "count", "key presses",
       "Number of key presses in the 30-s repetition of the task (num_of_keys in the authors' table)."),
      ("correct_key_presses", "correct_key_per_block", "count", "key presses",
       "Number of key presses that the authors' script scored as correct in the repetition (correct_key_per_block)."),
      ("accuracy", "accuracy", "proportion", "proportion",
       "Correct key presses divided by key presses in the repetition (accuracy in the authors' table)."),
      ("repetition_duration", "block_duration", "value", "0.1 ms (Presentation time units; 300000 = 30 s)",
       "Length of the repetition's segment of the Presentation log (block_duration in the authors' table).")]
t1 = table("young_exp1/variables_per_block.csv")
subj1 = sorted({r["subj"] for r in t1})
stim_cond = Counter((r["stim"], r["cond"]) for r in t1)
for subj in subj1:
    for task in ("B_4", "Train_4", "B_5", "Train_5"):
        for r in [x for x in t1 if x["subj"] == subj and x["task"] == task]:
            kind, ses_dir = task.split("_")
            for mname, col, stat, unit, desc in M1:
                add("Experiment 1", "young", "healthy young adults", "", subj, 1 if ses_dir == "4" else 2,
                    ("baseline" if kind == "B" else f"training repetition {int(r['block'])}") + f" (set {r['cond']})",
                    "baseline" if kind == "B" else "learning",
                    r["stim"].replace("HF_control", "HF control"), mname, desc, stat, float(r[col]), unit)
checks.internal("Experiment 1: the letter A or B (cond) is crossed with the stimulation (427 rows of each combination), so it "
                "is not the stimulation", set(stim_cond.values()) == {427} and len(stim_cond) == 4, str(dict(stim_cond)))
per_ses1 = Counter((r["subj"], r["task"].split("_")[0], r["task"].split("_")[1]) for r in t1)
checks.internal("Experiment 1: every participant has a baseline repetition and 60 training repetitions in each of 2 sessions",
                set(per_ses1.values()) == {1, 60} and len(per_ses1) == 14 * 4 and
                all(per_ses1[(s, "B", d)] == 1 and per_ses1[(s, "Train", d)] == 60 for s in subj1 for d in ("4", "5")))
checks.internal("Experiment 1: accuracy = correct key presses / key presses for every row",
                all(abs(float(r["accuracy"]) - float(r["correct_key_per_block"]) / float(r["num_of_keys"])) < 1e-9
                    for r in t1 if float(r["num_of_keys"]) > 0))

# ---------------------------------------------------------------- Experiment 2 (raw per-trial text files)
M2 = [("key_presses", "count", "key presses", "Number of key presses in the trial (the file's 'keypresses' line)."),
      ("max_complete_sequences", "count", "sequences",
       "Maximum number of complete 9-key sequences the trial's key presses could hold (the file's 'Maximum complete sequences' line)."),
      ("correct_sequences", "count", "sequences", "Number of correct sequences in the trial (the file's 'correct sequences' line)."),
      ("success_rate", "value", "percent", "Success rate of the trial as printed in the file ('Successrate', in percent).")]
ORDER = ["B", "Train", "Train_R", "Post", "FU1", "FU2"]
PHASE = {"B": "baseline", "Train": "learning", "Train_R": "other", "Post": "test", "FU1": "test", "FU2": "test"}
TRIAL_RE = re.compile(r"Trial (\d+)\s+(\d+) keypresses\s+Maximum complete sequences (\d+)\s+(\d+) correct sequences\s+"
                      r"Successrate ([\d.]+) %")
seq_by = {}
parsed = {}
letters = defaultdict(set)
for exp_dir, group, pop, hand in (("young_exp2", "young", "healthy young adults", "right"),
                                  ("older_exp2", "older", "healthy older adults", "right")):
    files = {}
    for n in z.namelist():
        m = re.fullmatch(re.escape(BASE + exp_dir + "/raw/") + r"(\d+[YO]\d+)_(\d)_(?:(B|Train|Post|FU1|FU2)_([AB])|Train_([AB])_R)\.txt", n)
        if m:
            files[(m.group(1), int(m.group(2)), m.group(3) or "Train_R")] = (m.group(4) or m.group(5), n)
    codes = sorted({k[0] for k in files})
    cond = {r["CODE"]: r for r in table(f"{exp_dir}/randomization/cond_{group}.csv")}
    auth = {(r["CODE"], r["SESSION"], r["BLOCK"]): int(float(r["var"])) for r in table(f"{exp_dir}/number_of_seq.csv")}
    mism, n_cmp, bad_cond, bad_n = 0, 0, 0, 0
    ex_mism = []
    for code in codes:
        for ses in (1, 2):
            for kind in ORDER:
                if (code, ses, kind) not in files:
                    bad_n += 1
                    continue
                letter, name = files[(code, ses, kind)]
                stim = cond[code][f"CONDITION{ses}"]
                letters[(exp_dir, code, ses)].add(letter)
                trials = [tuple(map(float, m)) for m in TRIAL_RE.findall(z.read(name).decode("utf-8", "replace"))]
                patt = re.search(r"patternlength=9\s+([\d ]+)", z.read(name).decode("utf-8", "replace")).group(1).split()
                assert len(patt) == 9 and trials, name
                parsed[(exp_dir, code, ses, kind)] = (trials, letter)
                for tno, kp, mx, cs, sr in trials:
                    lab = (kind if len(trials) == 1 else f"{kind}-T{int(tno)}") + f" (set {letter}, pattern {''.join(patt)})"
                    for mname, stat, unit, desc in M2:
                        v = {"key_presses": kp, "max_complete_sequences": mx, "correct_sequences": cs, "success_rate": sr}[mname]
                        add("Experiment 2", group, pop, hand, code, ses, lab, PHASE[kind], "iTBS" if stim == "iTBS" else "HF control",
                            mname, desc, stat, v, unit)
                    key = (code, f"ses-{ses}", "B" if kind == "B" else f"{kind}-T{int(tno)}")
                    if kind == "Train_R":     # the authors' Train-T4 is the Train_R file's trial 4 (see the check below)
                        key = (code, f"ses-{ses}", "Train-T4") if int(tno) == 4 else None
                    elif kind == "Train" and int(tno) == 4:
                        key = None
                    if key in auth:
                        n_cmp += 1
                        mism += auth[key] != cs
                        if auth[key] != cs and len(ex_mism) < 4:
                            ex_mism.append((code, key[1], key[2], auth[key], cs))
    checks.internal(f"{exp_dir}: a file of each of the 6 block types for every participant and session, and one letter A or B per "
                    f"participant and session", bad_n == 0 and all(len(v) == 1 for k, v in letters.items() if k[0] == exp_dir),
                    f"{len(codes)} participants; missing {bad_n}")
    checks.internal(f"{exp_dir}: correct sequences per trial equal the authors' number_of_seq.csv in all compared blocks",
                    mism == 0 and n_cmp == len(auth) == 15 * 2 * 17, f"{n_cmp} blocks compared, {mism} differ {ex_mism}")

# the same 7 trials are scored twice: Train_R against a second pattern (only trial 4 has correct sequences there)
r_ok, t4, why, odd = True, 0, Counter(), []
for (ed, code, ses, kind), (trials, letter) in parsed.items():
    if kind == "Train":
        tr = parsed[(ed, code, ses, "Train_R")][0]
        why["same key counts"] += [a[1] for a in trials] == [a[1] for a in tr]
        why["same max sequences"] += [a[2] for a in trials] == [a[2] for a in tr]
        if [a[1] for a in trials] != [a[1] for a in tr]:
            odd.append([i + 1 for i, (a, b) in enumerate(zip(trials, tr)) if a[1] != b[1]])
        why["Train trial 4 has no correct sequence"] += trials[3][3] == 0
        why["Train_R trial 4 has correct sequences"] += tr[3][3] > 0
        why["Train_R other trials have none"] += all(x[3] == 0 for i, x in enumerate(tr) if i != 3)
        t4 += 1
r_ok = (all(v == t4 for k, v in why.items() if not k.startswith("same")) and why["same key counts"] == 59
        and why["same max sequences"] == 59 and len(odd) == 1)
checks.internal("in 59 of the 60 Train/Train_R pairs the 7 trials have the same key press counts (one pair differs in some "
                "trials, listed in the detail); trial 4 has no correct sequence in "
                "Train but has the correct sequences the authors list as Train-T4 in Train_R, and Train_R trials 1-3 and 5-7 "
                "have none", r_ok and t4 == 60, f"{t4} pairs; passing: {dict(why)}; trials that differ in the one odd pair: {odd}")
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']} " + c.get("detail", ""))

# ---------------------------------------------------------------- published numbers
SRC = "Methods and Results, Experiments 1 and 2"
codes2 = {g: {k[1] for k in parsed if k[0] == f"{g}_exp2"} for g in ("young", "older")}
checks.published("Experiment 2: older adults", len(codes2["older"]), 15, 0, "Results, Experiment 2 (N = 15, right-handed, 9 females)")
checks.published("Experiment 2: young adults", len(codes2["young"]), 15, 0, "Results, Experiment 2 (N = 15, right-handed, 8 females)")
checks.published("Experiment 1: participants who performed the full protocol", len(subj1), 14, 0,
                 "Methods, Experiment 1 (fourteen out of 15 participants performed the full protocol)")
checks.published("Experiment 2: training blocks of 90 s per session",
                 len({re.match(r"Train-T(\d+)", r["block"]).group(1) for r in rows
                      if r["experiment"] == "Experiment 2" and r["block"].startswith("Train-T")}), 7, 0,
                 "Methods, Experiment 2 (seven 90-s blocks)")
checks.published("Experiment 1: training repetitions per session (six blocks of ten 30-s repetitions)",
                 max(per_ses1[(s_, "Train", d)] for s_ in subj1 for d in ("4", "5")), 60, 0,
                 "Methods, Experiment 1 (six blocks, each of ten 30-s repetitions)")
checks.published("Experiment 1: median repetition length in seconds (30-s repetitions)",
                 st.median(float(r["block_duration"]) for r in t1) / 10000, 30.0, 0.05, "Methods, Experiment 1 (30-s repetitions)")
checks.published("Experiment 2: training sessions per participant", len({k[2] for k in parsed}), 2, 0,
                 "Methods, Experiment 2 (two main training sessions)")
for c in checks.items:
    if c["kind"] == "published":
        print(f"  published: {c['what']}: computed {c['computed']:.3f} reported {c['reported']} ok={c['ok']}")

meta = dict(
    study="Wessel et al. 2023",
    paper=dict(authors="Wessel MJ, Beanato E, Popa T, Windel F, Vassiliadis P, Menoud P, Beliaeva V, Violante IR, Abderrahmane H, Dzialecka P, Park CH, Maceira-Elvira P, et al.",
               year=2023, title="Noninvasive theta-burst stimulation of the human striatum enhances striatal activity and motor skill learning",
               venue="Nat Neurosci 26(11):2005-2016", doi="10.1038/s41593-023-01457-7"),
    lab="Hummel (EPFL)",
    data_url="https://zenodo.org/records/14965889",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor summary format with participant codes replaced: the authors' per-repetition table of "
             "Experiment 1 (variables_per_block.csv) and the per-trial text files of Experiment 2 became one row per measure, "
             "participant, session and block. The deposit's behavioural files were read from inside its 124 GB Zenodo.zip with "
             "HTTP range requests (434 small members); the MRI data, the baseline-corrected tables, the Experiment 1 raw "
             "Presentation logs (one line per key press) and the nine-interval lines of the Experiment 2 files were not converted."),
    how=[
        "One row is one measure of one participant in one block of the sequential finger tapping task (a 9-key sequence typed "
        "as fast and accurately as possible while receiving striatal theta-burst tTIS, condition iTBS, or high-frequency "
        "control stimulation, in two sessions at least 3 days apart). Experiment 1 (young adults, fMRI) has a 30-s baseline "
        "and 60 training repetitions of 30 s per session (block is the repetition); Experiment 2 (young and older adults, "
        "group) has baseline B, training blocks of 90 s (Train-T1 to T7), a block Train_R in the files, and Post, FU1 and "
        "FU2 retests of 3 trials each.",
        "Experiment 1 measures are the authors' own per-repetition values (key presses, key presses scored correct, "
        "accuracy, and the repetition length in the Presentation log's 0.1-ms units); Experiment 2 measures are the four "
        "lines printed for each trial in the raw files (key presses, maximum complete sequences, correct sequences, success "
        "rate in percent), and the correct sequences equal the authors' number_of_seq.csv in every block.",
        "The Train_R files hold the same 7 trials as the Train files (same key press counts in 59 of 60 pairs), scored against a second 9-key "
        "pattern: only trial 4 has correct sequences there (and none in the Train file), and those are the values the "
        "authors list as Train-T4, so the fourth block was typed on the second pattern; both scorings are kept, the second "
        "as block type Train_R with phase other. The letter A or B in every file "
        "name (cond in the Experiment 1 table) is not the stimulation condition (it is crossed with it) and is not explained "
        "in the deposit; it is kept as 'set A' or 'set B' in the block label, and the stimulation (iTBS or high-frequency "
        "control) comes from the randomization tables (Experiment 2) and the stim column (Experiment 1). Subject codes are "
        "unique per experiment; ages and sex are not in the deposit.",
    ],
    unit_note="measure of one participant in one block of the sequential finger tapping task",
    differences=[],
)
stdlib.release(ID, "summary", rows, meta, checks)
