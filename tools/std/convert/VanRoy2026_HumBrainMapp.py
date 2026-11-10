"""Van Roy, Temudo, Koppelmans, Hoedlmoser, Albouy & King (2026), Hum Brain Mapp: bimanual explicit finger tapping task (FTT)
in 22 children (7-11 y) and 23 adults in an MRI scanner; training 1 and 2 (10 blocks each), a 2-block test about 1 min later and
a 10-block retest about 5 h later; one row per key press.

Raw files (untrusted data, read only) in raw_extra/VanRoy2026_HumBrainMapp/, from Zenodo record 16891677 (CC BY 4.0):
  FTT_Lifespan_OBS_E<n>.txt   the task log of participant n (one line per event: block starts, rests, one 'rep <key> at <time>' per
                              key press). These 45 files (and the .mat twins, not used) were extracted from the 5.5 GB archive
                              data_subjfolders.zip, Behavior/ folders only, with HTTP range requests (no other member was read;
                              the archive's fMRI results are not behavioural data); SOURCES.json lists each member.
  FiguresTables_SourceData.xlsx   the authors' per-participant source data of Table 1 and the figures (ages, genders, absolute and
                              normalised transition time and accuracy per block, offline changes); used for demographics and checks.
The task logs also hold other tasks (mapping, psychomotor vigilance, choice reaction, sequence check); only the four FTT runs that the
paper analyses are converted.

Run from the site folder: python3 -I tools/std/convert/VanRoy2026_HumBrainMapp.py
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "VanRoy2026_HumBrainMapp"
RAW = stdlib.raw_dir(ID)
SEQ = [4, 2, 1, 3, 4, 1]
LEGAL = {(4, 2), (2, 1), (1, 3), (3, 4), (4, 1), (1, 4)}   # transitions of the cyclic sequence 4-2-1-3-4-1-(4)
RUNS = [  # log label, authors' run name, session, epoch, phase, block_type, number of blocks, first block number (authors')
    ("TRAINING-Run1-Training", "Training 1", 1, 1, "learning", "training", 10, 1),
    ("TRAINING-Run2-Training", "Training 2", 1, 2, "learning", "training", 10, 11),
    ("SL-TEST-Training", "Test", 1, 3, "test", "test", 2, 1),
    ("SL-RETEST-Retest", "Retest", 2, 4, "test", "test", 10, 1),
]
LABELS = [r[0] for r in RUNS]
SEX = {"Female": "F", "Male": "M", "Gender fluid": "other", "Non-binary": "other"}
GROUP = {"Child": "Children", "Adult": "Adults"}
POPULATION = {"Children": "healthy children (7-11 years)", "Adults": "healthy adults (18-30 years)"}
DEVICE = ("MR-compatible keypad in each hand, index and middle fingers pressing four keys (1 left middle, 2 left index, 3 right "
          "index, 4 right middle finger); the 6-element sequence of finger numbers stayed on the screen during practice blocks; "
          "3 T MRI scanner")

checks = stdlib.Checks()

# ---------------------------------------------------------------- hashes of the raw files
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}


def sheet_rows(wb, name):
    return [r for r in wb[name].iter_rows(values_only=True)]


wb = openpyxl.load_workbook(RAW / "FiguresTables_SourceData.xlsx", read_only=True, data_only=True)
checks.internal("FiguresTables_SourceData.xlsx matches the downloaded hash",
                stdlib.sha256(RAW / "FiguresTables_SourceData.xlsx") == src["FiguresTables_SourceData.xlsx"]["sha256"])

# demographics (Table 1 source data) in the authors' order: children, then adults
demo = {}
for r in sheet_rows(wb, "Table1 - demographics"):
    if r and isinstance(r[1], str) and re.fullmatch(r"E\d+", r[1]):
        demo[r[1]] = dict(group=GROUP[r[0]], age=float(r[2]), sex=SEX[r[3]], me=r[5], sleep_q=r[7], sleep_dur=r[8], sss1=r[9],
                          pvt1=r[10], offline=r[12], sss2=r[13], pvt2=r[14])
checks.internal("Table 1 source data lists 45 participants (22 children, 23 adults)",
                len(demo) == 45 and sum(d["group"] == "Children" for d in demo.values()) == 22)


def num_sheet(name, ncols):
    out = {}
    for r in sheet_rows(wb, name):
        if r and isinstance(r[1], str) and re.fullmatch(r"E\d+", r[1]):
            out[r[1]] = np.array([np.nan if (v is None or isinstance(v, str)) else float(v) for v in r[2:2 + ncols]])
    return out


src_tt, src_acc = num_sheet("FigureS4a - Abs TT", 32), num_sheet("FigureS4b - Abs Acc", 32)
src_ntt, src_nacc = num_sheet("Figure3a - Norm TT", 32), num_sheet("Figure3b - Norm Acc", 32)
src_off = num_sheet("Figure5 - Macro-offline changes", 3)   # age, speed change, accuracy change


# ---------------------------------------------------------------- parse a task log
def parse_log(text):
    """-> {run label: [attempt, ...]} in file order; attempt = dict(start, blocks, rest, stopped, quit);
    block = dict(t0, p=[(key, time), ...]); rest = [(number of blocks before it, key, time, rest start)]"""
    att = {lab: [] for lab in LABELS}
    cur, state, rest_t0 = None, None, None
    for line in text.split("\n"):
        l = line.rstrip("\r")
        m = re.search(r"\t:\t(.+?) (START|STOP)\s*$", l)
        if m:
            lab, what = m.group(1), m.group(2)
            if lab in att and what == "START":
                cur = dict(label=lab, start=float(l.split("\t")[0]), blocks=[], rest=[], stopped=False, quit=False)
                att[lab].append(cur)
            elif lab in att and what == "STOP":
                if cur is not None and cur["label"] == lab:
                    cur["stopped"] = True
                cur = None
            else:
                cur = None            # other tasks, and the experimenter's mis-typed (WRONG) restarts
            state = None
            continue
        if cur is None:
            continue
        if re.match(r"[\d.]+\t:\tquit\s*$", l):
            cur["quit"] = True
            continue
        m = re.match(r"[\d.]+\t:\tPractice at ([\d.]+) Block (\d+)\s*$", l)
        if m:
            cur["blocks"].append(dict(t0=float(m.group(1)), p=[]))
            state = "block"
            continue
        m = re.match(r"[\d.]+\t:\tRest at ([\d.]+)", l)
        if m:
            state, rest_t0 = "rest", float(m.group(1))
            continue
        m = re.match(r"[\d.]+\t:\trep (\d) at ([\d.]+)\s*$", l)
        if m:
            key, t = int(m.group(1)), float(m.group(2))
            if state == "block":
                cur["blocks"][-1]["p"].append((key, t))
            elif state == "rest":
                cur["rest"].append((len(cur["blocks"]), key, t, rest_t0))
    return att


def block_measures(p):
    keys = np.array([k for k, _ in p])
    ts = np.array([t for _, t in p])
    dt = np.diff(ts)
    ok = np.array([(int(a), int(b)) in LEGAL for a, b in zip(keys[:-1], keys[1:])])
    return float(np.median(dt)), 100.0 * float(ok.mean())


# ---------------------------------------------------------------- build the rows
COLS = ["experiment", "group", "subject", "person", "age", "sex", "population", "session", "block", "trial", "phase", "time_s",
        "condition", "excluded_by_authors", "block_number", "block_type", "epoch", "key_pressed", "correct", "interval_ms",
        "element_type", "sequence", "sequence_position", "repetition", "time_in_block_ms"]
col = {c: [] for c in COLS}
tt_mine, acc_mine = {}, {}
rest_gaps = []
n_rest = n_excl_blocks = 0
excl_blocks = []
for pid, d in demo.items():
    text = (RAW / f"FTT_Lifespan_OBS_{pid}.txt").read_bytes().decode("utf-8")
    att = parse_log(text)
    trial = 0
    bn = {1: 0, 2: 0}      # blocks so far in each session (block_number counts every block, aborted attempts included)
    tts, accs = [], []
    for lab, run, session, epoch, phase, btype, nb, first in RUNS:
        attempts = [a for a in att[lab] if a["blocks"]]
        # blocks the authors analysed: the last attempt with blocks; participant 103's training 1 run was never completed in one
        # go: the authors' 10 blocks are blocks 1-5 of the first and blocks 1-5 of the second attempt (verified against source data)
        if pid == "E103" and lab == RUNS[0][0]:
            used = {(0, i) for i in range(5)} | {(1, i) for i in range(5)}
        else:
            used = {(len(attempts) - 1, i) for i in range(min(nb, len(attempts[-1]["blocks"])))}
        order = [(ai, bi) for ai, a in enumerate(attempts) for bi in range(len(a["blocks"]))]
        numbering = {k: n for n, k in enumerate(sorted(used, key=lambda k: order.index(k)))}
        run_tt, run_acc = [], []
        for ai, a in enumerate(attempts):
            rest_by_n = {}
            for n, key, t, rt0 in a["rest"]:
                rest_by_n.setdefault(n, []).append((key, t, rt0))
            # presses during the initial rest (before block 1) come first, then each block followed by the rests after it
            seq = [("rest", 0)] + [x for bi in range(len(a["blocks"])) for x in (("block", bi), ("rest", bi + 1))]
            for kind, idx in seq:
                if kind == "rest":
                    for key, t, rt0 in rest_by_n.get(idx, []):
                        trial += 1
                        n_rest += 1
                        rest_gaps.append(t - rt0)
                        prev = f"{run} block {first + numbering[(ai, idx - 1)]}" if idx and (ai, idx - 1) in used else None
                        label = f"Rest after {prev}" if prev else (f"Rest before {run} block {first}" if idx == 0 else
                                                                  f"Rest after {run} aborted attempt {ai + 1} block {idx}")
                        for c, v in (("block", label), ("trial", trial), ("time_s", t - a["start"]), ("condition", "rest"),
                                     ("excluded_by_authors", True), ("block_number", None), ("block_type", ""),
                                     ("key_pressed", str(key)), ("correct", None), ("interval_ms", None), ("element_type", ""),
                                     ("sequence", ""), ("sequence_position", None), ("repetition", None),
                                     ("time_in_block_ms", None)):
                            col[c].append(v)
                        col["experiment"].append(None)
                        for c, v in (("group", d["group"]), ("subject", pid), ("person", pid), ("age", d["age"]), ("sex", d["sex"]),
                                     ("population", POPULATION[d["group"]]), ("session", session), ("phase", phase),
                                     ("epoch", epoch)):
                            col[c].append(v)
                    continue
                b = a["blocks"][idx]
                assert len(b["p"]) == 48, (pid, lab, ai, idx, len(b["p"]))
                bn[session] += 1
                block_no = bn[session]
                is_used = (ai, idx) in used
                label = f"{run} block {first + numbering[(ai, idx)]}" if is_used else f"{run} aborted attempt {ai + 1} block {idx + 1}"
                if not is_used:
                    n_excl_blocks += 1
                    excl_blocks.append((pid, label))
                else:
                    m_tt, m_acc = block_measures(b["p"])
                    run_tt.append(m_tt)
                    run_acc.append(m_acc)
                prev_t = None
                for j, (key, t) in enumerate(b["p"]):
                    trial += 1
                    ok = None if j == 0 else (b["p"][j - 1][0], key) in LEGAL
                    for c, v in (("experiment", None), ("group", d["group"]), ("subject", pid), ("person", pid), ("age", d["age"]),
                                 ("sex", d["sex"]), ("population", POPULATION[d["group"]]), ("session", session),
                                 ("block", label), ("trial", trial), ("phase", phase), ("time_s", t - a["start"]),
                                 ("condition", "sequence"), ("excluded_by_authors", not is_used), ("block_number", block_no),
                                 ("block_type", btype), ("epoch", epoch), ("key_pressed", str(key)), ("correct", ok),
                                 ("interval_ms", None if j == 0 else round((t - b["p"][j - 1][1]) * 1000, 3)),
                                 ("element_type", "sequence"), ("sequence", "-".join(map(str, SEQ))),
                                 ("sequence_position", j % 6 + 1), ("repetition", j // 6 + 1),
                                 ("time_in_block_ms", round((t - b["t0"]) * 1000, 3))):
                        col[c].append(v)
        tts += run_tt + [np.nan] * (nb - len(run_tt))
        accs += run_acc + [np.nan] * (nb - len(run_acc))
    tt_mine[pid], acc_mine[pid] = np.array(tts), np.array(accs)

t = pd.DataFrame(col)
n = len(t)
for c, v in (("dataset", ID), ("study", ID), ("experiment", ""), ("subject_excluded_by_authors", False), ("handedness", ""),
             ("setting", "lab"), ("device", DEVICE), ("task", "sequence learning"), ("unit", "keypress"), ("pert_type", "none"),
             ("pert_value", 0), ("pert_unit", ""), ("feedback", "normal"), ("rt_ms", ""), ("rt_correct_ms", ""), ("stimulus", ""),
             ("correct_key", ""), ("triplet_type", "")):
    t[c] = v

# ---------------------------------------------------------------- internal checks
ids = list(demo)
n_blocks = {p: int(np.sum(~np.isnan(tt_mine[p]))) for p in ids}
checks.internal("45 participants with a log; 1,439 analysed blocks of 48 key presses (32 each, except one child's training 2: 9)",
                len(ids) == 45 and sum(n_blocks.values()) == 45 * 32 - 1 and sum(1 for p in ids if n_blocks[p] < 32) == 1,
                f"{sum(n_blocks.values())} blocks")
used_rows = t[~t.excluded_by_authors]
checks.internal("every analysed block has 48 key presses and the analysed rows number 69,072",
                (used_rows.groupby(["subject", "block"]).size() == 48).all() and len(used_rows) == 1439 * 48, f"{len(used_rows)} rows")
checks.internal("rows kept as logged but not analysed by the authors: 3 blocks of aborted run attempts (2 + 1, 48 key presses "
                f"each) and {n_rest} key presses during the 20 s rests", n_excl_blocks == 3 and (t.excluded_by_authors.sum() == 3 * 48 + n_rest),
                f"excluded blocks {n_excl_blocks}, rest presses {n_rest}")
checks.internal("every key press logged during a rest lies within 0 to 20.1 s of that rest's start",
                min(rest_gaps) >= 0 and max(rest_gaps) <= 20.1, f"{min(rest_gaps):.3f} to {max(rest_gaps):.3f} s")
checks.internal("key codes are 1-4 only", set(t.key_pressed) == {"1", "2", "3", "4"})


def same(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and (np.nanmax(np.abs(a - b)) if np.isfinite(a).any() else 0) < tol)


for what, mine, theirs, tol in (("median transition time per block (s)", tt_mine, src_tt, 6e-4),
                                ("% correct transitions per block", acc_mine, src_acc, 6e-4)):
    ok_n = sum(same(mine[p], theirs[p], tol) for p in ids)
    checks.internal(f"{what}: equal to the authors' source data (Figure S4) for every block of every participant", ok_n == 45,
                    f"{ok_n} of 45 participants equal")
ntt = {p: tt_mine[p] / tt_mine[p][0] for p in ids}
nacc = {p: acc_mine[p] / acc_mine[p][0] for p in ids}
for what, mine, theirs in (("normalised transition time", ntt, src_ntt), ("normalised accuracy", nacc, src_nacc)):
    ok_n = sum(same(mine[p], theirs[p], 6e-4) for p in ids)
    checks.internal(f"{what} (divided by the first block of training 1): equal to the authors' source data (Figure 3)",
                    ok_n == 45, f"{ok_n} of 45 participants equal")
# macro-offline change: end of training (the 2 test blocks) against the start of retest (its first 2 blocks)
off_sp = {p: np.mean(ntt[p][20:22]) - np.mean(ntt[p][22:24]) for p in ids}
off_ac = {p: np.mean(nacc[p][22:24]) - np.mean(nacc[p][20:22]) for p in ids}
d_sp = max(abs(off_sp[p] - src_off[p][1]) for p in ids)
d_ac = max(abs(off_ac[p] - src_off[p][2]) for p in ids)
off_note = None
if d_sp < 6e-4 and d_ac < 6e-4:
    checks.internal("macro-offline changes in normalised speed and accuracy (positive = improvement): equal to the authors' "
                    "Figure 5 source data for every participant", True, f"max abs difference {d_sp:.1e}, {d_ac:.1e}")
else:   # the paper's description of the offline change does not reproduce the authors' own values: use their values
    off_sp = {p: src_off[p][1] for p in ids}
    off_ac = {p: src_off[p][2] for p in ids}
    off_note = ("The macro-offline changes computed from these rows as the paper describes them (mean of the 2 test blocks minus "
                "mean of the first 2 retest blocks, each divided by the first block of training 1) differ from the values in the "
                f"authors' Figure 5 source data (largest difference {d_sp:.2f} in speed, {d_ac:.3f} in accuracy), so the "
                "macro-offline statistics checked here (t, df, Hedges' g) are computed from the authors' source-data values.")

# ---------------------------------------------------------------- published numbers
grp = pd.Series({p: demo[p]["group"] for p in ids})
ch, ad = grp[grp == "Children"].index, grp[grp == "Adults"].index
T1 = "Methods, Participants / Table 1"
checks.published("children: participants", len(ch), 22, 0, T1)
checks.published("adults: participants", len(ad), 23, 0, T1)
for g, idx, (nf, nm, nx, ma, sa) in (("children", ch, (14, 7, 1, 9.5, 1.3)), ("adults", ad, (16, 6, 1, 23.4, 3.3))):
    sx = pd.Series({p: demo[p]["sex"] for p in idx})
    checks.published(f"{g}: female participants", (sx == "F").sum(), nf, 0, T1)
    checks.published(f"{g}: male participants", (sx == "M").sum(), nm, 0, T1)
    checks.published(f"{g}: non-binary or gender-fluid participants", (sx == "other").sum(), nx, 0, T1)
    ages = np.array([demo[p]["age"] for p in idx])
    checks.published(f"{g}: mean age (years)", ages.mean(), ma, 0.05, T1)
    checks.published(f"{g}: SD of age (years)", ages.std(ddof=1), sa, 0.05, T1)


def hours(v):
    return v.hour + v.minute / 60 if hasattr(v, "hour") else np.nan


for g, idx, (so, sd_, sdur) in (("children", ch, (5.1, 0.7, 9.83)), ("adults", ad, (5.1, 0.4, 8.3))):
    off = np.array([demo[p]["offline"] for p in idx], float)
    checks.published(f"{g}: mean offline period between training and retest (h)", off.mean(), so, 0.05, T1)
    checks.published(f"{g}: SD of the offline period (h)", off.std(ddof=1), sd_, 0.05, T1)
    checks.published(f"{g}: mean sleep duration the night before training (h)", np.nanmean([hours(demo[p]["sleep_dur"]) for p in idx]),
                     sdur, 0.05, T1)


def hedges(a, b):
    na, nb_ = len(a), len(b)
    sp = math.sqrt(((na - 1) * np.var(a, ddof=1) + (nb_ - 1) * np.var(b, ddof=1)) / (na + nb_ - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb_) - 9))


R3 = "Results 3.2.3, macro-offline changes"
a, b = np.array([off_sp[p] for p in ch]), np.array([off_sp[p] for p in ad])
tst = stats.ttest_ind(a, b, equal_var=True)
checks.published("macro-offline change in speed, children vs adults: t(43)", tst.statistic, 2.059, 0.0005, R3)
checks.published("macro-offline change in speed: one-tailed p", tst.pvalue / 2, 0.023, 0.0005, R3)
checks.published("macro-offline change in speed: Hedges' g", hedges(a, b), 0.60, 0.005, R3)
a, b = np.array([off_ac[p] for p in ch]), np.array([off_ac[p] for p in ad])
tst = stats.ttest_ind(a, b, equal_var=False)
checks.published("macro-offline change in accuracy, children vs adults: Welch t", tst.statistic, 2.527, 0.0005, R3)
checks.published("macro-offline change in accuracy: Welch df", tst.df, 28.685, 0.0005, R3)
d_avg = (np.mean(a) - np.mean(b)) / math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
g_welch = d_avg * (1 - 3 / (4 * tst.df - 1))
print(f"  [info] accuracy effect size: pooled-SD Hedges' g {hedges(a, b):.4f}; Welch-test version {g_welch:.4f}")
checks.published("macro-offline change in accuracy: Hedges' g (Welch-test version: square root of the mean of the two variances, "
                 "correction with the Welch df)", g_welch, 0.74, 0.005, R3)

R2 = "Results 3.2.1, normalised transition time"
for what, sl, ref, dfs in (("training 1 (10 blocks)", slice(0, 10), 8.85, (1, 43)), ("training 2 (10 blocks)", slice(10, 20), 4.10, (1, 42)),
                           ("retest (10 blocks)", slice(22, 32), 0.03, (1, 43))):
    m = pd.Series({p: np.mean(ntt[p][sl]) if not np.isnan(ntt[p][sl]).any() else np.nan for p in ids}).dropna()
    gg = [m[[p for p in m.index if grp[p] == k]] for k in ("Children", "Adults")]
    checks.published(f"group effect F on normalised transition time, {what}: F{dfs}, n = {len(m)}",
                     stats.f_oneway(*gg).statistic, ref, 0.005, R2)
    checks.internal(f"degrees of freedom of the {what} group effect = {dfs}", len(m) - 2 == dfs[1])

for c in checks.items:
    if c["kind"] == "published":
        print(f"  {'ok ' if c['ok'] else 'BAD'} {c['what']}: computed {c['computed']:.4f}, reported {c['reported']}")
    elif not c["ok"] or c.get("detail"):
        print(f"  {'ok ' if c['ok'] else 'FAILED'} [internal] {c['what']} {c.get('detail', '')}")

# ---------------------------------------------------------------- release
names = [c[1] for c in stdlib.columns_for("keypress")]
for c in names:
    if c not in t.columns:      # a module column added later stays empty
        t[c] = ""
t = t[[c for c in names if c in t.columns]]
part_rows = [dict(person=p, offline_period_h=demo[p]["offline"], sleep_quality=demo[p]["sleep_q"], sleep_duration_h=hours(demo[p]["sleep_dur"]),
                  sss_training=demo[p]["sss1"], sss_retest=demo[p]["sss2"],
                  pvt_mean_rt_training_ms=demo[p]["pvt1"] if not isinstance(demo[p]["pvt1"], str) else None,
                  pvt_mean_rt_retest_ms=demo[p]["pvt2"] if not isinstance(demo[p]["pvt2"], str) else None) for p in ids]
part_cols = [("offline_period_h", "number", "hours", "Time between the end of training and the start of retest (authors' Table 1)."),
             ("sleep_quality", "integer", "1-6", "Self-rated sleep quality of the night before training (authors' Table 1)."),
             ("sleep_duration_h", "number", "hours", "Self-reported sleep duration of the night before training (authors' Table 1)."),
             ("sss_training", "integer", "1-7", "Stanford Sleepiness Scale before training."),
             ("sss_retest", "integer", "1-7", "Stanford Sleepiness Scale before retest."),
             ("pvt_mean_rt_training_ms", "number", "ms", "Mean reaction time on the psychomotor vigilance task before training; empty if not recorded."),
             ("pvt_mean_rt_retest_ms", "number", "ms", "Mean reaction time on the psychomotor vigilance task before retest; empty if not recorded.")]
entry = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
meta = dict(
    study=ID,
    paper={k: entry["papers"][0][k] for k in ("authors", "year", "title", "venue", "doi")},
    lab=entry["lab"],
    data_url="https://zenodo.org/records/16891677",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted the authors' task logs (Behavior folders of data_subjfolders.zip) of the finger tapping task into one table "
             "of key presses, added each participant's group, age and gender from the authors' Table 1 source data, and replaced "
             "participant identifiers with codes. The mapping, psychomotor vigilance, choice reaction and sequence-check tasks in "
             "the same logs, the MATLAB twins of the logs and the fMRI results of the deposit are not included."),
    how=[
        "One row is one key press in the bimanual finger tapping task: participants tapped the 6-element sequence 4-2-1-3-4-1 "
        "(1 left middle, 2 left index, 3 right index, 4 right middle finger) as fast and accurately as possible while the "
        "sequence stayed on the screen; a block ends after 48 key presses (about 8 repetitions) and is followed by a 20 s rest. "
        "key_pressed is the key code 1-4; there is no cue per element, so stimulus and rt_ms are empty. interval_ms is the time "
        "from the previous key press of the block (the authors' transition time; empty for the first press) and time_in_block_ms "
        "the time from the start of the block, both from the authors' log; time_s is seconds from the start of the run's log.",
        "correct is true when the transition from the previous key press to this one follows the order of the sequence "
        "(4>2, 2>1, 1>3, 3>4, 4>1, 1>4), which is how the authors count correct transitions (47 per block); it is empty for the "
        "first press of a block. sequence_position and repetition count the presses from the start of the block (1-6 and 1-8) "
        "and so ignore errors. epoch is the authors' run: 1 Training 1, 2 Training 2 (blocks 11-20 in their numbering), "
        "3 Test (2 blocks about 1 min after training 2), 4 Retest (10 blocks, session 2, about 5 h later, awake).",
        "excluded_by_authors marks rows kept as logged that are not in the authors' analysed blocks: key presses logged during the "
        "20 s rests (block 'Rest ...') and the 3 blocks of run attempts that were quit and restarted; the analysed blocks "
        "reproduce the authors' per-block transition time and accuracy (source data) exactly. One participant's Training 1 run "
        "was never completed in one go: the authors' 10 blocks are blocks 1-5 of the first and 1-5 of the second attempt.",
        "group is Children (7-11 years) or Adults (18-30 years); sex is the authors' gender (non-binary and gender-fluid as other).",
    ],
    unit_note="key press (one tap of the finger sequence tapping task)",
    differences=[
        "One child has 9 instead of 10 blocks in Training 2 (the run was quit after block 9); the authors have no value for it "
        "either, and the paper's Training 2 statistics use 21 children.",
        *([off_note] if off_note else []),
        "The paper's text does not describe the restarted runs or the key presses made during rests; both are in the logs and are "
        "flagged here.",
    ],
)
stdlib.release(ID, "keypress", t, meta, checks, participants=part_rows, participant_columns=part_cols)
