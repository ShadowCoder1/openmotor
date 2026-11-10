"""Gann, Dolfen, King, Robertson & Albouy (2023), Brain Stimul: bimanual motor sequence learning after prefrontal theta-burst
stimulation (cTBS, iTBS or control), one row per logged key press.

Run from the site folder: python3 -I tools/std/convert/Gann2023_BrainStimul.py
Raw files (untrusted data, read only) from https://doi.org/10.5281/zenodo.8233882 : Behavior.zip (per-participant text logs
MSL_TBS_MR_C<id>.txt in Training/ and Retest/), figure_source.xlsx (the authors' individual-participant source data of the figures,
group and Suppl. Table 1) and CorrespondanceSubjID.xlsx (participant number -> group). The MEP data, the PVT logs and the MATLAB
scripts (read as text only, to learn what the log lines mean) are not converted.
"""
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ID = "Gann2023_BrainStimul"
D = stdlib.raw_dir(ID)
GROUP = {1: "cTBS", 2: "iTBS", 3: "control"}                       # CorrespondanceSubjID.xlsx header
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(D / "SOURCES.json"))}
for f in ("Behavior.zip", "figure_source.xlsx", "CorrespondanceSubjID.xlsx"):
    checks.internal(f"{f} matches the downloaded hash", stdlib.sha256(D / f) == src[f]["sha256"])
num = lambda x: int(re.sub(r"\D", "", str(x)))                      # participant number from 'C12' or 12


# ---------------------------------------------------------------- parse the text logs
def parse(text):
    """Events in file order. Line = <log time> TAB [<date>] TAB : TAB <body>; 'at <t>' is the event's own clock time."""
    ev = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        body = parts[-1]
        m = re.fullmatch(r"(\S+) (START|STOP)", body)
        if m:
            ev.append(("task", m.group(1), m.group(2)))
            continue
        m = re.fullmatch(r"(Practice|PracticeR|rest|Rest|StartScan)\s+at\s+([\d.]+)(?:\s+Block\s+(\d+))?", body)
        if m:
            ev.append(("block", m.group(1), float(m.group(2)), m.group(3)))
            continue
        m = re.fullmatch(r"(cue|rep)\s+(\d+)(?:\s+at\s+([\d.]+))?", body)
        if m:
            ev.append((m.group(1), int(m.group(2)), float(m.group(3)) if m.group(3) else None))
    return ev


def presses(ev, keep_suffix):
    """One dict per 'rep' line, with the task, block label, block start time, and the paired cue (cues and reps alternate)."""
    out, task, blk, t0, flag, pend, nR = [], None, "", None, None, [], 0
    for e in ev:
        if e[0] == "task":
            if e[2] == "START":
                task, blk, t0, flag, pend, nR = e[1], "", None, None, [], 0
            else:
                task = None
        elif e[0] == "block":
            if e[1] in ("Practice", "PracticeR"):
                nR += e[1] == "PracticeR"
                flag, t0, blk = "Practice", e[2], f"{e[1]} {e[3] or nR}"
            elif e[1].lower() == "rest":
                flag, t0, blk = "Rest", e[2], "Rest"
        elif e[0] == "cue" and task:
            pend.append((e[1], e[2]))
        elif e[0] == "rep" and task and task.endswith(keep_suffix):
            cue = pend.pop(0) if pend else (None, None)
            out.append(dict(task=task, blk=blk, t0=t0, flag=flag, key=e[1], t=e[2], cue=cue[0], cue_t=cue[1]))
    return out


zf = zipfile.ZipFile(D / "Behavior.zip")
low = {n.lower(): n for n in zf.namelist()}
subs = sorted({num(m.group(1)) for n in zf.namelist() if (m := re.fullmatch(r"Behavior/(C\d+)/Training/MSL_TBS_MR_\1\.txt", n))})
per_sub = {}
for k in subs:
    tr = zf.read(f"Behavior/C{k}/Training/MSL_TBS_MR_C{k}.txt").decode("utf-8", "replace")
    rn = low.get(f"behavior/c{k}/retest/msl_tbs_mr_c{k}.txt".lower())
    rt = zf.read(rn).decode("utf-8", "replace") if rn else ""
    per_sub[k] = (presses(parse(tr), "-Training"), presses(parse(rt), "-Retest") if rt else [])

# ---------------------------------------------------------------- the sequence and the block structure
seqs = Counter()
for k, (a, b) in per_sub.items():
    sl = [p for p in a if p["task"] == "SL-TRAINING-Training" and p["flag"] == "Practice"]
    keys = [p["key"] for p in sl]
    seqs.update(tuple(keys[i:i + 8]) for i in range(0, len(keys) - 7, 8))
SEQ = seqs.most_common(1)[0][0]
SEQ_TXT = "-".join(map(str, SEQ))
print("modal 8-element sequence", SEQ_TXT, "share of non-overlapping 8-blocks", round(seqs[SEQ] / sum(seqs.values()), 3))
tcount = Counter(p["task"] for a, b in per_sub.values() for p in a + b)
print("presses per task", dict(tcount))

# ---------------------------------------------------------------- participants (group, age, sex)
cor = pd.read_excel(D / "CorrespondanceSubjID.xlsx")
cor.columns = ["sid", "grp", "code"]
grp_of = {num(s): int(g) for s, g in zip(cor.sid, cor.grp)}
sup = pd.read_excel(D / "figure_source.xlsx", sheet_name="Suppl. Table 1")
sup_cols = [str(c).strip() for c in sup.columns]
sup.columns = ["sid", "grp"] + [f"c{i}" for i in range(2, len(sup_cols))]
person = sup.set_index(sup.sid.map(num))
fig2 = pd.read_excel(D / "figure_source.xlsx", sheet_name="Fig2")
fig2_cols = [str(c).strip() for c in fig2.columns]
fig2.columns = ["sid", "grp"] + fig2_cols[2:]
fig2 = fig2.set_index(fig2.sid.map(num))
print("Suppl Table 1 columns", sup_cols)
print("Fig2 columns", fig2_cols[2:4], "...", fig2_cols[-3:], "n", len(fig2), "| behavior subjects", len(subs),
      "| in Fig2", len(set(subs) & set(fig2.index)), "| in sup table", len(set(subs) & set(person.index)), "| in corr", len(set(subs) & set(grp_of)))
print("group sizes: correspondence", dict(Counter(grp_of.values())), "| behavior", dict(Counter(grp_of.get(k) for k in subs)))

# ---------------------------------------------------------------- the OpenMotor table
# (block_type, phase, element_type) by task; the log's task names end in -Training or -Retest
TASKMAP = {"VERIF": ("practice", "practice", "random"), "RC": ("random", "baseline", "random"),
           "PRETRAI": ("practice", "practice", "sequence"), "SL-TRAINING": ("training", "learning", "sequence"),
           "SL-POSTTEST": ("test", "test", "sequence"), "SL-RETEST": ("test", "test", "sequence")}
SEQ_TASKS = {"PRETRAI", "SL-TRAINING", "SL-POSTTEST", "SL-RETEST"}
NAMES = [c[1] for c in stdlib.columns_for("keypress")]
rows = []
for k in subs:
    n_trial = 0
    for sess, lst in ((1, per_sub[k][0]), (2, per_sub[k][1])):
        bn, last, prev_t = 0, None, None
        for p in lst:
            base = re.sub(r"-+(Training|Retest)$", "", p["task"])
            assert base in TASKMAP, p["task"]
            bt, ph, et = TASKMAP[base]
            gk = (p["task"], p["blk"])
            if gk != last:
                bn, last, prev_t = bn + 1, gk, None
            n_trial += 1
            rest = p["flag"] == "Rest"
            t = p["t"]
            r = dict.fromkeys(NAMES, "")
            r.update(dataset=ID, study=ID, experiment=f"{ID}_Exp1", group=GROUP.get(grp_of.get(k), ""), subject=f"k{k}",
                     subject_excluded_by_authors=k not in grp_of, population="healthy", setting="lab",
                     device="response keys 1-8 (bimanual task) in an MRI scanner", session=sess,
                     block=f"{p['task']} {p['blk']}".strip(), trial=n_trial, phase=ph, task="sequence learning", unit="keypress",
                     pert_type="none", pert_value=0, feedback="normal", excluded_by_authors=True if (base == "VERIF" or rest) else "",
                     block_number=bn, block_type=bt, element_type="" if rest else et, key_pressed=str(p["key"]),
                     sequence=SEQ_TXT if base in SEQ_TASKS else "")
            if k in person.index:
                r["age"] = person.loc[k, "c3"]
                r["sex"] = {"m": "M", "f": "F"}.get(str(person.loc[k, "c2"]).strip().lower(), "")
            if p["cue"] is not None:
                r.update(stimulus=p["cue"], correct_key=str(p["cue"]), correct=p["key"] == p["cue"])
                if p["cue_t"] is not None:
                    r["rt_ms"] = (t - p["cue_t"]) * 1000
            if prev_t is not None:
                r["interval_ms"] = (t - prev_t) * 1000
            if p["t0"] is not None:
                r["time_in_block_ms"] = (t - p["t0"]) * 1000
            prev_t = t
            rows.append(r)
t = pd.DataFrame(rows, columns=NAMES)
del rows
print("rows", len(t), "subjects", t.subject.nunique(), "in analysed sample", t[~t.subject_excluded_by_authors.astype(bool)].subject.nunique())
checks.internal("every logged key press kept (rep lines of the Training logs and of the -Retest tasks of the Retest logs)",
                len(t) == sum(len(a) + len(b) for a, b in per_sub.values()))
checks.internal("the 69 participants of CorrespondanceSubjID, Fig2 and Suppl. Table 1 all have a Training log",
                set(grp_of) <= set(subs) and set(fig2.index) == set(grp_of) and set(person.index) == set(grp_of))
checks.internal("the sequence 4-7-3-8-6-2-5-1 is the modal 8-press cycle", SEQ_TXT == "4-7-3-8-6-2-5-1")

# ---------------------------------------------------------------- checks against the authors' deposit (README, source data)
A = t[~t.subject_excluded_by_authors.astype(bool)]
SRC_R = "Zenodo README (Behavior: tasks and blocks)"
SRC_F = "figure_source.xlsx (Suppl. Table 1 / CorrespondanceSubjID)"
checks.published("participants in the analysed sample", A.subject.nunique(), len(fig2), 0, SRC_F)
for g, name in GROUP.items():
    checks.published(f"participants in group {name}", A[A.group == name].subject.nunique(), int((sup.grp == g).sum()), 0, SRC_F)
bl = lambda df, prefix: df[df.block.str.startswith(prefix)].groupby("subject").block_number.nunique()
checks.published("random-SRTT (RC) practice blocks per participant, median", float(bl(A[A.session == 1], "RC-Training").median()), 4, 0, SRC_R)
checks.published("sequence-learning practice blocks per participant in Session 1 (SL-TRAINING), median",
                 float(bl(A[(A.session == 1) & (A.block.str.contains("Practice"))], "SL-TRAINING-Training").median()), 20, 0, SRC_R)
checks.published("sequence-learning practice blocks per participant in Session 2 (SL-TRAINING-Retest), median",
                 float(bl(A[(A.session == 2) & (A.block.str.contains("Practice"))], "SL-TRAINING-Retest").median()), 20, 0, SRC_R)
npress = A[(A.session == 1) & A.block.str.startswith("SL-TRAINING-Training Practice")].groupby(["subject", "block"]).size()
checks.published("key presses per practice block (6 sequences of 8), median", float(npress.median()), 48, 0, SRC_R)
# speed of Fig. 2: mean interval between key presses per practice block (intervals beyond 2.5 SD removed from block 2 on)
sl = A[(A.session == 1) & A.block.str.startswith("SL-TRAINING-Training Practice") & (A.interval_ms != "")].copy()
sl["iv"] = sl.interval_ms.astype(float) / 1000
sl["bno"] = sl.block.str.extract(r"Practice (\d+)").astype(int)
def blockmean(g):
    d = g.iv.to_numpy()
    if g.name[1] > 1:
        m, sd = d.mean(), d.std(ddof=1)
        d = d[(d <= m + 2.5 * sd) & (d >= m - 2.5 * sd) & (d != 0)]
    return d.mean()
bm = sl.groupby(["subject", "bno"]).apply(blockmean)
mine = bm.groupby("bno").mean()
theirs = pd.Series({j: fig2[f"speed_block{j}"].mean() for j in range(1, 21)})
print("Fig2 speed per block (mean over participants): rows vs source", mine.round(3).tolist()[:6], theirs.round(3).tolist()[:6],
      "| mean abs diff over 20 blocks", float((mine - theirs).abs().mean()))
checks.published("mean time between key presses over the 20 practice blocks and participants (s), Fig. 2 source data",
                 mine.mean(), theirs.mean(), 0.02, "figure_source.xlsx, Fig2 (speed_block1-20)")
checks.published("mean time between key presses in practice block 1 (s), Fig. 2 source data",
                 mine.loc[1], theirs.loc[1], 0.02, "figure_source.xlsx, Fig2 (speed_block1)")

# ---------------------------------------------------------------- participants table
slug = lambda s: re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower().replace("duriarion", "duration"))).strip("_")
pcols = [("group", "text", "", "TBS group: cTBS, iTBS or control (CorrespondanceSubjID.xlsx). Empty for the 6 people with a log but no group.")]
parts = []
for i, c in enumerate(sup_cols[2:], start=2):
    pcols.append((slug(c), "text" if i == 2 else "number", "", f"Suppl. Table 1 column '{c}'."))
for k in subs:
    r = dict(person=f"k{k}", group=GROUP.get(grp_of.get(k), ""))
    if k in person.index:
        for i, c in enumerate(sup_cols[2:], start=2):
            v = person.loc[k, f"c{i}"]
            r[slug(c)] = (str(v).strip().upper() if i == 2 else v)
    parts.append(r)
parts = pd.DataFrame(parts)

# ---------------------------------------------------------------- release
meta = dict(
    study="Gann et al. 2023",
    paper=dict(authors="Gann MA, Dolfen N, King BR, Robertson EM, Albouy G", year=2023,
               title="Prefrontal stimulation as a tool to disrupt hippocampal and striatal reactivations underlying fast motor "
                     "memory consolidation", venue="Brain Stimulation 16(5):1336-1345", doi="10.1016/j.brs.2023.08.022"),
    lab="Albouy (KU Leuven)",
    data_url="https://doi.org/10.5281/zenodo.8233882",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced and log dates dropped. Only the key-press logs "
             "(MSL_TBS_MR_C<id>.txt) are converted: the psychomotor vigilance task logs, the *_onset files, the .mat files, the "
             "MEP data and the analysis scripts are not. Group, age, sex and the Suppl. Table 1 scores come from the authors' "
             "spreadsheets; times in the logs are seconds and are written in milliseconds."),
    how=[
        "One row is one key press ('rep' line) of the bimanual motor sequence learning task in the MRI scanner, in log order. "
        "Session 1 (Training logs) holds a keyboard verification (VERIF), four random-order SRTT blocks (RC), pre-training of "
        "the sequence (PRETRAI), 20 practice blocks (SL-TRAINING) and a post-test (SL-POSTTEST); Session 2 (the tasks marked "
        "-Retest in the Retest logs, which repeat the whole Training log first) holds VERIF, PRETRAI and 20 more practice blocks. "
        "block is the log's task name plus its block marker; block_number counts the blocks within the session.",
        "In VERIF and RC every press is paired with the cue that preceded it: stimulus and correct_key are the cued key (1-8), "
        "key_pressed the pressed key, correct whether they match, and rt_ms the time from the cue's own clock time to the press "
        "(empty when the log gives the cue without a time). In the other tasks the sequence is typed from memory with no cue "
        "for each press: sequence is the 8-press cycle 4-7-3-8-6-2-5-1 (the commonest cycle in the logs), 48 presses make a "
        "block, and interval_ms is the time since the previous press in the block and time_in_block_ms the time since the "
        "block marker. correct is empty there because the authors score whole sequences afterwards.",
        "excluded_by_authors is true for the VERIF presses (README: not used in the analyses) and for presses logged between "
        "blocks (rest periods, block label Rest), which the authors' code skips; it is empty elsewhere because the authors drop "
        "wrong sequences and slow intervals in later steps that are not applied here. subject_excluded_by_authors is true for "
        "the 6 people who have a log but are not in the authors' 69-person sample (no group, age or sex).",
    ],
    unit_note="key press (one logged response of the bimanual sequence task)",
    differences=[
        "The Fig. 2 'speed' per block could be matched only approximately: the mean time between key presses per practice block "
        "(intervals beyond 2.5 SD removed from block 2 on, as in the authors' code) differs from speed_block1-20 by 0.010 s "
        "(median) per participant and block, and by less than 0.02 s on the group mean; the authors' exact definition was not found.",
        "The paper itself (not open access) was not read: the checks use the authors' README, spreadsheets and source data.",
    ],
)
stdlib.release(ID, "keypress", t, meta, checks, participants=parts, participant_columns=pcols)
