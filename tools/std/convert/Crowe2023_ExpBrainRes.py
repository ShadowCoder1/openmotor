"""Crowe, Smeets & Brenner (2023), Exp Brain Res: finger interception of a moving virtual target on a screen while the gap between two red items
(obstacles, or items to ignore) changes and the target jumps sideways. Main experiment (24 participants, 12 obstacles and 12 irrelevant-items
group) and replication (12 participants, obstacles), 400 trials each, Optotrak finger position at 500 Hz; one row per trial (recording),
module timeseries with one samples CSV per trial.

Run from the site folder: python3 -I tools/std/convert/Crowe2023_ExpBrainRes.py
Raw files (untrusted data, read only) in raw_extra/Crowe2023_ExpBrainRes/, from https://osf.io/4fjru/: MainExperiment.zip and
ReplicationExperiment.zip, read with zipfile. Inside, Participant<p>_1/Participant<p>_<trial>.txt (one per trial: lines 'time x y z' in
the Optotrak frame, plus event lines such as '0.000 condition: <name>', '0.300 target jump x -0.5', '0.760 hit target') and
Participant<p>_1/input/inputCondition<1|2> (the experiment's input files). The authors' ReadIn.m, ProcessData.m,
ResponseModulationMain.m and AnalyseMainExperiment.m were read as text (never run): their line handling is re-implemented below to
reproduce the paper's response-modulation statistics. Mac resource-fork files (._*) in the zips are ignored.
"""
import json
import math
import re
import statistics as st
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Crowe2023_ExpBrainRes"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
CORE_NAMES = [c[1] for c in stdlib.CORE]
MOD_NAMES = [c[1] for c in stdlib.load_module("timeseries")]
CONDITIONS = ["obstacles outward target right early", "obstacles outward target left early", "obstacles inward target right early",
              "obstacles inward target left early", "obstacles outward target right late", "obstacles outward target left late",
              "obstacles inward target right late", "obstacles inward target left late"]  # the authors' Conditions.csv, in order
OBSTACLE_GROUP = {1, 2, 3, 5, 7, 10, 11, 13, 17, 18, 20, 23}  # the authors' ResponseModulationMain.m
NUM = re.compile(r"-?\d+(\.\d+)?$")
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
SAMPLES_DIR = Path("tmp_crowe_samples")
SAMPLES_DIR.mkdir(exist_ok=True)

rows, extra = [], []
n_unknown_lines = Counter()
resp = {}  # (experiment, participant) -> condition -> list of 500-sample velocity vectors (authors' ProcessData)
cond_counts = {}
jump_counts = Counter()
no_jump = []
n_missing = [0]
n_empty_trials = []
for exp, zname, top in (("main", "MainExperiment.zip", "MainExperiment"), ("replication", "ReplicationExperiment.zip", "Replication Experiment")):
    assert stdlib.sha256(RAW / zname) == src[zname]["sha256"]
    z = zipfile.ZipFile(RAW / zname)
    names = {i.filename: i for i in z.infolist() if not i.is_dir() and "/._" not in i.filename}
    parts = sorted({int(m.group(1)) for n in names for m in [re.match(re.escape(top) + r"/Participant(\d+)_1/", n)] if m})
    for p in parts:
        grp = ""
        if exp == "main":
            ic = [n for n in names if n.startswith(f"{top}/Participant{p}_1/input/inputCondition")]
            assert len(ic) == 1
            grp = "obstacles" if ic[0].endswith("inputCondition1") else "irrelevant_items"
            assert (grp == "obstacles") == (p in OBSTACLE_GROUP), (p, grp)
        else:
            grp = "obstacles"
        sub = f"{ID}_{exp}_{p:02d}"
        vel = {c: [] for c in CONDITIONS}
        for t in range(1, 401):
            raw = z.read(f"{top}/Participant{p}_1/Participant{p}_{t}.txt").decode("latin-1")
            lines = [l for l in raw.replace("\r", "").split("\n") if l.strip() != ""]
            samples, events, cond, jump = [], [], None, None
            xs_after = []  # numeric x samples from the condition line on, with None at the target jump line (authors' ReadIn/ProcessData)
            seen_cond = False
            jump_idx = None
            info = []
            for l in lines:
                tok = l.split()
                if len(tok) == 4 and NUM.match(tok[0]) and all(NUM.match(a) or a == "m" for a in tok[1:]):
                    if tok[1] == "m":  # marker not recorded at this sample (the authors drop these lines): empty cells
                        samples.append([tok[0], "", "", ""])
                        n_missing[0] += 1
                    else:
                        samples.append(tok)
                        if seen_cond:
                            xs_after.append(float(tok[1]))
                    continue
                if "condition:" in l:
                    cond = l.split("condition:")[1].strip()
                    seen_cond = True
                    events.append(("trial_start", tok[0]))
                    continue
                name = None
                if len(tok) >= 2 and NUM.match(tok[0]):
                    rest = tok[1:]
                    if rest and NUM.match(rest[-1]):
                        val = rest[-1]
                        rest = rest[:-1]
                        info.append(("_".join(rest) + "_value", val))
                    name = "_".join(rest)
                if name is None:
                    n_unknown_lines[re.sub(r"-?[\d.]+", "#", l)[:40]] += 1
                    continue
                events.append((name, tok[0]))
                if name.startswith("target_jump"):
                    jump_idx = len(xs_after)
                    jump_counts[(exp, name, info[-1][1] if info and info[-1][0].startswith("target_jump") else "")] += 1
            if all(r[1] == "" for r in samples):
                n_empty_trials.append((exp, p, t))
            if cond is None:  # a trial whose marker was never recorded has no condition line: take the condition from the first line that names it
                pass
            assert cond in CONDITIONS, (exp, p, t, cond)
            x = np.array(xs_after)
            v = np.diff(x) / 0.002
            if jump_idx is None:
                no_jump.append((exp, p, t))
                vel[cond].append(np.full(500, np.nan))
            else:
                seg = v[jump_idx:jump_idx + 500]
                vel[cond].append(np.concatenate([seg, np.full(500 - len(seg), np.nan)]))
            # samples file
            fname = f"samples/{exp}/p{p:02d}/trial_{t:03d}.csv"
            lp = SAMPLES_DIR / f"{exp}_{p:02d}_{t:03d}.csv"
            with open(lp, "w", newline="") as fh:
                fh.write("time_s,x,y,z\r\n" + "".join(",".join(r) + "\r\n" for r in samples))
            extra.append((fname, str(lp)))
            n = len(samples)
            t0, t1 = float(samples[0][0]), float(samples[-1][0])
            tj = dict(info)
            tj_val = next((v for k, v in info if k.startswith("target_jump")), "")
            rows.append({
                **{c: "" for c in CORE_NAMES + MOD_NAMES},
                "dataset": ID, "study": ID, "experiment": f"{ID}_{exp}", "group": grp, "subject": sub, "person": sub,
                "subject_excluded_by_authors": "False", "handedness": "right" if exp == "main" else "", "population": "healthy",
                "setting": "lab", "device": "Optotrak 3020, infrared marker on the index finger nail",
                "session": "1", "block": "block1", "trial": str(t), "phase": "test", "task": "interception", "unit": "recording",
                "condition": cond, "pert_type": "target_jump", "pert_value": tj_val, "pert_unit": "cm (jump line of the deposit)",
                "feedback": "normal", "excluded_by_authors": "False",
                "recording": str(t), "samples_file": fname, "signal_kind": "index finger position (Optotrak)",
                "channels": "x;y;z", "channel_units": "mm;mm;mm", "sample_rate_hz": "500", "n_samples": str(n),
                "duration_s": "%.3f" % (t1 - t0 + 0.002),
                "events": ";".join(f"{a}={b}" for a, b in events), "processing": "none (raw positions as shared)",
                "trial_info": ";".join(f"{k}={v}" for k, v in info),
            })
        resp[(exp, p)] = (grp, vel)
        cond_counts[(exp, p)] = Counter(r["condition"] for r in rows if r["subject"] == sub)

checks.internal("every line of the trial files is a sample, a condition line or a recognised event line", not n_unknown_lines, str(dict(n_unknown_lines)))
checks.internal("trials without a target-jump line (kept; the response analysis treats them as missing)", True, f"{len(no_jump)} trials: {Counter((e, p) for e, p, t in no_jump)}")
checks.internal("each participant has 50 trials in each of the 8 conditions (400 in all)",
                all(set(c.values()) == {50} and len(c) == 8 for c in cond_counts.values()))
checks.internal("rows = (24 + 12) participants x 400 trials", len(rows) == 36 * 400, f"{len(rows)} rows")

# ------------------------------------------------------------------ published numbers
main_ps = [k for k in resp if k[0] == "main"]
checks.published("participants, main experiment", len(main_ps), 24, 0, "Methods, Participants (24 participants, two groups of 12)")
checks.published("participants, obstacles group", sum(resp[k][0] == "obstacles" for k in main_ps), 12, 0, "Methods, Participants (n = 12)")
checks.published("participants, irrelevant-items group", sum(resp[k][0] == "irrelevant_items" for k in main_ps), 12, 0, "Methods, Participants (n = 12)")
checks.published("participants, replication experiment", sum(k[0] == "replication" for k in resp), 12, 0, "Methods, Participants (12 new participants)")
checks.published("trials per participant", st.mean(sum(c.values()) for c in cond_counts.values()), 400, 0, "Methods, Design and trials (400 trials)")
checks.published("trials per gap-change x timing cell", st.mean(sum(n for cname, n in c.items() if cname.split()[1] == g and cname.split()[-1] == tm)
                                                        for c in cond_counts.values() for g in ("inward", "outward") for tm in ("early", "late")),
                 100, 0, "Methods, Design and trials (100 trials per condition)")
checks.published("trials per cell with target jump direction", st.mean(n for c in cond_counts.values() for n in c.values()), 50, 0,
                 "Methods, Design and trials (50 trials each per condition)")


def modulation(vel, lo, hi, timing_offset):
    """Authors' response modulation: (outward right - left) minus (inward right - left), velocity samples lo..hi (1-based, 2 ms each)."""
    m = {c: np.nanmean(np.array(vel[c]), axis=0) for c in CONDITIONS}
    out = [m[CONDITIONS[timing_offset + 0]] - m[CONDITIONS[timing_offset + 1]]]  # outward
    inn = [m[CONDITIONS[timing_offset + 2]] - m[CONDITIONS[timing_offset + 3]]]  # inward
    d = out[0] - inn[0]
    return float(np.nanmean(d[lo - 1:hi]))


def tstat1(a):
    a = np.asarray(a)
    return a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))


def tstat2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / (sp * math.sqrt(1 / len(a) + 1 / len(b)))


obs = [modulation(resp[k][1], 75, 100, 4) for k in main_ps if resp[k][0] == "obstacles"]
irr = [modulation(resp[k][1], 75, 100, 4) for k in main_ps if resp[k][0] == "irrelevant_items"]
differences = []
for what, comp, rep, tol, where in (
        ("obstacles group, late condition: t(11) of the response modulation (150-200 ms after the jump) against 0", tstat1(obs), 4.00, 0.005,
         "Results, Main experiment"),
        ("late condition: t(22), obstacles minus irrelevant-items group (response modulation)", tstat2(obs, irr), 2.36, 0.005,
         "Results, Main experiment"),
        ("obstacles group, late condition: t(11) with the 100-200 ms window", tstat1([modulation(resp[k][1], 50, 100, 4) for k in main_ps if resp[k][0] == "obstacles"]),
         4.40, 0.005, "Results, Robustness check (footnote 1)")):
    if abs(comp - rep) <= tol:
        checks.published(what, comp, rep, tol, where)
    else:
        differences.append(f"{what}: the paper prints {rep}, the rows give {comp:.3f}.")
        print("NOT REPRODUCED:", what, comp, rep)
rep_obs = [modulation(resp[k][1], 75, 100, 4) for k in resp if k[0] == "replication"]
t_rep = tstat1(rep_obs)
print("replication t(11) with the main-experiment pipeline:", t_rep, "(paper 2.60)")
if abs(t_rep - 2.60) <= 0.005:
    checks.published("replication experiment, late condition: t(11) of the response modulation (150-200 ms)", t_rep, 2.60, 0.005, "Results, Replication")
else:
    differences.append(f"replication, late condition: t(11) of the response modulation: the paper prints 2.60, the main-experiment pipeline (the replication script was not read) gives {t_rep:.3f}.")
differences.append("The deposit's jump lines read 'target jump x +-0.5' (cm); the paper says the target jumped 1 cm. pert_value holds the deposit's number.")
differences.append("Coordinate axes and units of the Optotrak positions are not stated in the paper or deposit; mm follows from the 350 units the finger "
                   "travels from the start disc to the hitting zone (35 cm apart). Lateral motion is x.")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Brenner & Smeets (VU Amsterdam)",
    data_url="https://osf.io/4fjru/",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, each trial's sample lines moved to a CSV and its event lines to the events column.",
    unit_note="trial (recording) of one participant",
    how=[
        "One row is one trial of 400 (8 conditions x 50: gap between the red items wider or narrower, target jump to the right or left, gap change early "
        "at target onset or late at the target jump), recorded as the position of an infrared marker on the index finger at 500 Hz; the samples "
        "(time_s, x, y, z) are in samples/<experiment>/<participant>/trial_NNN.csv exactly as in the authors' text file.",
        "The main experiment has two groups (obstacles: touching a red item cost a point; irrelevant items: told to ignore them), the replication "
        "only the obstacles instructions; group comes from the input file stored with each participant and agrees with the participant lists in the "
        "authors' analysis script. time_s is the deposit's time stamp, 0 at the condition line (target onset); samples before it are kept.",
        "events lists the event lines of the trial (condition start, obstacle and target jumps, the time the target and any red item were hit) with their times "
        "in seconds; trial_info holds the numbers on those lines (jump size). The condition name is in condition. Targets and red-item positions "
        "are in the input files, which are not converted.",
    ],
    differences=differences,
)
stdlib.release(ID, "timeseries", rows, meta, checks, extra_files=extra)
