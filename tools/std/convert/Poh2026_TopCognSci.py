"""Poh, Al-Fawakhiri, Tam, McDougle & Taylor (2026), Topics in Cognitive Science: generalization of visuomotor rotation learning
over target direction and over abstract target shape (5 experiments, 96 participants), KINARM End-Point robot, one row per reach in
the 90-column reaching format.

Run from the site folder on the PC: python3 -I tools/std/convert/Poh2026_TopCognSci.py
Source: OSF https://osf.io/s5tvc/ (CC-By Attribution 4.0 as stated on the node): 288 zips, one per experiment, participant and session
(Pre, Rot, Post), each holding one KINARM Dexterit-E c3d file per trial. Because of the size (9.5 GB) the converter processes ONE zip at
a time: download to raw_extra/<id>/, reduce every trial to a few numbers (cached in tmp_samples/<id>/), delete the zip.
The 16 xls files of Experiment 4 (similarity ratings of one group) are not converted. Nothing from the deposit is executed.
"""
import hashlib
import json
import math
import re
import shutil
import statistics as st
import sys
import time
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

HERE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_DIR))
sys.path.insert(0, str(HERE_DIR.parent))
import stdlib  # noqa: E402
import Poh2026_TopCognSci_c3d as C  # noqa: E402
from Poh2026_TopCognSci_files import FILES  # noqa: E402

ID = "Poh2026_TopCognSci"
RAW = stdlib.raw_dir(ID)
CACHE = stdlib.HERE / "tmp_samples" / ID
MIN_FREE = 3e9
checks = stdlib.Checks()


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


def tf(x):
    return "True" if x else "False"


def wrap(a):
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


# ---------------------------------------------------------------- stage 1: one zip at a time
def download(url, dest, size):
    RAW.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        while shutil.disk_usage(RAW).free < MIN_FREE + size:
            print("[wait] PC disk low", flush=True)
            time.sleep(300)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OpenMotor-curation/1.0 (research data index)"})
            h = hashlib.sha256()
            n = 0
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
                    h.update(b)
                    n += len(b)
            if n != size:
                raise IOError(f"size {n} != {size}")
            return h.hexdigest()
        except Exception as e:  # noqa: BLE001
            print(f"[retry {attempt}] {dest.name}: {e}", flush=True)
            time.sleep(30 * (attempt + 1))
    raise SystemExit(f"download failed: {url}")


NAME = re.compile(r"^(?P<pre>[A-Za-z]+?)(?P<ses>Pre|Rot|Post)(?P<n>\d+)\.zip$", re.I)
zips = [(e, f, s, u) for e, f, s, u in FILES if f.lower().endswith(".zip")]
CACHE.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
src_path = RAW / "SOURCES.json"
sources = {x["file"]: x for x in json.load(open(src_path))} if src_path.exists() else {}
import zipfile  # noqa: E402

t0 = time.time()
for i, (exp, fn, size, url) in enumerate(zips):
    cache = CACHE / (f"{exp}_{fn}.json")
    if cache.exists():
        continue
    dest = RAW / f"{exp}_{fn}"
    digest = download(url, dest, size)
    trials = []
    with zipfile.ZipFile(dest) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".c3d") and not n.endswith("common.c3d"))
        for n in names:
            r = C.reduce_trial(z.read(n))
            r["file"] = n.split("/")[-1]
            trials.append(r)
    json.dump(trials, open(cache, "w"))
    sources[dest.name] = dict(file=dest.name, url=url, bytes=size, sha256=digest, fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(list(sources.values()), open(src_path, "w"), indent=1)
    dest.unlink()
    print(f"[{i + 1}/{len(zips)}] {exp} {fn}: {len(trials)} trials, {time.time() - t0:.0f} s", flush=True)

# ---------------------------------------------------------------- stage 2: tables
SESSIONS = ["Pre", "Rot", "Post"]
persons = OrderedDict()  # (exp, group, n) -> {session: trials}
for exp, fn, size, url in zips:
    m = NAME.match(fn)
    ses = m["ses"].capitalize()
    grp = m["pre"] if exp == "E4" else ""
    key = (exp, grp, int(m["n"]))
    persons.setdefault(key, {})[ses] = json.load(open(CACHE / f"{exp}_{fn}.json"))
persons = OrderedDict(sorted(persons.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])))
checks.internal("every participant has Pre, Rot and Post sessions", all(set(v) == set(SESSIONS) for v in persons.values()))
EXP_N = Counter(k[0] for k in persons)
for e, n in (("E1", 16), ("E2", 16), ("E3", 16), ("E4", 32), ("E5", 16)):
    checks.published(f"Experiment {e[1]}: participants", EXP_N[e], n, 0,
                     "Methods, 'Design of Experiment %s' (n=%d)" % (e[1], n))
checks.published("participants in all five experiments", len(persons), 96, 0, "Methods, Participants (96 participants)")

FB = {1.0: "cursor", 2.0: "none", 4.0: "clamp"}
names = [c[1] for c in stdlib.reach90_columns()]
rows = []
by_person = OrderedDict()
bad_fb = Counter()
for pi, (key, ses_d) in enumerate(persons.items(), start=1):
    exp, grp, n = key
    tp_all = []
    seq = []
    for si, ses in enumerate(SESSIONS):
        for r in sorted(ses_d[ses], key=lambda r: (r["tp"], r["file"])):
            seq.append((si, ses, r))
    tps = [r["tp"] for _, _, r in seq]
    checks.internal(f"{exp} {grp} {n}: trial numbers strictly increase across Pre, Rot, Post", all(b > a for a, b in zip(tps, tps[1:])))
    pid = f"{ID}_{exp}_{grp}{n}"
    by_person[pid] = []
    for j, (si, ses, r) in enumerate(seq):
        fbt = FB[r["feedback_type"]]
        rot = r["rotation"]
        ta = r["target_angle"] % 360.0
        ha = None if r["hand_deg"] is None else wrap(r["hand_deg"] - ta)
        cursor = fbt in ("cursor", "clamp")
        cerr = wrap(r["cursor_deg"] - ta) if cursor and r["cursor_deg"] is not None else None
        if ses == "Pre":
            phase = "baseline"
        elif fbt == "none":
            phase = "probe"
        elif rot != 0:
            phase = "learning"
        else:
            phase = "other"
        row = dict.fromkeys(names, "")
        row.update(
            dataset=ID, experiment=f"{ID}_Exp{exp[1]}", study=ID, subject=pid, person=pid, id_type="lab", person_link="assumed_unique",
            group=grp, prior_exposure="unknown", subject_excluded_by_authors="False", handedness="right", device="kinarm",
            setting="lab", population="healthy",
            trial=str(j), trial_orig=str(r["tp"]), session="0", block=ses, phase=phase, trial_kind="reach",
            break_before=tf(j == 0 or seq[j - 1][1] != ses),
            target_angle=num(ta), target_id=str(int(round(ta * 2))), target_distance_norm="1", target_visible_during_move="True",
            target_event="none", target_jump="0", probe_kind="none",
            instr_goal="hand_to_target" if fbt == "clamp" else "cursor_to_target",
            instr_strategy="forbidden" if fbt == "clamp" else "unspecified", instr_disclosure="exists" if fbt == "clamp" else "",
            instr_report="none", cue_elements=(f"target_shape:{int(r['shape'])}" if r["shape"] > 0 else ""), fb_cued="True",
            hand_used="right", hand_angle=num(ha), hand_measure="fixed_distance", rt_ms=num(r["rt"]), mt_ms=num(r["mt"]),
            excluded_by_authors=tf(r["is_error"] == 1),
            speed_warning="1" if (r["too_slow"] > 0 or r["too_fast"] > 0) else "0", other_warning="1" if r["too_slow_rt"] > 0 else "0",
            fb_visible="online" if cursor else "none",
            fb_mode=("clamped" if fbt == "clamp" else ("veridical" if rot == 0 else "rotated")) if cursor else "none",
            fb_form="cursor" if cursor else "none",
            online_cursor=("perturbed" if (fbt == "clamp" or rot != 0) else "veridical") if cursor else "none",
            fb_endpoint_transform_only="False",
            rotation=num(rot) if fbt == "cursor" else "", clamp=num(rot) if fbt == "clamp" else "",
            cursor_error=num(cerr), fb_delay_ms="0" if cursor else "", fb_duration_ms="1000" if cursor else "",
            outcome_channels="cursor" if cursor else "",
        )
        if cursor and cerr is not None:
            row["hand_angle_fb"] = ""
        rows.append(row)
        by_person[pid].append((ses, r, ha, cerr, fbt))

# ---------------------------------------------------------------- internal checks (sign and structure)
n_trials = len(rows)
checks.internal("every trial file kept: rows equal the number of c3d trial files", n_trials == sum(len(v) for p in persons.values() for v in p.values()))
no_ha = sum(1 for r in rows if r["hand_angle"] == "")
checks.internal("hand direction found on at least 99% of trials", no_ha / n_trials < 0.01, f"{no_ha} of {n_trials} trials without one")
diffs = []
for pid, lst in by_person.items():
    for ses, r, ha, cerr, fbt in lst:
        if fbt == "cursor" and ha is not None and cerr is not None and r["rotation"] != 0:
            diffs.append(abs(wrap(cerr - ha - r["rotation"])))
diffs.sort()
checks.internal("sign: on rotated-cursor trials the cursor lies at hand direction + rotation (median |difference| < 5 deg)",
                diffs and diffs[len(diffs) // 2] < 5, f"median {diffs[len(diffs) // 2]:.2f} deg over {len(diffs)} trials")
dc = sorted(abs(wrap(cerr - r["rotation"])) for lst in by_person.values() for ses, r, ha, cerr, fbt in lst if fbt == "clamp" and cerr is not None)
checks.internal("sign: on clamp trials the cursor is exactly at target + clamp (counter-clockwise positive)", dc and dc[-1] < 1e-3,
                f"max |difference| {dc[-1]:.2e} deg over {len(dc)} trials")
late = defaultdict(list)
for pid, lst in by_person.items():
    for ses, r, ha, cerr, fbt in lst:
        if ses == "Rot" and fbt in ("cursor", "clamp") and r["rotation"] != 0 and ha is not None:
            late[pid].append((r["rotation"], ha, fbt))
opp = {pid: (st.mean(h for _, h, _ in v[-10:]) < 0) == (v[-1][0] > 0) for pid, v in late.items() if len(v) >= 10}
rot_ok = [ok for pid, ok in opp.items() if late[pid][-1][2] == "cursor"]
clamp_ok = [ok for pid, ok in opp.items() if late[pid][-1][2] == "clamp"]
checks.internal("compensation sign: late in learning the hand direction opposes the rotation for at least 95% of rotation participants "
                "and 80% of clamp participants", sum(rot_ok) / len(rot_ok) >= 0.95 and sum(clamp_ok) / len(clamp_ok) >= 0.8,
                f"{sum(rot_ok)} of {len(rot_ok)} rotation and {sum(clamp_ok)} of {len(clamp_ok)} clamp participants")

# ---------------------------------------------------------------- published numbers
def asymptote(exp_key):
    vals = []
    for pid, lst in by_person.items():
        if not pid.startswith(f"{ID}_{exp_key}_"):
            continue
        rot_tr = [x for x in lst if x[0] == "Rot" and x[4] in ("cursor", "clamp") and x[1]["rotation"] != 0][:80]
        anchor = (rot_tr[0][1]["target_angle"], rot_tr[0][1]["shape"])
        base = [x[2] for x in lst if x[0] == "Pre" and x[2] is not None and (x[1]["target_angle"], x[1]["shape"]) == anchor]
        b = st.mean(base) if base else 0.0
        s = 1.0 if rot_tr[0][1]["rotation"] > 0 else -1.0
        vals.append(st.mean(-s * wrap(x[2] - b) for x in rot_tr[-10:] if x[2] is not None))
    return st.mean(vals), len(vals)


ASY = {}
for e in ("E1", "E2", "E3", "E4", "E5"):
    ASY[e] = asymptote(e)
    print("asymptote", e, ASY[e], flush=True)
json.dump(ASY, open(CACHE / "asymptotes.json", "w"))
PAPER_ASY = {"E1": 42.85, "E2": 42.38, "E3": 43.12, "E5": 8.68}
checks.internal("adaptation asymptote has the expected size and sign: 40-45 deg for the 45 deg rotations, 5-12 deg for the 20 deg clamp",
                all(40 < ASY[e][0] < 45 for e in ("E1", "E2", "E3", "E4")) and 5 < ASY["E5"][0] < 12,
                ", ".join(f"{e} {ASY[e][0]:.2f}" for e in ASY))

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Taylor (Princeton) and McDougle (Yale)",
    data_url="https://osf.io/s5tvc/", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format from the KINARM c3d trial files, participant codes replaced, hand direction computed "
            "from the recorded hand path.",
    unit_note="reach (one KINARM c3d trial)",
    how=[
        "One row is one reach: Pre (match-to-sample rating phase), Rot (rotation or clamp training, then in Experiments 1-3 the "
        "generalization trials) and Post sessions of 96 participants in five experiments (16 per experiment, two groups of 16 in "
        "Experiment 4); every trial file the authors shared is kept, in order (trial_orig is the task program's trial number).",
        "hand_angle is the direction of the hand path from the hand at movement onset (smoothed speed above 5 cm/s after the target "
        "appeared) to where the hand first reached the task program's 10 cm radius from the start, minus the target direction, "
        "counter-clockwise positive (the paper describes 9 cm; the logged radius is 0.1 m). The sign was checked against the logged "
        "cursor: with a +45 deg rotation the cursor ended at hand direction + 45 deg, and hand directions late in learning opposed the "
        "rotation or clamp.",
        "Pairs of trials (an anchor trial with cursor feedback, then a probe trial) follow the paper; probe trials have no cursor "
        "(fb_visible none, phase probe). rotation is the cursor rotation, clamp the clamped cursor offset (Experiment 5, 20 deg); "
        "cursor_error is the logged final cursor direction minus the target. cue_elements target_shape:N is the shape index the task "
        "program logged (1-24); the index-to-roundness mapping is inferred from the paper, not documented in the deposit.",
        "Post sessions: the deposit does not describe them. In Experiments 1-3 they hold veridical-cursor pair trials after learning "
        "(phase other), in Experiments 4 and 5 the generalization probes. The two Experiment 4 groups are labelled with the authors' "
        "file names (DOnly, SquircleDirection); which is the Direction or Shape Emphasis group is not stated in the deposit. "
        "Similarity ratings (spoken and typed by the experimenter) are not in the c3d files and are not included.",
    ],
    differences=[
        "The adaptation asymptotes (mean of the last 10 learning trials, baseline corrected, compensation positive) in the preprint of this "
        "paper are 42.85, 42.38, 43.12 (Experiments 1-3) and 8.68 (Experiment 5); recomputed here from the hand directions they are "
        + ", ".join(f"{ASY[e][0]:.2f}" for e in ("E1", "E2", "E3", "E5"))
        + ". They sit 0.4 to 0.7 deg lower, so they are not listed as reproduced. Likely causes: the preprint defines the endpoint at 9 cm "
        "and the task program logged a 10 cm radius (computing Experiment 1 at 9 cm gives 42.67), and the authors' outlier removal and exact "
        "baseline are not specified in enough detail to repeat. Only the participant counts are listed as reproduced.",
        "The checks use the 2021 bioRxiv preprint of the paper (the journal text could not be read), so numbers may differ from the final article.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
