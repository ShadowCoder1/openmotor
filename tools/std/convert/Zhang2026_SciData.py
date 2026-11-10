"""Zhang et al. (2026), Sci Data: sEMG dataset comparing anatomical and equidistant (random) electrode placement for hand gesture
recognition. Only the expert gesture annotations (label.csv, 621 files) are converted; the 37 GB of EMG and IMU signals are not.

Run from the site folder: python3 -I tools/std/convert/Zhang2026_SciData.py
Raw files (untrusted data, read only) in raw_extra/Zhang2026_SciData/, from https://doi.org/10.34894/QRGIZQ (DataverseNL):
  README.txt, participants.csv        deposit documentation and the participant table (semicolon separated, decimal comma)
  labels/<participant>__<mode>__sequence_<NN>.csv   the deposit's <participant>/<mode>/sequence_<NN>/label.csv, one per recording
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zhang2026_SciData"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

NAMES = {1: "relax", 2: "wave in", 3: "make a fist", 4: "wave out", 5: "spread fingers", 6: "stretch out the index finger",
         7: "victory sign", 8: "pistol gesture", 9: "four", 10: "thumb up", 11: "take the mug", 12: "pick up the card",
         13: "take the paper cup", 14: "writing posture"}
GROUP = {**{k: "hand movement" for k in range(1, 6)}, **{k: "finger gesture" for k in range(6, 11)},
         **{k: "functional grip" for k in range(11, 15)}}

# ---- participant table
with open(RAW / "participants.csv", encoding="utf-8-sig", newline="") as f:
    rd = list(csv.reader(f, delimiter=";"))
hdr = [h.strip() for h in rd[0]]
ptab = {}
for r in rd[1:]:
    d = dict(zip(hdr, r))
    pid = d["ID"].strip()
    age = d["Age"].strip()
    hand = d["dominant hand"].strip().lower()
    ptab[pid] = dict(
        age=int(age) if age.isdigit() else None,
        sex={"male": "M", "female": "F"}[d["Gender"].strip().lower()],
        handedness={"right": "right", "rigtht": "right", "left": "left"}[hand],
        n_rand=int(d["Random sequence"]) if d["Random sequence"].strip().isdigit() else None,
        n_anat=int(d["Anatomical sequence"]) if d["Anatomical sequence"].strip().isdigit() else None)
checks.internal("participant table has 64 rows", len(ptab) == 64, str(len(ptab)))
ALIAS = {"HG_M7873Q0": "HG_M7873Q"}   # the one folder name that differs from the participants table by a trailing 0

# ---- label files
fn_re = re.compile(r"(HG_\w+?)__(anatomical|random)__sequence_(\d+)\.csv")
files = []
for p in sorted((RAW / "labels").glob("*.csv")):
    m = fn_re.fullmatch(p.name)
    assert m, p.name
    files.append((m.group(1), m.group(2), int(m.group(3)), p))
checks.internal("621 label files, one per recording folder of the deposit", len(files) == 621, str(len(files)))

rows = []
trial = {}
n_us_files = n_empty = n_double = n_swapped = n_stop_before_start = n_missing_gesture_recs = 0
dup_files = []
recs = {}   # (pid, mode) -> number of recordings (parts) with data
order = {"anatomical": 0, "random": 1}
files.sort(key=lambda t: (t[0], order[t[1]], t[2]))
for pid, mode, seq, p in files:
    with open(p, encoding="utf-8-sig", newline="") as f:
        rr = list(csv.reader(f))
    assert rr[0] == ["ID", "sequence", "timestamp_start", "timestamp_stop", "gesture_code"], rr[0]
    body = rr[1:]
    if not body:
        n_empty += 1
        continue
    assert all(len(r) == 5 and r[0] == pid and int(r[1]) == seq for r in body), p.name
    # a file can hold two recordings one after the other: each starts with the 'preparation' segment
    cuts = [i for i, r in enumerate(body) if r[4] == "preparation"]
    starts = cuts if cuts and cuts[0] == 0 else [0] + cuts
    parts = [body[a:b] for a, b in zip(starts, starts[1:] + [len(body)])]
    if len(parts) > 1:
        n_double += 1
    nums = []
    for r in body:
        for j in (2, 3):
            try:
                nums.append(float(r[j]))
            except ValueError:
                assert r[j] in ("end", "finished"), (p.name, r)
    micro = max(nums) > 1000
    assert micro == (min(nums) > 1000) or min(nums) > 1000 or not micro, p.name   # whole file in one unit
    assert not micro or min(nums) > 1000, ("file mixes units", p.name)
    n_us_files += micro
    div = 1e6 if micro else 1.0
    if len(parts) == 2:
        dup_files.append(p.name)
        assert [r[2:] for r in parts[0]] == [r[2:] for r in parts[1]], ("two recordings that differ", p.name)
    for k, part in enumerate(parts, 1):
        rec = f"{mode}_seq{seq:02d}" + (f"_part{k}" if len(parts) > 1 else "")
        if k == 1:
            recs[(pid, mode)] = recs.get((pid, mode), 0) + 1
        seen = {}
        ng = 0
        for si, r in enumerate(part, 1):
            code = r[4].strip()
            start = float(r[2]) / div
            stop_raw = r[3].strip()
            swapped = code == "end" and stop_raw == "finished"
            if swapped:
                code, stop_raw = "finished", "end"
                n_swapped += 1
            stop = float(stop_raw) / div if stop_raw not in ("end",) else None
            if code == "preparation":
                stype, gname, ggroup, rep = "preparation", "", "other", None
            elif code == "T":
                stype, gname, ggroup, rep = "transition", "", "other", None
            elif code == "finished":
                stype, gname, ggroup, rep = "finished", "", "other", None
            else:
                g = int(code)
                assert 1 <= g <= 14, (p.name, r)
                stype, gname, ggroup = "gesture", NAMES[g], GROUP[g]
                seen[g] = seen.get(g, 0) + 1
                rep = seen[g]
                ng += 1
            if stop is not None and stop < start:
                n_stop_before_start += 1
            trial[pid] = trial.get(pid, 0) + 1
            pi = ptab[ALIAS.get(pid, pid)]
            rows.append(dict(
                dataset=ID, study="Zhang et al. 2026 Sci Data", experiment=ID + "_Exp1", group="", subject=pid, person=pid,
                subject_excluded_by_authors=False, age=pi["age"], sex=pi["sex"], handedness=pi["handedness"],
                population="healthy", setting="lab",
                device="Delsys Trigno Avanti and Quattro sEMG/IMU sensors on the non-dominant forearm (signals not included here); Azure Kinect video used for scoring",
                session=1, block=f"{mode} placement, sequence {seq:02d}" + (f" (recording {k} of {len(parts)} in the file)" if len(parts) > 1 else ""),
                trial=trial[pid], phase="test", time_s=None, task="hand gestures and grasps", unit="gesture segment",
                condition=f"{mode} electrode placement", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                excluded_by_authors=False,
                recording=rec, electrode_placement=mode, sequence_order=seq, segment_index=si, segment_type=stype,
                gesture_code=code, gesture_name=gname, gesture_group=ggroup, repetition=rep, start_s=start, stop_s=stop,
                duration_s=(stop - start) if stop is not None else None))
        if ng < 28:
            n_missing_gesture_recs += 1

# ---- internal checks
import collections
rows_by_p = collections.Counter(r["person"] for r in rows)
checks.internal("every participant with a label folder has rows; 64 participants", len(rows_by_p) == 64, str(len(rows_by_p)))
checks.internal("6 label files are empty (header only) and not converted", n_empty == 6, str(n_empty))
checks.internal("all gesture codes 1-14 appear", {r["gesture_code"] for r in rows if r["segment_type"] == "gesture"} == {str(i) for i in range(1, 15)})
checks.internal("the 5 files that hold two recordings hold the same annotation twice (identical times and codes)", len(dup_files) == 5, str(len(dup_files)))
checks.internal("scored gestures last under 65 s (9 of them over 12 s, longest 61 s)",
                all(r["duration_s"] < 65 for r in rows if r["segment_type"] == "gesture" and r["duration_s"] is not None))
checks.internal("every row of every non-empty label file kept", len(rows) == 35872, str(len(rows)))

# ---- published numbers (deposit documentation: README.txt and participants.csv; the article text was not readable)
ng_rows = [r for r in rows if r["segment_type"] == "gesture"]
checks.published("distinct gestures", len({r["gesture_code"] for r in ng_rows}), 14, 0, "README.txt, gesture_code list; dataset description (14 distinct hand gestures)")
checks.published("participants with labelled recordings (all 64 participants of participants.csv)", len(rows_by_p), 64,
                 0, "deposit folder structure (one folder per participant) and participants.csv")
got_full = sum(1 for pid in rows_by_p if recs.get((pid, "anatomical"), 0) == 5 and recs.get((pid, "random"), 0) == 5)
tab_full = sum(1 for v in ptab.values() if v["n_anat"] == 5 and v["n_rand"] == 5)
checks.published("participants with five anatomical and five random recordings (participants.csv: 'Anatomical sequence' and 'Random sequence' both 5; "
                 "three of them have one recording fewer in the deposit)", got_full, tab_full, 3, "participants.csv")
# each recording has all 14 gestures twice -> ten repetitions per mode over five recordings
per = collections.Counter((r["person"], r["electrode_placement"], r["gesture_code"]) for r in ng_rows)
full = [pid for pid in rows_by_p if ptab[ALIAS.get(pid, pid)]["n_anat"] == 5 and ptab[ALIAS.get(pid, pid)]["n_rand"] == 5]
reps = [per[(pid, m, str(g))] for pid in full for m in ("anatomical", "random") for g in range(1, 15)]
checks.published("median scored repetitions of a gesture in one placement mode of a participant with 5+5 recordings",
                 float(np.median(reps)), 10, 0, "README.txt / dataset description: 14 distinct gestures performed ten times per mode")
print("participants with 5+5 recordings:", len(full), "share of gesture-mode cells with exactly 10 scored repetitions:",
      round(float(np.mean(np.array(reps) == 10)), 3))
print("files in microseconds:", n_us_files, "files with two recordings:", n_double, "swapped end rows:", n_swapped,
      "stop before start:", n_stop_before_start, "recordings with <28 scored gestures:", n_missing_gesture_recs)
for pid in sorted(rows_by_p):
    a = recs.get((pid, "anatomical"), 0); b = recs.get((pid, "random"), 0); t = ptab[ALIAS.get(pid, pid)]
    if (a, b) != (t["n_anat"], t["n_rand"]):
        print("MISMATCH", pid[-3:], "got", a, b, "table", t["n_anat"], t["n_rand"], "empty files:", sum(1 for q in files if q[0] == pid and q[3].stat().st_size < 60))
long_g = [(r["recording"], r["gesture_code"], r["duration_s"]) for r in ng_rows if r["duration_s"] is not None and r["duration_s"] > 12]
print("long gestures", long_g[:10], len(long_g))
print("full", got_full, tab_full)
for ch in checks.items:
    print("  ", ch["kind"], ch["ok"], ch["what"], ch.get("computed", ""), ch.get("reported", ""))

diffs = [
    f"{n_us_files} of the 615 non-empty label files hold their segment times as integers in microseconds (values up to about 150,000,000, "
    "for example 6068511 for 6.07 s); the README says seconds. start_s and stop_s here are in seconds: those files were divided by 1,000,000. "
    "The other files already give seconds with two decimals.",
    f"{n_double} label files (all anatomical recordings of one participant) hold the same annotation twice in a row (each copy starts with a preparation segment, "
    "times and codes identical); both copies are kept, as recording ..._part1 and ..._part2, so that no row of the files is lost: "
    "drop the part2 recordings to count each recording once. Six label files are empty (header only) and give no rows: the five "
    "anatomical recordings of one participant (the participants table notes that the label file is empty) and one random recording of another.",
    f"The README says each recording has all 14 gestures twice, 28 scored gestures; {n_missing_gesture_recs} recordings have fewer scored gestures "
    "(the authors did not mark every gesture), and 9 scored gestures last more than 12 s (longest 61 s, probably a missed transition). Segments are kept exactly as scored, including "
    f"{n_stop_before_start} segments whose stop time is earlier than the start time. The final segment's stop is the string 'end' in the files "
    "and empty here; in one file the 'end' and 'finished' labels are in swapped columns, which was put right.",
    "participants.csv lists no recording counts for two participants, but the deposit has five anatomical recordings for each; for three participants whose "
    "table says five random (or anatomical) recordings, the deposit has one fewer label file or an empty one.",
    "The Sci Data article text could not be read (the publisher page was not reachable), so the numbers checked are those of the deposit's own README and "
    "participants table, not figures from the article.",
]
meta = dict(
    study="Zhang et al. 2026",
    paper=dict(authors="Zhang Y, Garcia Arias LF, Wang X, Pruijm J, Roossien CC, Cao M, Wilhelm E",
               title="A surface electromyography dataset comparing anatomical and equidistant electrode placement for hand gesture recognition",
               venue="Scientific Data", year=2026, doi="10.1038/s41597-026-08111-4"),
    lab="Wilhelm (University of Groningen)", data_url="https://doi.org/10.34894/QRGIZQ",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format from the expert gesture annotation files (label.csv) only. The surface EMG and IMU signal files "
             "(sensor_data.csv, 621 files, 37 GB) and the participants' weight and height are not included. Participant identifiers were replaced by codes. "
             "Segment times written in microseconds in 299 files were converted to seconds."),
    how=[
        "One row is one scored segment of one recording: a performed gesture, the preparation before it, a transition between gestures, or the end of the "
        "recording. In each recording the participant, wearing sEMG and IMU sensors on the non-dominant forearm, performed 14 cued hand gestures twice "
        "(5 common hand movements, 5 finger gestures, 4 functional grips) in one of 10 randomized orders (sequence_order). A participant made up to five recordings with "
        "the sensors over eight named forearm muscles (anatomical) and up to five with the sensors spaced around the forearm (random).",
        "start_s and stop_s are seconds from the start of the Azure Kinect video of the recording, as scored by expert raters. gesture_code is the authors' code "
        "(1-14 gestures, T transition, preparation, finished). repetition counts the repeats of a gesture within the recording. The recording order within a participant "
        "is not known from the deposit: trial numbers the rows by placement mode (anatomical first, as in the protocol) and then by sequence number.",
        "Consent: the deposit states that all participants gave written informed consent and that the Medical Ethics Review Board of the University Medical Center Groningen "
        "issued a waiver (M22.307087); the deposit records no per-participant limits on use.",
    ],
    unit_note="scored gesture segment of one recording",
    differences=diffs,
)
stdlib.release(ID, "gesture_labels", rows, meta, checks)
