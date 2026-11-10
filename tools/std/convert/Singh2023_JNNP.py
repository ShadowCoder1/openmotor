"""Singh et al. (2023), J Neurol Neurosurg Psychiatry: interval-timing task in Parkinson's disease, one row per trial.

Run from the site folder: python3 -I tools/std/convert/Singh2023_JNNP.py
Raw files (untrusted data, read only) in raw_extra/Singh2023_JNNP/, from https://openneuro.org/datasets/ds004579 (CC0):
  participants.tsv                          group, age, sex, MoCA and UPDRS per participant (the ID and EEG-file columns,
                                            which hold the study's own participant numbers, are not read)
  sub-NNN_task-IntervalTiming_events.tsv    the EEG event markers of each participant (onset in s from the recording start)
Marker values, from task-IntervalTiming_events.json: S1 short-interval cue (text "Short interval", 3-s trials), S2 long-interval
cue (7-s trials), S3 solid-box go cue (the interval starts), S4 spacebar press, S5 spacebar release, S6 distractor vowel,
S7 feedback. Two participants (and the first part of a third) carry the same markers as ASCII codes of the digits
(S49 = '1' ... S55 = '7'): the sequence cue, go, distractors, press, release, feedback is identical, so they are read as S1-S7.
The deposit's R3 events repeat S events at the same latency and are ignored.
"""
import csv
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Singh2023_JNNP"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
SRC = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}


def tsv(name):
    with open(RAW / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def code_of(value):
    """S1..S7 (also as ASCII digits S49..S55) -> 1..7; anything else -> None."""
    m = re.fullmatch(r"S\s*(\d+)", value.strip())
    if not m:
        return None
    c = int(m.group(1))
    c = c - 48 if 49 <= c <= 55 else c
    return c if 1 <= c <= 7 else None


def ms(x):
    return round(x * 1000, 3)


# ---------------------------------------------------------------- participants
parts = tsv("participants.tsv")
checks.internal("participants.tsv matches the downloaded hash", stdlib.sha256(RAW / "participants.tsv") == SRC["participants.tsv"]["sha256"])
checks.internal("139 participants: 94 with Parkinson's disease and 45 controls (README)",
                len(parts) == 139 and Counter(p["GROUP"] for p in parts) == {"PD": 94, "Control": 45}, str(Counter(p["GROUP"] for p in parts)))


def num(x):
    x = (x or "").strip()
    return None if x in ("", "n/a", "NA") else float(x)


# ---------------------------------------------------------------- trials
NAMES = [c[1] for c in stdlib.columns_for("tapping") if c[1] != "person"]
rows = []
n_hash_ok = 0
n_cues = n_nogo = n_nopress = n_prebefore = n_multi = n_ascii_trials = n_extra_go = 0
unmapped = Counter()
lens_short, lens_long = [], []
by_part_target = {}
ascii_rt = {}
for p in parts:
    sid = p["participant_id"]
    fn = f"{sid}_task-IntervalTiming_events.tsv"
    n_hash_ok += stdlib.sha256(RAW / fn) == SRC[fn]["sha256"]
    ev = []
    for r in tsv(fn):
        v = r["value"].strip()
        c = code_of(v)
        if c is None:
            unmapped[re.sub(r"\s+", "", v)] += 1
            continue
        raw_code = int(re.fullmatch(r"S\s*(\d+)", v).group(1))
        ev.append((float(r["onset"]), c, raw_code >= 49))
    # a trial runs from one cue (S1 / S2) up to the next cue
    starts = [i for i, e in enumerate(ev) if e[1] in (1, 2)]
    for k, i0 in enumerate(starts):
        seg = ev[i0:(starts[k + 1] if k + 1 < len(starts) else len(ev))]
        cue_t, cue_c, is_ascii = seg[0]
        gos = [t for t, c, _ in seg if c == 3]
        go = gos[0] if gos else None
        # a second go marker inside the segment belongs to a trial whose cue marker is missing: nothing after it is used
        end = gos[1] if len(gos) > 1 else float("inf")
        n_extra_go += len(gos) > 1
        seg = [e for e in seg if e[0] < end]
        presses = [t for t, c, _ in seg if c == 4 and go is not None and t >= go]
        press = presses[0] if presses else None
        n_cues += 1
        n_ascii_trials += is_ascii
        n_nogo += go is None
        n_nopress += (go is not None and press is None)
        n_prebefore += (go is None and any(c == 4 for _, c, _ in seg))
        n_multi += len(presses) > 1
        release = next((t for t, c, _ in seg if c == 5 and press is not None and t >= press), None)
        distract = [ms(t - go) for t, c, _ in seg if c == 6 and go is not None and t >= go]
        target = 3000 if cue_c == 1 else 7000
        row = {n: None for n in NAMES}
        row.update(
            dataset=ID, study="Singh et al. 2023 JNNP", experiment="interval timing", group=p["GROUP"], subject=sid,
            subject_excluded_by_authors=None, age=num(p["AGE"]), sex=p["GENDER"], handedness=None,
            population="Parkinson's disease" if p["GROUP"] == "PD" else "healthy", setting="lab",
            device="computer keyboard (spacebar) and screen, 64-channel EEG recorded at the same time",
            session=1, block=None, trial=k + 1, phase="other", time_s=cue_t, task="finger tapping and timing", unit="trial",
            condition="short interval (3 s)" if cue_c == 1 else "long interval (7 s)",
            pert_type="none", pert_value=0, pert_unit=None, feedback=None, excluded_by_authors=None,
            tapping_trial=k + 1, event_in_trial=1, stimulus=None,
            tap_time_ms=ms(press - go) if press is not None else None,
            tap_offset_ms=ms(release - go) if (release is not None and go is not None) else None,
            target_interval_ms=target,
            distractor_times_ms=";".join(f"{d:g}" for d in distract) if distract else None,
            feedback_marker=any(c == 7 for _, c, _ in seg),
        )
        rows.append(row)
        if press is not None:
            if not is_ascii:
                (lens_short if cue_c == 1 else lens_long).append(press - go)
            by_part_target.setdefault((sid, target), []).append(ms(press - go))
            if is_ascii:
                ascii_rt.setdefault((sid, target), []).append(ms(press - go))

checks.internal("all 139 event files match their downloaded hashes", n_hash_ok == 139, f"{n_hash_ok}")
checks.internal(f"{n_cues} trials read: every cue marker (S1/S2) starts one trial; {n_cues - n_nogo} have the go cue",
                n_cues == len(rows) and n_cues > 11000, f"{n_nogo} trials without a go marker, {n_nopress} without a press, {n_extra_go} segments with a second go marker (its cue marker is missing), {n_multi} with more than one press")
checks.internal("presses of the participants with S1-S7 markers fall within the trial length the README gives (6 s for 3-s intervals, 14 s for 7-s intervals, plus 50 ms)",
                max(lens_short) <= 6.05 and max(lens_long) <= 14.05 and min(lens_short + lens_long) > 0,
                f"latest press {max(lens_short):.3f} s (short), {max(lens_long):.3f} s (long)")
asc_subj = sorted(s for s in {k[0] for k in ascii_rt} if len(ascii_rt.get((s, 3000), [])) >= 10 and len(ascii_rt.get((s, 7000), [])) >= 10)
asc_ok = len(asc_subj) == 2 and all(1500 < st.median(ascii_rt[(s, 3000)]) < 5500 and 4500 < st.median(ascii_rt[(s, 7000)]) < 9500
                                    and st.median(ascii_rt[(s, 3000)]) < st.median(ascii_rt[(s, 7000)]) for s in asc_subj)
checks.internal("participants whose markers are ASCII digits (S49-S55), read as S1-S7: median press about 3 s after short cues and about 7 s after long cues (these trials run longer, up to 18-20 s in the paper's description)",
                asc_ok, ", ".join(f"{s}: {st.median(ascii_rt[(s, 3000)]):.0f} / {st.median(ascii_rt[(s, 7000)]):.0f} ms" for s in asc_subj)
                + f"; the third participant has {sum(len(v) for k, v in ascii_rt.items() if k[0] not in asc_subj)} such trials")
checks.internal("short-interval presses are earlier than long-interval presses overall (3-s vs 7-s target)",
                st.median(lens_short) < 4 < st.median(lens_long), f"medians {st.median(lens_short):.2f} s and {st.median(lens_long):.2f} s")
checks.internal("every participant has trials", len({r["subject"] for r in rows}) == 139)

# ---------------------------------------------------------------- published numbers (Table 1 and Table 2)
SRC_T2 = "Table 2, interval timing (7-s intervals; Figure 1E-F)"
import numpy as np  # noqa: E402

grp = {p["participant_id"]: p["GROUP"] for p in parts}
ctrl_meds = [float(np.median(v)) for (s, t), v in by_part_target.items() if t == 7000 and grp[s] == "Control"]
q1, med, q3 = (float(np.percentile(ctrl_meds, q)) for q in (25, 50, 75))
checks.published("Control group, 3rd quartile of participants' median 7-s press time (ms)", q3, 6918, 5, SRC_T2)
checks.published("Control group, 1st quartile of participants' median 7-s press time (ms)", q1, 5476, 15, SRC_T2)
checks.published("Control group, median of participants' median 7-s press time (ms)", med, 5889, 100,
                 SRC_T2 + "; the paper's controls number 41 here 45")
ctrl = [p for p in parts if p["GROUP"] == "Control"]
checks.published("Control group mean MoCA", st.mean(num(p["MOCA"]) for p in ctrl), 26.7, 0.06, "Table 1 (49 controls; 45 in the deposit)")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    else:
        print(f"  [{'ok' if ch['ok'] else 'FAILED'}] {ch['what']} {ch.get('detail', '')}")
print("unmapped marker values (ignored):", dict(unmapped.most_common(8)))

# ---------------------------------------------------------------- release
participants = [dict(person=p["participant_id"], moca=num(p["MOCA"]), updrs=num(p["UPDRS"])) for p in parts]
participant_columns = [
    ("moca", "number", "0-30", "Montreal Cognitive Assessment score, as the deposit gives it."),
    ("updrs", "number", "", "UPDRS score of the patients, as the deposit gives it (the paper's Table 1 lists the motor part, UPDRS III, 0-56). Empty for controls."),
]
meta = dict(
    study="Singh et al. 2023",
    paper=dict(authors="Singh A, Cole RC, Espinoza AI, Wessel JR, Cavanagh JF, Narayanan NS", year=2023,
               title="Evoked mid-frontal activity predicts cognitive dysfunction in Parkinson's disease",
               venue="J Neurol Neurosurg Psychiatry 94(11):945-953", doi="10.1136/jnnp-2022-330154"),
    lab="Narayanan (University of Iowa)",
    data_url="https://openneuro.org/datasets/ds004579",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced: the EEG event markers of each participant were "
             "grouped into one row per trial (cue, go cue, spacebar press and release, distractor vowels, feedback); the EEG "
             "recordings, the deposit's own participant and EEG-file numbers and the repeated R3 events were not carried over."),
    how=[
        "One row is one trial of the interval-timing task of 94 people with Parkinson's disease and 45 controls: a text cue "
        "('Short interval' or 'Long interval'), then a solid box appeared and stayed on while the participant pressed the "
        "keyboard spacebar when they judged the target interval to have elapsed (3 s after short cues, 7 s after long cues, "
        "never told to them), with distractor vowels at random times. The deposit holds the EEG event markers only, so the "
        "trials were rebuilt from them: a trial runs from one cue marker to the next.",
        "tap_time_ms is the spacebar press in ms after the go cue (the appearance of the box); tap_offset_ms is the release, "
        "target_interval_ms the interval asked for, distractor_times_ms the distractor vowels in ms after the go cue and "
        "feedback_marker says that the recording holds a feedback event in the trial (the deposit does not describe it). "
        "Trials without a press (no response in the 6 s or 14 s the trial lasted) and the trials whose go marker is missing "
        "keep their row with the press time empty; time_s is the cue's time from the start of the EEG recording.",
        "Markers are the deposit's S1-S7; two participants, and the first part of a third, carry them as ASCII codes of the "
        "digits (S49-S55), which are read as S1-S7 because the order of events is identical and their press times match the "
        "3-s and 7-s targets. The deposit does not say which participants the paper left out, so the exclusion columns are empty.",
    ],
    unit_note="trial of the interval-timing task (one spacebar press)",
    differences=[
        "Table 2 of the paper (7-s intervals) rests on 41 controls and 43 + 31 + 18 patients in three cognitive-status groups; the deposit holds 45 controls and 94 patients "
        "without the split, so only the control group can be compared: its quartiles of participants' median press times "
        f"match ({q1:.0f} and {q3:.0f} ms against 5476 and 6918) while the median is {med:.0f} ms against 5889 (the paper has four controls fewer; the deposit does not say which).",
        "Table 1 lists 49 controls, 47 PD, 34 PDMCI and 19 PDD participants overall; control MoCA and age means are recomputed from the 45 controls the deposit holds.",
    ],
)


class Rows:
    def __iter__(self):
        return iter(rows)

    def __len__(self):
        return len(rows)


stdlib.release(ID, "tapping", Rows(), meta, checks, participants=participants, participant_columns=participant_columns)
