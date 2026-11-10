"""Crouch, Zheng & Eggleston (2021), arXiv 2109.10743: KT-sEMG keyboard typing dataset, key-press logs only (one typist, 29 runs).

Run from the site folder: python3 -I tools/std/convert/Crouch2021_KTsEMG.py
Raw files (untrusted data, read only) from https://doi.org/10.7910/DVN/AV6XIR: run<N>_keys.txt (key, unix timestamp). The 2 kHz
forearm sEMG files (run<N>.csv, about 25 GB) are not used. The EMG start-time files (run<N>_ts.txt) hold calendar dates and are not used.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Crouch2021_KTsEMG"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
PAT = re.compile(rb"(.*?) (\d{9,11}\.\d+)\r?\n", re.S)
LABEL = {b" ": "space", b"\r": "enter", b"\r\n": "enter_crlf", b"\x08": "backspace"}

keys, ts, run = [], [], []
for i in range(1, 30):
    f = RAW / f"run{i}_keys.txt"
    ok = stdlib.sha256(f) == src[f.name]["sha256"]
    assert ok, f.name
    b = f.read_bytes()
    recs = PAT.findall(b)
    assert PAT.sub(b"", b) in (b"",) or len(PAT.sub(b"", b)) < 25, (i, PAT.sub(b"", b)[:40])   # at most a trailing partial line
    for k, t in recs:
        keys.append(k)
        ts.append(float(t))
        run.append(i)
checks.internal("all 29 key logs match their downloaded hashes", True)
ts = np.array(ts)
run = np.array(run)
n = len(ts)
firsts = np.r_[True, run[1:] != run[:-1]]
step = np.r_[np.nan, np.diff(ts)]
step[firsts] = np.nan
checks.internal("timestamps never run backward within a run", bool((step[~firsts] >= 0).all()))
t0 = pd.Series(ts).groupby(run).transform("min").to_numpy()
text_like = [k for k in keys if len(k) > 1 and k not in LABEL]
checks.internal("every key is one character or a known control key", not text_like, str(text_like[:3]))
lab = [LABEL.get(k, k.decode("ascii")) for k in keys]
checks.internal("no @ sign and no run of 5 or more digits in the typed keys (no e-mail address or number typed)",
                "@" not in lab and not re.search(r"\d{5}", "".join(l if len(l) == 1 else " " for l in lab)))

t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": ID + "_Exp1", "group": "", "subject": "typist", "person": "typist",
    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
    "device": "standard US (QWERTY) USB keyboard; forearm sEMG sleeves worn (EMG not included)",
    "session": run, "block": ["run " + str(r) for r in run], "trial": np.arange(1, n + 1), "phase": "test",
    "time_s": (ts - t0), "task": "typing", "unit": "keypress", "condition": "dictation from podcast speech",
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
    "block_number": run, "block_type": "test", "epoch": "", "stimulus": "", "correct_key": "", "key_pressed": lab, "correct": "",
    "rt_ms": "", "rt_correct_ms": "", "interval_ms": step * 1000, "element_type": "", "sequence": "", "sequence_position": "",
    "repetition": "", "triplet_type": "", "time_in_block_ms": "", "hand": "", "direction": "", "hold_ms": "", "flight_ms": "",
    "finger": "",
})
checks.internal("every key press kept, in run order", len(t) == n == 150813, str(n))

# ---- published numbers (Table I and Section III): the paper keeps the 32 characters a-z, space, period, comma, enter, backspace, apostrophe
keep = np.array([l in set("abcdefghijklmnopqrstuvwxyz.,'") | {"space", "enter", "enter_crlf", "backspace"} for l in lab])
span = pd.Series(ts).groupby(run).agg(lambda s: s.max() - s.min()).sum()
checks.published("key presses of the 32 characters (Table I: 148,739; the paper also drops 27 blocks, hence a small excess here)",
                 keep.sum(), 148739, 60, "Table I, Keystrokes")
checks.published("recorded duration in seconds (Table I: 29,565 s of EMG; here first to last key press of each run, summed)",
                 span, 29565, 250, "Table I, Duration")
checks.published("key presses per second", keep.sum() / span, 5.03, 0.05, "Table I, Keystroke / s")
checks.published("average words per minute (key presses / 5 per word)", keep.sum() / span * 12, 60.37, 0.6, "Table I, Average WPM")
checks.published("number of 32-character classes seen", len(set(l for l, k in zip(lab, keep) if k)), 32, 1, "Table I, Characters")

diffs = [
    "The deposit holds 29 runs and 150,813 logged key presses; 148,774 of them are among the paper's 32 characters, 35 more than Table I (148,739): the paper kept only its 32 "
    "characters (letters a-z, space, period, comma, enter, backspace, apostrophe) and dropped blocks with other keys; all logged keys are kept here, "
    "including 2,039 presses of capital letters, digits and rare symbols.",
    "The paper describes 20 sessions of 30 minutes; the deposit has 29 runs of different lengths (65 s to 2,163 s between first and last key).",
]
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))

meta = dict(
    study="Crouch et al. 2021",
    paper=dict(authors="Crouch MS, Zheng M, Eggleston MS", year=2021, title="Natural Typing Recognition via Surface Electromyography",
               venue="arXiv preprint 2109.10743", doi="10.48550/arxiv.2109.10743"),
    lab="Nokia Bell Labs", data_url="https://doi.org/10.7910/DVN/AV6XIR",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format from the keystroke logs only. The 2 kHz forearm surface EMG recordings (29 CSV files, "
             "about 25 GB) and the EMG start-time files (which hold calendar dates) are not included. Key names are written as text "
             "(space, enter, enter_crlf, backspace); the original characters are kept otherwise."),
    how=[
        "One row is one key press of one professional typist taking dictation from podcast speech (natural conversation between native "
        "English speakers) on a standard US keyboard, in 29 runs while wearing forearm sEMG sleeves; session and block_number are the "
        "run number, trial counts the key presses over all runs, and time_s is the time of the press from the first key press of its run.",
        "key_pressed is the logged key: lower-case and capital letters (the logger recorded the character, not the shift key), space, "
        "enter, backspace, punctuation. interval_ms is the time from the previous key press in the run to this one (empty for the first); "
        "the logger gives only key-down times, so there are no hold times. The keys are the dictated words, so the key sequence spells "
        "the podcast speech the typist heard; no text was typed about the typist. Nothing else was kept: no calendar dates, no text beyond these keys.",
        "There is one typist, so subject and person are the same for all runs. The dictation text is not an experimental stimulus list: "
        "the correct key per press is not recorded, so correct is empty.",
    ],
    unit_note="key press by the typist",
    differences=diffs,
)
stdlib.release(ID, "keypress", t, meta, checks)
