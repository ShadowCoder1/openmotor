"""Kliger Amrani & Zion Golumbic (2020), Data in Brief 32:106044: spontaneous tapping (SMT) and synchronization-continuation tapping to
isochronous tone sequences at ten tempi, 20 adults, one row per tap.

Raw files, raw_extra/KligerAmrani2020_DataBrief/ (OSF h3yut, no license stated; MATLAB code and read_me.docx read as text, never run):
  SMTdata.mat        FreeTaps (3 x 60 cell): tap times (s) of each spontaneous trial, 3 trials per session, 3 sessions per participant
                     (participant = 3 consecutive columns); subName holds participant initials (never carried)
  SYNC_CONTdata.mat  rates_part3 (40 x 20): index (1-10) of the tempo in deltaTvec = 250, 350, 450, 550, 650, 800, 1000, 1400, 1800,
                     2200 ms of each of 40 trials per participant (4 blocks of the 10 tempi in random order);
                     tapTimesSync / tapTimesCont (40 x 20 cell): tap times (s) while the tones played, and after they stopped
  PPTdata.mat        perceptual comfort ratings of rhythms (no movement): not converted
Tap times are in seconds in the deposit; ms here. The deposit does not state the clock origin: synchronization times start at the
first tap of the trial after the tones began, continuation times restart near zero after the stop sign, so each phase has its own clock.
The tone onsets themselves are not in the deposit, so stimulus_time_ms and asynchrony_ms stay empty.
Run from the site folder: python3 -I tools/std/convert/KligerAmrani2020_DataBrief.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "KligerAmrani2020_DataBrief"
SRC = "Kliger Amrani & Zion Golumbic 2020, Data in Brief"
RAW = stdlib.raw_dir(ID)
ISI = [250, 350, 450, 550, 650, 800, 1000, 1400, 1800, 2200]
DEV = "custom electro-optic finger tapper; tones (440 Hz) through headphones in a sound-attenuated booth"


def taps(cell):
    x = cell
    while isinstance(x, np.ndarray):
        x = x.reshape(-1)[0] if x.size else ""
    s = str(x).strip().strip("[]")
    return [float(v) for v in s.replace(",", " ").split()] if s else []


def main():
    checks = stdlib.Checks()
    smt = sio.loadmat(RAW / "SMTdata.mat")
    sc = sio.loadmat(RAW / "SYNC_CONTdata.mat")
    ppt = sio.loadmat(RAW / "PPTdata.mat")
    F = smt["FreeTaps"]
    names = [str(n.reshape(-1)[0]) for n in smt["subName"].reshape(-1)]
    uniq = [names[3 * k] for k in range(20)]
    assert all(names[3 * k] == names[3 * k + 1] == names[3 * k + 2] for k in range(20))
    pn = [str(n.reshape(-1)[0]) for n in ppt["subName"].reshape(-1)]
    checks.internal("participant order of the SMT and PPT files is the same (the code pairs columns by index)", pn == uniq)
    rates, S, C = sc["rates_part3"], sc["tapTimesSync"], sc["tapTimesCont"]
    rows = []
    smt_med = {}
    for p in range(20):
        code = f"P{p + 1:02d}"
        base = dict(dataset=ID, study=ID, experiment="", group="", subject=code, person=code, subject_excluded_by_authors=False,
                    age="", sex="", handedness="", population="healthy", setting="lab", device=DEV, task="finger tapping and timing",
                    unit="tap", pert_type="none", pert_value=0, pert_unit="", excluded_by_authors=False, phase="test", time_s="")
        tt = 0
        trial = 0
        meds = []
        for k in range(9):  # trial k of participant p: column 3p + k // 3, row k % 3 (MATLAB linear order)
            t = taps(F[k % 3, 3 * p + k // 3])
            tt += 1
            trial += 1
            arr = np.array(t) * 1000
            iti = np.diff(arr)
            meds.append(float(np.median(iti)))
            for i, v in enumerate(arr):
                rows.append({**base, "session": k // 3 + 1, "block": f"spontaneous session {k // 3 + 1}, trial {k % 3 + 1}", "trial": trial,
                             "condition": "spontaneous", "feedback": "none", "tapping_trial": tt, "event_in_trial": i + 1,
                             "stimulus": "", "tap_time_ms": v, "iti_ms": iti[i - 1] if i else "", "ioi_ms": "", "stimulus_played": False})
        smt_med[p] = meds
        for r in range(40):
            isi = ISI[int(rates[r, p]) - 1]
            for ph, cell in (("synchronization", S[r, p]), ("continuation", C[r, p])):
                t = np.array(taps(cell)) * 1000
                iti = np.diff(t)
                tt += 1
                trial += 1
                for i, v in enumerate(t):
                    rows.append({**base, "session": r // 10 + 1, "block": f"sync-continuation block {r // 10 + 1}", "trial": trial,
                                 "condition": ph, "feedback": "normal" if ph == "synchronization" else "none",
                                 "tapping_trial": tt, "event_in_trial": i + 1, "stimulus": f"{isi} ms ISI", "tap_time_ms": v,
                                 "iti_ms": iti[i - 1] if i else "", "ioi_ms": isi if ph == "synchronization" else "",
                                 "stimulus_played": ph == "synchronization"})
    # checks
    spon = [r for r in rows if r["condition"] == "spontaneous"]
    n_tr = len({(r["person"], r["tapping_trial"]) for r in spon})
    checks.published("participants", len({r["person"] for r in rows}), 20, 0, f"{SRC}, 2.1 Experimental design")
    checks.internal("20 x 9 spontaneous trials", n_tr == 180, str(n_tr))
    per = {}
    for r in spon:
        per.setdefault((r["person"], r["tapping_trial"]), []).append(r)
    checks.internal("every spontaneous trial has 31 taps (30 intervals)", all(len(v) == 31 for v in per.values()))
    ms = np.array([np.mean([meds_ for meds_ in (np.mean(smt_med[p][0:3]), np.mean(smt_med[p][3:6]), np.mean(smt_med[p][6:9]))]) for p in range(20)]) / 1000
    checks.published("median across participants of SMT (s)", float(np.median(ms)), 0.73, 0.005, f"{SRC}, 1. Data description (Fig. 1A)")
    checks.published("lowest participant SMT (s), reported as 0.4", float(ms.min()), 0.4, 0.05, f"{SRC}, 1. Data description")
    checks.published("highest participant SMT (s), reported as 1.5", float(ms.max()), 1.5, 0.05, f"{SRC}, 1. Data description")
    cv = []
    for p in range(20):
        sm = np.array([np.mean(smt_med[p][a:a + 3]) for a in (0, 3, 6)])
        cv.append(np.std(sm, ddof=1) / np.mean(sm))  # MATLAB std
    checks.published("participants with CV of SMT across sessions above 0.25", int(sum(c > 0.25 for c in cv)), 10, 0, f"{SRC}, 1. Data description")
    sync_n = [len(taps(S[r, p])) for r in range(40) for p in range(20)]
    checks.internal("800 synchronization and 800 continuation trials", len(sync_n) == 800 and all(len(taps(C[r, p])) > 0 for r in range(40) for p in range(20)))
    checks.internal("every tempo appears 4 times per participant", all(sorted(np.bincount(rates[:, p])[1:]) == [4] * 10 for p in range(20)))
    checks.internal("no tap times out of order", all(v >= 0 for r in rows if r["iti_ms"] != "" for v in [r["iti_ms"]]))
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Turned the three MATLAB files into one table with a row per tap (spontaneous trials first, then the 40 synchronization-"
                 "continuation trials with their synchronization and continuation taps), converted seconds to milliseconds, computed "
                 "iti_ms as the difference between consecutive taps of a trial, replaced participant initials with codes, and left out "
                 "the perceptual-preference ratings (PPTdata.mat), which are not movement."),
        how=["One row is one finger tap. Spontaneous trials: 3 trials in each of 3 sessions per participant, 31 taps (30 intervals) each, at the most "
             "comfortable rate. Synchronization-continuation trials: 40 per participant (4 blocks of the 10 tempi, tone intervals 250 to "
             "2200 ms, in random order); the participant tapped along with a tone sequence (synchronization), then, after a stop sign, "
             "reproduced the tempo without tones (continuation). The two phases of a trial are separate tapping trials.",
             "Times are in ms. The deposit does not state the time origin: synchronization times start with the trial's tones, "
             "continuation times restart near zero after the stop sign, so the two phases do not share a clock. Tone onsets are not in the "
             "deposit, so stimulus_time_ms and asynchrony_ms are empty; ioi_ms holds the tone interval for synchronization taps.",
             "Spontaneous trials come first in the file, then the synchronization-continuation trials; the three tasks were interleaved in "
             "the experiment but the deposit does not give the interleaving, so trial is not the order in time across tasks."],
        unit_note="tap", differences=["The article says participants tapped 30 taps per spontaneous trial; every trial in the deposit holds 31 taps "
                                     "(30 intervals), so the rows are kept as shared."])
    stdlib.release(ID, "tapping", rows, meta, checks)


main()
