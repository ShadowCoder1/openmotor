"""Tomassini, Vercillo, Torricelli & Morrone (2018), Proc R Soc B: tapping and the perception of visual time (main experiment).

Run from the site folder: python3 -I tools/std/convert/Tomassini2018_ProcRSocB.py
Raw file (untrusted data, read only) in raw_extra/Tomassini2018_ProcRSocB/, from the Zenodo copy (record 5003924) of the Dryad
deposit https://doi.org/10.5061/dryad.h9v504t:
  MainExp_rawdata.zip       subj<NN>/subj<NN>-<block>.mat, MATLAB cell array 'all_data' with one struct per trial
  README_for_MainExp_rawdata.docx   the authors' field list (Trial, Tapping, Sound, LED_STIM, LED_FIX, STIM_LAT_postmov,
                            STIM_LAT_premov, SOA, RESP, REJECTED); the fields beat and tap hold the authors' extracted tone and tap
                            events (rows: onset, offset, duration in ms at the 1 kHz sampling, and for taps the interval from the
                            previous event onset), which the README does not list; their meaning is read from the data (the checks
                            below show tap interval = difference of tap onsets, first tap = time since the 4th tone onset).
The 10-s voltage traces (Tapping, Sound, LED_STIM, LED_FIX) are not converted: the extracted events are.
"""
import collections
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tomassini2018_ProcRSocB"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("MainExp_rawdata.zip matches the downloaded hash", stdlib.sha256(RAW / "MainExp_rawdata.zip") == src["MainExp_rawdata.zip"]["sha256"])

z = zipfile.ZipFile(RAW / "MainExp_rawdata.zip")
names = sorted((i.filename for i in z.infolist() if i.filename.endswith(".mat") and not i.filename.startswith("__MACOSX")),
               key=lambda s: (re.search(r"subj(\d+)/", s).group(1), int(re.search(r"-(\d+)\.mat", s).group(1))))


def get(e, k):
    return e[k][0, 0] if k in e.dtype.names else None


def sc(v):
    if v is None:
        return None
    a = np.asarray(v).ravel()
    return a[0].item() if a.size else None


def ev(v):
    return [np.asarray(x).ravel().astype(float) for x in v.ravel()]


DEVICE = ("tapping button read as a voltage at 1 kHz, 800-Hz tones from loudspeakers, probe flashes from a yellow LED "
          "(trial recorded for 10 s)")
trials = []            # one dict per trial with extracted events
n_cells, n_noevent, n_nodata = 0, 0, 0
blocks = collections.defaultdict(set)
bad = collections.Counter()
for n in names:
    subj = re.search(r"(subj\d+)/", n).group(1)
    blk = int(re.search(r"-(\d+)\.mat", n).group(1))
    blocks[subj].add(blk)
    m = loadmat(io.BytesIO(z.read(n)))
    for e in m["all_data"].ravel():
        n_cells += 1
        tap = get(e, "tap")
        if tap is None:
            n_nodata += 1          # the two placeholder cells at the start of each block (README: discard)
            continue
        taps, beats = ev(tap), ev(get(e, "beat"))
        if len(taps) != 4 or len(beats) != 4:
            n_noevent += 1         # tone and tap events empty: nothing extracted by the authors
            continue
        t = np.array([x[:4] for x in taps])
        b = np.array([x[:3] for x in beats])
        so = sc(get(e, "STIM_ONSET"))
        post, pre = sc(get(e, "STIM_LAT_postmov")), sc(get(e, "STIM_LAT_premov"))
        bad["tap interval != difference of tap onsets"] += not np.allclose(np.diff(t[:, 0]), t[1:, 3])
        bad["first tap interval != tap onset - 4th tone onset"] += abs((t[0, 0] - b[3, 0]) - t[0, 3]) > 1
        bad["probe latency after tap 3"] += abs(so - t[2, 0] - post) > 1
        bad["probe latency to tap 4"] += abs(so - t[3, 0] - pre) > 1
        bad["offset - onset != duration"] += not (np.allclose(t[:, 1] - t[:, 0], t[:, 2]) and np.allclose(b[:, 1] - b[:, 0], b[:, 2]))
        resp = sc(get(e, "RESP"))
        trials.append(dict(subj=subj, block=blk, trial=int(sc(get(e, "Trial"))), rej=int(sc(get(e, "REJECTED"))), resp=resp,
                           soa=float(sc(get(e, "SOA"))), post=float(post), pre=float(pre), t=t, b=b))
checks.internal("each trial's tap events are consistent with the authors' latencies (interval = onset difference, first interval from "
                "the 4th tone, probe latencies from tap 3 and 4 within 1 ms, duration = offset - onset)",
                sum(bad.values()) == 0, str(dict(bad)))
by = collections.defaultdict(list)
for t in trials:
    by[t["subj"]].append(t)
subjects = sorted(by)
acc = {s: sum(1 for t in by[s] if t["rej"] == 0) for s in subjects}
last_iti = np.array([t["t"][3, 3] for t in trials if t["rej"] == 0])
print(f"{n_cells} cells, {n_nodata} without events (placeholders), {n_noevent} with empty events, {len(trials)} trials with events")
checks.published("participants in the main experiment", len(subjects), 8, 0, "Methods, Participants (eight in the main experiment)")
checks.published("blocks of trials per participant (mean)", np.mean([len(blocks[s]) for s in subjects]), 26.75, 0.005,
                 "Methods, Procedure (26.75 +- 0.9 blocks per subject)")
checks.published("accepted trials per participant (mean)", np.mean(list(acc.values())), 1076, 1, "Results (1076 +- 48 trials per subject)")
checks.published("mean last inter-tap interval of accepted trials (ms)", last_iti.mean(), 1014, 0.5, "Results (inter-tap = 1014 +- 73 ms, mean +- s.d.)")
checks.published("SD of the last inter-tap interval (ms)", last_iti.std(ddof=1), 73, 0.5, "Results (inter-tap = 1014 +- 73 ms)")
checks.internal("tone events: 4 tones per trial 50 ms long, onsets about 1050 ms apart (1000-ms silence between tones, Methods)", all(np.abs(t["b"][:, 2] - 50).max() <= 5 for t in trials)
                and 1030 < np.mean([np.diff(t["b"][:, 0]).mean() for t in trials]) < 1070,
                f"mean tone interval {np.mean([np.diff(t['b'][:, 0]).mean() for t in trials]):.0f} ms")
checks.internal("probe interval (SOA) between 50 and 300 ms", min(t["soa"] for t in trials) >= 50 and max(t["soa"] for t in trials) <= 300)
checks.internal("responses are 0, 1 or missing (3 trials)", set(t["resp"] for t in trials) <= {0, 1, None}
                and sum(t["resp"] is None for t in trials) == 3)


class Rows:
    def __len__(self):
        return 8 * len(trials)

    def __iter__(self):
        for subj in subjects:
            seq = 0
            for k, t in enumerate(by[subj], start=1):
                common = dict(
                    dataset=ID, study="Tomassini et al. 2018 Proc R Soc B", experiment="main experiment", group="", subject=subj,
                    person=subj, subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy",
                    setting="lab", device=DEVICE, session=1, block=f"block {t['block']}, trial {t['trial']}", phase="test", time_s="",
                    task="finger tapping and timing", unit="tap", condition=f"probe interval {t['soa']:g} ms", pert_type="none",
                    pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=bool(t["rej"]), tapping_trial=k,
                    stimulus="4 tones at 1 Hz, then 4 self-paced taps with a visual probe between taps 3 and 4",
                    probe_interval_ms=t["soa"], probe_after_tap3_ms=t["post"], probe_to_tap4_ms=t["pre"],
                    probe_judged_longer=("" if t["resp"] is None else bool(t["resp"])))
                for j in range(4):       # the four tones of the pacing sequence
                    seq += 1
                    yield dict(common, trial=seq, event_in_trial=j + 1, stimulus_time_ms=float(t["b"][j, 0]), stimulus_played=True,
                               ioi_ms=float(t["b"][j, 0] - t["b"][j - 1, 0]) if j else "")
                for j in range(4):       # the four continuation taps
                    seq += 1
                    yield dict(common, trial=seq, event_in_trial=5 + j, tap_time_ms=float(t["t"][j, 0]), tap_offset_ms=float(t["t"][j, 1]),
                               iti_ms=float(t["t"][j, 3]), stimulus_played=False)


meta = dict(
    study="Tomassini et al. 2018",
    paper=dict(authors="Tomassini A, Vercillo T, Torricelli F, Morrone MC", year=2018,
               title="Rhythmic motor behaviour influences perception of visual time", venue="Proc R Soc B 285(1888):20181597",
               doi="10.1098/rspb.2018.1597"),
    lab="Morrone (University of Pisa)",
    data_url="https://doi.org/10.5061/dryad.h9v504t",
    license="CC0 1.0 (as stated on the Zenodo copy of the Dryad deposit)",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced: for every trial, the authors' extracted tone events "
             "(beat) and tap events (tap) were written as eight rows (four tones, four taps) with the trial's probe values on each row. "
             "The 10-s voltage traces of the button, sound and LEDs, the second-level fields (LED latencies, pulse parameters) and the "
             "422 placeholder cells without events (the README's first two cells of a block, to be discarded) were not converted; 106 trials whose tone and "
             "tap events are empty in the deposit (36 accepted, 70 rejected by the authors) are not written."),
    how=[
        "In each trial the participant heard four 50-ms tones at about 1 Hz, then continued the rhythm by tapping a button four times "
        "with no sound; between the 3rd and 4th tap two LED flashes defined a short visual interval that was compared with a "
        "previously internalized 150-ms interval (judged longer or shorter). One trial is eight rows: the four tones (stimulus_time_ms; "
        "stimulus_played true; ioi_ms the interval from the previous tone) and the four taps (tap_time_ms, tap_offset_ms; "
        "stimulus_played false). Times are ms from the start of the 10-s recording (the authors' samples at 1 kHz). iti_ms is the "
        "authors' interval ending with the tap: from the previous tap onset, and for the first tap from the 4th tone onset.",
        "probe_interval_ms is the SOA of the flashes, probe_after_tap3_ms and probe_to_tap4_ms the authors' latencies of the probe from "
        "the 3rd and the 4th tap onset, probe_judged_longer the response (1 longer, 0 shorter). Trials the authors' REJECTED flag marks "
        "(technical problems or self-reported distraction, per the README) are excluded_by_authors. Eight participants, 26-32 "
        "blocks each (the files of the deposit); the paper's control experiment and the voltage traces are not in this ZIP.",
        "Participant age and sex are not in the deposit (the paper gives group means only). Rows keep the order of the blocks (file "
        "number) and of the trials within each block.",
    ],
    unit_note="tone or tap of a trial (four tones then four taps)",
    differences=["The README does not list the beat and tap event fields; their meaning was read from the data and verified against the "
                 "README's own fields (the probe latencies from tap 3 and 4 are reproduced within 1 ms from the tap onsets).",
                 "The paper's inter-tap interval of 1014 +- 73 ms is reproduced as the mean and SD of the last (4th) interval of the accepted "
                 "trials; the mean of intervals 2-4 is 1007 ms."],
)
stdlib.release(ID, "tapping", Rows(), meta, checks)
