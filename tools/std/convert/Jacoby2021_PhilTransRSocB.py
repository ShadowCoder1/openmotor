"""Jacoby, Polak & London (2021), Phil Trans R Soc B: Malian jembe/dundun trios, one row per drum-stroke onset.

Run from the site folder: python3 -I tools/std/convert/Jacoby2021_PhilTransRSocB.py
Raw files (untrusted data, read only) in raw_extra/Jacoby2021_PhilTransRSocB/, from https://osf.io/8wyav/:
  time_E<1-4>-<J1,J2,D>.txt.filt.txt   onset times (s, 3 decimals) of each instrument over the whole recording session of each
                                       ensemble, after the authors' 50-ms false-alarm filter (E = ensemble, J1 = Jembe 1,
                                       J2 = Jembe 2, D = Dundun); the unfiltered detections (time_*.txt), the Sonic Visualiser
                                       sessions and the 12 raw audio files are not used
  processed-data_E1-ALL.take.<n>.line.<A|B>.rec.<r>_<instrument>.csv   the authors' metric annotation of Ensemble 1, takes 1-11
  README_raw_onset_extraction.txt      how the onsets were extracted (read as text)
The annotation files hold strokes that are also in the onset series (matched on the onset time), so annotation columns are filled
for those rows and left empty for the rest.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Jacoby2021_PhilTransRSocB"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
INS = {"J1": "Jembe 1 (lead)", "J2": "Jembe 2 (accompaniment)", "D": "Dundun (timeline)"}
FILE_INS = {"Jembe-1": "J1", "Jembe-2": "J2", "Dundun-1": "D"}

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
used = ([f"time_E{e}-{i}.txt.filt.txt" for e in "1234" for i in INS]
        + sorted(n for n in src if n.startswith("processed-data_E1-ALL.take.")))
checks.internal("every file read matches its downloaded hash", all(stdlib.sha256(RAW / n) == src[n]["sha256"] for n in used),
                f"{len(used)} files")

# ---------------------------------------------------------------- the onset series
series, fseries = {}, {}
for e in "1234":
    for i in INS:
        v = [l.strip() for l in (RAW / f"time_E{e}-{i}.txt.filt.txt").read_text().splitlines() if l.strip()]
        assert all(re.fullmatch(r"\d+\.\d{3}", x) for x in v)
        series[(e, i)] = [int(x.replace(".", "")) for x in v]  # ms
        fseries[(e, i)] = [float(x) for x in v]               # seconds as floating point, for the authors' 50-ms rule
checks.internal("12 series (4 ensembles x Jembe 1, Jembe 2, Dundun), each strictly increasing in time",
                len(series) == 12 and all(all(b > a for a, b in zip(s, s[1:])) for s in series.values()))
n_all = sum(len(s) for s in series.values())
# the authors' filter drops an onset less than 0.05 s after the previous one; evaluated in floating point on the seconds in
# the files, as their software would, it drops 11 onsets that are exactly 50 ms after the previous one (and keeps 4 others)
close = {k: {j for j in range(1, len(s)) if s[j] - s[j - 1] < 0.05} for k, s in fseries.items()}
n_close = sum(len(v) for v in close.values())

# ---------------------------------------------------------------- the authors' annotation (Ensemble 1, takes 1-11)
ann = {}
takes = defaultdict(dict)
for n in sorted(x for x in src if x.startswith("processed-data_E1-ALL.take.")):
    m = re.fullmatch(r"processed-data_E1-ALL\.take\.(\d+)\.line\.([AB])\.rec\.(\d+)_(.+)\.csv", n)
    take, line, rec, ins = int(m.group(1)), m.group(2), int(m.group(3)), FILE_INS[m.group(4)]
    takes[take][ins] = (line, rec)
    with open(RAW / n, newline="") as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == ["Onset_time", "Cycle_number", "Metric_location", "Metric_location_index", "Phase",
                                 "Is_included_in_grid"], rd.fieldnames
        for r in rd:
            key = (ins, int(round(float(r["Onset_time"]) * 1000)))
            assert key not in ann, key
            fl = lambda x: None if x == "NaN" else float(x)  # noqa: E731
            ann[key] = dict(take=take, line=line, cycle=int(r["Cycle_number"]) if r["Cycle_number"] != "NaN" else None,
                            loc=fl(r["Metric_location"]), idx=None if r["Metric_location_index"] == "NaN" else int(float(r["Metric_location_index"])),
                            phase=fl(r["Phase"]), ing=r["Is_included_in_grid"] == "1")
e1 = {i: set(series[("1", i)]) for i in INS}
matched = sum(1 for (i, t) in ann if t in e1[i])
checks.internal("every annotated stroke of Ensemble 1 is an onset of the same instrument's series (same time to the ms)",
                matched == len(ann), f"{matched} of {len(ann)} annotation rows matched")
checks.internal("annotation covers takes 1-11 for the three instruments, line A and B alternating, rec = (take+1)//2",
                sorted(takes) == list(range(1, 12)) and all(sorted(v) == ["D", "J1", "J2"] for v in takes.values())
                and all(len({x[0] for x in v.values()}) == 1 and len({x[1] for x in v.values()}) == 1 for v in takes.values())
                and all(next(iter(v.values()))[0] == "AB"[(t - 1) % 2] and next(iter(v.values()))[1] == (t + 1) // 2
                        for t, v in takes.items()))

# ---------------------------------------------------------------- the OpenMotor table
NAMES = [c[1] for c in stdlib.columns_for("tapping") if c[1] != "person"]


class Rows:
    def __iter__(self):
        for e in "1234":
            for i in INS:
                s, bad = series[(e, i)], close[(e, i)]
                for j, t in enumerate(s):
                    a = ann.get((i, t)) if e == "1" else None
                    row = dict(
                        dataset=ID, study=ID, experiment=f"Ensemble {e}", group="", subject=f"E{e}-{i}",
                        subject_excluded_by_authors=False, age="", sex="", handedness="",
                        population="professional drummers (Mali)", setting="lab",
                        device="piezo transducer on the drum skin, 4-track recorder (studio); onsets by Sonic Visualiser Onsets DS",
                        session=1, block="", trial=j + 1, phase="test", time_s=t / 1000, task="finger/hand", unit="tap",
                        condition=INS[i], pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                        excluded_by_authors=j in bad, tapping_trial=1, event_in_trial=j + 1, stimulus="", tap_time_ms=t)
                    if a:
                        row.update(take_number=a["take"], lineup=a["line"], cycle_number=a["cycle"], metric_location=a["loc"],
                                   metric_location_index=a["idx"], phase_beats=a["phase"], in_metric_grid=a["ing"])
                    yield row

    def __len__(self):
        return n_all


rows = Rows()
SRC = "Methods, Onset detection and Data Preparation (corpus counts)"
checks.published("ensembles", len({k[0] for k in series}), 4, 0, "Methods, Corpus (four trio ensembles)")
checks.published("drummers (one Dundun and two jembe players in each of the 4 trios; 12 series here, one per seat)",
                 len(series), 12, 0, "Methods, Corpus (12 professional drummers)")
checks.published("drumstroke onsets after the 50-ms false-alarm filter (raw data-points)", n_all - n_close, 158263, 0, SRC)
n_exact50 = sum(1 for s in series.values() for a, b in zip(s, s[1:]) if b - a == 50)
checks.internal("no onset is closer than 50 ms to the previous one; 15 are exactly 50 ms after it, and the 11 of them that "
                "fall under 0.05 s in floating point are flagged excluded_by_authors, because the paper's total is reached "
                "only without them", all(b - a >= 50 for s in series.values() for a, b in zip(s, s[1:])) and n_exact50 == 15
                and n_close == 11 and all(series[k][j] - series[k][j - 1] == 50 for k, v in close.items() for j in v),
                f"{n_close} flagged of {n_exact50} onsets exactly 50 ms after the previous one")
checks.internal("every series starts after 0 s and ends within 3 hours of its session",
                all(0 < s[0] and s[-1] < 10_800_000 for s in series.values()),
                "last onset (s): " + ", ".join(f"E{k[0]}-{k[1]} {s[-1] / 1000:.0f}" for k, s in series.items()))
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']} " + (f"computed {c['computed']:.1f} reported {c['reported']}" if c["kind"] == "published" else c.get("detail", "")))

meta = dict(
    study="Jacoby et al. 2021",
    paper=dict(authors="Jacoby N, Polak R, London J", year=2021,
               title="Extreme precision in rhythmic interaction is enabled by role-optimized sensorimotor coupling: analysis and modelling of West African drum ensemble music",
               venue="Phil Trans R Soc B 376(1835):20200331", doi="10.1098/rstb.2020.0331"),
    lab="Jacoby (Max Planck Institute for Empirical Aesthetics)",
    data_url="https://osf.io/8wyav/",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format: the 12 filtered onset series (4 ensembles x 3 instruments) became one row per "
             "drum-stroke onset, in time order, with the authors' metric annotation (take, lineup, cycle, grid location, phase) "
             "added to the strokes of Ensemble 1 takes 1-11 that their processed files cover. The unfiltered detections, the "
             "Sonic Visualiser sessions and the raw audio were not converted, and the drummers' names, which the audio file names "
             "carry, were not used."),
    how=[
        "One row is one drum-stroke onset, from the piezo transducer on one drum in a studio recording of the piece Suku by "
        "a trio of one Dundun and two jembe players (Jembe 1 = lead seat, Jembe 2 = accompaniment seat, Dundun = timeline); "
        "subject is one seat of one ensemble (for example Ensemble 1, Jembe 1), not one drummer, because the two jembe "
        "players swapped seats and roles after every take and the deposit does not say who played which take beyond lineup "
        "A or B.",
        "tap_time_ms and time_s are the authors' onset times in the continuous recording of the whole session of that "
        "ensemble (ms, from the 3-decimal seconds in the files), so one tapping_trial is the whole session and the pauses "
        "between takes are inside it; there is no external stimulus, so stimulus, asynchrony_ms and the interval columns "
        "are empty.",
        "For Ensemble 1 takes 1-11 the authors' processed files give the take number, the lineup, the metric cycle, the "
        "grid location the stroke was assigned to, its phase within the cycle (beats from the downbeat) and whether it was "
        "included in the grid (in_metric_grid); the other rows, and the strokes between takes, carry no annotation.",
        "excluded_by_authors marks 11 onsets that the deposit's filtered series still holds exactly 50 ms after the "
        "previous onset: the paper says onsets within a 50-ms window of an initial onset were discarded, and its total of "
        "158,263 onsets is reached only without these 11 (they are the ones that fall under 0.05 s in floating-point "
        "arithmetic; 4 other onsets also lie exactly 50 ms after the previous one and are kept).",
    ],
    unit_note="drum-stroke onset in a studio recording of a Malian drum trio",
    differences=["The paper counts 158,263 raw drumstroke onsets; the deposit's filtered series hold 158,274, of which 11 lie "
                 "exactly 50 ms after the previous onset (the edge of the paper's filter window) and are flagged. The paper analyses 72 takes of the "
                 "four ensembles, but the deposit's processed files (with take numbers) cover only Ensemble 1 takes 1-11, so "
                 "its final corpus of 150,717 grid-assigned strokes cannot be rebuilt from the deposit."],
)
stdlib.release(ID, "tapping", rows, meta, checks)
