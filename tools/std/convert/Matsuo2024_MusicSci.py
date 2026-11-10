"""Matsuo & Sakaguchi (2024), Music & Science: synchronization-continuation finger tapping, Experiments 1-3, one row per bar.

Run from the site folder: python3 -I tools/std/convert/Matsuo2024_MusicSci.py
Raw files (untrusted data, read only) in raw_extra/Matsuo2024_MusicSci/, from the Zenodo copy (record 8335982) of the Dryad
deposit https://doi.org/10.5061/dryad.xgxd254ns:
  barlength_raw_exp<X><Y>.csv   X = experiment 1-3, Y = condition A-G; row 1 participant ID, row 2 trial, then one row per bar
                                (first column: bar index, negative = synchronization phase, 0 and up = continuation phase),
                                bar lengths in seconds, NaN = missing
  barlength_ratio_exp<X>.csv    the authors' mean bar length in the initial and later ranges (internal check)
The force_ratio_ and ITI_ratio_ files (figure data for Figures 6 and 7) are medians over taps and are not converted.
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Matsuo2024_MusicSci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

# conditions as the README codes them (condition number in the summary files) and the paper's Figure 1 describes them
COND = {
    1: dict(A="A: six rhythm patterns of the bar, control (regular tapping)", B="B: rhythm pattern B", C="C: rhythm pattern C",
            D="D: rhythm pattern D (three-beat bar)", E="E: rhythm pattern E", F="F: rhythm pattern F"),
    2: dict(A="A: control, regular tapping", B="B: accent pattern B, passive (feedback tone intensity set by the program)",
            C="C: accent pattern C, passive", D="D: accent pattern D, passive",
            E="E: accent pattern B, active (participant reproduces the accent by tapping strength)",
            F="F: accent pattern C, active", G="G: accent pattern D, active"),
    3: dict(A="A: control, regular tapping", B="B: crescendo of feedback tones", C="C: diminuendo of feedback tones",
            D="D: ascending scale of feedback tones", E="E: descending scale of feedback tones"),
}
ALTERED = {1: set(), 2: set("BCD"), 3: set("BCDE")}  # feedback tones modulated by the program
NOMINAL_MS = 60000 / 132   # beat interval, 132 bpm (the paper rounds it to 455 ms; the authors' ratio files use 454.5)


def read(name):
    with open(RAW / name, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def num(s):
    s = s.strip()
    return None if s.lower() in ("nan", "") else float(s)


src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
rows = []
nan_cells = 0
per_exp = {}          # exp -> list of participant ids in order of first appearance
tr_data = {}          # (exp, cond, pid, trial) -> list of (bar index, ms)
for exp, conds in COND.items():
    pids = []
    for c in conds:
        name = f"barlength_raw_exp{exp}{c}.csv"
        checks.internal(f"{name} matches the downloaded hash", stdlib.sha256(RAW / name) == src[name]["sha256"])
        t = read(name)
        assert t[0][0] == "ID" and t[1][0] == "trial", name
        ids = [int(float(x)) for x in t[0][1:]]
        trs = [int(float(x)) for x in t[1][1:]]
        for j, (pid, tr) in enumerate(zip(ids, trs), start=1):
            if pid not in pids:
                pids.append(pid)
            series = []
            for k, line in enumerate(t[2:], start=1):
                v = num(line[j])
                if v is None:
                    nan_cells += 1
                series.append((k, int(float(line[0])), None if v is None else round(v * 1000, 3)))
            tr_data[(exp, c, pid, tr)] = series
    per_exp[exp] = pids

for exp, pids in per_exp.items():
    for pid in pids:
        seq = 0
        tn = 0
        for c in COND[exp]:
            for tr in (1, 2):
                series = tr_data.get((exp, c, pid, tr))
                if series is None:
                    continue
                tn += 1
                for k, bi, ms in series:
                    if ms is None:
                        continue
                    seq += 1
                    sync = bi < 0
                    rows.append(dict(
                        dataset=ID, study="Matsuo & Sakaguchi 2024 Music & Science", experiment=f"Experiment {exp}", group="",
                        subject=f"E{exp}_P{pid}", person=f"E{exp}_P{pid}", subject_excluded_by_authors=False,
                        age="", sex="", handedness="", population="healthy", setting="lab",
                        device="force sensor bar as response key sampled at 320 Hz (Arduino), feedback tone on each tap, target tones from a loudspeaker",
                        session=1, block="synchronization" if sync else "continuation", trial=seq,
                        phase="baseline" if sync else "test", time_s="", task="finger tapping and timing", unit="tap",
                        condition=COND[exp][c], pert_type="none", pert_value=0, pert_unit="",
                        feedback="altered" if c in ALTERED[exp] else "normal", excluded_by_authors=False,
                        tapping_trial=tn, event_in_trial=k, stimulus=f"condition {c}", stimulus_played=sync,
                        bar_length_ms=ms))

# ---------------------------------------------------------------- checks
by = {}
for r in rows:
    by.setdefault(r["experiment"], set()).add(r["person"])
for exp in (1, 2, 3):
    n = len(per_exp[exp])
    checks.published(f"Experiment {exp}: participants analysed", n, 14,
                     0, "Results, Experiment %d (preprint e32fv: 15 of 28 completed all conditions, one dropped for large "
                        "variability = 14; 14 of 18; 14 of 15)" % exp if exp != 3 else
                     "Results, Experiment 3 (14 of 15 participants used; preprint e32fv)")
    checks.published(f"Experiment {exp}: trials per condition (14 participants x 2 trials)",
                     len({(k[2], k[3]) for k in tr_data if k[0] == exp and k[1] == "A"}), 28, 0,
                     "Results, Experiment 1 (28 trials); same design in Experiments 2 and 3")
sync = [r["bar_length_ms"] for r in rows if r["block"] == "synchronization" and r["experiment"] != "Experiment 1"
        or (r["block"] == "synchronization" and "three-beat" not in r["condition"])]
checks.published("mean bar length in the synchronization phase (4 beats x 455 ms)", st.mean(sync), 1820, 10,
                 "Results, Experiment 1 (bar length of the original tempo 1820 ms = 455 ms x 4)")
d_sync = [r["bar_length_ms"] for r in rows if r["block"] == "synchronization" and "three-beat" in r["condition"]]
checks.internal("condition D of Experiment 1 has three-beat bars (1365 ms = 3 x 455 ms)", abs(st.mean(d_sync) - 1365) < 10,
                f"{st.mean(d_sync):.1f} ms")

# the authors' summary files: mean bar length in the initial (bar indices 6-10) and later (51-55; 66-70 in 1D) ranges / nominal bar of 4 x 60000/132 ms
def summary_check(exp):
    t = read(f"barlength_ratio_exp{exp}.csv")
    ids = [int(float(x)) for x in t[0][2:]]
    trs = [int(float(x)) for x in t[1][2:]]
    worst, n, bad = 0.0, 0, []
    ordered = list(COND[exp])
    for line in t[2:]:
        cnum, rng = int(float(line[0])), int(float(line[1]))
        c = ordered[cnum - 1]
        for j, v in enumerate(line[2:]):
            if num(v) is None:
                continue
            pid, tr = ids[j], trs[j]
            nominal = NOMINAL_MS * (3 if (exp == 1 and c == "D") else 4)
            lo, hi = (6, 10) if rng == 1 else ((66, 70) if (exp == 1 and c == "D") else (51, 55))
            vals = [ms for k, bi, ms in tr_data[(exp, c, pid, tr)] if ms is not None and lo <= bi <= hi]
            ratio = st.mean(vals) / nominal
            if abs(ratio - float(v)) > 6e-4:
                bad.append((c, rng, tr, round(ratio, 4), float(v)))
            else:
                worst = max(worst, abs(ratio - float(v)))
            n += 1
    return worst, n, bad

for exp in (1, 2, 3):
    worst, n, bad = summary_check(exp)
    print("summary file cells that differ from the rows:", exp, bad)
    checks.internal(f"Experiment {exp}: the authors' initial/later-range bar-length ratios (barlength_ratio file) are reproduced "
                    f"from the rows ({n - len(bad)} of {n} values; {len(bad)} differ)", len(bad) <= 2 and worst < 6e-4,
                    f"largest difference of the others {worst:.2e}; differing cells (condition, range, trial, rows, file): {bad}")
checks.internal("every bar length is between 300 and 6000 ms", all(300 <= r["bar_length_ms"] <= 6000 for r in rows))
checks.internal("missing (NaN) bar lengths are not written as rows", nan_cells > 0, f"{nan_cells} NaN cells in the raw grids, "
                f"{len(rows)} rows written")

meta = dict(
    study="Matsuo & Sakaguchi 2024",
    paper=dict(authors="Matsuo H, Sakaguchi Y", year=2024,
               title="Effects of Rhythm and Accent Patterns on Tempo-Keeping Property of Finger Tapping",
               venue="Music & Science 7:20592043241276959", doi="10.1177/20592043241276959"),
    lab="Sakaguchi (University of Electro-Communications, Tokyo)",
    data_url="https://doi.org/10.5061/dryad.xgxd254ns",
    license="CC0 1.0 (as stated on the Zenodo copy of the Dryad deposit)",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted to the OpenMotor format with the authors' bar-length grids (participant x trial in columns, bars in rows) "
             "unpacked into one row per bar, in seconds changed to ms; cells the authors marked NaN (missing bars and padding) "
             "are not written as rows; the figure-data files (average force ratio, ITI ratio, summary bar-length ratios) "
             "were not converted because they are medians or means of the rows."),
    how=[
        "One row is one bar of the finger-tapping trials in Experiments 1-3: four taps (three in condition D of Experiment 1) "
        "tapped on a force-sensor key at 132 bpm (beat 455 ms), first in time with target tones (synchronization phase, bar "
        "index negative in the deposit) and then without them (continuation phase, index 0 and up). bar_length_ms is the "
        "authors' length of the bar, the sum of its inter-tap intervals; the deposit holds no single tap times, so the rows "
        "are bars and tap_time_ms and iti_ms are empty. The unit column is 'tap' because the unit list has no 'bar'.",
        "The deposit holds the 14 participants per experiment that the authors analysed (28 trials of each condition), "
        "two formal trials per condition; participant IDs are separate for each experiment and the deposit does not say whether "
        "someone took part in more than one, so every experiment's participant is coded as a different person. The authors "
        "dropped the other participants (Experiment 1: 13 who could not do the task and one with large variability; "
        "Experiment 2: 4; Experiment 3: 1) before sharing, so subject_excluded_by_authors is false for everyone.",
        "Condition letters are the authors' (Figure 1 of the paper). Experiment 1: six rhythm patterns A-F with A the control; "
        "Experiment 2: A control, B-D three accent patterns in the passive condition, E-G the same three in the active "
        "condition (README order; feedback tones set by the program in passive, by tapping strength in active); Experiment 3: "
        "A control, then crescendo, diminuendo, ascending scale and descending scale of the feedback tones (README order, "
        "whose list of codes repeats one number). feedback is altered where the program modulated the feedback tones. "
        "Rows are in the order the authors list conditions (the order run: Figure 1 order); trial counts rows within the participant.",
        "Age, sex and handedness are not in the deposit.",
    ],
    unit_note="bar of four taps (three in Experiment 1 condition D)",
    differences=["The deposit and the published paper's analysed samples (14 per experiment) agree with the preprint (PsyArXiv e32fv, "
                 "read in place of the journal text, which was not reachable); the preprint's Wilcoxon z values for the later-range "
                 "change in Experiment 3 (-2.66, -2.05, -3.44) were not reproduced from the rows (-3.4, -2.6, -3.6 for the mean of "
                 "bars 51-55 against 1820 ms), so only counts and the nominal bar length are used as published checks.",
                 "In the authors' barlength_ratio_exp3 file, two later-range values (the last participant's trial 2 in conditions B and D) read 0.765 and 0.739, while the deposit's bar lengths for those trials give 0.956 and 0.924; the other 278 values of that file, and all values of the Experiment 1 and 2 files, are reproduced from the rows."],
)
stdlib.release(ID, "tapping", rows, meta, checks)
