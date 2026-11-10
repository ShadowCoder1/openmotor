"""Voudouris, Smeets & Brenner (2016) PLoS ONE 11(1):e0146864: fixation biases towards the index finger in almost-natural grasping.

Raw files: raw_extra/Voudouris2016_PLoSONE/AllData.zip (Zenodo 4970099, the Dryad deposit's copy of 'All data'; README.rtf documents the
columns and the stimlist condition codes). Layout: E<experiment>_S<participant>/E<e>_S<p>_<trial> (one tab-separated text file per
trial, 10 ms samples) and E<e>_S<p>_stimlist (trial number, condition code). Nothing in the deposit is run.
Module: timeseries (one row per trial recording, samples as CSV).

Run from the site folder: python3 -I tools/std/convert/Voudouris2016_PLoSONE.py
"""
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Voudouris2016_PLoSONE"
RAW = stdlib.raw_dir(DS)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
FPAT = re.compile(r"E(\d)_S(\d+)/E\d_S\d+_(\d+)$")
SPAT = re.compile(r"E(\d)_S(\d+)/E\d_S\d+_stimlist$")

COND = {
    1: {1: "middle orientation", 2: "clockwise orientation", 3: "anti-clockwise orientation"},
    2: {1: "middle orientation", 2: "clockwise orientation", 3: "anti-clockwise orientation"},
    3: {1: "middle position on support", 2: "far end of support", 3: "close end of support"},
    4: {1: "start close", 2: "start far"},
    5: {1: "obstacle at index finger's contact point", 2: "obstacle at thumb's contact point"},
    6: {1: "middle orientation", 2: "clockwise orientation", 3: "anti-clockwise orientation"},
    7: {1: "middle orientation", 2: "clockwise orientation", 3: "anti-clockwise orientation"},
}
OBJ = {1: "block", 2: "block (start LED at near edge)", 3: "sphere on support", 4: "block", 5: "block with protrusion", 6: "block", 7: "block"}
# names of the two digit markers (readme columns 6-9 and 10-15) per experiment
DIG = {e: ("index", "thumb") for e in range(1, 6)}
DIG[6] = ("little", "thumb")
DIG[7] = ("left_index", "right_index")


def channels(e, extras):
    a, b = DIG[e]
    ch = ["gaze_h_deg", "gaze_v_deg", "start_led_h_deg", "start_led_v_deg",
          f"{a}_h_deg", f"{a}_v_deg", f"{b}_h_deg", f"{b}_v_deg"]
    if extras:
        ch += [f"undocumented_{i:02d}" for i in range(1, 15)]
    ch += [f"{a}_x", f"{a}_y", f"{a}_z", f"{b}_x", f"{b}_y", f"{b}_z", "object_x", "object_y", "object_z",
           "left_eye_h_px", "left_eye_v_px", "right_eye_h_px", "right_eye_v_px"]
    return ch


def units(ch):
    u = []
    for c in ch:
        u.append("deg" if c.endswith("_deg") or c.startswith("undocumented") else "px" if c.endswith("_px") else "mm")
    return u


def parse(text):
    """Rows as (time_ms, values). Rows of 32 fields are fully missing in the first 18 fields; the last 13 are the Cartesian and eye columns."""
    out, extras, bad = [], None, 0
    for line in text.strip("\n").split("\n"):
        f = line.rstrip("\t").split("\t")
        if len(f) < 10:
            assert all(x == "NaN" for x in f), f
            bad += 1
            continue
        n = len(f)
        assert n in (22, 32, 36), n
        t = float(f[0])
        tail = f[-13:]
        if n == 32:
            assert all(x == "NaN" for x in f[1:19]), f[:20]
            head = ["NaN"] * 8
            mid = ["NaN"] * 14
        else:
            head = f[1:9]
            mid = f[9:23] if n == 36 else []
        out.append((t, head + mid + tail, n))
    return out, bad


def num(x):
    return "" if x == "NaN" else x.lstrip("+")


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "AllData.zip")
    files, stims = {}, {}
    for n in z.namelist():
        if n.endswith("/") or "/._" in n or "DS_Store" in n:
            continue
        m = FPAT.match(n)
        if m:
            files[(int(m[1]), int(m[2]), int(m[3]))] = n
        s = SPAT.match(n)
        if s:
            stims[(int(s[1]), int(s[2]))] = n
    subs = sorted({k[:2] for k in files})
    assert set(subs) == set(stims)
    sidx = {s: i + 1 for i, s in enumerate(subs)}  # stdlib assigns S0001... in order of first appearance, which is this order

    rows, extra = [], []
    dts, n_cond = [], defaultdict(list)
    dropped = 0
    layouts = defaultdict(set)
    stim_ok = True
    for (e, p) in subs:
        st = [r.split() for r in z.read(stims[(e, p)]).decode("latin1").strip().split("\n")]
        assert [int(r[0]) for r in st] == list(range(1, len(st) + 1))
        trials = sorted(k[2] for k in files if k[:2] == (e, p))
        stim_ok &= trials == list(range(1, len(st) + 1))
        cnt = defaultdict(int)
        for k in trials:
            data, bad = parse(z.read(files[(e, p, k)]).decode("latin1"))
            dropped += bad
            nm = {n for _, _, n in data}
            extras = 36 in nm
            assert not (36 in nm and 22 in nm)
            layouts[e].add("36" if extras else "22")
            ch = channels(e, extras)
            width = len(ch)
            fname = f"{DS}_S{sidx[(e, p)]:04d}_trial{k:03d}.csv"
            lines = ["time_s," + ",".join(ch)]
            ts = []
            for t, v, n in data:
                if n == 22 or n == 36:
                    assert len(v) == width, (len(v), width)
                elif n == 32 and not extras:
                    v = v[:8] + v[22:]
                assert len(v) == width, (e, p, k, len(v), width)
                ts.append(t)
                lines.append(f"{t / 1000:.2f}," + ",".join(num(x) for x in v))
            dts += [b - a for a, b in zip(ts, ts[1:]) if b > a]
            extra.append((f"samples/{fname}", "\n".join(lines) + "\n"))
            code = int(st[k - 1][1])
            cnt[code] += 1
            lab = COND[e][code]
            rows.append(dict(
                dataset=DS, study=DS, experiment=f"Experiment {e}", group="", subject=f"E{e}_S{p}", person=f"S{p}",
                subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                device="Optotrak 3020 (100 Hz) with EyeLink II (500 Hz, gaze resampled to 100 Hz); head, digit and object markers; "
                       f"object: {OBJ[e]}",
                session=1, block=f"Experiment {e}", trial=k, phase="test", time_s="", task="grasping and lifting", unit="recording",
                condition=lab, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=k, samples_file=f"samples/{fname}",
                signal_kind="gaze direction of the eyes, positions of the start LED and of two digit markers in degrees of visual angle, "
                            "Cartesian positions of the digit and object markers, and eye-tracker pixel positions, during reach-to-grasp and lift",
                channels=";".join(ch), channel_units=";".join(units(ch)), sample_rate_hz=100, n_samples=len(data),
                duration_s=(ts[-1] - ts[0]) / 1000, events="", processing="none beyond the authors' conversion (missing samples empty)",
                trial_info=f"stimlist_condition={code}"))
        n_cond[e] += list(cnt.values())

    persons = {r["person"] for r in rows}
    per_exp = defaultdict(set)
    per_exp_rows = defaultdict(int)
    for r in rows:
        per_exp[r["experiment"]].add(r["person"])
        per_exp_rows[r["experiment"]] += 1
    P = "Methods (Participants) of each experiment"
    for e, want in [(1, 8), (2, 7), (3, 8), (4, 8), (5, 8), (7, 7)]:
        ck.published(f"participants in Experiment {e}", len(per_exp[f"Experiment {e}"]), want, 0, P)
    for e, want in [(1, 528), (2, 462), (3, 528), (4, 352), (5, 352), (7, 462)]:
        ck.published(f"trials recorded in Experiment {e}", per_exp_rows[f"Experiment {e}"], want, 0,
                     f"Results of Experiment {e} (trials before discarding)")
    ck.published("participants in the whole study", len(persons), 14, 0, "Methods, Participants (P1-P14)")
    ck.published("replications of each condition (all subject x condition cells)", sum(sum(v) for v in n_cond.values()) / sum(len(v) for v in n_cond.values()), 22, 0, "Methods, Procedure (each condition 22 times)")
    ck.published("sampling interval of the recordings (ms)", sorted(dts)[len(dts) // 2], 10, 0, "Methods (Optotrak at 100 Hz; gaze resampled to 100 Hz)")
    ck.internal("every subject has trials 1..N matching its stimlist", stim_ok)
    ck.internal("all replication counts are 22", all(c == 22 for v in n_cond.values() for c in v))
    ck.internal("one row and one sample file per recording", len(rows) == len(extra) == len(files), f"{len(rows)} rows")

    meta = dict(
        study=DS,
        paper=dict(authors="Voudouris D, Smeets JBJ, Brenner E", year=2016,
                   title="Fixation Biases towards the Index Finger in Almost-Natural Grasping", venue="PLoS ONE 11(1):e0146864",
                   doi="10.1371/journal.pone.0146864"),
        lab="Brenner & Smeets (Vrije Universiteit Amsterdam)",
        data_url="https://doi.org/10.5061/dryad.1bm8c (copy used: https://doi.org/10.5281/zenodo.4970099)",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Each trial's tab-separated text file became one CSV of samples with time in seconds (the files give milliseconds); "
                 "NaN became an empty cell and a leading + sign was dropped. The condition code of the stimlist file became a text label "
                 "(from the deposit's README) and is also in trial_info. Participant and folder names became codes; participants with the same "
                 "number in different experiments are treated as the same person (the paper numbers P1-P14 across experiments). "
                 "One row without a time stamp (all NaN) in one recording was dropped."),
        unit_note="recording (one grasping trial) of one participant",
        how=["One row is one trial: the participant looked at a start LED, reached to and grasped an object with a precision grip, and lifted it "
             "while gaze (EyeLink II), head, digit and object positions (Optotrak) were recorded. Experiments differ in object, object orientation or "
             "position, and start position (Experiment 1: block rotated 90, clockwise or anti-clockwise; 2: same with the start LED at the near edge; "
             "3: sphere at the middle, far or close end of a support; 4: start close or far; 5: protrusion at the index or thumb side; 6: thumb and little "
             "finger instead of thumb and index; 7: two index fingers). Each condition was repeated 22 times. subject is a participant in one experiment, "
             "person a participant over experiments.",
             "Samples (100 Hz, time_s from the file's own clock in seconds): gaze_h/v_deg is gaze direction, start_led_h/v_deg the start LED position, "
             "<digit>_h/v_deg the two digit markers, all in degrees of visual angle as the deposit's README states; <digit>_x/y/z and object_x/y/z are "
             "Cartesian marker positions (unit not stated in the deposit, mm inferred from their size); the four eye_px columns are the eye tracker's "
             "pixel positions of the left and right eye (README columns 19-22). In Experiment 6 the first digit is the little finger, in Experiment 7 the "
             "first digit is the left and the second the right index finger (README); the README lists the columns of the first digit as the index finger "
             "and the second as the thumb otherwise.",
             "Most files (all but Experiment 3) have 14 more columns after the digit-angle columns than the README describes (36 instead of 22). They are "
             "kept as undocumented_01 to undocumented_14 (degrees-like values that change little within a trial); their meaning is not given in the deposit. "
             "Experiment 3 files have only the 22 documented columns. Some rows have 32 fields (all measurements missing at the start and end of "
             "trials when only the Cartesian and eye columns were recorded); they are placed in the same columns with the missing ones empty.",
             "trial_info holds stimlist_condition, the deposit's code for the trial's condition."],
        differences=["The deposit has no exclusion flags; the paper discarded some trials (66, 40, 41, 28, 36, 77 and 56 in Experiments 1-7) for equipment failure "
                     "or uncertain contact times, but which ones is not in the deposit, so excluded_by_authors is False in every row.",
                     "Experiment 6 has 8 participant folders and 528 recordings in the deposit; the paper reports 7 participants (P5-P7, P9, P11, P13, P14) and "
                     "462 trials. The extra folder is labelled with a number the paper does not list for Experiment 6. It is released as in the deposit and "
                     "no Experiment 6 count was used as a check.",
                     "One recording starts with a row stamped 2210 ms followed by rows from 20 ms; it is kept as shared.",
                     "The paper's gaze biases and contact asynchronies are computed from event detection (thresholds, contact times) that the deposit does "
                     "not provide, so they were not rebuilt; the published checks are counts of participants and trials, replications and the sampling rate."],
    )
    stdlib.release(DS, "timeseries", rows, meta, ck, extra_files=extra)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
    print("layouts", dict(layouts), "dropped rows", dropped)


if __name__ == "__main__":
    main()
