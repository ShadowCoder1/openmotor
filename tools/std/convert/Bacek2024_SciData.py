"""Bacek et al. (2024) Sci Data 11:646: 21 neurotypical adults walking on a dual-belt treadmill at three speeds x five
metronome step frequencies, with and without a locked left knee (orthosis); one row per stride.

Source: Figshare 10.26188/24270022 (CC BY 4.0), the segmented MAT version (article 24270064), one ZIP per participant of
1-2.3 GB. Each ZIP holds SubN_segMechanics.mat (ground reaction forces, centre of pressure, joint angles and the gait
events) and SubN_segEnergetics.mat (EMG and metabolic data, 1.4 GB). Only the 100-200 MB segMechanics member of each ZIP
was range-read from Figshare on the PC (raw_extra/Bacek2024_SciData/prestep, never the whole ZIP), its GaitIndex events
(heel strikes and toe-offs of both feet at 1 kHz, the authors' force-plate segmentation) were written to
stride_events_SubN.json, and the MAT file was deleted. This converter reads those JSON files.
The per-stride timing below is arithmetic on the authors' event times; no joint angles, forces, EMG or metabolic values
are carried (see the README).

Run from the site folder: python3 -I tools/std/convert/Bacek2024_SciData.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Bacek2024_SciData"
MODULE = "walking"

# Table 1 of the paper: id, age (years), height (m), body mass (kg), sex, leg length (m), preferred cadence (steps/min),
# preferred walking speed (m/s)
T1 = """1 35 1.72 74 M 0.87 104 1.12
2 33 1.74 70 M 0.89 94 1.00
3 33 1.86 86 M 0.97 105 1.40
4 25 1.75 76 M 0.87 109 1.00
5 32 1.68 51 M 0.82 112 1.05
6 43 1.73 80 F 0.91 107 1.21
7 27 1.67 64 M 0.86 109 0.94
8 24 1.76 84 M 0.91 105 1.20
9 23 1.60 65 M 0.82 99 1.15
10 22 1.88 77 M 0.95 95 1.05
11 24 1.75 90 M 0.95 103 1.20
12 26 1.74 82 M 0.84 105 1.10
13 29 1.73 82 M 0.87 98 0.96
14 30 1.72 79 M 0.87 91 0.94
15 27 1.55 50 F 0.78 114 0.98
16 28 1.55 51 F 0.84 121 1.23
17 29 1.65 72 F 0.84 105 1.10
18 25 1.80 80 M 0.93 105 1.25
19 57 1.56 59 F 0.74 104 1.03
20 29 1.74 66 M 0.93 96 1.10
21 26 1.85 88 M 0.96 95 1.01"""
# Table 3 of the paper: preferred cadence (steps/min) at 0.4, 0.8, 1.1 m/s, free (c1) walking
T3_FREE = {1: (58, 86, 104), 2: (57, 83, 99), 3: (52, 76, 95), 4: (73, 92, 107), 5: (81, 101, 114), 6: (52, 82, 101),
           7: (93, 105, 111), 8: (70, 89, 102), 9: (56, 81, 99), 10: (63, 83, 97), 11: (70, 91, 101), 12: (72, 93, 106),
           13: (68, 90, 101), 14: (57, 81, 96), 15: (71, 107, 117), 16: (78, 106, 116), 17: (69, 95, 105),
           18: (56, 77, 97), 19: (68, 107, 111), 20: (59, 82, 96), 21: (52, 84, 99)}
SPEEDS = [1.1, 0.8, 0.4]  # Table 2: T1-T5 1.1 m/s, T6-T10 0.8, T11-T15 0.4 (T16-T30 the same with the orthosis)
FREQ = [-10, -5, 0, 10, 20]


def main():
    raw = stdlib.raw_dir(DS)
    ck = stdlib.Checks()
    tab = {}
    for line in T1.splitlines():
        i, age, h, m, sex, leg, cad, spd = line.split()
        tab[int(i)] = dict(age=int(age), height=float(h), mass=float(m), sex=sex, leg=float(leg), cad=int(cad), spd=float(spd))
    rows = []
    part = []
    n_tests = {}
    dur = []
    free_rate = {}
    for n in range(1, 22):
        ev = json.load(open(raw / f"stride_events_Sub{n}.json"))
        t = tab[n]
        part.append(dict(person=f"Sub{n}", height_m=t["height"], mass_kg=t["mass"], leg_length_m=t["leg"],
                         preferred_cadence_spm=t["cad"], preferred_speed_mps=t["spd"]))

        def order(k):
            if k.startswith("T"):
                return (0, int(k[1:]))
            return (1, 0 if "Pref_Start" in k else 1, k)
        trial = 0
        n_tests[n] = 0
        for key in sorted(ev, key=order):
            e = ev[key]
            lhs, rhs, lto, rto = (np.array(e[c], float) / 1000.0 for c in ("LHS_1kHz", "RHS_1kHz", "LTO_1kHz", "RTO_1kHz"))
            m = len(lhs)
            assert len(rhs) == len(lto) == len(rto) == m, (n, key)
            assert e["n_cycles_angles"] == m - 1, (n, key)
            if key.startswith("T"):
                tn = int(key[1:])
                constrained = tn > 15
                k = (tn - 1) % 15
                speed, freq = SPEEDS[k // 5], FREQ[k % 5]
                block, session, phase, wt = f"T{tn}", "", "test", tn
                n_tests[n] += 1
                dur.append(lhs[-1])
                if not constrained and freq == 0:
                    free_rate.setdefault(n, {})[speed] = 120.0 / np.mean(np.diff(lhs))
            else:
                where, sess = key.split("/")
                speed, freq, constrained, wt = t["spd"], None, False, None
                block, session, phase = where.replace("_", " ").lower() + " walking", int(sess[-1]) - 1, "baseline"
                # session numbers in the paper: preparation day is session 1, data days are sessions 2 and 3
                session = int(sess[-1])
            for i in range(m - 1):
                stride_l = lhs[i + 1] - lhs[i]
                stride_r = rhs[i + 1] - rhs[i]
                stance_l = lto[i] - lhs[i]
                stance_r = rto[i] - rhs[i]
                trial += 1
                rows.append(dict(
                    dataset=DS, study="Bacek 2024 Sci Data", experiment="", group="", subject=f"Sub{n}", person=f"Sub{n}",
                    subject_excluded_by_authors=False, age=t["age"], sex=t["sex"], handedness="", population="healthy",
                    setting="lab", device="dual-belt instrumented treadmill, Vicon motion capture", session=session,
                    block=block, trial=trial, phase=phase, time_s=lhs[i], task="walking", unit="stride",
                    condition=(f"{speed:g} m/s, " + (f"{freq:+d}% step frequency, " if freq is not None else "") +
                               ("left knee locked" if constrained else "free")),
                    pert_type="knee_orthosis", pert_value=1 if constrained else 0, pert_unit="left knee locked (1) or free (0)",
                    feedback="normal", excluded_by_authors=False,
                    walking_context="treadmill", walking_trial=wt, belt_speed_fast_mps=speed, belt_speed_slow_mps=speed,
                    stride_start_s=lhs[i], target_step_frequency_pct=freq,
                    stride_time_left_s=stride_l, stride_time_right_s=stride_r,
                    stance_time_left_s=stance_l, stance_time_right_s=stance_r,
                    swing_time_left_s=stride_l - stance_l, swing_time_right_s=stride_r - stance_r,
                    stance_pct_left=100 * stance_l / stride_l, stance_pct_right=100 * stance_r / stride_r,
                    swing_pct_left=100 * (stride_l - stance_l) / stride_l, swing_pct_right=100 * (stride_r - stance_r) / stride_r,
                    step_time_left_s=(lhs[i] - rhs[i - 1]) if i > 0 else None, step_time_right_s=rhs[i] - lhs[i],
                    double_support_left_leading_s=(rto[i - 1] - lhs[i]) if i > 0 else None,
                    double_support_left_trailing_s=lto[i] - rhs[i],
                ))
    import pandas as pd
    df = pd.DataFrame(rows)

    # ---- checks
    ck.published("participants", df.person.nunique(), 21, 0, "Abstract; Methods, Participants")
    full = sum(1 for n in n_tests if n_tests[n] == 30)
    ck.published("participants who completed the entire protocol (all 30 walking tests)", full, 19, 0,
                 "Methods, Participants (19 completed the entire protocol)")
    ck.published("test conditions in the design (3 speeds x 5 step frequencies x 2 constraints)",
                 df[df.phase == "test"].block.nunique(), 30, 0, "Abstract; Experimental conditions")
    # preferred cadence vs measured step rate at 0% in free walking (Table 3 is the authors' value, rounded to 1 step/min)
    diffs = []
    for n, d in free_rate.items():
        for j, sp in enumerate((0.4, 0.8, 1.1)):
            if sp in d:
                diffs.append((n, sp, d[sp], T3_FREE[n][j]))
    ck.published("mean step rate over the free-walking tests at the preferred step frequency (steps/min, 62 participant-speed values)",
                 np.mean([a for _, _, a, _ in diffs]), np.mean([b for _, _, _, b in diffs]), 0.5,
                 "Table 3 (preferred cadence per speed, printed to 1 step/min)")
    ck.internal("measured step rate at preferred frequency vs Table 3 (steps/min), median absolute difference",
                True, f"{np.median([abs(a - b) for _, _, a, b in diffs]):.2f} over {len(diffs)} values")
    ck.internal("every stride has positive times", bool((df.stride_time_left_s > 0).all() and (df.stride_time_right_s > 0).all()))
    ck.internal("stance shorter than stride", bool((df.stance_pct_left < 100).all() and (df.stance_pct_right < 100).all()))
    ck.internal("tests last about 5 minutes (min, max of last heel strike, s)", True, f"{min(dur):.0f}, {max(dur):.0f}")
    ck.internal("tests per participant", True, str(n_tests))
    ck.internal("rows", True, str(len(df)))

    pcols = [("height_m", "number", "m", "Height, from Table 1 of the paper."),
             ("mass_kg", "number", "kg", "Body mass, from Table 1."),
             ("leg_length_m", "number", "m", "Leg length (frontal pelvic marker to lateral ankle marker), Table 1."),
             ("preferred_cadence_spm", "number", "steps/min", "Preferred step frequency at the preferred walking speed, measured in the preparation session, Table 1."),
             ("preferred_speed_mps", "number", "m/s", "Preferred treadmill speed found by the staircase method, Table 1.")]
    meta = dict(
        study=DS,
        paper=dict(authors="Bacek T, Sun M, Liu H, Chen Z, Manzie C, Burdet E, Kulic D, Oetomo D, Tan Y", year=2024,
                   title="A biomechanics and energetics dataset of neurotypical adults walking with and without kinematic constraints",
                   venue="Sci Data 11:646", doi="10.1038/s41597-024-03444-4"),
        lab="Tan / Oetomo (University of Melbourne)",
        data_url="https://doi.org/10.26188/24270022",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("From the segmented MAT files, kept only the gait event times (heel strikes and toe-offs at 1 kHz) and "
                 "turned them into per-stride timing; age, sex, height, mass, leg length and preferred cadence and speed "
                 "come from the paper's Table 1. Ground reaction forces, centre of pressure, joint angles, EMG and "
                 "metabolic data (about 60 GB across the deposit) are not included; participant labels are recoded."),
        how=["One row is one stride of the left foot (left heel strike to the next left heel strike) in one 5-minute walking test; the "
             "right foot's stride, stance, swing and step timings are those of the right-foot cycle that starts at the right heel strike "
             "following that left heel strike. Times are in seconds from the start of the recording, from the authors' force-plate events "
             "(heel strike and toe-off at a 5% body-weight threshold, after their correction of false and missing steps).",
             "Tests T1-T15 are free walking and T16-T30 walking with the passive orthosis locking the left knee (pert_type knee_orthosis, "
             "pert_value 1); within each 15, T1-T5 ran at 1.1 m/s, T6-T10 at 0.8 and T11-T15 at 0.4, and within each speed the five "
             "metronome step frequencies (-10, -5, 0, +10, +20 % of the preferred frequency) were in randomised order. "
             "The tests are labelled by the authors' T number in block; the session each test was in is not stored per test, so session is "
             "empty for them. The 6-minute preferred-speed walks at the start and end of each data session are included (block 'pref start "
             "walking' and 'pref end walking', phase baseline, belt speed = the participant's preferred speed, no imposed frequency).",
             "Double support (left leading) runs from the left heel strike to the previous right foot's toe-off, and is empty for the first "
             "stride of a recording; double support (left trailing) runs from the right heel strike to the left toe-off. Step time of the "
             "left foot runs from the right heel strike to the next left heel strike, and of the right foot from the left to the next right "
             "heel strike. Fast and slow belts were the same speed, so belt_speed_fast and belt_speed_slow are equal and fast_leg is empty.",
             "Joint-angle curves (101 points per stride, hip, knee, ankle, pelvis) and the energetics file (EMG, metabolic cost) were not "
             "carried: reading them would need the whole 1.4 GB energetics member of each ZIP."],
        unit_note="stride",
        differences=["A few tests are shorter than the 5 minutes the paper describes (the last heel strike falls between 134 s and 320 s after the start; the paper says test data were trimmed or removed when a participant had to stop), and two participants have only 15 and 13 of the 30 tests; the paper says one participant could not walk with the orthosis and two dropped out after two of three sessions."],
    )
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE)])
    for c in ck.items:
        print('  ', c.get('what')[:100], c.get('computed', ''), c.get('reported', ''), c.get('detail', '')[:200])
    stdlib.release(DS, MODULE, df, meta, ck, participants=part, participant_columns=pcols)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source']})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:80]}: {c['detail'][:300]}")


if __name__ == "__main__":
    main()
