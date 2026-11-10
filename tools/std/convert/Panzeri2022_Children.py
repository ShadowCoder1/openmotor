"""Panzeri et al. (2022), Children 9(11):1760: gait of 6 children and teenagers with hemiparesis after acquired brain injury,
recorded on the GRAIL (Motek) instrumented treadmill with Vicon motion capture before (T0) and after (T1) a month of VR-based gait rehabilitation.
One row per gait cycle (20 per side per assessment); samples are the authors' 101-point time-normalised curves of 119 variables.

Run from the site folder: python3 -I tools/std/convert/Panzeri2022_Children.py
Raw files (untrusted data, read only) in raw_extra/Panzeri2022_Children/ from https://zenodo.org/records/5862254 (S1_T0.csv ... S6_T1.csv).
Demographics were typed in from Table 1 of the paper (the deposit has none).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Panzeri2022_Children"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

# Table 1: subject -> age, sex, impaired side, months since injury, aetiology
T1 = {1: (9.8, "M", "left", 34, "resective surgery for epilepsy"), 2: (18.3, "F", "right", 63, "stroke"),
      3: (7.8, "M", "right", 11, "stroke"), 4: (15.9, "M", "left", 17, "traumatic brain injury"),
      5: (12.8, "F", "left", 74, "brain tumor"), 6: (14.3, "M", "right", 1, "brain tumor")}


def unit(name):
    n = name.replace("_filtered", "")
    if "COP" in n:
        return "m"
    if "Force" in n:
        return "N"
    if n in ("Left.State", "Right.State", "Belt.Speed") and n != "Belt.Speed":
        return "not stated"
    if n == "Belt.Speed":
        return "m/s"
    if n == "Timekey":
        return "s"
    if n.startswith("Marker_"):
        return "not stated"
    if n.startswith("FP") and "Moment" in n:
        return "not stated"
    if n.startswith("Moment ") or n.startswith("Power "):
        return "not stated"
    if n.startswith("Rotation") or n.endswith("Progression"):
        return "deg"
    if n.endswith("Stance.Swing"):
        return "percent of cycle"
    if n.endswith("Time"):
        return "s"
    if n.endswith("Length") or n.endswith("Width"):
        return "m"
    if n == "Walking.Speed":
        return "m/s"
    raise ValueError(name)


rows, extra = [], []
nchan = None
cycles_per = []
cont = []


def base(s, k, trial, **kw):
    age, sex, side_imp, mo, etio = T1[s]
    r = dict(
        dataset=ID, study=ID, experiment=ID, group="", subject=str(s), person=str(s), subject_excluded_by_authors=False,
        age=age, sex=sex, handedness="", population="hemiparesis after acquired brain injury (children and teenagers)",
        setting="clinic", device="GRAIL (Motek) instrumented treadmill, Vicon 10 cameras 100 Hz, force plates 1000 Hz",
        session=k + 1, phase="baseline" if k == 0 else "test", time_s="",
        task="walking", unit="recording", condition="T0 before rehabilitation" if k == 0 else "T1 after rehabilitation",
        pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False, trial=trial, recording=trial)
    r.update(kw)
    return r


for s in range(1, 7):
    age, sex, side_imp, mo, etio = T1[s]
    trial = 0
    for k in (0, 1):
        d = pd.read_csv(RAW / f"S{s}_T{k}.csv", header=None, low_memory=False)
        if d.shape[1] == 104:
            for side in ("left", "right"):
                blk = d[d[0] == side]
                names = list(dict.fromkeys(blk[1]))
                assert len(names) == 119
                if nchan is None:
                    nchan = names
                assert names == nchan, (s, k, side)
                cyc = sorted(blk[2].unique())
                cycles_per.append(len(cyc))
                for c in cyc:
                    trial += 1
                    A = np.array([blk[(blk[1] == nm) & (blk[2] == c)].iloc[0, 3:].values.astype(float) for nm in names]).T
                    assert A.shape == (101, 119) and not np.isnan(A).any()
                    df = pd.DataFrame(A, columns=names)
                    df.insert(0, "cycle_percent", np.arange(101))
                    fn = f"samples/{ID}_S{s:04d}_T{k}_{side}_cycle{int(c) + 1:02d}.csv"
                    extra.append((fn, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
                    rows.append(base(
                        s, k, trial, block=f"T{k} {side} cycles",
                        samples_file=fn, signal_kind="gait cycle curves: joint angles, moments and powers, pelvis and trunk angles, force-plate signals, spatiotemporal values",
                        channels=";".join(names), channel_units=";".join(unit(n) for n in names), sample_rate_hz="", n_samples=101,
                        duration_s="", events="",
                        processing="time-normalised to 101 points (0-100% of the gait cycle) by the authors; FP*_filtered channels are the authors' filtered force-plate signals",
                        trial_info=f"assessment=T{k};cycle_group={side};cycle={int(c) + 1};impaired_side={side_imp}"))
        else:
            names = [str(x) for x in d.iloc[0]]
            A = d.iloc[1:].astype(float).values
            assert not np.isnan(A).any() and len(names) == len(set(names))
            trial += 1
            n = len(A)
            df = pd.DataFrame(A, columns=names)
            df.insert(0, "time_s", np.arange(n) / 100.0)
            fn = f"samples/{ID}_S{s:04d}_T{k}_continuous.csv"
            extra.append((fn, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
            cont.append((s, k, n, len(names)))
            rows.append(base(
                s, k, trial, block=f"T{k} continuous",
                samples_file=fn, signal_kind="continuous treadmill walking: belt speed, force-plate signals (raw and filtered), marker positions (raw and filtered), joint angles, moments, powers, gait events and spatiotemporal values",
                channels=";".join(names), channel_units=";".join(unit(x) for x in names), sample_rate_hz=100, n_samples=n,
                duration_s=n / 100.0, events="",
                processing="as shared by the authors (raw and *_filtered channels side by side, filter not described); time_s is sample number / 100 Hz (the deposit's Timekey column is kept as a channel)",
                trial_info=f"assessment=T{k};impaired_side={side_imp}"))
D = pd.DataFrame(rows)
part = pd.DataFrame([dict(person=str(s), impaired_side=v[2], months_since_injury=v[3], etiology=v[4]) for s, v in T1.items()])
pcols = [("impaired_side", "category", "left, right", "Side of the hemiparesis (Table 1 of the paper)."),
         ("months_since_injury", "number", "months", "Time from the brain injury to the study (Table 1)."),
         ("etiology", "text", "", "Cause of the brain injury (Table 1).")]

P = checks.published
ages = [v[0] for v in T1.values()]
pp = D.drop_duplicates("person")
P("participants", D.person.nunique(), 6, 0, "Results, first sentence")
P("males", int((pp.sex == "M").sum()), 4, 0, "Abstract")
P("mean age (years)", pp.age.mean(), 13.13, 0.05, "Abstract (computed from the one-decimal ages of Table 1)")
checks.internal("12 files all read; 7 cycle files x 2 sides, 5 continuous files", len(cycles_per) == 14 and len(cont) == 5 and len(D) == sum(cycles_per) + 5)
checks.internal("every participant has both assessments", all(set(g.session) == {1, 2} for _, g in D.groupby("person")))
checks.internal("cycle recordings have 101 samples and 119 channels", all(r["n_samples"] in (101, ) or r["sample_rate_hz"] == 100 for r in rows) and len(nchan) == 119)

meta = dict(
    paper=dict(authors="Panzeri D, Genova C, Poggi G, Strazzer S, Biffi E", year=2022,
               title="Visual Feedback and Virtual Reality in Gait Rehabilitation of Hemiparetic Children and Teenagers after Acquired Brain Injury: A Pilot Study",
               venue="Children 9(11):1760", doi="10.3390/children9111760"),
    lab="Biffi (IRCCS Eugenio Medea, Bosisio Parini)",
    data_url="https://zenodo.org/records/5862254", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("The deposit has two layouts. Seven of the 12 CSV files (6 participants x T0/T1) hold, for the labels left and right, 119 variables x a number of gait cycles x 101 points; "
             "each cycle became one samples CSV (101 rows by 119 variables, values unchanged, the long label rows turned into columns). "
             "The other five files (three T0 and two T1) are continuous recordings of about 24 to 33 s with one row per sample and 405 to 410 columns; each became one samples CSV with the columns unchanged. "
             "Age, sex, impaired side, time from injury and etiology were typed in from Table 1 of the paper (the deposit has no clinical file)."),
    how=["One row is either one gait cycle (cycle files) or one continuous treadmill recording (continuous files) of one participant on the GRAIL at one assessment (T0 before, T1 after a month of 20 GRAIL sessions plus 20 physiotherapy sessions). "
         "No biofeedback was given during the recordings. In the cycle files the authors give the same number of cycles for each variable under the label left and under the label right; the deposit does not say what the label means "
         "(presumably the leg whose heel strike starts the cycle), so it is kept as cycle_group in trial_info and block. The deposit's own cycle counts differ between files.",
         "Cycle samples (samples/*_cycleNN.csv) have cycle_percent (0 to 100, the authors' time normalisation) then the 119 variables in the authors' names and order: force-plate signals FP0 and FP1 (raw and filtered), "
         "joint angles (Rotation, degrees), moments and powers of hip, knee and ankle, pelvic and trunk angles, foot progression, and per-cycle spatiotemporal values (step, stride, stance, swing times, lengths, speed). "
         "Continuous samples (samples/*_continuous.csv) have time_s (sample number / 100, since the deposit's Timekey column is a coarse clock that repeats values in two files; Timekey is kept as a channel) then belt speed, the same variables, marker positions and left/right gait-state codes.",
         "Units are inferred, not stated by the deposit: angles in degrees, centre of pressure in m, force in N, times in s, lengths in m, speeds in m/s; moments, powers, marker positions and state codes are marked 'not stated'."],
    differences=["The paper's abstract gives a mean age of 13.13 years; the one-decimal ages of Table 1 average 13.15 (rounding of the ages).",
                 "The paper reports clinical scores (6MWT, GMFM) and describes gait curves only in figures; none of those numbers are in the deposit, so only participant counts, sex and age were checked."],
    unit_note="gait cycle (recording)",
)
stdlib.release(ID, "timeseries", D, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
