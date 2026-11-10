"""Megardon, Ludwig & Sumner (2017) J Neurophysiol: trajectory curvature of the second saccade in saccade sequences.

Raw files (OSF t96t2 / component grh62), all read with plain readers, nothing executed:
  deviation-and-duration_all-participants.csv  one row per kept double-stimulus trial (the authors' analysis table)
  saccade-sequences_all-participants.h5        pandas HDF5 store read with h5py (numeric blocks only): saccade event times per trial
The authors' scripts (createDurationFileFromTrajectoryFile.py, extractSaccadeTrajectories.py, Functions/eyetraj.py and the
eyetools library on GitHub) were read as text to define the columns. The trajectory pickle is never opened.

Run from the site folder: python3 -I tools/std/convert/Megardon2017_JNeurophysiol.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Megardon2017_JNeurophysiol"
MODULE = "saccade"


def read_events():
    f = h5py.File(stdlib.raw_dir(DS) / "saccade-sequences_all-participants.h5", "r")
    frames = []
    for k in sorted(f.keys()):
        g = f[k]
        cols = [x.decode() for x in g["block0_items"][:]]
        q = pd.DataFrame(g["block0_values"][:], columns=cols)
        q["marker"] = g["block1_values"][:, 0]
        q["rejected"] = g["block2_values"][:, 0]
        q["file"] = k
        frames.append(q)
    ev = pd.concat(frames, ignore_index=True)
    ev["P"] = ev["file"].str[:2]
    return ev, sorted(f.keys())


def main():
    d = pd.read_csv(stdlib.raw_dir(DS) / "deviation-and-duration_all-participants.csv")
    ev, files = read_events()
    ck = stdlib.Checks()

    # ---- session and trial number from the authors' merged trial id (second file of a participant is offset by 800 x its file index)
    d["fidx"] = ((d["TRIAL_ID"] - 1) // 800).astype(int)
    d["session"] = np.where(d["fidx"] == 0, 1, 2)
    d["raw_id"] = (d["TRIAL_ID"] - 800 * d["fidx"]).astype(int)
    d["trial"] = d.groupby("Participant").cumcount() + 1
    ck.internal("rows are in participant, session, trial order",
                bool(d.groupby("Participant")["TRIAL_ID"].apply(lambda s: (s.diff().dropna() > 0).all()).all()
                     and (d.groupby("Participant")["session"].apply(lambda s: (s.diff().dropna() >= 0).all())).all()))
    # file index of each (participant, session): participants have two consecutive files in the sorted store
    fidx_of_file = {k: i for i, k in enumerate(files)}
    sess_file = {}
    for p in d["Participant"].unique():
        ks = [k for k in files if k[:2] == p]
        assert len(ks) == 2
        sess_file[(p, 1)], sess_file[(p, 2)] = ks
        assert set(d.loc[(d.Participant == p) & (d.session == 2), "fidx"]) == {fidx_of_file[ks[1]]}

    # ---- cross-check the table against the event store: times of fixation offset and of the two saccades
    piv = ev[ev["TRIAL_TYPE"] == 2].pivot_table(index=["file", "TRIAL_ID"], columns="marker", values="time", aggfunc="first")
    code = ev[ev["TRIAL_TYPE"] == 2].groupby(["file", "TRIAL_ID"])["TRIAL_CODE"].first()
    keys = [(sess_file[(p, s)], r) for p, s, r in zip(d["Participant"], d["session"], d["raw_id"])]
    sub = piv.reindex(pd.MultiIndex.from_tuples(keys, names=["file", "TRIAL_ID"]))
    t_fix = (sub[3] - sub[0]).to_numpy()
    t_isi = (sub[3] - sub[2]).to_numpy()
    code_m = code.reindex(pd.MultiIndex.from_tuples(keys, names=["file", "TRIAL_ID"])).to_numpy()
    ck.internal("every table row is found in the event store with the same trial code",
                bool(np.all(code_m == d["TRIAL_CODE"].to_numpy())))
    ck.internal("time from fixation offset and intersaccadic interval equal the event-store timestamp differences",
                bool(np.allclose(t_fix, d["timeFromFix"]) and np.allclose(t_isi, d["timeFromSacc1"])))
    ck.internal("first saccade end from fixation offset (nonInterSacc) equals marker 2 minus marker 0",
                bool(np.allclose(sub[2] - sub[0], d["nonInterSacc"])))

    # ---- authors' exclusion rule (Methods): rejecting more than 40% of trials, from the event store's rejected flags
    tr = ev.groupby(["P", "file", "TRIAL_ID"])["rejected"].max().reset_index()
    rej_rate = tr.groupby("P")["rejected"].mean()
    excluded = set(rej_rate[rej_rate > 0.40].index)
    keep_n = int((rej_rate <= 0.40).sum())
    ck.internal("each participant has 2 sessions of 800 trials in the event store",
                bool(tr.groupby("P")["TRIAL_ID"].size().eq(1600).all()))
    ck.published("participants recruited", d["Participant"].nunique(), 14, 0, "Methods, Participants")
    ck.published("participants left after excluding those who rejected more than 40% of trials", keep_n, 11, 0,
                 "Methods, Participants (14 recruited, 3 excluded)")
    ck.published("sessions per participant", d.groupby("Participant")["session"].nunique().mean(), 2, 0, "Methods, Experimental procedure")

    # ---- standardized rows
    s1 = np.where(d["S1 Duration"] == "shortS1", "short", "long")
    gp = np.where(d["Gap Duration"] == "shortGap", "short", "long")
    cond = [f"S1 {a}, gap {b}, fixation {c}" for a, b, c in zip(s1, gp, d["FixationSide"])]
    info = [f"trial_in_session={r}; trial_code={int(c)}; first_saccade_end_from_fixation_offset_ms={int(n)}"
            for r, c, n in zip(d["raw_id"], d["TRIAL_CODE"], d["nonInterSacc"])]
    out = pd.DataFrame({
        "dataset": DS, "study": DS, "experiment": "two-saccade sequence", "group": "",
        "subject": d["Participant"].to_numpy(),
        "subject_excluded_by_authors": d["Participant"].isin(excluded).to_numpy(),
        "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
        "device": "EyeLink 2000 eye tracker, monitor",
        "session": d["session"].to_numpy(), "block": [f"session {s}" for s in d["session"]],
        "trial": d["trial"].to_numpy(), "phase": "test", "time_s": "",
        "task": "eye movements", "unit": "trial", "condition": cond,
        "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
        "time_from_fixation_offset_ms": d["timeFromFix"].to_numpy(),
        "intersaccadic_interval_ms": d["timeFromSacc1"].to_numpy(),
        "initial_deviation_deg": d["initialDev"].to_numpy(),
        "trial_info": info,
    })
    ck.internal("all 6190 rows of the authors' table kept, 14 participants", len(out) == 6190 and out["subject"].nunique() == 14)
    ck.internal("intersaccadic interval at least 80 ms in every row (the authors' rejection rule)", bool((d["timeFromSacc1"] >= 80).all()))

    # ---- published effects, on the 11 analysed participants
    a = d[~d["Participant"].isin(excluded)]
    isi = a.groupby("S1 Duration")["timeFromSacc1"].mean()
    ck.published("intersaccadic interval: long minus short S1 duration (ms)", isi["longS1"] - isi["shortS1"], 267, 0.5,
                 "Results, intersaccadic interval")
    isg = a.groupby("Gap Duration")["timeFromSacc1"].mean()
    ck.published("intersaccadic interval: long minus short gap (ms)", isg["longGap"] - isg["shortGap"], 31, 0.5,
                 "Results, intersaccadic interval")
    fs = a.groupby("S1 Duration")["timeFromFix"].mean()
    ck.published("time from fixation offset to saccade 2: long minus short S1 (ms)", fs["longS1"] - fs["shortS1"], 272, 0.5,
                 "Results, time from fixation offset")
    n_cell = a.groupby(["Participant", "S1 Duration", "Gap Duration"]).size().mean()
    ck.published("deviation data points per participant and condition", n_cell, 125, 0.5, "Methods, Participants/Data analysis")

    meta = dict(
        study=DS,
        paper=dict(authors="Megardon G, Ludwig C, Sumner P", year=2017,
                   title="Trajectory curvature in saccade sequences: spatiotopic influences vs. residual motor activity",
                   venue="J Neurophysiol 118(2):1310-1320", doi="10.1152/jn.00110.2017"),
        lab="Sumner (Cardiff University) / Ludwig (University of Bristol)",
        data_url="https://osf.io/t96t2/", license="none stated", license_url="",
        changes=("Used the authors' trial table (deviation-and-duration_all-participants.csv) with participant codes; session and "
                 "within-session trial number were recovered from the authors' merged trial ids, and the event store "
                 "(saccade-sequences_all-participants.h5) was used only to cross-check the timings and to apply the "
                 "authors' 40% rejection rule. Saccade trajectories (a Python pickle, never opened), saccade positions and "
                 "control and single-saccade trials stay only in the original deposit."),
        how=["One row is one double-stimulus trial that the authors kept for the curvature analysis: a participant fixated a cross on "
             "the left or right, the fixation cross went off, stimulus S1 appeared above or below after a short or long gap and stayed "
             "for a short or long time, then a second stimulus S2 (its vertical mirror, 60 deg away) appeared and the participant "
             "made a second saccade to it. Condition gives S1 duration, gap and fixation side.",
             "time_from_fixation_offset_ms is the time from fixation offset to second-saccade onset and intersaccadic_interval_ms the "
             "time from the end of the first saccade to second-saccade onset (ms, Eyelink 2000).",
             "initial_deviation_deg is the authors' initialDev: perpendicular distance of the second saccade's path from the straight "
             "line between its start and end, 20 ms after its onset, in deg, with trajectories mirrored so that all saccades go the same "
             "way. Following the paper, leftward deviation is negative and rightward positive; the sign was not rechecked from the "
             "mirroring code.",
             "subject_excluded_by_authors marks the 3 of 14 participants who rejected more than 40% of their trials (computed from the "
             "deposit's rejected flags, following the Methods); the deposit's table still contains their rows. Age, sex and "
             "handedness were not shared per participant (the paper gives 25-30 years, 9 male of 14)."],
        unit_note="double-stimulus trial (the second saccade)",
        differences=["The paper reports a Gap effect of 301 ms on the time from fixation offset to saccade 2; the converted rows give "
                     "302.4 ms (300.1 ms from participant means), so this number is not used as a check.",
                     "The 3 excluded participants are identified here by applying the paper's 40% rule to the deposit's rejected flags; "
                     "the paper does not list who they are."],
    )
    stdlib.release(DS, MODULE, out.to_dict("records"), meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
