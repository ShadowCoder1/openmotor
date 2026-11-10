"""Thenaisie et al. (2022) Sci Transl Med 14(661):eabo1800: gait of 12 people with Parkinson's disease (with deep brain
stimulation leads in the subthalamic nucleus) on a ladder-like floor task (small steps, big steps, mixed), stairs and an
obstacle; one row per foot strike (step), no neural data.

Source: Zenodo 6867624 (CC BY 4.0), 4_GAIT_DATA_allpatients.zip (12.5 GB). The zip was never fetched whole: its central
directory and, per walking recording, two members were range-read from Zenodo on the PC: PDxx_Tn_gaitEVents_info.mat (3 KB:
left and right foot-strike times, the authors' locomotor state at each foot strike, and gait-initiation times) and
PDxx_Tn_epoched.mat (30-57 MB, of which only the task name TASK was kept; the file also holds band power, EMG and synergy
traces, which were loaded into memory by the MAT reader but discarded unread). The two state tables are MATLAB table objects,
decoded without MATLAB with the MCOS reader of tools/std/convert/Liu2024_RSOS_mcos.py. Everything kept
was written to events.json (raw_extra/Thenaisie2022_SciTranslMed/) and the members were deleted. This converter reads that file.
Folders OLD, old, LeftBC and RightBC of the zip hold earlier copies or per-channel variants of the same recordings (no task name
stored there); only the top-level copy of each recording is used.

Run from the site folder: python3 -I tools/std/convert/Thenaisie2022_SciTranslMed.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Thenaisie2022_SciTranslMed"
MODULE = "walking"
PHASE = {4: "initiation", 1: "continuous walking", 3: "termination"}
STEP_CM = {"SMALL LADDER": 47, "BIG LADDER": 70}


def main():
    raw = stdlib.raw_dir(DS)
    ck = stdlib.Checks()
    R = json.load(open(raw / "events.json"))
    rows = []
    bout_stats = []  # per bout: n state-4, n state-3 steps, init matched
    n_unmatched_init = 0
    n_before_first_init = 0
    for key in sorted(R, key=lambda k: (k.split("/")[0], int(re.search(r"_T(\d+)_", k).group(1)))):
        r = R[key]
        if key.count("/") != 1 or r.get("TASK") is None:
            continue
        pd_id, tn = key.split("/")[0], int(re.search(r"_T(\d+)_", key).group(1))
        task = r["TASK"].strip()
        strikes = []  # (time, side, state)
        for side, pre in (("left", "LFS"), ("right", "RFS")):
            t = np.array(r[pre + "_time"], float)
            st = np.array(r[pre + "_state"]["state"], float)
            assert len(t) == len(st) and np.all(np.diff(t) > 0), key
            strikes += [(float(a), side, int(b)) for a, b in zip(t, st)]
        strikes.sort()
        init = np.array([x for x in r["initFS_time"] if x > 0], float)
        assert np.all(np.diff(init) > 0), key
        # walking bout of each strike: the last gait-initiation time at or before it
        bouts = [int(np.searchsorted(init, s[0] - 1e-9, side="right")) for s in strikes]
        n_before_first_init += sum(1 for b in bouts if b == 0)
        bouts = [max(b, 1) for b in bouts]
        last = {}
        trial0 = len(rows)
        for i, (t, side, st) in enumerate(strikes):
            # next strike of the same foot in the same bout
            nxt = next((u for u in strikes[i + 1:] if u[1] == side), None)
            same_bout_nxt = nxt is not None and bouts[strikes.index(nxt)] == bouts[i]
            prev = strikes[i - 1] if i > 0 and bouts[i - 1] == bouts[i] and strikes[i - 1][1] != side else None
            rows.append(dict(
                dataset=DS, study="Thenaisie 2022 Sci Transl Med", experiment="", group="", subject=pd_id, person=pd_id,
                subject_excluded_by_authors=False, age=None, sex="", handedness="", population="Parkinson's disease",
                setting="clinic", device="foot-mounted gyroscopes (gait events), subthalamic deep brain stimulation leads",
                session=None, block=task, trial=len(rows) - trial0 + 1, phase="other", time_s=t, task="walking", unit="step",
                condition=task.lower(), pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                excluded_by_authors=False, walking_context="overground", walking_trial=tn, heel_strike_side=side,
                stride_start_s=t, stride_time_s=(nxt[0] - t) if same_bout_nxt else None,
                step_time_s=(t - prev[0]) if prev else None, walking_bout=bouts[i],
                gait_bout_phase=PHASE.get(st), target_step_length_cm=STEP_CM.get(task),
                _state=st, _bout=bouts[i], _rec=(pd_id, tn), _t=t))
        # trial index within the patient: continue numbering over recordings
    df = pd.DataFrame(rows)
    # trial = order within the patient's sequence
    df["trial"] = df.groupby("subject").cumcount() + 1
    # ---- checks on the inferred state code (independent of the deposit's data: the paper defines the states)
    for (rec, b), g in df.groupby(["_rec", "_bout"]):
        bout_stats.append((rec, b, int((g._state == 4).sum()), int((g._state == 3).sum()), int(len(g))))
    n4 = np.array([x[2] for x in bout_stats])
    n3 = np.array([x[3] for x in bout_stats])
    # the first strike of every bout carries code 4 (gait initiation) when it exists
    first_state = df.sort_values("_t").groupby(["_rec", "_bout"]).apply(lambda g: g.iloc[0]._state)
    ck.published("patients retained for gait experiments", df.person.nunique(), 12, 0,
                 "Methods, Participants (12 patients retained for gait experiments; Fig 5G n = 12)")
    ck.published("foot strikes per bout in the gait-termination state (last 2 gait cycles), median", np.median(n3[n3 > 0]), 2, 0,
                 "Methods, Identification of gait events and definition of locomotor states (termination = last 2 gait cycles)")
    ck.published("foot strikes per bout in the gait-initiation state (ends at the first heel strike), median", np.median(n4[n4 > 0]), 1, 0,
                 "Methods, Identification of gait events and definition of locomotor states (initiation ends at the first heel strike)")
    ck.internal("share of bouts whose first foot strike has code 4 (initiation)", True, f"{(first_state == 4).mean():.3f} of {len(first_state)} bouts")
    ck.internal("foot strikes before the first gait-initiation time (given bout 1)", True, str(n_before_first_init))
    ck.internal("rows", True, f"{len(df)} steps, {df.subject.nunique()} patients, {df.groupby('subject').walking_trial.nunique().sum()} recordings")
    ck.internal("tasks", True, str(df.block.value_counts().to_dict()))
    ck.internal("step times positive and under 5 s", bool(((df.step_time_s.dropna() > 0) & (df.step_time_s.dropna() < 5)).all()),
                f"median {df.step_time_s.median():.3f} s")
    df = df.drop(columns=["_state", "_bout", "_rec", "_t"])

    meta = dict(
        study=DS,
        paper=dict(authors="Thenaisie Y, Lee K, Moerman C, Scafa S, Galvez A, Pirondini E, Burri M, Ravier J, Puiatti A, Accolla E, Wicki B, Zacharia A, et al.",
                   year=2022, title="Principles of gait encoding in the subthalamic nucleus of people with Parkinson's disease",
                   venue="Sci Transl Med 14(661):eabo1800", doi="10.1126/scitranslmed.abo1800"),
        lab="Courtine / Bloch (EPFL and CHUV Lausanne)",
        data_url="https://zenodo.org/records/6867624",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Kept only the left and right foot-strike times of each walking recording, the authors' locomotor state at each "
                 "foot strike, the walking-bout starts and the task name; derived step time and stride time as differences of "
                 "those times. No neural (STN local field potential), EMG, kinematic or synergy data are included. Patient "
                 "numbers are recoded."),
        how=["One row is one foot strike (a step) of one patient in one walking recording (the authors' T number), in time order. "
             "Times are seconds from the start of the recording, from the authors' gyroscope-based gait-event detection, "
             "validated manually (Methods). step_time_s is the time from the previous foot strike of the other foot and stride_time_s "
             "the time to the next strike of the same foot, both left empty when that strike falls in another walking bout.",
             "A walking bout is one stand, straight walk and stop; bout numbers count the deposit's gait-initiation times (initFS_time). "
             "Tasks (block and condition) are the deposit's own names: SMALL LADDER and BIG LADDER (feet placed on lines marked 47 or "
             "70 cm apart; target_step_length_cm), MIX LADDER (interleaved small and big steps), STAIRS and OBSTACLE.",
             "gait_bout_phase comes from the deposit's numeric state code at each foot strike. The deposit does not name the codes; "
             "they are read as 4 = gait initiation, 1 = continuous walking and 3 = gait termination, because 4 is carried by the first "
             "foot strike of the bout, 3 by the last two and 1 by those between (the checks list the shares), matching the paper's "
             "state definitions. Standing and turning have no foot strikes.",
             "The deposit's state tables carry a sample index that is not on one sampling rate in about half of the recordings, so "
             "the state of each strike is matched to the strike by row order (both tables have exactly one row per foot-strike time "
             "in every recording). The deposit also lists a 0/1 'force' flag per foot strike (0 in small-step, 1 in big-step recordings) "
             "that is not carried. Age, sex and clinical scores of the patients are in the paper's Table S1, not in the deposit."],
        unit_note="foot strike (step)",
        differences=["The paper states that 15 patients performed the walking tasks and 12 were retained for gait analysis; "
                     "the deposit's gait data hold the 12 retained patients."],
    )
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE)])
    for c in ck.items:
        print('  ', c.get('what')[:100], c.get('computed', ''), c.get('reported', ''), c.get('detail', '')[:200])
    stdlib.release(DS, MODULE, df, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source']})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:80]}: {c['detail'][:300]}")


if __name__ == "__main__":
    main()
