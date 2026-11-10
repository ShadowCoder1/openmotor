"""Jacobsen & Ferris (2023) J Physiol (and Jacobsen & Ferris 2024 PeerJ, same data): mobile-EEG split-belt walking, abrupt
then gradual 2:1 adaptation, 30 young adults. The OpenNeuro deposit (ds004475, CC0) is dominated by 128-channel EEG (1.7 GB per
participant); the walking behaviour it shares is the gait events the authors detected from the treadmill force plates, in
sub-XXX_task-task_events.tsv (right and left heel strike RHS/LHS, toe-off RTO/LTO, plus condition markers). This converter
reads only those events files and participants.tsv; the EEG, head models and MRI are not downloaded.

Raw files: raw_extra/Jacobsen2023_JPhysiol/<sub>_events.tsv (30), participants.tsv, README, dataset_description.json,
data_dictionary.xlsx (read as text by me; abbreviations hard-coded below). One row per stride of the right leg (the fast leg:
the right belt was 1.2 m/s, the left 0.6 m/s in the split conditions). Run: python3 -I tools/std/convert/Jacobsen2023_JPhysiol.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Jacobsen2023_JPhysiol"
MODULE = "walking"
RAW = stdlib.raw_dir(DS)
STARTS = ["B1", "B2", "B3", "SB1", "P1", "SB2", "P2"]
PHASE = {"B1": "baseline", "B2": "baseline", "B3": "baseline", "SB1": "learning", "SB2": "learning",
         "P1": "aftereffect", "P2": "aftereffect"}
COND = {"B1": "tied belts 0.6 m/s (pre-adaptation)", "B2": "tied belts 1.2 m/s (pre-adaptation)",
        "B3": "tied belts 0.9 m/s (pre-adaptation)", "SB1": "abrupt split-belt adaptation",
        "P1": "post-adaptation 1", "SB2": "gradual split-belt adaptation", "P2": "post-adaptation 2"}
DEVICE = ("Bertec instrumented split-belt treadmill with force plates (1000 Hz); gait events from vertical force (10 N threshold); "
          "128-channel dual-layer EEG, neck EMG and Optitrack motion capture recorded but not part of this table")
MAXSTRIDE = 3.0  # s: longer right-heel-strike intervals are treadmill stops or recording gaps, not strides


def segments(ev):
    """Condition state at each time from the markers: a start marker (B1..P2) opens its block, 'End <block>' closes it,
    'pause <block>' suspends and 'continue <block>' resumes. Returns a function time -> block ('' outside any block)."""
    ev = ev.sort_values("onset", kind="stable")
    changes, cur, paused = [(-1.0, "")], "", False
    for t, v in zip(ev["onset"], ev["value"]):
        if v in STARTS:
            cur, paused = v, False
        elif v.startswith("End ") and v[4:] == cur:
            cur = ""
        elif v.startswith("pause "):
            paused = True
        elif v.startswith("continue "):
            cur, paused = v[9:], False
        else:
            continue
        changes.append((t, "" if paused else cur))
    times = np.array([c[0] for c in changes])
    labs = [c[1] for c in changes]
    return lambda t: labs[int(np.searchsorted(times, t, side="right")) - 1], {s: [c[0] for c in changes if c[1] == s] for s in STARTS}


def main():
    part = pd.read_csv(RAW / "participants.tsv", sep="\t", dtype=str)
    part["sub"] = part["participant_id"]
    ck = stdlib.Checks()
    frames, dur, info = [], {s: [] for s in STARTS}, []
    n_events = 0
    for _, p in part.iterrows():
        d = pd.read_csv(RAW / f"{p['sub']}_events.tsv", sep="\t", dtype={"value": str})
        d = d.dropna(subset=["onset"]).sort_values("onset", kind="stable").reset_index(drop=True)
        blk, firsts = segments(d)
        # block durations: start marker to the next marker that ends it (End <block>) or the next start
        marks = d[d["value"].isin(STARTS) | d["value"].str.startswith("End ")]
        for s in STARTS:
            st = marks[marks["value"] == s]["onset"]
            en = marks[marks["value"] == "End " + s]["onset"]
            if len(st) and len(en):
                dur[s].append(float(en.iloc[-1] - st.iloc[0]))
        ev = {k: d.loc[d["value"] == k, "onset"].to_numpy() for k in ("RHS", "LHS", "RTO", "LTO")}
        n_events += sum(len(v) for v in ev.values())
        rhs = ev["RHS"]
        sb2_start = float(d.loc[d["value"] == "SB2", "onset"].iloc[0])
        rows, n_gap, n_odd = [], 0, 0
        for i in range(len(rhs) - 1):
            t0, t1 = rhs[i], rhs[i + 1]
            if t1 - t0 > MAXSTRIDE:
                n_gap += 1
                continue
            r = dict(stride_start_s=t0, stride_time_s=t1 - t0)
            between = d[(d["onset"] > t0) & (d["onset"] < t1) & d["value"].isin(["RHS", "LHS", "RTO", "LTO"])]["value"].tolist()
            if between == ["LTO", "LHS", "RTO"]:
                lto = ev["LTO"][np.searchsorted(ev["LTO"], t0, side="right")]
                lhs = ev["LHS"][np.searchsorted(ev["LHS"], t0, side="right")]
                rto = ev["RTO"][np.searchsorted(ev["RTO"], t0, side="right")]
                j = np.searchsorted(ev["LTO"], lhs, side="right")  # slow leg's toe-off that ends the stance begun at lhs
                r.update(step_time_slow_s=lhs - t0, step_time_fast_s=t1 - lhs, stance_time_fast_s=rto - t0,
                         double_support_slow_s=lto - t0, double_support_fast_s=rto - lhs)
                if j < len(ev["LTO"]) and ev["LTO"][j] - lhs < MAXSTRIDE:
                    r["stance_time_slow_s"] = ev["LTO"][j] - lhs
            else:
                n_odd += 1
            rows.append(r)
        s = pd.DataFrame(rows)
        n = len(s)
        s["block"] = [blk(t) for t in s["stride_start_s"]]
        s["trial"] = np.arange(1, n + 1)
        st, sl = s["stride_time_s"], s.get("stance_time_slow_s")
        s["step_time_asymmetry"] = (s["step_time_fast_s"] - s["step_time_slow_s"]) / st
        s["stance_time_asymmetry"] = (s["stance_time_slow_s"] - s["stance_time_fast_s"]) / st
        s["double_support_asymmetry_s"] = s["double_support_slow_s"] - s["double_support_fast_s"]
        # belt speeds set by the protocol: tied baselines, abrupt split at 2:1 (1.2 right / 0.6 left after a 1.5 s ramp), gradual ramp
        fast = np.full(n, np.nan); slow = np.full(n, np.nan); ratio = np.zeros(n); pt = np.array(["none"] * n, dtype=object)
        for b, (f_, s_) in {"B1": (.6, .6), "B2": (1.2, 1.2), "B3": (.9, .9), "SB1": (1.2, .6)}.items():
            m = (s["block"] == b).to_numpy()
            fast[m], slow[m], ratio[m] = f_, s_, f_ / s_
        m = (s["block"] == "SB2").to_numpy()
        ramp = np.clip(0.001 * (s["stride_start_s"].to_numpy() - sb2_start), 0, 0.3)
        fast[m], slow[m] = (0.9 + ramp)[m], (0.9 - ramp)[m]
        ratio[m] = fast[m] / slow[m]
        ratio[s["block"].isin(["P1", "P2"]).to_numpy()] = 1.0
        pt[s["block"].isin(["SB1", "SB2"]).to_numpy()] = "split_belt"
        blocks = s["block"].tolist()
        out = pd.DataFrame(dict(
            dataset=DS, study=DS, experiment="", group="", subject=p["sub"], subject_excluded_by_authors=False,
            age=float(p["Age"]), sex=p["Sex"], handedness="", population="healthy", setting="lab", device=DEVICE,
            session=1, block=blocks, trial=s["trial"], phase=[PHASE.get(b, "other") for b in blocks],
            time_s=s["stride_start_s"], task="walking", unit="stride", condition=[COND.get(b, "") for b in blocks],
            pert_type=pt, pert_value=np.where(np.array(blocks) == "", np.nan, ratio), pert_unit="ratio", feedback="normal",
            excluded_by_authors=False, fast_leg="right", belt_speed_fast_mps=fast, belt_speed_slow_mps=slow,
            walking_trial=1, walking_context="treadmill"))
        for c in ["stride_start_s", "stride_time_s", "step_time_fast_s", "step_time_slow_s", "stance_time_fast_s",
                  "stance_time_slow_s", "double_support_fast_s", "double_support_slow_s", "step_time_asymmetry",
                  "stance_time_asymmetry", "double_support_asymmetry_s"]:
            out[c] = s[c] if c in s else np.nan
        frames.append(out)
        info.append((len(rhs), n, n_gap, n_odd, int((out.block == "").sum())))
    df = pd.concat(frames, ignore_index=True)
    info = np.array(info)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in df.columns and c != "person":
            df[c] = np.nan
    df = df[[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"]]

    # ---- internal checks
    ck.internal("30 participants, each with strides", df.subject.nunique() == 30 and len(part) == 30)
    ck.internal("one row per right heel strike pair: rows = RHS - 1 - gaps", (info[:, 1] == info[:, 0] - 1 - info[:, 2]).all())
    ck.internal("typical stride time 0.7-1.6 s", 0.7 < df.stride_time_s.median() < 1.6, f"median {df.stride_time_s.median():.3f}")
    ok = df.dropna(subset=["step_time_fast_s", "step_time_slow_s"])
    ck.internal("step times add up to the stride time", np.allclose(ok.step_time_fast_s + ok.step_time_slow_s, ok.stride_time_s, atol=1e-6))
    ck.internal("share of strides with a regular RHS-LTO-LHS-RTO event order is above 95%", 1 - info[:, 3].sum() / info[:, 1].sum() > 0.95,
                f"{info[:, 3].sum()} irregular of {info[:, 1].sum()}; gaps {info[:, 2].sum()}; strides outside any block {info[:, 4].sum()}")

    # ---- published numbers (Jacobsen & Ferris 2024 PeerJ, Participant demographics and Methods; same sample as the J Physiol paper)
    ck.published("participants analysed", len(part), 30, 0, "PeerJ 2024, Results: Participant demographics")
    ck.published("women", int((part["Sex"] == "F").sum()), 15, 0, "PeerJ 2024, Table 1")
    ck.published("men", int((part["Sex"] == "M").sum()), 15, 0, "PeerJ 2024, Table 1")
    ages = part["Age"].astype(float)
    ck.published("mean age (years)", ages.mean(), 22.8, 0.05, "PeerJ 2024, Table 1")
    ck.published("SD of age (years)", ages.std(), 2.6, 0.05, "PeerJ 2024, Table 1")
    ck.published("youngest age", ages.min(), 19, 0, "PeerJ 2024, Results (range 19-29)")
    ck.published("oldest age", ages.max(), 29, 0, "PeerJ 2024, Results (range 19-29)")
    for s, minutes, src in [("SB1", 15, "abrupt adaptation: 2:1 for the entire 15-min period"),
                            ("P1", 10, "post-adaptation #1: 10 minutes"), ("P2", 5, "post-adaptation #2: 5 min")]:
        ck.published(f"mean {s} duration (min)", np.mean(dur[s]) / 60, minutes, 0.5, "PeerJ 2024, Methods: " + src +
                     " (tolerance = the printed whole minutes +-0.5 min)")
    ck.published("mean SB2 duration (min): 5 min ramp + 2 min at 2:1", np.mean(dur["SB2"]) / 60, 7, 0.5,
                 "PeerJ 2024, Methods: gradual adaptation (tolerance +-0.5 min)")

    meta = dict(
        study=DS,
        paper=dict(authors="Jacobsen NA, Ferris DP", year=2023,
                   title="Electrocortical activity correlated with locomotor adaptation during split-belt treadmill walking",
                   venue="J Physiol 601(17):3921-3944", doi="10.1113/jp284505"),
        lab="Ferris (University of Florida)",
        data_url="https://openneuro.org/datasets/ds004475",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Used only the deposit's gait-event files (heel strikes and toe-offs per participant) and participants.tsv: the "
                 "EEG, EMG, head models and MRI are not included. Derived per-stride step, stance and double-support times from "
                 "the event times as the walking module defines them, and gave participant codes."),
        how=["One row is one stride of one participant: from a right heel strike to the next one, with the strides in order. "
             "The right leg walked on the fast belt (1.2 m/s) and the left on the slow belt (0.6 m/s) in the split conditions, so "
             "fast = right and slow = left in every _fast and _slow column. Times are in seconds from the start of the EEG recording.",
             "The deposit shares no kinematics, so step lengths and their asymmetry are empty. Step, stance and double-support times "
             "were computed here from the authors' event times (heel strike RHS/LHS, toe-off RTO/LTO, detected by the authors from "
             "vertical ground reaction force with a 10 N threshold) with the walking module's formulas; they are filled only for strides "
             "whose events come in the order RHS, LTO, LHS, RTO, RHS. Intervals between right heel strikes longer than 3 s "
             "(treadmill stops, recording gaps) are not strides and are left out.",
             "block is the authors' condition marker (B1, B2, B3: tied belts at 0.6, 1.2, 0.9 m/s; SB1: abrupt 2:1 split; P1: "
             "post-adaptation; SB2: gradual split; P2: post-adaptation), taken from the start markers and closed by the matching "
             "'End' markers; strides outside any block (between End and the next start, or during a 'pause') have an empty block. "
             "Belt speeds are the protocol's: tied speeds in B1-B3, 1.2/0.6 m/s from the start of SB1 (the first 1.5 s of ramp at "
             "0.2 m/s^2 are not modelled), and in SB2 a linear ramp of 0.001 m/s^2 from 0.9 m/s to 1.2 and 0.6 m/s over the first 5 min "
             "(computed from the paper's description, with the ramp assumed to start at the SB2 marker). Belt speeds in the "
             "post-adaptation blocks are empty (the paper says tied, 1:1, without a speed). pert_value is the fast:slow ratio."],
        unit_note="stride (from a right heel strike to the next one)",
        differences=["The deposit has 30 of the 33 participants recruited; the three discarded by the authors are not in it. "
                     "The paper's step length asymmetry could not be reproduced because the deposit holds no motion capture.",
                     "The paper's exact belt speeds during the gradual ramp were measured from ankle markers (not shared); the speeds "
                     "here are the protocol's nominal values. Published checks use the PeerJ paper (same dataset, open access); the "
                     "J Physiol paper's full text was not available."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source'][:60]})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:60]}: {c['detail'][:600]}")
    print("per-subject [RHS, strides, gaps, odd, unblocked] mean:", info.mean(0))


if __name__ == "__main__":
    main()
