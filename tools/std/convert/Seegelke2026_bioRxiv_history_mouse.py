"""Rows of the Seegelke & Heed prime-probe obstacle-avoidance mouse experiments (PsychoPy files A054, A061, A069, ...) in the
90-column reaching format. One row per reach: each trial has a prime reach (or, on stop / imagery / no-movement trials, no
reach) and a probe reach to the same target. Shared by the imagery and history converters (copied, not imported across ids)."""
import math

import csv
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = [r['column'] for r in csv.DictReader(open(Path(__file__).resolve().parents[3] / 'source/columns.csv', newline='', encoding='utf-8'))]

TARGET_RADIUS_DEG = math.degrees(math.atan(0.5 / 8.0))  # 1 cm circle at 8 cm


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def num(x):
    if x is None or x != x:
        return ""
    return "%.8g" % (0.0 if x == 0 else float(x))


def tf(x):
    return "True" if x else "False"


def prepare(df, practice_first_block=True):
    """Adds the authors' exclusion flags (rules reproduced from the papers' counts) and practice flag."""
    df = df.copy()
    first = df.groupby(["subID", "expName"]).blocks_thisN.transform("min")
    df["_practice"] = df.blocks_thisN == first
    go = df.stop_signal_prime == "go"
    df["_multipeak"] = ((df.num_peak_vel_Prime > 2) & go) | (df.num_peak_vel_Probe > 2)
    df["_rtmt"] = ((((df.RT_Prime < 100) | (df.RT_Prime > 1200) | (df.MT_Prime > 1000)) & go)
                   | (df.RT_Probe < 100) | (df.RT_Probe > 1200) | (df.MT_Probe > 1000))
    df["_err"] = df.Error_Any == 1
    return df


def trial_rows(df, ID, experiment, label_of_stop, person_prefix, demo, excluded_subs, order_key=None, exclude_reason=None):
    """df: prepared frame of one PsychoPy file. label_of_stop: name of the non-go trial type ('imagery', 'no_movement', 'nogo').
    demo: dict subID -> (age, sex, handedness). excluded_subs: dict subID -> reason. Returns list of row dicts."""
    out = []
    names = None
    target_ids = {}
    for sub, g in df.groupby("subID", sort=True):
        if order_key is not None:
            g = g.assign(_o=g.expName.map(order_key(g))).sort_values("_o", kind="stable")
        age, sex, hand = demo.get(sub, ("", "", ""))
        seq = 0
        t_row = 0
        for _, r in g.iterrows():
            t_row += 1
            go = r.stop_signal_prime == "go"
            tt = "execution" if go else label_of_stop
            practice = bool(r._practice)
            excluded = practice or bool(r._err) or bool(r._multipeak) or bool(r._rtmt)
            for role in ("prime", "probe"):
                R = role.capitalize()
                obstacle = r[f"obstacle_{role}"] == "yes"
                tgt = float(r[f"deg_targets_{role}"]) % 360.0
                tid = int(r[f"target_location_{role}"])
                is_reach = go or role == "probe"
                cues = []
                if obstacle:
                    cues.append("obstacle")
                if role == "prime" and not go:
                    cues.append("red_target")
                ha = r.get(f"ReachDiff2_sig_{R}")
                row = dict.fromkeys(NAMES, "")
                row.update(
                    dataset=ID, experiment=experiment, study=ID, subject=f"{person_prefix}{sub}", person=f"{person_prefix}{sub}",
                    id_type="lab", person_link="unknown", group="", prior_exposure="unknown",
                    subject_excluded_by_authors=tf(sub in excluded_subs), subject_exclusion_reason=excluded_subs.get(sub, ""),
                    age=num(age) if age != "" else "", sex=sex, handedness=hand, device="mouse", setting="lab", population="healthy",
                    trial=str(seq), trial_orig=str(t_row), session="0",
                    block=f"b{int(r.blocks_thisN):02d}_{role}_{tt}", phase="practice" if practice else "other",
                    trial_kind="reach" if is_reach else "other", break_before=tf(seq == 0 or (t_row > 1 and r.trials_thisN == 0 and role == "prime")),
                    target_angle=num(tgt), target_id=str(tid), target_size_deg=num(TARGET_RADIUS_DEG), target_distance_norm="1",
                    target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
                    instr_goal="cursor_to_target", instr_strategy="unspecified", instr_disclosure="none", instr_report="none",
                    cue_elements="|".join(cues), fb_cued="True", hand_used="right",
                    hand_measure="initial_direction", excluded_by_authors=tf(excluded),
                    fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical",
                    fb_endpoint_transform_only="False", rotation="0", fb_delay_ms="0", outcome_channels="cursor",
                )
                if is_reach:
                    row["hand_angle"] = num(wrap(ha)) if ha == ha else ""
                    row["rt_ms"] = num(r[f"RT_{R}"])
                    row["mt_ms"] = num(r[f"MT_{R}"])
                out.append(row)
                seq += 1
    return out
