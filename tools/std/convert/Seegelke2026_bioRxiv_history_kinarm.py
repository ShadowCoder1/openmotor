"""Rows of the Seegelke & Heed Kinarm experiments (hand selection A039, A056, A096, B098; obstacle avoidance A073, A074, A080)
in the 90-column reaching format. One row per reach: every trial has a prime and a probe movement. Imported by the history converter.

hand_angle is the direction of the hand 100 ms after the hand left the start (the authors' V_HandStart_100_Ori, degrees, counter-clockwise
positive in the robot frame, x to the right and y away from the body) minus the direction from the hand's start position to the target
centre (V_HandStartTarget_Ori). In the obstacle-avoidance files this equals minus the authors' ReachError100_TC_*_sig_adj (target minus hand),
checked to 1e-3 degrees in the converter. The same formula is applied to the hand-selection files, which carry the two direction columns
but not the authors' error column.
"""
import numpy as np

import Seegelke2026_bioRxiv_history_mouse as M

NAMES, num, tf, wrap = M.NAMES, M.num, M.tf, M.wrap


def hand_angles(df, role):
    R = role.capitalize()
    h = df[f"V_HandStart_100_Ori_{R}"]
    ref = df[f"V_HandStartTarget_Ori_{R}"]
    return wrap(h - ref)


def kinarm_rows(df, ID, experiment, kind, prefix, demo, excluded_subs, cfg):
    """df: one Kinarm file. kind: 'hs' (hand selection) or 'oa' (obstacle avoidance).
    cfg keys: nongo (label of the non-go trial type or None), excl_col (column with the authors' trial-error flag), group_col (or None),
    practice_col/practice_value (or None), passive_col (hand selection: column naming the movement mode), feedback_col (or None),
    rt_cols/mt_cols (prime and probe column names), target_radius_deg (or None)."""
    out = []
    ha = {r: hand_angles(df, r).to_numpy() for r in ("prime", "probe")}
    cols = {k: df[k].to_numpy() for k in set(df.columns)}
    subj = cols["subjNr"]
    # trial order inside a subject = file order (blocks are non-decreasing in every file)
    seq_of = {}
    n = len(df)
    for i in range(n):
        s = subj[i]
        t_row = seq_of.get(s, 0) + 1
        seq_of[s] = t_row
        age, sex, hand = demo.get(int(s), ("", "", ""))
        bc = cfg["block_col"]
        block_no = int(cols[bc][i]) if bc is not None and cols[bc][i] == cols[bc][i] else 0
        prev_block = seq_of.get(("blk", s), None)
        new_block = prev_block != block_no
        seq_of[("blk", s)] = block_no
        err_trial = cols["Error_Trial"][i] == 1
        flag = cols[cfg["excl_col"]][i]
        excluded_trial = err_trial or (flag == 1)
        if kind == "oa":  # more than two velocity peaks in either movement: removed by the paper's rule, not flagged in Error_Any
            excluded_trial = excluded_trial or cols["num_peak_vel_Prime"][i] > 2 or cols["num_peak_vel_Probe"][i] > 2
        phase = "other"
        if cfg.get("practice_col") is not None and cols[cfg["practice_col"]][i] == cfg["practice_value"]:
            phase = "practice"
            excluded_trial = True
        # trial type of this row
        passive = False
        if kind == "hs":
            if cfg["nongo"] is not None:
                go = cols["StopSignal_Str"][i] == "go"
                tt = "execution" if go else cfg["nongo"]
                prime_is_reach = go
            else:  # B098: movement mode
                mode = cols["movement_mode_str"][i]
                passive = mode != "active"
                tt = mode
                prime_is_reach = not passive
            tgt = {"prime": cols["V_StartTarget_Ori_Prime"][i], "probe": cols["V_StartTarget_Ori_Probe"][i]}
            tid = {"prime": cols["Target_Prime"][i], "probe": cols["Target_Probe"][i]}
            hands = {"prime": cols["Hand_Prime_Str"][i], "probe": cols["Hand_Probe_Str"][i]}
            obstacle = {"prime": False, "probe": False}
        else:
            tts = cols["TrialType_str"][i]
            if cfg["nongo"] is not None:  # A073: go / stop (stop = prime blocked by force channels)
                go = tts == "go"
                tt = "execution" if go else cfg["nongo"]
                prime_is_reach = go
            elif cfg.get("passive_types"):  # A074: active / passiveFB / passiveNoFB
                passive = tts != "active"
                tt = tts
                prime_is_reach = not passive
            else:  # A080: vision_on / vision_off
                tt = tts
                prime_is_reach = True
            tgt = {"prime": cols["TargetAngle_Prime"][i], "probe": cols["TargetAngle_Probe"][i]}
            tid = {"prime": "", "probe": ""}
            hands = {"prime": "", "probe": ""}
            obstacle = {"prime": cols["ObstaclePrime_str"][i] == "yes", "probe": cols["ObstacleProbe_str"][i] == "yes"}
        fbv = "online"
        if cfg.get("feedback_col") is not None and cols[cfg["feedback_col"]][i] == "NoFB":
            fbv = "none"
        if cfg.get("vision_off_value") is not None and tts == cfg["vision_off_value"]:
            fbv = "none"
        for role in ("prime", "probe"):
            R = role.capitalize()
            is_reach = prime_is_reach if role == "prime" else True
            cues = []
            if kind == "hs":
                cues.append("hand_color")
            if obstacle[role]:
                cues.append("obstacle")
            if role == "prime" and kind == "hs" and cfg["nongo"] is not None and tt != "execution":
                cues.append(cfg["nongo_cue"])
            if cfg.get("hidden_targets") and cols["file_label"][i].find("NotVisible") >= 0:
                cues.append("targets_hidden")
            if cfg.get("vision_off_value") is not None and tts == cfg["vision_off_value"]:
                cues.append("vision_off_at_go_cue")
            tvals = tgt[role]
            row = dict.fromkeys(NAMES, "")
            row.update(
                dataset=ID, experiment=experiment, study=ID, subject=f"{prefix}{int(s)}", person=f"{prefix}{int(s)}",
                id_type="lab", person_link="unknown",
                group=(str(cols[cfg["group_col"]][i]) if cfg.get("group_col") else ""), prior_exposure="unknown",
                subject_excluded_by_authors=tf(int(s) in excluded_subs), subject_exclusion_reason=excluded_subs.get(int(s), ""),
                age=num(age) if age != "" else "", sex=sex, handedness=hand, device="kinarm", setting="lab", population="healthy",
                trial=str(2 * (t_row - 1) + (0 if role == "prime" else 1)), trial_orig=str(t_row), session="0",
                block=f"b{block_no:02d}_{role}_{tt}", phase=phase,
                trial_kind=("reach" if is_reach else ("proprio_probe" if passive and role == "prime" else "other")),
                break_before=tf(role == "prime" and new_block),
                target_angle=num(float(tvals)) if tvals == tvals else "", target_id=str(tid[role]) if tid[role] != "" else "",
                target_distance_norm="1", target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
                instr_goal="cursor_to_target", instr_strategy="unspecified", instr_disclosure="none", instr_report="none",
                cue_elements="|".join(cues), fb_cued="True", hand_measure="initial_direction",
                excluded_by_authors=tf(excluded_trial),
                fb_visible=fbv, fb_mode="veridical", fb_form="cursor", online_cursor="veridical",
                fb_endpoint_transform_only="False", rotation="0", fb_delay_ms="0", outcome_channels="cursor",
            )
            if cfg.get("target_radius_deg") is not None:
                row["target_size_deg"] = num(cfg["target_radius_deg"])
            if hands[role]:
                row["hand_used"] = hands[role]
            if is_reach:
                a = ha[role][i]
                row["hand_angle"] = num(a) if a == a else ""
                rt = cols[cfg["rt_cols"][role]][i]
                mt = cols[cfg["mt_cols"][role]][i]
                row["rt_ms"] = num(rt) if rt == rt else ""
                row["mt_ms"] = num(mt) if mt == mt else ""
            out.append(row)
    return out
