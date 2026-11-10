"""Hadjiosif, Dajaj, Ranjan & Smith (2025), Sci Rep: tiny visual latencies impair implicit sensorimotor learning.
Experiments 1a (24 participants, 2 latency supersets) and 1b (18 participants, 3 supersets): 30 deg visuomotor rotation with
aim reports at visual-feedback latencies of 25, 85 and 300 ms; one row per outward reach, 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Hadjiosif2025_SciRep.py
Raw files (untrusted data, read only) in raw_extra/Hadjiosif2025_SciRep/: exp1a_data_R2.mat (Git LFS, 160 MB) and
exp1b_data_R2.mat from https://github.com/AlkisMH/Latency (folder SciReports, the revision-2 data). Each holds final_data, one
struct per participant, with one entry per movement (outward and return reaches interleaved; the outward ones are the odd
entries) and tgt_file, one matrix per rest-break-separated block. The authors' .m files (main_analysis_R2.m, buildEpochIndexVectors.m,
organizeLearningAndBaselineData.m, reorganizeData.m) were read as text for the epoch indices and the meaning of the fields.
subject_list (participant initials) is never read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hadjiosif2025_SciRep"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
LAT = {0: 25, 85: 85, 300: 300}


def num(x):
    if x is None or x != x:
        return ""
    return "%.8g" % (0.0 if x == 0 else float(x))


def seg_of(k, o):
    """Epoch of 1-based outward trial k in a superset whose training starts after trial o (authors' buildEpochIndexVectors)."""
    r = k - o
    if -56 <= r <= 0:
        return "baseline57"
    if 1 <= r <= 16:
        return "baseline_train_dir"
    if 17 <= r <= 136:
        return "training"
    if 137 <= r <= 250:
        return "generalization1"
    if 251 <= r <= 310:
        return "extended_training"
    if 311 <= r <= 424:
        return "generalization2"
    if 425 <= r <= 454:
        return "washout"
    return None


PHASE = {"preamble": "practice", "baseline57": "baseline", "baseline_train_dir": "baseline", "training": "learning",
         "extended_training": "learning", "generalization1": "probe", "generalization2": "probe", "washout": "washout"}

short = []
bad_clock = []
rows = []
adapt = {}   # (exp, person, superset) -> late-learning values
nper = {}
for exp, fn, nsets in (("Exp1a", "exp1a_data_R2.mat", 2), ("Exp1b", "exp1b_data_R2.mat", 3)):
    fd = sio.loadmat(RAW / fn, squeeze_me=True, struct_as_record=False)["final_data"]
    O = [380 + 511 * j for j in range(nsets)]  # authors: o = cumsum([380, 511, 511])
    ntr_full_exp = O[-1] + 454
    nper[exp] = len(fd)
    for pi, p in enumerate(fd):
        n_all = len(p.pert)
        ntr_full, ntr = ntr_full_exp, n_all // 2
        assert n_all % 2 == 0 and ntr_full - 30 <= ntr <= ntr_full, (exp, pi, n_all, ntr_full)
        short.append((exp, pi, ntr_full - ntr))
        tf = np.vstack([np.asarray(b, float)[1:-1] for b in p.tgt_file])
        assert tf.shape[0] == n_all, (tf.shape, n_all)
        tfo = tf[0::2]
        blen = [(np.asarray(b).shape[0] - 2) // 2 for b in p.tgt_file]
        blk = np.repeat(np.arange(len(blen)), blen)
        assert len(blk) == ntr
        first_of_block = np.r_[True, blk[1:] != blk[:-1]]
        hd0 = np.asarray(p.head_dir_peak_sp0, float)[0::2]
        ai0 = np.asarray(p.aim_dir0, float)[0::2]
        pert = np.asarray(p.pert, float)[0::2]
        vfb_f = np.asarray(p.vfb, float)[0::2]
        vfb = np.nan_to_num(vfb_f).astype(int)
        tg = np.asarray(p.tgt_angle, float)[0::2]
        st = np.array([float(x) for x in np.asarray(p.start_and_end_times, dtype=object)[0::2, 0]])
        tl = np.asarray(p.t_last_trial, float)
        t_abs = np.empty(ntr)
        for b in range(len(blen)):
            ix = np.where(blk == b)[0]
            t_abs[ix] = st[ix] - st[ix[-1]] + tl[b]
        t_rel = t_abs - t_abs[0]
        if not np.all(np.diff(t_rel[~np.isnan(t_rel)]) > 0):
            bad_clock.append((exp, pi))
            t_rel = np.full(ntr, np.nan)
        ss_of = np.full(ntr, -1)
        seg = [None] * ntr
        for k in range(1, ntr + 1):
            if k <= 380 - 57:
                seg[k - 1] = "preamble"
                continue
            for j, o in enumerate(O):
                s = seg_of(k, o)
                if s is not None and not (j > 0 and k - O[j] <= 0 and False):
                    # baseline57 of superset j+1 lies just before o; washout of superset j ends 57 trials before the next o
                    seg[k - 1] = s
                    ss_of[k - 1] = j
                    break
        sid = f"{ID}_{exp}_{pi + 1:03d}"
        for k in range(ntr):
            s = seg[k]
            j = ss_of[k]
            lo = max(0, O[j] - 56)
            lv = tfo[lo:O[j] + 455, 1]
            lat = LAT[int(np.bincount(lv[~np.isnan(lv)].astype(int)).argmax())] if s != "preamble" else 25
            if np.isnan(hd0[k]) and np.isnan(vfb_f[k]):
                row = dict.fromkeys(names, "")
                row.update(dataset=ID, experiment=f"{ID}_{exp}", study=ID, subject=sid, person=sid, id_type="lab",
                           person_link="assumed_unique", prior_exposure="unknown", subject_excluded_by_authors="False",
                           device="tablet", setting="lab", population="healthy", trial=str(k), trial_orig=str(k + 1), session="0",
                           block=("pre_superset" if s == "preamble" else f"superset{j + 1}_latency{lat}ms_{s}"), phase=PHASE[s],
                           trial_kind="missing", break_before="True" if first_of_block[k] else "False", excluded_by_authors="True")
                rows.append(row)
                continue
            aim_valid = (s in ("baseline_train_dir", "training", "extended_training")) or (
                s in ("generalization1", "generalization2") and tfo[k, 4] == 0) or (s == "washout" and abs(ai0[k]) > 0)
            row = dict.fromkeys(names, "")
            rot = pert[k] if vfb[k] == 1 else 0.0
            ha = -hd0[k]
            row.update(
                dataset=ID, experiment=f"{ID}_{exp}", study=ID, subject=sid, person=sid, id_type="lab", person_link="assumed_unique",
                group="", prior_exposure="unknown", subject_excluded_by_authors="False", handedness="", device="tablet",
                setting="lab", population="healthy", trial=str(k), trial_orig=str(k + 1), session="0",
                block=("pre_superset" if s == "preamble" else f"superset{j + 1}_latency{lat}ms_{s}"),
                phase=PHASE[s], trial_kind="reach", break_before="True" if first_of_block[k] else "False",
                time_s=num(t_rel[k]), target_angle=num((90.0 + tg[k]) % 360.0), target_distance_norm="1",
                target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
                instr_goal="hand_to_target" if (vfb[k] == 0) else "cursor_to_target",
                instr_strategy="allowed" if aim_valid else ("remove" if vfb[k] == 0 else "unspecified"),
                instr_disclosure="exists", instr_report="aim_pre" if aim_valid else "none",
                aim_report_timing="pre" if aim_valid else "", fb_cued="False", hand_used="right",
                hand_angle=num(ha), hand_angle_fb=num(ha), hand_measure="peak_velocity",
                aim_report=num(-ai0[k]) if aim_valid else "", excluded_by_authors="False",
                fb_visible="online" if vfb[k] == 1 else "none",
                fb_mode=("rotated" if (vfb[k] == 1 and rot != 0) else "veridical") if vfb[k] == 1 else "none",
                fb_form="cursor" if vfb[k] == 1 else "none",
                online_cursor=("perturbed" if rot != 0 else "veridical") if vfb[k] == 1 else "none",
                fb_endpoint_transform_only="False", rotation=num(rot),
                cursor_error=num(ha + rot) if vfb[k] == 1 else "",
                outcome_channels="cursor" if vfb[k] == 1 else "")
            rows.append(row)
        # late-learning value per superset (last 20 trials of the 120 training trials, not after a break), authors' sign
        for j, o in enumerate(O):
            ix = [k for k in range(o + 17 - 1 + 100, o + 136) if not first_of_block[k]]
            base = np.nanmean(hd0[o + 1 - 1: o + 16][-15:])
            basea = np.nanmean(ai0[o + 1 - 1: o + 16][-15:])
            sg = np.sign(pert[o + 17])
            tot = np.nanmean((hd0[ix] - base)) * sg
            exp_ = np.nanmean((ai0[ix] - basea)) * sg
            adapt[(exp, pi, j)] = (int(tfo[o + 20, 1]), tot, exp_)
        # internal: time increases

checks.internal("two participants have an inconsistent block clock (time_s left empty for them)", len(bad_clock) == 2, str(bad_clock))
checks.published("participants, Experiment 1a", nper["Exp1a"], 24, 0, "Results (Experiments 1a (n = 24) and 1b (n = 18))")
checks.published("participants, Experiment 1b", nper["Exp1b"], 18, 0, "Results (Experiments 1a (n = 24) and 1b (n = 18))")
checks.published("participants in total", nper["Exp1a"] + nper["Exp1b"], 42, 0, "Results (participants (n = 42))")
import collections
per_block = collections.Counter(r["block"].split("_", 1)[1] if r["block"] != "pre_superset" else "pre" for r in rows)
n_part = nper["Exp1a"] + nper["Exp1b"]
sets_total = 2 * nper["Exp1a"] + 3 * nper["Exp1b"]
for seg_name, n_exp in (("baseline_train_dir", 16), ("training", 120), ("extended_training", 60)):
    n_here = sum(v for k, v in per_block.items() if '_' in k and k.split('_', 1)[1] == seg_name)
    checks.published(f"{seg_name} trials per superset", n_here / sets_total, n_exp, 0,
                     {"baseline_train_dir": "Methods, Experiment schedule (16 additional no-VMR trials in the training direction)",
                      "training": "Methods, Experiment schedule (120 training trials)",
                      "extended_training": "Methods, Experiment schedule (60-trial extended training block)"}[seg_name])
gens = sum(v for k, v in per_block.items() if "generalization" in k) / sets_total
checks.published("generalization trials per superset (2 blocks of 114)", gens, 228, 0, "Methods, Experiment schedule (114-trial generalization blocks)")
for lat, rep_tot, rep_exp in ((25, 29.5, 9.1), (85, 28.8, 15.0), (300, 29.0, 18.4)):
    v = [(t, e) for (l, t, e) in adapt.values() if l == (0 if lat == 25 else lat)]
    checks.published(f"overall late-learning adaptation at {lat} ms, deg (mean of {len(v)} supersets)", np.nanmean([a for a, _ in v]), rep_tot, 1.0,
                     "Results, 'Implicit adaptation increases...' (29.5, 28.8, 29.0 deg)")
    checks.published(f"explicit late-learning adaptation at {lat} ms, deg", np.nanmean([b for _, b in v]), rep_exp, 1.5,
                     "Results, explicit adaptation (9.1, 15.0, 18.4 deg)")
checks.internal("each superset's latency is constant over its training trials",
                all(len({r['block'].split('_')[1] for r in rows if r['subject'] == s and r['block'].startswith(f'superset{j}_') and 'training' in r['block']}) == 1
                    for s in list({r['subject'] for r in rows})[:6] for j in (1, 2)))
checks.internal("hand angles within +-180", all(abs(float(r["hand_angle"])) <= 180 for r in rows if r["hand_angle"]))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Smith (Harvard)",
    data_url="https://github.com/AlkisMH/Latency", license="none stated", license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, hand and aim angles negated to counter-clockwise positive, "
            "return movements dropped.",
    unit_note="outward reach of one participant",
    differences=[
        "The late-learning means were recomputed here from the rows without the authors' across-participant outlier rejection (3 IQR), so they agree with the paper within about 1 deg rather than exactly.",
        "The paper describes a 25-trial washout block; the authors' index vectors and the file hold 30 trials, which are all kept (block ending _washout).",
        "time_s is empty for two participants (one in each experiment) whose block start times in the file are not consistent (t_last_trial runs backwards at one block boundary).",
        "One Experiment 1a participant has 57 reaches (a whole baseline block) with no data at all in the file; they are written as placeholder rows (trial_kind missing, excluded_by_authors True) so that trial_orig stays continuous.",
        "Fifteen of the 24 Experiment 1a participants have 30 fewer trials than the others (the authors' code pads the missing last block of 30 reaches, the washout of the last superset, with NaN); they are kept as they are.",
        "The first 323 outward trials of each participant (block pre_superset) are not described in the paper beyond a 17-trial familiarization; they include trials with random 15 and 30 degree rotations and trials without a cursor. They are kept and labelled."],
    how=[
        "One row is one outward reach on a digitizing tablet toward a target on a screen above the hand; the return reaches in the authors' file (every second entry) are not included. Each participant did 2 (Experiment 1a) or 3 (Experiment 1b) supersets, each at a different visual-feedback latency and with the opposite 30 deg rotation: 57 baseline reaches to 19 directions, 16 more baseline reaches and 120 training reaches in one direction, a 114-reach no-feedback generalization block, 60 extended training reaches, a second generalization block and 30 washout reaches. The block label names the superset, its total latency (25, 85 or 300 ms; the file codes them 0, 85, 300) and the epoch. Latency acts on the cursor throughout the movement; reach90 has no column for it, so it is only in block.",
        "hand_angle is the authors' head_dir_peak_sp0: the direction of the hand at peak speed relative to the target direction seen from the movement onset position (degrees), and aim_report is aim_dir0, the aim marker direction relative to the target. In the authors' file both have the same sign as the rotation (pert) although they compensate for it, and the paper calls +30 a counter-clockwise rotation, so they were read as clockwise-positive and negated here: rotation is the authors' pert (+30 counter-clockwise) and hand_angle = -head_dir_peak_sp0, giving counter-clockwise-positive angles with compensation opposite to the rotation. This sign is inferred, not stated in the deposit.",
        "target_angle is 90 + the authors' tgt_angle (their 0 is the 12 o'clock direction and positive is counter-clockwise: in tgt_file the first target, tgt_angle -45, lies up and to the right). The authors' per-trial target angle varies by a few degrees around the nominal direction and is kept as stored.",
        "break_before is True on the first reach of each of the authors' tgt_file blocks (rest breaks of at least one minute: every 20 to 30 training reaches and every 57 generalization reaches). time_s is the trial start time, rebuilt from start_and_end_times and t_last_trial as in the authors' ITI code, counted from the first trial.",
        "aim reports exist on the 16 extra baseline, training and extended training reaches and on generalization reaches of the aim-report kind (tgt_file column 5 equal to 0); other generalization reaches asked participants to aim their hand through the target (instr_strategy remove, no aim_report). Reaches without a cursor have fb_visible none. Trials the authors removed as outliers (about 0.9%) are not marked: their rule needs the whole sample and the file holds no flag."],
)
stdlib.release(ID, "reach90", rows, meta, checks)
