"""Al-Fawakhiri & McDougle (2024), J Exp Psychol Gen 153(8):2160-2173: independent influences of movement distance and visual
distance on Fitts' law. Reaches to targets at a short or a long distance in a visuomotor gain-adaptation design: in blocks of
adaptation trials (with feedback) one distance is shown with a different hand-to-cursor extent mapping, followed by test
trials without feedback. Five experiments (104 participants), one row per trial, in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/AlFawakhiri2024_JExpPsycholGen.py
Raw files (untrusted data, read only) in raw_extra/AlFawakhiri2024_JExpPsycholGen/, from
https://github.com/ilestz/FittsLaw_MSI (no license file in the repository):
  gain-online_e1.json  27 Prolific participants, continuous cursor feedback          = paper Experiment 2
  gain-online_e3.json  27 Prolific participants, endpoint feedback only               = paper Experiment 3
  gain-online_e2.json  30 Prolific participants, endpoint feedback, difficulty clamp  = paper Experiment 4b
  inlab/GainAdaptSub001-010.json.gz  10 lab participants, cursor feedback (Octave/Psychtoolbox)   = Experiment 1
  inlab/GainAdaptSub101-110.json.gz  10 lab participants, endpoint feedback, difficulty clamp      = Experiment 4a
The mapping of file to experiment was made from the preprint's participant counts, success rates and the authors' analysis
scripts (analyzeGainAdapt*.Rmd, read as text, never run).
Read-only: nothing of the deposit is executed.
"""
import gzip
import json
import math
import statistics as st
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "AlFawakhiri2024_JExpPsycholGen"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
PREPRINT = "bioRxiv 2023.09.20.558709 (preprint of this paper), Results"


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


def tf(x):
    return "True" if x else "False"


def wrap(a):
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


rows = []
person_key = {}       # (pseudonymous) keys used only inside this run; stdlib replaces them with codes
overlap = Counter()


def base(experiment, subj, pers, setting, id_type, link, device, hand, blk, trial, phase, kind, brk):
    r = dict.fromkeys(names, "")
    r.update(dataset=ID, study=ID, experiment=experiment, subject=subj, person=pers, id_type=id_type, person_link=link,
             prior_exposure="unknown", subject_excluded_by_authors="False", setting=setting, population="healthy",
             device=device, hand_used=hand, trial_kind=kind, phase=phase, block=blk, trial=str(trial), session="0",
             break_before=tf(brk), excluded_by_authors="False")
    return r


# ---------------------------------------------------------------- online experiments
ONLINE = [("e1", "Exp2_online_cursor", "online", 150, 300), ("e3", "Exp3_online_endpoint", "endpoint", 150, 300),
          ("e2", "Exp4b_online_endpoint_clamp", "endpoint", 150, 300)]
DEVICE = {"mouse": "mouse", "trackpad": "trackpad", "trackball": "other"}
n_by_exp = {}
for fid, exp, fbv, _, _ in ONLINE:
    d = json.load(open(RAW / f"gain-online_{fid}.json"))
    n_by_exp[exp] = len(d)
    maxvis = 300.0
    for s, sub in enumerate(d):
        cfg = sub["config"]
        pid = "online:" + str(cfg["id"])
        overlap[pid] += 1
        trials = sorted([(t["trial_number"], ph, t) for ph in ("adapt", "test") for t in sub["data"][ph]], key=lambda z: z[0])
        blen = 21 if fid == "e1" else 27
        for j, (tn, ph, t) in enumerate(trials):
            training = ph == "adapt"
            tid = t["trial_id"]
            vis, mov = float(t["vis_dist"]), float(t["move_dist"])
            manip = tid in (3, 4)
            r = base(exp, f"{ID}_{exp}_{s}", pid, "online", "prolific", "verified", DEVICE[cfg["device"]], cfg["hand"],
                     f"{'adapt' if training else 'test'}{tn // blen}", j, "learning" if training else "probe", "reach", j == 0)
            ang = float(t["target_angle"])
            r.update(trial_orig=str(tn), target_angle=num(ang), target_id=str(int(ang) * 1000 + int(vis)),
                     target_size_deg=num(math.degrees(math.atan(t["target_size_radius"] / vis))),
                     target_distance_norm=num(vis / maxvis), target_visible_during_move="True", target_event="none",
                     target_jump="0", probe_kind="none", instr_goal="cursor_to_target", instr_strategy="unspecified",
                     instr_disclosure="none", fb_cued="False", hand_measure="endpoint", hand_extent="",
                     hit=tf(t["hit"]).replace("True", "1").replace("False", "0"),
                     reward=("1" if t["hit"] else "0") if training else "",
                     score=num(t["points"]) if training else "",
                     speed_warning="1" if t["punish_types"] & 6 else "0", other_warning="1" if t["punish_types"] & ~6 else "0",
                     outcome_channels="cursor|hit_color|points" if training and fbv == "online" else (
                         "hit_color|points" if training else ""))
            if training:
                r.update(fb_visible=fbv, fb_form="cursor", online_cursor="none" if fbv == "endpoint" else (
                    "perturbed" if fid == "e1" and manip else "veridical"))
                if fid == "e1":
                    r.update(fb_mode="gain" if manip else "veridical", gain=num(vis / mov), fb_endpoint_transform_only="False")
                elif fid == "e3":
                    r.update(fb_mode="gain" if manip else "veridical", gain=num(vis / mov))
                else:
                    r.update(fb_mode="other" if manip else "veridical")
            else:
                r.update(fb_visible="none", fb_mode="none", fb_form="none", online_cursor="none")
            if t.get("reaction_time") is not None and t["movement_data"]:
                last = t["movement_data"][-1]
                x, y = last["raw_x"], last["raw_y"]
                ha = wrap(math.degrees(math.atan2(y, x)) - ang)
                ext = math.hypot(x, y)
                r.update(hand_angle=num(ha), hand_extent=num(ext / vis), rt_ms=num(t["reaction_time"]),
                         mt_ms=num(t["reach_time"]))
                if training and fid == "e1":
                    cx, cy = last["cursor_x"], last["cursor_y"]
                    r.update(cursor_error=num(wrap(math.degrees(math.atan2(cy, cx)) - ang)),
                             cursor_extent_error=num(math.hypot(cx, cy) / vis - 1.0))
            rows.append(r)

# ---------------------------------------------------------------- in-lab experiments
LAB = [("Exp1_lab_cursor", [f"{i:03d}" for i in range(1, 11)], "online", 106.0),
       ("Exp4a_lab_endpoint_clamp", [f"{i}" for i in range(101, 111)], "endpoint", 106.0)]
LABELS = {1: ("learning", "reach"), 2: ("probe", "reach"), 3: ("other", "other"), 4: ("other", "other"),
          5: ("other", "other"), 6: ("practice", "reach")}
for exp, files, fbv, maxvis in LAB:
    n_by_exp[exp] = len(files)
    for s, fn in enumerate(files):
        dd = json.load(gzip.open(RAW / f"GainAdaptSub{fn}.json.gz"))
        T, B = dd["trials"], dd["block"]
        n = len(T["label"])
        radius = B["target"]["size"] / 2.0
        pid = f"lab:{fn}"
        blk, prev_lab, next_break = -1, None, True
        for i in range(n):
            lab = T["label"][i]
            phase, kind = LABELS[lab]
            if lab == 1 and prev_lab != 1:
                blk += 1
            tid = T["trial_id"][i]
            vis, mov = float(T["vis_dist"][i]), float(T["move_dist"][i])
            ang = float(T["target"][i]["angle"])
            gain = float(T["gain"][i])
            training = lab in (1, 6)
            r = base(exp, f"{ID}_{exp}_{s}", pid, "lab", "lab", "assumed_unique", "unknown", "",
                     {1: f"adapt{blk}", 2: f"test{blk}", 3: "instruction", 4: "instruction", 5: "instruction", 6: "tutorial"}[lab],
                     i, phase, kind, next_break or i == 0)
            next_break = lab in (3, 4, 5)
            prev_lab = lab
            if kind == "other":
                r["excluded_by_authors"] = "True"
                rows.append(r)
                continue
            manip = tid in (3, 4)
            sc = T["score_data"]
            r.update(trial_orig=str(i + 1), target_angle=num(ang), target_id=str(int(ang) * 1000 + int(vis)),
                     target_size_deg=num(math.degrees(math.atan(radius / vis))), target_distance_norm=num(vis / maxvis),
                     target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
                     instr_goal="cursor_to_target", instr_strategy="unspecified", instr_disclosure="none", fb_cued="False",
                     hand_measure="endpoint", rt_ms=num(1000 * sc["reaction_time"][i]), mt_ms=num(1000 * sc["movement_time"][i]),
                     hit="1" if sc["hit"][i] else "0", reward=("1" if sc["hit"][i] else "0") if training else "",
                     score=num(sc["points"][i]) if training else "", speed_warning="1" if T["too_slow"][i] else "0",
                     other_warning="1" if T["was_restarted"][i] else "0")
            # excluded: the authors' scripts use only the adaptation and test trials (tutorial rows left out)
            if lab == 6:
                r["excluded_by_authors"] = "True"
            # endpoint: last sample of the reach frames (start_state 2), as in the authors' analysis
            f = T["frames"][i]
            xs = []
            for st_, ev in zip(f["start_state"], f["input_events"]):
                if st_ == 2:
                    ex = ev["x"] if isinstance(ev["x"], list) else [ev["x"]]
                    ey = ev["y"] if isinstance(ev["y"], list) else [ev["y"]]
                    xs += [(a, b) for a, b in zip(ex, ey) if a is not None and b is not None]
            if xs:
                x, y = xs[-1]
                ha = wrap(math.degrees(math.atan2(y, x)) - ang)
                ext = math.hypot(x, y)
                r.update(hand_angle=num(ha), hand_extent=num(ext / vis))
            else:
                ha = ext = None
            fbon = (T["online_feedback"][i] or T["endpoint_feedback"][i])
            if fbon and training:
                if fbv == "online":
                    r.update(fb_visible="online", fb_mode="gain" if manip else "veridical", gain=num(gain), fb_form="cursor",
                             online_cursor="perturbed" if manip else "veridical", fb_endpoint_transform_only="False",
                             fb_duration_ms=num(1000 * B["feedback_duration"]), outcome_channels="cursor|hit_color|points")
                    if ha is not None:
                        r.update(cursor_error=num(ha), cursor_extent_error=num(gain * ext / vis - 1.0))
                else:
                    r.update(fb_visible="endpoint", fb_mode="other" if manip else "veridical", fb_form="cursor",
                             online_cursor="none", fb_duration_ms=num(1000 * B["feedback_duration"]),
                             outcome_channels="cursor|hit_color|points")
            else:
                r.update(fb_visible="none", fb_mode="none", fb_form="none", online_cursor="none")
            rows.append(r)

# ---------------------------------------------------------------- checks
reach = [r for r in rows if r["trial_kind"] == "reach"]
by_exp = OrderedDict()
for r in rows:
    by_exp.setdefault(r["experiment"], OrderedDict()).setdefault(r["subject"], []).append(r)
checks.internal("every experiment has the number of participants of the deposit", {e: len(v) for e, v in by_exp.items()} == n_by_exp,
                str({e: len(v) for e, v in by_exp.items()}))
paper_n = {"Exp1_lab_cursor": 10, "Exp2_online_cursor": 27, "Exp3_online_endpoint": 27, "Exp4a_lab_endpoint_clamp": 10,
           "Exp4b_online_endpoint_clamp": 30}
for e, n in paper_n.items():
    checks.published(f"{e}: participants", len(by_exp[e]), n, 0, PREPRINT + ", Participants")
checks.internal("rows per participant: online 336 / 432 / 432; lab 480 and 699 (including instruction rows)",
                all(len(v) == {"Exp2_online_cursor": 336, "Exp3_online_endpoint": 432, "Exp4b_online_endpoint_clamp": 432,
                               "Exp1_lab_cursor": 480, "Exp4a_lab_endpoint_clamp": 699}[e] for e, d in by_exp.items()
                    for v in d.values()))
checks.internal("hand_angle within (-180, 180]", all(-180 < float(r["hand_angle"]) <= 180 for r in rows if r["hand_angle"]))
print("punish_types check: speed_warning share", st.mean(float(r["speed_warning"]) for r in reach if r["speed_warning"] != ""))
print("online ids appearing in more than one online file:", sum(1 for k, v in overlap.items() if v > 1), "of", len(overlap))


def rate(rs):
    return st.mean(float(r["hit"]) for r in rs)


def person_rate(exp, select):
    out = []
    for v in by_exp[exp].values():
        sel = [r for r in v if r["phase"] == "learning" and r["trial_kind"] == "reach" and r["excluded_by_authors"] == "False"
               and select(r)]
        out.append(rate(sel))
    return st.mean(out)


def vis_long(r):
    return float(r["target_distance_norm"]) > 0.75 if r["experiment"].startswith(("Exp2", "Exp3", "Exp4b")) else float(
        r["target_distance_norm"]) > 0.75


def manip(r):
    return r["fb_mode"] in ("gain", "other")


def mov_long(r):  # movement distance = visual distance for unmanipulated; swapped for manipulated
    return vis_long(r) != manip(r)


for e, tot, sec in (("Exp3_online_endpoint", 0.414, "Experiment 3, Results, Success rate"),
                    ("Exp1_lab_cursor", 0.910, "Experiment 1, Results, Success rate")):
    checks.published(f"{e}: overall success rate in the adaptation trials", person_rate(e, lambda r: True), tot, 0.0005,
                     PREPRINT + ", " + sec)
checks.published("Exp1_lab_cursor: success rate on unmanipulated trials",
                 100 * person_rate("Exp1_lab_cursor", lambda r: not manip(r)), 93.7, 0.05, PREPRINT + ", Experiment 1, Results")
checks.published("Exp1_lab_cursor: success rate on manipulated trials",
                 100 * person_rate("Exp1_lab_cursor", manip), 88.4, 0.05, PREPRINT + ", Experiment 1, Results")
checks.published("Exp3_online_endpoint: success rate lower on long visual distance trials by (points)",
                 100 * (person_rate("Exp3_online_endpoint", lambda r: not vis_long(r)) -
                        person_rate("Exp3_online_endpoint", vis_long)), 32.6, 0.05, PREPRINT + ", Experiment 3, Results")
checks.published("Exp3_online_endpoint: success rate lower on manipulated trials by (points)",
                 100 * (person_rate("Exp3_online_endpoint", lambda r: not manip(r)) - person_rate("Exp3_online_endpoint", manip)),
                 7.7, 0.05, PREPRINT + ", Experiment 3, Results")
checks.published("Exp4a_lab_endpoint_clamp: success rate lower with the longer movement by (points)",
                 100 * (person_rate("Exp4a_lab_endpoint_clamp", lambda r: not mov_long(r)) -
                        person_rate("Exp4a_lab_endpoint_clamp", mov_long)), 27, 0.5, PREPRINT + ", Experiment 4a, Results")
checks.published("Exp4b_online_endpoint_clamp: success rate lower with the longer movement by (points)",
                 100 * (person_rate("Exp4b_online_endpoint_clamp", lambda r: not mov_long(r)) -
                        person_rate("Exp4b_online_endpoint_clamp", mov_long)), 30, 0.5, PREPRINT + ", Experiment 4b, Results")

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="McDougle (Yale)",
    data_url="https://github.com/ilestz/FittsLaw_MSI",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format with participant codes replacing the Prolific identifiers and lab file numbers; "
            "the authors' browser (position samples) and Psychtoolbox frame logs were reduced to one row per trial "
            "(endpoint direction and extent); browser, battery, institution and date fields were not carried over.",
    unit_note="trial of one participant sequence; lengths are in the authors' units (pixels online, millimetres in the lab) "
              "before being divided by the target distance",
    how=[
        "One row is one trial: 336 (Experiment 2), 432 (Experiments 3 and 4b) per online participant, and 480 (Experiment 1) "
        "or 699 (Experiment 4a) per lab participant; the lab rows include the instruction screens (trial_kind other, "
        "excluded_by_authors True, no values) and the 3 tutorial reaches of Experiment 4a (phase practice). "
        "Adaptation trials (phase learning) alternate with 6 test trials without feedback (phase probe) in mini-blocks of one "
        "trial type (short or long target, movement the same as or different from the visual distance).",
        "The file-to-experiment mapping was derived from the preprint's counts and success rates: gain-online_e1 is the "
        "paper's Experiment 2, e3 is Experiment 3 and e2 is Experiment 4b; the lab files 001-010 are Experiment 1 and 101-110 "
        "Experiment 4a.",
        "hand_angle is the direction of the last recorded position relative to the target, counter-clockwise positive "
        "(atan2 of the y-up hand position minus target_angle; the authors use atan of the target-rotated position, which is "
        "the same whenever the hand ended in front of the start); hand_measure is endpoint. hand_extent is the distance of "
        "that last position from the start divided by the displayed target distance (vis_dist: 150 or 300 px online, 53 or "
        "106 mm in the lab); target_distance_norm is the displayed distance over the largest in the experiment.",
        "fb_mode gain with the gain column (cursor extent = gain x hand extent, gain = vis_dist / move_dist = 0.5 or 2) "
        "is recorded in the files for Experiment 2 (the cursor samples show it) and Experiment 1 (gain field). For "
        "Experiment 3 the files do not log the feedback position (cursor equals hand in every sample); the gain label is the "
        "authors' own model variable (gain = vis_dist / move_dist) and is an inference. In the difficulty-clamp experiments "
        "(4a, 4b) the manipulated trials are labelled fb_mode other; in 4a the deposit's gain field holds the shift "
        "(vis_dist minus move_dist: -53 or +53 mm) and the direction of the shift is not documented, so translation_xy is "
        "left empty. cursor_error and cursor_extent_error are filled only where the cursor position follows from a stated "
        "gain.",
        "Outcome columns: hit is the authors' hit flag (also on test trials, where no feedback was given); score is the "
        "points awarded on feedback trials. speed_warning marks a movement or reaction-time penalty (punish_types bits 2 and "
        "4 online; too_slow in the lab) and other_warning any other penalty bit online, or a restarted trial in the lab; the "
        "bit meanings were inferred from the values and are not documented. Trials without a recorded reaction time (about 50 "
        "online) have empty movement columns, as in the authors' analysis (NA).",
    ],
    differences=[
        "Two preprint numbers for the online Experiment 2 did not reproduce and are not used as checks: the overall success "
        "rate in the adaptation trials (96.8% printed; 96.88% from the rows) and the drop in success on manipulated trials "
        "(2.1 points printed; 1.73 from the rows). The deposit has 15 adaptation trials per block where the preprint and "
        "the authors' script describe 12, so the paper probably analysed a subset the deposit does not identify.",
        "The deposit's online Experiment 2 file has 15 adaptation and 6 test trials per mini-block (16 blocks, 336 trials); "
        "the preprint and the authors' script describe 12 adaptation trials per block. The rows are kept as shared.",
        "The 3 tutorial trials of the online Experiments 3 and 4b are not in the files (trial_orig starts at 3).",
        "The participant counts and success rates were checked against the preprint (the published JEP:General text was not "
        "reachable); the paper's other statistics (mixed models, BIC) were not recomputed because they depend on the "
        "authors' trial exclusions, which the deposit does not list. Participant ages and sexes are not in the deposit.",
    ],
)
for c in checks.items:
    print("CHK", "ok " if c["ok"] else "BAD", c["what"], round(c.get("computed", 0), 4), c.get("reported", ""))
stdlib.release(ID, "reach90", rows, meta, checks)
