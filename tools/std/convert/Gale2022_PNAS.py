"""Gale et al. (2022), PNAS: 45 deg clockwise visuomotor rotation with an MRI-compatible isometric force sensor, 32 participants,
two sessions on consecutive days, one row per trial, in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Gale2022_PNAS.py
Raw files (untrusted data, read only) in raw_extra/Gale2022_PNAS/, from https://openneuro.org/datasets/ds004021 (CC0): only the
BIDS behavioural files sub-XX_ses-YY_task-{rotation,washout}_events.tsv, participants.tsv, README, dataset_description.json
(the 100+ GB of fMRI images are not needed). Files are stored here as <path with / replaced by __>.

Sign convention. The README defines the events columns: response_time (s, target onset to movement start) and response_error
('angle error between the location of the target and the final cursor position on the circular ring', deg). Its sign is not
stated. The paper (Results, 'Participants launched the cursor...') says the cursor was rotated 45 deg CLOCKWISE relative to
the force direction, and in the files the mean error of the first 10 rotation trials is about +35 deg, so response_error is
positive for clockwise cursor errors. OpenMotor is counter-clockwise positive, so cursor_error = -response_error, rotation =
-45, and hand_angle = cursor_error - rotation (cursor = hand + rotation). The events files do not say which of the eight target
locations was used, so target_angle is empty; hand_angle is relative to the target regardless.
"""
import csv
import json
import statistics as st
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gale2022_PNAS"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()


def tsv(name):
    return list(csv.DictReader(open(RAW / name, newline="", encoding="utf-8"), delimiter="\t"))


part = [r["participant_id"] for r in tsv("participants.tsv")]
names = [c[1] for c in stdlib.reach90_columns()]
rows = []
by = OrderedDict()
for i, p in enumerate(part, start=1):
    seq = 0
    for ses in (1, 2):
        for task in ("rotation", "washout"):
            ev = tsv(f"{p}__ses-{ses:02d}__func__{p}_ses-{ses:02d}_task-{task}_events.tsv")
            for e in ev:
                lab = e["trial_type"]
                rot = -45.0 if lab == "rotation" else 0.0
                miss = e["response_error"] in ("n/a", "")
                cerr = None if miss else -float(e["response_error"])
                hand = None if miss else cerr - rot
                row = dict.fromkeys(names, "")
                row.update(
                    dataset=ID, experiment=f"{ID}_Main", study=ID, subject=p, person=p, id_type="lab",
                    person_link="assumed_unique", prior_exposure="unknown", subject_excluded_by_authors="False",
                    handedness="right", device="other", setting="lab", population="healthy",
                    trial=str(seq), trial_orig=e["trial_number"], session=str(ses - 1), block=lab,
                    phase={"baseline": "baseline", "rotation": "learning", "washout": "washout"}[lab], trial_kind="reach",
                    target_id="", target_visible_during_move="True", target_event="none", target_jump="0",
                    probe_kind="none", instr_goal="cursor_to_target", instr_strategy="unspecified", instr_disclosure="none",
                    instr_report="none", fb_cued="True", hand_used="right", hand_measure="other",
                    rt_ms=("%.8g" % (1000 * float(e["response_time"]))) if e["response_time"] not in ("n/a", "") else "",
                    excluded_by_authors="False", fb_visible="endpoint", fb_mode="rotated" if rot else "veridical",
                    fb_form="cursor", online_cursor="perturbed" if rot else "veridical", fb_endpoint_transform_only="False",
                    rotation="%.8g" % rot if rot else "0", outcome_channels="cursor",
                )
                if not miss:
                    row.update(hand_angle="%.8g" % hand, hand_angle_fb="%.8g" % hand, cursor_error="%.8g" % cerr)
                rows.append(row)
                by.setdefault(p, []).append((ses, lab, e, miss, cerr, hand))
                seq += 1
# fb_visible: the cursor is launched by a brief force pulse, so it is shown at the end ring position (endpoint)

# ---- checks
sess_rot = {p: [v for v in vs if v[0] == 1 and v[1] in ("baseline", "rotation")] for p, vs in by.items()}
checks.published("participants analysed", len(by), 32, 0, "Results (N = 32 perform a visuomotor rotation task)")
checks.published("trials per participant in the scan with the rotation (120 baseline + 320 rotation)",
                 st.mean(len(v) for v in sess_rot.values()), 440, 0, "Materials and Methods ('a total of 440 trials')")
checks.published("baseline trials per participant", st.mean(sum(1 for x in v if x[1] == "baseline") for v in sess_rot.values()),
                 120, 0, "Materials and Methods (120 trials with no rotation)")
checks.published("rotation trials per participant", st.mean(sum(1 for x in v if x[1] == "rotation") for v in sess_rot.values()),
                 320, 0, "Materials and Methods (320 trials with 45 deg rotation)")
checks.internal("every participant has 2 x (440 + 120) rows", all(len(v) == 1120 for v in by.values()))
for ses in (1, 2):
    base = [x[4] for v in by.values() for x in v if x[0] == ses and x[1] == "baseline" and not x[3]]
    first = [x[4] for v in by.values() for x in v if x[0] == ses and x[1] == "rotation" and int(x[2]["trial_number"]) <= 130 and not x[3]]
    late = [x[5] for v in by.values() for x in v if x[0] == ses and x[1] == "rotation" and int(x[2]["trial_number"]) > 400 and not x[3]]
    checks.internal(f"session {ses}: baseline cursor error near 0 (|mean| < 5 deg), first 10 rotation trials cursor error"
                    " counter-clockwise (< -15 deg, i.e. clockwise displaced by the rotation), late hand angle compensating"
                    " (> +30 deg)", abs(st.mean(base)) < 5 and st.mean(first) < -15 and st.mean(late) > 30,
                    f"{st.mean(base):.1f}, {st.mean(first):.1f}, {st.mean(late):.1f}")
checks.internal("missing responses (n/a) are kept as empty hand angle, not dropped",
                sum(1 for v in by.values() for x in v if x[3]) == sum(1 for r in rows if r["hand_angle"] == ""))
checks.internal("rows = 32 x 1120", len(rows) == 32 * 1120)

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Gallivan & Flanagan (Queen's University)", data_url="https://openneuro.org/datasets/ds004021",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="converted to the OpenMotor format, participant codes replaced, response_error sign flipped to counter-clockwise "
            "positive, hand angle derived as cursor error minus the rotation.",
    unit_note="trial of one participant sequence",
    how=[
        "One row is one trial: per participant 2 sessions (consecutive days) x (120 baseline + 320 trials with the cursor rotated "
        "45 deg clockwise, then 120 washout trials); trial counts all sessions in order, session is 0 or 1.",
        "The authors shared the angle between the target and the final cursor position (response_error) and the response time; "
        "its sign was flipped so that cursor_error is counter-clockwise positive (rotation -45 is clockwise), and hand_angle = "
        "cursor_error - rotation.",
        "The files do not say which of the eight target locations was cued, so target_angle is empty; hand_angle is relative to the "
        "target. Trials with no response ('n/a') are kept with empty angles. The isometric force sensor means hand_measure is other. "
        "The fMRI images are not included. No age or sex per participant was shared.",
    ],
    differences=["The paper analyses 32 of 40 recruited participants (8 excluded for head motion or missing scans); the deposit holds "
                 "only those 32, so no participant is flagged.",
                 "The paper does not say whether it used the first or both sessions; both are converted."],
)
stdlib.release(ID, "reach90", rows, meta, checks)
