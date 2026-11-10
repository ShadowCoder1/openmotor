"""Franken, Hartsuiker, Johansson, Hall, Wartenberg & Lind (2019), J Acoust Soc Am 146(6):4108-4121: responses to unexpected
200-ms pitch shifts of auditory feedback (+-25, +-100 cents and null shifts) while vocalizing [e], heard through three kinds of
headphones with different passive sound attenuation (Experiment 2; Experiment 1 measured the headphones with a dummy
microphone and is not movement data).

Raw files, raw_extra/Franken2019_JASA/ (OSF vm84u, fetched with tools/std/pc.py fetch):
  oppfoll.csv   subject, headphones (1 BD, 2 CB, 3 ER), perturbation (cents), trialNumber, pertDirection, pertMag, type_slope
                one row per pitch-shift epoch the authors kept; type_slope 1 opposing, -1 following, 0 null shift, NA unclassified
  peaks.csv     subject, headphones, perturbation, ..., respMag, respPeakLat, respMagOpp, NtrialsOpp, respMagFoll, NtrialsFoll
                one row per participant x headphones x shift condition (the average response contour's peak)
  exp2_script.R (authors' code, read as text): labels, the five participants dropped before the statistics, sign conventions
  grandAverages.csv (group-average contours) is not carried.
Run from the site folder: python3 -I tools/std/convert/Franken2019_JASA.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Franken2019_JASA"
PAPER_SRC = "Franken et al. 2019, J Acoust Soc Am 146(6):4108-4121 (accepted manuscript, OSF vecw6)"
HP = {1: "BeyerDynamic DT 770 Pro (BD)", 2: "custom-built headphones (CB)", 3: "Etymotic ER-3C insert earphones (ER)"}
# the five participants the authors' exp2_script.R removes before the statistics (3 with badly fitting ER earphones, 2 with
# too few epochs: paper, Experiment 2, Analysis)
DROPPED = ["S3", "S6", "S13", "S23", "S42"]
DEVICE = "DPA 4088-B head-mounted microphone, Eventide Eclipse pitch shifter, Aphex HeadPod 4 amplifier, three headphone types"


def build(checks):
    raw = stdlib.raw_dir(ID)
    o = pd.read_csv(raw / "oppfoll.csv")
    p = pd.read_csv(raw / "peaks.csv")
    key = ["subject", "headphones", "perturbation"]
    assert not p.duplicated(key).any()
    o = o.merge(p.drop(columns=["pertDirection", "pertMag"]), on=key, how="left", validate="m:1")
    o["pid"] = o.subject.str[1:].astype(int)
    o = o.sort_values(["pid", "trialNumber"], kind="stable").reset_index(drop=True)
    cell_n = o.groupby(key).trialNumber.transform("size")
    o["cell_n"] = cell_n
    nz = lambda x: "" if pd.isna(x) else float(x)
    ni = lambda x: "" if pd.isna(x) else int(x)
    rows = []
    for k, r in enumerate(o.itertuples(index=False)):
        dropped = r.subject in DROPPED
        shift = int(r.perturbation)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Experiment 2", "group": "", "subject": r.subject,
            "subject_excluded_by_authors": dropped, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE, "session": 1, "block": HP[int(r.headphones)], "trial": int(r.trialNumber),
            "phase": "test", "time_s": "", "task": "speech", "unit": "utterance",
            "condition": "null shift" if shift == 0 else f"{shift:+d} cents",
            "pert_type": "pitch_shift" if shift else "none", "pert_value": shift, "pert_unit": "cents",
            "feedback": "altered" if shift else "normal", "excluded_by_authors": bool(dropped or r.cell_n < 20),
            "stimulus": "sustained vowel [e] (Dutch 'EE')", "segment": "pitch-shift epoch",
            "measure_window": "F0 contour from 200 ms before to 700 ms after the 200-ms pitch shift onset (cents re the 100 ms before onset)",
            "masking_noise": "none", "feedback_latency_ms": 14.3,
            "authors_trial_number": int(r.trialNumber),
            "pitch_response_type": {1.0: "opposing", -1.0: "following"}.get(r.type_slope, ""),
            "pitch_response_peak_cents": nz(r.respMag) if shift else "",
            "pitch_response_peak_latency_s": nz(r.respPeakLat), "pitch_response_opposing_cents": nz(r.respMagOpp),
            "n_epochs_opposing": ni(r.NtrialsOpp), "pitch_response_following_cents": nz(r.respMagFoll),
            "n_epochs_following": ni(r.NtrialsFoll),
        })
    return rows, o, p


def internal_checks(a, o, p, checks):
    checks.internal("30337 epochs from 49 participants, as in the authors' file", len(a) == 30337 and a.subject.nunique() == 49)
    checks.internal("trial numbers are unique within each participant and within 1-720 (3 blocks x 80 trials x 3 shifts)",
                    bool(not a.duplicated(["subject", "trial"]).any() and a.trial.between(1, 720).all()))
    checks.internal("each participant heard each headphone type in one block of consecutive trial numbers",
                    bool(all(g.groupby("block").trial.apply(lambda s: s.max() - s.min() < 240).all() for _, g in a.groupby("subject"))))
    checks.internal("at most 240 epochs per participant and headphone, with at most 40 per shift size and 80 null",
                    bool(a.groupby(["subject", "block"]).size().max() <= 240
                         and a.groupby(["subject", "block", "pert_value"]).size().max() <= 80
                         and a[a.pert_value != 0].groupby(["subject", "block", "pert_value"]).size().max() <= 41))
    checks.internal("shift sizes are 0, +-25, +-100 cents; null shifts have no response class",
                    set(a.pert_value) == {-100, -25, 0, 25, 100} and (a[a.pert_value == 0].pitch_response_type == "").all())
    checks.internal("peaks.csv: 42 participants x 3 headphones x 5 conditions = 630 rows, all matched to epoch rows",
                    len(p) == 630 and p.subject.nunique() == 42 and o.merge(p, on=["subject", "headphones", "perturbation"]).shape[0] > 0)
    checks.internal("response sign: mean peak is positive for downward shifts and negative for upward shifts (opposing = against the shift)",
                    bool(p[p.pertDirection == -1].respMag.mean() > 0 > p[p.pertDirection == 1].respMag.mean()))


def published_checks(a, checks):
    ep = a[a.pert_value != 0]
    cls = ep.pitch_response_type
    checks.published("participants in Experiment 2 (before exclusions)", a.subject.nunique(), 49, 0, f"{PAPER_SRC}, Experiment 2, Participants")
    checks.published("analysed epochs (null shifts excluded)", int(cls.isin(["opposing", "following"]).sum()), 19857, 0,
                     f"{PAPER_SRC}, Results, Proportion of opposing responses")
    checks.published("epochs classified as opposing", int((cls == "opposing").sum()), 13377, 0, f"{PAPER_SRC}, Results, Proportion of opposing responses")
    checks.published("epochs classified as following", int((cls == "following").sum()), 6480, 0, f"{PAPER_SRC}, Results, Proportion of opposing responses")
    checks.published("share of epochs classified as opposing (%)", 100 * (cls == "opposing").sum() / cls.isin(["opposing", "following"]).sum(),
                     67, 0.5, f"{PAPER_SRC}, Results, Proportion of opposing responses")
    kept = a[~a.subject_excluded_by_authors]
    checks.published("participants left after the 3 + 2 exclusions", kept.subject.nunique(), 44, 0,
                     f"{PAPER_SRC}, Experiment 2, Analysis (49 - 3 - 2)")
    cell = kept.groupby(["subject", "block", "pert_value"]).size()
    cell = cell[cell >= 20]
    checks.published("average pitch contours (conditions with at least 20 epochs, null shifts included)", len(cell), 656, 0,
                     f"{PAPER_SRC}, Experiment 2, Analysis")
    nonnull = cell[cell.index.get_level_values(2) != 0]
    checks.published("mean epochs per non-null contour", nonnull.mean(), 35.2, 0.05, f"{PAPER_SRC}, Experiment 2, Analysis")
    checks.published("minimum epochs per non-null contour", nonnull.min(), 20, 0, f"{PAPER_SRC}, Experiment 2, Analysis")
    nul = cell[cell.index.get_level_values(2) == 0]
    checks.published("mean epochs per null-shift contour", nul.mean(), 70.2, 0.05, f"{PAPER_SRC}, Experiment 2, Analysis")
    checks.published("minimum epochs per null-shift contour", nul.min(), 33, 0, f"{PAPER_SRC}, Experiment 2, Analysis")
    return nonnull.max(), kept


def main():
    checks = stdlib.Checks()
    rows, o, p = build(checks)
    a = pd.DataFrame(rows)
    internal_checks(a, o, p, checks)
    mx, kept = published_checks(a, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    print("max epochs per non-null contour:", mx)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Merged the authors' per-epoch response-type file with their per-condition peak-response file, ordered each "
                 "participant's epochs by trial number, added the pitch shift of each epoch from the authors' labels, dropped the "
                 "group-average contours and the Experiment 1 headphone measurements, and replaced participant numbers with codes."),
        how=[
            "One row is one pitch-shift epoch (one 200-ms shift of the feedback, or a null shift) in Experiment 2; each vocalization of "
            "the vowel [e] held three shifts. Epochs the authors discarded (tracking errors or mistimed shifts) are absent, so trial "
            "(the authors' trialNumber, 1-720 over the whole session) has gaps. The block column names the headphones worn.",
            "pert_value is the shift in cents (positive = pitch raised). pitch_response_type is the authors' per-epoch class from the "
            "slope of the produced F0 contour 60-260 ms after shift onset (opposing = against the shift, following = with it); it is empty "
            "for null shifts and for 343 epochs the authors left unclassified.",
            "pitch_response_peak_cents, the latency and the opposing and following peaks are the authors' values for the average response "
            "contour of that participant, headphones and shift size (peaks.csv), repeated on each epoch of that condition; a positive value is a "
            "rise in produced pitch, so opposing responses are positive for downward shifts. peaks.csv covers 42 of the 49 participants; "
            "its null-shift rows hold no response (value 0).",
            "The five participants that the authors' script removes before its statistics (S-numbers withheld here) are marked "
            "subject_excluded_by_authors; conditions with fewer than 20 epochs, which the paper rejects, are marked excluded_by_authors.",
        ],
        unit_note="pitch-shift epoch of one participant",
        differences=[
            "The paper reports 19857 analysed epochs (13377 opposing, 6480 following); these counts are reproduced only by the file as "
            "shared, which still holds the five participants whose data the authors drop before the statistics. After dropping them the "
            "file has 18401 classified epochs.",
            "Epochs discarded per participant: the paper reports a mean of 77.3 (10.7% of 720), a maximum of 264 and four participants above 200; "
            "the shared file gives 100.9 (14.0%), 399 and seven over all 49 participants, and 88.1, 325 and five over the 44 kept. Not used as checks.",
            f"The largest number of epochs behind a non-null contour is {int(mx)} in the shared file; the paper states a maximum of 40 (ranging from 20).",
            "peaks.csv holds 42 participants and 15 contours each (630 rows), not the 44 participants and 656 contours described in the paper; "
            "the epoch file reproduces the 656.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
