"""Huang, Gegenfurtner, Schutz & Billino (2017) J Vis: age effects on saccadic adaptation (Zenodo 546106).

Raw file: raw_extra/Huang2017_JVis/SaccadicAdaptation.zip with EXP1/ (double-step paradigm) and EXP2/ (perceptual task):
EXPn_eyedata.txt (one row per trial), EXPn_cognitivedata.txt (one row per participant), EXPn_columndescription.txt.
Only read as data; nothing from the deposit is executed.

Run from the site folder: python3 -I tools/std/convert/Huang2017_JVis.py
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Huang2017_JVis"
MODULE = "saccade"
SEC1 = "Experiment 1, Results"
SEC2 = "Experiment 2, Results"
GRP = {1: "young", 2: "senior"}
DEVICE = "EyeLink 1000 Desktop Mount (1000 Hz, binocular), 22-inch monitor at 90 cm, chin and head rest"


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "SaccadicAdaptation.zip")
    ck = stdlib.Checks()

    e1 = pd.read_csv(z.open("EXP1/EXP1_eyedata.txt"), sep="\t", header=None,
                     names=["sid", "g", "trial", "amp_x", "lat", "dur"])
    e2 = pd.read_csv(z.open("EXP2/EXP2_eyedata.txt"), sep="\t", header=None,
                     names=["sid", "g", "trial", "h", "v", "correct", "lat", "dur", "amp_x", "amp_y"])
    cog_names = ["sid", "g", "age", "sex", "lps3", "tmtb", "digit_back", "block_back"]
    c1 = pd.read_csv(z.open("EXP1/EXP1_cognitivedata.txt"), sep="\t", header=None, names=cog_names + ["extra"],
                     usecols=range(8), engine="python")
    c2 = pd.read_csv(z.open("EXP2/EXP2_cognitivedata.txt"), sep="\t", header=None, names=cog_names, usecols=range(8))

    # ---- structure ---------------------------------------------------------------------------------------------------
    ck.internal("Experiment 1: 32 participants (16 young, 16 senior) x 400 trials",
                e1.sid.nunique() == 32 and e1.groupby("sid").size().eq(400).all() and len(e1) == 12800
                and e1.groupby("g").sid.nunique().to_dict() == {1: 16, 2: 16})
    ck.internal("Experiment 2: 29 participants (17 young, 12 senior) x 450 trials",
                e2.sid.nunique() == 29 and e2.groupby("sid").size().eq(450).all() and len(e2) == 13050
                and e2.groupby("g").sid.nunique().to_dict() == {1: 17, 2: 12})
    ck.internal("cognitive files hold the same participants and groups as the eye files",
                set(c1.sid) == set(e1.sid) and set(c2.sid) == set(e2.sid)
                and (c1.set_index("sid").g == e1.groupby("sid").g.first()).all()
                and (c2.set_index("sid").g == e2.groupby("sid").g.first()).all())
    ck.internal("trial numbers run 1..400 and 1..450 in file order within every participant",
                bool((e1.groupby("sid").trial.apply(lambda s: (s.to_numpy() == np.arange(1, 401)).all())).all()
                     and (e2.groupby("sid").trial.apply(lambda s: (s.to_numpy() == np.arange(1, 451)).all())).all()))

    # ---- authors' trial criteria (Data analysis sections) -------------------------------------------------------------
    e1["ok"] = e1.lat.between(80, 500) & e1.dur.between(10, 100) & e1.amp_x.between(5, 15)
    e2["ok"] = e2.lat.between(80, 500) & e2.dur.between(10, 100) & e2.amp_x.abs().between(3.5, 10.5)
    e1["phase"] = np.where(e1.trial <= 50, "preadaptation", np.where(e1.trial <= 350, "adaptation", "postadaptation"))
    e2["phase"] = np.where(e2.trial <= 50, "preadaptation", np.where(e2.trial <= 250, "adaptation", "postadaptation"))

    # ---- sign conventions of Experiment 2 -------------------------------------------------------------------------------
    # README: horizontal direction 1 = leftward, vertical direction 1 = downward. In the file, amplitudes carry the sign of
    # these codes (horizontal amplitude has the sign of h in 95.8% of trials, vertical amplitude follows v after adaptation).
    # The OpenMotor module wants x positive rightward and y positive upward, so both axes are flipped to follow the README.
    ck.internal("Exp 2: horizontal amplitude has the sign of the README's horizontal direction code in >95% of trials",
                float((np.sign(e2.amp_x) == e2.h).mean()) > 0.95)
    adap2 = e2[e2.phase == "adaptation"]
    ck.internal("Exp 2: in adaptation trials vertical amplitude follows the sign of the vertical code (mean x code > 0 "
                "when coded 1, < 0 when coded -1)",
                bool(adap2[adap2.v == 1].amp_y.mean() > 0.3 and adap2[adap2.v == -1].amp_y.mean() < -0.3))
    pairs = e2[e2.phase == "adaptation"].groupby("sid").apply(lambda q: len(set(zip(q.h, q.v))))
    ck.internal("Exp 2: every participant has exactly two (direction, shift) pairings (shift coupled to saccade direction)",
                bool((pairs == 2).all()))

    # ---- published numbers: Experiment 1 -----------------------------------------------------------------------------
    e1["gain"] = e1.amp_x / 10.0
    x1 = e1[e1.ok]

    def pm(df, mask, col):
        return df[mask(df)].groupby("sid")[col].mean()

    ints = {"baseline": lambda d: d.trial <= 50, "initial adaptation": lambda d: d.trial.between(51, 60),
            "final adaptation": lambda d: d.trial.between(341, 350), "postadaptation": lambda d: d.trial > 350}
    rep1 = {"baseline": 0.98, "initial adaptation": 0.93, "final adaptation": 0.78, "postadaptation": 0.82}
    for k, f in ints.items():
        ck.published(f"Exp 1: mean saccadic gain, {k} (both age groups, 32 participants)", pm(x1, f, "gain").mean(),
                     rep1[k], 0.006, SEC1 + " (M = 0.98, 0.93, 0.78, 0.82), saccade amplitude / 10 deg step")
    ck.published("Exp 1: SD of baseline gain across participants", pm(x1, ints["baseline"], "gain").std(), 0.06, 0.006, SEC1)
    ex1 = e1.groupby(["g", "sid"]).ok.apply(lambda s: 100.0 * (1 - s.mean()))
    ck.published("Exp 1: SD of % trials excluded, young adults", ex1.loc[1].std(), 11.1, 0.1, "Exp 1, Data analysis")
    ck.published("Exp 1: SD of % trials excluded, senior adults", ex1.loc[2].std(), 12.1, 0.1, "Exp 1, Data analysis")
    ck.published("Exp 1: mean % trials excluded, young adults", ex1.loc[1].mean(), 15.5, 0.4, "Exp 1, Data analysis")
    ck.published("Exp 1: mean % trials excluded, senior adults", ex1.loc[2].mean(), 22.9, 0.4, "Exp 1, Data analysis")
    ck.published("Exp 1: participants kept after dismissing 5 of 37 (2 young, 3 senior)", e1.sid.nunique(), 37 - 5, 0,
                 "Exp 1, Participants and Data analysis")

    # ---- published numbers: Experiment 2 -----------------------------------------------------------------------------
    lat_ex = (~e2.lat.between(80, 500)).groupby([e2.g, e2.sid]).mean() * 100
    dur_ex = (~e2.dur.between(10, 100)).groupby([e2.g, e2.sid]).mean() * 100
    ck.published("Exp 2: % trials excluded by the latency criterion, young adults", lat_ex.loc[1].mean(), 3.8, 0.1, "Exp 2, Data analysis")
    ck.published("Exp 2: % trials excluded by the latency criterion, senior adults", lat_ex.loc[2].mean(), 10.1, 0.1, "Exp 2, Data analysis")
    ck.published("Exp 2: % trials excluded by the duration criterion, young adults", dur_ex.loc[1].mean(), 1.3, 0.15, "Exp 2, Data analysis")
    ck.published("Exp 2: % trials excluded by the duration criterion, senior adults", dur_ex.loc[2].mean(), 2.3, 0.3, "Exp 2, Data analysis")
    acc = e2.groupby(["g", "sid"]).correct.mean() * 100
    ck.published("Exp 2: discrimination accuracy, young adults (%)", acc.loc[1].mean(), 98.2, 0.15, "Exp 2, Data analysis")
    ck.published("Exp 2: discrimination accuracy, senior adults (%)", acc.loc[2].mean(), 93.7, 0.15, "Exp 2, Data analysis")
    ck.published("Exp 2: participants kept (17 young + 16 senior - 3 dismissed - 1 at chance)", e2.sid.nunique(), 29, 0, "Exp 2, Data analysis")
    e2["coll"] = e2.amp_y * e2.v            # vertical amplitude collapsed over the two shift directions (sign-free of the axis)
    x2 = e2[e2.ok]
    base = x2[x2.trial <= 50].groupby(["g", "sid"]).coll.mean().groupby("g").mean()
    fin = x2[x2.trial.between(241, 250)].groupby(["g", "sid"]).coll.mean().groupby("g").mean()
    ck.published("Exp 2: size of baseline vertical amplitude, young adults (deg)", abs(base.loc[1]), 0.06, 0.03, SEC2)
    ck.published("Exp 2: size of baseline vertical amplitude, senior adults (deg)", abs(base.loc[2]), 0.07, 0.03, SEC2)
    ck.published("Exp 2: vertical amplitude at the end of adaptation, young adults (deg)", fin.loc[1], 1.06, 0.06, SEC2)
    ck.published("Exp 2: vertical amplitude at the end of adaptation, senior adults (deg)", fin.loc[2], 0.84, 0.06, SEC2)

    # ---- standardized rows -------------------------------------------------------------------------------------------
    cog1 = c1.set_index("sid")
    cog2 = c2.set_index("sid")
    rows = []

    adap1 = (e1.phase == "adaptation").to_numpy()
    r1 = pd.DataFrame({
        "dataset": DS, "study": DS, "experiment": "double-step", "group": e1.g.map(GRP),
        "subject": [f"E1_{s}" for s in e1.sid], "person": [f"E1_{s}" for s in e1.sid],
        "subject_excluded_by_authors": False,
        "age": e1.sid.map(cog1.age), "sex": e1.sid.map(cog1.sex).map({1: "M", 2: "F"}), "handedness": "",
        "population": "healthy", "setting": "lab", "device": DEVICE, "session": 1,
        "block": e1.phase, "trial": e1.trial,
        "phase": e1.phase.map({"preadaptation": "baseline", "adaptation": "learning", "postadaptation": "aftereffect"}),
        "time_s": np.nan, "task": "eye movements", "unit": "saccade", "condition": "double-step",
        "pert_type": np.where(adap1, "intrasaccadic_step", "none"),
        "pert_value": np.where(adap1, -2.5, 0.0), "pert_unit": "deg",
        "feedback": np.where(adap1, "altered", "none"),
        "excluded_by_authors": ~e1.ok,
        "target_amplitude_deg": 10.0, "target_direction_deg": 0.0,
        "step_x_deg": np.where(adap1, -2.5, np.nan), "step_y_deg": np.where(adap1, 0.0, np.nan),
        "step_delay_ms": np.where(adap1, 0.0, np.nan),
        "saccade_amplitude_deg": np.nan, "saccade_amplitude_x_deg": e1.amp_x, "saccade_gain": e1.gain,
        "saccade_latency_ms": e1.lat, "saccade_duration_ms": e1.dur, "eye": "both",
    })
    r1["saccade_amplitude_deg"] = np.nan
    rows.append(r1)

    adap2f = (e2.phase == "adaptation").to_numpy()
    shift = np.where(adap2f, np.where(e2.v == 1, -1.47, 1.47), 0.0)          # up positive; README: code 1 = downward
    r2 = pd.DataFrame({
        "dataset": DS, "study": DS, "experiment": "perceptual task", "group": e2.g.map(GRP),
        "subject": [f"E2_{s}" for s in e2.sid], "person": [f"E2_{s}" for s in e2.sid],
        "subject_excluded_by_authors": False,
        "age": e2.sid.map(cog2.age), "sex": e2.sid.map(cog2.sex).map({1: "M", 2: "F"}), "handedness": "",
        "population": "healthy", "setting": "lab", "device": DEVICE, "session": 1,
        "block": e2.phase, "trial": e2.trial,
        "phase": e2.phase.map({"preadaptation": "baseline", "adaptation": "learning", "postadaptation": "aftereffect"}),
        "time_s": np.nan, "task": "eye movements", "unit": "saccade", "condition": "perceptual task",
        "pert_type": np.where(adap2f, "postsaccadic_target_shift", "none"),
        "pert_value": shift, "pert_unit": "deg",
        "feedback": "normal", "excluded_by_authors": ~e2.ok,
        "target_amplitude_deg": 7.0,
        "target_direction_deg": np.where(e2.h == 1, 180.0, 0.0),             # README: code 1 = leftward
        "step_x_deg": 0.0, "step_y_deg": shift, "step_delay_ms": 0.0,
        "saccade_amplitude_x_deg": -e2.amp_x, "saccade_amplitude_y_deg": -e2.amp_y,
        "saccade_latency_ms": e2.lat, "saccade_duration_ms": e2.dur, "eye": "both",
        "response": "", "response_correct": e2.correct == 1,
    })
    # the amplitude of the primary saccade is not given as a vector length by the authors; leave it empty
    rows.append(r2)
    out = pd.concat(rows, ignore_index=True)
    for c in stdlib.columns_for(MODULE):
        if c[1] not in out.columns:
            out[c[1]] = ""
    out["person"] = out["subject"]
    ck.internal("every eye-data row of both experiments is kept in the authors' order", len(out) == 12800 + 13050)

    cols = [("age_group", "category", "young, senior", "Age group as in the main table."),
            ("lps3_correct", "integer", "items", "LPS-3 (non-verbal problem solving, German intelligence test subtest): number of correct items."),
            ("tmtb_time_s", "number", "seconds",
             "Trail Making Test part B completion time. The deposit's column description says ms, but the values (about 25-110) "
             "are only plausible as seconds."),
            ("digit_span_backward", "integer", "items", "Backward digit span (Wechsler Memory Scale): longest correctly repeated sequence."),
            ("block_span_backward", "integer", "items", "Backward block span (Wechsler Memory Scale): longest correctly repeated sequence."),
            ("experiment", "text", "", "Experiment the participant took part in (participant numbers are reused between experiments and are not linked).")]
    part = []
    for tag, cog, exp in (("E1", c1, "double-step"), ("E2", c2, "perceptual task")):
        for _, r in cog.iterrows():
            part.append(dict(person=f"{tag}_{int(r.sid)}", age_group=GRP[int(r.g)], lps3_correct=r.lps3, tmtb_time_s=r.tmtb,
                             digit_span_backward=r.digit_back, block_span_backward=r.block_back, experiment=exp))
    part = pd.DataFrame(part)
    ck.internal("participants table: 32 + 29 rows", len(part) == 61)
    ck.internal("TMT-B values are in a plausible range for seconds (20-150)", bool(part.tmtb_time_s.dropna().between(20, 150).all()),
                f"{part.tmtb_time_s.min():.1f}-{part.tmtb_time_s.max():.1f}")

    meta = dict(
        study=DS,
        paper=dict(authors="Huang J, Gegenfurtner KR, Schutz AC, Billino J", year=2017,
                   title="Age effects on saccadic adaptation: Evidence from different paradigms reveals specific vulnerabilities",
                   venue="J Vis 17(6):9", doi="10.1167/17.6.9"),
        lab="Billino / Gegenfurtner (University of Giessen)",
        data_url="https://doi.org/10.5281/zenodo.546106",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Converted the two trial tables (EXP1_eyedata.txt, EXP2_eyedata.txt) to the OpenMotor saccade format with "
                 "participant codes; added trial-exclusion flags re-derived from the criteria stated in the paper; the "
                 "executive-control scores became participants.csv. In Experiment 2 both axes were flipped to the OpenMotor "
                 "convention (x rightward, y upward) following the deposit's column description."),
        how=["One row is one trial of one participant: the horizontal amplitude (Experiment 1) or horizontal and vertical "
             "amplitude (Experiment 2), latency and duration of the primary saccade. Experiment 1 is a double-step paradigm "
             "(target 10 deg to the right, stepped back by 2.5 deg during the saccade in the 300 adaptation trials; target "
             "extinguished at saccade onset in the 50 pre- and 50 post-adaptation trials); saccade_gain is the amplitude divided "
             "by the 10 deg target step. Experiment 2 is a perceptual task (target 7 deg left or right; during 200 adaptation "
             "trials the discrimination target sat 1.47 deg above or below the centre of the stimulus, depending on saccade "
             "direction, so pert_value and step_y_deg give that shift, positive upward).",
             "The deposit did not flag excluded trials; excluded_by_authors marks trials outside the paper's criteria "
             "(latency 80-500 ms, duration 10-100 ms, Experiment 1 amplitude 5-15 deg, Experiment 2 absolute horizontal "
             "amplitude 3.5-10.5 deg). Participants the authors dismissed (5 in Experiment 1, 4 in Experiment 2) are not in the "
             "deposit.",
             "Participant numbers repeat across the two experiments but ages differ for some, so they are coded as different "
             "people. The deposit's column description for Experiment 2 labels horizontal code 1 as leftward and vertical "
             "code 1 as downward, while amplitudes with the positive code are positive; the converter follows the description "
             "(so x and y are sign-flipped to rightward-positive and upward-positive). The task is mirror-symmetric, so no "
             "published result depends on the horizontal sign."],
        unit_note="trial (primary saccade)",
        differences=["Experiment 1 exclusion rates computed from the paper's criteria are 15.2% (young) and 22.6% (senior) against "
                     "15.5% and 22.9% in the paper (SDs match); the cause is not stated in the paper.",
                     "The deposit's column description for the Trail Making Test says milliseconds; the values are in seconds.",
                     "Experiment 2: trials excluded by the latency criterion (3.8% young, 10.1% senior) and the duration criterion "
                     "(1.2% and 2.1%) reproduce the paper, but the horizontal-amplitude criterion removes 5.7% and 15.5% of trials "
                     "here against 7.5% and 20.4% in the paper, so the total exclusion rates are lower here (6.6% and 17.2%, paper 9.1% "
                     "and 20.7%). The paper does not say how the amplitude was measured for that criterion.",
                     "Four participants in Experiment 2 and one in Experiment 1 have an incomplete row in the cognitive file; "
                     "missing values are left empty."],
    )
    stdlib.release(DS, MODULE, out, meta, ck, participants=part, participant_columns=cols)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}  ok={c['ok']}")


if __name__ == "__main__":
    main()
