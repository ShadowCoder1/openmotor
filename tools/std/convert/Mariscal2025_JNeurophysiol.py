"""Mariscal, Sombric & Torres-Oviedo (2025) J Neurophysiol: young and older adults adapted to a split-belt treadmill at
slow or self-selected (fast) speeds; generalization to overground walking. 4 groups x 8, one row per stride.

Raw files: raw_extra/Mariscal2025_JNeurophysiol/<group>__OG<n>KinReduceParams.mat, 32 files from the OSF component
'Parameter Published' (osf.io/shf5g) of project whces (no license stated), in folders Young Slow, Young Fast, Old Slow and
Old Fast. Each holds one labTools adaptationData object whose parameterSeries keeps 8 parameters per stride (eventType,
bad, good, trial, initTime, stepLengthAsym, alphaAsym, Xasym). Decoded from the MAT-file's own object store with
Mariscal2025_JNeurophysiol_labtools.py (the pilot's reader); nothing from the deposit or labTools is executed.
Paper text read from the NIH manuscript (PMC12301796).

Run from the site folder: python3 -I tools/std/convert/Mariscal2025_JNeurophysiol.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Mariscal2025_JNeurophysiol_labtools import MatObjects, number, strides, text  # noqa: E402

DS = "Mariscal2025_JNeurophysiol"
MODULE = "walking"
GROUPS = {"YoungSlow": "Young Slow", "YoungFast": "Young Fast", "OldSlow": "Old Slow", "OldFast": "Old Fast"}
# deposit file order within each OSF folder (fetched in this order)
PHASE = {"OG base": "baseline", "fast base": "baseline", "slow base": "baseline", "Slow base": "baseline",
         "SS base": "baseline", "Medium Speed SS": "baseline", "TM base": "baseline", "self selected": "other",
         "Self Selective": "other", "short split": "practice", "adaptation": "learning", "catch": "aftereffect",
         "Catch": "aftereffect", "re-adaptation": "learning", "OG post": "transfer", "TM post": "washout"}
SPLIT = {"short split", "adaptation", "re-adaptation"}
OG = {"OG base", "OG post"}


def main():
    raw = stdlib.raw_dir(DS)
    files = sorted(raw.glob("*__*.mat"), key=lambda p: (list(GROUPS).index(p.name.split("__")[0]), p.name))
    ck = stdlib.Checks()
    frames, info = [], []
    for p in files:
        g = p.name.split("__")[0]
        mo = MatObjects(p.read_bytes())
        a = mo.variable(mo.top["adaptData"])
        st, sub = strides(a), a["subData"]
        col, block, n = st["col"], st["block"], st["n"]
        assert (block != "").all() and not (set(block) - set(PHASE)), set(block) - set(PHASE)
        sex = {"Female": "F", "Male": "M"}.get(text(sub["sex"]).strip(), "")
        age = number(sub["age"])
        subj = f"{g}:{p.name}"
        info.append(dict(group=g, sex=sex, age=age, n=n))
        split = np.isin(block, list(SPLIT))
        f = dict(dataset=DS, study=DS, experiment="", group=GROUPS[g], subject=subj, subject_excluded_by_authors=False,
                 age=age if age > 0 else np.nan, sex=sex, handedness=text(sub["dominantArm"]).lower(),
                 population="healthy older adults" if g.startswith("Old") else "healthy young adults", setting="lab",
                 device=np.where(np.isin(block, list(OG)), "overground walkway; motion capture",
                                 "Bertec split-belt treadmill; motion capture"),
                 session=1, block=block, trial=np.arange(1, n + 1), phase=[PHASE[b] for b in block], time_s=np.nan,
                 task="walking", unit="stride", condition="", pert_type=np.where(split, "split_belt", "none"),
                 pert_value=np.where(block == "adaptation", np.nan, np.where(split, 2.0, 0.0)),
                 pert_unit=np.where(split, "ratio", ""), feedback="normal", excluded_by_authors=col["bad"] == 1,
                 fast_leg=text(sub["dominantLeg"]).lower(), walking_trial=st["trial"],
                 stride_start_s=col["initTime"].round(8), step_length_asymmetry=col["stepLengthAsym"].round(8),
                 leading_position_asymmetry=col["alphaAsym"].round(8), trailing_position_asymmetry=col["Xasym"].round(8),
                 walking_context=np.where(np.isin(block, list(OG)), "overground", "treadmill"), surface_incline_deg=0.0)
        frames.append(pd.DataFrame(f))
    df = pd.concat(frames, ignore_index=True)
    ex = df.excluded_by_authors

    cnt = pd.Series([i_["group"] for i_ in info]).value_counts().to_dict()
    ck.internal("32 participants, 8 per group, one file each", len(info) == 32 and set(cnt.values()) == {8}, str(cnt))
    ck.internal("every stride the deposit shares is kept, in its order", len(df) == sum(i_["n"] for i_ in info), f"{len(df)} rows")
    gd = df[~ex]
    d = (gd.leading_position_asymmetry + gd.trailing_position_asymmetry - gd.step_length_asymmetry).abs().dropna()
    d2 = (gd.leading_position_asymmetry - gd.trailing_position_asymmetry - gd.step_length_asymmetry).abs().dropna()
    ck.internal("leading_position_asymmetry + trailing_position_asymmetry is close to step_length_asymmetry on good "
                "strides (approximate, not exact: this labTools version took X_fast at the stride's opening slow heel "
                "strike), and much closer than their difference", np.median(d) < 0.02 and np.median(d2) > 2 * np.median(d),
                f"median |sum - SLA| {np.median(d):.4f}, median |difference - SLA| {np.median(d2):.4f}, n={len(d)}")
    good_tm = df[~ex & (df.walking_context == "treadmill")].step_length_asymmetry.abs()
    ck.internal("on good treadmill strides |step_length_asymmetry| < 1 (the large values sit on bad or overground "
                "strides and are kept as shared)", (good_tm < 1).mean() > 0.999, f"max {good_tm.max():.2f}")
    ck.internal("excluded_by_authors = labTools 'bad' flag", 0 < ex.mean() < 0.12, f"{int(ex.sum())} strides ({100 * ex.mean():.2f}%)")

    women = {g: sum(i_["sex"] == "F" for i_ in info if i_["group"] == g) for g in GROUPS}
    for g, rep in (("YoungFast", 2), ("YoungSlow", 5), ("OldFast", 5), ("OldSlow", 4)):
        ck.published(f"{GROUPS[g]}: number of participants", cnt[g], 8, 0, "Methods, Participants")
        ck.published(f"{GROUPS[g]}: number of women", women[g], rep, 0, "Methods, Participants")
    ck.published("young adults: women (7 of 16)", women["YoungFast"] + women["YoungSlow"], 7, 0, "Methods, Participants")
    ck.published("older adults: women (9 of 16)", women["OldFast"] + women["OldSlow"], 9, 0, "Methods, Participants")
    ys = np.array([i_["age"] for i_ in info if i_["group"] == "YoungSlow"])
    ck.published("Young Slow: mean age (22.17 yrs)", ys.mean(), 22.17, 0.01, "Methods, Participants")
    ck.published("Young Slow: SD of age (1.03)", ys.std(ddof=1), 1.03, 0.01, "Methods, Participants")
    ages = {g: np.array([i_["age"] for i_ in info if i_["group"] == g]) for g in GROUPS}

    meta = dict(
        study=DS,
        paper=dict(authors="Mariscal DM, Sombric CJ, Torres-Oviedo G", year=2025,
                   title="Age and self-selected walking speed impact the generalization of locomotor memories across contexts",
                   venue="J Neurophysiol 133(5):1410-1421", doi="10.1152/jn.00432.2023"),
        lab="Torres-Oviedo (University of Pittsburgh)", data_url="https://osf.io/whces/", license="none stated",
        license_url="",
        changes=("Read the deposit's MATLAB (labTools) objects without the lab's code and kept every stride with the three "
                 "asymmetry measures the deposit shares, the recording number, the stride start time and the bad-stride "
                 "flag; participant IDs, dates, height and weight are left out."),
        how=["One row is one stride (slow-leg heel strike to the next) of one participant, in the deposit's order; the "
             "deposit keeps per stride only step length asymmetry and its two parts, leading_position_asymmetry (labTools "
             "alphaAsym) and trailing_position_asymmetry (Xasym), each divided by stride length, so per-leg lengths and "
             "times are empty. Their sum is close to, but not exactly, step length asymmetry (median difference about "
             "0.01), because this labTools version measured the fast leg's trailing position at the stride's first slow "
             "heel strike.",
             "Fast is the dominant leg. The first adaptation block ramped the belts linearly from tied to 2:1 over 600 "
             "strides (pert_value empty there); short split and re-adaptation were 2:1, catch 10 tied strides. Belt speeds "
             "are left empty: the paper gives 0.55 and 1.1 m/s for the slow groups (one participant 0.5 and 1.0) and "
             "speeds tailored to each person for the fast groups (mean 1.17 +- 0.09 m/s young, 1.10 +- 0.12 m/s older; "
             "slow belt 2/3 and fast belt 4/3 of the mean), while the deposit's condition notes give other values for "
             "some sessions.",
             "Signs are labTools': fast minus slow, positive when the fast leg's step is longer. OG blocks are overground "
             "on a walkway; 'self selected' blocks are treadmill walking at a self-set pace used to find the speed.",
             "excluded_by_authors marks labTools' bad strides (missing or disordered events, stride time out of range)."],
        unit_note="stride (from a slow-leg heel strike to the next one)",
        differences=[
            "One Young Fast participant's age is 0 in the deposit (left empty here); the paper's Young Fast age (20 +- 8.67) "
            "appears to include it. Mean ages here: Young Fast %.2f (with the 0), Old Fast %.2f (paper 71.2), Old Slow %.2f "
            "(paper 75)." % (ages["YoungFast"].mean(), ages["OldFast"].mean(), ages["OldSlow"].mean()),
            "No outcome statistic was checked: the paper's epochs are not documented in the deposit."],
    )
    names = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in names if c not in df.columns})[names]
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
