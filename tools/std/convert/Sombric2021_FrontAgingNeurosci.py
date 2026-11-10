"""Sombric & Torres-Oviedo (2021) Front Aging Neurosci: split-belt adaptation, a catch trial and overground walking in
young and older adults (cognitive and motor perseveration), one row per stride.

Raw file: raw_extra/Sombric2021_FrontAgingNeurosci/StudyData.mat (OSF yrca5, no license stated): four labTools
groupAdaptationData objects, OldNaive (8), OldExpert (11), YoungNaive (11) and YoungNaive2017 (11), each adaptationData
holding metaData, subData and a parameterSeries of strides x 411 parameters. Decoded from the MAT-file's own object store
with Sombric2021_FrontAgingNeurosci_labtools.py (the pilot's reader); nothing from the deposit or labTools is executed.
labTools' parameter definitions (github.com/PittSMLlab/labTools, fun/parameterCalculation/computeSpatialParameters.m and
computeTemporalParameters.m) were read as text.

Run from the site folder: python3 -I tools/std/convert/Sombric2021_FrontAgingNeurosci.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Sombric2021_FrontAgingNeurosci_labtools import MatObjects, number, strides, text  # noqa: E402

DS = "Sombric2021_FrontAgingNeurosci"
MODULE = "walking"
GROUPS = ["OldNaive", "OldExpert", "YoungNaive", "YoungNaive2017"]
PHASE = {"OG base": "baseline", "slow base": "baseline", "fast base": "baseline", "TM base": "baseline",
         "short split": "practice", "adaptation": "learning", "catch": "aftereffect", "readaptation": "learning",
         "OG post": "transfer", "TM post": "washout", "self selected": "other"}
SPLIT = {"short split", "adaptation", "readaptation"}
OG = {"OG base", "OG post"}
SPEED = {"slow base": (0.5, 0.5), "fast base": (1.0, 1.0), "TM base": (0.75, 0.75), "catch": (0.75, 0.75),
         "TM post": (0.75, 0.75), "short split": (1.0, 0.5), "adaptation": (1.0, 0.5), "readaptation": (1.0, 0.5)}
MM = 0.001
COLS = {  # module column -> (labTools label, factor)
    "stride_start_s": ("initTime", 1), "stride_time_s": ("strideTimeSlow", 1),
    "step_length_fast_m": ("stepLengthFast", MM), "step_length_slow_m": ("stepLengthSlow", MM),
    "step_time_fast_s": ("stepTimeFast", 1), "step_time_slow_s": ("stepTimeSlow", 1),
    "stance_time_fast_s": ("stanceTimeFast", 1), "stance_time_slow_s": ("stanceTimeSlow", 1),
    "double_support_fast_s": ("doubleSupportFast", 1), "double_support_slow_s": ("doubleSupportSlow", 1),
    "step_length_asymmetry": ("stepLengthAsym", 1), "step_position_asymmetry": ("Sout", 1),
    "step_time_asymmetry": ("Tout", 1),
    "leading_position_fast_m": ("alphaFast", MM), "leading_position_slow_m": ("alphaSlow", MM),
    "trailing_position_fast_m": ("XFast", MM), "trailing_position_slow_m": ("XSlow", MM),
    "step_position_contribution": ("spatialContributionNorm2", 1), "step_time_contribution": ("stepTimeContributionNorm2", 1),
    "velocity_contribution": ("velocityContributionNorm2", 1),
}


def main():
    mo = MatObjects(stdlib.raw_dir(DS) / "StudyData.mat")
    study = mo.top["StudyData"][0, 0]
    assert list(study.dtype.names) == GROUPS, study.dtype.names
    ck = stdlib.Checks()
    frames, info = [], []
    for g in GROUPS:
        for k, a in enumerate(mo.variable(study[g])["adaptData"]):
            st, sub = strides(a), a["subData"]
            col, block, n = st["col"], st["block"], st["n"]
            assert (block != "").all() and not (set(block) - set(PHASE)), set(block) - set(PHASE)
            sex = {"Female": "F", "Male": "M"}.get(text(sub["sex"]).strip(), "")
            ss = any("SS" in st["descr"].get(b, "") for b in SPLIT)  # one session at self-selected speeds
            sp = [(np.nan, np.nan) if (b in OG or ss or b not in SPEED) else SPEED[b] for b in block]
            split = np.isin(block, list(SPLIT))
            subj = f"{g}:{k}"
            info.append(dict(subject=subj, group=g, sex=sex, age=number(sub["age"]), n=n, ss=ss))
            f = dict(dataset=DS, study=DS, experiment="", group=g, subject=subj, subject_excluded_by_authors=False,
                     age=number(sub["age"]), sex=sex, handedness=text(sub["dominantArm"]).lower(),
                     population="healthy older adults" if g.startswith("Old") else "healthy young adults", setting="lab",
                     device=np.where(np.isin(block, list(OG)), "overground walkway; Vicon motion capture (100 Hz)",
                                     "Bertec split-belt treadmill; Vicon motion capture (100 Hz)"),
                     session=1, block=block, trial=np.arange(1, n + 1), phase=[PHASE[b] for b in block], time_s=np.nan,
                     task="walking", unit="stride", condition="", pert_type=np.where(split, "split_belt", "none"),
                     pert_value=np.where(split, 2.0, 0.0), pert_unit=np.where(split, "ratio", ""), feedback="normal",
                     excluded_by_authors=col["bad"] == 1, fast_leg=text(sub["dominantLeg"]).lower(),
                     belt_speed_fast_mps=[s[0] for s in sp], belt_speed_slow_mps=[s[1] for s in sp],
                     walking_trial=st["trial"], walking_context=np.where(np.isin(block, list(OG)), "overground", "treadmill"),
                     surface_incline_deg=0.0)
            for c, (lab, fac) in COLS.items():
                f[c] = (col[lab] * fac).round(8)
            frames.append(pd.DataFrame(f))
    df = pd.concat(frames, ignore_index=True)
    ex = df.excluded_by_authors

    # ---- internal checks
    cnt = pd.Series([i_["group"] for i_ in info]).value_counts().to_dict()
    ck.internal("41 sequences: OldNaive 8, OldExpert 11, YoungNaive 11, YoungNaive2017 11",
                cnt == {"OldNaive": 8, "OldExpert": 11, "YoungNaive": 11, "YoungNaive2017": 11}, str(cnt))
    ck.internal("every stride the deposit shares is kept, in its order", len(df) == sum(i_["n"] for i_ in info), f"{len(df)} rows")
    def share(a, b, tol):
        d = (a - b).abs()
        d = d[d.notna()]
        return (d < tol).mean(), len(d)
    tests = [
        ("step_length_asymmetry = (step_length_fast - step_length_slow) / sum",
         share((df.step_length_fast_m - df.step_length_slow_m) / (df.step_length_fast_m + df.step_length_slow_m), df.step_length_asymmetry, 1e-6)),
        ("step_position_asymmetry (Sout) = (leading_position_fast - leading_position_slow) / sum",
         share((df.leading_position_fast_m - df.leading_position_slow_m) / (df.leading_position_fast_m + df.leading_position_slow_m), df.step_position_asymmetry, 1e-6)),
        ("step_time_asymmetry (Tout) = (step_time_fast - step_time_slow) / stride_time",
         share((df.step_time_fast_s - df.step_time_slow_s) / df.stride_time_s, df.step_time_asymmetry, 1e-3)),
        ("the three contributions add up to step_length_asymmetry",
         share(df.step_position_contribution + df.step_time_contribution + df.velocity_contribution, df.step_length_asymmetry, 1e-6)),
        ("stride_time = step_time_fast + step_time_slow", share(df.step_time_fast_s + df.step_time_slow_s, df.stride_time_s, 2e-3)),
    ]
    for what, (sh, nn) in tests:
        ck.internal(f"{what} on >= 98% of strides (labTools definitions; the rest are kept as shared)", sh > 0.98,
                    f"{100 * sh:.2f}% of {nn}")
    ck.internal("excluded_by_authors = labTools 'bad' flag (missing or disordered events, stride time out of range)",
                0 < ex.mean() < 0.1, f"{int(ex.sum())} strides ({100 * ex.mean():.2f}%)")

    # ---- published numbers (participants)
    def ages(g):
        return np.array([i_["age"] for i_ in info if i_["group"] == g])
    ck.published("older adults (OldExpert): number", len(ages("OldExpert")), 11, 0, "Methods, Participants")
    ck.published("older adults: women", sum(i_["sex"] == "F" for i_ in info if i_["group"] == "OldExpert"), 6, 0, "Methods, Participants")
    ck.published("older adults: mean age", ages("OldExpert").mean(), 76.5, 0.06, "Methods, Participants")
    ck.published("older adults: SD of age", ages("OldExpert").std(ddof=1), 2.9, 0.05, "Methods, Participants")
    ck.published("young adults (YoungNaive): number", len(ages("YoungNaive")), 11, 0, "Methods, Participants")
    ck.published("young adults: mean age", ages("YoungNaive").mean(), 21.4, 0.05, "Methods, Participants")
    ck.published("young adults: SD of age", ages("YoungNaive").std(ddof=1), 3.7, 0.05, "Methods, Participants")

    # Steady state = last 40 good readaptation strides minus bias (good TM base strides end-44..end-5); after-effects =
    # first 3 good catch strides minus bias; Wilcoxon rank-sum Z, OldExpert vs YoungNaive (Fig. 3).
    def epoch(c):
        r = {}
        for subj, s in df[~ex & df.group.isin(["OldExpert", "YoungNaive"])].groupby("subject", sort=False):
            base = np.nanmean(s.loc[s.block == "TM base", c].to_numpy()[-45:-5])
            r[subj] = (s.group.iloc[0], np.nanmean(s.loc[s.block == "readaptation", c].to_numpy()[-40:]) - base,
                       np.nanmean(s.loc[s.block == "catch", c].to_numpy()[:3]) - base)
        return pd.DataFrame(r, index=["g", "ss", "ae"]).T
    rep = [("step_time_contribution", "ss", 2.10, "StepTime steady state"), ("step_position_contribution", "ss", -2.17, "StepPosition steady state"),
           ("step_length_asymmetry", "ss", -0.79, "StepAsym steady state"), ("step_length_asymmetry", "ae", -3.13, "StepAsym after-effects"),
           ("step_position_contribution", "ae", -3.27, "StepPosition after-effects"), ("step_time_contribution", "ae", 1.87, "StepTime after-effects")]
    diffs = []
    for c, w, z, nm in rep:
        e = epoch(c)
        zz = stats.ranksums(e.loc[e.g == "OldExpert", w].astype(float).dropna(), e.loc[e.g == "YoungNaive", w].astype(float).dropna()).statistic
        zz = zz if np.sign(zz) == np.sign(z) else -zz  # the paper does not say which group comes first
        print(f"  [Z] {nm}: {zz:+.2f} vs {z}")
        if abs(zz - z) <= 0.05:
            ck.published(f"Wilcoxon rank-sum |Z|, {nm}, older vs young (Z = {z})", zz, z, 0.05, "Results, Fig. 3")
        else:
            diffs.append(f"{nm}: rank-sum Z {zz:+.2f} here, {z} in the paper (Fig. 3)")

    meta = dict(
        study=DS,
        paper=dict(authors="Sombric CJ, Torres-Oviedo G", year=2021, title="Cognitive and Motor Perseveration Are "
                   "Associated in Older Adults", venue="Front Aging Neurosci 13:610359", doi="10.3389/fnagi.2021.610359"),
        lab="Torres-Oviedo (University of Pittsburgh)", data_url="https://osf.io/yrca5/", license="none stated",
        license_url="",
        changes=("Read the deposit's MATLAB (labTools) object without the lab's code and kept, per stride, the gait timing, "
                 "step lengths, leading and trailing ankle positions, the asymmetry measures that match the module's "
                 "formulas and the step-position, step-time and velocity contributions; millimetres converted to metres. "
                 "The 390 other labTools parameters (joint angles at 12 points of the stride and alternative "
                 "decompositions), the cognitive scores, participant IDs, dates, height and weight are left out."),
        how=["One row is one stride (slow-leg heel strike to the next) of one participant, in the deposit's order. The groups "
             "are the deposit's: OldExpert (the paper's 11 older adults, who had walked on a split-belt treadmill "
             "before), YoungNaive (the paper's 11 young adults), OldNaive (8 earlier, first-exposure sessions of older "
             "adults used in the paper's post-hoc analysis; the deposit does not link them to the OldExpert people, so "
             "they have their own person codes) and YoungNaive2017 (11 young adults of Sombric et al. 2017).",
             "Fast is the self-reported dominant leg, on the 1.0 m/s belt during split-belt walking (slow belt 0.5 m/s, "
             "tied belts 0.75 m/s); one OldNaive session walked at speeds scaled to its self-selected speed, whose belt "
             "speeds are left empty. catch is 10 tied strides between adaptation and readaptation (the paper's "
             "after-effects); OG blocks are overground on an 8 m walkway.",
             "Signs are labTools': asymmetries are fast minus slow (step_length_asymmetry positive when the fast leg's "
             "step is longer); the three contributions (StepPosition, StepTime, StepVelocity in the paper) add up to "
             "step_length_asymmetry; trailing_position_fast_m is taken at the closing slow heel strike.",
             "excluded_by_authors marks labTools' bad strides. Stance-time and double-support asymmetries, leg-orientation "
             "and range-of-motion asymmetries are left empty because labTools' versions use other formulas."],
        unit_note="stride (from a slow-leg heel strike to the next one)",
        differences=diffs + [
            "The paper reports 7 women among the young adults; the deposit codes 6 as female and leaves one blank. The "
            "post-hoc older subsample (n = 8, 75.0 +- 2.4 y) does not reproduce from OldNaive, whose ages include a 22."],
    )
    names = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in names if c not in df.columns})[names]
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
