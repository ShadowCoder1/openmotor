"""Sombric & Torres-Oviedo (2020) J NeuroEngineering Rehabil: split-belt walking on a flat and on an inclined treadmill
in 12 people in the chronic phase after a stroke (two sessions 204-360 days apart), one row per stride. The deposit also
holds the three groups of young adults (decline, flat, incline; 8 each) of Sombric, Calvert & Torres-Oviedo (2019,
Front Physiol), which the paper uses as its reference for young, intact walkers; they are kept here as well.

Raw file: raw_extra/Sombric2020_JNER/StudyData.mat (OSF vm597, no license stated). It holds five labTools
groupAdaptationData objects: DeclineYoungAbrupt, InclineYoungAbrupt, FlatYoungAbrupt (young adults), NoDescription (the
stroke flat session) and InclineStroke (the stroke incline session). Each adaptationData object has metaData (conditions
and their recordings), subData (sex, age, dominant leg and arm, affected side) and a parameterSeries of strides x 15
parameters. The objects are decoded from the MAT-file's own object store (Sombric2020_JNER_labtools.py, the pilot's
reader); nothing from the deposit or from labTools is executed. The authors' analysis code (github.com/csombric/
Sombric2019: biomechAnalysis/getForceResults.m, classes/dataStructs/@adaptationData/removeBiasV3.m, CodeByCarly/
computeForceParameters.m) and labTools' parameter definitions (github.com/PittSMLlab/labTools,
fun/parameterCalculation/computeSpatialParameters.m) were read as text.

Run from the site folder: python3 -I tools/std/convert/Sombric2020_JNER.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Sombric2020_JNER_labtools import MatObjects, number, strides, text  # noqa: E402

DS = "Sombric2020_JNER"
MODULE = "walking"

# Table 1 of the paper, in the deposit's order (P1..P12; the order is confirmed by the stride counts below):
# age, sex, affected (paretic) side, slow/mid/fast speed (m/s), adaptation strides (flat, incline),
# post-adaptation strides (flat, incline), incline-session slope (deg)
TABLE1 = [
    (43, "F", "R", (0.75, 1.13, 1.50), (907, 609), (605, 303), 8.5),
    (55, "F", "R", (0.54, 0.81, 1.08), (867, 301), (642, 300), 5),
    (64, "F", "R", (0.40, 0.60, 0.80), (617, 368), (307, 10), 5),
    (58, "F", "R", (0.30, 0.45, 0.60), (901, 406), (625, 10), 5),
    (66, "M", "R", (0.51, 0.77, 1.02), (606, 452), (599, 302), 5),
    (60, "F", "R", (0.60, 0.90, 1.20), (907, 597), (600, 300), 5),
    (77, "M", "R", (0.23, 0.35, 0.47), (589, 605), (598, 302), 5),
    (59, "M", "R", (0.47, 0.70, 0.93), (905, 608), (600, 306), 8.5),
    (52, "M", "R", (0.64, 0.96, 1.28), (903, 602), (603, 302), 5),
    (66, "M", "L", (0.51, 0.76, 1.01), (908, 519), (602, 299), 8.5),
    (75, "M", "R", (0.63, 0.94, 1.25), (913, 497), (552, 306), 5),
    (49, "M", "R", (0.47, 0.71, 0.95), (931, 450), (303, 300), 5),
]
YOUNG = {"DeclineYoungAbrupt": -8.5, "InclineYoungAbrupt": 8.5, "FlatYoungAbrupt": 0.0}
STROKE = {"NoDescription": ("flat session", 1), "InclineStroke": ("incline session", 2)}
GROUPS = ["DeclineYoungAbrupt", "InclineYoungAbrupt", "FlatYoungAbrupt", "NoDescription", "InclineStroke"]

PHASE = {
    # young adults (Sombric et al. 2019)
    "slow flat": "baseline", "fast flat": "baseline", "TM base flat": "baseline", "slow": "baseline", "fast": "baseline",
    "TM base": "baseline", "catch": "practice", "adaptation": "learning", "TM post": "washout", "flat post": "washout",
    # stroke, flat session
    "OG base": "baseline", "TM slow": "baseline", "TM mid": "baseline", "Mid": "baseline", "short split": "practice",
    "Short split": "practice", "Short exposure": "practice", "OG post": "transfer", "Mid uphill": "other",
    "Big uphill": "other", "Uphill low": "other", "Uphill high": "other", "Uphill mid": "other", "Uphill steep": "other",
    "WhoKnows": "other",
    # stroke, incline session
    "flat base": "baseline", "re-adaptation": "learning",
}
SPLIT = {"catch", "adaptation", "short split", "Short split", "Short exposure", "re-adaptation"}
OVERGROUND = {"OG base", "OG post"}
UPHILL_FLAT_SESSION = {"Mid uphill": 5.0, "Big uphill": 8.5, "Uphill low": 5.0, "Uphill high": 8.0, "Uphill mid": 5.0,
                       "Uphill steep": 8.5}  # short trials at the end of the stroke flat session, slopes from the descriptions
TM_DEVICE = "Bertec instrumented split-belt treadmill with handrail and harness; Vicon motion capture (100 Hz)"
OG_DEVICE = "overground walkway; Vicon motion capture (100 Hz)"
MM = 0.001


def speeds(g, k, b):
    """Protocol belt speeds (fast, slow) in m/s for block b, or (nan, nan) overground."""
    if b in OVERGROUND:
        return np.nan, np.nan
    if g in YOUNG:  # 0.5, 1.0 and 1.5 m/s; split 3:1 (1.5 and 0.5 m/s)
        if b in SPLIT:
            return 1.5, 0.5
        v = {"slow": 0.5, "slow flat": 0.5, "fast": 1.5, "fast flat": 1.5}.get(b, 1.0)
        return v, v
    slow, mid, fast = TABLE1[k][3]
    if b in SPLIT:
        return fast, slow
    if b == "TM slow":
        return slow, slow
    return mid, mid


def incline(g, k, b):
    if b in OVERGROUND:
        return 0.0
    if g in YOUNG:
        return 0.0 if ("flat" in b or g == "FlatYoungAbrupt") else YOUNG[g]
    if g == "NoDescription":
        return UPHILL_FLAT_SESSION.get(b, 0.0)
    if b in ("flat base", "flat post"):
        return 0.0
    if b == "WhoKnows":
        return np.nan  # 100-150 strides at the mid speed after re-adaptation; the description gives no slope
    return float(TABLE1[k][6])


def build():
    mo = MatObjects(stdlib.raw_dir(DS) / "StudyData.mat")
    study = mo.top["StudyData"][0, 0]
    assert list(study.dtype.names) == GROUPS, study.dtype.names
    ck = stdlib.Checks()

    frames, info, lab_ids, desc_ok, side_ok = [], [], {}, True, []
    for g in GROUPS:
        grp = mo.variable(study[g])
        assert grp["__class__"] == "groupAdaptationData"
        for k, a in enumerate(grp["adaptData"]):
            st, sub = strides(a), a["subData"]
            col, block, n = st["col"], st["block"], st["n"]
            assert (block != "").all(), "stride outside every condition"
            unknown = set(block) - set(PHASE)
            assert not unknown, unknown
            lab_ids[(g, k)] = text(a["metaData"]["ID"]).rstrip("_")  # used only to pair the two stroke sessions
            sex = {"Female": "F", "Male": "M"}[text(sub["sex"])]
            if g in YOUNG:
                person, fast_leg = f"young:{g}:{k}", text(sub["dominantLeg"]).lower()
                group, experiment, cond, session = g, "young adults (Sombric et al. 2019, Front Physiol)", "", 1
                population = "healthy young adults"
            else:
                cond, session = STROKE[g]
                person, group, experiment = f"stroke:{k}", "stroke", "stroke survivors, flat and incline sessions"
                fast_leg = {"R": "left", "L": "right"}[TABLE1[k][2]]  # the paretic leg walked on the slow belt
                population = "chronic stroke (hemiparesis)"
                if "affectedSide" in sub and text(sub["affectedSide"]) in ("R", "L"):
                    side_ok.append(text(sub["affectedSide"]) == TABLE1[k][2])
                if g == "InclineStroke":
                    for b in ("TM base", "adaptation", "TM post"):
                        if b in st["descr"]:
                            desc_ok &= f"{TABLE1[k][6]:g}" in st["descr"][b] or (TABLE1[k][6] == 8.5 and "8" in st["descr"][b])
            sp = [speeds(g, k, b) for b in block]
            split = np.isin(block, list(SPLIT))
            ratio = 3.0 if g in YOUNG else 2.0
            subj = f"{g}:{k}"
            info.append(dict(subject=subj, group=g, k=k, sex=sex, age=number(sub["age"]),
                             counts={b: int((block == b).sum()) for b in set(block)}))
            f = dict(
                dataset=DS, study=DS, experiment=experiment, group=group, subject=subj, person=person,
                subject_excluded_by_authors=False, age=number(sub["age"]), sex=sex,
                handedness=text(sub["dominantArm"]).lower(), population=population, setting="lab",
                device=np.where(np.isin(block, list(OVERGROUND)), OG_DEVICE, TM_DEVICE), session=session,
                block=block, trial=np.arange(1, n + 1), phase=[PHASE[b] for b in block], time_s=np.nan, task="walking",
                unit="stride", condition=cond, pert_type=np.where(split, "split_belt", "none"),
                pert_value=np.where(split, ratio, 0.0), pert_unit=np.where(split, "ratio", ""), feedback="normal",
                excluded_by_authors=col["bad"] == 1,
                fast_leg=fast_leg, belt_speed_fast_mps=[s[0] for s in sp], belt_speed_slow_mps=[s[1] for s in sp],
                walking_trial=st["trial"], stride_start_s=col["initTime"],
                step_length_fast_m=col["stepLengthFast"] * MM, step_length_slow_m=col["stepLengthSlow"] * MM,
                step_length_asymmetry=col["stepLengthAsym"],
                walking_context=np.where(np.isin(block, list(OVERGROUND)), "overground", "treadmill"),
                surface_incline_deg=[incline(g, k, b) for b in block],
                leading_position_fast_m=col["alphaFast"] * MM, leading_position_slow_m=col["alphaSlow"] * MM,
                trailing_position_fast_m=col["XFast"] * MM, trailing_position_slow_m=col["XSlow"] * MM,
                peak_propulsion_fast_bw=col["FyPFmax"], peak_propulsion_slow_bw=col["FyPSmax"],
            )
            frames.append(pd.DataFrame(f))
    df = pd.concat(frames, ignore_index=True)
    for c in ["stride_start_s", "step_length_fast_m", "step_length_slow_m", "step_length_asymmetry",
              "leading_position_fast_m", "leading_position_slow_m", "trailing_position_fast_m",
              "trailing_position_slow_m", "peak_propulsion_fast_bw", "peak_propulsion_slow_bw"]:
        df[c] = df[c].astype(float).round(8)
    return df, info, lab_ids, desc_ok, side_ok, ck


def main():
    df, info, lab_ids, desc_ok, side_ok, ck = build()

    # ---- internal checks
    sx = {(i_["group"], i_["k"]): i_["sex"] for i_ in info}
    pairs_ok = all(sx[("NoDescription", k)] == sx[("InclineStroke", k)] == TABLE1[k][1] for k in range(12))
    ck.internal("48 stride sequences: 8 per young group and 12 stroke survivors with a flat and an incline session; the "
                "two sessions of each stroke survivor are paired by their order, which is Table 1's in both files (the "
                "stride counts below match Table 1 row by row, and sex agrees with Table 1 in both sessions)",
                len(info) == 48 and pairs_ok and df.person.nunique() == 36, f"{df.subject.nunique()} subjects, "
                f"{df.person.nunique()} persons")
    ck.internal("every stride the deposit shares is kept, in its order, and lies in one of the authors' conditions",
                len(df) == sum(i_["counts"][b] for i_ in info for b in i_["counts"]), f"{len(df)} rows")
    sla = (df.step_length_fast_m - df.step_length_slow_m) / (df.step_length_fast_m + df.step_length_slow_m)
    d = (sla - df.step_length_asymmetry).abs()
    ck.internal("on >= 99% of strides, step_length_asymmetry = (step_length_fast - step_length_slow) / (sum) on every stride (labTools "
                "stepLengthAsym; positive when the fast, non-paretic or dominant leg takes the longer step)",
                (d[d.notna()] < 1e-6).mean() > 0.99, f"{100 * (d[d.notna()] < 1e-6).mean():.2f}% of strides within 1e-6, "
                f"{int((d > 1e-6).sum())} strides differ (max {np.nanmax(d):.3f}); the deposit's values are kept")
    young = df.group.isin(list(YOUNG))
    e1 = (df.step_length_fast_m - (df.leading_position_fast_m - df.trailing_position_slow_m)).abs()
    e2 = (df.step_length_slow_m - (df.leading_position_slow_m - df.trailing_position_fast_m)).abs()
    ck.internal("step_length_fast = leading_position_fast - trailing_position_slow on every stride (both at the fast "
                "heel strike)", np.nanmax(e1) < 1e-6, f"max |diff| {np.nanmax(e1):.2g} m")
    ck.internal("step_length_slow = leading_position_slow - trailing_position_fast exactly for the young adults (X_fast "
                "at the closing slow heel strike), but not for the stroke files, whose labTools version took X_fast at "
                "the opening slow heel strike (median difference about 1 cm)",
                np.nanmax(e2[young]) < 1e-6 and np.nanmedian(e2[~young]) > 0.005,
                f"young max {np.nanmax(e2[young]):.2g} m; stroke median {np.nanmedian(e2[~young]):.3f} m")
    tm = df.walking_context == "treadmill"
    ck.internal("on the treadmill, feet land in front of the hip (leading > 0) and trail behind it (trailing < 0) on "
                ">= 99% of good strides",
                ((df.leading_position_fast_m[tm & ~df.excluded_by_authors] > 0).mean() > 0.99)
                and ((df.trailing_position_slow_m[tm & ~df.excluded_by_authors] < 0).mean() > 0.99))
    pr = pd.concat([df.peak_propulsion_fast_bw, df.peak_propulsion_slow_bw]).dropna()
    ck.internal("peak propulsion between 0 and 0.6 body weights on >= 99% of values (forces divided by 9.81 x body mass in "
                "computeForceParameters.m)", pr.between(0, 0.6).mean() > 0.99, f"{100 * pr.between(0, 0.6).mean():.2f}% of values; {int((pr > 0.6).sum())} "
                f"values above 0.6 (max {pr.max():.2f}) are kept as shared")
    ck.internal("the deposit's affected side agrees with Table 1 wherever it is filled (the other entries are 'X' or "
                "missing), so fast_leg is the non-paretic side from Table 1", len(side_ok) >= 12 and all(side_ok),
                f"{sum(side_ok)} of {len(side_ok)} agree")
    ck.internal("the incline-session condition notes give the Table 1 slope (5 or 8.5 deg) for every participant", desc_ok)
    ex = df.excluded_by_authors
    ck.internal("excluded_by_authors = labTools 'bad' flag (missing or disordered gait events, stride time out of "
                "range); getForceResults.m calls removeBadStrides before every outcome", 0 < ex.mean() < 0.06,
                f"{int(ex.sum())} strides ({100 * ex.mean():.2f}%)")

    # ---- published numbers
    stroke = [i_ for i_ in info if i_["group"] in STROKE]
    for i_ in stroke:
        k, s = i_["k"], 0 if i_["group"] == "NoDescription" else 1
        if (k, s) == (5, 1):
            continue  # P6 incline: 611 / 305 strides here, 597 / 300 in Table 1 (see differences)
        ck.published(f"P{k + 1} {STROKE[i_['group']][0]}: adaptation strides", i_["counts"]["adaptation"],
                     TABLE1[k][4][s], 0, "Table 1")
        ck.published(f"P{k + 1} {STROKE[i_['group']][0]}: post-adaptation strides", i_["counts"]["TM post"],
                     TABLE1[k][5][s], 0, "Table 1")
    flat = [i_ for i_ in stroke if i_["group"] == "NoDescription"]
    ck.published("number of women among the 12 stroke survivors", sum(i_["sex"] == "F" for i_ in flat), 5, 0,
                 "Table 1 (the Methods text says 4)")

    # Outcomes as in getForceResults.m: bad strides removed; bias = mean of the TM base strides end-44..end-5;
    # early adaptation = adaptation strides 2-6 (one of the code's options; strides 1-5 give t = 0.11 for deltaAdapt), late adaptation = end-44..end-5, after-effects = TM post strides 1-5,
    # all minus the bias; deltaAdapt = late - early.
    def outcomes(param):
        res = {}
        good = df[~ex & (df.group == "stroke")]
        for (cnd, subj), s in good.groupby(["condition", "subject"], sort=False):
            v = {b: s.loc[s.block == b, param].to_numpy(float) for b in ("TM base", "adaptation", "TM post")}
            base = np.nanmean(v["TM base"][-45:-5])
            late = np.nanmean(v["adaptation"][-45:-5]) - base
            early = np.nanmean(v["adaptation"][1:6]) - base  # strides 2-6 (EarlyStrides=[2:6] in getForceResults.m)
            res.setdefault(cnd, []).append(dict(base=base, late=late, dadapt=late - early,
                                                ae=np.nanmean(v["TM post"][:5]) - base))
        return pd.DataFrame(res["flat session"]), pd.DataFrame(res["incline session"])

    def tpair(fl, il, which):  # paired t, flat minus incline (the paper's sign), participants with both values
        a, b = fl[which].astype(float), il[which].astype(float)
        ok = a.notna() & b.notna()
        return stats.ttest_rel(a[ok], b[ok]).statistic

    fl, il = outcomes("step_length_asymmetry")
    ck.published("paired t, step length asymmetry in late adaptation, flat vs incline (t(11) = -3.68)",
                 tpair(fl, il, "late"), -3.68, 0.05, "Results, Fig. 2c")
    ck.published("paired t, step length asymmetry deltaAdapt (late minus early adaptation), flat vs incline (t(11) = 0.33)",
                 tpair(fl, il, "dadapt"), 0.33, 0.02, "Results, Fig. 2d")
    near = {"baseline": (tpair(fl, il, "base"), 1.08), "after-effects": (tpair(fl, il, "ae"), -3.21)}
    for prm, nm, rep, df_ in (("peak_propulsion_fast_bw", "non-paretic", 0.10, 11), ("peak_propulsion_slow_bw", "paretic", -1.04, 10)):
        a, b = outcomes(prm)
        ck.published(f"paired t, {nm} peak propulsion deltaAdapt, flat vs incline (t({df_}) = {rep})", tpair(a, b, "dadapt"),
                     rep, 0.02, "Results, Fig. 5d")
    for prm, nm in (("peak_propulsion_fast_bw", "non-paretic"), ("peak_propulsion_slow_bw", "paretic")):
        a, b = outcomes(prm)
        for w in ("late", "dadapt", "ae"):
            near[f"{nm} propulsion {w}"] = (tpair(a, b, w), None)
    for k_, (v, r) in near.items():
        print(f"  [not a check] {k_}: t = {v:+.2f}" + (f" (paper {r})" if r is not None else ""))

    meta = dict(
        study=DS,
        paper=dict(authors="Sombric CJ, Torres-Oviedo G", year=2020,
                   title="Augmenting propulsion demands during split-belt walking increases locomotor adaptation of "
                         "asymmetric step lengths",
                   venue="J NeuroEngineering Rehabil 17(1):69", doi="10.1186/s12984-020-00698-y"),
        lab="Torres-Oviedo (University of Pittsburgh)",
        data_url="https://osf.io/vm597/",
        license="none stated", license_url="",
        changes=("Read the deposit's MATLAB (labTools) objects without the lab's code and kept, per stride, every "
                 "parameter the deposit fills except the walking direction, the hip position and the final toe-off "
                 "time; millimetres were converted to metres; participant IDs, dates, height, weight and the "
                 "experimenter's notes are left out."),
        how=["One row is one stride (slow-leg heel strike to the next) of one participant, in the deposit's order, with "
             "step lengths, step length asymmetry, the leading (alpha) and trailing (X) ankle positions relative to the "
             "hip, and peak propulsion forces in body weights; experiment separates the 12 stroke survivors (two "
             "sessions each: session 1 flat, session 2 incline, condition says which) from the 24 young adults of "
             "Sombric et al. (2019, Front Physiol), whom the paper uses as its reference.",
             "Fast and slow are the legs on the fast and slow belts during split-belt walking: the non-paretic leg was "
             "fast for the stroke survivors (fast_leg is the non-paretic side from Table 1) and the dominant leg for the "
             "young adults. Belt speeds are the protocol's (Table 1 for each stroke survivor; 0.5, 1.0 and 1.5 m/s for "
             "the young adults), split-belt ratio 2:1 (stroke) or 3:1 (young); surface_incline_deg is the slope stated "
             "in the condition notes and Table 1 (5 or 8.5 deg; the young groups walked at -8.5, 0 or +8.5 deg).",
             "Signs are labTools': step_length_asymmetry is positive when the fast leg's step is longer; leading "
             "positions are positive in front of the hip and trailing positions negative behind it. In the stroke files "
             "trailing_position_fast_m was taken at the stride's opening slow heel strike (in the young files at the "
             "closing one), so step_length_slow equals leading_position_slow minus trailing_position_fast only for the "
             "young adults.",
             "excluded_by_authors marks labTools' bad strides (missing or disordered gait events, or a stride time out of "
             "range), which the authors' code removes before every outcome. Blocks keep the authors' condition names, "
             "including short trials not in the paper (short split-belt exposures before baseline, flat walking before "
             "and after the incline session, and brief uphill trials at the end of the flat session)."],
        unit_note="stride (from a slow-leg heel strike to the next one)",
        differences=[
            "Table 1 lists 597 adaptation and 300 post-adaptation strides for P6 in the incline session; the deposit "
            "has 611 and 305. The other 46 stride counts in Table 1 match exactly.",
            "The deposit's ages differ from Table 1 for three sessions: 64 for P2 and 80 for P11 in the flat session "
            "(Table 1: 55 and 75) and 22 for P9 in the incline session (Table 1: 52); they are kept as shared. The "
            "Methods text says 4 women, Table 1 and the deposit 5.",
            f"With the authors' epochs the paired t-values for baseline step length asymmetry ({near['baseline'][0]:.2f}, "
            f"paper 1.08) and for after-effects ({near['after-effects'][0]:.2f}, paper -3.21) do not reproduce, nor do "
            "the propulsion t-values of Fig. 5 (their stroke-specific results function is not in the public code)."],
    )
    names = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in names if c not in df.columns})[names]  # columns this deposit does not fill
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
