"""Sombric, Harker, Sparto & Torres-Oviedo (2017) Front Aging Neurosci: split-belt adaptation, catch, re-adaptation and
overground transfer in young and older adults, on a first visit and (for 8 + 8) a second visit about 6 weeks later; one row
per stride.

Raw files (OSF yh48z, folder 'Data for Reproducibility', no license stated): TreadmillData.zip and OvergroundData.zip, each
with the same 38 session files OG<n>params.mat (labTools adaptationData with a paramData stride table), processed for the
treadmill and for the overground analyses; FinalCogScores.mat (a 92 x 11 matrix without column names, left out).
Treadmill strides are taken from TreadmillData and overground strides from OvergroundData; the other copy of each is left
out. Files named OG2<n> are second visits of OG<n>. Decoded with Sombric2017_FrontAgingNeurosci_labtools.py (the pilot's
reader); nothing from the deposit or labTools is executed.

Run from the site folder: python3 -I tools/std/convert/Sombric2017_FrontAgingNeurosci.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Sombric2017_FrontAgingNeurosci_labtools import MatObjects, number, strides, text  # noqa: E402

DS = "Sombric2017_FrontAgingNeurosci"
MODULE = "walking"
PHASE = {"OG base": "baseline", "slow base": "baseline", "fast base": "baseline", "TM base": "baseline",
         "short split": "practice", "adaptation": "learning", "catch": "aftereffect", "re-adaptation": "learning",
         "OG post": "transfer", "TM post": "washout", "calibration": "other"}
SPLIT = {"short split", "adaptation", "re-adaptation"}
SPEED = {"slow base": (0.5, 0.5), "fast base": (1.0, 1.0), "TM base": (0.75, 0.75), "catch": (0.75, 0.75),
         "TM post": (0.75, 0.75), "short split": (1.0, 0.5), "adaptation": (1.0, 0.5), "re-adaptation": (1.0, 0.5)}
MM = 0.001
COLS = {"stride_start_s": ("initTime", 1), "stride_time_s": ("strideTimeSlow", 1),
        "step_length_fast_m": ("stepLengthFast", MM), "step_length_slow_m": ("stepLengthSlow", MM),
        "step_time_fast_s": ("stepTimeFast", 1), "step_time_slow_s": ("stepTimeSlow", 1),
        "stance_time_fast_s": ("stanceTimeFast", 1), "stance_time_slow_s": ("stanceTimeSlow", 1),
        "double_support_fast_s": ("doubleSupportFast", 1), "double_support_slow_s": ("doubleSupportSlow", 1),
        "step_length_asymmetry": ("stepLengthAsym", 1), "step_position_asymmetry": ("Sout", 1),
        "step_time_asymmetry": ("Tout", 1), "leading_position_fast_m": ("alphaFast", MM),
        "leading_position_slow_m": ("alphaSlow", MM), "trailing_position_fast_m": ("XFast", MM),
        "trailing_position_slow_m": ("XSlow", MM)}
CONTRIB = {"step_position_contribution": "spatialContribution", "step_time_contribution": "stepTimeContribution",
           "velocity_contribution": "velocityContribution"}


def load(z, name):
    mo = MatObjects(z.read(name))
    a = mo.variable(mo.top["adaptData"])
    return a, strides(a), [text(t) for t in a["data"]["trialTypes"]]


def main():
    raw = stdlib.raw_dir(DS)
    ztm, zog = zipfile.ZipFile(raw / "TreadmillData.zip"), zipfile.ZipFile(raw / "OvergroundData.zip")
    names = [i.filename for i in ztm.infolist()]
    assert sorted(names) == sorted(i.filename for i in zog.infolist()) and len(names) == 38
    num = lambda s: int(s[2:-10])  # noqa: E731  OG<n>params.mat -> n
    visit2 = {n for n in names if num(n) >= 200}
    first = {n for n in names if num(n) < 200}
    assert all(f"OG{num(n) - 200}params.mat" in first for n in visit2) and len(visit2) == 16
    ck = stdlib.Checks()
    frames, info = [], []
    for n in sorted(first, key=num) + sorted(visit2, key=num):
        a, s_tm, types = load(ztm, n)
        _, s_og, types_og = load(zog, n)
        assert types == types_og, n
        sub = a["subData"]
        age, sex = number(sub["age"]), {"Female": "F", "Male": "M"}.get(text(sub["sex"]).strip(), "")
        parts = []
        for src, kind in ((s_tm, "TM"), (s_og, "OG")):
            keep = np.array([types[t - 1] == kind for t in src["trial"]])
            d = {c: (src["col"][lab][keep] * f if lab in src["col"] else np.full(keep.sum(), np.nan)) for c, (lab, f) in COLS.items()}
            slsum = src["col"]["stepLengthFast"][keep] + src["col"]["stepLengthSlow"][keep]
            for c, lab in CONTRIB.items():
                norm = lab + "Norm2"
                d[c] = src["col"][norm][keep] if norm in src["col"] else src["col"][lab][keep] / slsum
            d.update(walking_trial=src["trial"][keep], block=src["block"][keep],
                     excluded_by_authors=src["col"]["good"][keep] == 0 if "good" in src["col"] else False,
                     _src=kind, _i=np.arange(keep.sum()))
            parts.append(pd.DataFrame(d))
        f = pd.concat(parts, ignore_index=True).sort_values(["walking_trial", "_i"], kind="stable").reset_index(drop=True)
        assert (f.block != "").all() and not (set(f.block) - set(PHASE)), set(f.block) - set(PHASE)
        older = age > 60
        v = 2 if n in visit2 else 1
        person = f"OG{num(n) - 200 if v == 2 else num(n)}"  # internal key only; persons are re-coded by release()
        info.append(dict(group="older" if older else "young", visit=v, sex=sex, age=age, n=len(f)))
        b = f.block.to_numpy()
        split = np.isin(b, list(SPLIT))
        og = f._src.to_numpy() == "OG"
        f = f.drop(columns=["_src", "_i"]).assign(
            dataset=DS, study=DS, experiment="", group="older adults" if older else "young adults",
            subject=f"{n}", person=person, subject_excluded_by_authors=False, age=age, sex=sex,
            handedness=text(sub["dominantArm"]).lower(), population="healthy older adults" if older else "healthy young adults",
            setting="lab", device=np.where(og, "9.2 m overground walkway; Vicon motion capture (100 Hz)",
                                           "Bertec split-belt treadmill; Vicon motion capture (100 Hz)"),
            session=v, trial=np.arange(1, len(f) + 1), phase=[PHASE[x] for x in b], time_s=np.nan, task="walking",
            unit="stride", condition=f"visit {v}", pert_type=np.where(split, "split_belt", "none"),
            pert_value=np.where(split, 2.0, 0.0), pert_unit=np.where(split, "ratio", ""), feedback="normal",
            fast_leg=text(sub["dominantLeg"]).lower(),
            belt_speed_fast_mps=[np.nan if o else SPEED.get(x, (np.nan, np.nan))[0] for x, o in zip(b, og)],
            belt_speed_slow_mps=[np.nan if o else SPEED.get(x, (np.nan, np.nan))[1] for x, o in zip(b, og)],
            walking_context=np.where(og, "overground", "treadmill"), surface_incline_deg=0.0)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    for c in list(COLS) + list(CONTRIB):
        df[c] = df[c].astype(float).round(8)
    ex = df.excluded_by_authors.astype(bool)
    df["excluded_by_authors"] = ex

    I = pd.DataFrame(info)
    ck.internal("38 sessions: 22 first visits (11 young, 11 older) and 16 second visits (8 + 8); second-visit files OG2<n> "
                "pair with first-visit files OG<n>", len(I) == 38 and df.person.nunique() == 22,
                str(I.groupby(["visit", "group"]).size().to_dict()))
    ck.internal("treadmill strides from TreadmillData, overground strides from OvergroundData, each in recording order",
                len(df) == I.n.sum(), f"{len(df)} rows; {int((df.walking_context == 'overground').sum())} overground")
    d = ((df.step_length_fast_m - df.step_length_slow_m) / (df.step_length_fast_m + df.step_length_slow_m) - df.step_length_asymmetry).abs().dropna()
    ck.internal("step_length_asymmetry = (step_length_fast - step_length_slow) / sum on >= 98% of strides",
                (d < 1e-6).mean() > 0.98, f"{100 * (d < 1e-6).mean():.2f}% of {len(d)}")
    d = (df.step_position_contribution + df.step_time_contribution + df.velocity_contribution - df.step_length_asymmetry).abs().dropna()
    ck.internal("the three contributions add up to step_length_asymmetry on >= 98% of strides", (d < 1e-6).mean() > 0.98,
                f"{100 * (d < 1e-6).mean():.2f}% of {len(d)}")
    dd = ((df.step_time_fast_s - df.step_time_slow_s) / df.stride_time_s - df.step_time_asymmetry).abs().dropna()
    ck.internal("step_time_asymmetry (Tout) = (step_time_fast - step_time_slow) / stride_time on >= 98% of strides",
                (dd < 2e-3).mean() > 0.98, f"{100 * (dd < 2e-3).mean():.2f}% of {len(dd)}")

    diffs = []
    for v, g, nn, women, m, sd in ((1, "young", 11, 5, 25.4, 5.5), (1, "older", 11, 4, 77.2, 2.8),
                                   (2, "young", 8, 4, 27.0, 5.7), (2, "older", 8, 2, 76.6, 2.5)):
        s = I[(I.visit == v) & (I.group == g)]
        ck.published(f"visit {v}, {g} adults: number", len(s), nn, 0, "Methods, Participants")
        for what, val, rep, tol in ((f"visit {v}, {g} adults: women", (s.sex == "F").sum(), women, 0),
                                    (f"visit {v}, {g} adults: mean age", s.age.mean(), m, 0.05),
                                    (f"visit {v}, {g} adults: SD of age", s.age.std(ddof=1), sd, 0.05)):
            if abs(val - rep) <= tol:
                ck.published(what, val, rep, tol, "Methods, Participants")
            else:
                diffs.append(f"{what}: {val:.2f} here, {rep} in the paper")
    print("\\n".join(diffs))

    meta = dict(
        study=DS,
        paper=dict(authors="Sombric CJ, Harker HM, Sparto PJ, Torres-Oviedo G", year=2017,
                   title="Explicit Action Switching Interferes with the Context-Specificity of Motor Memories in Older Adults",
                   venue="Front Aging Neurosci 9:40", doi="10.3389/fnagi.2017.00040"),
        lab="Torres-Oviedo (University of Pittsburgh)", data_url="https://osf.io/yh48z/", license="none stated", license_url="",
        changes=("Read the deposit's MATLAB (labTools) objects without the lab's code. Each session is shared twice (TreadmillData "
                 "and OvergroundData, processed for the treadmill and the overground analyses); treadmill strides are taken "
                 "from the first and overground strides from the second. Kept per stride: gait timing, step lengths, "
                 "leading and trailing ankle positions, the asymmetries that match the module's formulas and the "
                 "step-position, step-time and velocity contributions (in TreadmillData only in mm, here divided by the "
                 "stride's step-length sum, as the paper normalizes them). Millimetres converted to metres; the other labTools "
                 "parameters, the unlabelled cognitive-score matrix, participant IDs, dates, height and weight are left out."),
        how=["One row is one stride (slow-leg heel strike to the next) of one participant in one visit, in recording order. "
             "session and condition give the visit (1, or 2 about six weeks later for 8 young and 8 older adults); a "
             "person's two visits share the person code.",
             "Fast is the dominant leg (the leg used to kick a ball), on the 1.0 m/s belt during split-belt walking (slow "
             "belt 0.5 m/s, tied 0.5, 0.75 or 1.0 m/s as the block says); catch is 10 tied strides at 0.75 m/s; OG blocks "
             "are 6 minutes on a 9.2 m walkway. Treadmill strides have no stride start time or trailing positions (not in "
             "TreadmillData).",
             "Signs are labTools': asymmetries are fast minus slow; the three contributions (StepPosition, StepTime, "
             "StepVelocity in the paper) add up to step_length_asymmetry.",
             "excluded_by_authors is labTools' good flag inverted (no stride is flagged in these files); the paper's own "
             "overground bias removal (by position on the walkway) and removal of turning steps are not applied here."],
        unit_note="stride (from a slow-leg heel strike to the next one)",
        differences=diffs + ["No outcome statistic was checked: the overground measures need the authors' position-by-position "
                             "bias removal, which the deposit does not document."],
    )
    cols = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in cols if c not in df.columns})[cols]
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
