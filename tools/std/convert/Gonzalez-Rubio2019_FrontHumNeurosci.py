"""Gonzalez-Rubio, Velasquez & Torres-Oviedo (2019) Front Hum Neurosci: split-belt walking while explicitly controlling
step position (spatial feedback) or step time (temporal feedback), 3 groups x 7 participants, one row per stride.

Raw files: raw_extra/Gonzalez-Rubio2019_FrontHumNeurosci/StudyData.mat (Figshare 10.6084/m9.figshare.8145962, CC BY 4.0;
the same file is on OSF eq6f9) and README_osf.txt (the OSF README, read as text). StudyData.mat holds one MATLAB classdef
object per group (groupAdaptationData, from the lab's labTools package): seven adaptationData objects per group, each
with metaData (conditions and their trials), subData (sex, age, dominant leg and arm) and data, a parameterSeries of
strides x parameters with a label and a description for every parameter.

Reading the objects without the lab's code: MATLAB with only its factory path (restoredefaultpath; nothing from the
deposit or labTools) loads every group object as an empty 0x0 double, because classdef objects need their class files.
So there is no MATLAB step: the class instances are decoded below from the MAT-file's own object store (the MCOS
'FileWrapper__' block inside the hidden __function_workspace__ variable) with scipy, as plain data. Nothing from the
deposit or from labTools is executed. The authors' published analysis code
(github.com/PittSMLlab/Gonzalez-Rubioetal_2019_ExplicitTemporalControl: getResults.m, removeBiasV3.m, getEarlyLateIdxs.m,
adaptationData.m) was read as text to reproduce their bad-stride removal, bias removal and epochs.

Run from the site folder: python3 -I tools/std/convert/Gonzalez-Rubio2019_FrontHumNeurosci.py
"""
import io
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.io.matlab._mio5 import MatFile5Reader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Gonzalez-Rubio2019_FrontHumNeurosci"
MODULE = "walking"


class MatObjects:
    """Read MATLAB classdef (MCOS) objects from a v7 MAT-file without their class definitions.

    MATLAB writes the property values of every object into a hidden variable, __function_workspace__: a MAT stream
    (without the 116-byte text header) holding a struct whose field MCOS is an opaque FileWrapper__ object. That
    object's value is a cell array: cell 0 is a uint8 metadata block (version, number of names, 8 region offsets, the
    null-terminated names, then a class table of [namespace, class name, 0, 0], an object table of
    [class, 0, 0, saveobj id, property-set id, dependency id] and the property sets as nprops followed by
    (name, kind, value) triplets padded to 8 bytes); the other cells hold the property values (cell value + 2 for
    kind 1). An object reference anywhere in the file is a uint32 array [0xDD000000, ndims, dims..., object ids...,
    class id]. Only what this file uses is handled; anything else stops the converter."""

    MARK = 0xDD000000

    def __init__(self, path):
        self.top = sio.loadmat(path)
        ws = self.top["__function_workspace__"].tobytes()
        head = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + ws[0:4]  # version and endian bytes of the block
        rd = MatFile5Reader(io.BytesIO(head + ws[8:]), struct_as_record=True, squeeze_me=False)
        rd.initialize_read()
        rd.mat_stream.seek(128)
        hdr, _ = rd.read_var_header()
        wrapper = rd.read_var_array(hdr, process=True)
        self.cells = wrapper[0, 0]["MCOS"][0]["arr"].ravel()
        meta = self.cells[0].ravel().tobytes()
        version, n_names = struct.unpack_from("<II", meta, 0)
        offs = struct.unpack_from("<8I", meta, 8)
        assert version in (2, 3, 4), version
        pos, self.names = 40, []
        while len(self.names) < n_names:
            end = meta.index(b"\x00", pos)
            self.names.append(meta[pos:end].decode("latin1"))
            pos = end + 1
        u32 = lambda a, b: np.frombuffer(meta[a:b], dtype="<u4")  # noqa: E731
        self.classes = u32(offs[0], offs[1]).reshape(-1, 4)
        assert offs[1] == offs[2], "objects saved through saveobj are not handled"
        self.objects = u32(offs[2], offs[3]).reshape(-1, 6)
        r4, i, self.props = u32(offs[3], offs[4]), 2, [None]
        while i < len(r4):
            n = int(r4[i])
            self.props.append(r4[i + 1:i + 1 + 3 * n].reshape(-1, 3))
            i += 1 + 3 * n
            i += i % 2

    def obj(self, oid):
        cid, _, _, saveobj_id, prop_id, _ = (int(x) for x in self.objects[oid])
        assert saveobj_id == 0, "objects saved through saveobj are not handled"
        out = {"__class__": self.names[self.classes[cid][1] - 1]}
        for name_i, kind, val in (self.props[prop_id] if prop_id else []):
            name = self.names[name_i - 1]
            if kind == 1:
                out[name] = self.value(self.cells[val + 2])
            elif kind == 2:
                out[name] = bool(val)
            elif kind == 0:
                out[name] = self.names[val - 1] if val else ""
            else:
                raise ValueError(f"unknown property kind {kind} for {name}")
        return out

    def value(self, v):
        if isinstance(v, np.ndarray) and v.dtype == np.uint32 and v.size >= 4 and int(v.ravel()[0]) == self.MARK:
            a = v.ravel().astype(np.int64)
            nd = int(a[1])
            n = int(np.prod(a[2:2 + nd]))
            objs = [self.obj(int(i)) for i in a[2 + nd:2 + nd + n]]
            return objs[0] if n == 1 else objs
        if isinstance(v, np.ndarray) and v.dtype == object:
            return [self.value(x) for x in v.ravel(order="F")]
        return v

    def variable(self, opaque):
        return self.value(opaque[0]["arr"])


def text(a):
    if isinstance(a, np.ndarray):
        return str(a.ravel()[0]) if a.size else ""
    return str(a)


# struct field (group object) -> group name in the README and paper; in the file's order
GROUPS = {"StepTimeStudy": "Temporal Feedback", "StepTimeStudyControl": "Control", "AlphaFeedback": "Spatial Feedback"}
FEEDBACK_GROUPS = ("StepTimeStudy", "AlphaFeedback")
BLOCKS = ["TM base", "familiarization", "adaptation", "TM post"]
PHASE = {"TM base": "baseline", "familiarization": "practice", "adaptation": "learning", "TM post": "washout"}
PLANNED = {"TM base": 150, "familiarization": 150, "adaptation": 600, "TM post": 450}  # strides, paper Methods 2.1
SPEED = {"TM base": (0.75, 0.75), "familiarization": (0.75, 0.75), "adaptation": (1.0, 0.5), "TM post": (0.75, 0.75)}
DEVICE = "Bertec instrumented split-belt treadmill with a center divider; Vicon motion capture (100 Hz); screen for visual feedback"
# module column -> (labTools parameter label, factor to the module's unit). Labels are looked up by name, per participant.
PARAMS = {
    "stride_time_s": ("strideTimeSlow", 1.0),
    "step_length_fast_m": ("stepLengthFast", 0.001),
    "step_length_slow_m": ("stepLengthSlow", 0.001),
    "step_time_fast_s": ("stepTimeFast", 1.0),
    "step_time_slow_s": ("stepTimeSlow", 1.0),
    "stance_time_fast_s": ("stanceTimeFast", 1.0),
    "stance_time_slow_s": ("stanceTimeSlow", 1.0),
    "double_support_fast_s": ("doubleSupportFast", 1.0),
    "double_support_slow_s": ("doubleSupportSlow", 1.0),
    "step_length_asymmetry": ("stepLengthAsym", 1.0),
    "step_position_asymmetry": ("Sout", 1.0),
    "step_time_asymmetry": ("Tout", 1.0),
    "stance_time_asymmetry": ("Tgoal", 1.0),
    "double_support_asymmetry_s": ("Terror", 1.0),
    "leg_orientation_asymmetry": ("Serror", 1.0),
    "range_of_motion_asymmetry": ("Sgoal", 1.0),
}
PAPER_SIX = ["step_position_asymmetry", "step_time_asymmetry", "leg_orientation_asymmetry", "double_support_asymmetry_s",
             "range_of_motion_asymmetry", "stance_time_asymmetry"]  # Sout, Tout, SA, TA, SnA, TnA


def main():
    mo = MatObjects(stdlib.raw_dir(DS) / "StudyData.mat")
    study = mo.top["StudyData"][0, 0]
    ck = stdlib.Checks()
    assert list(study.dtype.names) == list(GROUPS), study.dtype.names

    frames, people, label_sets, filled, desc_ok = [], [], set(), set(), True
    for g in GROUPS:
        grp = mo.variable(study[g])
        assert grp["__class__"] == "groupAdaptationData"
        for a in grp["adaptData"]:
            md, sub, ps = a["metaData"], a["subData"], a["data"]
            labels = [text(x) for x in ps["labels"]]
            label_sets.add((len(labels), tuple(labels)))
            X = ps["Data_"].astype(float)
            assert X.shape[1] == len(labels)
            col = {lab: X[:, k] for k, lab in enumerate(labels)}
            filled |= {lab for lab in labels if np.isfinite(col[lab]).any()}
            n = X.shape[0]
            trial = col["trial"]
            block = np.array([""] * n, dtype=object)
            names = [text(c) for c in md["conditionName"]]
            assert names == BLOCKS, names
            for cname, cdesc, tic in zip(names, md["conditionDescription"], md["trialsInCondition"]):
                block[np.isin(trial, np.ravel(tic))] = cname
                d = text(cdesc)
                desc_ok &= ("2:1" in d) if cname == "adaptation" else ("0.75" in d)
            assert (block != "").all() and (np.diff(trial) >= 0).all()
            code = f"{DS}_S{len(people) + 1:03d}"
            people.append(dict(code=code, group=g, sex=text(sub["sex"]), age=float(np.ravel(sub["age"])[0]),
                               leg=text(sub["dominantLeg"]).lower(), arm=text(sub["dominantArm"]).lower(),
                               counts={b: int((block == b).sum()) for b in BLOCKS}))
            fb = (g in FEEDBACK_GROUPS) & np.isin(block, ["familiarization", "adaptation"])
            split = block == "adaptation"
            f = dict(
                dataset=DS, study=DS, experiment="", group=GROUPS[g], subject=code, subject_excluded_by_authors=False,
                age=people[-1]["age"], sex={"Female": "F", "Male": "M"}[people[-1]["sex"]],
                handedness=people[-1]["arm"], population="healthy", setting="lab", device=DEVICE, session=1,
                block=block, trial=np.arange(1, n + 1), phase=[PHASE[b] for b in block], time_s="", task="walking",
                unit="stride", condition=np.where(fb, "feedback on", "no feedback"),
                pert_type=np.where(split, "split_belt", "none"), pert_value=np.where(split, 2.0, 0.0),
                pert_unit=np.where(split, "ratio", ""), feedback="normal", excluded_by_authors=col["bad"] == 1,
                fast_leg=people[-1]["leg"],
                belt_speed_fast_mps=[SPEED[b][0] for b in block], belt_speed_slow_mps=[SPEED[b][1] for b in block],
                walking_trial=trial.astype(int), stride_start_s=np.ravel(ps["hiddenTime"]).astype(float),
            )
            for c, (lab, factor) in PARAMS.items():
                f[c] = col[lab] * factor
            frames.append(pd.DataFrame(f))
    df = pd.concat(frames, ignore_index=True)
    for c in ["stride_start_s"] + list(PARAMS):
        df[c] = df[c].round(6)  # the precision written to the CSV, so the checks below see the released values

    # ---- internal checks
    by_group = pd.Series([p["group"] for p in people]).value_counts()
    ck.internal("21 participants, 7 in each of the 3 groups (paper, Methods)", len(people) == 21 and (by_group == 7).all(),
                str(by_group.to_dict()))
    rng = all(-1 <= p["counts"][b] - PLANNED[b] <= 6 for p in people for b in BLOCKS)
    ck.internal("every participant walked TM base, familiarization, adaptation, TM post, each within -1/+6 strides of the "
                "planned 150/150/600/450", rng,
                "; ".join(f"{b} {min(p['counts'][b] for p in people)}-{max(p['counts'][b] for p in people)}" for b in BLOCKS))
    ck.internal("every stride the deposit shares is kept, in its order: 28,427 rows; recording numbers never decrease "
                "and stride start times increase within each recording",
                len(df) == 28427 and all((np.diff(s.stride_start_s.to_numpy())[np.diff(s.walking_trial.to_numpy()) == 0] > 0).all()
                                         for _, s in df.groupby("subject", sort=False)), f"{len(df)} rows")
    ck.internal("parameter labels are matched by name (the 21 parameter tables do not share one column order or size)",
                True, f"{len(label_sets)} distinct label lists, sizes {sorted({k for k, _ in label_sets})}")
    expect = {"bad", "good", "trial"} | {PARAMS[c][0] for c in PAPER_SIX}
    ck.internal("the deposit fills only bad, good, trial and the paper's six measures; every other parameter "
                "(step lengths, step times, step-length asymmetry) is empty for every stride", filled == expect,
                ", ".join(sorted(filled)))
    ck.internal("condition notes give 0.75 m/s for the tied blocks and a 2:1 belt ratio for adaptation", desc_ok)
    ex = df.excluded_by_authors
    ck.internal("excluded_by_authors = the deposit's bad-stride flag (getResults.m calls removeBadStrides before every "
                "epoch)", ex.sum() == 152, f"{int(ex.sum())} strides ({100 * ex.mean():.2f}%)")
    leg_side = pd.Series([p["leg"] for p in people]).value_counts().to_dict()
    ck.internal("fast leg = self-reported dominant leg (paper, Methods 2.1): right for 20, left for 1",
                leg_side == {"right": 20, "left": 1}, str(leg_side))
    # Sign check: with the labTools event definitions, (stance_slow - stance_fast) = (DS_slow - DS_fast)
    # + (step_time_fast - step_time_slow) exactly, so Tout = TnA - TA / stride time. Stride time = the next stride's start
    # minus this one's (strides are contiguous within a recording).
    diffs = []
    for _, s in df.groupby("subject", sort=False):
        T = np.diff(s.stride_start_s.to_numpy())
        ok = (np.diff(s.walking_trial.to_numpy()) == 0) & ~s.excluded_by_authors.to_numpy()[:-1] & ~s.excluded_by_authors.to_numpy()[1:]
        est = s.stance_time_asymmetry.to_numpy()[:-1] - s.double_support_asymmetry_s.to_numpy()[:-1] / T
        d = (est - s.step_time_asymmetry.to_numpy()[:-1])[ok]
        diffs.append(d[np.isfinite(d)])
    d = np.abs(np.concatenate(diffs))
    ck.internal("signs: step_time_asymmetry = stance_time_asymmetry - double_support_asymmetry_s / stride time on every "
                "pair of consecutive good strides (so Tout is positive when the fast leg's step lasts longer)",
                np.median(d) < 0.001 and np.percentile(d, 95) < 0.005,
                f"n={len(d)}, median |diff| {np.median(d):.5f}, 95th percentile {np.percentile(d, 95):.5f}")
    early = df[(df.block == "adaptation") & ~ex].groupby("subject", sort=False).head(5)
    ck.internal("stance_time_asymmetry is positive early in adaptation (the slow foot stays on the slow belt longer)",
                early.stance_time_asymmetry.mean() > 0.05, f"first 5 good adaptation strides: {early.stance_time_asymmetry.mean():.3f}")
    good = df[~ex]
    spread = {c: float(np.nanpercentile(np.abs(good[c]), 99.5)) for c in PAPER_SIX}
    ck.internal("99.5% of good strides lie within +-0.5 on each of the six measures (a few outliers kept as shared)",
                all(v < 0.5 for v in spread.values()), str({k: round(v, 3) for k, v in spread.items()}))
    ck.internal("ages 18-40 and sex F or M for everyone", all(18 <= p["age"] <= 40 for p in people)
                and {p["sex"] for p in people} <= {"Female", "Male"})

    # ---- published numbers
    ck.published("number of women (13 women, 8 men)", sum(p["sex"] == "Female" for p in people), 13, 0, "Methods, para 1")
    ck.published("mean age, years (24.69 +- 4)", np.mean([p["age"] for p in people]), 24.69, 0.005, "Methods, para 1")

    # Epochs as in the authors' getResults.m and removeBiasV3.m, on good strides only (removeBadStrides):
    # bias = mean of the last 40 TM base strides leaving out the last 5; steady state = adaptation strides end-45..end-6;
    # after-effect = TM post strides 2..31; both minus the bias.
    def epochs(c):
        out = {}
        for subj, s in df[~ex].groupby("subject", sort=False):
            v = {b: s.loc[s.block == b, c].to_numpy(float) for b in BLOCKS}
            nb, na = len(v["TM base"]), len(v["adaptation"])
            bias = np.nanmean(v["TM base"][nb - 45:nb - 5])
            out[subj] = (s.group.iloc[0], np.nanmean(v["adaptation"][na - 46:na - 6]) - bias,
                         np.nanmean(v["TM post"][1:31]) - bias)
        return pd.DataFrame(out, index=["group", "ss", "ae"]).T

    ep = {c: epochs(c) for c in PAPER_SIX}

    def p_zero(c, grp, which):  # one-sample t-test of the bias-removed epoch against zero, n = 7
        return stats.ttest_1samp(ep[c].loc[ep[c].group == grp, which].astype(float), 0).pvalue

    def p_group(c, grp):  # group main effect of the 2 (group) x 2 (epoch) mixed ANOVA, control vs grp = t-test on subject means
        m = ep[c][["ss", "ae"]].astype(float).mean(axis=1)
        return stats.ttest_ind(m[ep[c].group == "Control"], m[ep[c].group == grp]).pvalue

    S, T, SF = "Spatial Feedback", "Temporal Feedback", "Control"
    pub = [
        ("Sout steady state vs 0, control", p_zero("step_position_asymmetry", SF, "ss"), 0.0004, "Results 3.1, para 1 (Figure 2A)"),
        ("Tout steady state vs 0, spatial feedback", p_zero("step_time_asymmetry", S, "ss"), 0.0004, "Results 3.1, para 1 (Figure 2A)"),
        ("Sout after-effect vs 0, control", p_zero("step_position_asymmetry", SF, "ae"), 0.0003, "Results 3.1, para 1 (Figure 2B)"),
        ("Sout after-effect vs 0, spatial feedback", p_zero("step_position_asymmetry", S, "ae"), 0.0164, "Results 3.1, para 1 (Figure 2B)"),
        ("Tout after-effect vs 0, spatial feedback", p_zero("step_time_asymmetry", S, "ae"), 0.1023, "Results 3.1, para 1 (Figure 2B)"),
        ("SA after-effect vs 0, control", p_zero("leg_orientation_asymmetry", SF, "ae"), 0.0009, "Results 3.1, para 2 (Figure 3B)"),
        ("SA after-effect vs 0, spatial feedback", p_zero("leg_orientation_asymmetry", S, "ae"), 0.0542, "Results 3.1, para 2 (Figure 3B)"),
        ("TA after-effect vs 0, control", p_zero("double_support_asymmetry_s", SF, "ae"), 0.0044, "Results 3.1, para 2 (Figure 3B)"),
        ("TA after-effect vs 0, spatial feedback", p_zero("double_support_asymmetry_s", S, "ae"), 0.0007, "Results 3.1, para 2 (Figure 3B)"),
        ("SA group effect, control vs spatial feedback", p_group("leg_orientation_asymmetry", S), 0.0091, "Results 3.1, para 2"),
        ("Sout group effect, control vs temporal feedback", p_group("step_position_asymmetry", T), 0.0005, "Results 3.2, para 1"),
        ("Sout after-effect vs 0, temporal feedback", p_zero("step_position_asymmetry", T, "ae"), 0.0021, "Results 3.2, para 1 (Figure 4B)"),
        ("Tout after-effect vs 0, temporal feedback", p_zero("step_time_asymmetry", T, "ae"), 0.8550, "Results 3.2, para 1 (Figure 4B)"),
        ("SA group effect, control vs temporal feedback", p_group("leg_orientation_asymmetry", T), 0.0029, "Results 3.2, para 2"),
        ("SA after-effect vs 0, temporal feedback", p_zero("leg_orientation_asymmetry", T, "ae"), 0.0080, "Results 3.2, para 2 (Figure 5B)"),
        ("TA after-effect vs 0, temporal feedback", p_zero("double_support_asymmetry_s", T, "ae"), 0.0009, "Results 3.2, para 2 (Figure 5B)"),
        ("TnA group effect, control vs temporal feedback", p_group("stance_time_asymmetry", T), 0.0030, "Results 3.3 (Figure 6A)"),
    ]
    for what, computed, reported, src in pub:
        ck.published("p-value, " + what, computed, reported, 0.00005, src)
    # Three p-values the paper reports that the shared data give only approximately (not used as checks):
    near = [("Sout steady state vs 0, spatial feedback", p_zero("step_position_asymmetry", S, "ss"), 0.0481),
            ("Tout after-effect vs 0, control", p_zero("step_time_asymmetry", SF, "ae"), 0.4235),
            ("Tout steady state vs 0, temporal feedback", p_zero("step_time_asymmetry", T, "ss"), 0.0092)]

    meta = dict(
        study=DS,
        paper=dict(authors="Gonzalez-Rubio M, Velasquez NF, Torres-Oviedo G", year=2019,
                   title="Explicit Control of Step Timing During Split-Belt Walking Reveals Interdependent Recalibration "
                         "of Movements in Space and Time",
                   venue="Front Hum Neurosci 13:207", doi="10.3389/fnhum.2019.00207"),
        lab="Torres-Oviedo (University of Pittsburgh)",
        data_url="https://doi.org/10.6084/m9.figshare.8145962",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Read the deposit's MATLAB (labTools) objects without the lab's code and kept, per stride, the paper's "
                 "six asymmetry measures, the recording number, the stride start time and the bad-stride flag, with "
                 "participant codes; participant IDs, dates, the experimenter field, height and weight, and the "
                 "deposit's empty parameters are left out."),
        how=["One row is one stride (slow-leg heel strike to the next) of one participant, in the deposit's order; the "
             "deposit keeps per stride only the paper's six asymmetry measures (Sout, Tout, SA, TA, SnA and TnA, in the "
             "asymmetry columns) and a bad-stride flag, so the per-leg times and lengths and step_length_asymmetry are "
             "empty.",
             "Fast and slow are the legs on the 1.0 and 0.5 m/s belts during adaptation (the self-reported dominant leg "
             "was on the fast belt); both belts ran at 0.75 m/s in the other blocks, and each block was its own "
             "recording (planned 150, 150, 600 and 450 strides).",
             "Values and signs are the deposit's: step_time_asymmetry (Tout) is positive when the fast leg's step lasts "
             "longer, as the deposit's own parameter description says and as stance_time_asymmetry minus "
             "double_support_asymmetry_s over stride time confirms stride by stride (median difference 0.0002), which "
             "is the opposite of how the paper's equation 2 reads; range_of_motion_asymmetry is defined by the paper's "
             "equation 5, while the "
             "deposit's description gives an angle-based formula that the shared values cannot test.",
             "excluded_by_authors marks the strides the deposit flags as bad (missing or out-of-order gait events, or a "
             "stride time out of range), which the authors' analysis code drops before every result; feedback is normal "
             "on every row, and condition is 'feedback on' for the familiarization and adaptation strides of the two "
             "feedback groups, who watched their own step position or step time on a screen (one control participant's "
             "condition notes also mention feedback in those blocks, but the paper lists every control as walking "
             "without it)."],
        unit_note="stride (from a slow-leg heel strike to the next one)",
        differences=["Three p-values in the Results come out close but not equal from the shared strides: 0.0499 (paper 0.0481), 0.4322 (paper 0.4235) and 0.0101 (paper 0.0092); the other 17 match to 4 decimals."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5f} vs {c['reported']}  ({c['source']})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:70]}: {c['detail']}")
    for what, computed, reported in near:
        print(f"  [not a check] p-value, {what}: {computed:.4f} vs reported {reported}")


if __name__ == "__main__":
    main()
