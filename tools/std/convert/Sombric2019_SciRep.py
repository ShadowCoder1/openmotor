"""Sombric, Gonzalez-Rubio & Torres-Oviedo (2019) Sci Rep: active and passive perception of step length before and
after split-belt walking. The deposit shares the split sessions of the three active-perception groups (9 each) and the
passive-perception group (8): one row per stride, with step lengths and leg positions normalized by the authors.

Raw files (OSF fex8t, folder DataFiles): raw_extra/Sombric2019_SciRep/NormalizedStudyData.mat, a struct of four MATLAB
groupAdaptationData objects (the lab's labTools classes), each holding one adaptationData object per participant with
metaData (conditions, their descriptions and trials), subData (sex, age, dominant leg and arm) and data, a
parameterSeries of strides x parameters. The objects are decoded from the MAT-file's object store with the reader of
tools/std/convert/Gonzalez-Rubio2019_FrontHumNeurosci.py (copied below, with the payload field name of newer scipy);
nothing from the deposit or labTools is executed. R_Active.mat, R_Passive.mat and R_PT.mat (per-participant epoch
results) were fetched but are not converted. The authors' README (github.com/PittSMLlab/Sombric2019Perception) states
the normalization: each step length divided by the speed-specific baseline, minus 1.

Run from the site folder: python3 -I tools/std/convert/Sombric2019_SciRep.py
"""
import io
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Sombric2019_SciRep"
MODULE = "walking"


def _payload(o):
    """The data field of a scipy MatlabOpaque record: 'arr' in older scipy, '_ObjectMetadata' in newer."""
    names = o.dtype.names
    return o["arr"] if "arr" in names else o["_ObjectMetadata"]


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
        self.cells = _payload(wrapper[0, 0]["MCOS"][0]).ravel()
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
        return self.value(_payload(opaque[0] if opaque.ndim else opaque))


def text(a):
    if isinstance(a, np.ndarray):
        return str(a.ravel()[0]) if a.size else ""
    return str(a)



GROUPS = {"PerceptionAbrupt": ("active perception (Experiment 2)", "Short-MidSpeed"),
          "PerceptionAbruptLong": ("active perception (Experiment 2)", "Long-MidSpeed"),
          "PerceptionAbruptSlowShort": ("active perception (Experiment 2)", "Short-SlowSpeed"),
          "PassivePerceptionGradual": ("passive perception (Experiment 1)", "")}
PHASE = {"tm base": "baseline", "medium baseline": "baseline", "familiarization": "practice", "map": "test",
         "error clamp": "baseline", "adaptation": "learning", "catch": "aftereffect", "re-adaptation": "learning",
         "washout error clamp": "aftereffect", "washout": "washout", "washout map": "test", "analog": "baseline",
         "pa analog": "aftereffect"}
FEEDBACK = {"familiarization": "normal", "map": "none", "washout map": "none", "error clamp": "altered",
            "washout error clamp": "altered"}
PARAMS = {"step_length_fast_rel": "stepLengthFast", "step_length_slow_rel": "stepLengthSlow",
          "leading_position_fast_rel": "alphaFast", "leading_position_slow_rel": "alphaSlow",
          "trailing_position_fast_rel": "XFast", "trailing_position_slow_rel": "XSlow"}
DEVICE = ("Bertec instrumented split-belt treadmill; Vicon motion capture (100 Hz); Oculus virtual reality headset showing "
          "a step-length grid in the perceptual blocks; instrumented handrail touched lightly")


def txt(v):
    """Text of a MATLAB char array, also when it sits inside a cell (a list here)."""
    while isinstance(v, list):
        v = v[0] if v else ""
    return text(v)


NUM = r"(\d*\.?\d+)"


def speeds(desc):
    """Belt speeds (fast, slow) in m/s from the authors' condition description; NaN when it gives none."""
    for pat in (NUM + r"\s*m/s\s+and\s+" + NUM + r"\s*m/s", NUM + r"\s+(?:and\s+)?" + NUM + r"\s*m/s"):
        m = re.search(pat, desc)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return max(a, b), min(a, b)
    m = re.search(NUM + r"\s*m/s", desc)
    return (float(m.group(1)),) * 2 if m else (np.nan, np.nan)


def main():
    mo = MatObjects(stdlib.raw_dir(DS) / "NormalizedStudyData.mat")
    S = mo.top["NormalizedStudyData"][0, 0]
    assert list(S.dtype.names) == list(GROUPS), S.dtype.names
    ck = stdlib.Checks()
    frames, people, descs, unassigned, targets_ok = [], [], {}, 0, True
    for g, (exp, gname) in GROUPS.items():
        grp = mo.variable(S[g])
        assert grp["__class__"] == "groupAdaptationData"
        for a in grp["adaptData"]:
            md, sub, ps = a["metaData"], a["subData"], a["data"]
            labels = [text(x) for x in ps["labels"]]
            X = ps["Data_"].astype(float)
            col = {lab: X[:, k] for k, lab in enumerate(labels)}
            n, trial = X.shape[0], col["trial"]
            block = np.array([""] * n, dtype=object)
            fast, slow = np.full(n, np.nan), np.full(n, np.nan)
            for cn, cd, tic in zip(md["conditionName"], md["conditionDescription"], md["trialsInCondition"]):
                m = np.isin(trial, np.ravel(tic))
                block[m] = txt(cn)
                f, s = speeds(txt(cd))
                fast[m], slow[m] = f, s
                descs.setdefault((g, txt(cn)), (txt(cd), f, s, int(m.sum())))
            unassigned += int((block == "").sum())
            code = f"{DS}_S{len(people) + 1:03d}"
            age = float(np.ravel(sub["age"])[0])
            people.append(dict(code=code, exp=exp, sex=text(sub["sex"]), age=age if age >= 18 else np.nan, age_raw=age,
                               counts=pd.Series(block).value_counts().to_dict()))
            low = [b.lower() for b in block]
            tgt = col.get("target", np.full(n, np.nan))
            cond = np.array([""] * n, dtype=object)
            if np.isfinite(tgt).any():
                lv = np.unique(np.round(tgt[np.isfinite(tgt)], 9))
                targets_ok &= len(lv) == 3
                for lab, v in zip(["short target", "comfortable target", "long target"], lv):
                    cond[np.isfinite(tgt) & (np.abs(tgt - v) < 1e-6)] = lab
                targets_ok &= bool(np.allclose(np.diff(lv), 0.06, atol=1e-6))
            split = np.isfinite(fast) & (fast != slow)
            ramp = np.array([b == "adaptation" for b in low]) & (g == "PassivePerceptionGradual")
            f = dict(
                dataset=DS, study=DS, experiment=exp, group=gname, subject=code, subject_excluded_by_authors=False,
                age=people[-1]["age"], sex={"Female": "F", "Male": "M"}[people[-1]["sex"]],
                handedness=text(sub["dominantArm"]).lower(), population="healthy", setting="lab", device=DEVICE,
                session=np.nan, block=block, trial=np.arange(1, n + 1), phase=[PHASE.get(b, "other") for b in low],
                time_s=np.nan, task="walking", unit="stride", condition=cond,
                pert_type=np.where(split | ramp, "split_belt", "none"),
                pert_value=np.where(ramp, np.nan, np.where(split, fast / np.where(split, slow, 1), 0.0)),
                pert_unit=np.where(split | ramp, "ratio", ""),
                feedback=[FEEDBACK.get(b, "normal") for b in low], excluded_by_authors=col["bad"] == 1,
                fast_leg="unknown", belt_speed_fast_mps=fast, belt_speed_slow_mps=slow,
                walking_trial=trial.astype(int), stride_start_s=np.ravel(ps["hiddenTime"]).astype(float),
                target_step_length=tgt,
            )
            for c, lab in PARAMS.items():
                f[c] = col[lab]
            frames.append(pd.DataFrame(f))
    df = pd.concat(frames, ignore_index=True)

    # ---- internal checks
    ck.internal("every stride falls in one of the authors' conditions", unassigned == 0, f"{unassigned} unassigned")
    ck.internal("each active participant's target takes three values, 0.06 apart (the paper's targets are 6 cm apart)",
                targets_ok)
    act_counts = [p["counts"] for p in people if p["exp"].startswith("active")]
    rng = {b: (min(c.get(b, 0) for c in act_counts), max(c.get(b, 0) for c in act_counts))
           for b in ("Error Clamp", "Washout Error Clamp", "Map", "Washout Map")}
    plan = {"Error Clamp": 455, "Washout Error Clamp": 455, "Map": 150, "Washout Map": 150}
    ck.internal("active perceptual blocks hold the planned 455 strides and map tests the planned 150, within -5/+2",
                all(plan[b] - 5 <= lo and hi <= plan[b] + 2 for b, (lo, hi) in rng.items()), str(rng))
    tm = df[(df.block == "TM base") & ~df.excluded_by_authors]
    ck.internal("normalized step lengths average about 0 in TM base, as the normalization implies",
                abs(tm.step_length_fast_rel.mean()) < 0.05 and abs(tm.step_length_slow_rel.mean()) < 0.05,
                f"{tm.step_length_fast_rel.mean():.4f}, {tm.step_length_slow_rel.mean():.4f}")
    nospeed = df.belt_speed_fast_mps.isna()
    ok_nospeed = df.experiment.str.startswith("passive") & df.block.isin(["adaptation", "Map", "Analog", "PA Analog", "Washout Map"])
    ck.internal("belt speeds read from every condition description, except the passive group's gradual ramp and its "
                "standing tasks", (nospeed <= ok_nospeed).all(),
                "; ".join(f"{k[0][:14]}/{k[1]}: {v[1]}/{v[2]} m/s, {v[3]} strides" for k, v in descs.items()))
    bad_age = [p["age_raw"] for p in people if not p["age_raw"] >= 18]
    ck.internal("ages: all but two participants 19-39 years; the deposit's age is below 1 year for two participants "
                "(left empty)", len(bad_age) == 2 and all(19 <= p["age"] <= 39 for p in people if p["age"] == p["age"]),
                str([round(a, 2) for a in bad_age]))

    # ---- published numbers
    P = pd.DataFrame(people)
    mean_age_note = []
    for exp, n, nf, am, asd, src in [("active perception (Experiment 2)", 27, 16, 25.1, 5.4, "Methods, Subjects"),
                                     ("passive perception (Experiment 1)", 8, 2, 24.8, 4.8, "Methods, Subjects")]:
        q = P[P.exp == exp]
        ck.published(f"{exp}: participants", len(q), n, 0, src)
        ck.published(f"{exp}: women", int((q.sex == "Female").sum()), nf, 0, src)
        mean_age_note.append(f"{exp}: mean age {q.age.mean():.2f} +- {q.age.std():.2f} over {int(q.age.notna().sum())} "
                             f"valid ages (paper {am} +- {asd})")

    meta = dict(
        study=DS,
        paper=dict(authors="Sombric C, Gonzalez-Rubio M, Torres-Oviedo G", year=2019,
                   title="Split-Belt walking induces changes in active, but not passive, perception of step length",
                   venue="Sci Rep 9(1):16442", doi="10.1038/s41598-019-52860-9"),
        lab="Torres-Oviedo (University of Pittsburgh)",
        data_url="https://osf.io/fex8t/",
        license="none stated",
        license_url="",
        changes=("Read the deposit's MATLAB (labTools) objects without the lab's code and kept, per stride, the authors' "
                 "normalized step lengths, leading and trailing leg positions, the step-length target, the recording "
                 "number, the stride start time and the bad-stride flag, with participant codes; participant IDs, dates, "
                 "height, weight and the empty parameters are left out."),
        how=["One row is one stride of one participant, in the deposit's order. The deposit holds only the split session "
             "of each participant (the tied control sessions in the paper are not shared) and, for the passive group, "
             "only the walking: the standing passive-perception reports are not in it.",
             "The step lengths (ankle to ankle at heel strike) and the leading (alpha) and trailing (X) leg positions "
             "relative to the hip are as the authors normalized them: divided by the participant's mean in baseline walking "
             "at the same speed, minus 1, so 0 is the baseline value; they fill the *_rel columns, and the metre columns are "
             "empty. target_step_length is the deposit's target parameter unchanged: its three values per participant are "
             "0.06 apart, like the paper's targets 6 cm apart, which suggests the target in metres minus 1; they are "
             "labelled short, comfortable and long target (in increasing order) in condition.",
             "Fast and slow are the legs on the 1.5 and 0.5 m/s belts; which side was fast is not stated, so fast_leg is "
             "unknown. Belt speeds come from the authors' condition descriptions (the passive group's adaptation ramped "
             "gradually to 3:1, so its speeds and ratio are empty); feedback is altered in the perceptual blocks (only 35% "
             "of the step-length error was shown in the virtual-reality headset), none in the map tests and normal "
             "elsewhere.",
             "excluded_by_authors is the deposit's bad-stride flag. block keeps the authors' condition names; phase maps "
             "the baseline perceptual block (Error Clamp) to baseline and the post-adaptation one (Washout Error Clamp) "
             "to aftereffect."],
        unit_note="stride (from a heel strike to the next one of the same leg)",
        differences=["Ages do not reproduce the paper's means: the deposit's age is below 1 year for two participants "
                     "(left empty here) and the others give " + "; ".join(mean_age_note) + ".",
                     "The condition descriptions in the deposit give 650 and 150 strides for adaptation and re-adaptation "
                     "of the active groups, where the paper says 600 and 200; the rows follow the deposit."],
    )
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in df.columns and c != "person":
            df[c] = np.nan  # module columns this deposit does not fill (the module may gain columns from other datasets)
    df = df[[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"]]
    stdlib.release(DS, MODULE, df, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source']})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:60]}: {c['detail'][:1500]}")


if __name__ == "__main__":
    main()
