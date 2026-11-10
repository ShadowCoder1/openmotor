"""Gonzalez-Rubio, Iturralde & Torres-Oviedo (2026) Sci Rep: Weber's law for perceived belt-speed differences while
walking. 39 young adults (13 per group) made two-alternative forced-choice judgements ("which leg is slower?") on a
split-belt treadmill at a common reference mean speed (1.05 m/s) and at a group-specific testing speed (0.7, 1.4 or
1.75 m/s), over two sessions; one row per analysed perceptual trial.

Raw files (OSF nfb28, folder Data; the same files are on github.com/PittSMLlab/Gonzalez-Rubioetal.-2026-WeberPerception):
raw_extra/Gonzalez-Rubio2026_SciRep/Data.csv (the authors' trial table), JND_Choices.mat (their per-participant JNDs
from mixed-effects logistic fits, a cell array of MATLAB table objects), group1/2/3_JND_pertSize.csv (their JNDs from
drift-diffusion fits to reaction times) and READ_ME.txt. The authors' Weber_Figures.m and 'DDM RT fit.ipynb' were read
as text (never run) for the speed codes (0 = 1050, 1 = 700, 2 = 1400, 3 = 1750 mm/s), the participant ranges of the
groups (1-13 comfortable, 14-26 slow, 27-39 fast) and the row order of the JND tables. The MATLAB tables are decoded
from the MAT-file's object store with the reader of tools/std/convert/Gonzalez-Rubio2019_FrontHumNeurosci.py (copied
below, with the payload field name of newer scipy versions); nothing from the deposit is executed.

The reference-speed trials of this table are, trial for trial, the analysed answered trials of Gonzalez-Rubio2025_eNeuro
(same participants in the same order; checked when this converter was written).

Run from the site folder: python3 -I tools/std/convert/Gonzalez-Rubio2026_SciRep.py
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

DS = "Gonzalez-Rubio2026_SciRep"
MODULE = "perception"


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



SPEED_MM_S = {0: 1050.0, 1: 700.0, 2: 1400.0, 3: 1750.0}  # Weber_Figures.m
GROUPS = {"Comfortable": (range(1, 14), 2), "Slow": (range(14, 27), 1), "Fast": (range(27, 40), 3)}  # subIDs, test code
JND_TABLE_ORDER = ["Slow", "Comfortable", "Fast"]  # JND_Choices{model}{g} and group<g>_JND_pertSize.csv
FLIPPED_SUB = 32
DEVICE = ("split-belt treadmill; a hand-held controller in each hand (back button) for the choice; noise-cancelling "
          "headphones playing white noise and the cues; a drape hiding the feet; a screen showing the question")
STIMULUS = "belt-speed difference, right belt minus left belt (positive: left belt slower)"


def main():
    raw = stdlib.raw_dir(DS)
    ck = stdlib.Checks()
    d = pd.read_csv(raw / "Data.csv")
    assert list(d.columns) == ["pertSize", "reactionTime", "initialResponse", "subID", "blockNo", "speed", "leftResponse",
                               "rightResponse", "noResponse"], d.columns
    group_of = {s: g for g, (r, _) in GROUPS.items() for s in r}
    codes = d.groupby("subID").speed.apply(lambda s: tuple(sorted(set(s))))
    ck.internal("each participant walked at the reference code 0 and at the testing code of the group the authors' code "
                "assigns by participant number (1-13 comfortable = 2, 14-26 slow = 1, 27-39 fast = 3)",
                all(codes[s] == (0, GROUPS[group_of[s]][1]) for s in codes.index) and len(codes) == 39)
    # response follows the authors' choice columns (leftResponse/rightResponse), which their fits use; for one participant
    # those keep the recorded side on null trials while initialResponse is reversed there
    resp = np.where(d.leftResponse == 1, -1, np.where(d.rightResponse == 1, 1, np.nan))
    ck.internal("every trial has exactly one of leftResponse and rightResponse (the table holds answered trials only; "
                "noResponse is 0 on every row)", ((d.leftResponse + d.rightResponse) == 1).all() and (d.noResponse == 0).all())
    differs = resp != d.initialResponse.to_numpy()
    ck.internal("the choice columns agree with initialResponse except on the null trials of the one participant whose "
                "responses the authors reversed", (differs == ((d.subID == FLIPPED_SUB) & (d.pertSize == 0)).to_numpy()).all(),
                f"{int(differs.sum())} rows")
    stim = d.pertSize.astype(float)
    correct = np.where(stim == 0, None, resp == -np.sign(stim))
    mean_speed = d.speed.map(SPEED_MM_S).to_numpy()
    df = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment="", group=[group_of[s] for s in d.subID], subject=[f"s{s}" for s in d.subID],
        subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population="healthy", setting="lab",
        device=DEVICE, session=np.nan, block=[f"block {b}" for b in d.blockNo],
        trial=d.groupby("subID").cumcount().to_numpy() + 1, phase="test", time_s=np.nan, task="walking", unit="trial",
        condition=np.where(d.speed == 0, "reference", "testing"),
        pert_type=np.where(stim == 0, "none", "split_belt"), pert_value=stim.to_numpy(), pert_unit=np.where(stim == 0, "", "mm/s"),
        feedback="masked", excluded_by_authors=False,
        stimulus=STIMULUS, stimulus_value=stim.to_numpy(), stimulus_unit="mm/s", reference_value=mean_speed,
        response=[str(int(r)) for r in resp], response_label=np.where(resp == -1, "left slower", "right slower"),
        response_value=np.nan, response_unit="", response_correct=correct,
        reaction_time_s=d.reactionTime.map(lambda v: float(format(v, ".8g"))).to_numpy(), bout=np.nan,
    ))

    # ---- the authors' per-participant JNDs -> participants.csv
    mo = MatObjects(raw / "JND_Choices.mat")
    J = mo.top["JND_Choices"]
    jnd = {}
    for mi, model in enumerate(["abs", "wf"]):
        for gi, g in enumerate(JND_TABLE_ORDER):
            t = mo.variable(J[0, mi][0, gi])
            assert t["__class__"] == "table" and [text(v) for v in t["varnames"]] == ["Reference", "Testing"]
            jnd[(model, g)] = [np.ravel(x).astype(float) for x in t["data"]]
    ddm = {g: pd.read_csv(raw / f"group{gi + 1}_JND_pertSize.csv") for gi, g in enumerate(JND_TABLE_ORDER)}
    people = []
    for g, (subs, code) in GROUPS.items():
        for k, s in enumerate(subs):
            people.append(dict(
                person=f"s{s}", speed_group=g, testing_speed_mps=SPEED_MM_S[code] / 1000,
                jnd_glmm_reference_mm_s=jnd[("abs", g)][0][k], jnd_glmm_testing_mm_s=jnd[("abs", g)][1][k],
                jnd_glmm_reference_fraction=jnd[("wf", g)][0][k], jnd_glmm_testing_fraction=jnd[("wf", g)][1][k],
                jnd_ddm_reference_mm_s=float(ddm[g].Reference[k]), jnd_ddm_testing_mm_s=float(ddm[g].Testing[k])))
    P = pd.DataFrame(people)
    pcols = [
        ("speed_group", "category", "Slow, Comfortable, Fast", "Group, named by its testing speed."),
        ("testing_speed_mps", "number", "m/s", "Mean belt speed of the group's testing condition (the reference was 1.05 m/s)."),
        ("jnd_glmm_reference_mm_s", "number", "mm/s",
         "The authors' just noticeable difference at the reference speed, ln(3) / slope of the participant's logistic "
         "psychometric function from their mixed-effects fit with the belt-speed difference in mm/s (JND_Choices.mat, model 1)."),
        ("jnd_glmm_testing_mm_s", "number", "mm/s", "The same at the participant's testing speed."),
        ("jnd_glmm_reference_fraction", "number", "fraction of the mean belt speed",
         "The authors' JND at the reference speed from their mixed-effects fit with the difference divided by the mean "
         "speed (Weber fraction; JND_Choices.mat, model 2)."),
        ("jnd_glmm_testing_fraction", "number", "fraction of the mean belt speed", "The same at the testing speed."),
        ("jnd_ddm_reference_mm_s", "number", "mm/s",
         "The authors' JND at the reference speed predicted from a drift-diffusion model fitted to the participant's mean "
         "reaction times alone (group<k>_JND_pertSize.csv)."),
        ("jnd_ddm_testing_mm_s", "number", "mm/s", "The same at the testing speed."),
    ]

    # ---- internal checks
    nb = d.groupby(["subID", "blockNo"]).size()
    ck.internal("every row of Data.csv kept, in order: 13,059 answered trials of 39 participants", len(df) == 13059)
    ck.internal("stimuli are 0, +-25, +-50, +-100, +-150, +-200, +-300 mm/s",
                set(d.pertSize) == {0, 25, -25, 50, -50, 100, -100, 150, -150, 200, -200, 300, -300})
    ck.internal("38 participants have six blocks that alternate the two speeds (one participant's four block labels mix "
                "them, as in the deposit)", (d.groupby("subID").blockNo.nunique() == 6).sum() == 38, str(nb.value_counts().to_dict()))
    ck.internal("reaction times between -0.7 and 11 s (6 slightly negative: answers before the start cue, kept by the "
                "authors, whose DDM fits allow a negative non-decision time)", d.reactionTime.between(-0.71, 11).all()
                and int((d.reactionTime < 0).sum()) == 6)
    ck.internal("JND tables: 13 participants per group; Weber-fraction JNDs = mm/s JNDs / mean speed within 2%",
                all(len(v[0]) == 13 for v in jnd.values()) and np.allclose(
                    P.jnd_glmm_reference_fraction * 1050, P.jnd_glmm_reference_mm_s, rtol=0.02))

    # ---- published numbers
    ck.published("participants", df.subject.nunique(), 39, 0, "Methods, Participants")
    for g in GROUPS:
        ck.published(f"participants in the {g} group", df[df.group == g].subject.nunique(), 13, 0, "Methods, Participants")
    ck.published("mean reference-speed JND as a Weber fraction, % (authors' per-participant estimates)",
                 100 * P.jnd_glmm_reference_fraction.mean(), 4.807, 0.0005, "Results, first section (Fig. 2B)")
    src = "Results, DDM section (Fig. 4B)"
    for g, rep in [("Slow", 0.0), ("Fast", 0.002), ("Comfortable", 0.021)]:
        q = P[P.speed_group == g]
        p = stats.ttest_rel(q.jnd_ddm_reference_mm_s, q.jnd_ddm_testing_mm_s).pvalue
        what = (f"paired t-test p, DDM JND, reference vs {g.lower()} speed" + (" (reported p < 0.001)" if rep == 0 else ""))
        ck.published(what, p, rep, 0.001 if rep == 0 else 0.0005, src)

    diffs = []
    for g in ["Slow", "Comfortable", "Fast"]:
        q = P[P.speed_group == g]
        diffs.append((g, q.jnd_glmm_testing_mm_s.mean(), q.jnd_glmm_testing_mm_s.std() / np.sqrt(13),
                      stats.ttest_rel(q.jnd_glmm_reference_mm_s, q.jnd_glmm_testing_mm_s).pvalue))
    ref = P.jnd_glmm_reference_mm_s
    meta = dict(
        study=DS,
        paper=dict(authors="Gonzalez-Rubio M, Iturralde PA, Torres-Oviedo G", year=2026,
                   title="Weber's Law in walking: sensory scaling is observed in multi-sensory, dynamic tasks",
                   venue="Sci Rep 16(1):25767", doi="10.1038/s41598-026-54948-5"),
        lab="Torres-Oviedo (University of Pittsburgh)",
        data_url="https://osf.io/nfb28/",
        license="none stated",
        license_url="",
        changes=("Kept every row of the authors' trial table and wrote each response with its meaning and correctness, the "
                 "speed codes as mean belt speeds and the groups by name; added their per-participant JND estimates "
                 "(mixed-effects and drift-diffusion) as participants.csv; participant numbers are replaced by codes."),
        how=["One row is one analysed perceptual trial: after a belt-speed difference was set around a constant mean speed "
             "(reference_value, mm/s: 1050 in the reference condition, 700, 1400 or 1750 in the group's testing condition), "
             "an audio cue opened a response window (8 strides; 9 s for the fast group) in which the participant pressed the "
             "back button of the left or right hand-held controller to say which leg was slower.",
             "stimulus_value and pert_value are the difference in mm/s, right belt minus left belt (positive: left belt "
             "slower; the deposit's data README says m/s, but the values and the authors' code are in mm/s); response is -1 "
             "for 'left slower' and 1 for 'right slower'; reaction_time_s runs from the start cue to the press (6 are "
             "slightly negative, kept as shared); response_correct is empty on null trials.",
             "The authors removed the trials without an answer and the first trial after each treadmill start before "
             "sharing, so excluded_by_authors is False on every row. For one participant the authors reversed the "
             "responses; response follows their left/right choice columns, which their fits use and which are not reversed "
             "on that participant's 48 null trials (the deposit's initialResponse column is). block is the authors' block "
             "number (blocks alternate the two speeds); which blocks fell on which of the two days is not recorded, so "
             "session is empty. Vision of the feet and sound were masked (drape, white noise).",
             "participants.csv holds, per participant, the group, the testing speed and the authors' JNDs as shared: from "
             "their mixed-effects logistic fits (JND_Choices.mat) and from drift-diffusion fits to reaction times "
             "(JNDs_RT). The reference-speed trials are the same trials as in the dataset of Gonzalez-Rubio et al. (2025, "
             "eNeuro), from the same participants."],
        unit_note="perceptual trial (one judgement of which leg was slower)",
        differences=[
            "The paper's 37 non-response trials cannot be counted: the shared table holds answered trials only (13,059, "
            "8 fewer than the 13,104 planned minus 37), and one participant's block labels give 4 blocks instead of 6, so "
            "the paper's 234 blocks are not reproduced by counting labels.",
            f"The mixed-effects JNDs in JND_Choices.mat give reference {ref.mean():.3f} +- {ref.std() / np.sqrt(39):.3f} mm/s "
            "(paper 50.478 +- 3.057) and testing " + "; ".join(f"{g.lower()} {m:.3f} +- {s:.3f}" for g, m, s, _ in diffs)
            + " mm/s (paper slow 37.941 +- 4.657, comfortable 53.594 +- 4.346, fast 77.588 +- 8.760), and paired t-test "
            "p-values " + ", ".join(f"{g.lower()} {p:.4f}" for g, _, _, p in diffs)
            + " (paper 0.010, 0.385, < 0.001): close to the paper except the comfortable test, so the deposited estimates "
            "seem to come from a slightly different fit than the one reported; the paper's own mixed-model fit is not "
            "re-run here.",
        ],
    )
    stdlib.release(DS, MODULE, df, meta, ck, participants=P, participant_columns=pcols)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source']})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:70]}: {c['detail']}")
    print("  differences:", meta["differences"])


if __name__ == "__main__":
    main()
