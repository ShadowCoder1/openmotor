"""Sadeghi, Ingram & Wolpert (2018), PLoS ONE: adaptive coupling and generalization of force-field learning, 5 groups x 10
participants, one row per movement (Out and Return reaches), in the force_field_reaching module.

Run from the site folder: python3 -I tools/std/convert/Sadeghi2018_PLoSOne.py
Raw files (untrusted data, read only) in raw_extra/Sadeghi2018_PLoSOne/, from https://doi.org/10.1371/journal.pone.0207482
(figshare 7400021): pone.0207482.s001.mat (a MATLAB table T, which scipy loads only as an opaque object) and
pone.0207482.s002.m (the authors' description of its columns, read as text, never run).
The table is decoded without MATLAB (MATLAB's licence on the PC has expired): it is a MAT-file whose table object is stored
in the hidden __function_workspace__; the reader below follows the one in Gonzalez-Rubio2019_FrontHumNeurosci.py.
"""
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Sadeghi2018_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
MODULE = "force_field_reaching"
cols = stdlib.columns_for(MODULE)
names = [c[1] for c in cols]


class MatObjects:
    MARK = 0xDD000000

    def __init__(self, path):
        self.top = sio.loadmat(path)
        ws = self.top["__function_workspace__"].tobytes()
        head = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + ws[0:4]
        rd = MatFile5Reader(io.BytesIO(head + ws[8:]), struct_as_record=True, squeeze_me=False)
        rd.initialize_read()
        rd.mat_stream.seek(128)
        hdr, _ = rd.read_var_header()
        wrapper = rd.read_var_array(hdr, process=True)
        self.cells = wrapper[0, 0]["MCOS"][0]["_ObjectMetadata"].ravel()
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
        assert offs[1] == offs[2]
        self.objects = u32(offs[2], offs[3]).reshape(-1, 6)
        r4, i, self.props = u32(offs[3], offs[4]), 2, [None]
        while i < len(r4):
            n = int(r4[i])
            self.props.append(r4[i + 1:i + 1 + 3 * n].reshape(-1, 3))
            i += 1 + 3 * n
            i += i % 2

    def obj(self, oid):
        cid, _, _, saveobj_id, prop_id, _ = (int(x) for x in self.objects[oid])
        assert saveobj_id == 0
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
                raise ValueError(kind)
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
        return self.value(opaque[0][2])


def num(x):
    if x is None or x != x:
        return ""
    return "%.8g" % (0.0 if float(x) == 0 else float(x))


mo = MatObjects(RAW / "pone.0207482.s001.mat")
T = mo.variable(mo.top["T"])
vn = [str(np.asarray(v).ravel()[0]) for v in T["varnames"]]
assert vn == ["TrialNumber", "Out_Ret_Trials", "PostBreakTrials", "Field", "Adaptation", "KinematicError"], vn
tn, oret, pb, fld, ad, ke = [np.asarray(x) for x in T["data"]]
N, S, G = ad.shape
assert (N, S, G) == (746, 10, 5) and fld.shape == (N, 1, G) and ke.shape == ad.shape
assert (tn.ravel() == np.arange(1, N + 1)).all()
oret = oret.ravel().astype(int)
pb = pb.ravel().astype(bool)

# Phases, from the paper's design (Methods: Phases and trial counts) matched against the Field sequence of every group:
# 14 lead-in pairs, Probe1 79 pairs, washout 32, training 110, washout 32, Probe2 79, final washout 27 (373 Out-Return pairs).
SEG = [(0, 14, "Start", "baseline"), (14, 93, "Probe1", "test"), (93, 125, "Washout1", "washout"), (125, 235, "Training", "learning"),
       (235, 267, "Washout2", "washout"), (267, 346, "Probe2", "test"), (346, 373, "Washout3", "washout")]


def seg_of(t):
    p = t // 2
    for a, b, nm, ph in SEG:
        if a <= p < b:
            return nm, ph
    raise ValueError(t)


rows = []
for g in range(G):
    for s in range(S):
        sub = f"{ID}_G{g + 1}_S{s + 1:02d}"
        for t in range(N):
            f = int(fld[t, 0, g])
            tt = ["null", "field", "channel"][f]
            blk, ph = seg_of(t)
            row = dict.fromkeys(names, "")
            row.update(
                dataset=ID, study=ID, experiment=f"{ID}_Main", group=f"G{g + 1}", subject=sub, person=sub,
                subject_excluded_by_authors="False", population="healthy", setting="lab", device="vBOT planar robotic manipulandum",
                session="1", block=blk, trial=str(t + 1), phase=ph, task="reaching", unit="trial",
                condition="Out" if oret[t] == 1 else "Return",
                pert_type={"null": "none", "field": "force_field", "channel": "error_clamp"}[tt],
                pert_value="15" if tt == "field" else "0", pert_unit="Ns/m", feedback="normal",
                excluded_by_authors="True" if pb[t] else "False",
                trial_type=tt, trial_in_condition=str(t // 2 + 1),
                mpe_cm=num(ke[t, s, g]) if tt != "channel" else "",
                force_compensation_pct=num(100.0 * ad[t, s, g]) if tt == "channel" else "",
            )
            rows.append(row)

# ---------------------------------------------------------------- checks
R = rows
checks.internal("rows = 746 trials x 10 participants x 5 groups", len(R) == 746 * 10 * 5)
checks.internal("channel trials carry a force compensation and no MPE; other trials the reverse",
                all((r["force_compensation_pct"] != "") == (r["trial_type"] == "channel") and (r["mpe_cm"] != "") == (r["trial_type"] != "channel") for r in R))
checks.published("participants (5 groups of 10)", len({r["subject"] for r in R}), 50, 0, "Methods, Participants")
checks.published("groups", len({r["group"] for r in R}), 5, 0, "Methods, Participants (G1 to G5)")
g1 = [r for r in R if r["subject"].endswith("G1_S01")]


def count(rows_, **kw):
    return sum(all(r[k] == v for k, v in kw.items()) for r in rows_)


checks.published("Probe1: Out-Return pairs (158 trials)", count(g1, block="Probe1") / 2, 79, 0, "Methods, Phases (Probe1)")
checks.published("Probe1: Out exposure trials", count(g1, block="Probe1", condition="Out", trial_type="field"), 65, 0, "Methods, Phases (Probe1)")
checks.published("Probe1: Out channel trials", count(g1, block="Probe1", condition="Out", trial_type="channel"), 14, 0, "Methods, Phases (Probe1)")
checks.published("Probe1: Return trials that are channel trials", count(g1, block="Probe1", condition="Return", trial_type="channel"), 79, 0, "Methods, Phases (Probe1)")
checks.published("Training (G1): exposure pairs", count(g1, block="Training", condition="Out", trial_type="field"), 90, 0, "Methods, Phases (Training)")
checks.published("Training (G1): Out-Return pairs with a channel trial (20 interspersed channel pairs)",
                 count(g1, block="Training", condition="Out", trial_type="channel"), 20, 0, "Methods, Phases (Training)")
g5 = [r for r in R if r["subject"].endswith("G5_S01")]
checks.published("Training (G5): exposure on Return trials only", count(g5, block="Training", condition="Return", trial_type="field"), 90, 0, "Methods, Phases (G5)")
checks.published("Training (G5): exposure trials on Out", count(g5, block="Training", condition="Out", trial_type="field"), 0, 0, "Methods, Phases (G5)")
checks.published("Washout after Probe1: null pairs", count(g1, block="Washout1", condition="Out", trial_type="null"), 30, 0, "Methods, Phases (washout of 30 null pairs and 2 channel pairs)")
checks.published("Washout after Probe1: trials (64)", count(g1, block="Washout1"), 64, 0, "Methods, Phases (washout of 30 null pairs and 2 channel pairs)")
checks.published("Probe2, G3: Out trials are channel trials", count([r for r in R if r["subject"].endswith("G3_S01")], block="Probe2", condition="Out", trial_type="channel") , 79, 0, "Methods, Phases (Probe2: G3 channel)")
brk = [t for t in range(746) if pb[t]]
checks.published("trials excluded after the rest break before Probe2 (all 14 trials)", sum(1 for r in g1 if r["excluded_by_authors"] == "True" and int(r["trial"]) > 650), 14, 0, "Methods, Rest breaks")
checks.published("trials excluded after the first rest break", sum(1 for r in g1 if r["excluded_by_authors"] == "True" and int(r["trial"]) <= 190), 10, 0, "Methods, Rest breaks (10 trials after each break)")
fc = [float(r["force_compensation_pct"]) for r in R if r["force_compensation_pct"]]
checks.internal("force compensation values between -150 and 250 percent", -150 < min(fc) and max(fc) < 250, f"{min(fc):.0f} to {max(fc):.0f}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Wolpert (Cambridge / Columbia)",
    data_url="https://plos.figshare.com/articles/dataset/Adaptive_coupling_influences_generalization_of_sensorimotor_learning/7400021",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted from a MATLAB table to the OpenMotor force-field reaching format, participant codes replaced, adaptation (a fraction) written as percent; phases and blocks named from the paper's design.",
    unit_note="movement (Out or Return reach) of one participant",
    how=[
        "One row is one movement: 746 per participant (373 Out-Return pairs, in the authors' trial order), 10 participants in each of 5 groups (G1 to G5), 50 in all. condition says Out or Return; trial_in_condition is the pair number. Trial type is the authors' Field code: 0 null, 1 force field (velocity-dependent curl field of 15 Ns/m), 2 channel.",
        "mpe_cm is the authors' kinematic error (maximum perpendicular error), on null and field trials; the deposit gives no unit and the values (about 1 for field trials) are of centimetre size, as in the paper's figures. force_compensation_pct is the authors' Adaptation measure on channel trials (the slope of the force pushed into the channel wall on the ideal compensating force, 0 = none, 1 = full) times 100.",
        "The authors flipped signs so that clockwise and counter-clockwise field groups could be combined: pert_value is 15 Ns/m on every field trial and MPE signs follow the authors' flipped convention. Trials right after a rest break are the authors' exclusions (10 trials after each of four breaks, all 14 trials of Probe2's break) and are marked excluded_by_authors.",
        "The deposit holds the data from the first pairs of Probe1 on; the 40 + 14 familiarization pairs of the paper are not in it. block and phase were assigned from the paper's design and checked against each group's Field sequence: Start (14 pairs: channel and null lead-in), Probe1 (79), Washout1 (32), Training (110), Washout2 (32), Probe2 (79), Washout3 (27). Groups differ in what the Out and Return trials have in Training and Probe2 (G1 to G4 exposure on both, G5 on Return only; Probe2 Out: G1 and G5 the training field, G2 null, G3 channel, G4 the opposite field).",
    ],
)
stdlib.release(ID, MODULE, rows, meta, checks)
