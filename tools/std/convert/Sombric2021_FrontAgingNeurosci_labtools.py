"""Copy for Sombric2021_FrontAgingNeurosci of tools/std/convert/Sombric2020_JNER_labtools.py. Reader for the Torres-Oviedo lab's labTools objects (adaptationData, groupAdaptationData) in MAT-files, without the
lab's code. Copied from tools/std/convert/Gonzalez-Rubio2019_FrontHumNeurosci.py (class MatObjects), with two additions:
newer scipy (1.15+) names the fields of a MATLAB opaque object _TypeSystem, _Class and _ObjectMetadata instead of s0, s1,
s2 and arr, and older labTools files keep the stride table in a paramData object (Data, labels, indsInTrial,
trialTypes) instead of a parameterSeries (Data_, labels, description_, hiddenTime).

Only the MAT-file's own object store is decoded, as plain data; nothing from a deposit or from labTools is executed.
"""
import io
import struct

import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader


class MatObjects:
    """Read MATLAB classdef (MCOS) objects from a v7 MAT-file without their class definitions (see the pilot converter
    Gonzalez-Rubio2019_FrontHumNeurosci.py for the layout of the FileWrapper__ block)."""

    MARK = 0xDD000000

    def __init__(self, path_or_bytes):
        src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
        self.top = sio.loadmat(src)
        ws = self.top["__function_workspace__"].tobytes()
        head = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + ws[0:4]  # version and endian bytes of the block
        rd = MatFile5Reader(io.BytesIO(head + ws[8:]), struct_as_record=True, squeeze_me=False)
        rd.initialize_read()
        rd.mat_stream.seek(128)
        hdr, _ = rd.read_var_header()
        wrapper = rd.read_var_array(hdr, process=True)
        self.cells = self._arr(wrapper[0, 0]["MCOS"][0]).ravel()
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

    @staticmethod
    def _arr(opaque_item):
        names = opaque_item.dtype.names
        return opaque_item["arr" if "arr" in names else "_ObjectMetadata"]

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
        return self.value(self._arr(opaque[0]))


def text(a):
    if isinstance(a, np.ndarray):
        return str(a.ravel()[0]) if a.size else ""
    return str(a)


def number(a):
    v = np.ravel(a)
    return float(v[0]) if v.size else float("nan")


def strides(adapt):
    """One adaptationData object -> dict of per-stride arrays: every labTools parameter by label, plus 'trial'
    (recording number) and 'block' (the authors' condition name, from metaData.trialsInCondition), in the file's order."""
    md, ps = adapt["metaData"], adapt["data"]
    labels = [text(x) for x in ps["labels"]]
    X = np.asarray(ps["Data_"] if "Data_" in ps else ps["Data"], dtype=float)
    assert X.ndim == 2 and X.shape[1] == len(labels), (X.shape, len(labels))
    col = {lab: X[:, k] for k, lab in enumerate(labels)}
    if "trial" in col:
        trial = col["trial"].astype(int)
    else:  # paramData: indsInTrial{t} lists the rows (1-based) of recording t
        trial = np.zeros(X.shape[0], int)
        for t, ix in enumerate(ps["indsInTrial"], start=1):
            ix = np.ravel(ix).astype(int)
            trial[ix - 1] = t
        assert (trial > 0).all(), "rows not in any recording"
    block = np.full(X.shape[0], "", dtype=object)
    conds = [text(c) for c in md["conditionName"]]
    for cname, tic in zip(conds, md["trialsInCondition"]):
        tic = np.ravel(tic).astype(int) if isinstance(tic, np.ndarray) else np.array([], int)
        hit = np.isin(trial, tic)
        assert not (block[hit] != "").any(), f"recording in two conditions: {cname}"
        block[hit] = cname
    descr = {text(c): text(d) for c, d in zip(md["conditionName"], md["conditionDescription"])}
    return dict(col=col, labels=labels, trial=trial, block=block, conds=conds, descr=descr, n=X.shape[0])
