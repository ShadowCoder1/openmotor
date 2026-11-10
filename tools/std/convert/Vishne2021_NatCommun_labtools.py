"""Reader for MATLAB table objects (subject_info, aq_data) in the Vishne et al. (2021) MAT-file, without MATLAB. Copied from
tools/std/convert/Sombric2020_JNER_labtools.py (class MatObjects, itself copied from Gonzalez-Rubio2019_FrontHumNeurosci.py);
it decodes only the MAT-file's own object store, as plain data. Nothing from the deposit is executed.
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
