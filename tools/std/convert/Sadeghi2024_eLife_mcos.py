"""MATLAB class-object reader copied from the Gonzalez-Rubio2019 converter (decodes the MCOS object store with scipy; runs nothing from the deposit).
Added for Sadeghi2024_eLife: string_object(), which decodes MATLAB string objects (saved through saveobj as uint64 words holding UTF-16 text)."""
import io,struct,sys
import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader


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
        # (changed for Sadeghi2024_eLife: string objects are saved through saveobj; their text is decoded by string_object below)
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
        return self.value(opaque[0]["_ObjectMetadata"])


def string_object(o, ref):
    """Text of a MATLAB string object. ref is the (MCOS, 'string', uint32 array) tuple that scipy returns for it; the array is
    [0xDD000000, ndims, dims..., object id, class id], and object id k is stored in cell k + 1 of the object store as
    [1, ndims, dims..., then per string: number of characters, UTF-16 text packed in uint64 words]."""
    oid = int(np.asarray(ref[0][2]).ravel()[4])
    a = o.cells[oid + 1].ravel()
    assert int(a[0]) == 1
    nd = int(a[1])
    n = int(np.prod(a[2:2 + nd]))
    q, out = 2 + nd, []
    for _ in range(n):
        nc = int(a[q])
        q += 1
        nw = (nc + 3) // 4
        out.append(a[q:q + nw].astype("<u8").tobytes()[:2 * nc].decode("utf-16-le"))
        q += nw
    return out[0] if n == 1 else out
