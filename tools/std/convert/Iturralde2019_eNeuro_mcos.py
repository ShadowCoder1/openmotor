"""MATLAB class-object reader for v7.3 (HDF5) MAT-files, adapted from the v7 reader in Liu2024_RSOS_mcos.py (decodes the MCOS
object store with h5py, as plain data; runs nothing from the deposit).

In a v7.3 file the hidden object store is /#subsystem#/MCOS, a cell array of HDF5 object references: cell 0 is the uint8
metadata block (same layout as in v7 files) and the other cells hold the property values. An object reference anywhere
in the file is a uint32 array [0xDD000000, ndims, dims..., object ids..., class id]. Data sets are stored with reversed
dimensions (MATLAB is column-major), so arrays are transposed on reading."""
import struct

import h5py
import numpy as np


class H5Objects:
    MARK = 0xDD000000

    def __init__(self, path):
        self.f = h5py.File(path, "r")
        self.cache = {}
        cells = self.f["#subsystem#/MCOS"]
        self.refs = [cells[0, i] for i in range(cells.shape[1])]
        meta = np.asarray(self.f[self.refs[0]]).ravel().astype("uint8").tobytes()
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
        self.objcache = {}

    def top(self, name):
        return self.dec(self.f[name])

    def cell(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dec(self.f[self.refs[idx]])
        return self.cache[idx]

    def obj(self, oid):
        if oid in self.objcache:
            return self.objcache[oid]
        cid, _, _, saveobj_id, prop_id, _ = (int(x) for x in self.objects[oid])
        assert saveobj_id == 0, "objects saved through saveobj are not handled"
        out = {"__class__": self.names[self.classes[cid][1] - 1]}
        self.objcache[oid] = out
        for name_i, kind, val in (self.props[prop_id] if prop_id else []):
            name = self.names[name_i - 1]
            if kind == 1:
                out[name] = self.cell(val + 2)
            elif kind == 2:
                out[name] = bool(val)
            elif kind == 0:
                out[name] = self.names[val - 1] if val else ""
            else:
                raise ValueError(f"unknown property kind {kind} for {name}")
        return out

    def dec(self, x):
        if isinstance(x, h5py.Group):
            return {k: self.dec(x[k]) for k in x.keys()}
        cls = x.attrs.get("MATLAB_class", b"")
        cls = cls.decode() if isinstance(cls, bytes) else str(cls)
        if h5py.check_ref_dtype(x.dtype):
            a = np.empty(x.shape, dtype=object)
            for idx in np.ndindex(x.shape):
                a[idx] = self.dec(self.f[x[idx]])
            return a.T
        v = x[()]
        if isinstance(v, np.ndarray) and v.dtype == np.uint32 and v.size >= 4 and int(v.ravel()[0]) == self.MARK:
            a = v.ravel().astype(np.int64)
            nd = int(a[1])
            n = int(np.prod(a[2:2 + nd]))
            objs = [self.obj(int(i)) for i in a[2 + nd:2 + nd + n]]
            return objs[0] if n == 1 else objs
        if cls == "char":
            return "".join(chr(c) for c in np.asarray(v).ravel(order="F"))
        if isinstance(v, np.ndarray):
            return v.T
        return v
