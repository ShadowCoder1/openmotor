"""Reader for MATLAB 'table' objects in a v7 MAT-file without MATLAB (copy used by Burroni2025_JASAEL.py; same method as
the MatTables class of Tang2022_JNeurophysiol.py, which decodes the table from the file's own object store with scipy:
nothing from the deposit is executed). Columns are returned as numpy arrays; cell columns as lists of arrays/strings."""
import io
import struct

import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader


class MatTables:
    MARK = 0xDD000000

    def __init__(self, path):
        self.top = sio.loadmat(path)
        ws = self.top["__function_workspace__"].tobytes()
        head = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + ws[0:4]
        rd = MatFile5Reader(io.BytesIO(head + ws[8:]), struct_as_record=True, squeeze_me=False)
        rd.initialize_read()
        rd.mat_stream.seek(128)
        hdr, _ = rd.read_var_header()
        mc = rd.read_var_array(hdr, process=True)[0, 0]["MCOS"]
        self.cells = mc[0]["arr" if "arr" in mc.dtype.names else "_ObjectMetadata"].ravel()
        meta = self.cells[0].ravel().tobytes()
        _, n_names = struct.unpack_from("<II", meta, 0)
        offs = struct.unpack_from("<8I", meta, 8)
        pos, self.names = 40, []
        while len(self.names) < n_names:
            end = meta.index(b"\x00", pos)
            self.names.append(meta[pos:end].decode("latin1"))
            pos = end + 1
        u32 = lambda a, b: np.frombuffer(meta[a:b], dtype="<u4")  # noqa: E731
        self.classes = u32(offs[0], offs[1]).reshape(-1, 4)
        self.objects = u32(offs[2], offs[3]).reshape(-1, 6)
        r4, i, self.props = u32(offs[3], offs[4]), 2, [None]
        while i < len(r4):
            n = int(r4[i])
            self.props.append(r4[i + 1:i + 1 + 3 * n].reshape(-1, 3))
            i += 1 + 3 * n
            i += i % 2

    def string_column(self, ref):
        """A MATLAB string array saved by saveobj (object store version 4): its payload is the cell 1 + sid, a uint64 stream
        [version, ndims, dims..., one length per element, UTF-16 text of all elements]."""
        cid, _, _, sid, _, _ = (int(x) for x in self.objects[int(ref[4])])
        assert self.names[self.classes[cid][1] - 1] == "string" and sid > 0
        # In this file the string payloads are not all at cell 1 + sid (the first three are at cells 2-4, the next three near the
        # end of the cell store). table() pairs the n-th string column (by sid) with the n-th uint64 stream that starts
        # [1, 2, nrows, 1]; the pairing is checked there by the decoded text.
        k = self.sidcell[sid]
        b = np.asarray(self.cells[k]).tobytes()
        w = np.frombuffer(b[: len(b) // 8 * 8], dtype="<u8")
        ver, nd = int(w[0]), int(w[1])
        dims = [int(x) for x in w[2:2 + nd]]
        n = int(np.prod(dims))
        lens = [int(x) for x in w[2 + nd:2 + nd + n]]
        text = b[(2 + nd + n) * 8:]
        assert ver == 1 and 2 * sum(lens) <= len(text), (ver, dims, sum(lens), len(text))
        out, pos = [], 0
        for L in lens:
            out.append(text[pos:pos + 2 * L].decode("utf-16-le"))
            pos += 2 * L
        return np.array(out, dtype=object)

    def table(self, var):
        """dict: column name -> numpy array (numeric) or list (cell columns); plus the order of names under '_names'."""
        v = self.top[var]
        a = np.asarray(v[0]["arr" if "arr" in v.dtype.names else "_ObjectMetadata"]).ravel().astype(np.int64)
        assert int(a[0]) == self.MARK and int(a[1]) == 2 and int(a[2] * a[3]) == 1, a[:6]
        cid, _, _, sid, pid, _ = (int(x) for x in self.objects[int(a[4])])
        assert sid == 0 and self.names[self.classes[cid][1] - 1] == "table"
        o = {self.names[ni - 1]: self.cells[val + 2] for ni, kind, val in self.props[pid] if kind == 1}
        names = [str(np.asarray(x).ravel()[0]) for x in o["varnames"].ravel()]
        nrows = int(np.asarray(o["nrows"]).ravel()[0])
        out = {"_names": names, "_nrows": nrows}
        pay = [k for k, c in enumerate(self.cells) if np.asarray(c).dtype == np.uint64 and np.asarray(c).size > 4
               and list(np.asarray(c).ravel()[:4]) == [1, 2, nrows, 1]]
        sids = []
        for col in o["data"].ravel():
            col = np.asarray(col)
            if col.dtype == np.uint32 and col.shape == (6, 1) and int(col[0, 0]) == self.MARK:
                cid0 = int(self.objects[int(col[4, 0])][0])
                if self.names[self.classes[cid0][1] - 1] == "string":
                    sids.append(int(self.objects[int(col[4, 0])][3]))
        assert len(sids) == len(pay), (sids, pay)
        self.sidcell = dict(zip(sorted(sids), pay))
        for nm, col in zip(names, o["data"].ravel()):
            col = np.asarray(col)
            if col.dtype == np.uint32 and col.shape == (6, 1) and int(col[0, 0]) == self.MARK:
                ref = col.ravel().astype(np.int64)
                c2, _, _, _, p2, _ = (int(x) for x in self.objects[int(ref[4])])
                if self.names[self.classes[c2][1] - 1] == "categorical":
                    d = {self.names[ni - 1]: self.cells[val + 2] for ni, kind, val in self.props[p2] if kind == 1}
                    cats = [str(np.asarray(x).ravel()[0]) for x in d["categoryNames"].ravel()]
                    out[nm] = np.array([cats[int(c) - 1] for c in np.asarray(d["codes"]).ravel()], dtype=object)
                    continue
                out[nm] = self.string_column(ref)
            elif col.dtype == object:
                out[nm] = [np.asarray(x) for x in col.ravel(order="F")]
            else:
                out[nm] = col
        return out

