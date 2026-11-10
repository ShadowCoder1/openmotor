"""Reader for MATLAB 'table' objects in a v7 MAT-file without MATLAB (helper of Burroni_ItalianLength_OSF.py; same method as
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
        b = np.asarray(self.cells[1 + sid]).tobytes()
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
        for nm, col in zip(names, o["data"].ravel()):
            col = np.asarray(col)
            if col.dtype == np.uint32 and col.shape == (6, 1) and int(col[0, 0]) == self.MARK:
                out[nm] = self.string_column(col.ravel().astype(np.int64))
            elif col.dtype == object:
                out[nm] = [np.asarray(x) for x in col.ravel(order="F")]
            else:
                out[nm] = col
        return out

