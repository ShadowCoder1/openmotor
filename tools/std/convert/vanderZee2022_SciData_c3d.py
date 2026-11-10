"""Plain C3D reader (Intel/DEC/SGI, integer or float data) for vanderZee2022_SciData. Reads points and analog channels with struct/numpy only;
nothing in a C3D file is executed."""
import struct

import numpy as np


def parse(b):
    ptr = (b[0] - 1) * 512
    ptype = b[ptr + 3] - 83  # 84 intel, 85 dec, 86 sgi
    en = ">" if ptype == 3 else "<"
    dec = ptype == 2
    u16 = lambda o: struct.unpack_from(en + "H", b, o)[0]
    i16 = lambda o: struct.unpack_from(en + "h", b, o)[0]

    def f32(o):
        if dec:
            w = b[o:o + 4]
            return struct.unpack("<f", bytes([w[2], w[3], w[0], w[1]]))[0] / 4.0
        return struct.unpack_from(en + "f", b, o)[0]
    h = dict(npts=u16(2), nanalog=u16(4), first=u16(6), last=u16(8), scale=f32(12), dstart=u16(16), aps=u16(18), rate=f32(20))
    pos = ptr + 4
    groups, params = {}, {}
    while True:
        nl = struct.unpack_from("b", b, pos)[0]
        gid = struct.unpack_from("b", b, pos + 1)[0]
        nlen = abs(nl)
        name = b[pos + 2:pos + 2 + nlen].decode("latin1").strip()
        nxt = i16(pos + 2 + nlen)
        cur = pos + 2 + nlen + 2
        if gid < 0:
            groups[-gid] = name
        else:
            t = struct.unpack_from("b", b, cur)[0]
            nd = b[cur + 1]
            dims = list(b[cur + 2:cur + 2 + nd])
            o = cur + 2 + nd
            n = int(np.prod(dims)) if nd else 1
            if t == -1:
                s = b[o:o + n].decode("latin1")
                val = [s[i * dims[0]:(i + 1) * dims[0]].strip() for i in range(n // dims[0])] if nd >= 2 else s.strip()
            elif t == 1:
                val = list(struct.unpack_from("%db" % n, b, o))
            elif t == 2:
                val = [i16(o + 2 * i) for i in range(n)]
            elif t == 4:
                val = [f32(o + 4 * i) for i in range(n)]
            else:
                val = None
            params[(groups.get(gid, str(gid)), name)] = val
        if nxt == 0:
            break
        pos = pos + 2 + nlen + nxt
    return h, params, en, dec


def data(b, h, params, en, dec):
    """Returns (points mm as float (n, npts, 3) with NaN where missing, analog as float (n*aps, nch) after (raw - offset) * scale * gen_scale)."""
    npts, na, aps = h["npts"], h["nanalog"], h["aps"]
    sc = h["scale"]
    nch = na // aps if aps else 0
    per = npts * 4 + na
    off = (h["dstart"] - 1) * 512
    n = h["last"] - h["first"] + 1
    flt = sc < 0
    n = min(n, (len(b) - off) // (per * (4 if flt else 2)))
    assert not (flt and dec), 'DEC float data not supported'
    dt = np.dtype(np.float32 if flt else np.int16).newbyteorder(en)
    arr = np.frombuffer(b, dtype=dt, count=n * per, offset=off).reshape(n, per)
    p4 = arr[:, :npts * 4].reshape(n, npts, 4)
    pts = p4[:, :, :3].astype(np.float64) * (sc if sc > 0 else 1.0)
    missing = (p4[:, :, :3] == 0).all(axis=2) | (p4[:, :, 3] == -1)
    pts[missing] = np.nan
    an = None
    if nch:
        raw = arr[:, npts * 4:].reshape(n, aps, nch).reshape(n * aps, nch).astype(np.float64)
        offs = np.array(params[("ANALOG", "OFFSET")], dtype=float)
        scl = np.array(params[("ANALOG", "SCALE")], dtype=float) * params[("ANALOG", "GEN_SCALE")][0]
        an = (raw - offs) * scl
    return pts, an, n
