"""Minimal C3D reader (numpy only) for the Grouvel2024 deposit: header, parameter section, 3D points and analog channels."""
import struct

import numpy as np


def read_c3d(buf):
    b = bytes(buf)
    pstart = b[0]
    assert b[1] == 0x50
    # parameter section
    off = (pstart - 1) * 512
    proc = b[off + 3]  # 84 Intel, 85 DEC, 86 SGI (big endian)
    assert proc in (84, 85), proc
    dec = proc == 85
    pos = off + 4
    groups, params = {}, {}
    while True:
        nlen = struct.unpack_from("b", b, pos)[0]
        gid = struct.unpack_from("b", b, pos + 1)[0]
        name = b[pos + 2: pos + 2 + abs(nlen)].decode("latin1").strip()
        nxt = struct.unpack_from("<h", b, pos + 2 + abs(nlen))[0]
        here = pos + 2 + abs(nlen)
        if nlen == 0 and gid == 0:
            break
        if gid < 0:  # group
            groups[-gid] = name
        else:
            t = struct.unpack_from("b", b, here + 2)[0]
            nd = b[here + 3]
            dims = [b[here + 4 + i] for i in range(nd)]
            p = here + 4 + nd
            n = int(np.prod(dims)) if nd else 1
            if t == -1:
                if nd <= 1:
                    val = b[p: p + n].decode("latin1")
                    val = [val]
                else:
                    w = dims[0]
                    cnt = int(np.prod(dims[1:]))
                    val = [b[p + i * w: p + (i + 1) * w].decode("latin1").strip() for i in range(cnt)]
                    p += w * cnt
            elif t == 1:
                val = list(np.frombuffer(b, np.int8, n, p))
            elif t == 2:
                val = list(np.frombuffer(b, "<i2", n, p))
            elif t == 4:
                if dec:
                    val = list(_dec_float(np.frombuffer(b, "<u4", n, p)))
                else:
                    val = list(np.frombuffer(b, "<f4", n, p))
            else:
                val = None
            params[(groups.get(gid, str(gid)), name)] = val
        if nxt == 0:
            break
        pos = here + nxt
    return b, params, dec


def _dec_float(u):
    u = u.astype("<u4")
    u = ((u & 0xFFFF) << 16) | (u >> 16)
    f = u.view("<f4") / 4.0
    return f


def c3d_points(buf):
    """Return dict with labels, units, rate, frames, points (frames x n x 3, mm as stored), residual, analog (samples x ch), analog labels, events."""
    b, P, dec = read_c3d(buf)
    g = lambda grp, nm: P.get((grp, nm))
    npts = int(g("POINT", "USED")[0])
    nfr = int(g("POINT", "FRAMES")[0])
    if nfr == -1 or nfr < 0:
        nfr = int(np.frombuffer(b, "<u2", 1, 16)[0])
    scale = float(g("POINT", "SCALE")[0])
    rate = float(g("POINT", "RATE")[0])
    labels = [s.strip() for s in g("POINT", "LABELS")][:npts]
    units = [s.strip() for s in (g("POINT", "UNITS") or [])]
    dstart = int(g("POINT", "DATA_START")[0])
    nan_ = int(g("ANALOG", "USED")[0]) if g("ANALOG", "USED") else 0
    arate = float(g("ANALOG", "RATE")[0]) if g("ANALOG", "RATE") else 0.0
    spf = int(round(arate / rate)) if (nan_ and rate) else 0
    pos = (dstart - 1) * 512
    if scale < 0:
        per = npts * 4 + nan_ * spf
        arr = np.frombuffer(b, "<f4", nfr * per, pos).reshape(nfr, per).copy() if not dec else _dec_float(np.frombuffer(b, "<u4", nfr * per, pos)).reshape(nfr, per)
        pt = arr[:, : npts * 4].reshape(nfr, npts, 4)
        xyz, res = pt[:, :, :3], pt[:, :, 3]
        an = arr[:, npts * 4:].reshape(nfr * spf, nan_) if nan_ else None
    else:
        per = npts * 4 + nan_ * spf
        arr = np.frombuffer(b, "<i2", nfr * per, pos).reshape(nfr, per).astype(float)
        pt = arr[:, : npts * 4].reshape(nfr, npts, 4)
        xyz, res = pt[:, :, :3] * scale, pt[:, :, 3]
        an = arr[:, npts * 4:].reshape(nfr * spf, nan_) if nan_ else None
    ev = {}
    for k in ("LABELS", "TIMES", "CONTEXTS", "USED"):
        ev[k] = g("EVENT", k)
    return dict(labels=labels, units=units, rate=rate, frames=nfr, xyz=xyz, res=res, analog=an, analog_rate=arate,
                analog_labels=[s.strip() for s in (g("ANALOG", "LABELS") or [])][:nan_], events=ev, params=P, first_frame=int(np.frombuffer(b, "<u2", 1, 6)[0]))
