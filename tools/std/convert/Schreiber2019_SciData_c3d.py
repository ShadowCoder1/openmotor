"""Minimal C3D reader (Intel/little-endian files, as written by Qualisys), written for this converter. Reads bytes only;
nothing in a deposit is executed. Returns header values, parameters, 3D points, analog channels and events."""
import struct

import numpy as np


def read_c3d(data):
    b = bytes(data)
    pstart, key = b[0], b[1]
    assert key == 0x50, "not a C3D file"
    npts, nanalog_total, first, last = struct.unpack_from("<HHHH", b, 2)
    scale = struct.unpack_from("<f", b, 12)[0]
    dstart = struct.unpack_from("<H", b, 16)[0]
    ratio = struct.unpack_from("<H", b, 18)[0]
    rate = struct.unpack_from("<f", b, 20)[0]
    # parameter section
    pos = (pstart - 1) * 512
    ptype = b[pos + 3]
    assert ptype == 84, f"processor type {ptype} not handled (Intel only)"
    pos += 4
    groups, params = {}, {}
    while True:
        nchar = struct.unpack_from("<b", b, pos)[0]
        gid = struct.unpack_from("<b", b, pos + 1)[0]
        if nchar == 0:
            break
        name = b[pos + 2:pos + 2 + abs(nchar)].decode("latin1").strip()
        off_pos = pos + 2 + abs(nchar)
        off = struct.unpack_from("<h", b, off_pos)[0]
        nxt = off_pos + off
        if gid < 0:
            groups[-gid] = name
        else:
            t = struct.unpack_from("<b", b, off_pos + 2)[0]
            nd = b[off_pos + 3]
            dims = list(b[off_pos + 4:off_pos + 4 + nd])
            dpos = off_pos + 4 + nd
            n = int(np.prod(dims)) if nd else 1
            if t == -1:
                raw = b[dpos:dpos + n]
                if nd >= 2:
                    w = dims[0]
                    val = [raw[i * w:(i + 1) * w].decode("latin1").strip() for i in range(n // w)] if w else []
                else:
                    val = raw.decode("latin1").strip()
            else:
                dt = {1: "<i1", 2: "<i2", 4: "<f4"}[t]
                val = np.frombuffer(b, dtype=dt, count=n, offset=dpos).reshape(dims[::-1] if nd > 1 else (n,)) if n else np.array([])
            params[(gid, name)] = val
        if off == 0:
            break
        pos = nxt
    P = {}
    for (gid, name), v in params.items():
        P[f"{groups.get(gid, gid)}:{name}"] = v
    n_frames = int(P["POINT:FRAMES"][0]) if "POINT:FRAMES" in P else last - first + 1
    if last - first + 1 > n_frames:
        n_frames = last - first + 1
    n_an = int(P["ANALOG:USED"][0]) if "ANALOG:USED" in P else 0
    ratio = int(ratio) if ratio else (int(P["ANALOG:RATE"][0] / P["POINT:RATE"][0]) if n_an else 0)
    dpos = (dstart - 1) * 512
    is_float = scale < 0
    per_frame = npts * 4 + n_an * ratio
    if is_float:
        arr = np.frombuffer(b, dtype="<f4", count=n_frames * per_frame, offset=dpos).reshape(n_frames, per_frame)
        pts = arr[:, :npts * 4].reshape(n_frames, npts, 4)
        an = arr[:, npts * 4:].reshape(n_frames * ratio, n_an) if n_an else np.zeros((0, 0))
        if n_an:
            an = an.astype(np.float64)
    else:
        arr = np.frombuffer(b, dtype="<i2", count=n_frames * per_frame, offset=dpos).reshape(n_frames, per_frame)
        pts = arr[:, :npts * 4].reshape(n_frames, npts, 4).astype(np.float64)
        pts[:, :, :3] *= scale
        an = arr[:, npts * 4:].reshape(n_frames * ratio, n_an).astype(np.float64) if n_an else np.zeros((0, 0))
    if n_an and not is_float:
        sc = np.asarray(P.get("ANALOG:SCALE", np.ones(n_an)), dtype=float)[:n_an]
        off = np.asarray(P.get("ANALOG:OFFSET", np.zeros(n_an)), dtype=float)[:n_an]
        gen = float(np.asarray(P.get("ANALOG:GEN_SCALE", [1.0]))[0])
        an = (an - off) * sc * gen
    out = dict(rate=float(P["POINT:RATE"][0]) if "POINT:RATE" in P else rate, scale=scale, ratio=ratio, n_frames=n_frames,
               first=first, last=last, params=P, points=pts,
               point_labels=[s for s in P.get("POINT:LABELS", [])][:npts], analog=an,
               analog_labels=[s for s in P.get("ANALOG:LABELS", [])][:n_an],
               analog_rate=float(P["ANALOG:RATE"][0]) if "ANALOG:RATE" in P else 0.0)
    return out
