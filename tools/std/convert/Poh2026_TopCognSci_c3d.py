"""Minimal C3D reader (Intel/little-endian, as written by KINARM Dexterit-E; adapted from the Schreiber2019 reader) plus the
per-trial reduction for Poh et al. Reads bytes only; nothing in the deposit is executed."""
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


import math

LABEL_KEYS = ("target_angle", "rotation", "feedback_type", "aiming_targets", "success", "reactionTime", "ReachTime",
              "Endpoint_cursor_x", "Endpoint_cursor_y", "tooSlow", "tooFast", "tooSlowRT", "radius", "workspace_center_x",
              "workspace_center_y", "gradual")
ONSET_SPEED = 0.05  # m/s (the paper: movement onset when speed first exceeds 5 cm/s)


def reduce_trial(data):
    """One KINARM c3d trial -> dict of the per-trial values used by the converter (all plain floats)."""
    d = read_c3d(data)
    L = [l.strip("\x00") for l in d["analog_labels"]]
    A = d["analog"]
    P = d["params"]
    pl = [l.strip("\x00") for l in d["point_labels"]]
    xy = d["points"][:, pl.index("Right_Hand"), :2].astype(float)
    ch = {k: A[:, L.index(k)] for k in LABEL_KEYS}
    cx, cy, R = float(np.median(ch["workspace_center_x"])), float(np.median(ch["workspace_center_y"])), float(np.median(ch["radius"]))
    evl = [str(s).strip("\x00").strip() for s in P["EVENTS:LABELS"]]
    evt = [float(t) for t in np.asarray(P["EVENTS:TIMES"]).ravel()]
    rate = d["rate"]
    t_show = None
    for lab, t in zip(evl, evt):
        if lab.startswith("SHOW_TARGET"):
            t_show = t  # keep the last one
    out = dict(tp=int(P["TRIAL:TP"][0]), is_error=int(P["TRIAL:IS_ERROR"][0]), n=int(d["n_frames"]),
               target_angle=float(np.median(ch["target_angle"])), rotation=float(np.median(ch["rotation"])),
               feedback_type=float(np.median(ch["feedback_type"])), shape=float(np.median(ch["aiming_targets"])),
               gradual=float(np.median(ch["gradual"])), success=float(ch["success"][-1]), rt=float(ch["reactionTime"][-1]),
               mt=float(ch["ReachTime"][-1]), too_slow=float(ch["tooSlow"].max()), too_fast=float(ch["tooFast"].max()),
               too_slow_rt=float(ch["tooSlowRT"].max()), radius=R, hand_deg=None, hand_onset_ms=None, extent=None,
               cursor_deg=None)
    # hand direction: from the hand at movement onset to the hand where it first is R from the start (the deposit's radius)
    k = int(round((t_show if t_show is not None else 0.0) * rate))
    w = 5
    ker = np.ones(w) / w
    sx = np.convolve(xy[:, 0], ker, mode="same")
    sy = np.convolve(xy[:, 1], ker, mode="same")
    sp = np.hypot(np.gradient(sx), np.gradient(sy)) * rate
    idx = np.nonzero(sp[k:] > ONSET_SPEED)[0]
    if len(idx):
        on = k + int(idx[0])
        r = np.hypot(xy[on:, 0] - cx, xy[on:, 1] - cy)
        cr = np.nonzero(r >= R)[0]
        if len(cr):
            ce = on + int(cr[0])
            dx, dy = xy[ce, 0] - xy[on, 0], xy[ce, 1] - xy[on, 1]
            out["hand_deg"] = math.degrees(math.atan2(dy, dx))
            out["hand_onset_ms"] = (on - k) * 1000.0 / rate
            out["extent"] = float(math.hypot(dx, dy))
    ex, ey = float(ch["Endpoint_cursor_x"][-1]), float(ch["Endpoint_cursor_y"][-1])
    if ex != 0.0 or ey != 0.0:
        out["cursor_deg"] = math.degrees(math.atan2(ey - cy, ex - cx))
    return out
