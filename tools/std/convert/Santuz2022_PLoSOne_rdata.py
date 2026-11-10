"""Minimal reader for R's XDR serialization (RData / RDS, gzip-compressed): data only, nothing is evaluated.
Supports lists, vectors, strings, pairlists, attributes, references; closures and environments are refused."""
import gzip
import struct

import numpy as np


class R:
    def __init__(self, b):
        self.b, self.i, self.refs = b, 0, []

    def int(self):
        v = struct.unpack(">i", self.b[self.i:self.i + 4])[0]
        self.i += 4
        return v

    def dbl(self, n):
        a = np.frombuffer(self.b, dtype=">f8", count=n, offset=self.i).astype(float)
        self.i += 8 * n
        return a

    def item(self):
        flags = self.int()
        t, has_attr, has_tag = flags & 0xFF, bool(flags & 0x200), bool(flags & 0x400)
        is_obj = bool(flags & 0x100)
        if t == 254 or t == 242 or t == 241 or t == 253 or t == 251:  # NIL, empty/base/global env, missing arg
            return None
        if t == 255:  # REFSXP
            idx = flags >> 8
            if idx == 0:
                idx = self.int()
            return self.refs[idx - 1]
        if t == 1:  # SYMSXP
            s = self.item()
            self.refs.append(s)
            return s
        if t == 9:  # CHARSXP
            n = self.int()
            if n == -1:
                return None
            s = self.b[self.i:self.i + n].decode("latin1")
            self.i += n
            return s
        if t == 2:  # LISTSXP: attr, tag, car, cdr
            attr = self.item() if has_attr else None
            tag = self.item() if has_tag else None
            car = self.item()
            cdr = self.item()
            out = [(tag, car)] + (cdr if isinstance(cdr, list) else [])
            return out
        if t in (10, 13):  # LGL, INT
            n = self.int()
            a = np.frombuffer(self.b, dtype=">i4", count=n, offset=self.i).astype(np.int64)
            self.i += 4 * n
            a = np.where(a == -2147483648, -1, a)
            val = a
        elif t == 14:
            n = self.int()
            val = self.dbl(n)
        elif t == 16:  # STRSXP
            n = self.int()
            val = [self.item() for _ in range(n)]
        elif t == 19:  # VECSXP
            n = self.int()
            val = [self.item() for _ in range(n)]
        elif t == 238:
            raise ValueError("ALTREP object not supported")
        else:
            raise ValueError(f"R type {t} not supported at byte {self.i}")
        if has_attr:
            attrs = self.item()
            return Obj(val, dict(attrs))
        return Obj(val, {})


class Obj:
    def __init__(self, val, attrs):
        self.val, self.attrs = val, attrs

    @property
    def names(self):
        n = self.attrs.get("names")
        return n.val if n is not None else None


def load(path):
    b = gzip.decompress(open(path, "rb").read())
    assert b[:5] == b"RDX3\n", b[:8]
    r = R(b)
    assert b[5:7] == b"X\n"
    r.i = 7
    assert r.int() == 3
    r.int(); r.int()
    n = r.int()
    r.i += n  # native encoding string
    top = r.item()
    return {tag: val for tag, val in top}
