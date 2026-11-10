"""Seekable read-only file over HTTP range requests (the signed redirect is resolved again for every 8 MB block), so that
zipfile can read single members of a 6 GB archive without storing it. Reads bytes only."""
import io, urllib.request, urllib.error, time

class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (re-resolves the signed redirect on each block)."""
    def __init__(self, url, size, block=8 << 20):
        self.url, self.size, self.block, self.pos = url, size, block, 0
        self.cache_start, self.cache = -1, b""
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def _fetch(self, start):
        end = min(start + self.block, self.size) - 1
        for a in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "OpenMotor-curation/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    b = r.read()
                if len(b) != end - start + 1: raise IOError("short read")
                self.cache_start, self.cache = start, b
                return
            except Exception as e:
                time.sleep(5 * (a + 1))
        raise IOError("range fetch failed")
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = []
        while n > 0:
            if not (self.cache_start <= self.pos < self.cache_start + len(self.cache)):
                self._fetch(self.pos)
            o = self.pos - self.cache_start
            c = self.cache[o:o + n]
            out.append(c); self.pos += len(c); n -= len(c)
        return b"".join(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
