"""Read single members out of a big remote ZIP with HTTP range requests (read-only, with a 8 MB read-ahead buffer)."""
import io
import time
import urllib.request


class HttpFile(io.RawIOBase):
    CHUNK = 8 << 20

    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.buf_start, self.buf = 0, b""
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, wh=0):
        self.pos = off if wh == 0 else self.pos + off if wh == 1 else self.size + off
        return self.pos

    def _fetch(self, a, n):
        n = min(n, self.size - a)
        for k in range(10):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{a + n - 1}"})
                return urllib.request.urlopen(req, timeout=180).read()
            except Exception:  # noqa: BLE001  (429 or a dropped connection: back off and retry)
                time.sleep(5 * (k + 1))
        raise IOError("range request kept failing")

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        end = self.pos + n
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            self.buf_start, self.buf = self.pos, self._fetch(self.pos, max(n, self.CHUNK))
        d = self.buf[self.pos - self.buf_start: end - self.buf_start]
        self.pos += len(d)
        return d

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)
