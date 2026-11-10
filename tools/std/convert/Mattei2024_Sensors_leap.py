"""Streaming reader for the pretty-printed Leap Motion JSON dumps (MongoDB export: one big array of frame objects) of
Mattei et al. 2024. Yields one parsed top-level frame at a time, so a 925 MB file is never held in memory."""
import json
import re

BR = re.compile(r"[{}]")


def frames(path, chunk=1 << 24):
    pending = ""
    depth = 0
    with open(path, encoding="utf8") as f:
        while True:
            s = f.read(chunk)
            if not s:
                break
            start = 0 if depth > 0 else None
            for mm in BR.finditer(s):
                i = mm.start()
                if s[i] == "{":
                    if depth == 0:
                        start = i
                        pending = ""
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        yield json.loads(pending + s[start:i + 1])
                        pending = ""
                        start = None
            if depth > 0:
                pending += s[start:]
