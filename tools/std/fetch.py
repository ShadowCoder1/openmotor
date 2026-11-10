"""Download one public file into raw_extra/<id>/ and record its source and hash.

usage: python3 -I tools/std/fetch.py <dataset_id> <url> <file_name>

The file is untrusted data: it is written into the dataset's own folder, never executed, and read only by the
converter with python3 -I. Archives are not unpacked here (the converter reads them with zipfile)."""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]


def main(ds_id, url, name):
    assert "/" not in name and not name.startswith("."), name
    d = HERE / "raw_extra" / ds_id
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    req = urllib.request.Request(url, headers={"User-Agent": "OpenMotor-curation/1.0 (research data index)"})
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
        for b in iter(lambda: r.read(1 << 20), b""):
            f.write(b)
            h.update(b)
            n += len(b)
    src = d / "SOURCES.json"
    recs = json.load(open(src)) if src.exists() else []
    recs = [x for x in recs if x["file"] != name] + [dict(file=name, url=url, bytes=n, sha256=h.hexdigest(),
                                                           fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))]
    json.dump(recs, open(src, "w"), indent=1)
    print(name, n, h.hexdigest()[:16])


if __name__ == "__main__":
    main(*sys.argv[1:4])
