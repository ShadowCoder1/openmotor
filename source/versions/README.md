# OpenMotor snapshots

Each file here, `<YYYY-MM-DD>.json`, lists every dataset OpenMotor offered for download on that date: the exact file
(name, URL, size, sha256), its rows and participants, its format (`module`) and its license as listed on the site.
A result computed from OpenMotor can name one snapshot, and anyone can later fetch exactly the same files, however many
datasets have been added since. Snapshots are written by `python3 -I tools/snapshot.py`.

## What a snapshot holds

- `snapshot`: the date. `created_utc`: when it was written. `site_commit`: the commit of this repository it was made from.
- `sha256`: the snapshot's own hash, the sha256 of its `datasets` list (sorted by id) written as compact JSON:
  `json.dumps(datasets, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, UTF-8. Two snapshots with the same
  files have the same hash.
- `verified`: when the files were compared with what GitHub serves (size and GitHub's sha256 digest of each release
  asset) and any that differed from their record. Such an entry carries the sha256 of the file GitHub serves, which is
  what anyone fetching it gets, and the record's value as `record_sha256`.
- `datasets`: one entry per file: `id`, `study` (the study page, `#study/<study>`), `module` (`reach90` is the 90-column
  reaching format; the others are the task formats on the Data Format page), `release` (the GitHub release tag),
  `file`, `url`, `bytes`, `sha256`, `rows`, `participants`, `license`, and `version` once a dataset has more than one.

## How to cite a snapshot

Cite the original paper of every dataset you use (each ZIP's README gives it), and the snapshot by its date and the first
16 characters of its hash, with a link to the file in this repository, for example:

> OpenMotor snapshot 2026-11-10 (sha256 8e873cfe712e8d69), https://github.com/physicalintelligencelab/openmotor/blob/main/source/versions/2026-11-10.json

If you used only some datasets, list their ids and sha256 values from the snapshot (in a supplement or your code).

## How to fetch exactly those files later

Every file stays at its URL: a dataset that changes is published under a new name (`<id>.v2.zip`, `<id>.v3.zip`, ...)
by `tools/std/versioning.py`, and the earlier file is never replaced. Download from the URLs in the snapshot and
compare each file's sha256 with the snapshot:

```python
import hashlib, json, urllib.request
snap = json.load(urllib.request.urlopen(
    "https://raw.githubusercontent.com/physicalintelligencelab/openmotor/main/source/versions/2026-11-10.json"))
datasets = snap["datasets"]
assert hashlib.sha256(json.dumps(datasets, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                      .encode("utf-8")).hexdigest() == snap["sha256"]
for d in datasets:
    if d["id"] not in {"Codol2018_SciRep", "Tsay2020_JNeurophysiol"}:   # the datasets you need
        continue
    data = urllib.request.urlopen(d["url"]).read()
    assert hashlib.sha256(data).hexdigest() == d["sha256"], d["id"]
    open(d["file"], "wb").write(data)
```

For many or large files, download with a tool that streams to disk (`curl -LO <url>`, `gh release download`) and check
with `shasum -a 256 <file>`.

Files published before versioning was in place (before the first snapshot, 2026-11-10) were uploaded with replacement;
a snapshot names the bytes served on its date, and the sha256 check tells you if a file has changed since.
