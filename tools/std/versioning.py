"""Publish a dataset ZIP as a GitHub release asset without ever overwriting an earlier version.

    from versioning import publish_zip
    res = publish_zip("Smith2024_JNeurophysiol", "source/release_extra/Smith2024_JNeurophysiol.zip", "standardized")

Rules
  - The first version is <id>.zip. A later version whose data differ is uploaded as <id>.v2.zip, <id>.v3.zip, ...;
    nothing is uploaded with --clobber, so every file a snapshot (source/versions/) or a paper points to stays fetchable.
  - The new file is compared with the newest published version: the same sha256 means no upload. A ZIP rebuilt from
    the same data has a different sha256 (each member stores the time it was written), so the members' contents are
    compared too (names and the sha256 of each member's bytes); the same data means no upload either, and the record
    keeps pointing at the published file.
  - The version history is rebuilt from the release's assets each time (stdlib.release() rewrites the record and drops
    'url' and 'versions'), then written into the record:
        versions: [{version, release, file, url, sha256, bytes, date, content_sha256?}, ...]  oldest first
        url, zip_sha256, zip_bytes: the newest version
  - After an upload the asset is listed again and its size and GitHub digest must equal the local file's.

The record is a dict (updated in place) or a path (read, updated and written back); by default
source/release_extra_records/<id>.json when it exists. GitHub calls go through a small class (GitHub) so tests can
replace it (tools/std/test_versioning.py)."""
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
REPO = "physicalintelligencelab/openmotor"
TAG = "standardized"
ID_RE = re.compile(r"[A-Za-z0-9_\-]+")
MAX_ASSETS = 1000  # GitHub: "Up to 1000 release assets may be associated with a single release."
MAX_COMPARE_BYTES = 300_000_000  # largest published file downloaded to compare contents when its content hash is unknown


class GitHub:
    """The four calls publish_zip needs, through the gh command line (GH_TOKEN or a logged-in gh)."""

    def __init__(self, repo=REPO):
        self.repo = repo

    def _gh(self, *args):
        return subprocess.run(["gh", *args], capture_output=True, text=True)

    def assets(self, tag):
        """[{name, size, digest ('sha256:...' or None), created_at}] of the release with this tag."""
        r = self._gh("api", f"repos/{self.repo}/releases/tags/{tag}")
        if r.returncode != 0:
            raise RuntimeError(f"release {tag!r} of {self.repo} not found: {r.stderr.strip()[-300:]}")
        rid = json.loads(r.stdout)["id"]
        r = self._gh("api", "--paginate", "--slurp", f"repos/{self.repo}/releases/{rid}/assets?per_page=100")
        if r.returncode != 0:
            raise RuntimeError(f"cannot list the assets of {tag!r}: {r.stderr.strip()[-300:]}")
        return [{"name": a["name"], "size": a["size"], "digest": a.get("digest"), "created_at": a["created_at"]}
                for page in json.loads(r.stdout) for a in page]

    def upload(self, tag, path):
        """Upload a file under its own name; fails (returns False) if an asset of that name exists. Never clobbers."""
        r = self._gh("release", "upload", tag, str(path), "-R", self.repo)
        if r.returncode != 0:
            if "already exists" in r.stderr or "already_exists" in r.stderr:
                return False
            raise RuntimeError(f"upload of {Path(path).name} failed: {r.stderr.strip()[-300:]}")
        return True

    def download(self, tag, name, dest_dir):
        r = self._gh("release", "download", tag, "--pattern", name, "--dir", str(dest_dir), "-R", self.repo)
        if r.returncode != 0:
            raise RuntimeError(f"download of {name} failed: {r.stderr.strip()[-300:]}")
        return Path(dest_dir) / name


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def content_sha256(zip_path):
    """sha256 over the members' names and the sha256 of each member's bytes, in name order (member times ignored)."""
    h = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            m = hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    m.update(b)
            h.update(name.encode("utf-8") + b"\0" + m.hexdigest().encode() + b"\n")
    return h.hexdigest()


def file_name(ds, n):
    return f"{ds}.zip" if n == 1 else f"{ds}.v{n}.zip"


def version_of(ds, name):
    """1 for <id>.zip, N for <id>.vN.zip (N >= 2), None for any other asset."""
    if name == f"{ds}.zip":
        return 1
    m = re.fullmatch(re.escape(ds) + r"\.v([0-9]+)\.zip", name)
    return int(m.group(1)) if m and int(m.group(1)) >= 2 else None


def asset_url(repo, tag, name):
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def _digest(a):
    d = a.get("digest") or ""
    return d[len("sha256:"):] if d.startswith("sha256:") else None


def history(ds, assets, repo, tag, record=None):
    """The published versions of one dataset, oldest first, from the release assets. sha256 comes from GitHub's digest,
    else from the record (its versions list, or zip_sha256 when its url names that file); None if neither has it."""
    known = {}
    for v in (record or {}).get("versions") or []:
        known[v.get("file")] = v
    if record and record.get("url") and record.get("zip_sha256"):
        known.setdefault(record["url"].rsplit("/", 1)[-1], {"sha256": record["zip_sha256"], "bytes": record.get("zip_bytes")})
    out = []
    for a in assets:
        n = version_of(ds, a["name"])
        if n is None:
            continue
        k = known.get(a["name"]) or {}
        sha = _digest(a) or (k.get("sha256") if k.get("bytes") in (None, a["size"]) else None)
        e = {"version": n, "release": tag, "file": a["name"], "url": asset_url(repo, tag, a["name"]), "sha256": sha,
             "bytes": a["size"], "date": (a.get("created_at") or "")[:10] or k.get("date")}
        if k.get("content_sha256") and k.get("sha256") == sha:
            e["content_sha256"] = k["content_sha256"]
        out.append(e)
    return sorted(out, key=lambda e: e["version"])


def _all_history(gh, ds, repo, tag, also_tags, record):
    """history() over the earlier releases in also_tags and the release `tag`; also returns the assets of `tag`."""
    vers, cur = [], []
    for t in [*also_tags, tag]:
        assets = gh.assets(t)
        if t == tag:
            cur = assets
        vers += history(ds, assets, repo, t, record)
    order = {t: i for i, t in enumerate([*also_tags, tag])}
    return sorted(vers, key=lambda e: (e["version"], order[e["release"]])), cur


def _sha_of_published(gh, v, max_bytes):
    """sha256 and content hash of a published version whose digest is unknown (downloads it if small enough)."""
    if v["bytes"] > max_bytes:
        return None, None
    with tempfile.TemporaryDirectory() as d:
        p = gh.download(v["release"], v["file"], d)
        return sha256_file(p), content_sha256(p)


def _content_of_published(gh, v, max_bytes):
    if v.get("content_sha256"):
        return v["content_sha256"]
    if v["bytes"] > max_bytes:
        return None
    with tempfile.TemporaryDirectory() as d:
        p = gh.download(v["release"], v["file"], d)
        if v.get("sha256") and sha256_file(p) != v["sha256"]:
            raise RuntimeError(f"{v['file']}: the downloaded file does not match its recorded sha256")
        return content_sha256(p)


def _upload_as(gh, tag, src, name):
    """Upload src under the asset name `name` (a hard link or copy with that name in a temporary folder)."""
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return gh.upload(tag, dst)


def publish_zip(dataset_id, zip_path, tag=TAG, *, repo=REPO, record=None, record_path=None, gh=None, also_tags=(),
                compare_content=True, max_compare_bytes=MAX_COMPARE_BYTES):
    """Publish zip_path for dataset_id under the release `tag`, never overwriting a published file.

    also_tags: earlier releases that hold versions too (GitHub allows 1,000 assets per release; when 'standardized' is
    full, new files go to a new tag and the old one is passed here, so version numbers continue across both).
    Returns {status: 'published' (first version) | 'new_version' | 'unchanged', version, file, url, sha256, bytes,
    versions, note}. The record (dict or path; default source/release_extra_records/<id>.json if it exists) gets
    'versions' and points url / zip_sha256 / zip_bytes at the newest version."""
    if not ID_RE.fullmatch(dataset_id):
        raise ValueError(f"bad dataset id {dataset_id!r}")
    zip_path = Path(zip_path)
    gh = gh or GitHub(repo)
    if record is None:
        record_path = Path(record_path) if record_path else HERE / "source/release_extra_records" / f"{dataset_id}.json"
        record = None
        if record_path.exists():
            with open(record_path) as f:
                record = json.load(f)
    elif record_path:
        record_path = Path(record_path)
    sha, size = sha256_file(zip_path), zip_path.stat().st_size
    local_content = content_sha256(zip_path)

    vers, cur = _all_history(gh, dataset_id, repo, tag, also_tags, record)
    note = ""
    status = None
    if vers:
        latest = vers[-1]
        if latest["sha256"] is None:
            latest["sha256"], c = _sha_of_published(gh, latest, max_compare_bytes)
            if c:
                latest["content_sha256"] = c
        if latest["sha256"] == sha:
            status, note = "unchanged", "same sha256 as the newest published version"
            latest.setdefault("content_sha256", local_content)
        elif compare_content:
            pc = _content_of_published(gh, latest, max_compare_bytes)
            if pc == local_content:
                status, note = "unchanged", "same data as the newest published version (the ZIP was rebuilt)"
                latest["content_sha256"] = pc
            elif pc is None:
                note = "newest version too large to compare contents; published as a new version"
    if status is None:
        if len(cur) >= MAX_ASSETS:
            raise RuntimeError(f"release {tag!r} has {len(cur)} assets (GitHub's limit is {MAX_ASSETS}): publish to a new "
                               f"tag and pass {tag!r} in also_tags")
        n = (vers[-1]["version"] + 1) if vers else 1
        for _ in range(5):  # another publisher may take a name between listing and upload: then take the next one
            if _upload_as(gh, tag, zip_path, file_name(dataset_id, n)):
                break
            n += 1
        else:
            raise RuntimeError(f"{dataset_id}: no free version name after 5 tries")
        status = "published" if n == 1 else "new_version"
        a = None
        for wait in (0, 2, 5, 10):  # the asset list can lag the upload by a moment
            time.sleep(wait)
            a = next((x for x in gh.assets(tag) if x["name"] == file_name(dataset_id, n)), None)
            if a:
                break
        if a is None or a["size"] != size or (_digest(a) and _digest(a) != sha):
            raise RuntimeError(f"{file_name(dataset_id, n)}: the uploaded asset does not match the local file")
        # listed again: includes any version another publisher added meanwhile
        vers, _ = _all_history(gh, dataset_id, repo, tag, also_tags, {"versions": vers})
        for v in vers:
            if v["version"] == n and v["release"] == tag:
                v.update(sha256=sha, content_sha256=local_content, date=v["date"] or dt.date.today().isoformat())
    newest = vers[-1]
    if record is not None:
        record["versions"] = vers
        record["url"], record["zip_sha256"], record["zip_bytes"] = newest["url"], newest["sha256"], newest["bytes"]
        if record_path:
            with open(record_path, "w") as f:
                json.dump(record, f, indent=1)
    return {"status": status, "version": newest["version"], "file": newest["file"], "url": newest["url"],
            "sha256": newest["sha256"], "bytes": newest["bytes"], "versions": vers, "note": note}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Publish a dataset ZIP as a new release asset version (never overwrites).")
    ap.add_argument("dataset_id")
    ap.add_argument("zip_path")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--also-tag", action="append", default=[], help="an earlier release that holds versions too")
    a = ap.parse_args()
    res = publish_zip(a.dataset_id, a.zip_path, a.tag, repo=a.repo, also_tags=a.also_tag)
    print(json.dumps({k: v for k, v in res.items() if k != "versions"}, indent=1))
