"""Write a dated snapshot of every downloadable OpenMotor dataset: source/versions/<YYYY-MM-DD>.json.

usage: python3 -I tools/snapshot.py [--date YYYY-MM-DD] [--no-verify] [--replace]

One entry per downloadable file: id, study, module (the data format), url, file, release, bytes, sha256, rows,
participants, license, and version when the dataset has more than one (tools/std/versioning.py). Entries come from the
first release's source/release/manifest.json and from source/release_extra_records/<id>.json, with the URLs and license
labels the site shows (tools/build_page.py).

The snapshot's own hash ("sha256") is the sha256 of its "datasets" list, sorted by id, written as compact JSON
(json.dumps(datasets, sort_keys=True, separators=(",", ":"), ensure_ascii=False), UTF-8). It identifies the exact set
of files: two snapshots with the same files have the same hash. source/versions/README.md says how to cite and fetch one.

Unless --no-verify, each file is compared with the release asset GitHub serves (size and GitHub's sha256 digest).
A file GitHub serves with a different sha256 than its record is listed under "verified.mismatches", and its entry
carries the sha256 of the served file (the bytes anyone fetching the snapshot gets) plus "record_sha256".
An existing snapshot of the same date is never overwritten with different contents unless --replace."""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "source/versions"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def datasets():
    cfg = load(HERE / "source/release_config.json")
    repo, tag = cfg["repo"], cfg["bundle_tag"]
    published = set(cfg["publish_kinds"])
    out = []
    for m in load(HERE / "source/release/manifest.json").values():  # the first release (90-column reaching format)
        if m["kind"] not in published:
            continue
        out.append({"id": m["name"], "study": m["catalog_study"], "module": "reach90", "release": tag, "file": m["file"],
                    "url": f"https://github.com/{repo}/releases/download/{tag}/{m['file']}", "bytes": m["bytes"],
                    "sha256": m["sha256"], "rows": m["rows"], "participants": m["persons"],
                    "license": cfg["license_labels"][m["kind"]]})
    listing = {e["id"]: e for e in load(HERE / "source/extra_studies.json")}
    for p in sorted((HERE / "source/release_extra_records").glob("*.json")):
        if p.name.endswith(".skip.json"):
            continue
        r = load(p)
        if not r.get("url"):
            continue
        url = r["url"].replace("https://github.com/ShadowCoder1/openmotor/", f"https://github.com/{repo}/")  # as the site does
        e = {"id": r["id"], "study": r["id"], "module": r["module"], "release": url.split("/")[-2], "file": url.split("/")[-1],
             "url": url, "bytes": r["zip_bytes"], "sha256": r["zip_sha256"], "rows": r["rows"],
             "participants": r.get("persons", r["subjects"]), "license": listing.get(r["id"], {}).get("license") or "Not stated"}
        if r.get("versions"):
            e["version"] = r["versions"][-1]["version"]
        out.append(e)
    ids = [e["id"] for e in out]
    assert len(ids) == len(set(ids)), "duplicate dataset ids"
    return sorted(out, key=lambda e: e["id"]), repo


def snapshot_hash(ds):
    return hashlib.sha256(json.dumps(ds, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def gh_assets(repo):
    """{(tag, name): asset} for every release of the repository (gh api, all pages)."""
    def api(path):
        r = subprocess.run(["gh", "api", "--paginate", "--slurp", path], capture_output=True, text=True, check=True)
        return [x for page in json.loads(r.stdout) for x in page]
    out = {}
    for rel in api(f"repos/{repo}/releases?per_page=100"):
        for a in api(f"repos/{repo}/releases/{rel['id']}/assets?per_page=100"):
            out[(rel["tag_name"], a["name"])] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args()
    dt.date.fromisoformat(a.date)
    ds, repo = datasets()
    verified = None
    if not a.no_verify:
        assets = gh_assets(repo)
        mism, missing = [], []
        for e in ds:
            x = assets.get((e["release"], e["file"]))
            if x is None:
                missing.append(e["id"])
                continue
            served = (x.get("digest") or "").removeprefix("sha256:") or None
            if x["size"] != e["bytes"] or (served and served != e["sha256"]):
                mism.append({"id": e["id"], "file": e["file"], "record_bytes": e["bytes"], "record_sha256": e["sha256"],
                             "served_bytes": x["size"], "served_sha256": served})
                e["record_sha256"] = e["sha256"]
                e["sha256"], e["bytes"] = served, x["size"]
        if missing:
            sys.exit(f"not on GitHub: {missing}")
        verified = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "files": len(ds),
                    "by": "size and GitHub's sha256 digest of each release asset", "mismatches": mism}
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    snap = {"snapshot": a.date, "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repo": repo, "site_commit": commit or None, "datasets_count": len(ds),
            "participants": sum(e["participants"] for e in ds), "rows": sum(e["rows"] for e in ds),
            "bytes": sum(e["bytes"] for e in ds), "sha256": snapshot_hash(ds),
            "hash_rule": "sha256 of json.dumps(datasets, sort_keys=True, separators=(',', ':'), ensure_ascii=False) in UTF-8",
            "verified": verified, "datasets": ds}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{a.date}.json"
    if p.exists() and not a.replace and load(p).get("sha256") != snap["sha256"]:
        sys.exit(f"{p.name} exists with other contents (sha256 {load(p).get('sha256')}); use --replace or another --date")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{p.relative_to(HERE)}: {len(ds)} datasets, sha256 {snap['sha256']}"
          + (f", {len(verified['mismatches'])} served file(s) differ from their record" if verified else ", not verified"))


if __name__ == "__main__":
    main()
