"""Tests of tools/std/versioning.py with GitHub replaced by an in-memory fake (no network, no gh).

usage: python3 -I tools/std/test_versioning.py   (or pytest tools/std/test_versioning.py)"""
import hashlib
import json
import shutil
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import versioning as V  # noqa: E402

DS = "Smith2024_JNeurophysiol"
TAG = "standardized"


class FakeGitHub:
    """Releases held in memory. upload() refuses an existing name, as gh does without --clobber."""

    def __init__(self, digests=True):
        self.releases = {TAG: {}}  # tag -> {name: bytes}
        self.created = {}          # name -> created_at
        self.uploads = []          # names, in order
        self.downloads = []
        self.digests = digests
        self.steal = set()         # names another publisher takes just before our upload

    @property
    def files(self):
        return self.releases[TAG]

    def assets(self, tag):
        return [{"name": n, "size": len(b), "created_at": self.created[n],
                 "digest": ("sha256:" + hashlib.sha256(b).hexdigest()) if self.digests else None}
                for n, b in self.releases[tag].items()]

    def upload(self, tag, path):
        name, files = Path(path).name, self.releases[tag]
        if name in self.steal:
            self.steal.discard(name)
            files[name], self.created[name] = b"someone else's file", "2026-11-10T00:00:00Z"
        if name in files:
            return False
        files[name] = Path(path).read_bytes()
        self.created[name] = "2026-11-%02dT12:00:00Z" % (10 + len(self.uploads))
        self.uploads.append(name)
        return True

    def download(self, tag, name, dest_dir):
        self.downloads.append(name)
        p = Path(dest_dir) / name
        p.write_bytes(self.releases[tag][name])
        return p


def make_zip(path, rows, when=(2026, 11, 10, 12, 0, 0)):
    """A dataset ZIP; `when` is the member time (a rebuild of the same data changes only this)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo(f"{DS}/{DS}.csv", when), "subject,trial,x\n" + "".join(f"S1,{i},{v}\n" for i, v in enumerate(rows)))
        z.writestr(zipfile.ZipInfo(f"{DS}/README.txt", when), "OpenMotor dataset\n")
    return path


class PublishZip(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.gh = FakeGitHub()
        self.rec_path = self.d / f"{DS}.json"
        self.rec_path.write_text(json.dumps({"id": DS, "rows": 3, "zip_sha256": "x", "zip_bytes": 0}))
        V.time.sleep = lambda s: None

    def tearDown(self):
        V.time.sleep = time.sleep
        shutil.rmtree(self.d)

    def publish(self, z, **kw):
        return V.publish_zip(DS, z, TAG, gh=self.gh, record_path=self.rec_path, **kw)

    def rec(self):
        return json.loads(self.rec_path.read_text())

    def test_first_publish_is_plain_name(self):
        z = make_zip(self.d / "a.zip", [1, 2, 3])
        r = self.publish(z)
        self.assertEqual((r["status"], r["file"], r["version"]), ("published", f"{DS}.zip", 1))
        rec = self.rec()
        self.assertEqual(rec["url"], f"https://github.com/{V.REPO}/releases/download/{TAG}/{DS}.zip")
        self.assertEqual(rec["zip_sha256"], V.sha256_file(z))
        self.assertEqual([v["file"] for v in rec["versions"]], [f"{DS}.zip"])

    def test_same_file_is_not_uploaded(self):
        z = make_zip(self.d / "a.zip", [1, 2, 3])
        self.publish(z)
        r = self.publish(z)
        self.assertEqual(r["status"], "unchanged")
        self.assertEqual(self.gh.uploads, [f"{DS}.zip"])

    def test_changed_data_becomes_v2_and_v3_and_nothing_is_overwritten(self):
        z1 = make_zip(self.d / "a.zip", [1, 2, 3])
        self.publish(z1)
        first = self.gh.files[f"{DS}.zip"]
        r2 = self.publish(make_zip(self.d / "b.zip", [1, 2, 4]))
        r3 = self.publish(make_zip(self.d / "c.zip", [1, 2, 5]))
        self.assertEqual((r2["status"], r2["file"]), ("new_version", f"{DS}.v2.zip"))
        self.assertEqual(r3["file"], f"{DS}.v3.zip")
        self.assertEqual(self.gh.files[f"{DS}.zip"], first)  # the first version is untouched
        rec = self.rec()
        self.assertEqual([v["version"] for v in rec["versions"]], [1, 2, 3])
        self.assertTrue(rec["url"].endswith(f"{DS}.v3.zip"))
        self.assertEqual(rec["zip_sha256"], rec["versions"][-1]["sha256"])
        self.assertEqual(rec["versions"][0]["sha256"], V.sha256_file(z1))
        self.assertEqual(rec["versions"][1]["date"], "2026-11-11")

    def test_rebuilt_zip_with_same_data_is_not_a_new_version(self):
        self.publish(make_zip(self.d / "a.zip", [1, 2, 3]))
        rebuilt = make_zip(self.d / "b.zip", [1, 2, 3], when=(2026, 11, 12, 9, 30, 0))
        self.assertNotEqual(V.sha256_file(rebuilt), V.sha256_file(self.d / "a.zip"))
        r = self.publish(rebuilt)
        self.assertEqual(r["status"], "unchanged")
        self.assertEqual(self.gh.uploads, [f"{DS}.zip"])
        self.assertEqual(self.rec()["zip_sha256"], V.sha256_file(self.d / "a.zip"))  # the record names the served file

    def test_history_survives_a_rewritten_record(self):
        """stdlib.release() rewrites the record without url or versions: the history comes back from the assets."""
        self.publish(make_zip(self.d / "a.zip", [1, 2, 3]))
        self.publish(make_zip(self.d / "b.zip", [1, 2, 4]))
        self.rec_path.write_text(json.dumps({"id": DS, "rows": 3, "zip_sha256": "y", "zip_bytes": 1}))
        r = self.publish(make_zip(self.d / "c.zip", [9]))
        self.assertEqual(r["file"], f"{DS}.v3.zip")
        self.assertEqual([v["file"] for v in self.rec()["versions"]], [f"{DS}.zip", f"{DS}.v2.zip", f"{DS}.v3.zip"])

    def test_without_digests_uses_record_then_download(self):
        self.gh.digests = False
        z = make_zip(self.d / "a.zip", [1, 2, 3])
        self.publish(z)
        self.assertEqual(self.publish(z)["status"], "unchanged")  # sha256 from the record, no download
        self.assertEqual(self.gh.downloads, [])
        self.rec_path.write_text(json.dumps({"id": DS}))         # record lost its hashes: download and hash
        self.assertEqual(self.publish(z)["status"], "unchanged")
        self.assertEqual(self.gh.downloads, [f"{DS}.zip"])

    def test_name_taken_meanwhile_takes_the_next_version(self):
        self.publish(make_zip(self.d / "a.zip", [1, 2, 3]))
        self.gh.steal.add(f"{DS}.v2.zip")
        r = self.publish(make_zip(self.d / "b.zip", [1, 2, 4]))
        self.assertEqual(r["file"], f"{DS}.v3.zip")
        self.assertEqual(self.gh.files[f"{DS}.v2.zip"], b"someone else's file")
        self.assertEqual([v["version"] for v in self.rec()["versions"]], [1, 2, 3])

    def test_other_datasets_assets_are_ignored(self):
        self.gh.files["Smith2024_JNeurophysiol_Exp2.zip"] = b"other"
        self.gh.created["Smith2024_JNeurophysiol_Exp2.zip"] = "2026-11-01T00:00:00Z"
        r = self.publish(make_zip(self.d / "a.zip", [1]))
        self.assertEqual((r["status"], r["file"]), ("published", f"{DS}.zip"))

    def test_versions_continue_in_a_new_release(self):
        """When 'standardized' is full, new files go to a new tag; numbering continues from the old release."""
        self.publish(make_zip(self.d / "a.zip", [1]))
        self.gh.releases["standardized-2"] = {}
        r = V.publish_zip(DS, make_zip(self.d / "b.zip", [2]), "standardized-2", gh=self.gh, record_path=self.rec_path,
                          also_tags=[TAG])
        self.assertEqual((r["file"], r["url"].split("/")[-2]), (f"{DS}.v2.zip", "standardized-2"))
        self.assertEqual([(v["release"], v["version"]) for v in self.rec()["versions"]], [(TAG, 1), ("standardized-2", 2)])

    def test_full_release_is_refused(self):
        for i in range(V.MAX_ASSETS):
            self.gh.files[f"Other{i}_X.zip"] = b"x"
            self.gh.created[f"Other{i}_X.zip"] = "2026-11-01T00:00:00Z"
        with self.assertRaises(RuntimeError):
            self.publish(make_zip(self.d / "a.zip", [1]))

    def test_record_as_dict(self):
        rec = {"id": DS}
        r = V.publish_zip(DS, make_zip(self.d / "a.zip", [1]), TAG, gh=self.gh, record=rec)
        self.assertEqual(rec["url"], r["url"])
        self.assertEqual(len(rec["versions"]), 1)

    def test_version_names(self):
        self.assertEqual(V.version_of(DS, f"{DS}.zip"), 1)
        self.assertEqual(V.version_of(DS, f"{DS}.v12.zip"), 12)
        self.assertIsNone(V.version_of(DS, f"{DS}.v1.zip"))
        self.assertIsNone(V.version_of(DS, f"{DS}_Exp2.zip"))
        with self.assertRaises(ValueError):
            V.publish_zip("../evil", self.d / "a.zip", TAG, gh=self.gh)


if __name__ == "__main__":
    unittest.main()
