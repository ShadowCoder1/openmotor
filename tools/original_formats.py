"""List the file formats of the data each study's authors shared, for the Studies table's Format column.

Reads only file names (and, for ZIP files, the archive's list of names); no data file is opened or changed.
Runs on the Mac or the PC:
    python original_formats.py <motor-centaur folder> <out.json> [dataset ...]
External datasets: data/external/<dataset>/raw. OpenMotor-origin datasets: the study's "Standardized Data"
folder in data/openmotor/OpenMotor Database. Datasets whose files are not on this machine are skipped.
"""
import collections
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT, OUT, WANT = Path(sys.argv[1]), sys.argv[2], set(sys.argv[3:])
KIND = {".csv": "CSV", ".tsv": "TSV", ".txt": "text", ".dat": "text", ".mat": "MATLAB", ".xlsx": "Excel", ".xls": "Excel",
        ".rdata": "R", ".rda": "R", ".rds": "R", ".json": "JSON", ".h5": "HDF5", ".hdf5": "HDF5", ".npy": "NumPy",
        ".npz": "NumPy", ".pkl": "Python pickle", ".pickle": "Python pickle", ".parquet": "Parquet", ".sav": "SPSS"}
NOT_DATA = ("readme", "license", "licence", "codebook", "dictionary", "description", "confirmation")


def names_in(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if f.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(p) as z:
                        out += [n for n in z.namelist() if not n.endswith("/")]
                except zipfile.BadZipFile:
                    pass
            else:
                out.append(p)
    return out


def kinds(names):
    c = collections.Counter()
    for n in names:
        low = n.replace("\\", "/").lower()
        base = low.rsplit("/", 1)[-1]
        if "__macosx/" in low or base.startswith(".") or any(w in base for w in NOT_DATA):
            continue
        k = KIND.get(os.path.splitext(base)[1])
        if k:
            c[k] += 1
    if len(c) > 1:
        c.pop("JSON", None)          # metadata sidecars next to the data (BIDS, VR logs)
    if len(c) > 1 and c.get("text", 0) <= 2:
        c.pop("text", None)          # a note or two next to the data files
    return dict(c.most_common())


studies = {r["dataset"]: r for r in csv.DictReader(open(ROOT / "data/studies.tsv", encoding="utf-8"), delimiter="\t")}
found = {}
for ds, r in studies.items():
    if WANT and ds not in WANT:
        continue
    if ds.startswith("ext_"):
        folder = ROOT / "data/external" / ds / "raw"
    else:
        folder = ROOT / "data/openmotor/OpenMotor Database" / r["folder"] / "Standardized Data"
    if folder.is_dir():
        found[ds] = kinds(names_in(folder))
json.dump(found, open(OUT, "w"), indent=1)
print(f"{len(found)} datasets scanned")
