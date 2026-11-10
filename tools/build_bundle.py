"""One ZIP with every dataset folder (CSV, README, columns.csv, LICENSE), plus catalog.csv, columns.csv and README.
Too big for GitHub Pages (100 MB file limit), so it is uploaded as a GitHub release asset."""
import shutil, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "source/bundle/openmotor-all-datasets.zip"
zips = sorted((HERE / "dist/data").glob("*.zip"))
assert len(zips) == 99, len(zips)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as big:
    big.write(HERE / "source/repo_README.md", "openmotor/README.md")
    big.write(HERE / "dist/catalog.csv", "openmotor/catalog.csv")
    big.write(HERE / "dist/columns.csv", "openmotor/columns.csv")
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                zi = zipfile.ZipInfo("openmotor/datasets/" + info.filename, date_time=info.date_time)
                zi.compress_type = zipfile.ZIP_DEFLATED  # a hand-made ZipInfo defaults to STORED (no compression)
                with z.open(info) as src, big.open(zi, "w") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
print(OUT.name, round(OUT.stat().st_size / 1e6, 1), "MB from", len(zips), "datasets")
