"""Kulvicius et al. (2023, Commun Med 3:112): centre-of-pressure time series of 1776 five-second pressure-mat snippets from 45
typically developing infants, labelled fidgety movements present (FM+) or absent (FM-).

Raw files, raw_extra/Kulvicius2023_CommunMed/ (Zenodo 8104097, fetched with tools/std/fetch.py):
  data.zip  5fold_split_1/input_5fold_split_1_fold_<k>_test.txt  = the authors' centre-of-pressure features (rows = snippets,
            3000 columns = 500 frames x 6 features x_top,y_top,p_top,x_bottom,y_bottom,p_bottom, values scaled 0..1), with
            sample_list_mat_..._test.txt (snippet file names) and output_..._test.txt (0/1 labels). The five test folds of one
            split together hold every snippet exactly once, so they give all 1776 snippets.
The raw 32x32 pressure frames (pressure_mat_data.zip, 1776 x 500 x 1024 values) are not carried: they would be about 1.8 GB of text.
Run from the site folder: python3 -I tools/std/convert/Kulvicius2023_CommunMed.py
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kulvicius2023_CommunMed"
SRC = "Kulvicius et al. 2023, Commun Med 3:112"
CH = ["x_top", "y_top", "p_top", "x_bottom", "y_bottom", "p_bottom"]


def main():
    checks = stdlib.Checks()
    z = zipfile.ZipFile(stdlib.raw_dir(ID) / "data.zip")
    recs = []
    for k in range(1, 6):
        base = f"data/5fold_split_1/%s_5fold_split_1_fold_{k}_test.txt"
        names = z.read(f"data/5fold_split_1/sample_list_mat_5fold_split_1_fold_{k}_test.txt").decode().split()
        X = np.loadtxt(io.BytesIO(z.read(base % "input")))
        outn = [n for n in z.namelist() if re.search(rf"output_5fold_split_1_fold_{k}_test\d*\.txt$", n)][0]
        y = np.array(z.read(outn).decode().split(), dtype=int)
        assert X.shape == (len(names), 3000) and len(y) == len(names), (k, X.shape, len(names), len(y))
        for i, n in enumerate(names):
            m = re.fullmatch(r"idxp(\d+)_(\d+)d_camera1_frames_(\d+)-(\d+)_mat_samples_(\d+)-(\d+)_label_([01])\.txt", n)
            assert m, n
            assert int(m.group(7)) == y[i]
            recs.append(dict(inf=int(m.group(1)), age_d=int(m.group(2)), f0=int(m.group(3)), f1=int(m.group(4)),
                             label=int(m.group(7)), X=X[i].reshape(500, 6)))
    recs.sort(key=lambda r: (r["inf"], r["age_d"], r["f0"]))
    checks.internal("every snippet appears once", len(recs) == 1776 and len({(r["inf"], r["age_d"], r["f0"]) for r in recs}) == 1776)
    checks.internal("features within 0..1", all(r["X"].min() >= 0 and r["X"].max() <= 1 for r in recs))
    sub_no, rows, extra, count = {}, [], [], {}
    for r in recs:
        s = sub_no.setdefault(r["inf"], len(sub_no) + 1)
        count[s] = count.get(s, 0) + 1
        n = count[s]
        code = f"{ID}_S{s:04d}"
        fn = f"samples/{code}_rec{n:03d}.csv"
        d = pd.DataFrame(r["X"], columns=CH)
        d.insert(0, "time_s", np.arange(500) / 100.0)
        extra.append((fn, d.to_csv(index=False, float_format="%.9g", lineterminator="\r\n")))
        rows.append(dict(
            dataset=ID, study=SRC, experiment="Fidgety movement classification", group="", subject=f"I{r['inf']}", person=f"I{r['inf']}",
            subject_excluded_by_authors=False, age=r["age_d"] / 365.25, sex="", handedness="", population="healthy (typically developing infants)",
            setting="lab", device="Tekscan Conformat pressure mat, 32 x 32 sensors, 471.4 x 471.4 mm, 100 Hz",
            session="", block=f"age {r['age_d']} days", trial=n, phase="test", time_s="", task="infant spontaneous movements (supine)",
            unit="recording", condition="fidgety movements present" if r["label"] else "fidgety movements absent",
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            recording=n, samples_file=fn, signal_kind="centre of pressure (x, y) and total pressure of the top and bottom halves of the pressure mat image",
            channels=";".join(CH), channel_units=";".join(["normalized 0-1"] * 6), sample_rate_hz=100, n_samples=500, duration_s=5,
            events="", processing="the authors' centre-of-pressure features computed from the pressure images, scaled to 0..1",
            trial_info=f"fidgety_movements={'present' if r['label'] else 'absent'};age_days={r['age_d']};video_frames={r['f0']}-{r['f1']}"))
    df = pd.DataFrame(rows)
    checks.internal("sample file names carry 45 subject codes", len({f.split("/")[1].split("_rec")[0] for f, _ in extra}) == 45)
    persons = df.drop_duplicates("person")
    checks.published("infants", len(persons), 45, 0, f"{SRC}, Abstract (N = 45)")
    checks.published("pressure snippets", len(df), 1776, 0, f"{SRC}, Abstract (1776 snippets)")
    checks.published("snippet length (s)", float(df.duration_s.iloc[0]), 5, 0, f"{SRC}, Abstract (5 s snippets)")
    checks.published("sampling rate (Hz)", float(df.sample_rate_hz.iloc[0]), 100, 0, "deposit README (pressure mat, 100 Hz)")
    checks.internal("ages within the 4-16 week sessions (20 to 130 days)", 20 <= df.age.min() * 365.25 and df.age.max() * 365.25 <= 130,
                    f"{df.age.min()*365.25:.0f}-{df.age.max()*365.25:.0f} days")
    s = json.load(open(stdlib.HERE / "source/extra_studies.json"))
    s = next(x for x in s if x["id"] == ID)
    p = dict(s["papers"][0])
    meta = dict(
        study=SRC, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Took the authors' centre-of-pressure feature matrices of the five test folds of the first cross-validation split "
                 "(together every snippet once), reshaped each snippet to 500 frames x 6 channels, one samples CSV per snippet, with "
                 "time in seconds; age in days from the file name converted to years. The raw 32 x 32 pressure frames are not included."),
        how=["One row is one 5 s snippet of the pressure mat under a supine infant (500 frames at 100 Hz). Its samples have time_s and the "
             "six channels x_top, y_top, p_top, x_bottom, y_bottom, p_bottom: centre of pressure position and total pressure of the top "
             "and the bottom area of the mat image, as the authors' feature script computed them and scaled to 0..1 (the scaling "
             "is as shipped; the deposit does not state units, so no physical unit is claimed).",
             "Each snippet was labelled by assessors from synchronized video as fidgety movements present (FM+) or absent (FM-); the label "
             "is the condition. age is the infant's age at the recording from the file name (days / 365.25); infants were recorded in "
             "biweekly sessions between 4 and 16 weeks of post-term age. Sex is not shared per infant. Rows are ordered by infant, age and video frame."],
        unit_note="5-second pressure-mat snippet (centre-of-pressure time series)")
    return stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)


if __name__ == "__main__":
    main()
