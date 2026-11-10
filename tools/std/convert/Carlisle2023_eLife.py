"""Carlisle & Kuo (2023), eLife 12:e81939: short walking bouts of 10 distances, foot-mounted IMUs, one row per bout file.

Run from the site folder: python3 -I tools/std/convert/Carlisle2023_eLife.py
Raw file (untrusted data, read only) in raw_extra/Carlisle2023_eLife/repo.zip, the GitHub archive of https://github.com/kuo-lab/short_walk_experiment:
  saved_walk_info/new/subject<S>distance<D>section<K>trial<T>.mat (left_walk_info and right_walk_info structs), analyze_short_walks_new.m (read as text only).
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Carlisle2023_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
TMP = RAW / "samples_tmp"
TMP.mkdir(exist_ok=True)
PERIOD = 0.0078  # s, PERIOD in the authors' analysis script
DIST = [1.1, 1.7, 2.2, 2.8, 3.3, 3.8, 5.1, 7, 9.1, 12.7]  # m, paper Methods (same list as actual_dist in the script)
FIELDS = [("P", 3, "m", ["x", "y", "z"]), ("V", 3, "m/s", ["x", "y", "z"]), ("Vm", 1, "m/s", [""]),
          ("euler", 3, "not stated", ["1", "2", "3"]), ("W", 3, "not stated", ["x", "y", "z"]),
          ("A", 3, "not stated", ["x", "y", "z"]), ("quaternion", 4, "unitless", ["w", "x", "y", "z"]), ("is_walking", 1, "flag 0/1", [""])]
CH, UN = [], []
for side in ("left", "right"):
    for f, n, u, ax in FIELDS:
        for a in ax:
            CH.append(f"{side}_{f}" + (f"_{a}" if a else ""))
            UN.append(u)

z = zipfile.ZipFile(RAW / "repo.zip")
pat = re.compile(r"subject(\d+)distance(\d+)section(\d+)trial(\d+)\.mat$")
files = sorted((tuple(map(int, m.groups())), n) for n in z.namelist() if "saved_walk_info/new/" in n for m in [pat.search(n)] if m)
rows, extra, count = [], [], {}
for (s, d, k, t), name in files:
    m = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)
    L, R = m["left_walk_info"], m["right_walk_info"]
    n = len(L.is_walking)
    cols = {"time_s": np.round(np.arange(n) * PERIOD, 4)}
    for side, w in (("left", L), ("right", R)):
        assert len(w.is_walking) == n
        for f, nn, u, ax in FIELDS:
            v = np.asarray(getattr(w, f), dtype=float).reshape(n, nn)
            for j, a in enumerate(ax):
                cols[f"{side}_{f}" + (f"_{a}" if a else "")] = v[:, j]
    count[s] = count.get(s, 0) + 1
    fn = f"samples/{ID}_S{s:04d}_distance{d:02d}_section{k}_trial{t}.csv"
    p = TMP / Path(fn).name
    pd.DataFrame(cols).to_csv(p, index=False, float_format="%.6g", lineterminator="\r\n")
    extra.append((fn, p))
    ev = []
    for side, w in (("left", L), ("right", R)):
        ev += [f"{side}_foot_flat_start={(int(a) - 1) * PERIOD:.4f}" for a in np.atleast_1d(w.FFstart)]
        ev += [f"{side}_foot_flat_end={(int(a) - 1) * PERIOD:.4f}" for a in np.atleast_1d(w.FFend)]
    # the authors' script treats sections 1 and 3 (subject 3: sections 1 and 4) as the main set
    main = k in ((1, 4) if s == 3 else (1, 3))
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group="", subject=str(s), person=str(s), subject_excluded_by_authors=False,
        age="", sex="", handedness="", population="healthy", setting="lab",
        device="Two foot-mounted inertial sensors (APDM Opal), outdoors", session=1,
        block=f"distance {DIST[d - 1]} m, section {k}", trial=count[s], phase="test", time_s="", task="walking", unit="recording",
        condition=f"walk distance {DIST[d - 1]} m", pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=count[s], samples_file=fn, signal_kind="foot IMU: integrated position, velocity, orientation, angular rate and acceleration of both feet, one short walking bout",
        channels=";".join(CH), channel_units=";".join(UN), sample_rate_hz=round(1 / PERIOD, 2), n_samples=n, duration_s=round((n - 1) * PERIOD, 4),
        events=";".join(ev), processing="authors' processed walk_info structs (saved_walk_info/new): IMU-derived fields as saved by the authors",
        trial_info=f"distance_m={DIST[d - 1]};distance_index={d};section={k};trial_in_section={t};main_set_in_authors_script={'yes' if main else 'no'}"))
df = pd.DataFrame(rows)
main = df.trial_info.str.contains("main_set_in_authors_script=yes")

checks.published("participants", df.person.nunique(), 10, 0, "Methods, Subjects (N = 10)")
checks.published("walking distances", df.trial_info.str.extract(r"distance_m=([\d.]+)")[0].astype(float).nunique(), 10, 0, "Methods (1.1 ... 12.7 m)")
checks.published("trials in the main (sidewalk) set", int(main.sum()), 400, 0, "Methods (a total of 400 trials)")
checks.published("trials per distance per participant in the main set", int(df[main].groupby(["person", "condition"]).size().mean()), 4, 0, "Methods (each distance four times)")
checks.internal("800 recordings, 80 per participant", len(df) == 800 and set(count.values()) == {80}, str(len(df)))
checks.internal("every recording has 42 channels plus time", all(len(r["channels"].split(";")) == 42 for r in rows))
checks.internal("main-set sections are 2 per participant-distance pair of trials", bool(main.sum() == 400))

meta = dict(
    paper=dict(authors="Carlisle RE, Kuo AD", year=2023, title="Optimization of energy and time predicts dynamic speeds for human walking",
               venue="eLife 12:e81939", doi="10.7554/eLife.81939"),
    lab="Kuo (University of Calgary)", data_url="https://github.com/kuo-lab/short_walk_experiment",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Read the MATLAB structs (left_walk_info and right_walk_info) of the 800 per-trial files and wrote one CSV per file with a time_s column (sample number x 0.0078 s); values are the authors' own, "
             "and the fields An, Anz and Az (variants of A) were left out. Foot-flat start and end sample numbers (FFstart, FFend) are listed in events as times. "
             "peak_speed_all_sub.mat (derived speeds) is not included."),
    how=["One row is one short walking bout: a participant walked a marked distance (1.1 to 12.7 m, 10 distances) starting and stopping at rest, with an inertial sensor on each foot. Each participant has "
         "80 files (10 distances x 4 sections x 2 trials). Positions P are in metres and velocities V in m/s as used in the authors' script; the units of euler, W (angular rate) and A are not stated in the deposit.",
         "The sampling interval is 0.0078 s (about 128 Hz), taken from PERIOD in the authors' script. Row order is distance, section, trial (file names), not necessarily the order walked.",
         "The paper analyses 400 sidewalk trials; the deposit does not label terrain. The authors' script analyses sections 1 and 3 (for one participant sections 1 and 4, because that participant's sections are out of order), "
         "which gives exactly 400 files, so trial_info marks those as main_set_in_authors_script=yes; the other sections are taken to be the grass trials the paper mentions, which is an inference. "
         "Sex and age are not shared per participant."],
    differences=["Published speeds and durations (for example peak speed range 1.21 to 1.78 m/s at 12.7 m) were not reproduced: they come from the authors' manual step analysis and are not in the per-trial files."],
    unit_note="walking bout file (both feet, 42 IMU channels)",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
