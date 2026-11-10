"""O'Day, Lee, Seagers, Hoffman, Jih-Schiff, Kidzinski, Delp & Bronte-Stewart 2022 (J NeuroEng Rehabil): seven people with Parkinson's disease
walked a turning-and-barrier course off medication and off DBS while wearing APDM inertial sensors (128 Hz); raters labelled freezing of gait
(FOG) from video. Deposit: GitHub stanfordnmbl/imu-fog-detection (BSD-3-Clause), data/raw/imus6_subjects7 (6 sensors, all 60 walks) and
data/raw/imus11_subjects4 (the 11-sensor version of 20 of those walks). One .xlsx per walk; the 11-sensor file is used where it exists.
The deposit's code is never run; the .xlsx files are read with openpyxl."""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "ODay2022_JNeuroengRehabil"
MODULE = "timeseries"
SEX = {1: "M", 2: "M", 3: "F", 4: "F", 5: "M", 6: "F", 7: "M"}
# Table 1: walks, % time FOG (mean, SD across walks), FOG events (mean, SD)
T1 = {1: (7, 19.8, 13.5, 4.4, 2.6), 2: (5, 12.3, 25.0, 1.8, 3.5), 3: (9, 23.7, 11.1, 4.2, 2.5), 4: (6, 53.7, 19.8, 8.0, 2.0),
      5: (9, 9.9, 11.3, 2.0, 2.2), 6: (14, 5.3, 5.8, 1.4, 1.6), 7: (10, 28.3, 19.8, 4.8, 3.2)}
LOC = {"chest": "chest", "lumbar": "lumbar", "head": "head", "wrist_r": "right wrist", "wrist_l": "left wrist", "thigh_r": "right thigh",
       "thigh_l": "left thigh", "ankle_r": "right ankle", "ankle_l": "left ankle", "foot_r": "right foot", "foot_l": "left foot"}


def read_xlsx(b):
    ws = next(iter(openpyxl.load_workbook(io.BytesIO(b), read_only=True, data_only=True)))
    it = ws.iter_rows(values_only=True)
    head = list(next(it))
    rows = [r for r in it if r and r[0] is not None]
    return pd.DataFrame(rows, columns=head)


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "repo.zip")
    files = {}
    for n in z.namelist():
        m = re.search(r"data/raw/(imus6_subjects7|imus11_subjects4)/(pt(\d+)_visit_([\d.]+)_tbc_walklr_(\d)_trial_(\d+))\.xlsx$", n)
        if m:
            files.setdefault(m[2], {})[m[1]] = (n, int(m[3]), m[4], int(m[5]), int(m[6]))
    out = Path("scratch") / "oday_samples"
    out.mkdir(parents=True, exist_ok=True)
    rows, extra, rec = [], [], {}
    stats = {}
    n_super = same = 0
    mins = 0.0
    for key in sorted(files, key=lambda k: (files[k][next(iter(files[k]))][1], float(files[k][next(iter(files[k]))][2]), files[k][next(iter(files[k]))][4], k)):
        v = files[key]
        n, s, visit, lr, trial = v.get("imus11_subjects4", v["imus6_subjects7"])
        d = read_xlsx(z.read(n))
        if "imus11_subjects4" in v and "imus6_subjects7" in v:
            d6 = read_xlsx(z.read(v["imus6_subjects7"][0]))
            n_super += 1
            c = [x for x in d6.columns if x in d.columns]
            same += int(len(d6) == len(d) and all(np.allclose(d6[x].to_numpy(float), d[x].to_numpy(float), equal_nan=True) for x in c) and len(c) == len(d6.columns))
        assert d.columns[0] == "subject_ID" and d.columns[1] == "time" and d.columns[-1] == "freeze_label"
        assert (d.subject_ID == s).all()
        chans = list(d.columns[2:])
        lab = d.freeze_label.to_numpy(float)
        t = d.time.to_numpy(float)
        dt = float(np.median(np.diff(t)))
        edges = np.diff(np.r_[0, lab, 0])
        on, off = np.where(edges == 1)[0], np.where(edges == -1)[0]
        stats[key] = (s, len(d) * dt / 60, 100 * lab.mean(), len(on))
        rec[s] = rec.get(s, 0) + 1
        fn = f"samples/{key}.csv"
        lp = out / f"{key}.csv"
        o = d.drop(columns=["subject_ID"]).rename(columns={"time": "time_s"})
        o.to_csv(lp, index=False, float_format="%.9g", lineterminator="\r\n")
        extra.append((fn, str(lp)))
        units = ["m/s^2" if c.endswith(("_ax", "_ay", "_az")) else ("rad/s" if c.endswith(("_gx", "_gy", "_gz")) else "0/1") for c in chans]
        sensors = sorted({re.match(r"imu_(.+)_[ag][xyz]$", c)[1] for c in chans if c.startswith("imu_")})
        ev = [f"fog_start={t[i]:.4f}" for i in on] and [x for a, b in zip(on, off) for x in (f"fog_start={t[a]:.4f}", f"fog_end={(t[b - 1] + dt):.4f}")]
        rows.append(dict(
            dataset=DS, study=DS, experiment="", group="", subject=f"p{s}", subject_excluded_by_authors=False, age="", sex=SEX[s], handedness="",
            population="Parkinson's disease", setting="clinic", device="APDM wearable inertial sensors (accelerometer + gyroscope), " + ", ".join(LOC[x] for x in sensors),
            session="", block=f"visit {visit}", trial=rec[s], phase="test", time_s=t[0], task="walking", unit="recording",
            condition="turning and barrier course, off medication and off DBS", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=False, recording=rec[s], samples_file=fn,
            signal_kind="inertial sensor accelerations and angular velocities with a video-rated freezing-of-gait label",
            channels=";".join(chans), channel_units=";".join(units), sample_rate_hz=128, n_samples=len(d), duration_s=(len(d) - 1) * dt,
            events=";".join(ev), processing="none stated for the shared signals (the paper downsampled to 64 Hz and windowed them only for its models)",
            trial_info=f"visit_number={visit};walklr={lr};trial_number={trial};sensors={len(sensors)}"))
    df = pd.DataFrame(rows)
    nonrep = []
    st = pd.DataFrame(stats.values(), columns=["s", "min", "pfog", "nfog"])
    ck.internal("each walk's 11-sensor file holds the 6-sensor file's columns and values unchanged", same == n_super, f"{same}/{n_super}")
    ck.internal("every walk is a 128 Hz recording", (df.sample_rate_hz == 128).all())
    ck.published("participants", df.subject.nunique(), 7, 0, "Methods, Participants / Table 1")
    ck.published("unique walks", len(df), 60, 0, "Results, IMU walking data")
    for s, (nw, pm, ps, nm, ns) in T1.items():
        g = st[st.s == s]
        ck.published(f"participant {s}: number of walks", len(g), nw, 0, "Table 1")
        ck.published(f"participant {s}: % time FOG, mean across walks", g.pfog.mean(), pm, 0.05, "Table 1")
        sd = g.pfog.std(ddof=1)
        if abs(sd - ps) <= 0.05:
            ck.published(f"participant {s}: % time FOG, SD across walks", sd, ps, 0.05, "Table 1")
        else:
            nonrep.append(f"participant {s} SD of % time FOG across walks: {sd:.3f} computed from the labels, {ps} printed in Table 1")
        ck.published(f"participant {s}: FOG events per walk, mean", g.nfog.mean(), nm, 0.05, "Table 1")
    nonrep.append(f"total walking time: {st['min'].sum():.1f} min from the rows (rows x sampling interval), 88.8 min in the paper")
    ck.published("FOG events in all walks", st.nfog.sum(), 211, 0, "Methods, Participants")
    nonrep.append(f"% of time freezing in the full dataset: {(st.pfog * st['min']).sum() / st['min'].sum():.1f} from the rows (walk-duration-weighted), 23.9 in the paper")
    ck.published("% time FOG, mean across all 60 walks", st.pfog.mean(), 19.7, 0.05, "Table 1, Total")
    meta = dict(
        study=DS,
        paper=dict(authors="O'Day J, Lee M, Seagers K, Hoffman S, Jih-Schiff A, Kidzinski L, Delp S, Bronte-Stewart H", year=2022,
                   title="Assessing inertial measurement unit locations for freezing of gait detection and patient preference",
                   venue="J NeuroEngineering Rehabil 19(1):20", doi="10.1186/s12984-022-00992-x"),
        lab="Delp / Bronte-Stewart (Stanford University)", data_url="https://github.com/stanfordnmbl/imu-fog-detection",
        license="BSD 3-Clause", license_url="https://github.com/stanfordnmbl/imu-fog-detection/blob/main/LICENSE",
        license_text="Copyright (c) 2021, Stanford Neuromuscular Biomechanics Laboratory. All rights reserved. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the copyright notice, the list of conditions and the disclaimer are retained, and the name of the copyright holder is not used to endorse derived products (BSD 3-Clause License of the repository, which holds the data).",
        changes=("Wrote each walk's spreadsheet to a CSV (time_s, the sensor channels and the freezing label, 9 significant digits, subject column removed) and listed the start and end of every "
                 "labelled freezing episode as events; where a walk was shared twice (6-sensor and 11-sensor versions) only the 11-sensor version, which contains the same columns and values, is kept; "
                 "participant numbers became codes and sex comes from the paper's Table 1."),
        how=["One row is one walk (one recording, about 1.5 minutes at 128 Hz) of one person with Parkinson's disease through a turning and barrier course, off medication and off deep brain stimulation. "
             "The samples CSV has time_s (the deposit's time column, in seconds from the start of the whole session, so a walk does not start at 0), the 3 accelerations (m/s^2) and 3 angular velocities (rad/s) "
             "of each sensor, and freeze_label (1 while the rater marked freezing of gait on the synchronised video, else 0). Eleven sensors (head, chest, lumbar, both wrists, both thighs, both ankles, both feet) "
             "for 20 walks of 4 participants, and the 6-sensor subset (chest, lumbar, ankles, feet) for the other walks; sensors that were not worn are empty columns or absent as the sensors field says.",
             "block is the clinic visit number from the file name (the paper says visits were up to 44 months apart; the unit of the number is not stated). 'walklr' in trial_info is the 0 or 1 flag in the file name, "
             "whose meaning the deposit does not explain, and trial_number is the number in the file name. Sex is from the paper's Table 1, assuming the deposit's participant number is the paper's participant number "
             "(the numbers of walks per participant agree). events are the start and end times of each run of freeze_label = 1 (end = time of the last labelled sample plus one sample)."],
        unit_note="walk (one recording of a turning-and-barrier course walk)",
        differences=nonrep + ["The listing says the licence is BSD; the repository's BSD 3-Clause licence is the only licence stated and covers the whole repository including the data."],
    )
    stdlib.release(DS, MODULE, df, meta, ck, extra_files=extra)


if __name__ == "__main__":
    main()
