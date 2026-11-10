"""Braun & Gegenfurtner (2016) J Vis: dynamics of oculomotor direction discrimination.

Raw files (Zenodo 10.5281/zenodo.150311, CC BY 4.0): PursuitTrained.zip, PursuitUntrained.zip, SaccTrained.zip, SaccUntrained.zip and
README.rtf. Each zip holds one folder per observer, one sub-folder per target direction, and one 4-column text file (*.pos) per
accepted trial: trial time (s), gaze x, gaze y, time relative to pursuit onset (s). Read as text only; the extra
*.dsac / *.purpos / *.saccpos files and the summary files in SaccUntrained and the percup.0.pos.mat files are not
documented in the README and are left out.

Run from the site folder: python3 -I tools/std/convert/Braun2016_JVis.py
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Braun2016_JVis"
MODULE = "timeseries"
ZIPS = [("PursuitTrained", "step-ramp pursuit", "trained"), ("PursuitUntrained", "step-ramp pursuit", "untrained"),
        ("SaccTrained", "ramp with initial saccade", "trained"), ("SaccUntrained", "ramp with initial saccade", "untrained")]
DIR_ORDER = ["-10", "-5", "-2", "0", "+2", "+5", "+10"]
DEVICE = {"trained": "EyeLink 1000 Tower Mount (right eye, 1000 Hz), CRT monitor 100 Hz at 47 cm",
          "untrained": "EyeLink 1000 table mount (right eye, 1000 Hz), Display++ monitor 120 Hz at 90 cm"}


def fmt(x):
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return format(float(x), ".8g")


def main():
    raw = stdlib.raw_dir(DS)
    rows, extra = [], []
    person_of = {}
    k = 0
    for zn, exp, grp in ZIPS:
        z = zipfile.ZipFile(raw / f"{zn}.zip")
        mem = [x for x in z.namelist() if x.endswith(".pos") and not x.startswith("__MACOSX")]
        parsed = []
        for x in mem:
            p = x.split("/")
            parsed.append((p[1], DIR_ORDER.index(p[2]), p[-1], x))
        parsed.sort()
        seq = {}
        for folder, di, fname, x in parsed:
            m = re.fullmatch(r"([a-z]+?)(\d\d)", folder)
            assert m, "unexpected observer folder name"
            who, suffix = m.group(1)[:2], m.group(2)
            a = np.loadtxt(io.StringIO(z.read(x).decode("ascii")), ndmin=2)
            assert a.shape[1] == 4 and np.isfinite(a).all()
            seq[folder] = seq.get(folder, 0) + 1
            k += 1
            rel = f"samples/{zn}_{k:05d}.csv"
            body = ["time_s,gaze_x,gaze_y,time_from_pursuit_onset_s"] + [",".join(fmt(v) for v in r) for r in a]
            extra.append((rel, ("\r\n".join(body) + "\r\n").encode()))
            tag = re.sub(r"^[a-zA-Z]+", "", fname[:-4])
            dirv = DIR_ORDER[di]
            person_of[(zn, folder)] = who
            rows.append(dict(
                dataset=DS, study=DS, experiment=exp, group=grp, subject=f"{zn}/{folder}", person=who,
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device=DEVICE[grp], session="", block=f"target vertical component {dirv} deg", trial=seq[folder], phase="test",
                time_s="", task="eye movements", unit="recording", condition=f"vertical component {dirv} deg",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=seq[folder], samples_file=rel, signal_kind="gaze position of the right eye",
                channels="gaze_x;gaze_y;time_from_pursuit_onset", channel_units="deg;deg;s", sample_rate_hz=1000,
                n_samples=len(a), duration_s=a[-1, 0] - a[0, 0],
                events=f"pursuit_onset={fmt(round(a[0, 0] - a[0, 3], 6))}",
                processing="filtered by the authors before sharing (second-order Butterworth low-pass, 30 Hz, per the paper)",
                trial_info=f"vertical_component_deg={dirv}; file_tag={tag}; observer_folder_suffix={folder[-2:]}"))
    ck = stdlib.Checks()
    import collections
    pers = {g: {r["person"] for r in rows if r["group"] == g} for g in ("trained", "untrained")}
    ck.published("trained observers", len(pers["trained"]), 4, 0, "Methods, Observers")
    ck.published("untrained (naive) observers", len(pers["untrained"]), 6, 0, "Methods, Observers")
    ck.published("target vertical components", len({r["condition"] for r in rows}), 7, 0, "Methods, Paradigms (0, +-2, +-5, +-10 deg)")
    ck.published("sampling rate (Hz)", float(np.median([r["sample_rate_hz"] for r in rows])), 1000, 0, "Methods, Eye movement recording")
    dts = []
    blob = dict(extra)
    for r in rows[::97]:
        t = np.loadtxt(io.StringIO(blob[r["samples_file"]].decode()), delimiter=",", skiprows=1, usecols=0)
        dts.append(np.diff(t).mean())
    ck.internal("samples spaced 1 ms in sampled files", bool(np.allclose(dts, 0.001, atol=1e-6)))
    cnt = collections.Counter((r["experiment"], r["group"]) for r in rows)
    ck.internal("trial counts per paradigm and group", True, str(dict(cnt)))
    ck.internal("every .pos file kept: 5727 + 3834 + 6222 + 4132 recordings", len(rows) == 5727 + 3834 + 6222 + 4132, str(len(rows)))
    ck.internal("untrained recordings: 12 observer folders per paradigm = 6 observers x 2 folders",
                len({r["subject"] for r in rows if r["group"] == "untrained" and r["experiment"] == "step-ramp pursuit"}) == 12)
    unt = collections.Counter(r["person"] for r in rows if r["group"] == "untrained" and r["experiment"] == "step-ramp pursuit")
    ck.internal("no untrained observer has more pursuit trials than the 5 blocks x 150 collected (750)",
                bool(all(v <= 750 for v in unt.values())), str(sorted(unt.values())))
    meta = dict(
        study=DS,
        paper=dict(authors="Braun DI, Gegenfurtner KR", year=2016, title="Dynamics of oculomotor direction discrimination",
                   venue="J Vis 16(13):4", doi="10.1167/16.13.4"),
        lab="Gegenfurtner (University of Giessen)", data_url="https://doi.org/10.5281/zenodo.150311",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Re-encoded each trial's text file as a CSV (time in seconds, gaze x and y, time from pursuit onset) listed in "
                 "one table with participant codes; folder and file names carried observer identifiers and were replaced by codes. "
                 "The per-observer percup.0.pos.mat files (oculometric proportions) and the undocumented .dsac, .purpos and .saccpos "
                 "files in SaccUntrained stay only in the original deposit."),
        how=["One row is one accepted trial of one observer: the right eye's gaze position at 1000 Hz from 100 ms before to about "
             "1 s after the target began to move, in samples/<file>.csv with columns time_s (the authors' trial time), gaze_x, gaze_y and "
             "time_from_pursuit_onset_s (the authors' fourth column, zero at the pursuit onset they determined; the event pursuit_onset "
             "gives that moment in trial time). The files were already filtered by the authors.",
             "experiment gives the paradigm: step-ramp pursuit (the target starts 4 deg to one side and moves back across the screen "
             "centre) or ramp with an initial saccade (the target starts at the centre, so pursuit begins with a saccade); group is "
             "trained (four observers, three highly practiced) or untrained (six naive observers); the vertical component of the "
             "target direction (0, +-2, +-5, +-10 deg from horizontal) comes from the sub-folder name, and in these files "
             "targets run toward positive gaze_x and positive vertical components toward positive gaze_y.",
             "The README gives no unit for gaze x and y: they are taken to be degrees of visual angle because gaze_x of the trained "
             "observers rises by about 20 per second, the 20 deg/s target speed in the paper. Target speeds of individual trials are not "
             "recorded in the deposit (the untrained observers saw 16 and 32 deg/s, the trained 20 deg/s in the main experiment); the "
             "two-digit suffix of each observer folder (06 or 12) is not explained by the README and is kept as observer_folder_suffix. "
             "The deposit contains accepted trials only and no trial times, so trial counts the files of an observer folder "
             "in the order of direction sub-folder and file name, not in time."],
        unit_note="accepted trial (gaze recording)",
        differences=["Per-trial target speeds, the trial order, and the trials the authors excluded (typically 10-20% of pursuit and 5-10% "
                     "of ramp trials per the Methods) are not in the deposit, which holds accepted trials only: the untrained observers have "
                     "519-693 pursuit recordings against the 750 trials (5 blocks of 150) collected, so the exclusion rate cannot be checked."],
    )
    # row order defines subject/person code order
    stdlib.release(DS, MODULE, rows, meta, ck, extra_files=extra)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']} vs {c['reported']}")


if __name__ == "__main__":
    main()
