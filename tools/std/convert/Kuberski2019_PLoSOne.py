"""Kuberski & Gafos (2019), PLoS ONE 14(3):e0213851: electromagnetic articulography of repetitive /ka/ and /ta/ at eight metronome rates (30 to 570 bpm),
tongue tip and tongue body sensors, 83.33 Hz.

Raw file, raw_extra/Kuberski2019_PLoSOne/ (Zenodo 2273898): signals.zip with <subject>/<syllable>/<rate>/<trial>/<sensor>/<coord>/occ.mat and seg.mat (MAT-file v7.3
holding hNSignal objects). The files are read as HDF5 without MATLAB: the object's property values (rate, time, data matrix) are the numeric cells of
#subsystem#/MCOS, the movement structure is a plain HDF5 struct. occ.mat (occlusal-plane aligned displacement) is carried; seg.mat (a time-varying PCA
transform of the same data) is not. Nothing from the deposit is run.
Run from the site folder: python3 -I tools/std/convert/Kuberski2019_PLoSOne.py
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kuberski2019_PLoSOne"
P = "Kuberski & Gafos 2019, "
RATES = [30, 90, 150, 210, 300, 390, 480, 570]


def text(f, ref):
    return "".join(chr(c) for c in f[ref][...].ravel())


def idx2t(t, i):
    # README: qonset/qoffset are (1-based, MATLAB) indices into the time vector; fractional values are interpolated
    return float(np.interp(i - 1, np.arange(len(t)), t))


def read(zf, name):
    f = h5py.File(io.BytesIO(zf.read(name)), "r")
    cells = [f[r] for r in f["#subsystem#/MCOS"][0]]
    rate = float(cells[2][...].ravel()[0])
    t = cells[3][...].ravel()
    d = cells[4][...]
    assert d.shape[0] == len(t) and cells[3].shape == (len(t), 1)
    mv = f["movs"]
    gest = []
    for i in range(mv["tags"].shape[0]):
        tg = f[mv["tags"][i, 0]]
        tags = [text(f, r) for r in f[tg[...].ravel()[0]][...].ravel()] if tg.dtype == object and tg.shape == (1, 1) and f[tg[...].ravel()[0]].dtype == object else []
        if not tags:
            tags = [text(f, r) for r in tg[...].ravel()] if tg.dtype == object else []
        if "gest" in tags:
            gest.append((idx2t(t, float(f[mv["qonset"][i, 0]][...].ravel()[0])), idx2t(t, float(f[mv["qoffset"][i, 0]][...].ravel()[0]))))
    return rate, t, d[:, 0], gest


def main():
    raw = stdlib.raw_dir(ID)
    zf = zipfile.ZipFile(raw / "signals.zip")
    recs = {}
    for n in zf.namelist():
        m = re.fullmatch(r"signals/([^/]+)/(ka|ta)/bpm(\d+)/trial(\d+)/(tb|tt)/([xyz])/occ\.mat", n)
        if m:
            sub, syl, bpm, tr, sen, co = m.groups()
            recs.setdefault((sub, syl, int(bpm), int(tr)), {})[f"{sen}_{co}"] = n
    subs = []
    order = sorted(recs, key=lambda k: (k[0], k[1], k[2], k[3]))
    for k in order:
        if k[0] not in subs:
            subs.append(k[0])
    code = {s: f"{ID}_S{i + 1:04d}" for i, s in enumerate(subs)}
    rows, files, nmov = [], [], 0
    rec_no = {}
    for k in order:
        sub, syl, bpm, tr = k
        ch = {}
        ev = []
        rate = None
        tref = None
        for name in ("tt_x", "tt_y", "tt_z", "tb_x", "tb_y", "tb_z"):
            if name not in recs[k]:
                continue
            r, t, x, gest = read(zf, recs[k][name])
            rate = r if rate is None else rate
            assert abs(r - rate) < 1e-9
            s = pd.Series(x, index=np.round(t, 6))
            ch[name] = s[~s.index.duplicated()]
            for a, b in gest:
                ev.append(f"{name}_gest_onset={a:.4f};{name}_gest_offset={b:.4f}")
            nmov += len(gest)
        df = pd.concat(ch, axis=1).sort_index()
        df.insert(0, "time_s", df.index.values)
        rec_no[sub] = rec_no.get(sub, 0) + 1
        fname = f"samples/{code[sub]}_{syl}_bpm{bpm:03d}_trial{tr}.csv"
        files.append((fname, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n").encode()))
        chans = list(ch)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Repetitive speech at eight metronome rates", "group": "", "subject": sub,
            "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy adults", "setting": "lab",
            "device": "electromagnetic articulograph (AG501), tongue tip and tongue body sensors", "session": 1, "block": f"{syl} {bpm} bpm",
            "trial": rec_no[sub], "phase": "test", "time_s": "", "task": "speech", "unit": "recording", "condition": f"/{syl}/ repeated at {bpm} bpm",
            "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
            "recording": rec_no[sub], "samples_file": fname, "signal_kind": "tongue sensor position (occlusal-plane aligned), tt = tongue tip, tb = tongue body",
            "channels": ";".join(chans), "channel_units": ";".join(["mm"] * len(chans)), "sample_rate_hz": rate, "n_samples": len(df),
            "duration_s": float(df.time_s.iloc[-1] - df.time_s.iloc[0]), "events": ";".join(ev),
            "processing": "decimated from 1250 Hz to 83.33 Hz by the authors, aligned to the occlusal plane; displacement column only",
            "trial_info": f"syllable={syl};metronome_bpm={bpm};authors_trial={tr}"})
    out = pd.DataFrame(rows)
    ch = stdlib.Checks()
    ch.published("sampling rate after decimation (Hz)", out.sample_rate_hz.median(), 83.33, 0.005, P + "Data processing (decimated to 83.33 Hz)")
    ch.published("number of metronome rates", out.condition.str.extract(r"at (\d+) bpm")[0].nunique(), 8, 0, P + "Methods (eight metronome rates)")
    ch.published("slowest rate (bpm)", int(out.condition.str.extract(r"at (\d+) bpm")[0].astype(int).min()), 30, 0, P + "Methods (30 bpm)")
    ch.published("fastest rate (bpm)", int(out.condition.str.extract(r"at (\d+) bpm")[0].astype(int).max()), 570, 0, P + "Methods (570 bpm)")
    ch.published("speakers", len(subs), 9, 0, P + "Abstract and Methods (nine adult speakers)")
    ch.internal("rates are exactly the paper's eight", sorted(set(int(x) for x in out.condition.str.extract(r"at (\d+) bpm")[0])) == RATES)
    ch.internal("every recording has at least one channel and 80+ samples", bool((out.n_samples > 80).all() and (out.channels != "").all()))
    ch.internal("recordings", len(out) == len(recs), str(len(out)))
    for c in ch.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {x: p0[x] for x in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Read the authors' MAT-file (HDF5) objects without MATLAB, kept the displacement signal of each sensor and coordinate, joined the channels of one trial on the authors' time vector, and wrote one samples CSV per recording. Movement onsets and offsets (seconds) of the movements the authors tagged as gestures are listed in events. Participant aliases were replaced. Corrected 11-09: the deposit stores qonset and qoffset as 1-based indices into the time vector (its README); the first release wrote the raw index numbers as if they were seconds, now they are converted to seconds on the time vector (fractional indices interpolated).",
        how=["One row is one recording: one subject saying one syllable (/ka/ or /ta/) over and over for one trial at one metronome rate (30 to 570 beats per minute). The samples CSV has time_s and up to six channels: tongue tip (tt) and tongue body (tb) sensor position along x, y, z in millimetres, aligned to the occlusal plane, at 83.33 Hz.",
             "Only the displacement signal is carried; velocity to jounce derivatives stored in the authors' objects and the seg.mat files (time-varying PCA of the same data) are not. Coordinate axes follow the authors' occlusal-plane alignment; their direction was not stated in the README and is not interpreted here.",
             "events lists, per channel, onset and offset times (20% of peak velocity, as the README defines qonset and qoffset) of movements the authors tagged gest, meaning movements contributing to sound production of the target syllable. Trials missing from the deposit are simply absent."],
        unit_note="recording", differences=["The deposit's README lists six subject aliases, but signals.zip holds nine speakers, matching the paper (five German, four English); all nine are included.",
                                            "The paper's power law results were not recomputed; checks cover the stated setup (sampling rate, rates, subject count)."])
    return stdlib.release(ID, "timeseries", out.to_dict("records"), meta, ch, extra_files=files)


if __name__ == "__main__":
    main()
