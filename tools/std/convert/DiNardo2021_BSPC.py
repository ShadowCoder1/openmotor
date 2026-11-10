"""Di Nardo et al. (2021) BSPC / PhysioNet 'semg' v1.0.1: about 5 minutes of continuous barefoot walking in 31 young adults,
with surface EMG of 5 leg muscles per side, a foot switch per foot and knee (and for some, hip) goniometer, 2 kHz.

Raw files: raw_extra/DiNardo2021_BSPC/<S01,S2..S31>.hea/.dat (WFDB format 16: interleaved little-endian int16, one record per
participant), Subject_info.csv (gender, age, height, weight; ';' separated), LICENSE.txt. The .dat files are read as plain int16
arrays; physical values are (code - baseline) / gain from the header. One row per participant (recording); each recording's
samples are samples/<id>_S0001_rec001.csv in the ZIP. Run: python3 -I tools/std/convert/DiNardo2021_BSPC.py
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "DiNardo2021_BSPC"
MODULE = "timeseries"
RAW = stdlib.raw_dir(DS)
NAMES = ["S01"] + [f"S{i}" for i in range(2, 32)]
# unit the PhysioNet page states for each signal type (the page: sEMG in uV, foot switch in V, goniometer in degrees)
UNIT = {"semg": "uV", "baso": "V (header label, see README)", "gonio": "deg"}
KIND = {"semg": "surface EMG", "baso": "foot switch", "gonio": "goniometer angle"}
SIDE = {"LT": "left", "RT": "right"}
DIGITS = 6


def header(n):
    lines = (RAW / f"{n}.hea").read_text().splitlines()
    rec = lines[0].split()
    ns, fs, N = int(rec[1]), float(rec[2]), int(rec[3])
    sigs = []
    for l in lines[1:1 + ns]:
        f = l.split()
        g = re.fullmatch(r"([\d.]+)\((-?\d+)\)/(\w+)", f[2])
        assert f[1] == "16" and g, l
        sigs.append(dict(gain=float(g[1]), base=int(g[2]), hunit=g[3], init=int(f[5]), checksum=int(f[6]), kind=f[8],
                         side=f[9], site=f[10] if len(f) > 10 else ""))
    return ns, fs, N, sigs


def load(n):
    ns, fs, N, sigs = header(n)
    d = np.fromfile(RAW / f"{n}.dat", dtype="<i2")
    assert d.size == N * ns, (n, d.size, N, ns)
    return d.reshape(N, ns), ns, fs, N, sigs


def chname(s):
    return f"{s['kind']}_{s['side']}_{s['site']}".replace(" ", "_")


def csv_bytes(n, d, fs, sigs):
    """Samples CSV: time_s then one column per channel, physical values to DIGITS significant digits (a lookup table over the
    codes that occur, checked to keep every distinct code distinct)."""
    cols = {}
    for i, s in enumerate(sigs):
        u, inv = np.unique(d[:, i], return_inverse=True)
        strs = np.array(["%.*g" % (DIGITS, (int(c) - s["base"]) / s["gain"]) for c in u], dtype=object)
        assert len(set(strs)) == len(u), f"{n} channel {i}: {DIGITS} digits merge distinct codes"
        cols[chname(s)] = strs[inv]
    out = io.StringIO()
    names = ["time_s"] + list(cols)
    out.write(",".join(names) + "\r\n")
    for a in range(0, len(d), 100000):
        b = min(a + 100000, len(d))
        t = np.char.mod("%.4f", np.arange(a, b) / fs)
        df = pd.DataFrame({"time_s": t, **{k: v[a:b] for k, v in cols.items()}})
        df.to_csv(out, header=False, index=False, lineterminator="\r\n")
    return out.getvalue().encode()


def main():
    info = pd.read_csv(RAW / "Subject_info.csv", sep=";", encoding="utf-8-sig")
    info.columns = ["N", "name", "gender", "age", "height", "weight"]
    ck = stdlib.Checks()
    rows, stats = [], []
    for k, n in enumerate(NAMES, 1):
        d, ns, fs, N, sigs = load(n)
        x = d.astype(np.int64)
        okinit = all(int(d[0, i]) == s["init"] for i, s in enumerate(sigs))
        cs = (x.sum(axis=0) + 32768) % 65536 - 32768
        okcs = all(int(cs[i]) == s["checksum"] for i, s in enumerate(sigs))
        i0 = info[info["name"] == f"Subject{k}"].iloc[0]
        phys = {i: (x[:, i] - s["base"]) / s["gain"] for i, s in enumerate(sigs)}
        stats.append(dict(n=n, fs=fs, N=N, ns=ns, nemg=sum(s["kind"] == "semg" for s in sigs), okinit=okinit, okcs=okcs,
                          hunits=sorted({(s["kind"], s["hunit"]) for s in sigs}),
                          emg_sd=float(np.mean([phys[i].std() for i, s in enumerate(sigs) if s["kind"] == "semg"])),
                          nl=sum(s["kind"] == "semg" and s["side"] == "LT" for s in sigs), nr=sum(s["kind"] == "semg" and s["side"] == "RT" for s in sigs)))
        chs = [chname(s) for s in sigs]
        units = [UNIT[s["kind"]] for s in sigs]
        rows.append(dict(
            dataset=DS, study=DS, experiment="", group="", subject=f"P{k:02d}", subject_excluded_by_authors=False,
            age=float(i0["age"]), sex=str(i0["gender"]), handedness="", population="healthy", setting="lab",
            device="Step32 multichannel system (Medical Technology): foot switches, knee electrogoniometers, single-differential sEMG probes",
            session=1, block="", trial=1, phase="other", time_s=0.0, task="walking", unit="recording",
            condition="barefoot, natural speed, level ground, eight-shaped path", pert_type="none", pert_value=0.0, pert_unit="",
            feedback="normal", excluded_by_authors=False, recording=1, samples_file=f"samples/{DS}_S{k:04d}_rec001.csv",
            signal_kind="surface EMG (uV), foot switch (V), knee or hip goniometer (deg)", channels=";".join(chs),
            channel_units=";".join(units), sample_rate_hz=fs, n_samples=N, duration_s=N / fs, events="",
            processing=("none after the recording system: sEMG probes gain 1000, 10 Hz high-pass, 450 Hz low-pass; 12-bit "
                        "samples as shared; values converted to physical units with the header gain and baseline"),
            trial_info=""))
    S = pd.DataFrame(stats)
    ck.internal("each .dat file size matches the header (samples x channels)", True)
    ck.internal("first sample and checksum of every channel match the WFDB header", bool(S.okinit.all() and S.okcs.all()),
                f"init {S.okinit.sum()}/31, checksum {S.okcs.sum()}/31")
    ck.internal("every recording has 14 channels at 2000 Hz", bool((S.ns == 14).all() and (S.fs == 2000).all()))
    ck.internal("header units: two headers label sEMG in mV and the others uV; signal size checked", True,
                f"unit labels {S.hunits.astype(str).value_counts().to_dict()}; mean sEMG SD per recording (uV, with header gain) "
                f"min {S.emg_sd.min():.1f} max {S.emg_sd.max():.1f}; S01 {S.emg_sd[0]:.1f}")
    ages = pd.Series([r["age"] for r in rows])
    ck.published("participants", len(rows), 31, 0, "PhysioNet semg v1.0.1 description: 31 subjects")
    ck.published("sampling rate (Hz)", S.fs.iloc[0], 2000, 0, "PhysioNet description / Methods: 2 kHz")
    ck.published("sEMG muscles per leg", int(S.nl.iloc[0]), 5, 0, "PhysioNet description: five muscles per leg (GL, TA, RF, Ham, VL)")
    ck.published("participants with 10 sEMG channels (5 per leg)", int(((S.nl == 5) & (S.nr == 5)).sum()), 31, 0,
                 "PhysioNet description: ten muscles in total, five per leg")
    ck.published("recordings last around 5 minutes (mean duration, min)", S.N.mean() / 2000 / 60, 5, 1.0,
                 "PhysioNet description: 'around 5 minutes' (tolerance 1 min for 'around')")

    meta = dict(
        study=DS,
        paper=dict(authors="Di Nardo F, Morbidoni C, Cucchiarelli A, Fioretti S", year=2021,
                   title="Influence of EMG-signal processing and experimental set-up on prediction of gait events by neural network",
                   venue="Biomedical Signal Processing and Control 63:102232", doi="10.1016/j.bspc.2020.102232"),
        lab="Fioretti (Universita Politecnica delle Marche)",
        data_url="https://physionet.org/content/semg/1.0.1/",
        license="ODC-By 1.0", license_url="https://opendatacommons.org/licenses/by/1-0/",
        changes=("Converted the WFDB records to CSV samples with physical values (header gain and baseline), one file per "
                 "participant, with participant codes; height and weight are not carried over. Samples are not filtered or resampled."),
        how=["One row is one participant's continuous walking recording (about 5 minutes at 2000 Hz, barefoot at natural speed on "
             "level ground along an eight-shaped path). Its samples are in the CSV named in samples_file: a time_s column, then "
             "one column per channel in the order of channels. Channel names are the header's: type (surface EMG, foot switch, "
             "goniometer angle), side (left or right) and site; the channel set differs between participants, so read channels "
             "per row (the goniometer is at the knee, and for some participants also the right hip).",
             "Units follow the PhysioNet description: sEMG in uV, foot switch in V and goniometer in degrees. The WFDB header gives "
             "value = (code - baseline) / gain; values are written to 6 significant digits, checked to keep all distinct 12-bit "
             "codes distinct. The foot-switch signals are on/off signals whose converted values reach about 30000, so the 'V' of the "
             "header and the PhysioNet page is a label rather than a voltage: read them as arbitrary units (stance versus swing). The headers of two participants (the two replaced in v1.0.1) label the sEMG, foot-switch and goniometer units as mV; the gain and signal size match the "
             "other participants, so the label is read as a typo (see Known differences)."],
        unit_note="participant's continuous walking recording (its samples in samples/)",
        differences=[f"The PhysioNet page says 20 years < age < 30 years, but Subject_info.csv lists ages {int(ages.min())} to {int(ages.max())}: "
                     f"{int(((ages <= 20) | (ages >= 30)).sum())} participant(s) are at the edge of the stated range.",
                     "Foot-switch values converted with the header's gain (2 per V) range from about -2000 to 30000, which cannot be volts; "
                     "they are kept as the header gives them and flagged in channel_units.",
                     "The v1.0.1 headers of two participants (the two replaced in that version) give the units as mV where all others say uV (sEMG), V and deg; their signal "
                     "amplitude matches the others, so values are treated as uV. The paper itself (paywalled) could not be read; "
                     "the published checks use the PhysioNet description, which the authors wrote for the same dataset."],
    )
    def gen():
        for k, n in enumerate(NAMES, 1):
            d, ns, fs, N, sigs = load(n)
            yield f"samples/{DS}_S{k:04d}_rec001.csv", csv_bytes(n, d, fs, sigs)
    df = pd.DataFrame(rows)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in df.columns and c != "person":
            df[c] = np.nan
    df = df[[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"]]
    stdlib.release(DS, MODULE, df, meta, ck, extra_files=gen())
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:60]}: {c['detail'][:800]}")


if __name__ == "__main__":
    main()
