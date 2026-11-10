"""Narayanan et al. (2014), J Acoust Soc Am 136(3):1307-1311: USC-TIMIT, real-time MRI and electromagnetic articulography (EMA)
database for speech production research. This converter standardizes the EMA part (four speakers reading the 460 MOCHA-TIMIT
sentences); the real-time MRI videos and all audio are not converted.

Raw files, raw_extra/Narayanan2014_JASA/ (Zenodo record 19422914, file USC-TIMIT.zip of 8.5 GB; only the members below were fetched with
HTTP range requests, each checked against the CRC32 in the ZIP's central directory; SOURCES.json lists all of them):
  ema/usctimit_ema_<speaker>_<first>_<last>.mat   MVIEW-format struct array (1x7) with the fields NAME, SRATE, SIGNAL: AUDIO (22,050 Hz)
      and six EMA sensors TT, TB, TD, UL, LL, JAW, each a samples x 3 matrix of positions at about 100.08 Hz; the sensors are listed in
      two different orders (two speakers each), so they are found by NAME; five consecutive sentences per file (92 files per speaker =
      460 sentences); speakers m1, m3, f1, f5 (Zenodo's EMA folders F1, F5, M1, M3)
  ema/usctimit_<ema|mri>_<speaker>_<first>_<last>.trans   the authors' time alignment: start_s,end_s,phone,word,sentence text,sentence
      number. The deposit names the transcripts of speaker m3 usctimit_mri_m3_...: they are byte-identical (CRC32 and size) to the
      transcripts in the MRI part of the ZIP, and this converter shows that they do not fit the audio inside m3's EMA files
  meta/list_of_sentences.txt (the 460 sentences), meta/README.txt (EMA data are MVIEW-compatible)
What the paper says (Sec. III, Table I): NDI Wave Speech Research System; six 5-DOF sensors (three on the tongue midline, lower incisor,
lower lip, upper lip); 100 Hz; missing points estimated by piecewise cubic Hermite interpolation and trajectories smoothed with a 20 Hz
low-pass filter; EMA for four of the ten speakers (ages 29, 26, 23 and 25). The paper, the deposit and the MVIEW readme state neither
the unit nor the axes of the positions; this converter works them out from the data and says so in the README.
Run from the site folder: python3 -I tools/std/convert/Narayanan2014_JASA.py
"""
import csv
import hashlib
import io
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Narayanan2014_JASA"
SRC = "Narayanan et al. 2014, J Acoust Soc Am"
SPEAKERS = [("m1", "M", 29), ("m3", "M", 26), ("f1", "F", 23), ("f5", "F", 25)]  # paper Table I (EMA speakers), in this order
SENSORS = ["TT", "TB", "TD", "UL", "LL", "JAW"]
CHANNELS = [f"{s}_{a}" for s in SENSORS for a in "xyz"]
NONSPEECH = {"sil", "sp"}
N_FILES = 92
DEVICE = "NDI Wave Speech Research System (six sensors, 100 Hz), microphone recorded in parallel"
KIND = ("EMA sensor positions (x, y, z) of the front-most tongue sensor TT, the middle tongue sensor TB, the rear-most tongue sensor "
        "TD, the upper lip UL, the lower lip LL and the lower incisor JAW, while reading sentences; the unit (mm) and the meaning of "
        "x, y and z were worked out from the data, the authors do not state them")
PROC = ("as shared: head movement corrected with reference sensors, missing points filled by piecewise cubic Hermite interpolation "
        "and trajectories smoothed with a 20 Hz low-pass filter (as the paper describes them); the audio channel of the file is not "
        "copied; sensor samples that are still missing in the file are empty cells")
# sha256 over the sorted lines "<name> <CRC32> <size>" of the 92 transcripts of speaker m3 in the MRI part of the ZIP, taken from the ZIP's
# central directory (USC-TIMIT/MRI/Data/M3/trans/); the transcripts in the EMA folder (USC-TIMIT/EMA/Data/M3/trans/) have the same names,
# CRC32s and sizes, and so do the raw files read here
M3_MRI_TRANS_DIGEST = "685890889c7aaf6f2ac1b2080fe4f700af4dade6dab1fe4741eb18e6acaa0021"
FIT_MIN_R = 0.5  # an alignment is kept for a speaker when in at least 90% of the files the speech labels track the audio level this well
FIT_MIN_SHARE = 0.9


def read_mat(path):
    """Sensors are found by NAME (the files of two speakers list them in a different order)."""
    st = sio.loadmat(str(path))[path.stem]
    names = [str(st["NAME"][0, i].ravel()[0]) for i in range(st.shape[1])]
    assert sorted(names) == sorted(["AUDIO"] + SENSORS), (path.name, names)
    ix = {nm: i for i, nm in enumerate(names)}
    rate = {nm: float(np.asarray(st["SRATE"][0, ix[nm]]).ravel()[0]) for nm in names}
    sig = {nm: np.asarray(st["SIGNAL"][0, ix[nm]], dtype=float) for nm in names}
    return tuple(names), rate, sig


def read_trans(path):
    """The authors' lines as text: start, end, phone, word, sentence text (may contain commas), sentence number (empty on pauses)."""
    b = path.read_bytes()
    assert b.isascii(), path.name
    rows = []
    for line in b.decode("ascii").splitlines():
        if not line.strip():
            continue
        p = line.split(",", 4)
        assert len(p) == 5, (path.name, line)
        txt, num = p[4].rsplit(",", 1)
        float(p[0]), float(p[1])
        rows.append((p[0], p[1], p[2], p[3], txt, num))
    return rows


def corr_lag(a, b, max_lag):
    """Pearson r between a and b at lag 0 and its maximum over lags (positive lag: b is later than a), with the best lag in samples."""
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    if a.std() == 0 or b.std() == 0:
        return 0.0, 0.0, 0
    res = {}
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a[: n - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: n + lag])
        m = min(len(aa), len(bb))
        if m >= 50:
            res[lag] = float(np.corrcoef(aa[:m], bb[:m])[0, 1])
    best = max(res, key=res.get)
    return res[0], res[best], best


def audio_level_db(audio, sr, hop_s=0.010):
    h = int(round(sr * hop_s))
    m = len(audio) // h
    return 20 * np.log10(np.sqrt((audio[: m * h].reshape(m, h) ** 2).mean(axis=1) + 1e-12)), h / sr


def interp_gaps(x):
    x = x.copy()
    for c in range(x.shape[1]):
        bad = np.isnan(x[:, c])
        if bad.any() and not bad.all():
            x[bad, c] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad, c])
    return x


def movavg(v, k):
    return np.convolve(v, np.ones(k) / k, mode="same")


raw = stdlib.raw_dir(ID)
ema = raw / "ema"
sent_bytes = (raw / "meta" / "list_of_sentences.txt").read_bytes()
SENTENCES = {}
for ln in sent_bytes.decode("ascii").splitlines():
    m = re.match(r"^(\d{3}) : (.*\S)\s*$", ln)
    assert m, ln
    SENTENCES[int(m.group(1))] = m.group(2)
assert sorted(SENTENCES) == list(range(1, 461))

rows, extra = [], []
S = {}  # per-speaker statistics for the checks and the README
all_rates, low_files, longest_gap = [], [], (0, "")
for si, (spk, sex, age) in enumerate(SPEAKERS, start=1):
    pcode = f"P{si:04d}"
    files = sorted(ema.glob(f"usctimit_ema_{spk}_*.mat"), key=lambda p: int(p.stem.split("_")[-2]))
    assert len(files) == N_FILES, (spk, len(files))
    assert [tuple(int(v) for v in f.stem.split("_")[-2:]) for f in files] == [(1 + 5 * k, 5 + 5 * k) for k in range(N_FILES)], spk
    st = S[spk] = dict(prefixes=set(), orders=set(), nan_vals=0, nan_samples=0, nan_recs=0, fit_r=[], fit_lag=[], sync_lag=[], sync_r=[],
                       ul_tt=[], tt_tb_td=[], ul_ll=[], ul_jaw=[], zr_tongue=[], xyr_tongue=[], tt_td=[], ul_ll_max=[], ul_y=[], hi20=[],
                       hi20_max=[], rate=[], aud_minus_ema=[], end_after=[], speech_sent=set(),
                       span_after_signal=0, neg_rows=0, neg_first=0, trans_empty=0, gaps=[], spill_files=0, spill_before=0, recs=[], psd=np.zeros(51), n_psd=0,
                       overlap=[])
    for ri, f in enumerate(files, start=1):
        a, b = f.stem.split("_")[-2:]
        a_i, b_i = int(a), int(b)
        cands = [ema / f"usctimit_{k}_{spk}_{a}_{b}.trans" for k in ("ema", "mri")]
        tp = [p for p in cands if p.exists()]
        assert len(tp) == 1, (f.name, tp)
        st["prefixes"].add(tp[0].name.split("_")[1])
        names, rate_d, sig = read_mat(f)
        st["orders"].add(names)
        file_sensors = [nm for nm in names if nm != "AUDIO"]
        rates = {rate_d[s] for s in SENSORS}
        assert len(rates) == 1 and rate_d["AUDIO"] == 22050.0, (f.name, rate_d)
        rate = rates.pop()
        ps = [sig[s] for s in SENSORS]
        n = ps[0].shape[0]
        assert all(p.shape == (n, 3) for p in ps), f.name
        audio = sig["AUDIO"].ravel()
        x = np.hstack(ps)  # columns TT_x TT_y TT_z TB_x ... JAW_z
        assert not np.isinf(x).any() and not np.isnan(audio).any(), f.name
        nanm = np.isnan(x).reshape(n, 6, 3)
        assert (nanm.all(axis=2) == nanm.any(axis=2)).all(), f.name  # a sample is missing for all three coordinates or for none
        miss = nanm[:, :, 0].sum(axis=0)  # missing samples per sensor
        n_nan = int(nanm.sum())
        st["nan_vals"] += n_nan
        st["nan_samples"] += int(miss.sum())
        st["nan_recs"] += int(n_nan > 0)
        for si_, s in enumerate(SENSORS):
            col = nanm[:, si_, 0]
            if col.any():
                runs = np.split(np.flatnonzero(col), np.flatnonzero(np.diff(np.flatnonzero(col)) > 1) + 1)
                mr = max(len(r) for r in runs)
                if mr > longest_gap[0]:
                    longest_gap = (mr, f"{mr} samples ({mr / rate:.2f} s) of sensor {s} in recording {ri} of speaker {ID}_{pcode}")
        st["rate"].append(rate)
        st["aud_minus_ema"].append(len(audio) / 22050.0 - n / rate)
        if not 99.9 < rate < 100.1:
            low_files.append((n, len(audio) / 22050.0))
        m_ = x.reshape(n, 6, 3)
        mean = np.nanmean(m_, axis=0)  # 6 x 3
        rng = np.nanmax(m_, axis=0) - np.nanmin(m_, axis=0)
        st["ul_tt"].append(bool(mean[3, 0] > mean[0, 0]))
        st["tt_tb_td"].append(bool(mean[0, 0] > mean[1, 0] > mean[2, 0]))
        st["ul_ll"].append(bool(mean[3, 1] > mean[4, 1]))
        st["ul_jaw"].append(bool(mean[3, 1] > mean[5, 1]))
        st["ul_y"].append(float(mean[3, 1]))
        st["zr_tongue"].append(float(rng[:3, 2].mean()))
        st["xyr_tongue"].append(float(rng[:3, :2].mean()))
        st["tt_td"].append(float(np.nanmedian(np.linalg.norm(ps[0] - ps[2], axis=1))))
        st["ul_ll_max"].append(float(np.nanmax(np.linalg.norm(ps[3] - ps[4], axis=1))))
        # share of the signal's power above 20 Hz (the paper: trajectories smoothed with a 20 Hz low-pass filter); gaps linearly filled for this test
        xi = interp_gaps(x)
        win = np.hanning(n)
        spec = np.abs(np.fft.rfft((xi - xi.mean(axis=0)) * win[:, None], axis=0)) ** 2 / (win ** 2).sum()
        fr = np.fft.rfftfreq(n, d=1.0 / rate)
        tot = np.maximum(spec[fr > 0].sum(axis=0), 1e-30)
        hi = spec[fr > 20].sum(axis=0) / tot
        st["hi20"].append(float(np.median(hi)))
        st["hi20_max"].append(float(hi.max()))
        grid = np.arange(0, 51.0)
        st["psd"] += np.array([np.interp(grid, fr, spec[:, c]) for c in range(spec.shape[1])]).mean(axis=0)
        st["n_psd"] += 1
        # is the EMA in step with the audio of the same file? level of the audio against the speed of the articulators
        v = np.linalg.norm(np.diff(xi.reshape(n, 6, 3), axis=0) * rate, axis=2)
        speed = movavg(v[:, [0, 1, 2, 4, 5]].mean(axis=1), 5)
        env_db0, hop = audio_level_db(audio, 22050.0)
        env_lin = movavg(10 ** (env_db0 / 20), 5)
        t_env, t_ema = (np.arange(len(env_lin)) + 0.5) * hop, (np.arange(n - 1) + 0.5) / rate
        la = np.log(np.interp(t_ema, t_env, env_lin) + 1e-6)
        lb = np.log(speed + 1e-3)
        _, r_best, lag_best = corr_lag(la, lb, int(round(0.6 * rate)))
        st["sync_lag"].append(lag_best / rate)
        st["sync_r"].append(r_best)
        # the authors' time alignment
        tr = read_trans(tp[0])
        tc = (np.arange(len(env_db0)) + 0.5) * hop
        ind = np.zeros(len(env_db0))
        for t0, t1, ph, wd, tx, nm in tr:
            if ph not in NONSPEECH and float(t1) > float(t0):
                ind[(tc >= float(t0)) & (tc < float(t1))] = 1.0
        r0, _, lagfit = corr_lag(env_db0, ind, int(round(3.0 / hop)))
        st["fit_r"].append(r0)
        st["fit_lag"].append(lagfit * hop)
        # events from the alignment: start of the first and end of the last speech sound of each sentence of the file (pauses left out;
        # intervals with end <= start are artefacts of how the authors cut their alignment into files and are left out)
        span = {}
        for t0, t1, ph, wd, tx, nm in tr:
            if nm.strip() and ph not in NONSPEECH and float(t1) > float(t0):
                k = int(nm)
                lo, hi_ = span.get(k, [1e9, -1e9])
                span[k] = [min(lo, float(t0)), max(hi_, float(t1))]
            if float(t1) < float(t0):
                st["neg_rows"] += 1
        st["neg_first"] += int(float(tr[0][1]) < float(tr[0][0]))
        st["trans_empty"] += int(not any(t[5].strip() for t in tr))
        for r1, r2 in zip(tr[:-1], tr[1:]):
            if float(r1[1]) >= float(r1[0]) and float(r2[1]) >= float(r2[0]):
                st["gaps"].append(float(r2[0]) - float(r1[1]))
        nums = {int(t[5]) for t in tr if t[5].strip()}
        if spk != "m3":
            # the files are chunks of one long recording: a sentence cut by the end of a file continues in the next one
            assert set(range(a_i, b_i + 1)) <= nums <= set(range(a_i - 1, b_i + 2)), (f.name, sorted(nums))
            st["spill_files"] += int((b_i + 1) in nums)
            st["spill_before"] += int((a_i - 1) in nums)
            assert set(range(a_i, b_i + 1)) <= set(span) <= set(range(a_i - 1, b_i + 2)), (f.name, sorted(span))
            for k in range(a_i, b_i + 1):
                st["speech_sent"].add(k)
            for txt_, num_ in sorted({(t[4], t[5]) for t in tr if t[5].strip() and 1 <= int(t[5]) <= 460}):
                words, listed = set(re.findall(r"[a-z0-9']+", txt_.lower())), set(re.findall(r"[a-z0-9']+", SENTENCES[int(num_)].lower()))
                st["overlap"].append(len(words & listed) / len(listed))
            st["span_after_signal"] += sum(1 for k in range(a_i, b_i + 1) if span[k][1] > n / rate + 0.05)
        st["end_after"].append(max(float(t[1]) for t in tr) - n / rate)
        # sample file
        t = np.arange(n) / rate
        buf = io.StringIO()
        buf.write("time_s," + ",".join(CHANNELS) + "\r\n")
        np.savetxt(buf, np.column_stack([t, x]), fmt="%.8g", delimiter=",", newline="\r\n")
        txt = buf.getvalue().replace("nan", "")
        assert "inf" not in txt
        arc = f"samples/{pcode}_rec{ri:03d}.csv"
        extra.append((arc, txt.encode("ascii")))
        proc = PROC
        if n_nan:
            proc += "; in this file " + ", ".join(f"{s} has {int(miss[i])} empty samples" for i, s in enumerate(SENSORS) if miss[i]) + f" of {n}"
        st["recs"].append(dict(ri=ri, a=a, b=b, n=n, rate=rate, arc=arc, proc=proc, tr=tr, span=span, pcode=pcode))
    # keep the alignment of a speaker only when it fits the audio inside the speaker's own EMA files
    st["fit_share"] = float(np.mean(np.array(st["fit_r"]) >= FIT_MIN_R))
    st["use_align"] = st["fit_share"] >= FIT_MIN_SHARE
    for r in st["recs"]:
        events = ""
        if st["use_align"]:
            events = ";".join(f"sentence_{k}_{w}={r['span'][k][i]:.8g}" for k in range(int(r["a"]), int(r["b"]) + 1)
                              for i, w in enumerate(("start", "end")))
            al = io.StringIO()
            cw = csv.writer(al, lineterminator="\r\n")
            cw.writerow(["start_s", "end_s", "phone", "word", "sentence_text", "sentence_number"])
            for tt in r["tr"]:
                cw.writerow(tt)
            extra.append((f"alignments/{r['pcode']}_rec{r['ri']:03d}.csv", al.getvalue().encode("ascii")))
        rows.append(dict(
            dataset=ID, study=SRC, experiment="", group="", subject=spk, person=spk, subject_excluded_by_authors=False, age=age, sex=sex,
            handedness=None, population="not stated (adult native speakers of General American English)", setting="lab", device=DEVICE,
            session=1, block=f"sentences {r['a']}-{r['b']}", trial=r["ri"], phase="other", time_s=None, task="speech", unit="recording",
            condition="", pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            recording=r["ri"], samples_file=r["arc"], signal_kind=KIND, channels=";".join(CHANNELS), channel_units=";".join(["mm"] * 18),
            sample_rate_hz=r["rate"], n_samples=r["n"], duration_s=r["n"] / r["rate"], events=events, processing=r["proc"]))
    all_rates += st["rate"]
extra.append(("sentences.txt", sent_bytes))
df = pd.DataFrame(rows)

# ---- numbers for the checks and the README ------------------------------------------------------------------------------------
rates = np.array(all_rates)
normal = (rates > 99.9) & (rates < 100.1)
low = rates[~normal]
nan_vals = sum(s["nan_vals"] for s in S.values())
nan_samples = sum(s["nan_samples"] for s in S.values())
nan_recs = sum(s["nan_recs"] for s in S.values())
aligned = [k for k, s in S.items() if s["use_align"]]
unaligned = [k for k, s in S.items() if not s["use_align"]]
assert unaligned == ["m3"] and S["m3"]["prefixes"] == {"mri"} and all(S[k]["prefixes"] == {"ema"} for k in aligned), (aligned, unaligned)
code = {spk: f"{ID}_P{i:04d}" for i, (spk, _, _) in enumerate(SPEAKERS, start=1)}
fit_med = {k: float(np.median(S[k]["fit_r"])) for k in S}
end_after = np.concatenate([S[k]["end_after"] for k in aligned])
end_after_n = int((end_after > 0.05).sum())
aud_gap = np.concatenate([S[k]["aud_minus_ema"] for k in S])
same_len = [k for k in S if np.abs(S[k]["aud_minus_ema"]).max() < 1e-3]
other_gap = np.concatenate([S[k]["aud_minus_ema"] for k in S if k not in same_len])
sync_ok = {k: float(np.mean(np.abs(S[k]["sync_lag"]) <= 0.3)) for k in S}
up_y = [float(np.mean(S[k]["ul_y"])) for k in S]
tt_td = {k: float(np.median(S[k]["tt_td"])) for k in S}
ll_max = {k: float(np.median(S[k]["ul_ll_max"])) for k in S}
hi20_typ = float(np.median(np.concatenate([S[k]["hi20"] for k in S])))
hi20_max = float(max(max(S[k]["hi20_max"]) for k in S))
psd_db = {k: 10 * np.log10(S[k]["psd"] / S[k]["n_psd"]) for k in S}
slope = lambda d, lo, hi: np.polyfit(np.arange(lo, hi + 1), d[lo:hi + 1], 1)[0]  # dB per Hz
slope_lo = {k: slope(psd_db[k], 10, 20) for k in S}
slope_hi = {k: slope(psd_db[k], 22, 32) for k in S}
n_neg = sum(S[k]["neg_rows"] for k in aligned)
n_neg_first = sum(S[k]["neg_first"] for k in aligned)
n_span_after = sum(S[k]["span_after_signal"] for k in aligned)
gaps_all = np.array([g for k in aligned for g in S[k]["gaps"]])
overlap_all = np.array([o for k in aligned for o in S[k]["overlap"]])
n_spill = sum(S[k]["spill_files"] for k in aligned)
n_spill_b = sum(S[k]["spill_before"] for k in aligned)
m3code = code["m3"]
m3_lines = sorted(f"{p.name} {zlib.crc32(p.read_bytes()) & 0xFFFFFFFF:08x} {p.stat().st_size}" for p in ema.glob("usctimit_mri_m3_*.trans"))
m3_digest_ok = len(m3_lines) == N_FILES and hashlib.sha256("\n".join(m3_lines).encode()).hexdigest() == M3_MRI_TRANS_DIGEST
m3_lag = np.array(S["m3"]["fit_lag"])
low_short = all(n / 100.0 < dur - 0.2 for n, dur in low_files)  # the low-rate files hold fewer samples than 100 Hz x the audio length
n_empty_m3 = S["m3"]["trans_empty"]

# ---- checks -------------------------------------------------------------------------------------------------------------------
ck = stdlib.Checks()
ck.published("speakers with EMA data", df["person"].nunique(), 4, 0, "Table I (EMA column)")
ck.published("male speakers with EMA data", df.loc[df["sex"] == "M", "person"].nunique(), 2, 0, "Table I")
ck.published("female speakers with EMA data", df.loc[df["sex"] == "F", "person"].nunique(), 2, 0, "Table I")
ck.published("EMA sensors per recording", len(file_sensors), 6, 0, "Sec. III (six 5-DOF sensors)")
ck.published("EMA sensors on the tongue midline (TT, TB, TD)", sum(nm in ("TT", "TB", "TD") for nm in file_sensors), 3, 0,
             "Sec. III (three sensors on the midline of the tongue)")
ck.published("EMA sensors on lower incisor, lower lip and upper lip (JAW, LL, UL)", sum(nm in ("JAW", "LL", "UL") for nm in file_sensors), 3, 0,
             "Sec. III (the other three sensors)")
ck.published("EMA sampling rate (Hz), median over the files' own rates", float(np.median(rates)), 100, 0.1, "Sec. III (100 Hz)")
cover = [sum(int(b) - int(a) + 1 for a, b in (r.block.split()[-1].split("-") for r in df[df["person"] == k].itertuples())) for k in
         [s for s, _, _ in SPEAKERS]]
ck.published("sentences read per speaker (sum over the speaker's files, smallest)", min(cover), 460, 0, "Sec. IV (the 460 sentences of MOCHA-TIMIT)")
ck.published("sentences read per speaker (sum over the speaker's files, largest)", max(cover), 460, 0, "Sec. IV")
ck.published("sentences with speech labelled in the alignment of each speaker whose alignment is kept (smallest)",
             min(len(S[k]["speech_sent"]) for k in aligned), 460, 0, "Sec. IV (460 sentences)")
ck.published("median share of a sensor signal's power above 20 Hz, over all recordings (20 Hz low-pass filter)", hi20_typ, 0, 0.001,
             "Sec. III (trajectories smoothed with a 20 Hz low-pass filter)")
ck.internal("92 files of 5 sentences for each of the 4 speakers, covering sentences 1-460 once", len(df) == 368 and
            (df.groupby("person").size() == N_FILES).all() and cover == [460] * 4, f"{len(df)} recordings")
orders = {k: sorted(S[k]["orders"]) for k in S}
ck.internal("sensors found by name in every file, all six with the same number of samples and 3 columns (asserted); each speaker's 92 files "
            "list them in one order, and the order differs between speakers", all(len(v) == 1 for v in orders.values()) and
            len({v[0] for v in orders.values()}) == 2, "; ".join(f"{k}: {' '.join(v[0][1:])}" for k, v in orders.items()))
ck.internal("missing sensor samples always lack all three coordinates (asserted); counted and left as empty cells",
            True, f"{nan_samples} sensor samples ({nan_vals} values) in {nan_recs} of 368 files")
ck.internal("axes: upper lip ahead of the tongue tip and tongue tip ahead of tongue body ahead of rear tongue on x, in 99% of the files "
            "of every speaker", all(np.mean(S[k]["ul_tt"]) >= 0.99 and np.mean(S[k]["tt_tb_td"]) >= 0.99 for k in S),
            "; ".join(f"{k}: {np.mean(S[k]['ul_tt']):.2f}, {np.mean(S[k]['tt_tb_td']):.2f}" for k in S))
ck.internal("axes: upper lip above lower lip and above lower incisor on y, in 99% of the files of every speaker",
            all(np.mean(S[k]["ul_ll"]) >= 0.99 and np.mean(S[k]["ul_jaw"]) >= 0.99 for k in S),
            "; ".join(f"{k}: {np.mean(S[k]['ul_ll']):.2f}, {np.mean(S[k]['ul_jaw']):.2f}" for k in S))
ck.internal("axes: the tongue sensors move less along z than along x and y (mean range on z below 0.6 of that on x and y) for every speaker",
            all(np.mean(S[k]["zr_tongue"]) < 0.6 * np.mean(S[k]["xyr_tongue"]) for k in S),
            "; ".join(f"{k}: {np.mean(S[k]['zr_tongue']):.1f} vs {np.mean(S[k]['xyr_tongue']):.1f}" for k in S))
ck.internal("unit: median distance between the tongue-tip and rear-tongue sensors is 15-60 units for every speaker (1.5-6 cm in mm), and the "
            "median of the largest lip distance per file is 20-60 units",
            all(15 <= v <= 60 for v in tt_td.values()) and all(20 <= v <= 60 for v in ll_max.values()),
            "; ".join(f"{k}: {tt_td[k]:.1f}, {ll_max[k]:.1f}" for k in S))
ck.internal("20 Hz filter: the average power spectrum falls at least 1.3 times as steeply (dB per Hz) between 22 and 32 Hz as between 10 "
            "and 20 Hz, for every speaker", all(slope_hi[k] <= 1.3 * slope_lo[k] for k in S),
            "; ".join(f"{k}: {slope_lo[k]:.1f} then {slope_hi[k]:.1f} dB/Hz" for k in S))
ck.internal("EMA and audio of the same file are in step: the audio level and the speed of the articulators correlate best within 0.3 s in "
            "at least 95% of the files of every speaker", all(v >= 0.95 for v in sync_ok.values()),
            "; ".join(f"{k}: {v:.2f}" for k, v in sync_ok.items()))
ck.internal(f"alignments: in at least {FIT_MIN_SHARE:.0%} of the files the speech labels track the audio level of the file (r >= {FIT_MIN_R}) "
            f"for the speakers whose alignment is kept", all(S[k]["fit_share"] >= FIT_MIN_SHARE for k in aligned),
            "; ".join(f"{k}: share {S[k]['fit_share']:.2f}, median r {fit_med[k]:.2f}" for k in S))
ck.internal("alignments: the text the authors give for a sentence number shares at least half of its words with that numbered sentence of the "
            "list (what was said, including slips, against what was asked) in every file, so the numbers point at the right sentences",
            overlap_all.min() >= 0.5, f"{len(overlap_all)} (file, sentence) pairs, lowest overlap {overlap_all.min():.2f}, "
            f"median {np.median(overlap_all):.2f}, {np.mean(overlap_all == 1):.0%} identical word sets")
ck.internal("the 92 transcripts named usctimit_mri_m3_ in the EMA folder have the names, CRC32s and sizes of the MRI part's transcripts of that "
            "speaker (digest taken from the ZIP's central directory)", m3_digest_ok, M3_MRI_TRANS_DIGEST[:16] + "...")
ck.internal("the alignment left out (transcripts named usctimit_mri_) does not fit its audio: share of files with r >= 0.5 below 50%",
            S["m3"]["fit_share"] < 0.5, f"share {S['m3']['fit_share']:.2f}, median r {fit_med['m3']:.2f}, median best shift {np.median(m3_lag):.2f} s")
ck.internal("every file has a sample file; every kept alignment has an alignment file",
            len([a for a, _ in extra if a.startswith("samples/")]) == len(df) and
            len([a for a, _ in extra if a.startswith("alignments/")]) == N_FILES * len(aligned))
print("recordings", len(df), "| NaN: sensor samples", nan_samples, "values", nan_vals, "files", nan_recs, "longest gap", longest_gap,
      "| rates: normal", int(normal.sum()), "low", np.round(np.sort(low), 2).tolist(), "range of normal", round(rates[normal].min(), 4), round(rates[normal].max(), 4),
      "| audio minus EMA s: %.3f..%.3f" % (aud_gap.min(), aud_gap.max()))
for k in S:
    s = S[k]
    print(k, "prefix", sorted(s["prefixes"]), "| fit r median %.2f share>=0.5 %.2f | best shift median %.2f range %.2f..%.2f | sync within 0.3 s %.2f | use_align %s" % (
        fit_med[k], s["fit_share"], np.median(s["fit_lag"]), min(s["fit_lag"]), max(s["fit_lag"]), sync_ok[k], s["use_align"]),
          "| files with next sentence started", s["spill_files"], "| sentence spans ending after the signal", s["span_after_signal"], "| neg-length rows", s["neg_rows"])
print("alignment ends after the signal (+0.05 s) in", end_after_n, "of", len(end_after), "files; by %.2f..%.2f s, median %.2f" % (
    end_after[end_after > 0.05].min(), end_after.max(), np.median(end_after[end_after > 0.05])), "| upper-lip y averages", np.round(up_y, 1).tolist(),
      "| tongue span", {k: round(v, 1) for k, v in tt_td.items()}, "| lip max", {k: round(v, 1) for k, v in ll_max.items()},
      "| hi20 median %.5f max %.4f" % (hi20_typ, hi20_max), "| slopes dB/Hz", {k: (round(slope_lo[k], 2), round(slope_hi[k], 2)) for k in S}, "| gaps >0.3 s:", int((gaps_all > 0.3).sum()), "max %.2f" % gaps_all.max(), "| overlap min %.2f" % overlap_all.min(),
      "| total sample bytes", sum(len(b) for a, b in extra) // 1_000_000, "MB")
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)

nan_by = ", ".join("%d in %d files for speaker %s" % (S[k]["nan_samples"], S[k]["nan_recs"], code[k]) for k in S if S[k]["nan_samples"])
meta = dict(
    paper=dict(authors="Narayanan S, Toutios A, Ramanarayanan V, Lammert A, Kim J, Lee S, Nayak K, Kim YC, Zhu Y, Goldstein L, Byrd D, "
                       "Bresch E, Ghosh P, Katsamanis A, Proctor M", year=2014,
               title="Real-time magnetic resonance imaging and electromagnetic articulography database for speech production research (TC)",
               venue="J Acoust Soc Am 136(3):1307-1311", doi="10.1121/1.4890284"),
    lab="Narayanan (USC SPAN / SAIL)", data_url="https://zenodo.org/records/19422914", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Only the EMA part of the deposit (4 of its 10 speakers) was converted: each EMA file became one row and its six sensor "
             "trajectories one CSV (samples/); for three speakers the authors' time alignment became one CSV per file (alignments/). "
             "The audio channel inside the files, the separate WAV files and all real-time MRI data were not converted. Sample times "
             "are the sample index divided by the file's own sampling rate; values are copied to 8 significant digits; the sensors are "
             "written in one order (the files use two); sensor samples missing in the files are empty cells; speaker labels are "
             "replaced by codes."),
    how=["One row is one EMA file of the deposit: one speaker reading five consecutive sentences of the 460-sentence MOCHA-TIMIT set "
         "(92 files per speaker; block gives the sentence numbers and sentences.txt lists the sentences). samples_file holds the 18 "
         "position channels (x, y, z of the front-most tongue sensor TT, the middle tongue sensor TB, the rear-most tongue sensor TD, "
         "the upper lip UL, the lower lip LL and the lower incisor JAW) at the file's own sampling rate, in this order for every "
         "speaker. x, y and z are the first, second and third column of each sensor's SIGNAL matrix in the file. session is 1 for every row: "
         "the deposit does not say whether a speaker's 92 files were recorded in one sitting.",
         "The unit and the axes are ASSUMED: the authors, the deposit and the paper do not state them. Positions are labelled mm because "
         f"the distances between the sensors are those of a human mouth only in millimetres (in the files' own units the median distance "
         f"between the front-most and the rear-most tongue sensor is {min(tt_td.values()):.0f}-{max(tt_td.values()):.0f} and the median "
         f"of the largest distance between the upper and the lower lip sensor in a file is {min(ll_max.values()):.0f}-"
         f"{max(ll_max.values()):.0f}; in centimetres the tongue would be metres long, in tenths of a millimetre the open mouth a few "
         "millimetres wide). In every "
         "speaker x runs front-back with larger values towards the lips (lips ahead of the tongue tip, tongue tip ahead of tongue "
         "body ahead of rear tongue), y runs vertically with larger values upwards (upper lip above lower lip and lower incisor), and "
         "z is the lateral axis (the tongue sensors move least along it); its sign is not known. The origin of the coordinates is not "
         f"the same for every speaker (the upper-lip sensor's y averages between {min(up_y):.0f} and {max(up_y):.0f} across speakers), so "
         "compare movements within a speaker rather than absolute positions across speakers.",
         "events gives, from the authors' time alignment (alignments/), the start of the first and the end of the last speech sound "
         "(phone label other than sil and sp) of each of the five sentences of the file, in seconds; pauses the authors attach to a "
         "sentence and intervals whose end is not after their start " + ("(all of them the first line of a file) " if n_neg == n_neg_first else "") +
         "are left out. The alignment files have all the authors' lines. The files are consecutive pieces: a sentence can begin at the "
         f"end of one file and continue at the start of the next, so {n_spill} of the {N_FILES * len(aligned)} alignments also label the first "
         f"sounds of the next file's first sentence (the number one above the file's last sentence) and {n_spill_b} the last sounds of "
         "the previous file's last sentence (the number one below the file's first sentence); events does not list these.",
         f"For speaker {m3code} there is no alignment and events is empty: the deposit's transcripts for this speaker do not belong to "
         "the EMA recordings (see Known differences).",
         "The paper says the EMA data were head-corrected, gap-filled by piecewise cubic Hermite interpolation and smoothed with a 20 Hz "
         "low-pass filter; none of this was repeated here. The first EMA sample is taken as time 0, the same as the first audio "
         "sample the file holds; in "
         f"{'every file' if min(sync_ok.values()) == 1 else 'at least ' + format(min(sync_ok.values()), '.0%') + ' of the files'} of every "
         "speaker the audio level and the speed of the articulators correlate best at an offset below 0.3 s."],
    unit_note="EMA file (five consecutive sentences read by one speaker)",
    differences=[
        "ASSUMED, not stated by the authors, the deposit or the paper: (1) the unit of the positions is mm; (2) x runs front-back (larger "
        "towards the lips), y runs vertically (larger upwards) and z is the lateral axis. They were worked out from the data (sensor "
        "distances; the order of the sensors along x and y; the small movement along z), as described under How this file was made. "
        "The sign of z and the origin of each speaker's coordinates (which differs between speakers) are not known.",
        f"The paper says missing sensor points were filled by piecewise cubic Hermite interpolation, but {nan_samples} sensor samples "
        f"({nan_vals} values, always all three coordinates of one sensor) are still missing in {nan_recs} of the 368 files "
        f"({nan_by}; the longest gap is {longest_gap[1]}). They are empty cells here, not filled, and processing names them for each file.",
        f"The paper gives a sampling rate of 100 Hz; the files carry a rate of their own: {rates[normal].min():.4f}-{rates[normal].max():.4f} Hz "
        f"in {int(normal.sum())} files and {low.min():.1f}-{low.max():.1f} Hz in {len(low)} files"
        + (" (those hold fewer samples than 100 Hz times the length of their audio)" if low_short else "") +
        ". time_s is the sample index divided by the file's own rate.",
        (f"For speaker {code[same_len[0]]} the EMA signal and the audio of a file have the same length (to 0.1 ms); for the other speakers "
         f"the EMA signal is {other_gap.min():.2f}-{other_gap.max():.2f} s shorter than the audio (median {np.median(other_gap):.2f} s). "
         if len(same_len) == 1 else
         f"The EMA signal is up to {aud_gap.max():.2f} s shorter than the audio of the same file (median {np.median(aud_gap):.2f} s). ") +
        "The audio inside the files is sampled at 22,050 Hz, where the paper says 44.1 kHz downsampled to 16 kHz; it is not part of this "
        "conversion.",
        f"The deposit's transcripts for speaker {m3code} are named usctimit_mri_..., not usctimit_ema_..., and are byte-identical (CRC32 "
        "and size, checked against the ZIP's file directory) to the transcripts of the MRI recordings of that speaker"
        + (f"; {n_empty_m3} of them has no sentence label at all" if n_empty_m3 else "") + ". They do not fit the audio inside this speaker's EMA files: the correlation between the audio level "
        f"and the speech labels is {fit_med['m3']:.2f} (median over files) against {min(fit_med[k] for k in aligned):.2f}-"
        f"{max(fit_med[k] for k in aligned):.2f} for the three other speakers; the labels fit best when shifted by {np.median(m3_lag):.1f} s "
        f"(median over files; from {m3_lag.min():.1f} to {m3_lag.max():.1f} s, positive meaning the label lies later than the sound), so "
        "no single correction repairs them. The EMA and the audio of these files are in step with each other (see How this file was made), so "
        "it is the transcripts that do not belong to them. They are therefore not copied: no alignments and no events for this speaker.",
        f"In {end_after_n} of the {len(end_after)} alignments that are kept the last interval ends {end_after[end_after > 0.05].min():.1f}-"
        f"{end_after.max():.1f} s after the end of the EMA signal (median {np.median(end_after[end_after > 0.05]):.1f} s), and "
        f"{n_neg} lines ({'all of them the first line of a file' if n_neg == n_neg_first else str(n_neg_first) + ' of them the first line of a file'}) "
        f"have an end before their start. {int((gaps_all > 0.3).sum())} gaps of more than 0.3 s (longest {gaps_all.max():.1f} s) separate consecutive "
        "lines. The alignment times are copied as the authors give them.",
        *([f"In {n_span_after} sentence{'s' if n_span_after > 1 else ''} the end given in events lies after the end of the EMA signal of the file "
           "(the alignment is longer than the signal there)."] if n_span_after else []),
        f"The deposit has two further transcripts and WAV files for speaker {code['f5']} (sentences 461-470) with no EMA file; they are not "
        "converted.",
        f"The 20 Hz low-pass filter shows in the data as a steeper roll-off above 20 Hz, not as an absence of power: the median share of a "
        f"sensor signal's power above 20 Hz is {hi20_typ:.3%}, and up to {hi20_max:.1%} in single channels.",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)
