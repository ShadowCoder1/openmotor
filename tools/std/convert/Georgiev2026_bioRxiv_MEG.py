"""Georgiev, Cabaraux, Munoz, Digileva, Mongold, Wens, Torbati, Moumdjian, Naeije, De Tiege & Bourguignon (2026, bioRxiv
10.64898/2026.07.07.736915): upper-limb movement execution during MEG, EMG and force recordings; one row per recording
(55 recordings of 41 people).

Raw files (OSF 9fgw8, no license stated): MovementExecutionTasksData/MEG*/<DHmovement|IHI|..._t5>*_tsss_mc_ica.fif, about
430-500 MB each. Georgiev2026_bioRxiv_MEG_jobs.py (run first on the PC) downloaded each file in turn, read ONLY its EMG, MISC
and STIM channels with a minimal FIF reader (Georgiev2026_bioRxiv_MEG_fif.py; the MEG channels were never copied out), stored
them in tmp_samples/Georgiev2026_bioRxiv_MEG/*.npz and deleted the raw file. This converter turns those channels into the
OpenMotor timeseries format and finds the cues and movement onsets with the authors' own published method (their
CM_movement_onset_bip_acc.m, re-implemented below in numpy).
Run from the PC's om_std folder: python3 -I tools/std/convert/Georgiev2026_bioRxiv_MEG.py
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Georgiev2026_bioRxiv_MEG"
TMP = Path("tmp_samples") / DS

# ---- the authors' movement-onset method (CM_movement_onset_bip_acc.m, slow_mvt = 1), re-implemented

def cosine_filter(L, kinds, wp, ws):
    """frequency-domain filter of length L; wp = cutoffs and ws = transition widths as fractions of Nyquist, as the authors call it."""
    f = np.abs(np.fft.fftfreq(L))*2  # 0..1 of Nyquist
    g = np.ones(L)
    for k, c, w in zip(kinds, wp, ws):
        lo, hi = c - w/2, c + w/2
        ramp = np.clip((f - lo)/(hi - lo), 0, 1)
        r = 0.5 - 0.5*np.cos(np.pi*ramp)  # 0 -> 1 raised cosine
        g *= r if k == 'high' else (1 - r)
    return g

def bip_onsets(bip, Fs):
    bip = np.abs(bip - np.median(bip))
    L = len(bip); L2 = L//20
    th = np.median(bip[:20*L2].reshape(20, L2).max(1))/3   # column-major reshape([L2 20]) -> 20 consecutive blocks
    pos = []
    bip = bip.copy()
    while (bip > th).any():
        p = int(np.argmax(bip > th))
        p = max(p - round(Fs/10), 0)
        p = int(np.argmax(bip[p:] > th/10)) + p
        pos.append(p)
        bip[:p + round(Fs/2) + 1] = 0
    pos = np.array(pos)
    return pos

def movement_onsets(bip, acc, Fs):
    """port of CM_movement_onset_bip_acc.m, slow_mvt = 1 branch (single accelerometer channel)"""
    bo = bip_onsets(bip, Fs)
    bo = bo[bo >= Fs/2]
    bo = bo[bo <= acc.shape[-1] - Fs*2.5]
    acc = np.atleast_2d(acc)
    F = cosine_filter(acc.shape[1], ['high', 'low'], [0.1/Fs*2, 45/Fs*2], [0.1/Fs*2, 5/Fs*2])
    a = np.real(np.fft.ifft(np.fft.fft(acc, axis=1) * F, axis=1))
    if a.shape[0] > 1:  # the script's multi-channel option: first singular vector across channels
        a = np.linalg.svd(a, full_matrices=False)[2][0]
    else:
        a = a[0]
    ds = 10
    tep = np.arange(0, int(Fs*0.5) + 1, ds)
    nb = len(bo)
    sd = np.zeros((200, nb))
    for j in range(nb):
        for w in range(200):
            idx = (bo[j] - int(Fs*0.5) + w*ds + tep).astype(int)
            idx = idx[(idx >= 0) & (idx < len(a))]
            sd[w, j] = np.std(detrend(a[idx], type='linear'), ddof=1)
    base = sd[:int(Fs*0.15/ds)].mean(0)
    bad = np.zeros(nb, bool)
    for _ in range(20):
        m = ~bad
        th = base[m].mean() + 3*base[m].std(ddof=1)
        bad = base > th
    sdm = sd.mean(0)
    for _ in range(20):
        m = ~bad
        th = sdm[m].mean() + 3*sdm[m].std(ddof=1)
        bad = bad | (sdm > th)
    on = np.full(nb, np.nan)
    th = 3.2
    while np.mean(np.isnan(on[~bad])) > 0.20:
        th -= 0.2
        for j in range(nb):
            k = np.flatnonzero(sd[:, j] - th*base[j] > 0)
            on[j] = k[0] + 1 if len(k) else np.nan   # MATLAB 1-based window index
        if th < 0.3: break
    on = (on - 1)*ds
    rt = np.nanmean(on[~bad])
    return bo, bo + on, bad, rt, th


# ---- build
MEG_CH_NOTE = {"EMG001": "EMG, V", "EMG002": "EMG, V"}
FORCE_CH = ("MISC001", "MISC002", "MISC003")


def kind_of(fname):
    base = fname.split("__", 1)[1]
    if base.startswith("DHmovement"):
        return "ipsilateral", "DHmovement" + re.match(r"DHmovement(\d*)", base).group(1)
    if base.startswith("IHI"):
        return "contralateral", "IHI" + re.match(r"IHI(\d*)", base).group(1)
    m = re.search(r"_t5_(\d+)", base)
    return "contralateral", "t5_" + m.group(1)


def build(files):
    rows, extra, trials = [], [], []
    persons = sorted({f.name.split("__")[0] for f in files})
    rec_count = {}
    stats = []
    for f in sorted(files, key=lambda p: (p.name.split("__")[0], kind_of(p.name)[0], p.name)):
        folder = f.name.split("__")[0]
        z = np.load(f)
        names = [str(x) for x in z["names"]]
        data = z["data"].astype(np.float64)
        fs = float(z["sfreq"])
        n = data.shape[1]
        exp, label = kind_of(f.name)
        pi = persons.index(folder) + 1
        pcode = f"{DS}_P{pi:03d}"
        scode = f"{DS}_P{pi:03d}_{'ipsi' if exp.startswith('ipsi') else 'contra'}"
        rec_count[scode] = rec_count.get(scode, 0) + 1
        rec = rec_count[scode]
        bip = data[names.index("MISC006")]
        acc = np.stack([data[names.index(c)] for c in ("MISC009", "MISC010", "MISC011")])  # three accelerometer axes
        bo, mo, bad_acc, rt, th = movement_onsets(bip, acc, fs)
        badch = data[names.index("bad")] if "bad" in names else np.zeros(n)
        for j in range(len(bo)):
            a, b = (int(mo[j]) - 1000, int(mo[j]) + 2500) if not np.isnan(mo[j]) else (None, None)
            nb = int(badch[max(a, 0):b].sum()) if a is not None else ""
            trials.append(dict(recording=f"{scode}_rec{rec:02d}", trial=j + 1, cue_onset_s=bo[j] / fs,
                               movement_onset_s=(mo[j] / fs) if not np.isnan(mo[j]) else "",
                               reaction_time_ms=(mo[j] - bo[j]) / fs * 1000 if not np.isnan(mo[j]) else "",
                               acc_flagged_by_authors_method=bool(bad_acc[j]), bad_samples_in_epoch=nb))
        fn = f"samples/{scode}_rec{rec:02d}.csv"
        df = pd.DataFrame(data.T, columns=names)
        df.insert(0, "time_s", np.arange(n) / fs)
        extra.append((fn, df.to_csv(index=False, float_format="%.7g", lineterminator="\r\n")))
        ev = ";".join(f"cue={bo[j] / fs:.3f};movement={mo[j] / fs:.3f}" if not np.isnan(mo[j]) else f"cue={bo[j] / fs:.3f}"
                      for j in range(len(bo)))
        units = ["V" if c.startswith("EMG") else ("V (raw transducer output, not calibrated to N)" if c in FORCE_CH
                 else "V (raw channel output)" if c.startswith("MISC") else "code" if c == "STI101" else "0/1 flag") for c in names]
        rows.append(dict(
            dataset=DS, study=DS, experiment=exp, group="", subject=scode, person=pcode, subject_excluded_by_authors=False,
            age="", sex="", handedness="right", population="healthy", setting="lab",
            device="MEG (306 channels) with surface EMG, a force transducer, accelerometer and audio-cue channel",
            session=1, block=label, trial=rec, phase="test", time_s="", task="finger movements", unit="recording",
            condition=exp, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            recording=rec, samples_file=fn,
            signal_kind="surface EMG of hand muscles, hand force, accelerometer, audio cue and artifact flag recorded alongside MEG (MEG not included)",
            channels=";".join(names), channel_units=";".join(units), sample_rate_hz=fs, n_samples=n, duration_s=n / fs,
            events=ev, processing=("none for these channels (the authors' Maxfilter/ICA cleaning changed MEG channels only); "
                                   "EMG scaled to volts with the file's calibration"),
            trial_info=f"n_cues={len(bo)};n_cues_with_movement_onset={int(np.sum(~np.isnan(mo)))};mean_reaction_time_ms={rt:.1f};"
                       f"first_samp={int(z['first_samp'])};n_meg_channels_in_file={int(z['n_meg'])}"))
        stats.append(dict(scode=scode, exp=exp, label=label, n_cues=len(bo), cue_iv=np.median(np.diff(bo)) / fs,
                          rt=rt, fs=fs, n=n, names=names))
    return pd.DataFrame(rows), extra, pd.DataFrame(trials), pd.DataFrame(stats)


def main():
    files = sorted(TMP.glob("*.npz"))
    df, extra, trials, st = build(files)
    ck = stdlib.Checks()
    ck.published("recordings (14 x 2 + 27 = 55)", len(df), 55, 0, "Methods: ipsilateral task in 14 people, contralateral in 41 (14 of them also ipsilateral + 27 new); OSF folder")
    ck.published("people with the ipsilateral task", df[df.experiment.str.startswith("ipsi")].person.nunique(), 14, 0, "Methods, Participants (group 1)")
    ck.published("people with the contralateral task", df[df.experiment.str.startswith("contra")].person.nunique(), 41, 0, "Methods, Participants (group 2)")
    ck.published("people in all", df.person.nunique(), 41, 0, "Methods, Participants (group 2 includes the 14 of group 1)")
    ck.published("MEG sampling rate, Hz", float(df.sample_rate_hz.iloc[0]), 1000, 0, "Methods, MEG recordings (1 kHz)")
    ck.internal("all recordings sampled at the same rate and last 5 min or more", df.sample_rate_hz.nunique() == 1 and df.duration_s.min() > 250,
                f"{df.duration_s.min():.0f}-{df.duration_s.max():.0f} s")
    ck.internal("cue interval median 3-4 s in every recording (paper: cued every 3-4 s)", st.cue_iv.between(3, 4.2).all(),
                f"{st.cue_iv.min():.2f}-{st.cue_iv.max():.2f}")
    ck.internal("mean cue-to-movement-onset time 100-1000 ms in every recording (sanity range)", st.rt.between(100, 1000).all(), f"{st.rt.min():.0f}-{st.rt.max():.0f}")
    ck.internal("cues per recording 60-110", st.n_cues.between(60, 110).all(), f"{st.n_cues.min()}-{st.n_cues.max()}, mean {st.n_cues.mean():.1f}")
    meta = dict(
        study=DS,
        paper=dict(authors="Georgiev C, Cabaraux P, Munoz NY, Digileva D, Mongold SJ, Wens V, Torbati AHM, Moumdjian L, Naeije G, De Tiege X, Bourguignon M",
                   year=2026, title="Assessment of sensorimotor cortical beta oscillations from peripheral electromyography and force recordings",
                   venue="bioRxiv (preprint; no journal version found, also posted at Research Square rs-11372464)", doi="10.64898/2026.07.07.736915"),
        lab="Bourguignon (Universite libre de Bruxelles)", data_url="https://osf.io/9fgw8/",
        license="none stated", license_url="",
        changes=("From each 430-500 MB MEG file only the EMG, MISC (force, accelerometer, audio cue), stimulus and artifact-flag channels were read "
                 "(MEG channels were not copied out of the files) and written at the original 1 kHz to one CSV per recording; cue and movement-onset "
                 "times were found with the authors' published method; participant folders are replaced by codes."),
        how=["One row is one MEG recording of an upper-limb movement task done by a healthy adult: a steady pinch grip held with the right hand while, every 3 to 4 s "
             "(random jitter), an auditory cue asked for a fast movement of the other (ipsilateral task: right arm rotated 5-10 degrees in about 0.5 s; contralateral task: "
             "fast closing and opening of the left fingers), per the paper's Methods. samples/<recording>.csv holds all samples at 1 kHz.",
             "Channels as the authors' scripts use them: EMG001 and EMG002 are surface EMG in volts (calibrated with the file's scale); MISC001, MISC002 or MISC003 is the right-hand "
             "force transducer (which one depends on the recording; the others are other analog inputs), MISC006 the recorded audio cue, MISC009-MISC011 accelerometer axes "
             "(movement onset uses their first singular vector, the script's multi-channel option; the authors' own analysis scripts use MISC011 alone), STI101 the trigger line (it changes in bursts and was not used by the authors for these tasks), 'bad' the authors' "
             "artifact flag per sample. Analog MISC channels are raw volts: the files hold no calibration to newtons or g.",
             "Cue and movement times (events column and trials.csv) are not in the deposit: they come from the authors' published script CM_movement_onset_bip_acc.m "
             "(slow-movement branch), re-implemented here: cues are the onsets of the audio-cue channel above one third of the median block maximum; movement onset is the first "
             "0.5 s window (stepped by 10 ms) where the detrended standard deviation of the 0.1-45 Hz accelerometer exceeds a multiple of its baseline. The script's cosine_filter function is not "
             "in the deposit, so its band-pass is approximated by raised-cosine edges of the stated widths; onsets can differ slightly from the authors'. acc_flagged marks cues "
             "whose accelerometer baseline or window variability the method itself calls outlying; bad_samples_in_epoch counts the authors' bad-sample flags in the -1 to 2.5 s "
             "window around movement onset.",
             "No age or sex per person is shared (only group means in the paper). The authors' 'DHmovement' recordings are the ipsilateral task and 'IHI'/'t5' the contralateral "
             "task (their ReadMe). The 35-person bimanual-contraction experiment has no raw recordings in the deposit (only derived coupling matrices) and is not included."],
        unit_note="recording (EMG, force and accelerometer at 1 kHz during a cued movement task)",
        differences=["The paper reports retained epochs (63 of 85 and 67 of 85 on average) after its own artifact rejection; here every detected cue is listed, so cue counts "
                     "are not the retained-epoch counts."],
    )
    stdlib.release(DS, "timeseries", df, meta, ck, extra_files=extra + [("trials.csv", trials.to_csv(index=False, lineterminator="\r\n"))])
    for c in ck.items:
        print(("  [published] " if c["kind"] == "published" else "  [internal] ") + c["what"][:90],
              (f"{c['computed']:.4g} vs {c['reported']:.4g}" if c["kind"] == "published" else c.get("detail", "")))


if __name__ == "__main__":
    main()
