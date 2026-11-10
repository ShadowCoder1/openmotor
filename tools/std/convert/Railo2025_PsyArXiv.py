"""Railo, Kukkola, Lehtimaki, Seppa & Terho: "Did you hear what you said? Inattentional deafness of pitch shifts in self-produced
vocalization" (PsyArXiv 3m6tx; v1 Nov 2025, v2 Jun 2026; no journal version found in Crossref or Europe PMC on 2026-11-09).
75 analysed adults voiced /a/ for 3 s while counting red flashes of a fixation cross; on one unannounced trial half of them heard
their pitch raised 75 cents for 200 ms (Inattention condition), then did Divided- and Full-attention blocks (20 trials each,
+-75 cents on half) reporting whether they noticed a shift.

Raw files, raw_extra/Railo2025_PsyArXiv/ (OSF b2zud; fetched with tools/std/pc.py fetch):
  InattentionExp_data.mat      data_INFO (MATLAB table, decoded with Railo2025_PsyArXiv_mcos.py, nothing run): ID, block, trial,
                               number_of_targets, number_response, number_accuracy, magnitude_data (semitones; 0.0001 = no shift,
                               +-0.75), first_response_data (-3..3 detection confidence), first_response_accuracy_data,
                               second_response_data (-3..3 felt a change of own pitch), ost_data (always 1, undocumented, not carried),
                               condition (inatten, divided, fullatt); single_trial_VOCdata (4160 x 195, cents re baseline, one
                               row per data_INFO row); time_axis (195 samples, -1000 to 1000 ms re shift onset)
  Collect_and_analyze_v2.m     the authors' script, read as text: excluded IDs [61 63 69 94 99 112]; samples beyond +-75 cents set to
                               NaN; VR1 = samples 118-137 (207-402 ms), VR2 = 137-175 (402-794 ms); critical trial = condition inatten,
                               block 1, trial 6.
Block 0 of Inattention holds the 6 practice trials. The order of the Divided and Full blocks was counterbalanced and is not
recorded in the file, so trial numbers follow the file (Inattention, Divided, Full).
Run from the site folder: python3 -I tools/std/convert/Railo2025_PsyArXiv.py
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Railo2025_PsyArXiv_mcos import MatObjects  # noqa: E402

ID = "Railo2025_PsyArXiv"
EXCLUDE = [61, 63, 69, 94, 99, 112]
SRC = "Railo et al., PsyArXiv 3m6tx v2"
COND = {"inatten": "Inattention", "divided": "Divided attention", "fullatt": "Full attention"}


def nz(x):
    return "" if pd.isna(x) else float(x)


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    f = raw / "InattentionExp_data.mat"
    o = MatObjects(f)
    t = o.variable(o.top["data_INFO"])
    names = [str(v[0]) for v in t["varnames"]]
    d = pd.DataFrame({n: ([x[0] for x in c] if isinstance(c, list) else np.asarray(c).ravel()) for n, c in zip(names, t["data"])})
    m = sio.loadmat(f)
    V, ta = m["single_trial_VOCdata"], m["time_axis"].ravel()
    assert len(d) == V.shape[0] and V.shape[1] == len(ta) == 195
    Vc = np.where(np.abs(V) > 75, np.nan, V)  # the authors' rule
    early = np.nanmean(Vc[:, 117:137], axis=1)
    late = np.nanmean(Vc[:, 136:175], axis=1)
    d["early"], d["late"] = early, late
    d["seq"] = d.groupby("ID").cumcount() + 1
    rows = []
    for i, r in enumerate(d.itertuples(index=False)):
        shifted = abs(r.magnitude_data) > 0.01
        cents = round(r.magnitude_data * 100) if shifted else 0
        asked = r.block == 1
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Inattentional deafness of pitch shifts", "group": "", "subject": f"P{int(r.ID)}",
            "subject_excluded_by_authors": int(r.ID) in EXCLUDE, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": "Audio-Technica AT2035 microphone, Motu MicroBook IIc, Beyerdynamic DT 770 Pro headphones, Audapter",
            "session": 1, "block": "practice" if r.block == 0 else "main", "trial": int(r.seq),
            "phase": "practice" if r.block == 0 else "test", "time_s": "", "task": "speech", "unit": "utterance",
            "condition": COND[r.condition], "pert_type": "pitch_shift" if shifted else "none", "pert_value": float(cents),
            "pert_unit": "cents", "feedback": "altered" if shifted else "masked", "excluded_by_authors": int(r.ID) in EXCLUDE,
            "stimulus": "/a/", "segment": "sustained vowel, 3 s", "measure_window": "",
            "duration_ms": "", "f0_hz": "", "f1_hz": "", "f2_hz": "", "f1_mel": "", "intensity_au": "", "shift_f1": "", "shift_f2": "",
            "masking_noise": "68 dB pink noise", "feedback_latency_ms": "",
            "f2_mel": "", "f1_change_mel": "", "f2_change_mel": "", "compensation_mel": "",
            "detection_rating": nz(r.first_response_data), "detection_correct": "" if pd.isna(r.first_response_accuracy_data) else bool(r.first_response_accuracy_data),
            "felt_change_rating": nz(r.second_response_data), "f0_response_early_cents": nz(r.early), "f0_response_late_cents": nz(r.late),
            "visual_task_targets": int(r.number_of_targets),
            "visual_task_response": "" if pd.isna(r.number_response) else int(r.number_response),
            "visual_task_correct": bool(r.number_accuracy), "f0_track_file": "samples/f0_change_tracks.csv", "f0_track_row": i + 1,
        })
    tr = pd.DataFrame(V, columns=[f"t_{x:.1f}ms" for x in ta])
    tr.insert(0, "row", np.arange(1, len(tr) + 1))
    tracks = tr.to_csv(index=False, lineterminator="\r\n", float_format="%.6g").encode()

    k = ~d.ID.isin(EXCLUDE)
    checks.internal("one track per table row and 52 trials for each of 80 participants",
                    V.shape[0] == len(d) and d.groupby("ID").size().eq(52).all() and d.ID.nunique() == 80)
    checks.internal("time axis is -1000 to 1000 ms in 195 samples", abs(ta[0] + 1000) < 1e-6 and abs(ta[-1] - 1000) < 1e-6)
    checks.published("participants after exclusions", d.ID[k].nunique(), 75, 0, f"{SRC}, Methods, Participants")
    c = d[k & (d.condition == "inatten") & (d.block == 1) & (d.trial == 6)]
    w, n = c[c.magnitude_data == 0.75], c[c.magnitude_data == 0.0001]
    checks.published("participants given the unexpected pitch shift", len(w), 34, 0, f"{SRC}, Results, Inattention condition")
    checks.published("shifted participants who did not notice it (inattentional deafness)", (w.first_response_data < 0).sum(), 17, 0,
                     f"{SRC}, Results, Inattention condition")
    checks.published("participants given no shift", len(n), 41, 0, f"{SRC}, Results, Inattention condition")
    checks.published("no-shift participants who correctly reported nothing unusual", (n.first_response_data < 0).sum(), 36, 0,
                     f"{SRC}, Results, Inattention condition")
    checks.published("% correct rejections", 100 * (n.first_response_data < 0).mean(), 87.8, 0.05, f"{SRC}, Results, Inattention condition")
    checks.published("SD of confidence in the 75 cent trials reported as not noticed", w[w.first_response_data < 0].first_response_data.std(),
                     0.60, 0.005, f"{SRC}, Results, Inattention condition")
    dv = d[k & (d.condition == "divided")].groupby("ID").number_accuracy.mean().mean()
    checks.published("visual task accuracy, Divided attention (%)", 100 * dv, 80.2, 0.05, f"{SRC}, Results, Visual task accuracy")
    out = np.abs(V[k.values]) > 75
    checks.published("% of vocalization samples beyond +-75 cents of baseline (analysed participants)", 100 * out.mean(), 2.6, 0.05,
                     f"{SRC}, Methods, Data preprocessing")
    for c_ in checks.items:
        if not c_["ok"]:
            print("FAILED:", c_)
    meta = dict(
        study=ID,
        paper=dict(authors="Railo H, Kukkola H, Lehtimaki L, Seppa S, Terho H",
                   title="Did you hear what you said? Inattentional deafness of pitch shifts in self-produced vocalization",
                   venue="PsyArXiv preprint (version 2)", year=2026, doi="10.31234/osf.io/3m6tx_v2"),
        lab="Railo (University of Turku)", data_url="https://osf.io/b2zud/", license="none stated", license_url="",
        changes=("Decoded the authors' MATLAB table, added the pitch-shift size in cents, the authors' two vocal-response window means "
                 "(computed from their track as in their script) and a file with the 195-sample pitch track of every trial, and "
                 "replaced participant numbers with codes."),
        how=[
            "One row is one 3 s vocalization of /a/ by one participant, in the file's order (Inattention practice and main trials, then "
            "Divided, then Full attention; the counterbalanced order of the last two is not recorded). Rows of the 6 participants the "
            "authors excluded (hearing problems or too variable pitch) are kept and marked.",
            "pert_value is the pitch shift heard in cents (+75, -75, or 0; the file stores semitones, 0.0001 meaning none), lasting "
            "200 ms and starting 1.0, 1.2 or 1.4 s after voicing began. The unannounced shift is trial 6 of block 'main' in the "
            "Inattention condition.",
            "detection_rating and felt_change_rating are the confidence reports (-3..-1 no, 1..3 yes); samples/f0_change_tracks.csv holds "
            "the pitch change in cents from the pre-shift baseline in 195 samples from -1000 to 1000 ms around the shift, row n belonging "
            "to row n of the main table (f0_track_row), exactly as in the deposit.",
            "f0_response_early_cents and f0_response_late_cents are means of that track over 200-400 ms and 400-800 ms, ignoring samples "
            "beyond +-75 cents as the authors' script does.",
        ],
        unit_note="vocalization of one participant",
        differences=[
            "Not reproduced: the paper's mean confidence of 'not noticed' reports (-2.11 with a shift, -1.85 without) came out as -2.12 and "
            "-1.89 here, and the Inattention visual-task accuracy of 87.5% as 87.6%; the deposit has 80 participants (one participant of "
            "the 82 had technical problems) and no age or sex.",
            "ost_data (constant 1, undocumented) is not carried.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks, extra_files=[("samples/f0_change_tracks.csv", tracks)])


if __name__ == "__main__":
    main()
