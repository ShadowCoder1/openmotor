"""Makino, Kobayashi & Nozaki (2026), eLife reviewed preprint: tonic feedback motor commands predict visuomotor learning.
Zenodo 10.5281/zenodo.18297833. Three experiments (10, 12 and 11 participants) with cursor shifts during reaches in a force
channel. The deposit shares only subject-level, condition-averaged force waveforms (the authors' scripts: raw force, position
and velocity are not shared), so the format is the timeseries module: one row per participant, condition and waveform.

Run from the site folder: python3 -I tools/std/convert/Makino2026_eLife.py   (raw files in raw_extra/Makino2026_eLife/)
Files used (untrusted data, read only): exp1.zip, exp2.zip, exp3.zip (df_exp?_NN.mat, MATLAB v5). The authors' MATLAB scripts
(programs.zip) were read as text, never run. Not converted: exp3/df_exp3_2_NN.mat (v7.3 files with jump-aligned feedback for the
four shift locations, undocumented in the scripts), exp3/group_exp3_fig6.mat (group-level derived data), data_figure8.zip,
figure8.zip, supplement.zip (model fits).

Waveform meaning, from the variable names and comments in the authors' scripts:
  meanper  condition-averaged feedback force (perturbation trials), aligned to movement onset, 1600 samples = -499..1100 ms
  meanpro  condition-averaged learning force (probe trials, cursor invisible), aligned to movement onset, same time base
  meanperj / meanproj  the same, aligned to the cursor-shift (jump) onset, 1400 samples = -499..900 ms
  (meanproj exists in Experiments 2 and 3 only)
Condition order (rows of the arrays; tp_re values 4.. = condition index + 3; tp_re 0 = trial set excluded by the authors):
  Exp 1: 1 no shift; (2,3), (4,5), (6,7), (8,9) = negative, positive shift at 1, 6, 11, 16 cm (difference script: (pos - neg)/2)
  Exp 2: pairs (1,2), (3,4), ... (11,12) = shift sizes 0.4, 0.8, 1.2, 1.6, 2.0, 3.0 cm in the figure order (difference (2k) - (2k-1))
  Exp 3: pairs (1,2) maintained; (3,4) 16 cm removed, (5,6) 16 cm reversed, (7,8) 11 cm removed, (9,10) 11 cm reversed,
         (11,12) 6 cm removed, (13,14) 6 cm reversed (the order of the labels of the authors' Fig 5i bar plot and of Fig 5b-d)
These assignments are inferred from the scripts and the paper's methods, not stated in the readme.
"""
import io
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Makino2026_eLife"
RAW = stdlib.raw_dir(ID)
ck = stdlib.Checks()
CORE = [c[1] for c in stdlib.CORE]
MODULE = "timeseries"
MCOLS = [c[1] for c in stdlib.load_module(MODULE)]
SHIFT_LOC = [1, 6, 11, 16]
SIZES2 = [0.4, 0.8, 1.2, 1.6, 2.0, 3.0]
EXP3 = [("maintained", None), ("removed", 16), ("reversed", 16), ("removed", 11), ("reversed", 11), ("removed", 6), ("reversed", 6)]

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for fn in ("exp1.zip", "exp2.zip", "exp3.zip"):
    ck.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])


def cond_info(exp, idx):
    """(label, pert_value cm, trial_info text) of condition index idx (1-based)."""
    if exp == 1:
        if idx == 1:
            return "no_shift", 0.0, "shift=none"
        loc = SHIFT_LOC[(idx - 2) // 2]
        sign = "positive" if idx % 2 == 1 else "negative"
        return f"shift_{sign}_at_{loc}cm", 3.0, f"shift_cm=3;shift_direction={sign};shift_location_cm={loc}"
    if exp == 2:
        k = (idx - 1) // 2
        member = "second" if idx % 2 == 0 else "first"
        return f"shift_{SIZES2[k]}cm_{member}", SIZES2[k], f"shift_cm={SIZES2[k]};shift_location_cm=1;pair_member={member}"
    lvl = (idx - 1) // 2
    member = "second" if idx % 2 == 0 else "first"
    kind, loc = EXP3[lvl]
    lab = "maintained_3cm" if loc is None else f"{kind}_at_{loc}cm"
    extra = "" if loc is None else f";second_event_location_cm={loc}"
    return f"{lab}_{member}", 3.0, f"shift_cm=3;shift_location_cm=1;level={kind}{extra};pair_member={member}"


def f8(a):
    return "%.8g" % a


rows, extra = [], []
n_part = {}
cycles = {}
ncond = {}
roc_all = []
for exp, nfiles, ncond_exp in ((1, 10, 9), (2, 12, 12), (3, 11, 14)):
    z = zipfile.ZipFile(RAW / f"exp{exp}.zip")
    names = sorted(n for n in z.namelist() if n.endswith(".mat") and "MACOSX" not in n and "_2_" not in n and "group" not in n)
    ck.internal(f"Experiment {exp}: {nfiles} participant files", len(names) == nfiles, str(len(names)))
    n_part[exp] = len(names)
    for fi, n in enumerate(names):
        m = sio.loadmat(io.BytesIO(z.read(n)))
        sid = int(m["subject_id"].ravel()[0])
        ck.internal(f"Exp {exp} file {fi + 1} subject number equals its file number", f"{sid:02d}" in n)
        person = f"{ID}_Exp{exp}_{fi + 1:02d}"
        tp = m["tp_re"].ravel().astype(int)
        counts = Counter(tp.tolist())
        n_excl = counts.get(0, 0)
        ck.internal(f"Exp {exp} file {fi + 1}: condition labels are exactly {4}..{3 + ncond_exp}",
                    sorted(k for k in counts if k) == list(range(4, 4 + ncond_exp)))
        ncond[exp] = ncond_exp
        sig = {"per": m["meanper"], "pro": m["meanpro"]}
        sigj = {"per": m["meanperj"]}
        if "meanproj" in m:
            sigj["pro"] = m["meanproj"]
        ck.internal(f"Exp {exp} file {fi + 1}: array shapes", all(a.shape == (ncond_exp, 1600) for a in sig.values())
                    and all(a.shape == (ncond_exp, 1400) for a in sigj.values()))
        roc = m["onsetROC_ms"].ravel() if "onsetROC_ms" in m else None
        if roc is not None:
            roc_all.append(roc)
        for ci in range(1, ncond_exp + 1):
            n_sets = counts.get(ci + 3, 0)
            cycles.setdefault(exp, []).append(n_sets)
            lab, pv, info = cond_info(exp, ci)
            for align, d, ns, t0 in (("movement", sig, 1600, -499), ("jump", sigj, 1400, -499)):
                for kind in ("per", "pro"):
                    if kind not in d:
                        continue
                    w = d[kind][ci - 1]
                    ch = "feedback_force" if kind == "per" else "learning_force"
                    arc = f"samples/exp{exp}_{fi + 1:02d}_c{ci:02d}_{kind}_{align}.csv"
                    t = (np.arange(ns) + t0) / 1000.0
                    body = ["time_s,%s" % ch] + [f"{f8(a)},{f8(b)}" for a, b in zip(t, w)]
                    extra.append((arc, "\r\n".join(body) + "\r\n"))
                    tinfo = f"condition_index={ci};{info};n_trial_sets={n_sets};sets_excluded_in_sequence={n_excl}"
                    if exp == 1 and ci > 1 and align == "jump" and roc is not None:
                        tinfo += f";roc_onset_after_jump_ms_for_this_shift_location={f8(roc[(ci - 2) // 2])}"
                    perturb = kind == "per"
                    rows.append(dict(
                        dataset=ID, study=ID, experiment=f"Experiment {exp}", group="", subject=person, person=person,
                        subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population="healthy",
                        setting="lab", device="force-channel reaching setup (cursor shifts during the reach); force sampled at 1 kHz",
                        session=1, block=lab, trial=len(rows) + 1, phase="learning", time_s=np.nan, task="reaching",
                        unit="recording", condition=lab,
                        pert_type="cursor_shift" if perturb else "none", pert_value=pv if perturb else 0,
                        pert_unit="cm", feedback="altered" if perturb else "none", excluded_by_authors=False,
                        recording=0, samples_file=arc,
                        signal_kind=("condition-averaged lateral force on perturbation trials (feedback response), "
                                     if perturb else "condition-averaged lateral force on probe trials, cursor invisible (learning response), ")
                        + ("aligned to movement onset" if align == "movement" else "aligned to cursor-shift onset"),
                        channels=ch, channel_units="N", sample_rate_hz=1000, n_samples=ns, duration_s=ns / 1000.0,
                        events="", processing=("average over the participant's trial sets of the condition, aligned as stated; "
                                               "baseline handling not stated in the shared files"),
                        trial_info=tinfo + f";alignment={align}"))
# recording index within participant
cnt = Counter()
for r in rows:
    cnt[r["person"]] += 1
    r["recording"] = cnt[r["person"]]
    r["trial"] = cnt[r["person"]]
for r in rows:
    for c in MCOLS:
        r.setdefault(c, "")
    r["time_s"] = ""

# ----------------------------------------------------------------------------- checks
for exp, n in ((1, 10), (2, 12), (3, 11)):
    ck.published(f"Experiment {exp} participants analysed", n_part[exp], n, 0,
                 "Methods: Experiment 1 (10), Experiment 2 (12), Experiment 3 (12 recruited, one excluded, 11 analysed)")
ck.published("participants analysed in total", sum(n_part.values()), 33, 0, "Results (N = 33)")
for exp, k in ((1, 9), (2, 12), (3, 14)):
    ck.published(f"Experiment {exp} number of cursor-shift condition types", ncond[exp], k, 0, f"Methods, Experiment {exp}")
for exp, k in ((1, 26), (2, 18), (3, 18)):
    ck.published(f"Experiment {exp} trial sets (cycles) per condition, most complete participant", max(cycles[exp]), k, 0,
                 f"Methods, Experiment {exp} ({k} cycles, one trial set per condition per cycle)")
ck.internal("trial-set count per condition never exceeds the planned cycles",
            all(max(cycles[e]) <= k for e, k in ((1, 26), (2, 18), (3, 18))))
roc = np.array(roc_all)
ck.published("Experiment 1 mean ROC onset of the feedback response after cursor shift (ms, all shifts and participants)",
             roc.mean(), 155.9, 1.0, "Results, Experiment 1 (155.9 +/- 13.9 ms)")
ck.internal("Experiment 1 ROC onset across shift locations: 10 participants x 4", roc.shape == (10, 4), str(roc.shape))
ck.internal("every waveform finite", all(np.isfinite(float(x.split(",")[1])) for _, body in extra[:200] for x in body.splitlines()[1:]))
ck.internal("rows = recordings with a samples file each", len(rows) == len(extra) == 10 * 9 * 3 + 12 * 12 * 4 + 11 * 14 * 4,
            f"{len(rows)} rows, {len(extra)} files")

meta = dict(
    paper=dict(authors="Makino Y, Kobayashi T, Nozaki D", year=2026,
               title="Tonic feedback motor commands predict visuomotor learning",
               venue="eLife (reviewed preprint)", doi="10.7554/elife.111537"),
    lab="Nozaki (University of Tokyo)", data_url="https://doi.org/10.5281/zenodo.18297833",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Only the subject-level waveform files of Experiments 1-3 were converted, one row per participant, condition and waveform; "
             "values and time bases kept as shared, the time axis written in seconds. df_exp3_2_*.mat, group_exp3_fig6.mat, the "
             "bootstrap data of Figure 8 and the model-fit files were not converted."),
    how=["One row is one waveform of one participant in one cursor-shift condition: the force on perturbation trials (feedback "
         "response) or on probe trials with the cursor hidden (learning response), each aligned to movement onset or to the "
         "shift onset. The samples (1 kHz, from -499 ms, force in N) are in the CSV named in samples_file.",
         "These are the authors' participant-level averages over their trial sets (five trials each: baseline, perturbation, probe "
         "and two null trials per the paper); no raw trials, hand angles or kinematics were shared, so none are given. "
         "trial_info lists the condition index, the number of trial sets averaged and the number the authors excluded in the "
         "participant's sequence (tp_re = 0).",
         "Condition labels (shift size, location, direction, maintained / removed / reversed) are inferred from the order of the "
         "arrays used in the authors' scripts and from the paper's methods; the shared files do not state the sign of the "
         "lateral axis, so shift direction is given only as positive / negative (Experiment 1, as in the script) or as the "
         "first / second member of a pair (Experiments 2 and 3, where the script subtracts the first from the second)."],
    unit_note="participant, condition and waveform (feedback or learning force, aligned to movement or shift onset)",
    differences=["The Zenodo record is titled 'Visuomotor learning is determined by the tonic component of feedback responses' "
                 "(an earlier title of the preprint); Crossref lists the preprint under the title above, posted 2026-06-29.",
                 "The ROC onset of the paper is computed from waveforms that the authors derive from these data; here the mean of the "
                 "authors' per-participant onset values (onsetROC_ms) is compared with the paper's 155.9 ms."])
stdlib.release(ID, MODULE, rows, meta, ck, extra_files=extra)
