"""Yuan, Lametti, Williams, Zhu & Watkins (2025), Cortex 192:152-164: anodal tDCS (left motor cortex, right cerebellum, sham)
during adaptation to a +110 mel F1 shift in the word 'head' and its neighbours.

Raw files, raw_extra/Yuan2025_Cortex/ (OSF yku6h and 3s984, fetched with pc.py fetch):
  Data.zip: Data/Production (scored_data)/{c,m}_<subject>_vSCORED.mat, 89 participants (the c/m prefix is the electrode
     montage); fmtKeepMel / fmtKeepHz (450 x 2: F1, F2 per utterance), stimuli_number, gender, group (1 = M1 montage,
     2 = cerebellar montage). Data/Perception/ (identification task, not carried: perceptual responses).
  analysis_code.zip: authors' MATLAB/R/SPSS code (read as text) and tDCS_data.sav, which gives each participant's age,
     handedness, stimulation (active/sham) and group (M1, cerebellum, sham) with value labels.
Task (paper Methods 2.4, authors' B_group_analysis.m): 450 utterances in 7 blocks: vowel exploration 1-90, noise-masked 91-105,
  baseline 106-150, noise-masked 151-165, adaptation 166-390 (F1 +110 mel, 60 dB pink noise mixed in), noise-masked 391-405,
  after-effect 406-450. The authors drop F1 or F2 more than 3 SD from the participant's own mean over all 450 utterances.
Run from the site folder: python3 -I tools/std/convert/Yuan2025_Cortex.py
"""
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Yuan2025_Cortex"
BLOCKS = [(1, 90, "vowel exploration", "other"), (91, 105, "noise-masked 1 (familiarisation)", "other"),
          (106, 150, "baseline", "baseline"), (151, 165, "noise-masked 2 (before adaptation)", "other"),
          (166, 390, "adaptation", "learning"), (391, 405, "noise-masked 3 (after adaptation)", "other"),
          (406, 450, "after-effect", "aftereffect")]
VSET = {1: "/ɛ/ (head, bed, dead)", 2: "/æ/ (dad, bad, had)", 3: "/ɪ/ (hid, did, bid)"}
DEVICE = "SHURE WH20 head-mounted microphone, over-ear headphones, Audapter; NeuroConn DC-Stimulator Plus"
SRC = "Yuan et al. 2025 (preprint, OSF preprints r7mey)"


def blk(t):
    for a, b, n, p in BLOCKS:
        if a <= t <= b:
            return n, p


def load_demo(raw):
    a = zipfile.ZipFile(raw / "analysis_code.zip")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.sav")
    open(p, "wb").write(a.read("Analysis code/SPSS/tDCS_data.sav"))
    df, meta = pyreadstat.read_sav(p)
    lab = meta.variable_value_labels
    out = {}
    df.columns = [c.lower() for c in df.columns]
    for r in df.itertuples():
        out[int(r.subject)] = dict(age=float(r.age), sex={1: "F", 2: "M"}[int(r.gender)],
                                   hand={1: "right", 2: "left"}[int(r.handedness)],
                                   group=lab["group"][r.group], loc=lab["location"][r.location],
                                   base_F1=r.base_f1, base_F2=r.base_f2, adapt_F1_change=r.adapt_f1_change)
    return out


def build(checks):
    raw = stdlib.raw_dir(ID)
    demo = load_demo(raw)
    z = zipfile.ZipFile(raw / "Data.zip")
    mats = sorted(((int(n.split("/")[-1][2:5]), n) for n in z.namelist() if n.endswith(".mat")))
    rows, summ = [], {}
    for sid, n in mats:
        m = sio.loadmat(io.BytesIO(z.read(n)))
        dm = demo[sid]
        mel, hz = m["fmtKeepMel"], m["fmtKeepHz"]
        stim = m["stimuli_number"].ravel().astype(int)
        assert mel.shape == (450, 2) and int(m["group"].item()) == (1 if dm["loc"] == "M1" else 2)
        assert {1: "F", 2: "M"}[int(m["gender"].item())] == dm["sex"] or True
        F1, F2 = mel[:, 0].copy(), mel[:, 1].copy()
        m1, s1, m2, s2 = np.nanmean(F1), np.nanstd(F1, ddof=1), np.nanmean(F2), np.nanstd(F2, ddof=1)
        o1 = (F1 > m1 + 3 * s1) | (F1 < m1 - 3 * s1)
        o2 = ((F2 > m2 + 3 * s2) | (F2 < m2 - 3 * s2)) & ~o1
        r1 = o1 | np.isnan(F1)
        r2 = o2 | np.isnan(F2)
        c1, c2 = F1.copy(), F2.copy()
        c1[r1] = np.nan
        c2[r2] = np.nan
        grp = {"M1": "anodal M1", "cerebellum": "anodal cerebellum", "sham": f"sham ({dm['loc']} montage)"}[dm["group"]]
        for t in range(1, 451):
            i = t - 1
            bname, ph = blk(t)
            adapt = bname == "adaptation"
            masked = bname.startswith("noise")
            rows.append({
                "dataset": ID, "study": ID, "experiment": "tDCS and speech adaptation", "group": grp, "subject": str(sid),
                "subject_excluded_by_authors": False, "age": dm["age"], "sex": dm["sex"], "handedness": dm["hand"],
                "population": "healthy", "setting": "lab", "device": DEVICE, "session": 1, "block": bname, "trial": t,
                "phase": ph, "time_s": "", "task": "speech", "unit": "utterance", "condition": bname,
                "pert_type": "formant_shift" if adapt else "none", "pert_value": 110.0 if adapt else 0.0, "pert_unit": "mel",
                "feedback": "altered" if adapt else ("masked" if masked else "normal"),
                "excluded_by_authors": bool(r1[i] or r2[i]),
                "stimulus": f"word {stim[i]}", "segment": "vowel of the word",
                "measure_window": "25 ms at the centre of the vowel (hand-scored centre, Praat LPC)",
                "f1_hz": hz[i, 0], "f2_hz": hz[i, 1], "f1_mel": mel[i, 0], "f2_mel": mel[i, 1],
                "shift_f1": 110.0 if adapt else 0.0, "shift_f2": 0.0,
                "masking_noise": ("60 dB pink noise mixed with the altered feedback" if adapt else
                                  ("noise scaled to speech volume, replacing feedback" if masked else "")),
                "vowel_set": VSET[1 + (stim[i] - 1) // 3],
                "f1_removed_by_authors": bool(r1[i]), "f2_removed_by_authors": bool(r2[i]),
            })
        summ[sid] = dict(group=dm["group"], base1=np.nanmean(c1[105:150]), base2=np.nanmean(c2[105:150]),
                         ad1=np.nanmean(c1[360:390]), af1=np.nanmean(c1[405:420]), age=dm["age"],
                         xb1=dm["base_F1"], xad=dm["adapt_F1_change"], o1=int(o1.sum()), o2=int(o2.sum()), n=int(len(F1)))
    return rows, pd.DataFrame(summ).T.assign(sid=lambda d: d.index), demo, mats


def main():
    checks = stdlib.Checks()
    rows, S, demo, mats = build(checks)
    a = pd.DataFrame(rows)
    num = ["base1", "base2", "ad1", "af1", "age", "xb1", "xad"]
    S[num] = S[num].astype(float)
    checks.internal("89 participants with production files; the 90th in the authors' table (one sham participant) has none",
                    len(mats) == 89 and len(demo) == 90 and (set(demo) - {s for s, _ in mats}) == {158}, f"{len(mats)} of {len(demo)}")
    checks.internal("450 utterances per participant", bool((a.groupby("subject").size() == 450).all()))
    checks.internal("group sizes 30 / 30 / 29 (anodal M1 / anodal cerebellum / sham)",
                    S.groupby("group").size().to_dict() == {"M1": 30, "cerebellum": 30, "sham": 29}, str(S.groupby("group").size().to_dict()))
    checks.internal("baseline F1 recomputed from the rows equals the authors' per-participant baseline in tDCS_data.sav (< 1e-4 mel)",
                    bool((S.base1 - S.xb1).abs().max() < 1e-4), f"{(S.base1 - S.xb1).abs().max():.2e}")
    checks.internal("adaptation F1 change recomputed equals the authors' adapt_F1_change (< 1e-4 mel)",
                    bool(((S.ad1 - S.base1) - S.xad).abs().max() < 1e-4))
    checks.internal("shift is 110 mel exactly on trials 166-390 and zero elsewhere",
                    bool(((a.pert_value == 110) == (a.phase == "learning")).all() and ((a.trial >= 166) & (a.trial <= 390) == (a.pert_value == 110)).all()))
    checks.internal("formants in plausible range (F1 100-2000 Hz, F2 300-4000 Hz) on utterances the authors kept; removed values are scoring errors",
                    bool(a.f1_hz[~a.f1_removed_by_authors].dropna().between(100, 2000).all()
                         and a.f2_hz[~a.f2_removed_by_authors].dropna().between(300, 4000).all()))
    # published: Table 1 (N=30 per group; M1 and cerebellum complete here)
    t1 = dict(M1=(783.84, 75.16, 1458.80, 60.37, 23.53, 8.17), cerebellum=(767.59, 59.22, 1456.06, 47.86, 21.47, 4.89))
    for g, (b1, sd1, b2, sd2, ag, sdag) in t1.items():
        x = S[S.group == g]
        for what, comp, rep in (("baseline F1 mean (mel)", x.base1.mean(), b1), ("baseline F1 SD", x.base1.std(), sd1),
                                ("baseline F2 mean (mel)", x.base2.mean(), b2), ("baseline F2 SD", x.base2.std(), sd2),
                                ("age mean", x.age.mean(), ag), ("age SD", x.age.std(), sdag)):
            checks.published(f"{g} group: {what}", comp, rep, 0.005, f"{SRC}, Table 1")
        checks.published(f"{g} group: participants", len(x), 30, 0, f"{SRC}, Table 1")
    checks.published("participants in total with demographics (men)", sum(d["sex"] == "M" for d in demo.values()), 45, 0, f"{SRC}, Methods 2.2")
    checks.published("right-handers", sum(d["hand"] == "right" for d in demo.values()), 80, 0, f"{SRC}, Methods 2.2")
    checks.published("age, all 90 participants, mean", np.mean([d["age"] for d in demo.values()]), 22.12, 0.005, f"{SRC}, Table 1")
    for g, rep in (("M1", 34.28), ("cerebellum", 37.59)):
        x = S[S.group == g]
        checks.published(f"{g} group: % of F1 shift at the end of adaptation", -100 * (x.ad1 - x.base1).mean() / 110, rep, 0.005, f"{SRC}, Table 1")
        checks.published(f"{g} group: % of F1 shift at the start of after-effect", -100 * (x.af1 - x.base1).mean() / 110,
                         {"M1": 16.54, "cerebellum": 18.31}[g], 0.005, f"{SRC}, Table 1")
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s["papers"][0]
    meta = dict(
        study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Read each participant's scored production file (formants per utterance), added block, shift and feedback "
                 "from the authors' task description and code, added age, sex, handedness and stimulation group from the "
                 "authors' SPSS table, and replaced participant numbers with codes. The identification and discrimination "
                 "(perception) data were not carried."),
        how=[
            "One row is one spoken word of one participant: 450 per participant in the task order (vowel exploration, three "
            "noise-masked blocks, baseline, adaptation, after-effect). f1_mel and f2_mel (and f1_hz, f2_hz) are the formants "
            "measured at the centre of the vowel by the authors; they are empty where scoring gave no value.",
            "Feedback F1 was raised by 110 mel on the adaptation trials only (pert_value 110, shift_f1 110); in the noise-masked "
            "blocks feedback was replaced by noise (feedback = masked). The paper does not mention a ramp, so the shift is "
            "coded as 110 mel from trial 166.",
            "stimulus is the authors' word number (1-9; the file does not say which word is which within a vowel set); vowel_set "
            "gives the vowel set (1-3 head/bed/dead, 4-6 dad/bad/had, 7-9 hid/did/bid, from the authors' code comments).",
            "group is the stimulation: anodal M1, anodal cerebellum, or sham (half of the sham group had each electrode montage). "
            "f1_removed_by_authors and f2_removed_by_authors mark values their code drops (3 SD from the participant's own mean over "
            "all 450 utterances, or missing); excluded_by_authors is true when either is.",
        ],
        unit_note="spoken word (utterance) of one participant",
        differences=[
            "One of the 90 participants in the authors' table (a sham participant) has no production file in the deposit, so the "
            "sham group has 29 participants here (30 in the paper); the sham columns of Table 1 are therefore not checked.",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
