"""Priplata, Niemi, Harry, Lipsitz & Collins (2003) Lancet 362:1123-1124: postural sway of 15 young and 12 elderly adults standing with
eyes closed on vibrating gel insoles, with (STIM) and without (NULL) subsensory noise. PhysioNet 'Noise Enhancement of Sensorimotor
Function' v1.0.0 (nesfdb), ODC-By 1.0. Raw: raw_extra/Priplata2003_Lancet/nesfdb.tar.gz (read in place; two-column text files).
Run from the site folder: python3 -I tools/std/convert/Priplata2003_Lancet.py"""
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Priplata2003_Lancet"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
GROUP = {"yng": "young adults", "eh": "healthy elderly"}
t = tarfile.open(raw / "nesfdb.tar.gz")
recs = []
for m in t.getmembers():
    if not m.isfile():
        continue
    p = m.name.split("/")
    assert len(p) == 4 and p[2] in ("NULL", "STIM"), m.name
    n = int(re.search(r"(\d\d)\.txt$", p[3]).group(1))
    recs.append((p[0], p[1], p[2], n, m))
# subject order: young first? keep the archive's order of groups as (eh, yng) sorted by id, trials by file number
recs.sort(key=lambda r: (r[0] != "yng", r[1], r[3]))
subs = []
for r in recs:
    if r[1] not in subs:
        subs.append(r[1])
rows, files = [], []
per = {}
rec_no = {}
for g, sid, cond, n, m in recs:
    a = np.loadtxt(t.extractfile(m))
    assert a.shape == (1800, 2) and np.isfinite(a).all()
    rec_no[sid] = rec_no.get(sid, 0) + 1
    k = rec_no[sid]
    si = subs.index(sid) + 1
    fn = f"samples/P{si:02d}_trial{k:02d}.csv"
    stim = cond == "STIM"
    ml, ap = a[:, 0], a[:, 1]
    r = np.hypot(ml - ml.mean(), ap - ap.mean())
    per.setdefault((g, sid, cond), []).append((np.ptp(ml), np.ptp(ap), r.mean()))
    rows.append(dict(
        dataset=DS, study="Priplata 2003 Lancet", experiment="", group=GROUP[g], subject=sid, subject_excluded_by_authors=False,
        population="healthy" if g == "yng" else "healthy older adults", setting="lab",
        device="Vicon motion capture, reflective marker on the right shoulder; gel insoles with 3 vibrating tactors per foot",
        session=1, block="", trial=k, phase="test", task="balance and posture", unit="recording",
        condition="noise (subsensory insole vibration)" if stim else "control (no vibration)",
        pert_type="subsensory_insole_vibration" if stim else "none", pert_value=0.9 if stim else 0.0,
        pert_unit="fraction of the participant's sensory threshold" if stim else "",
        feedback="normal", excluded_by_authors=False, recording=k, samples_file=fn,
        signal_kind="shoulder-marker displacement during quiet standing, eyes closed (stabilogram)",
        channels="mediolateral;anteroposterior", channel_units="deposit units;deposit units", sample_rate_hz=60, n_samples=1800,
        duration_s=30.0, events="", processing=f"as shared (marker displacement; see the how note); trial {n} in the participant's sequence",
        trial_info=f"file_position_in_authors_sequence={n}"))
    files.append((fn, a))
import pandas as pd  # noqa: E402
df = pd.DataFrame(rows).reindex(columns=[c[1] for c in stdlib.columns_for("timeseries") if c[1] != "person"])


def extra():
    for fn, a in files:
        yield fn, "time_s,mediolateral,anteroposterior\n" + "".join(f"{i / 60:.6f},{x[0]:.8g},{x[1]:.8g}\n" for i, x in enumerate(a))


# ---- checks
def gmean(g, cond, j):
    return float(np.mean([np.mean([v[j] for v in per[(gg, s, c)]]) for (gg, s, c) in per if gg == g and c == cond]))


ng = {g: len({s for (gg, s, c) in per if gg == g}) for g in GROUP}
ck.published("young participants", ng["yng"], 15, 0, "Letter, para 'We included 15 young ...'")
ck.published("elderly participants", ng["eh"], 12, 0, "Letter, para 'We included 15 young ...'")
ck.published("participants in total", len(subs), 27, 0, "Abstract")
tr = {}
for r in recs:
    tr.setdefault((r[0], r[1]), []).append(r[3])
ck.published("trials per young participant (10 noise + 10 control)", np.mean([len(v) for (g, s), v in tr.items() if g == "yng"]), 20, 0, "Methods")
ck.published("trials per elderly participant (5 noise + 5 control)", np.mean([len(v) for (g, s), v in tr.items() if g == "eh"]), 10, 0, "Methods")
ck.published("noise trials per young participant", np.mean([len(per[(g, s, c)]) for (g, s, c) in per if g == "yng" and c == "STIM"]), 10, 0, "Methods")
ck.published("noise trials per elderly participant", np.mean([len(per[(g, s, c)]) for (g, s, c) in per if g == "eh" and c == "STIM"]), 5, 0, "Methods")
ck.published("trial duration, s", df.duration_s.astype(float).mean(), 30, 0, "Methods (30 s stance trials)")
# the paper divides lengths by marker height (values ~0.7 of the deposit's), so compare scale-free noise/control ratios (Table)
for g, lab, pm in (("yng", "young", 0.0), ("eh", "elderly", 0.0)):
    pass
for g, lab, j, name, num, den in (("eh", "elderly", 0, "mediolateral range", 13.4, 15.7), ("yng", "young", 1, "anteroposterior range", 18.4, 19.9),
                                  ("yng", "young", 2, "mean radius", 4.8, 5.1), ("eh", "elderly", 1, "anteroposterior range", 21.4, 22.5),
                                  ("eh", "elderly", 2, "mean radius", 5.0, 5.5), ("yng", "young", 0, "mediolateral range", 13.7, 13.5)):
    ck.published(f"{lab} {name}, noise / control ratio", gmean(g, "STIM", j) / gmean(g, "NULL", j), num / den, 0.045, "Table (group means)")
ck.published("elderly mediolateral range falls with noise (deposit units, mm)", gmean("eh", "NULL", 0) - gmean("eh", "STIM", 0), 3.1, 0.6,
             "Table (15.7 to 13.4 in height-normalised units; about 3 mm at the deposit's scale)")
ck.internal("one row per file of the archive", len(df) == len(recs) == 420, f"{len(df)} recordings")
ck.internal("all recordings have 1800 samples at 60 Hz = 30 s", bool((df.n_samples == 1800).all()))

meta = dict(
    paper=dict(authors="Priplata AA, Niemi JB, Harry JD, Lipsitz LA, Collins JJ", year=2003,
               title="Vibrating insoles and balance control in elderly people", venue="The Lancet 362(9390):1123-1124",
               doi="10.1016/s0140-6736(03)14470-4"),
    lab="Collins (Boston University)", data_url="https://physionet.org/content/nesfdb/1.0.0/",
    license="Open Data Commons Attribution License v1.0 (ODC-By 1.0)", license_url="https://opendatacommons.org/licenses/by/1-0/",
    changes=("Kept all 420 trials (two-column text files) of 27 participants as one CSV of 1800 samples per trial, with a time column "
             "added (60 Hz) and participants coded. Nothing was filtered or rescaled."),
    how=["One row is one 30 s quiet-standing trial (eyes closed, feet 8 cm apart, on gel insoles). The samples file has time_s, "
         "mediolateral and anteroposterior displacement of a reflective marker on the right shoulder, recorded with Vicon at 60 Hz, "
         "columns 1 and 2 of the authors' text files.",
         "Noise trials (condition 'noise') had subsensory white-noise vibration (low-pass 100 Hz) under both feet at 90% of the "
         "participant's sensory threshold (pert_value 0.9); control trials had none. The two trial types were randomised in pairs; "
         "trial is the order of the trial in the authors' file numbering within the participant. Young participants did 10 + 10 "
         "trials, elderly 5 + 5.",
         "The PhysioNet page says the displacements are normalised by marker height, but the deposit's values are about 1.4 times the "
         "normalised values of the paper's table (consistent with millimetres of raw shoulder displacement divided by a marker "
         "height of roughly 1.4 m for the paper's table); the unit is therefore given as 'deposit units' and the published checks use "
         "noise/control ratios, which do not depend on scale. Age, sex and height per participant were not shared."],
    unit_note="30 s standing trial (samples in samples/)",
    differences=["Table values in the paper are divided by marker height and cannot be reproduced in absolute terms because heights were not shared; ratios of noise to control means were reproduced instead."],
)
stdlib.release(DS, "timeseries", df, meta, ck, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
