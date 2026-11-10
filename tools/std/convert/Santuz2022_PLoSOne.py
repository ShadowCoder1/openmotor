"""Santuz et al. (2022) PLoS ONE 17(6):e0269417: sex-specific muscle-synergy tuning in young and older adults walking
(young, older) and running (young) on a treadmill. 215 participants, 520 trials, two trials of >= 35 gait cycles each.

Raw files (Zenodo 6655815, CC BY 4.0; untrusted data, read as bytes only), raw_extra/Santuz2022_PLoSOne/:
  metadata.dat       tab separated: Code, Group (G1 young walking, G2 older walking, G3 young running), Sex, Locomotion,
                     Speed (m/s), Speed_type (Fixed or Preferred), Age, Height, Mass (one line per participant and locomotion)
  CYCLE_TIMES.RData  R list of 520 data frames (touchdown, stance), one per trial, named CYCLE_TIMES_<ID>_<sex>_<YOUNG|OLD>_<TW|TR>_<trial>;
                     touchdown = time of each touchdown of the recorded (right) leg in s, stance = its stance time in s.
                     Decoded with Santuz2022_PLoSOne_rdata.py (a data-only reader of R's serialization; nothing is evaluated).
Not converted: RAW_EMG.RData and RAW_DATA.RData (raw EMG, ~1.5 GB compressed: it would give several GB of samples, over the 1.9 GB
limit) and muscle_synergies.R (the authors' code; read only as text). Paper read from journals.plos.org.
Run from the site folder: python3 -I tools/std/convert/Santuz2022_PLoSOne.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Santuz2022_PLoSOne_rdata import load  # noqa: E402

ID = "Santuz2022_PLoSOne"
MODULE = "walking"
GRP = {"G1": "young adults, walking", "G2": "older adults, walking", "G3": "young adults, running"}


def main():
    raw = stdlib.raw_dir(ID)
    ck = stdlib.Checks()
    meta_rows = [l.rstrip("\r\n").split("\t") for l in open(raw / "metadata.dat", encoding="utf-8") if l.strip()]
    assert meta_rows[0] == ["Code", "Group", "Sex", "Condition", "Speed", "Speed_type", "Age", "Height", "Mass"], meta_rows[0]
    md = pd.DataFrame(meta_rows[1:], columns=meta_rows[0])
    md[["Speed", "Age", "Height", "Mass"]] = md[["Speed", "Age", "Height", "Mass"]].astype(float)
    ct = load(raw / "CYCLE_TIMES.RData")["CYCLE_TIMES"]
    frames = []
    seen = set()
    for nm, o in zip(ct.names, ct.val):
        m = re.fullmatch(r"CYCLE_TIMES_(ID\d{4})_([MF])_(YOUNG|OLD)_(TW|TR)_(\d\d)", nm)
        assert m, nm
        code, sex, age_g, loc, tri = m.groups()
        f = {n: c.val for n, c in zip(o.names, o.val)}
        td, st = np.asarray(f["touchdown"], float), np.asarray(f["stance"], float)
        row = md[(md.Code == code) & (md.Condition == ("Walking" if loc == "TW" else "Running"))]
        assert len(row) == 1, nm
        r = row.iloc[0]
        assert r.Sex == sex and ((r.Group == "G2") == (age_g == "OLD")) and ((r.Group == "G3") == (loc == "TR")), nm
        seen.add((code, loc))
        n = len(td)
        stride = np.append(np.diff(td), np.nan)
        frames.append(pd.DataFrame(dict(
            dataset=ID, study="Santuz et al. 2022, PLoS ONE", experiment=GRP[r.Group], group=GRP[r.Group],
            subject=f"{code}_{loc}", person=code, subject_excluded_by_authors=False, age=r.Age, sex=sex, handedness="",
            population="healthy older adults" if r.Group == "G2" else "healthy young adults", setting="lab",
            device="treadmill; recorded leg: right", session=1, block=f"trial {tri}", trial=np.arange(1, n + 1), phase="other",
            time_s=td, task="walking", unit="stride", condition=("running, " if loc == "TR" else "walking, ") + r.Speed_type.lower() + " speed",
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            fast_leg="unknown", belt_speed_fast_mps=r.Speed, belt_speed_slow_mps=r.Speed, walking_trial=int(tri),
            stride_start_s=td, stride_time_s=stride, stance_time_right_s=st, walking_context="treadmill", surface_incline_deg=0.0)))
    df = pd.concat(frames, ignore_index=True)
    df["stride_time_s"] = df.stride_time_s.round(6)
    persons = md.drop_duplicates("Code")
    pw = md[md.Group == "G1"]; po = md[md.Group == "G2"]; pr = md[md.Group == "G3"]
    ck.internal("every metadata line has its two trials and every trial a metadata line",
                len(md) == len(seen) and set(zip(md.Code, np.where(md.Condition == "Walking", "TW", "TR"))) == seen, f"{len(md)} participant x locomotion")
    ck.published("total trials in the data set (Zenodo description: 520 trials)", df.groupby(["subject", "walking_trial"]).ngroups, 520, 0, "Zenodo record 6655815, description")
    ck.published("participant x group entries (70 + 70 + 120 = 260; the paper's 215 counts the runners who also walked once, "
                 "and the deposit's codes do not identify them)", df.person.nunique(), 260, 0, "Methods, Participants (G1 70, G2 70, G3 120)")
    ck.published("G1 young walking: participants", len(pw), 70, 0, "Methods, Participants")
    ck.published("G1: women", (pw.Sex == "F").sum(), 35, 0, "Methods, Participants")
    ck.published("G2 older walking: participants", len(po), 70, 0, "Methods, Participants")
    ck.published("G2: women", (po.Sex == "F").sum(), 35, 0, "Methods, Participants")
    ck.published("G3 young running: participants", len(pr), 120, 0, "Methods, Participants")
    ck.published("G3: women", (pr.Sex == "F").sum(), 60, 0, "Methods, Participants")
    ck.published("older adults (65-84 yr): participants", len(po), 70, 0, "Methods, Participants")
    ck.published("youngest young participant (20 yr)", np.floor(md[md.Group != "G2"].Age.min()), 20, 0, "Methods, Participants (20 to 43 yr)")
    ck.published("oldest young participant (43 yr)", np.floor(md[md.Group != "G2"].Age.max()), 43, 0, "Methods, Participants (20 to 43 yr)")
    ck.published("older participants' age range starts at 65", np.floor(po.Age.min()), 65, 0, "Methods, Participants (65 to 84 yr)")
    ck.published("older participants' age range ends at 84", np.floor(po.Age.max()), 84, 0, "Methods, Participants (65 to 84 yr)")
    cyc = df.groupby(["subject", "walking_trial"]).size()
    ck.internal("every trial has at least 35 gait cycles (smallest trial %d)" % cyc.min(), cyc.min() >= 35)
    ck.internal("touchdowns increase within each trial and stance times are positive",
                bool((df.groupby(["subject", "walking_trial"]).stride_start_s.diff().dropna() > 0).all()) and bool((df.stance_time_right_s > 0).all()))
    long_stance = int((df.stance_time_right_s >= df.stride_time_s).sum())
    ck.internal("stance shorter than the stride on all but a handful of cycles (kept as shared)", long_stance < 20, f"{long_stance} cycles")

    meta = dict(
        study=ID,
        paper=dict(authors="Santuz A, Janshen L, Brull L, Munoz-Martel V, Taborri J, Rossi S, Arampatzis A", year=2022,
                   title="Sex-specific tuning of modular muscle activation patterns for locomotion in young and older adults",
                   venue="PLoS ONE 17(6):e0269417", doi="10.1371/journal.pone.0269417"),
        lab="Santuz / Arampatzis (Humboldt-Universitat zu Berlin)", data_url="https://zenodo.org/records/6655815",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Kept every gait cycle of the 520 trials in CYCLE_TIMES.RData (touchdown and stance time of the right leg), in order; "
                 "added age, sex and treadmill speed from metadata.dat; stride_time_s is the difference between consecutive "
                 "touchdowns (empty for the last cycle of a trial); height and body mass are left out; the raw EMG is not included."),
        how=["One row is one gait cycle of the right leg on a treadmill, from one touchdown to the next (stride_start_s and time_s are the "
             "touchdown time in seconds from the start of the trial; stance_time_right_s is the time that foot stayed on the belt). "
             "Each participant has two trials of at least 35 cycles; the paper analysed the first 30 cycles of each, but the deposit's "
             "cycles are all kept.",
             "Group G1 is young adults walking, G2 older adults walking and G3 young adults running (running rows are marked in "
             "condition); The paper says some runners also took part as walkers, but the deposit's participant codes do not show who, so every code is its own person. Speed is the treadmill speed (self-selected "
             "'preferred' or a 'fixed' speed near the average preferred speed), entered in both belt-speed columns because the belts "
             "were tied. Touchdown and lift-off were detected from foot accelerations, a pressure plate or motion capture, as the paper's Methods describe.",
             "The recorded raw EMG (13 right-leg muscles, 1 kHz) is in the deposit's RAW_EMG and RAW_DATA files, which are not converted."],
        unit_note="gait cycle of the right leg (touchdown to touchdown)",
        differences=["The paper counts 215 participants because some took part in two groups; the deposit does not identify them, so this file has 260 person codes (70 + 70 + 120).", "A few cycles (see the check list) have a stance time longer than the next touchdown interval; they are kept as the authors shared them.", "The deposit's cycle-time files hold all cycles (35 to 93 per trial); the paper analysed the first 30 per trial."])
    names = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in names + ["person"] if c not in df.columns})[["person"] + [c for c in names if c != "person"]]
    stdlib.release(ID, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
