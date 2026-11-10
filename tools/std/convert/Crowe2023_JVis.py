"""Crowe, Smeets & Brenner (2023) J Vis: finger taps on a target attached to a rolling disk (disk visible or invisible),
51 participants, 200 trials each in the deposit.

Raw file: raw_extra/Crowe2023_JVis/data.zip (read in place with zipfile; one folder per participant, per trial a .txt with the
authors' summary values and a .opt with the 500 Hz marker samples; the samples are used only to classify trials that have no
scored tap, with the authors' own rule from analyse_wheel.py, and are not released). Nothing in the zip is executed.

Run from the site folder: python3 -I tools/std/convert/Crowe2023_JVis.py"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Crowe2023_JVis"
MODULE = "screen_interception"
DEVICE = ("index-finger marker tracked by Optotrak 3020 (500 Hz); targets back-projected at 120 Hz (800 x 600 px) on a 1.25 x 1.0 m "
          "screen tilted back by 30 deg; participants stood and tapped the screen")


def opt_class(txt):
    """The authors' rule (analyse_wheel.py, summarize_performance) for trials with no 'target' line in the .txt."""
    up = False
    for line in txt.splitlines():
        w = line.split()
        if not w:
            continue
        if w[-1] == "m":
            return "missing"
        if not up and float(w[-1]) > 10:
            up = True
        elif up and float(w[-1]) < 5:
            return "soft"
    return "nothing"


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "data.zip")
    names = {n for n in z.namelist() if not n.startswith("__MACOSX")}
    txts = sorted((n for n in names if n.endswith(".txt")),
                  key=lambda n: (int(re.search(r"Participant(\d+)_", n).group(1)), int(re.search(r"_(\d+)\.txt$", n).group(1))))
    rows = []
    for n in txts:
        who, trial = re.search(r"Participant(\d+)_(\d+)\.txt$", n).groups()
        d = {}
        for l in z.read(n).decode("latin1").splitlines():
            if ":" in l:
                k, v = l.split(":", 1)
                d.setdefault(k.strip(), v.strip())
        tap = d["tap"].split() if "tap" in d else [None, None]
        tt = d["target at tap"].split() if "target at tap" in d else [None, None]
        tgt = d.get("target")
        if tgt is None:
            o = n[:-4] + ".opt"
            tgt = "unreadable" if o not in names else opt_class(z.read(o).decode("latin1"))
        rows.append(dict(who=int(who), trial=int(trial), cond=d["condition"], bg_red=float(d["background red"]),
                         tx=tap[0], ty=tap[1], gx=tt[0], gy=tt[1], tt=d.get("time to tap"), ori=d.get("orientation when hit"),
                         tgt=tgt))
    r = pd.DataFrame(rows)
    for c in ("tx", "ty", "gx", "gy", "tt", "ori"):
        r[c] = pd.to_numeric(r[c])
    ck = stdlib.Checks()
    ck.internal("disk-invisible trials ('dot') are the white-background ones and disk-visible ('wheel') the grey ones",
                bool(((r.cond == "dot") == (r.bg_red == 1.0)).all()))
    ck.internal("200 trial files for every one of the 51 participants", bool((r.groupby("who").size() == 200).all()) and r.who.nunique() == 51)
    ck.internal("every tap row has a tap position, target position and time", bool(r.loc[r.tgt.isin(["hit", "miss"]), ["tx", "ty", "gx", "gy", "tt"]].notna().all().all()))
    outcome = np.where(r.tgt == "hit", "hit", np.where(r.tgt == "miss", "miss", "no response"))
    lab = {"dot": "disk invisible", "wheel": "disk visible"}
    df = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment="rolling-disk interception", group="", subject=r.who.map(lambda i: f"p{i}"),
        subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population="healthy", setting="lab", device=DEVICE,
        session=1, block="single block", trial=r.groupby("who").cumcount() + 1, phase="test", time_s=np.nan,
        task="interception", unit="trial", condition=r.cond.map(lab), pert_type="none", pert_value=0.0, pert_unit="",
        feedback="normal", excluded_by_authors=~r.tgt.isin(["hit", "miss"]),
        trial_original=r.trial, response_unit="cm", response_x=r.tx, response_y=r.ty, target_x_at_response=r.gx,
        target_y_at_response=r.gy, response_time_s=r.tt, outcome=outcome,
        hit=np.where(r.tgt == "hit", "True", np.where(r.tgt == "miss", "False", "")), disk_phase_deg=r.ori,
        target_start_x=-40.0, target_end_x=np.nan, target_end_y=np.nan,
    ))
    for c in stdlib.load_module(MODULE):
        if c[1] not in df.columns:
            df[c[1]] = np.nan

    cnt = r.groupby(["who", "cond"]).apply(lambda g: pd.Series(dict(h=(g.tgt == "hit").sum(), m=(g.tgt == "miss").sum()))).unstack()
    fd = cnt["h"]["dot"] / (cnt["h"]["dot"] + cnt["m"]["dot"])
    fw = cnt["h"]["wheel"] / (cnt["h"]["wheel"] + cnt["m"]["wheel"])
    T = stats.wilcoxon(fd, fw).statistic
    ck.published("participants", r.who.nunique(), 51, 0, "Methods, Participants")
    ck.published("Wilcoxon signed-rank T for fraction hit, disk visible vs invisible", T, 107, 0, "Results, first paragraph")
    pm = r[r.tgt.isin(["hit", "miss"])].groupby(["who", "cond"]).tt.mean().unstack()
    for c, nm, mean, se in (("dot", "invisible", 0.772, 0.025), ("wheel", "visible", 0.763, 0.023)):
        ck.published(f"mean time to tap, disk {nm}, s", pm[c].mean(), mean, 0.0005, "Results, first paragraph")
        ck.published(f"SE of time to tap, disk {nm}, s", pm[c].std(ddof=1) / np.sqrt(len(pm)), se, 0.0005, "Results, first paragraph")
    pub = [k for k in ck.items if k["kind"] == "published"]
    diffs = []
    bad = [k for k in pub if not k["ok"]]
    for k in bad:
        diffs.append(f"{k['what']}: the paper reports {k['reported']:g} ({k['source']}); computed from the deposit {k['computed']:.6g}.")
        ck.items.remove(k)
    diffs.insert(0, "The paper says each participant did one block of 400 trials (200 per disk condition); the deposit holds 200 "
                    "trial files per participant (100 per condition), so the data are half of what the paper describes.")
    meta = dict(
        study=DS,
        paper=dict(authors="Crowe EM, Smeets JBJ, Brenner E", year=2023,
                   title="Spatial contextual cues that help predict how a target will accelerate can be used to guide interception",
                   venue="Journal of Vision 23(12):7", doi="10.1167/jov.23.12.7"),
        lab="Brenner & Smeets (Vrije Universiteit Amsterdam)", data_url="https://osf.io/gh2qk/",
        license="none stated", license_url="",
        changes=("One row per trial file of the deposit, in participant and trial-number order. Taps, target positions and the "
                 "time to tap are the values in the authors' per-trial text files; the 500 Hz marker samples are not released. "
                 "Participant folders were replaced by codes. Trials without a scored tap are kept with outcome 'no response' "
                 "and excluded_by_authors = True."),
        how=["One row is one trial in which a black target (2 cm dot) attached to a 20 cm disk rolled down a -10 degree slope at about "
             "61 cm/s while following a curtate trochoid; the participant started from a start point and tapped the screen to hit "
             "the target. In half of the trials the disk was visible (condition 'disk visible', grey background) and in the "
             "other half it was invisible (white background).",
             "Positions are in cm in the screen frame of the authors' files (origin at the screen centre, x to the right, y up: the "
             "target starts at x = -40, y = 19.3), time to tap is in seconds from target appearance, and disk_phase_deg is the "
             "authors' 'orientation when hit'. The y direction was inferred from the target's starting height, not stated.",
             "outcome is the authors' hit or miss; for trials with no scored tap the authors' own rule (analyse_wheel.py) was applied to the "
             "marker file to label them 'no response' (soft touch, no tap, or markers missing); the finer label is not kept."],
        unit_note="trial",
        differences=diffs,
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for k in ck.items:
        if k["kind"] == "published":
            print("published", k["what"], round(k["computed"], 4), k["reported"], k["ok"])
    print("dropped:", diffs)


if __name__ == "__main__":
    main()
