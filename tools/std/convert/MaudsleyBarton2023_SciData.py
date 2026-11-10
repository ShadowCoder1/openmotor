"""Maudsley-Barton & Yap (2023) Sci Data 10:633 (KINECAL): 90 people (healthy adults, older non-fallers, older self-reported fallers) doing
clinical balance movements in front of a Kinect V2 (30 fps). The full deposit is 180 GB (per-frame depth images and per-frame skeleton
text files), too large to convert. This converter uses the small per-recording files the authors shared inside each recording folder:
  - sway_metrics.csv (the authors' sway metrics, one row per recording) and start_end.csv,
  - the authors' cached stacked raw skeleton array <part>_cached_stacked_raw_XYZ_<movement>.npy: (3 axes, 26 joints incl. the authors' centre
    of mass, frames), joint positions in metres relative to the first frame's spine base (first frame dropped).
and register.csv (participants). Recordings that have neither file (about half: the per-frame skeleton files only) are not in this copy.
Raw files: raw_extra/MaudsleyBarton2023_SciData/ (PhysioNet kinecal 1.0.3, CC0), fetched file by file.
Run from the site folder: python3 -I tools/std/convert/MaudsleyBarton2023_SciData.py"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "MaudsleyBarton2023_SciData"
JOINTS = ["SPINEBASE", "SPINEMID", "NECK", "HEAD", "SHOULDERLEFT", "ELBOWLEFT", "WRISTLEFT", "HANDLEFT", "SHOULDERRIGHT", "ELBOWRIGHT",
          "WRISTRIGHT", "HANDRIGHT", "HIPLEFT", "KNEELEFT", "ANKLELEFT", "FOOTLEFT", "HIPRIGHT", "KNEERIGHT", "ANKLERIGHT", "FOOTRIGHT",
          "SPINESHOULDER", "HANDTIPLEFT", "THUMBLEFT", "HANDTIPRIGHT", "THUMBRIGHT", "COM"]
CH = [f"{j}_{a}" for j in JOINTS for a in "XYZ"]
UN = ["m"] * len(CH)
MOVES = ["Quiet-Standing-Eyes-Open", "Quiet-Standing-Eyes-Closed", "Foam-Quiet-Standing-Eyes-Open", "Foam-Quiet-Standing-Eyes-Closed",
         "Semi-Tandem-Balance", "Tandem-Balance", "Unilateral-Stance-Eyes-Open", "Unilateral-Stance-Eyes-Closed", "STS-5"]
MTEXT = {"Quiet-Standing-Eyes-Open": "quiet standing, eyes open", "Quiet-Standing-Eyes-Closed": "quiet standing, eyes closed",
         "Foam-Quiet-Standing-Eyes-Open": "quiet standing on foam, eyes open", "Foam-Quiet-Standing-Eyes-Closed": "quiet standing on foam, eyes closed",
         "Semi-Tandem-Balance": "semi-tandem stance", "Tandem-Balance": "tandem stance",
         "Unilateral-Stance-Eyes-Open": "single-leg stance, eyes open", "Unilateral-Stance-Eyes-Closed": "single-leg stance, eyes closed",
         "STS-5": "five sit-to-stand repetitions"}
SWAY = ["RDIST_ML", "RDIST_AP", "RDIST", "MDIST_ML", "MDIST_AP", "MDIST", "TOTEX_ML", "TOTEX_AP", "TOTEX", "MVELO_ML", "MVELO_AP", "MVELO",
        "MFREQ_ML", "MFREQ_AP", "MFREQ", "AREA_CE"]
GROUPTXT = {"HA": "healthy adult (under 65, no falls)", "NF": "older adult, no falls in the past year",
            "FHs": "older adult, one self-reported fall in the past year", "FHm": "older adult, two or more self-reported falls in the past year"}
PCOLS = [("group_name", "text", "", "Group in the authors' words (HA healthy adult under 65 with no falls; NF older non-faller; FHs older faller with one fall; FHm older faller with two or more falls)."),
         ("height_m", "number", "m", "Body height."), ("mass_kg", "number", "kg", "Body mass."), ("bmi", "number", "kg/m^2", "Body mass index, as in the deposit."),
         ("recorded_in_lab", "true/false", "", "Deposit column recorded_in_the_lab (1 = recorded in the lab; 0 = not; the deposit does not say where)."),
         ("clinically_at_risk", "true/false", "", "Labelled clinically at risk by the authors (impaired on at least 2 of SPPB, 3 m walk, five-times sit-to-stand, TUG).")]


def clean_move(folder, p):
    m = folder.split("_", 1)[1]
    return m.replace("_fail-16s", "")


def main():
    raw = stdlib.raw_dir(DS)
    R = raw / "rec"
    reg = pd.read_csv(raw / "register.csv").iloc[:, :9]
    reg["age_top_coded"] = reg.age.astype(str).str.startswith(">")
    reg["age"] = pd.to_numeric(reg.age, errors="coerce")
    reg["p"] = reg.part_id.str.replace("SPPB", "").astype(int)
    regd = reg.set_index("p").to_dict("index")
    ck = stdlib.Checks()
    rows, extra, sway_rows = [], [], []
    skipped = {"backup/copy": 0, "ramp": 0, "unmatched cache": 0}
    ratios = []
    seq = {}
    entries = []
    for p in sorted(os.listdir(R), key=int):
        for folder in sorted(os.listdir(R / p)):
            core = clean_move(folder, p)
            if "backup" in folder or "copy" in folder:
                skipped["backup/copy"] += 1
                continue
            if core.startswith("Ramp"):
                skipped["ramp"] += 1
                continue
            assert core in MOVES, core
            entries.append((int(p), folder, core))
    entries.sort(key=lambda e: (e[0], MOVES.index(e[2]), e[1]))
    for p, folder, core in entries:
        d = R / str(p) / folder
        fs = os.listdir(d)
        cands = []
        for f in fs:
            if f.endswith(".npy") and f.startswith(f"{p}_cached_stacked_raw_XYZ_"):
                suf = f[len(f"{p}_cached_stacked_raw_XYZ_"):-4].replace("Oepn", "Open")
                if suf == core:
                    cands.append(np.load(d / f, allow_pickle=False))
        arr = None
        if cands:
            if all(a.shape == cands[0].shape and np.array_equal(a, cands[0]) for a in cands):
                arr = cands[0]
            else:
                skipped["unmatched cache"] += 1
        sw = None
        if "sway_metrics.csv" in fs:
            t = pd.read_csv(d / "sway_metrics.csv")
            assert len(t) == 1
            sw = t.iloc[0]
        win = (None, None)
        if "start_end.csv" in fs:
            t = pd.read_csv(d / "start_end.csv")
            win = (t.iloc[0, 1], t.iloc[0, 2])
        if arr is None and sw is None and win == (None, None):
            continue
        if arr is None and sw is None:
            continue
        seq[p] = seq.get(p, 0) + 1
        r = regd[p]
        sf, n, dur = "", None, None
        if arr is not None:
            assert arr.shape[0] == 3 and arr.shape[1] == 26
            n = arr.shape[2]
            body = []
            for i in range(n):
                v = arr[:, :, i]
                body.append(",".join([f"{(i + 1) / 30:.4f}"] + [format(float(v[a, j]), ".8g") for j in range(26) for a in range(3)]))
            sf = f"samples/{p}_{core}.csv"
            extra.append((sf, ("time_s," + ",".join(CH) + "\r\n" + "\r\n".join(body) + "\r\n").encode("ascii")))
            dur = round(n / 30, 3)
            if sw is not None:
                from scipy import signal
                b, a_ = signal.butter(2, np.pi * 8 / (2 * 30), btype="low")
                ml = signal.filtfilt(b, a_, arr[0, 25]); ap = signal.filtfilt(b, a_, arr[2, 25])
                ml = (ml - ml.mean()) * 100; ap = (ap - ap.mean()) * 100
                ratios.append(float(np.sqrt((ml ** 2 + ap ** 2).mean()) / float(sw["RDIST"])) if float(sw["RDIST"]) > 0 else np.nan)
        eyes_closed = "Closed" in core
        rows.append(dict(
            dataset=DS, study=DS, experiment="", group=r["group"], subject=f"S{p:03d}", person=f"S{p:03d}", subject_excluded_by_authors=False,
            age=r["age"], sex={"m": "M", "f": "F"}[r["sex"]], handedness="", population=GROUPTXT[r["group"]],
            setting="lab" if int(r["recorded_in_the_lab"]) == 1 else "", device="Kinect V2 depth camera on a tripod, 30 fps",
            session=1, block=core, trial=seq[p], phase="test", time_s="", task="balance and posture", unit="recording",
            condition=MTEXT[core], pert_type="foam_surface" if core.startswith("Foam") else "none", pert_value=0, pert_unit="",
            feedback="none" if eyes_closed else "normal", excluded_by_authors=False,
            recording=seq[p], samples_file=sf, signal_kind="Kinect V2 skeleton joint positions (25 joints plus the authors' centre of mass)",
            channels=";".join(CH) if arr is not None else "", channel_units=";".join(UN) if arr is not None else "",
            sample_rate_hz=30 if arr is not None else None, n_samples=n, duration_s=dur, events="",
            processing="Raw Kinect joint positions with the first frame's spine base subtracted (first frame dropped); no filtering"
            if arr is not None else ""))
        srow = dict(subject=f"S{p:03d}", recording=seq[p], movement_folder=core, window_start=win[0], window_end=win[1],
                    has_samples=arr is not None)
        for k in SWAY:
            srow[k] = None if sw is None else sw[k]
        sway_rows.append(srow)
    df = pd.DataFrame(rows)
    # sway_metrics.csv keyed by the final subject codes (assigned by stdlib in order of first appearance)
    order = {}
    for s in df.subject:
        order.setdefault(s, f"{DS}_S{len(order) + 1:04d}")
    sw_df = pd.DataFrame(sway_rows)
    sw_df["subject"] = sw_df.subject.map(order)
    sw_csv = sw_df.to_csv(index=False, lineterminator="\r\n")
    sw_cols = ("column,description\r\nsubject,Participant sequence code as in the main table.\r\nrecording,Recording number as in the main table.\r\n"
               "movement_folder,The authors' movement label (folder name).\r\n"
               "window_start,Deposit start_end.csv 'start' (unit not documented; values 0 to 1).\r\n"
               "window_end,Deposit start_end.csv 'end' (unit not documented; usually 20, which is the 20 s trial length in the paper).\r\n"
               "has_samples,True when a skeleton samples file exists for this recording.\r\n"
               "RDIST_ML / RDIST_AP / RDIST,Root-mean-square distance of the centre of mass from its mean (mediolateral / anteroposterior / resultant), cm (the authors' code labels them cm; the paper gives no unit).\r\n"
               "MDIST_ML / MDIST_AP / MDIST,Mean absolute distance from the mean centre of mass position (same axes and unit).\r\n"
               "TOTEX_ML / TOTEX_AP / TOTEX,Total excursion: path length of the centre of mass (same axes and unit).\r\n"
               "MVELO_ML / MVELO_AP / MVELO,Mean velocity: total excursion divided by trial duration (cm/s).\r\n"
               "MFREQ_ML / MFREQ_AP / MFREQ,Mean frequency (Hz) as defined in the paper.\r\n"
               "AREA_CE,95% confidence ellipse area of the sway path (cm^2).\r\n")
    extra += [("sway_metrics.csv", sw_csv.encode("ascii")), ("sway_metrics_columns.csv", sw_cols.encode("ascii"))]
    prow = []
    for p, r in regd.items():
        if f"S{p:03d}" in set(df.subject):
            prow.append(dict(person=f"S{p:03d}", group_name=r["group"], height_m=r["height"], mass_kg=r["weight"], bmi=r["BMI"],
                             recorded_in_lab=bool(r["recorded_in_the_lab"]), clinically_at_risk=bool(r["clinically-at-risk"])))
    per = df.drop_duplicates("person")
    gc = per.group.value_counts()
    ck.published("healthy adults (HA)", gc.get("HA", 0), 33, 0, "Table 3 / Participants (33)")
    ck.published("older non-fallers (NF)", gc.get("NF", 0), 33, 0, "Table 3 (33)")
    ck.published("older fallers, one fall (FHs)", gc.get("FHs", 0), 15, 0, "Table 3 (15)")
    ck.published("older fallers, two or more falls (FHm)", gc.get("FHm", 0), 9, 0, "Table 3 (9)")
    ck.published("participants", len(per), 90, 0, "Abstract (90 participants)")
    ck.published("males in HA", ((per.group == "HA") & (per.sex == "M")).sum(), 22, 0, "Table 3 (HA 22 male, 11 female)")
    ck.published("males in NF", ((per.group == "NF") & (per.sex == "M")).sum(), 16, 0, "Table 3 (NF 16 male, 17 female)")
    ck.published("males in FHs", ((per.group == "FHs") & (per.sex == "M")).sum(), 7, 0, "Table 3 (FHs 7 male, 8 female)")
    ck.published("males in FHm", ((per.group == "FHm") & (per.sex == "M")).sum(), 2, 0, "Table 3 (FHm 2 male, 7 female)")
    ck.published("mean age of HA (y)", per[per.group == "HA"].age.mean(), 46.2, 0.2, "Table 3 (46.2)")
    nf = per[per.group == "NF"].age
    nf_all = (nf.sum() + 90 * (len(reg[(reg.group == "NF") & reg.age_top_coded]))) / (nf.notna().sum() + reg[(reg.group == "NF") & reg.age_top_coded].shape[0])
    ck.published("mean age of NF (y), the one age given as >89 counted as 90", nf_all, 73.3, 0.15, "Table 3 (73.3)")
    ck.internal("one participant's age is top-coded (>89) in the register and left empty; the rest max at 89 or below", reg.age_top_coded.sum() == 1 and per.age.max() <= 89, str(per.age.max()))
    ck.published("clinically at risk", len(reg[reg["clinically-at-risk"] == 1]), 6, 0, "Table 3 (Clinically-At-Risk 6)")
    ck.internal("all 90 register participants have at least one recording here", len(per) == 90)
    ck.internal("every samples file has 3 x 26 channels x frames of 30 fps data, 9 to 40 s", all(9 <= d <= 40 for d in df.duration_s.dropna()))
    ratios = [x for x in ratios if x == x]
    med = float(np.median(ratios)) if ratios else float("nan")
    ck.internal("units: centre-of-mass RMS distance from the cached skeleton (x100, 8 Hz low-pass) is within 0.7-1.8 times the authors' RDIST (median)", 0.7 < med < 1.8, f"median ratio {med:.2f} over {len(ratios)} recordings")
    ck.internal("sway metric rows equal recordings", len(sw_df) == len(df))
    meta = dict(
        paper=dict(authors="Maudsley-Barton S, Yap MH", year=2023, title="KINECAL: A Dataset for Falls-Risk Assessment and Balance Impairment Analysis",
                   venue="Sci Data 10(1):633", doi="10.1038/s41597-023-02375-w"),
        lab="Yap (Manchester Metropolitan University)", data_url="https://physionet.org/content/kinecal/1.0.3/",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("From each recording folder: the authors' cached skeleton array became samples/<participant>_<movement>.csv (joint x/y/z in metres, "
                 "26 joints incl. the authors' centre of mass; numbers kept to 8 significant digits), and the authors' sway_metrics.csv and "
                 "start_end.csv became rows of sway_metrics.csv. Participant numbers were replaced by codes. Not included: the per-frame depth images "
                 "and per-frame skeleton text files (180 GB), recordings that have neither a cached skeleton array nor a sway_metrics.csv, "
                 "duplicate backup/copy folders, 'Ramp' folders (not described in the paper), and the 3 m walk and Timed Up and Go recordings "
                 "(no cached files)."),
        how=["One row is one recording of one movement of one participant (quiet standing eyes open/closed on a firm surface or foam, semi-tandem, "
             "tandem and single-leg stance, five sit-to-stands). The samples file has time_s (frame number / 30; the first frame is dropped by the authors) and "
             "x, y, z (m) of 25 Kinect V2 joints plus the authors' centre of mass (mean of mid-spine and both hips); positions are relative to the "
             "first frame's spine base. Kinect x is mediolateral, y vertical and z anteroposterior (paper). sway_metrics.csv holds the authors' RDIST, "
             "MDIST, TOTEX, MVELO, MFREQ and AREA_CE for the recording.",
             "trial and recording count a participant's recordings in the order of the paper's movement list (the actual order of movements is not in the deposit). "
             "The age of one participant is given as >89 in the register and is left empty here. Participants whose folder had only some movements appear with only those. group is the authors' register group; the age and sex are "
             "as in register.csv. Recordings were stopped after 20 s, on loss of balance, or when the lifted leg touched down (paper); start_end.csv "
             "is kept as window_start and window_end.",
             "Participants gave written informed consent (paper); the paper and PhysioNet page state no further limits on use."],
        unit_note="recording of one movement (samples in samples/*.csv, authors' sway metrics in sway_metrics.csv)",
        differences=["The paper says participants were aged 18 to 92; the register's youngest is 23 and one age is given only as >89.", "Only the recordings with a cached skeleton array or sway_metrics.csv in their folder are converted (roughly half of the "
                     "990 participant x movement recordings, mostly the standing and stance tests); the rest exist only as 180 GB of per-frame files.",
                     "The register has 47 male and 43 female participants; summing the sex columns of the paper's group table gives the same 47 male.",
                     "A few folders hold sway_metrics.csv files in an older layout (different column names); their values are used unchanged. The sway units are not stated in "
                     "the paper; the authors' code labels them cm and the cached skeleton (metres) times 100 gives values of the same size as the sway files."])
    stdlib.release(DS, "timeseries", df, meta, ck, participants=prow, participant_columns=PCOLS, extra_files=extra)
    print("skipped folders", skipped)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))


main()
