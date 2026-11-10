"""Price, Hoogkamer and Huber (2026), IEEE Trans Neural Syst Rehabil Eng 34:2660-2668: 12 healthy adults walked 20 min on an
instrumented treadmill wearing adjustable-compliance footwear; the foot-ground compliance under one foot was made soft for 10 minutes
and then restored. Raw treadmill ground reaction forces (1 kHz, .mot) and motion-capture markers (100 Hz, .tsv) in three blocks per person.
One row per participant, block and signal type.

Run from the site folder: python3 -I tools/std/convert/Price2026_TNSRE.py
Raw files (untrusted data, read only) in raw_extra/Price2026_TNSRE/ from https://doi.org/10.6084/m9.figshare.31152274, saved by the fetch
script as <participant>_grf_<block>.mot, <participant>_mocap_<block>.tsv and participant_info.csv (Figshare keeps the folder in its
file listing only). Each file becomes a samples CSV and the raw file is deleted right after (small PC disk); reruns skip converted files.
The deposit's allParticicipantsOutcomeMeasures.mat (stride-wise derived measures) is not included.
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Price2026_TNSRE"
RAW = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "tmp_samples" / ID
TMP.mkdir(parents=True, exist_ok=True)
checks = stdlib.Checks()

info = pd.read_csv(RAW / "participant_info.csv", encoding="utf-8-sig")
info.columns = ["participant", "mass", "sex", "age", "height", "side", "speed", "hilo", "lohi"]
BLOCKS = {"pre_onearly": "pre_onearly", "base_onearly": "pre_onearly", "baseline_adapt_shoes": "pre_onearly",
          "onlate_postearly": "onlate_postearly", "adapt_post_shoes": "onlate_postearly", "postlate": "postlate"}
# base_onearly (two participants) is mapped to pre_onearly; the first participant's folder uses other names, mapped below by block length
ORDER = {"pre_onearly": 1, "onlate_postearly": 2, "postlate": 3}
COND = {"pre_onearly": "rigid footwear, then compliant under one foot (switch during the block)",
        "onlate_postearly": "compliant under one foot, then rigid again (switch during the block)",
        "postlate": "rigid footwear after the perturbation"}

# the P02 folder uses different mocap names and a shorter second block; keep each file under its own label
P02_LABEL = {"baseline_adapt_shoes": "pre_onearly", "adapt_post_shoes": "onlate_postearly", "onlate_postearly": "postlate"}
GRF_CH = None
rows, extra = [], []
trial_n = {}
for _, p in info.iterrows():
    pid = p.participant
    num = int(pid[1:])
    trial_n[pid] = 0
    for kind in ("mocap", "grf"):
        def lab(b):
            return P02_LABEL[b] if (pid == "P02" and kind == "mocap") else BLOCKS[b]
        items = {}
        for f in RAW.glob(f"{pid}_{kind}_*.*"):
            b = re.sub(rf"^{pid}_{kind}_", "", f.stem)
            items[lab(b)] = b
        for f in TMP.glob(f"{ID}_P{num:02d}_{kind}_*.csv"):
            blk = f.stem.split(f"_{kind}_")[1]
            items.setdefault(blk, None)
        assert sorted(items) == sorted(ORDER), (pid, kind, items)
        for blk in sorted(items, key=lambda x: ORDER[x]):
            b = items[blk] or blk
            out = TMP / f"{ID}_P{num:02d}_{kind}_{blk}.csv"
            src = RAW / (f"{pid}_{kind}_{b}.tsv" if kind == "mocap" else f"{pid}_{kind}_{b}.mot")
            if not out.exists():
                with open(src, "r", encoding="utf-8", errors="strict") as f, open(out.with_suffix(".part"), "w", newline="", encoding="utf-8") as g:
                    if kind == "mocap":
                        meta = {}
                        for line in f:
                            if line.startswith("Frame\t"):
                                hdr = line.rstrip("\r\n").split("\t")
                                while hdr and hdr[-1] == "":
                                    hdr.pop()
                                break
                            k = line.rstrip("\r\n").split("\t")
                            meta[k[0]] = k[1:]
                        assert float(meta["FREQUENCY"][0]) == 100.0 and int(meta["NO_OF_FRAMES"][0]) > 0, (src.name, list(meta)[:5])
                        names_c = hdr[1:]
                        g.write(",".join(["time_s"] + names_c[1:]).replace(" ", "_") + "\r\n")
                        n = 0
                        for line in f:
                            q = line.rstrip("\r\n").split("\t")
                            while len(q) > len(hdr) and q[-1] == "":
                                q.pop()
                            assert len(q) == len(hdr), (src.name, n)
                            g.write(",".join(q[1:]) + "\r\n")
                            n += 1
                        assert n == int(meta["NO_OF_FRAMES"][0]), (src.name, n, meta["NO_OF_FRAMES"])
                    else:
                        for line in f:
                            if line.startswith("endheader"):
                                break
                            if line.startswith("nRows="):
                                nrows = int(line.split("=")[1])
                        hdr = f.readline().rstrip("\r\n").split("\t")
                        hdr = [h for h in hdr if h]
                        g.write(",".join(hdr) + "\r\n")
                        n = 0
                        for line in f:
                            q = line.rstrip("\r\n").split("\t")
                            q = [x for x in q if x != ""] if len(q) > len(hdr) else q
                            assert len(q) == len(hdr), (src.name, n, len(q))
                            g.write(",".join(q) + "\r\n")
                            n += 1
                        assert n == nrows, (src.name, n, nrows)
                out.with_suffix(".part").replace(out)
                src.unlink()
            d = pd.read_csv(out, nrows=3)
            chans = list(d.columns[1:])
            tot = sum(1 for _ in open(out)) - 1
            last = float(pd.read_csv(out, usecols=[0]).iloc[-1, 0])
            trial_n[pid] += 1
            fn = f"samples/{ID}_P{num:02d}_{kind}_{blk}.csv"
            extra.append((fn, str(out)))
            ev = []
            if blk == "pre_onearly":
                ev.append(f"compliance_switch_high_to_low={p.hilo:g}")
            if blk == "onlate_postearly":
                ev.append(f"compliance_switch_low_to_high={p.lohi:g}")
            if kind == "mocap":
                sig, rate, units = "3D marker positions (Qualisys, 34 markers)", 100, ";".join(["mm"] * len(chans))
                proc = "none stated for the shared file (the paper filters markers at 6 Hz for analysis)"
            else:
                sig, rate = "treadmill ground reaction forces, centre of pressure and torques of the two belts", 1000
                units = ";".join("N" if "force" in c else ("m" if "_p" in c else "N*m") for c in chans)
                proc = "OpenSim .mot file as shared (the paper filters forces at 30 Hz, fourth-order zero-lag Butterworth, for analysis)"
            rows.append(dict(
                dataset=ID, study=ID, experiment=ID, group="", subject=pid, person=pid, subject_excluded_by_authors=False,
                age=p.age, sex={"F": "F", "M": "M"}.get(p.sex, "other"), handedness="", population="healthy adults", setting="lab",
                device="instrumented dual-belt treadmill (Bertec) and Qualisys 8-camera motion capture, adjustable compliance footwear",
                session=1, block=f"{b} ({blk})" if b != blk else blk, trial=trial_n[pid], phase="other", time_s="",
                task="walking", unit="recording", condition=COND[blk],
                pert_type="surface_compliance", pert_value="", pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=trial_n[pid], samples_file=fn, signal_kind=sig, channels=";".join(chans), channel_units=units,
                sample_rate_hz=rate, n_samples=tot, duration_s=round(last, 3) if kind == "mocap" else round(tot / 1000.0, 3),
                events=";".join(ev), processing=proc,
                trial_info=f"block={blk};perturbed_side={p.side};treadmill_speed_mps={p.speed:g};signal={kind}"))
D = pd.DataFrame(rows)
part = pd.DataFrame([dict(person=p.participant, mass_kg=p.mass, height_cm=p.height, perturbed_side={"L": "left", "R": "right"}[p.side],
                          treadmill_speed_mps=p.speed) for _, p in info.iterrows()])
pcols = [("mass_kg", "number", "kg", "Body mass measured on the treadmill force plates while wearing the footwear."),
         ("height_cm", "number", "cm", "Self-reported height."),
         ("perturbed_side", "category", "left, right", "Side of the foot whose ground compliance was made soft."),
         ("treadmill_speed_mps", "number", "m/s", "Walking speed.")]

P = checks.published
P("participants", D.person.nunique(), 12, 0, "II.A Participants")
P("left-side perturbation", int((info.side == "L").sum()), 8, 0, "II.C Experimental Protocol")
P("right-side perturbation", int((info.side == "R").sum()), 4, 0, "II.C Experimental Protocol")
P("female participants", int((info.sex == "F").sum()), 6, 0, "II.A Participants")
P("male participants", int((info.sex == "M").sum()), 4, 0, "II.A Participants")
P("other-sex participants", int((info.sex == "X").sum()), 2, 0, "II.A Participants")
P("age: mean", info.age.mean(), 25, 0.5, "II.A Participants")
P("age: SD", info.age.std(ddof=1), 6, 0.5, "II.A Participants")
P("height: mean (cm)", info.height.mean(), 166, 0.5, "II.A Participants")
P("height: SD (cm)", info.height.std(ddof=1), 5, 0.5, "II.A Participants")
P("weight: mean (kg)", info["mass"].mean(), 69.2, 0.05, "II.A Participants")
P("weight: SD (kg)", info["mass"].std(ddof=1), 9.4, 0.05, "II.A Participants")
P("participants walking at 1.0 m/s", int((info.speed == 1.0).sum()), 11, 0, "II.C Experimental Protocol (all but one)")
dd = D.assign(blk=D.trial_info.str.extract(r"block=([a-z_]+)")[0], sig=D.trial_info.str.extract(r"signal=(\w+)")[0]).pivot_table(index=["person", "blk"], columns="sig", values="duration_s")
ren = dd.loc[["P02", "P03", "P14"]]
checks.internal("renamed marker files (P02, P03, P14) match the length of the force file of the same block within 2 s (supports the block mapping)", bool((abs(ren.mocap - ren.grf) < 2).all()), str(ren.round(1).to_dict()))
LEN_DIFF = dd[abs(dd.mocap - dd.grf) >= 2].round(1)
print("marker/force length differences:", LEN_DIFF.to_dict())
checks.internal("6 recordings per participant (3 blocks x mocap and grf)", all(v == 6 for v in trial_n.values()) and len(D) == 72)

meta = dict(
    paper=dict(authors="Price M, Hoogkamer W, Huber ME", year=2026,
               title="Gait Adaptation to Asymmetric Foot-Ground Compliance Applied by Robotic Footwear",
               venue="IEEE Trans Neural Syst Rehabil Eng 34:2660-2668", doi="10.1109/tnsre.2026.3696575"),
    lab="Hoogkamer & Huber (UMass Amherst)",
    data_url="https://doi.org/10.6084/m9.figshare.31152274", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each of the 72 raw files (36 marker .tsv, 36 force .mot) became one samples CSV: tabs became commas, the header lines of the Qualisys and OpenSim files were dropped "
             "(frame numbers dropped, Time kept as time_s) and the column names had spaces replaced by underscores; no value was changed. Sex X in the deposit's participant table is written as other. "
             "The deposit's allParticicipantsOutcomeMeasures.mat (stride-wise measures derived by the authors) and its marker/force files for the unused participants are not part of this release."),
    how=["One row is one block of one participant's 20-minute walk, for one signal type (markers or forces). Blocks are the authors' pre_onearly (rigid shoes, then the compliance change on one side), "
         "onlate_postearly (compliant, then rigid again) and postlate (rigid). The compliance_switch events give the time in seconds from the start of the block at which the authors switched the footwear "
         "(participant info table, 'Stiffness hi/lo' and 'lo/hi'); marker and force files of a block are assumed to start together.",
         "Marker samples (100 Hz) are 34 markers x X, Y, Z in mm (Qualisys); force samples (1 kHz, OpenSim .mot) are, for each of the two belts, the ground force vector (N), centre of pressure position (m) and torque (N*m), "
         "in the deposit's names (the second belt's channels start with 1_). Which belt is left or right is not stated; the OpenSim y axis is vertical. Units are those of the OpenSim and Qualisys conventions, not stated in the files.",
         "Naming oddities in the deposit: two participants' marker files are named base_onearly (mapped to pre_onearly) and the first participant's marker files are named "
         "baseline_adapt_shoes, adapt_post_shoes and onlate_postearly, mapped to pre_onearly, onlate_postearly and postlate. These mappings are inferred from file sizes and checked against "
         "the lengths of the force files of the same block (see the checks), not stated by the authors; block shows the authors' file name where it differs."],
    differences=["Marker and force files of the same block differ in length for two blocks in the deposit: P08 postlate (markers 120 s, forces 60 s) and P12 pre_onearly (forces 110.4 s, markers 120 s); both are kept as shared. The paper's Table-free outcome measures (GRF impulse, push-off, braking, spatio-temporal asymmetries) come from the .mat file and were not recomputed; checks are participant counts and demographics. "
                 "The article itself is licensed CC BY-NC-ND while the data deposit is CC BY 4.0; the data license is used."],
    unit_note="recording (one block of one participant, one signal type)",
)
stdlib.release(ID, "timeseries", D, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
shutil.rmtree(TMP, ignore_errors=True)
