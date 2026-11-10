"""Tomassini et al. (2022), iScience 25(4):104096: interpersonal synchronization of rhythmic index-finger flexion-extension
(0.25 Hz) in 30 couples (main experiment) plus the deposit's other experiments, recorded with Vicon at 300 Hz: the three secondary tasks
(RealPartner/HandProne, RealPartner/Arm horizontal and vertical, VirtualPartner dot tracking) and the speed-dependency control experiment
(Control_Exp, 10 further couples). Not carried: the extra markers of couples 21-30 (elbow, wrist), the force/photodiode-like channels
(fdA, fdB, pp0-pp2; not documented), and trial_data/ (the authors' FieldTrip copy of the main-experiment positions).

Raw files, raw_extra/Tomassini2022_iScience/ (OSF cagh8, CC-By 4.0), named <path of the OSF file with / replaced by __>, for example
RealPartner__Main__dyad__inphase__C01_P.mat. Positions are struct data: markers, fs, pos(marker, block, time, xyz). Control_Exp files
(Control__Cnn.mat) hold dataCouple with Pos (partner x time, one axis) and a condition label per trial.
Paper read through PubMed Central (PMC8971945, NCBI efetch).
Run from the site folder: python3 -I tools/std/convert/Tomassini2022_iScience.py
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tomassini2022_iScience"
P = "Tomassini et al. 2022, "
R = stdlib.raw_dir(ID)
COND = {"solo": "solo", "dyad__inphase": "dyad in-phase", "dyad__antiphase": "dyad anti-phase"}
DEVICE = "Vicon motion capture, 10 cameras, retro-reflective markers on the right index finger"
PACE = {30: 0.25, 60: 0.5, 90: 0.75, 120: 1.0}  # control-experiment labels (movements per minute) -> cycle frequency (Hz), inferred
DOTCOND = {"DOTsolo_inphase": ("dot from solo kinematics, tracked in-phase", "solo"),
           "DOTsolo_antiphase": ("dot from solo kinematics, tracked anti-phase", "solo"),
           "DOTinphase_inphase": ("dot from dyad in-phase kinematics, tracked in-phase", "in-phase"),
           "DOTantiphase_antiphase": ("dot from dyad anti-phase kinematics, tracked anti-phase", "anti-phase")}


def load(name):
    return sio.loadmat(R / name, simplify_cells=True)["data"]


def names(d):
    m = d["markers"]
    return [str(x) for x in (m if np.ndim(m) else [m])]


def csv_bytes(t, cols, fmt="{:.8g}"):
    buf = io.StringIO()
    buf.write("time_s," + ",".join(c for c, _ in cols) + "\n")
    arr = np.column_stack([t] + [np.asarray(a, dtype=float) for _, a in cols])
    for r in arr:
        buf.write(",".join("" if np.isnan(v) else fmt.format(v) for v in r) + "\n")
    return buf.getvalue().encode()


def row(experiment, person, c, role, cond, b, path, chans, fs, n, t_info, extra_info=""):
    cols = chans
    return {
        "dataset": ID, "study": ID, "experiment": experiment, "group": "", "subject": person, "person": person,
        "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "right", "population": "healthy",
        "setting": "lab", "device": DEVICE, "session": 1, "block": f"trial {b + 1}", "trial": 0, "phase": "test", "time_s": "",
        "task": "finger tapping and timing", "unit": "recording", "condition": cond, "pert_type": "none", "pert_value": 0.0,
        "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
        "recording": 0, "samples_file": path,
        "signal_kind": "positions (" + ", ".join(nm for nm, _ in t_info) + ")",
        "channels": ";".join(c_ for c_, _ in cols), "channel_units": ";".join("mm" for _ in cols), "sample_rate_hz": fs,
        "n_samples": n, "duration_s": n / fs, "events": "", "processing": "none (positions as shared)",
        "trial_info": f"couple={c}; role={role}" + extra_info}


def real_partner(rows, extra, st, family, label, expkey):
    """Main, HandProne and Arm files: RealPartner__<family>__{solo,dyad__inphase,dyad__antiphase}__Cnn_<tag>.mat"""
    files = sorted(p.name for p in R.glob(f"RealPartner__{family}__*.mat"))
    for f in files:
        m = re.match(rf"RealPartner__{family}__(dyad__inphase|dyad__antiphase|solo)__C(\d\d)_(NP|P|A|B|solo)\.mat", f)
        assert m, f
        cond, c, tag = COND[m.group(1)], int(m.group(2)), m.group(3)
        d = load(f)
        mk, pos = names(d), np.asarray(d["pos"], dtype=float)
        t = np.asarray(d["time_axis"] if "time_axis" in d else d["time"], dtype=float).ravel()
        fs = float(d["fs"])
        st["fs"].add(fs)
        if m.group(1) == "solo":
            if tag in ("A", "B"):
                subj = [(tag, [("tip", mk.index("FingerTip")), ("base", mk.index("FingerBase"))])]
            else:
                subj = [("A", [("tip", mk.index("finger_tip1")), ("base", mk.index("finger_base1"))]),
                        ("B", [("tip", mk.index("finger_tip2")), ("base", mk.index("finger_base2"))])]
        else:
            tips = ["FingerTip1", "FingerTip2"] if "FingerTip1" in mk else ["finger_tip1", "finger_tip2"]
            subj = [("A", [("tip", mk.index(tips[0]))]), ("B", [("tip", mk.index(tips[1]))])]
        for role, chans in subj:
            person = f"couple{c:02d}{role}"
            st[expkey]["couples"].add(c)
            st[expkey]["persons"].add(person)
            for b in range(pos.shape[1]):
                cols = [(f"{nm}_{ax}_mm", pos[i, b, :, k]) for nm, i in chans for k, ax in enumerate("xyz")]
                path = f"samples/<<{person}>>_{label}_{m.group(1).replace('dyad__', 'dyad_')}_{b + 1}.csv"
                extra.append((path, csv_bytes(t, cols)))
                st[expkey]["dur"].add(round(len(t) / fs, 3))
                st[expkey]["n_per"][(person, cond)] = st[expkey]["n_per"].get((person, cond), 0) + 1
                rows.append(row(f"Real partner, {label}", person, c, role, cond, b, path, cols, fs, len(t), chans_info(chans)))


def chans_info(chans):
    return [(n, i) for n, i in chans]


def virtual_partner(rows, extra, st):
    for f in sorted(p.name for p in R.glob("VirtualPartner__*.mat")):
        m = re.match(r"VirtualPartner__(DOT\w+?)__C(\d\d)_\w+\.mat", f)
        assert m, f
        cond, _ = DOTCOND[m.group(1)]
        c = int(m.group(2))
        d = load(f)
        mk, pos = names(d), np.asarray(d["pos"], dtype=float)
        t = np.asarray(d["time"], dtype=float).ravel()
        fs = float(d["fs"])
        dot = np.asarray(d["dot_profile"], dtype=float)
        st["fs"].add(fs)
        dot_fs = dot.shape[1] / 150.0  # inferred: the dot profile covers the 150 s of a trial
        for role, k in (("A", 1), ("B", 2)):
            person = f"couple{c:02d}{role}"
            chans = [("tip", mk.index(f"finger_tip{k}")), ("base", mk.index(f"finger_base{k}"))]
            st["virt"]["couples"].add(c)
            st["virt"]["persons"].add(person)
            for b in range(pos.shape[1]):
                cols = [(f"{nm}_{ax}_mm", pos[i, b, :, kk]) for nm, i in chans for kk, ax in enumerate("xyz")]
                path = f"samples/<<{person}>>_virtual_{m.group(1)}_{b + 1}.csv"
                extra.append((path, csv_bytes(t, cols, "{:.7g}")))
                st["virt"]["dur"].add(round(len(t) / fs, 3))
                st["virt"]["n_per"][(person, cond)] = st["virt"]["n_per"].get((person, cond), 0) + 1
                r = row("Virtual partner (dot tracking)", person, c, role, cond, b, path, cols, fs, len(t), chans_info(chans),
                        extra_info=f"; dot_file=samples/<<couple{c:02d}A>>_dot_{m.group(1)}_{b + 1}.csv")
                rows.append(r)
            if role == "A":
                for b in range(dot.shape[0]):
                    extra.append((f"samples/<<{person}>>_dot_{m.group(1)}_{b + 1}.csv",
                                  csv_bytes(np.arange(dot.shape[1]) / dot_fs, [("dot_profile", dot[b])], "{:.7g}")))


def control(rows, extra, st):
    for f in sorted(p.name for p in R.glob("Control__C*.mat")):
        c = int(re.match(r"Control__C(\d\d)\.mat", f).group(1))
        d = sio.loadmat(R / f, simplify_cells=True)["dataCouple"]
        labels = [int(str(x).strip()) for x in d["condition"]]
        seen = {}
        for b, (lab, pos) in enumerate(zip(labels, d["Pos"])):
            pos = np.asarray(pos, dtype=float)
            t = np.arange(pos.shape[1]) / 300.0
            seen[lab] = seen.get(lab, 0) + 1
            for role, i in (("A", 0), ("B", 1)):
                person = f"ctrlcouple{c:02d}{role}"
                st["ctrl"]["couples"].add(c)
                st["ctrl"]["persons"].add(person)
                cond = f"dyad in-phase, {PACE[lab]:g} Hz"
                cols = [("tip_main_axis_mm", pos[i])]
                path = f"samples/<<{person}>>_control_{PACE[lab]:g}Hz_{seen[lab]}.csv"
                extra.append((path, csv_bytes(t, cols, "{:.7g}")))
                st["ctrl"]["dur"].add(round(len(t) / 300.0, 3))
                st["ctrl"]["n_per"][(person, cond)] = st["ctrl"]["n_per"].get((person, cond), 0) + 1
                st["ctrl"]["pace_couples"].setdefault(PACE[lab], set()).add(c)
                ac = pos[i] - np.nanmean(pos[i])
                sp = np.abs(np.fft.rfft(ac * np.hanning(len(ac)))) ** 2
                fr = np.fft.rfftfreq(len(ac), 1 / 300.0)
                sel = (fr > 0.1) & (fr < 2.5)
                st["ctrl"]["peak"].setdefault(PACE[lab], []).append(float(fr[sel][np.argmax(sp[sel])]))
                r = row("Control experiment (speed dependency)", person, c, role, cond, b, path, cols, 300.0, pos.shape[1], [("tip", 0)],
                        extra_info=f"; pace_label={lab}")
                r["signal_kind"] = "fingertip position along the main movement axis (one axis, as shared)"
                rows.append(r)


def new_stat():
    return {"couples": set(), "persons": set(), "dur": set(), "n_per": {}, "pace_couples": {}, "peak": {}}


def build():
    rows, extra = [], []
    st = {"fs": set(), "main": new_stat(), "hand": new_stat(), "armh": new_stat(), "armv": new_stat(), "virt": new_stat(), "ctrl": new_stat()}
    real_partner(rows, extra, st, "Main", "main task", "main")
    n_main = len(rows)
    real_partner(rows, extra, st, "HandProne", "hand prone", "hand")
    real_partner(rows, extra, st, "Arm__horizontal", "arm horizontal", "armh")
    real_partner(rows, extra, st, "Arm__vertical", "arm vertical", "armv")
    virtual_partner(rows, extra, st)
    control(rows, extra, st)
    # the authors' main-task rows keep the original experiment label
    for r in rows[:n_main]:
        r["experiment"] = "Real partner, main task"
    # subject codes as release() will assign them (order of first appearance), used in the sample file names
    codes = {}
    for r in rows:
        codes.setdefault(r["person"], f"{ID}_S{len(codes) + 1:04d}")
    sub = lambda s: re.sub(r"<<(\w+)>>", lambda m: codes[m.group(1)], s)  # noqa: E731
    for r in rows:
        r["samples_file"] = sub(r["samples_file"])
        r["trial_info"] = sub(r["trial_info"])
    extra = [(sub(p), b) for p, b in extra]
    cnt = {}
    for r in rows:
        r["trial"] = cnt[r["subject"]] = cnt.get(r["subject"], 0) + 1
        r["recording"] = r["trial"]
    return rows, extra, st, n_main


def main():
    checks = stdlib.Checks()
    rows, extra, st, n_main = build()
    M, H, AH, AV, V, C = (st[k] for k in ("main", "hand", "armh", "armv", "virt", "ctrl"))
    checks.published("couples in the main experiment", len(M["couples"]), 30, 0, P + "Participants (thirty couples, main experiment)")
    checks.published("participants in the main experiment", len(M["persons"]), 60, 0, P + "Participants (30 couples)")
    checks.published("sampling rate (Hz)", list(st["fs"])[0], 300, 0, P + "Kinematic data recording (300 Hz)")
    checks.published("trials per participant and condition", min(M["n_per"].values()), 2, 0, P + "Data collection (two trials per couple and condition)")
    checks.published("trials per participant and condition (max)", max(M["n_per"].values()), 2, 0, P + "Data collection")
    checks.published("conditions per participant", len({k[1] for k in M["n_per"]}), 3, 0, P + "Setup and procedure (solo, in-phase, anti-phase)")
    checks.published("shortest trial (s, 2.5 min)", min(M["dur"]), 150, 0.5, P + "Data collection (2.5 min trials)")
    checks.internal("every main-experiment participant has 6 recordings (3 conditions x 2 trials)", n_main == 6 * len(M["persons"]), f"{n_main} recordings")
    checks.internal("trial durations", True, str(sorted(M["dur"])))
    # extension
    checks.published("participants in all experiments (80 = 40 couples)", len(M["persons"]) + len(C["persons"]), 80, 0, P + "Participants (eighty participants formed 40 couples)")
    checks.published("couples in the control experiment", len(C["couples"]), 10, 0, P + "Participants (10 couples in the control experiment)")
    checks.published("couples in the hand-prone task", len(H["couples"]), 10, 0, P + "Setup and procedure (secondary tasks, 10 couples each)")
    checks.published("couples in the arm task", len(AH["couples"] | AV["couples"]), 10, 0, P + "Setup and procedure (secondary tasks, 10 couples each)")
    checks.published("couples in the virtual-partner task", len(V["couples"]), 10, 0, P + "Setup and procedure (Virtual partner, 10 couples)")
    checks.published("hand-prone trials per participant and condition", {min(H["n_per"].values()), max(H["n_per"].values())} == {2} and 2 or 0, 2, 0, P + "Data collection (two trials per condition)")
    checks.published("arm trials per participant and condition", {min(AH["n_per"].values()), max(AH["n_per"].values()), min(AV["n_per"].values()), max(AV["n_per"].values())} == {3} and 3 or 0, 3, 0,
                     P + "Data collection (three trials of 1.5 min in the Arm task)")
    checks.published("arm trial duration (s, 1.5 min)", min(AH["dur"] | AV["dur"]), 90, 0.5, P + "Data collection (1.5 min trials in the Arm task)")
    checks.published("arm task conditions per participant (solo, dyad in-phase)", len({k[1] for k in AH["n_per"]}), 2, 0, P + "Secondary tasks (only solo and dyad-in-phase tested in Arm)")
    checks.published("virtual-partner trials per participant and tracked-dot condition", {min(V["n_per"].values()), max(V["n_per"].values())} == {2} and 2 or 0, 2, 0, P + "Setup and procedure (2 trials per condition)")
    checks.published("control trial duration (s, 2.5 min)", min(C["dur"]), 150, 0.5, P + "Data collection (2.5 min trials)")
    checks.published("control couples at 0.25 Hz", len(C["pace_couples"][0.25]), 10, 0, P + "Control experiment (paces 0.25, 0.5, 0.75 Hz in all 10 couples)")
    checks.published("control couples at 0.5 Hz", len(C["pace_couples"][0.5]), 10, 0, P + "Control experiment")
    checks.published("control couples at 0.75 Hz", len(C["pace_couples"][0.75]), 10, 0, P + "Control experiment")
    checks.published("control couples at 1 Hz", len(C["pace_couples"][1.0]), 5, 0, P + "Control experiment (five couples also tested at 1 Hz)")
    meds = [float(np.median(v)) for _, v in sorted(C["peak"].items())]
    checks.internal("control: median spectral peak of the finger position rises with the pace label (labels read as 0.25, 0.5, 0.75, 1 Hz)",
                    all(x < y for x, y in zip(meds, meds[1:])), ", ".join(f"{k:g} Hz: {m:.2f}" for k, m in zip(sorted(C["peak"]), meds)))
    checks.internal("control: peak at the two slowest paces is close to the instructed pace", abs(meds[0] - 0.25) < 0.1 and abs(meds[1] - 0.5) < 0.15, f"{meds[0]:.2f}, {meds[1]:.2f} Hz")
    checks.internal("recordings in the main experiment kept as before", n_main == 360, f"{n_main}")
    checks.internal("all persons", len({r['person'] for r in rows}) == 80, f"{len({r['person'] for r in rows})} persons, {len(rows)} recordings")
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8")) if x["id"] == ID)
    p0 = s["papers"][0]
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC-By Attribution 4.0 International",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Split the authors' per-couple position arrays into one CSV per participant and trial (fingertip, plus finger base where the file has it), "
                "and replaced file codes. Samples were not resampled or filtered. Carries the main experiment, the three secondary tasks and the control experiment (see how).",
        how=["One row is one trial (2.5 min in the main task, hand-prone, virtual-partner and control experiments; 1.5 min in the arm task) of one participant in one condition. "
             "Main task: rhythmic index finger flexion-extension at about 0.25 Hz alone (solo), or with a partner in-phase or anti-phase, with the fingertip position in x (mediolateral), "
             "y (anteroposterior) and z (vertical), 300 Hz (experiment 'Real partner, main task'). Units are millimetres (Vicon default; inferred, the deposit does not state them). "
             "Trials of couples 21-30 are 152 s long, those of couples 1-20 150 s.",
             "Which marker belongs to which partner is inferred from the file layout: marker 1 (FingerTip1, finger_tip1) is partner A, marker 2 is partner B, and the solo files C..._A and C..._B (couples 1-20) are partners A and B; "
             "for couples 21-30 the single solo file holds both partners. The two partners of a couple share the same time axis in a dyad file (trial_info gives couple and role). The secondary tasks reuse these conventions: "
             "'hand prone' (couples 11-20, finger moved along the vertical axis with the hand prone; in-phase and anti-phase mean the opposite of the main task), 'arm horizontal' and 'arm vertical' (couples 1-10, whole-forearm movement at about 0.17 Hz, "
             "solo and dyad in-phase only, three 90 s trials) and 'Virtual partner' (couples 21-30, tracking a dot that replays a recorded fingertip velocity; condition says whose kinematics the dot showed and whether the tracking was in-phase or anti-phase; "
             "the dot profile of each trial is a separate samples file, trial_info dot_file, named with partner A's subject code; its sampling rate is inferred as 9000 samples over 150 s = 60 Hz and its unit is not stated).",
             "Control experiment (speed dependency), 10 further couples (20 participants who are not in the main experiment), dyad in-phase at different paces: the file holds eight (or, for 5 couples, six) trials, each with a label 30, 60, 90 or 120. "
             "These labels are read as movements per minute, that is 0.25, 0.5, 0.75 and 1 Hz cycle frequency (inferred; consistent with the paper: the labels 30, 60, 90 are in all couples, 120 in 5, and the spectral peak of the finger signals rises with the label, near the instructed frequency at 0.25 and 0.5 Hz and above it at 0.75 and 1 Hz, where movements were self-paced and ran faster than instructed: median peaks about 0.94 and 1.27 Hz). "
             "Each control recording has one position axis, in the units of the file (millimetres by inference), for partner A (first row) and partner B (second row; inferred). Control files have no time axis, so time is sample number / 300 Hz.",
             "Not carried: the elbow and wrist markers and extra signals of couples 21-30 (fdA, fdB, pp0, pp1, pp2; not documented in the deposit), and trial_data/ (the authors' FieldTrip copy of the main positions). "
             "Age and sex per participant are not in the files."],
        unit_note="recording (one participant, one condition, one trial)",
        differences=["The paper's spectral, phase-locking and submovement statistics were not recomputed. Where the main text gives only designs (not measured values), the published checks reproduce counts, trial structure and instructed paces; "
                     "the control experiment's pace labels are checked against the finger-signal spectra as internal checks only."])
    return stdlib.release(ID, "timeseries", rows, meta, checks, extra_files=extra)


if __name__ == "__main__":
    main()
