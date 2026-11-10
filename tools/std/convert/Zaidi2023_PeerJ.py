"""Zaidi & Harris-Love (2023), PeerJ: simple forward-reaching test (SRT) after stroke, Optotrak 3D hand-marker recordings at 300 Hz,
one row per recording (a set of ten timed reaches), in the OpenMotor timeseries format.

Run from the site folder: python3 -I tools/std/convert/Zaidi2023_PeerJ.py
Raw files (untrusted data, read only) in raw_extra/Zaidi2023_PeerJ/, from https://zenodo.org/records/7587404 (CC BY 4.0):
  S<nn>_SRT_{NP,P}{1,2}.xlsx  one recording per participant, arm (NP = non-paretic, P = paretic) and set (1, 2)
  S02/S04/S05/S06_SRT_P{1,2}_Original.xlsx  the same four participants' paretic-arm recordings (the deposit also holds shortened .xls
                              copies of them, which are not converted)
  S00_SRT.xlsx                one healthy control, one marker per arm (sheets Left, Right)
  Input to DataProcessing.xlsx  per participant: sex, age, paretic side, dominant side, maximum reach (cm), severity; sheet 'Further
                              Data': time since stroke
The authors' MATLAB scripts were read as text only (DataProcessing.m): marker 1 is the left hand and marker 2 the right hand (the script
reads columns 2-4 for the left arm and 5-7 for the right). Not converted: the PIH recordings (the examiner moves the participant's hand
passively along the ideal path, a calibration), S00_L and S00_R (the control's 8-marker trial sheets, undocumented).
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
import Zaidi2023_PeerJ_xl as xl  # noqa: E402

DS = "Zaidi2023_PeerJ"
MODULE = "timeseries"
RAW = stdlib.raw_dir(DS)
FS = 300.0
ck = stdlib.Checks()

# ---------------------------------------------------------------- participants
book = xl.read_xlsx(RAW / "Input to DataProcessing.xlsx")
inp = {r[2]: r for r in book["Input"][1:] if r and len(r) > 8 and r[2]}
further = {r[0]: r for r in book["Further Data"][2:] if r and r[0]}
assert len(inp) == 29 and set(inp) == set(further)
SIDE = {1: "left", 2: "right"}
part = {}
side_conflicts = []
for s, r in inp.items():
    f = further[s]
    # the 'Further Data' sheet agrees with the paper's Tables 1 and 2; the 'Input' sheet disagrees for S42 (checked below)
    paretic = {"L": "left", "R": "right"}[f[2]]
    dominant = {"L": "left", "R": "right"}[f[3]]
    if (SIDE[int(r[5])], SIDE[int(r[6])]) != (paretic, dominant):
        side_conflicts.append(s)
    sev = {1: "mild", 2: "severe"}[int(r[8])]
    assert f[5].lower() == sev and abs(float(f[4]) - float(r[7])) < 1e-9
    part[s] = dict(sex=r[3], age=float(r[4]), paretic=paretic, dominant=dominant, max_reach_cm=float(r[7]), severity=sev,
                   months=float(f[1]))

# ---------------------------------------------------------------- recordings
def num(v):
    if isinstance(v, str):
        v = v.strip()
        if v == "" or v.lower() == "nan":
            return None
        return float(v)
    return v


def parse(path, sheet=None):
    sh = xl.read_any(path)
    rows = sh[sheet] if sheet else next(iter(sh.values()))
    hi = next(i for i, r in enumerate(rows) if r and isinstance(r[0], str) and r[0].startswith("Frame"))
    hdr = rows[hi]
    if "\t" in str(hdr[0]):  # header stored as tab-joined text in the first cell(s)
        hdr = [t for c in hdr if c is not None for t in str(c).split("\t")]
    if len(hdr) == 13:  # 'Marker_1<tab>x' pairs
        hdr = ["Frame"] + [f"{hdr[i]} {hdr[i + 1]}" for i in range(1, 13, 2)]
    meta = [' '.join(str(c).replace('\t', ' ') for c in r if c is not None) for r in rows[:hi] if r and r[0]]
    data = []
    for r in rows[hi + 1:]:
        if r and isinstance(r[0], str) and "\t" in r[0]:  # some rows are stored tab-joined across the first cells
            toks = []
            for c in r:
                toks += c.split("\t") if isinstance(c, str) else [c]
            r = toks
        r = list(r) + [None] * (len(hdr) - len(r))
        data.append([num(v) for v in r[:len(hdr)]])
    return hdr, data, meta


def chan_names(hdr):
    out = []
    for h in hdr[1:]:
        h = h.strip().replace(" ", "_")
        out.append(h)
    return out


out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW.parent.parent / "scratch" / f"{DS}_samples"
out_dir.mkdir(parents=True, exist_ok=True)
rows, extra = [], []
n_rec = 0
rec_per = {}
bad_rate = 0
frames_ok = True
for s in sorted(part):
    pp = part[s]
    for arm_set in ("NP1", "NP2", "P1", "P2"):
        name = f"{s}_SRT_{arm_set}.xlsx"
        if not (RAW / name).exists():
            name = f"{s}_SRT_{arm_set}_Original.xlsx"
        if not (RAW / name).exists():
            continue
        hdr, data, meta = parse(RAW / name)
        assert hdr[0].startswith("Frame") and len(hdr) == 7, (name, hdr)
        assert any("Frequency: 300" in m for m in meta) and any("Units: mm" in m for m in meta), (name, meta)
        paretic_arm = arm_set.startswith("P")
        moving = pp["paretic"] if paretic_arm else {"left": "right", "right": "left"}[pp["paretic"]]
        fr = [r[0] for r in data]
        frames_ok &= all(b - a == 1 for a, b in zip(fr, fr[1:]))
        n_rec += 1
        rec_per[s] = rec_per.get(s, 0) + 1
        chans = ["marker1_left_hand_x", "marker1_left_hand_y", "marker1_left_hand_z",
                 "marker2_right_hand_x", "marker2_right_hand_y", "marker2_right_hand_z"]
        fname = f"samples/{s}_{arm_set}.csv"
        lp = out_dir / f"{s}_{arm_set}.csv"
        with open(lp, "w", newline="") as f:
            f.write("time_s," + ",".join(chans) + "\r\n")
            t0 = data[0][0]
            for r in data:
                f.write("%.6f," % ((r[0] - t0) / FS) + ",".join("" if v is None else "%.8g" % v for v in r[1:]) + "\r\n")
        extra.append((fname, str(lp)))
        rows.append(dict(
            dataset=DS, study=DS, experiment="", group=pp["severity"], subject=s, subject_excluded_by_authors=False,
            age=pp["age"], sex=pp["sex"], handedness=pp["dominant"], population="stroke", setting="lab",
            device="Optotrak Certus, two infrared markers on the dorsal hands", session=int(arm_set[-1]),
            block=f"{'paretic' if paretic_arm else 'non-paretic'} arm, set {arm_set[-1]}", trial=rec_per[s], phase="test",
            time_s="", task="reaching", unit="recording", condition=f"{'paretic' if paretic_arm else 'non-paretic'} arm",
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            recording=n_rec, samples_file=fname, signal_kind="3D positions of two infrared markers on the dorsal surface of the hands",
            channels=";".join(chans), channel_units=";".join(["mm"] * 6), sample_rate_hz=FS, n_samples=len(data),
            duration_s=round((len(data) - 1) / FS, 6), events="",
            processing="none by the authors in the shared file (raw Optotrak export); empty cells are frames where a marker was not visible",
            trial_info=f"moving_arm={moving};paretic_arm={pp['paretic']};file_variant={'original' if name.endswith('_Original.xlsx') else 'standard'};"
                       f"frames_in_file_header={meta[0].split(':')[1].strip()}",
        ))

# control: S00_SRT (one marker per arm; sheets Left, Right)
book = xl.read_xlsx(RAW / "S00_SRT.xlsx")
for k, (sheet, arm) in enumerate((("Left", "left"), ("Right", "right")), start=1):
    r_ = book[sheet]
    hi = next(i for i, r in enumerate(r_) if r and r[0] == "Frame")
    data = [r for r in r_[hi + 1:] if r and r[0] is not None]
    n_rec += 1
    chans = [f"{arm}_hand_x", f"{arm}_hand_y", f"{arm}_hand_z"]
    fname = f"samples/S00_{sheet}.csv"
    lp = out_dir / f"S00_{sheet}.csv"
    fr = [r[0] for r in data]
    with open(lp, "w", newline="") as f:
        f.write("time_s," + ",".join(chans) + "\r\n")
        for r in data:
            f.write("%.6f," % ((r[0] - fr[0]) / FS) + ",".join("" if v is None else "%.8g" % v for v in (list(r) + [None] * 4)[1:4]) + "\r\n")
    extra.append((fname, str(lp)))
    rows.append(dict(
        dataset=DS, study=DS, experiment="", group="control", subject="S00", subject_excluded_by_authors=False,
        age="", sex="", handedness="", population="healthy", setting="lab",
        device="Optotrak Certus, one infrared marker on the dorsal hand", session=1, block=f"{arm} arm", trial=k, phase="test",
        time_s="", task="reaching", unit="recording", condition=f"{arm} arm", pert_type="none", pert_value=0, pert_unit="",
        feedback="normal", excluded_by_authors=False, recording=n_rec, samples_file=fname,
        signal_kind="3D position of one infrared marker on the dorsal surface of the hand", channels=";".join(chans),
        channel_units=";".join(["mm"] * 3), sample_rate_hz=FS, n_samples=len(data), duration_s=round((len(data) - 1) / FS, 6),
        events="", processing="none by the authors in the shared file (frame numbers are not consecutive: the sheet keeps the frames "
                               "between the first and the last of the ten reaches only)" if not all(b - a == 1 for a, b in zip(fr, fr[1:])) else "none",
        trial_info=f"moving_arm={arm}"))
# frame numbers of the control are those of a longer session, so time_s is relative to the first listed frame
t_ctrl = [r for r in rows if r["subject"] == "S00"]

# ---------------------------------------------------------------- checks
mild = [p for p in part.values() if p["severity"] == "mild"]
sev = [p for p in part.values() if p["severity"] == "severe"]
S = "Methods, Participants and Tables 1 and 2"
ck.published("participants with mild impairment", len(mild), 15, 0, S)
ck.published("participants with severe impairment", len(sev), 14, 0, S)
ck.published("mean age, mild group (years)", st.mean(p["age"] for p in mild), 61.8, 0.05, "Results, group comparison (Table 1)")
ck.published("SD of age, mild group", st.stdev(p["age"] for p in mild), 9.8, 0.05, "Results, group comparison (Table 1)")
ck.published("mean age, severe group (years)", st.mean(p["age"] for p in sev), 62.14, 0.01, "Results, group comparison (Table 2)")
ck.published("SD of age, severe group", st.stdev(p["age"] for p in sev), 9.3, 0.05, "Results, group comparison (Table 2)")
ck.published("mean time since stroke, severe (months)", st.mean(p["months"] for p in sev), 62, 0.5, "Results, group comparison (Table 2)")
mild_months = st.mean(p["months"] for p in mild)
mild_right = sum(p["paretic"] == "right" for p in mild)
ck.published("mean maximum reach, mild (cm)", st.mean(p["max_reach_cm"] for p in mild), 35.58, 0.03, "Results, group comparison")
ck.published("mean maximum reach, severe (cm)", st.mean(p["max_reach_cm"] for p in sev), 16.86, 0.01, "Results, group comparison")
ck.published("SD of maximum reach, severe (cm)", st.stdev(p["max_reach_cm"] for p in sev), 12.0, 0.05, "Results, group comparison")
ck.published("severe group: right paretic arms", sum(p["paretic"] == "right" for p in sev), 6, 0, "Results, group comparison")
ck.published("mild group: paretic arm is the dominant arm", sum(p["paretic"] == p["dominant"] for p in mild), 9, 0, "Results, group comparison")
ck.published("severe group: paretic arm is the dominant arm", sum(p["paretic"] == p["dominant"] for p in sev), 4, 0, "Results, group comparison")
ck.published("men in the mild group", sum(p["sex"] == "M" for p in mild), 11, 0, "Table 1")
ck.published("men in the severe group", sum(p["sex"] == "M" for p in sev), 7, 0, "Table 2")
ck.internal("the two sheets of the deposit agree on paretic and dominant side for all participants except S42 and S47, where 'Further Data' "
            "(used) and 'Input' differ", side_conflicts == ["S42", "S47"], str(side_conflicts))
ck.internal("every one of the 29 stroke participants has recordings, and all are sampled at 300 Hz",
            len(rec_per) == 29 and all(r["sample_rate_hz"] == 300 for r in rows), f"{n_rec} recordings")
ck.internal("frame numbers are consecutive within every stroke-participant recording", frames_ok)
ck.internal("recordings: 2 control + stroke recordings; 4 per participant except where the deposit lacks the second set",
            n_rec == len(rows) and all(1 <= v <= 4 for v in rec_per.values()), str(sorted(rec_per.values())))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == DS)["papers"][0]
meta = dict(
    study=DS, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Harris-Love (George Mason University / MedStar National Rehabilitation Hospital)",
    data_url="https://zenodo.org/records/7587404", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="each recording's samples were written to a CSV with a time column (seconds from the first sample, 300 Hz) and the marker "
            "coordinates unchanged (mm); participant IDs became codes; sex, age and the other participant values come from the deposit's "
            "'Input to DataProcessing' workbook.",
    unit_note="recording (a set of ten timed forward reaches)",
    how=[
        "One row is one recording of the simple forward-reaching test: ten reaches to a target at 80% of the participant's maximum reach "
        "after random 'Go' signals, with an Optotrak Certus recording the 3D positions of two infrared markers on the backs of the hands at "
        "300 Hz (mm). Marker 1 is the left hand and marker 2 the right hand (as in the authors' script); condition says which arm "
        "reached, and trial_info gives the moving arm and the paretic side.",
        "Each stroke participant has up to four recordings: the paretic and the non-paretic arm, each in two sets on two separate days "
        "(session 1 and 2). The deposit lacks the second set of both arms for four participants. Empty cells are frames where a marker was not "
        "visible. The reaches inside a recording are not segmented; the authors' scripts find them with participant-specific peak settings.",
        "For four participants the deposit holds a long (_Original) and a shortened (.xls) file of each paretic recording; the long files "
        "are converted. One healthy control (group control) has one marker per arm. The passive hand-path (PIH) calibration recordings and "
        "the control's 8-marker trial sheets are not included.",
    ],
    differences=[
        "For two participants (S42 and S47 in the deposit) the deposit's two spreadsheets disagree on the paretic and dominant side; the 'Further Data' sheet "
        "(the one that also holds time since stroke) was used.",
        f"The paper reports a mean time since stroke of 39 months for the mild group; the deposit gives {mild_months:.1f} (the paper's Table 1 lists 8 months for the participant "
        "whom the deposit lists at 51). The paper reports 7 right and 8 left paretic arms in the mild group; the deposit's 'Further Data' sheet gives "
        f"{mild_right} right. The severe-group values and all other group numbers reproduce.",
        "The Zenodo record 7587404 is titled 'Informing upper extremity rehabilitation through endpoint kinematics of the functional reaching movement "
        "after stroke' (Jan 2023, CC BY 4.0, with the same participants); the PeerJ paper's own data statement points to a later record, "
        "10.5281/zenodo.7968664 (May 2023, 'Upper extremity kinematics: Development of a quantitative measure of impairment severity and dissimilarity "
        "after stroke', CC BY 4.0). The PeerJ paper is cited here because the groups, ages and reach capacities match it.",
        "One file header in S53 states 166895 frames but the sheet holds 45158 rows; the rows present are converted.",
    ],
)
pdf = [dict(person=s, severity=p["severity"], paretic_side=p["paretic"], months_since_stroke=p["months"],
            max_paretic_reach_cm=p["max_reach_cm"]) for s, p in part.items()]
stdlib.release(DS, MODULE, rows, meta, ck, participants=pdf,
               participant_columns=[("severity", "category", "mild, severe", "Impairment group from the upper-extremity Fugl-Meyer score (mild 38-66, severe 0-37)."),
                                    ("paretic_side", "category", "left, right", "Side of the stroke-affected arm."),
                                    ("months_since_stroke", "number", "months", "Time since stroke."),
                                    ("max_paretic_reach_cm", "number", "cm", "Maximum reach of the paretic arm.")],
               extra_files=extra)
