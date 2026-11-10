"""Varghese et al. (2024), npj Parkinson's Disease 10(1):9: the PADS (Parkinson's Disease Smartwatch) dataset.
469 participants (Parkinson's disease, differential diagnoses, healthy controls) wore one smartwatch on each wrist during the 11 steps of
a short neurological assessment; every step of every wrist is one accelerometer + gyroscope recording (about 100 Hz).
One row per recording (participant x step x wrist); the samples of each recording are a CSV inside the ZIP (samples/...).

Run from the site folder: python3 -I tools/std/convert/Varghese2024_npjParkinsonsDis.py
Raw file (untrusted data, read with zipfile only, never extracted or run), raw_extra/Varghese2024_npjParkinsonsDis/:
  pads-1.0.0.zip   the PhysioNet project (https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/, open access):
    movement/observation_NNN.json      per participant: device (Apple Watch model), sampling rate and, for each of the 11 steps
                                       ("record_name", in the order of the paper's Table 4), the number of rows and the two wrists'
                                       file names, channel names and units
    movement/timeseries/NNN_<Step>_<LeftWrist|RightWrist>.txt   no header; 7 comma-separated columns with 10 decimals:
                                       Time (s), Accelerometer_X/Y/Z (g), Gyroscope_X/Y/Z (rad/s)
    patients/patient_NNN.json          condition, comment, age, age at diagnosis, height, weight, gender, handedness, kinship, alcohol
    questionnaire/questionnaire_response_NNN.json   30 yes/no items of the non-motor symptoms questionnaire (NMS)
    LICENSE.txt, SHA256SUMS.txt        license text and the hash of every file (all verified here)
    preprocessed/ (the authors' machine-learning input, derived) and scripts/ (their code) are not converted.
  SHA256SUMS.txt   as downloaded separately.
"""
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Varghese2024_npjParkinsonsDis"
SRC = "Varghese et al. 2024, npj Parkinsons Dis"
RAW = stdlib.raw_dir(ID)
ZIPNAME = "pads-1.0.0.zip"
HEAD = "time_s,Accelerometer_X,Accelerometer_Y,Accelerometer_Z,Gyroscope_X,Gyroscope_Y,Gyroscope_Z\r\n"
CHANNELS = ["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z", "Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z"]
UNITS = ["g", "g", "g", "rad/s", "rad/s", "rad/s"]

# The paper's Table 4: step label, the deposit's step name (record_name), duration (s), what the participant did
STEPS = [
    ("1a", "Relaxed", 20, "resting with closed eyes while sitting"),
    ("1b", "RelaxedTask", 20, "resting while calculating serial sevens"),
    ("2", "StretchHold", 10, "lift and extend the arms"),
    ("3", "LiftHold", 10, "remain with the arms lifted"),
    ("4", "HoldWeight", 10, "hold a one-kilogram weight in each hand"),
    ("5", "PointFinger", 10, "point the index finger to the examiner's lifted hand"),
    ("6", "DrinkGlas", 10, "drink from a glass"),
    ("7", "CrossArms", 10, "cross and extend both arms"),
    ("8", "TouchIndex", 10, "bring both index fingers to each other"),
    ("9", "TouchNose", 10, "tap the own nose with the index finger"),
    ("10", "Entrainment", 20, "stomp in the pace the examiner sets, arms extended"),
]
STEP_LABEL = {s[1]: s[0] for s in STEPS}
DROPPED = {"LiftHold", "PointFinger", "TouchIndex"}  # steps 3, 5, 8: removed from the authors' machine-learning analysis (Methods)

# patient "condition" -> (group = the paper's class, population, key in Table 3)
CONDITION = {
    "Parkinson's": ("Parkinson's disease (PD)", "Parkinson's disease", "PD"),
    "Healthy": ("Healthy controls (HC)", "healthy", "HC"),
    "Essential Tremor": ("Differential diagnoses (DD)", "Essential Tremor", "Essential Tremor"),
    "Atypical Parkinsonism": ("Differential diagnoses (DD)", "Atypical Parkinsonism", "Atypical Parkinsonism"),
    "Multiple Sclerosis": ("Differential diagnoses (DD)", "Multiple Sclerosis", "Multiple Sclerosis"),
    "Other Movement Disorders": ("Differential diagnoses (DD)", "Other Movement Disorders", "Other"),
}

# The paper's Table 3 (participant samples), as printed: n, male, female, then mean and SD of age, age at diagnosis, height, weight
T3 = {
    "PD": (276, 195, 81, ("65.4", "9.6"), ("58.26", "10.62"), ("175.71", "9.77"), ("83.78", "18.1")),
    "HC": (79, 29, 50, ("62.9", "12.5"), None, ("170.48", "15.63"), ("78.75", "16.72")),
    "DD": (114, 57, 57, ("62.4", "11.5"), ("52.96", "16.52"), ("173.04", "8.73"), ("79.54", "17.84")),
    "Atypical Parkinsonism": (15, 8, 7, ("66.0", "7.96"), ("62.6", "8.21"), ("168.47", "8.1"), ("79.13", "18.64")),
    "Essential Tremor": (28, 18, 10, ("66.18", "10.84"), ("46.96", "22.89"), ("173.36", "9.6"), ("84.68", "21.88")),
    "Multiple Sclerosis": (11, 7, 4, ("54.09", "7.92"), ("43.82", "13.04"), ("178.36", "10.69"), ("83.55", "16.18")),
    "Other": (60, 24, 36, ("61.32", "12.32"), ("55.02", "13.32"), ("173.07", "7.64"), ("76.52", "15.45")),
}
T3_NAME = {"PD": "Parkinson's disease", "HC": "healthy controls", "DD": "differential diagnoses",
           "Atypical Parkinsonism": "atypical parkinsonism", "Essential Tremor": "essential tremor",
           "Multiple Sclerosis": "multiple sclerosis", "Other": "other differential diagnoses"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def sample_csv(data):
    """The deposit's text file (no header, LF) as a CSV with a header line and CRLF line ends; the values are copied as text."""
    lines = data.decode("ascii").splitlines()
    return (HEAD + "\r\n".join(lines) + "\r\n").encode("ascii")


def main():
    ck = stdlib.Checks()
    src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
    zp = RAW / ZIPNAME
    ck.internal("the project ZIP matches the hash recorded when it was downloaded", stdlib.sha256(zp) == src[ZIPNAME]["sha256"],
                f"{zp.stat().st_size} bytes")
    z = zipfile.ZipFile(zp)
    names = [n for n in z.namelist() if not n.endswith("/")]
    tops = {n.split("/")[0] for n in names}
    prefix = (next(iter(tops)) + "/") if len(tops) == 1 and all("/" in n for n in names) else ""
    members = {n[len(prefix):]: n for n in names}

    def read(rel):
        return z.read(members[rel])

    sums = {}
    for line in open(RAW / "SHA256SUMS.txt", encoding="utf-8"):
        line = line.rstrip("\r\n")
        if line:
            h, p = line.split(None, 1)
            sums[p.strip().lstrip("*")] = h
    ck.internal("the ZIP holds the 12,689 files listed in SHA256SUMS.txt plus SHA256SUMS.txt itself, which is the file downloaded separately",
                set(sums) <= set(members) and set(members) - set(sums) == {"SHA256SUMS.txt"} and len(sums) == 12689
                and read("SHA256SUMS.txt") == open(RAW / "SHA256SUMS.txt", "rb").read(), f"{len(members)} files")
    # ---- every file except the time series (those are hashed while they are read below)
    bad = [rel for rel, h in sums.items() if not rel.startswith("movement/timeseries/") and sha(read(rel)) != h]
    ck.internal("every file except the time series matches SHA256SUMS.txt", not bad, f"{len(sums) - 10318} files; mismatches {bad[:3]}")
    lic = read("LICENSE.txt").decode("utf-8", "replace")

    # ---- participants, sessions, time series (pass 1: read, verify, describe; the samples are copied in pass 2)
    ids = sorted(int(re.fullmatch(r"patients/patient_(\d{3})\.json", n).group(1)) for n in members if n.startswith("patients/"))
    ck.internal("patients, observations and questionnaires: one file per participant, numbered 1-469",
                ids == list(range(1, 470))
                and sorted(n for n in members if n.startswith("movement/observation_")) == [f"movement/observation_{i:03d}.json" for i in ids]
                and sorted(n for n in members if n.startswith("questionnaire/")) == [f"questionnaire/questionnaire_response_{i:03d}.json" for i in ids])
    q_ref = None
    rows, recs, people = [], [], []
    stat = dict(n_files=0, hash_bad=0, nonnum=0, nonmono=0, first_not0=0, rows_ne_json=0, bad_cols=0, header=0)
    dts, accmag = [], []
    layout_ref = None
    mn, mx = np.full(6, np.inf), np.full(6, -np.inf)
    for k, n in enumerate(ids, start=1):
        nnn = f"{n:03d}"
        pat = json.loads(read(f"patients/patient_{nnn}.json"))
        obs = json.loads(read(f"movement/observation_{nnn}.json"))
        qr = json.loads(read(f"questionnaire/questionnaire_response_{nnn}.json"))
        assert pat["id"] == nnn and obs["subject_id"] == nnn and qr["subject_id"] == nnn
        assert obs["sampling_rate"] == 100 and obs["bits"] == 32 and obs["data_type"] == "float"
        layout = [(x["record_name"], x["rows"], tuple(r["device_location"] for r in x["records"])) for x in obs["session"]]
        layout_ref = layout_ref or [(s_[1], 2048 if s_[2] == 20 else 1024, ("LeftWrist", "RightWrist")) for s_ in STEPS]
        assert layout == layout_ref, "step order, rows or wrists differ from the paper's Table 4"
        qitems = [(it["link_id"], it["text"]) for it in qr["item"]]
        q_ref = q_ref or qitems
        assert qitems == q_ref and len(qitems) == 30 and all(isinstance(it["answer"], bool) for it in qr["item"])
        scode, pcode = f"{ID}_S{k:04d}", f"{ID}_P{k:04d}"  # stdlib assigns the same codes (order of first appearance)
        group, population, t3key = CONDITION[pat["condition"]]
        dx = pat["age_at_diagnosis"]
        people.append(dict(
            person=pcode, key=t3key, group=group, age=pat["age"], dx_raw=dx, dx=None if (dx == 0 or pat["condition"] == "Healthy") else dx,
            height=pat["height"], weight=pat["weight"], gender=pat["gender"], comment=pat["disease_comment"],
            kin=pat["appearance_in_kinship"], kin1=pat["appearance_in_first_grade_kinship"], alcohol=pat["effect_of_alcohol_on_tremor"],
            watch=obs["device_id"], nms={f"nms_{lid}": it["answer"] for (lid, _), it in zip(qitems, qr["item"])}))
        rec_no = 0
        for sess in obs["session"]:
            step = sess["record_name"]
            for r in sess["records"]:
                rel = "movement/" + r["file_name"]
                assert r["channels"] == ["Time"] + CHANNELS and r["units"] == ["s"] + UNITS
                data = read(rel)
                stat["n_files"] += 1
                stat["hash_bad"] += sha(data) != sums[rel]
                txt = data.decode("ascii")
                stat["header"] += bool(re.search(r"[A-DF-Za-df-z]", txt.split("\n", 1)[0]))
                try:
                    a = pd.read_csv(io.BytesIO(data), header=None, dtype=np.float64).to_numpy()
                except Exception:  # noqa: BLE001
                    stat["nonnum"] += 1
                    continue
                stat["bad_cols"] += a.shape[1] != 7
                stat["nonnum"] += not np.isfinite(a).all()
                stat["rows_ne_json"] += a.shape[0] != sess["rows"]
                t = a[:, 0]
                stat["first_not0"] += t[0] != 0
                stat["nonmono"] += not (np.diff(t) > 0).all()
                dts.append(float(np.median(np.diff(t))))
                mn, mx = np.minimum(mn, a[:, 1:].min(axis=0)), np.maximum(mx, a[:, 1:].max(axis=0))
                accmag.append(float(np.median(np.sqrt((a[:, 1:4] ** 2).sum(axis=1)))))
                rec_no += 1
                wrist = "left" if r["device_location"] == "LeftWrist" else "right"
                assert r["device_location"] in ("LeftWrist", "RightWrist")
                arc = f"samples/{pcode}_{step}_{r['device_location']}.csv"
                rows.append(dict(
                    dataset=ID, study=SRC, experiment="PADS smartwatch assessment (11 steps, a watch on each wrist)", group=group,
                    subject=scode, person=pcode, subject_excluded_by_authors=False, age=pat["age"],
                    sex={"male": "M", "female": "F"}[pat["gender"]], handedness={"right": "right", "left": "left"}[pat["handedness"]],
                    population=population, setting="clinic", device=f"{obs['device_id']}, {wrist} wrist", session=1,
                    block=f"step {STEP_LABEL[step]}", trial=rec_no, phase="test", time_s=None, task="finger movements", unit="recording",
                    condition=step, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                    excluded_by_authors=step in DROPPED,
                    recording=rec_no, samples_file=arc, signal_kind="wrist accelerometer and gyroscope of a smartwatch (3 axes each)",
                    channels=";".join(CHANNELS), channel_units=";".join(UNITS), sample_rate_hz=100, n_samples=int(a.shape[0]),
                    duration_s=float(t[-1] - t[0]), events="",
                    processing="none by OpenMotor; the authors shifted the time stamps to start at 0 s (PhysioNet description)"))
                recs.append((arc, rel))
    df = pd.DataFrame(rows)
    pe = pd.DataFrame(people)

    # ---- internal checks
    ck.internal("every time-series file matches SHA256SUMS.txt (10,318 files)", stat["hash_bad"] == 0 and stat["n_files"] == 10318,
                f"{stat['n_files']} files")
    ck.internal("the observation files list exactly the time-series files of the ZIP",
                sorted(r[1] for r in recs) == sorted(n for n in members if n.startswith("movement/timeseries/")))
    ck.internal("every recording has 7 numeric columns (Time, 3 accelerometer, 3 gyroscope), no header line and no missing value",
                stat["bad_cols"] == 0 and stat["nonnum"] == 0 and stat["header"] == 0, str({k: int(v) for k, v in stat.items() if k != 'n_files'}))
    ck.internal("the number of rows of every file is the one the observation file states for that step", stat["rows_ne_json"] == 0)
    ck.internal("the time column starts at 0 and increases in every recording", stat["first_not0"] == 0 and stat["nonmono"] == 0)
    n_by = df.groupby("condition").n_samples.agg(["min", "max"])
    long_steps = sorted(n_by.index[n_by["min"] == 2048])
    ck.internal("steps 1a, 1b and 10 (Relaxed, RelaxedTask, Entrainment) have 2048 samples in every recording; the other eight 1024",
                long_steps == ["Entrainment", "Relaxed", "RelaxedTask"] and bool((n_by["max"] == n_by["min"]).all())
                and set(n_by["min"]) == {1024, 2048}, str({k: int(v) for k, v in n_by["min"].items()}))
    ck.internal("median sampling interval of every recording is within 9.8-10.2 ms", min(dts) > 0.0098 and max(dts) < 0.0102,
                f"{min(dts) * 1000:.3f}-{max(dts) * 1000:.3f} ms")
    ck.internal("two recordings (left and right wrist) for each participant and step", bool(
        df.groupby(["person", "condition"]).device.nunique().eq(2).all()) and bool(df.groupby(["person", "condition"]).size().eq(2).all()))
    ck.internal("each participant has the same watch model on both wrists", bool(df.groupby("person").device.apply(
        lambda s: s.str.replace(r", (left|right) wrist", "", regex=True).nunique()).eq(1).all()))
    ck.internal("the 30 questionnaire items and their wording are identical for all participants", True, f"{len(q_ref)} items")
    ck.internal("accelerometer values at rest are far below 1 g (no gravity component)", float(np.median(accmag)) < 0.1,
                f"median of the per-recording median magnitude {np.median(accmag):.4f} g")
    ck.internal("the age at diagnosis is 0 (unknown) for exactly 19 participants and equals the age for every healthy control",
                int((pe.dx_raw == 0).sum()) == 19 and bool((pe[pe.key == "HC"].dx_raw == pe[pe.key == "HC"].age).all()),
                f"{int((pe.dx_raw == 0).sum())} zeros")

    # ---- published numbers: Methods and Table 3
    M = f"{SRC}, Methods, Study data"
    steps = df.groupby("person").condition.nunique()
    ck.published("participants in the dataset", pe.person.nunique(), 469, 0, f"{SRC}, Introduction and Methods, Study data (Table 3 total)")
    ck.published("measurement steps (participant x step)", len(df.groupby(["person", "condition"])), 5159, 0, "PhysioNet record abstract (5159 measurement steps of 469 individuals)")
    ck.published("assessment steps per participant (minimum)", steps.min(), 11, 0, M)
    ck.published("assessment steps per participant (maximum)", steps.max(), 11, 0, M)
    dur_step = df.groupby("condition").duration_s.mean()
    ck.published("assessment steps that took 20 s", int((dur_step > 15).sum()), 3, 0, M)
    ck.published("assessment steps that took 10 s", int((dur_step < 15).sum()), 8, 0, M)
    ck.published("smartwatches per participant (one on each wrist): recordings per participant and step", float(df.groupby(["person", "condition"]).size().mean()), 2, 0, M)
    ck.published("recordings (5159 measurement steps x 2 smartwatches)", len(df), 2 * 5159, 0, f"PhysioNet record abstract (5159 steps) and {M} (two watches)")
    short = ~df.condition.isin(["Relaxed", "RelaxedTask", "Entrainment"])
    ck.published("samples in a recording of a 10-s step (minimum)", df.n_samples[short].min(), 1024, 0, "PhysioNet record, Data description (1024 data points per file)")
    ck.published("samples in a recording of a 10-s step (maximum)", df.n_samples[short].max(), 1024, 0, "PhysioNet record, Data description (1024 data points per file)")
    ck.published("sampling frequency (Hz): 1 / median sampling interval over all recordings", 1.0 / float(np.median(dts)), 100, 0.1, M)
    seg = df.assign(seg=df.n_samples // 1024).groupby(["person", "device"]).seg.sum()
    ck.published("10-s time series per participant and wrist (20-s steps cut in two parts)", seg.min(), 14, 0, M)
    ck.published("10-s time series per participant and wrist (maximum)", seg.max(), 14, 0, M)
    chan = df.assign(c=(df.n_samples // 1024) * 6).groupby("person").c.sum()
    ck.published("channels of time series per participant (14 x 2 arms x 2 sensors x 3 axes)", chan.min(), 168, 0, M)
    ck.published("channels of time series per participant (maximum)", chan.max(), 168, 0, M)

    T = "Table 3"
    sel = {"PD": pe.key == "PD", "HC": pe.key == "HC", "DD": pe.group.str.startswith("Differential"),
           "Atypical Parkinsonism": pe.key == "Atypical Parkinsonism", "Essential Tremor": pe.key == "Essential Tremor",
           "Multiple Sclerosis": pe.key == "Multiple Sclerosis", "Other": pe.key == "Other"}
    fill_dx = pe.dx_raw.where(pe.dx_raw != 0, pe.age)  # Table 3 is reproduced when an unknown (0) age at diagnosis is the age
    tol = lambda g, col: 0.05 + 1e-9 if (g in ("PD", "HC", "DD") and col == "age") else 0.005 + 1e-9  # noqa: E731
    for g, (n_, male, female, age, dxs, height, weight) in T3.items():
        x = pe[sel[g]]
        w = T3_NAME[g]
        ck.published(f"{w}: participants", len(x), n_, 0, f"{SRC}, {T}")
        ck.published(f"{w}: men", int((x.gender == "male").sum()), male, 0, f"{SRC}, {T}")
        ck.published(f"{w}: women", int((x.gender == "female").sum()), female, 0, f"{SRC}, {T}")
        for label, col, vals in (("age (years)", x.age, age), ("height (cm)", x.height, height), ("weight (kg)", x.weight, weight)):
            c = "age" if label.startswith("age") else label
            ck.published(f"{w}: {label}, mean", col.mean(), float(vals[0]), tol(g, c), f"{SRC}, {T}")
            ck.published(f"{w}: {label}, SD", col.std(ddof=1), float(vals[1]), tol(g, c), f"{SRC}, {T}")
        if dxs:
            v = fill_dx[sel[g]]
            ck.published(f"{w}: age at diagnosis (years; unknown, 0 in the deposit, taken as the age), mean", v.mean(), float(dxs[0]), tol(g, "dx"), f"{SRC}, {T}")
            ck.published(f"{w}: age at diagnosis (years; unknown, 0 in the deposit, taken as the age), SD", v.std(ddof=1), float(dxs[1]), tol(g, "dx"), f"{SRC}, {T}")
    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)

    # ---- participants.csv
    nms_cols = [(f"nms_{lid}", "true/false", "", f"Non-motor symptoms questionnaire (NMS), item {lid}, answered yes (True) or no (False): {tx}")
                for lid, tx in q_ref]
    common = pe.comment.value_counts()
    keep = set(common[common >= 5].index)
    pcols = [
        ("age_at_diagnosis", "number", "years",
         "Age at diagnosis. Empty where the deposit has 0 (unknown) and for healthy controls (the deposit repeats their age); "
         "the paper's Table 3 treats an unknown age at diagnosis as the age (see README)."),
        ("height_cm", "number", "cm", "Height as in the deposit. One healthy control has 55, an entry error kept as shared (Table 3 includes it)."),
        ("weight_kg", "number", "kg", "Weight as in the deposit."),
        ("disease_comment", "text", "",
         "The deposit's short diagnosis comment (German clinical abbreviations: IPS idiopathic Parkinson syndrome, THS deep brain "
         "stimulation; '-' means none). Kept only where at least 5 participants have exactly this wording; individual comments, "
         "which describe single clinical histories, are left empty."),
        ("kinship_with_pd", "true/false", "", "Kinship with PD, from the questionnaire (the deposit's appearance_in_kinship)."),
        ("kinship_with_pd_first_grade", "true/false", "",
         "The same for first-grade kinship (the deposit's appearance_in_first_grade_kinship); empty where the deposit has null."),
        ("effect_of_alcohol_on_tremor", "category", "No effect, Improvement, Worsening, Unknown", "Effect of alcohol on the tremor, from the questionnaire."),
        ("watch_model", "category", "Apple Watch Series 3, Apple Watch Series 4", "Smartwatch model, from the deposit's observation file (both wrists)."),
    ] + nms_cols
    parts = []
    for p in people:
        d = dict(person=p["person"], age_at_diagnosis=p["dx"], height_cm=p["height"], weight_kg=p["weight"],
                 disease_comment=p["comment"] if p["comment"] in keep else None, kinship_with_pd=p["kin"],
                 kinship_with_pd_first_grade=p["kin1"], effect_of_alcohol_on_tremor=p["alcohol"], watch_model=p["watch"])
        d.update(p["nms"])
        parts.append(d)

    # ---- release
    s, paper = paper_meta()
    n_series3 = int((pe.watch == "Apple Watch Series 3").sum())
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"],
        license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0)",
        license_url="https://physionet.org/content/parkinsons-disease-smartwatch/view-license/1.0.0/",
        license_text=lic,
        changes=("Converted the 10,318 time-series text files into one CSV each (samples/...; a header line added, the values copied "
                 "unchanged, CRLF line ends) with one table row per recording; participant numbers replaced by codes; the patient "
                 "and questionnaire files made into participants.csv (an age at diagnosis of 0, which the deposit uses for unknown, "
                 "and the age that the deposit repeats for healthy controls are left empty; rare free-text diagnosis comments are left "
                 "out). Not converted: the authors' preprocessed machine-learning files and their scripts."),
        how=[
            "One row is one recording: one Apple Watch on the left or right wrist of one participant during one of the 11 steps of the "
            "PADS assessment, seated in an armchair in a movement-disorders outpatient clinic. condition is the deposit's step name and "
            "block the step number of the paper's Table 4: 1a Relaxed (resting, eyes closed, 20 s), 1b RelaxedTask (resting while "
            "calculating serial sevens, 20 s), 2 StretchHold (lift and extend the arms), 3 LiftHold (remain with the arms lifted), "
            "4 HoldWeight (hold a one-kilogram weight in each hand), 5 PointFinger (point the index finger to the examiner's hand), "
            "6 DrinkGlas (drink from a glass), 7 CrossArms (cross and extend both arms), 8 TouchIndex (bring both index fingers to "
            "each other), 9 TouchNose (tap the own nose with the index finger), 10 Entrainment (stomp in the examiner's pace, 20 s); "
            "steps 2-9 last 10 s. The two watches recorded the same step at once, so each step has two rows (left wrist first); "
            "device gives the watch model of that participant and the wrist, and trial and recording both number the participant's "
            "22 recordings in the order of the steps. time_s of the main table is empty: the recordings have no common clock.",
            "The folder samples/ of the ZIP holds one CSV per recording, named in samples_file, with one line per sample: time_s (the deposit's Time column in seconds, which the "
            "authors shifted to start at 0), then the three accelerometer axes in g and the three gyroscope axes in rad/s, copied "
            "unchanged from the deposit's text files (10 decimals; those files have no header line). Samples were not resampled or "
            "filtered. Steps 1a, 1b and 10 have 2048 samples (about 20.5 s), the other eight 1024 samples (about 10.2 s) at "
            f"about 100 Hz (the median sampling interval is {np.median(dts) * 1000:.2f} ms); duration_s is the last time stamp. The "
            f"acceleration values are small (median magnitude {np.median(accmag):.3f} g), so they do not contain the constant 1 g of "
            "gravity.",
            "group is the paper's class (Parkinson's disease, differential diagnoses, healthy controls) and population the deposit's "
            "diagnosis group: Parkinson's disease, healthy, or one of the differential diagnoses Essential Tremor, Atypical "
            "Parkinsonism, Multiple Sclerosis and Other Movement Disorders. excluded_by_authors is True for steps 3, 5 and 8 "
            "(LiftHold, PointFinger, TouchIndex), which the authors removed from their machine-learning analysis because they gave "
            "no additional value to the classification; their recordings are in the deposit and here. The authors' other "
            "preprocessing for machine learning (removal of the first 0.5 s, gravity correction by l1 trend filtering, cutting "
            "the 20-s recordings in two) is not applied.",
            "participants.csv adds, per participant, the age at diagnosis, height, weight, the deposit's short diagnosis comment "
            "(only where at least 5 participants share the wording), kinship with PD, the effect of alcohol on tremor, the watch "
            "model and the 30 yes/no answers of the non-motor symptoms questionnaire (NMS); participants_columns.csv gives the "
            "wording of every item. Study registration: ClinicalTrials.gov NCT03638479 (PhysioNet record).",
        ],
        unit_note="recording (one wrist of one participant during one assessment step)",
        differences=[
            "The paper reports 504 participants and 5,544 steps recorded originally (Introduction, Methods); the deposit holds the "
            "469 participants and 5,159 steps of the analysis set (Table 3), so the 35 left out (2 with missing data, 33 of the "
            "youngest controls removed for age matching) are not here.",
            "Table 3's age at diagnosis (Parkinson's disease 58.26, differential diagnoses 52.96, multiple sclerosis 43.82, other "
            "55.02 years) is reproduced only when the deposit's 0 (unknown, 19 participants) is replaced by the participant's age, "
            "which the paper does not state; dropping the zeros instead gives 58.08, 52.23, 40.67 and 54.14. The deposit itself "
            "repeats the age for all healthy controls, for whom Table 3 shows no value. participants.csv leaves both empty.",
            f"The paper and the PhysioNet page name the Apple Watch Series 4 only; the deposit's observation files list the Series 3 "
            f"for {n_series3} of the 469 participants, and device and watch_model follow the observation files.",
            "The PhysioNet description says each file holds 1024 samples (10.24 s); the three 20-s steps (1a, 1b, 10) have 2048 "
            "samples per file, as the paper's Table 4 durations imply.",
            "The deposit gives one participant a height of 55 cm (an entry error); it is kept as shared, and Table 3's healthy-control "
            "height SD (15.63) is reproduced only with it.",
        ],
    )
    out = stdlib.OUT / f"{ID}.zip"

    def extra():
        for arc, rel in recs:
            yield arc, sample_csv(read(rel))

    rec = stdlib.release(ID, "timeseries", rows, meta, ck, participants=parts, participant_columns=pcols, extra_files=extra())
    size = out.stat().st_size
    print(f"ZIP {size / 1e6:.0f} MB; accelerometer range {np.round(mn[:3], 3).tolist()} .. {np.round(mx[:3], 3).tolist()} g; "
          f"gyroscope range {np.round(mn[3:], 3).tolist()} .. {np.round(mx[3:], 3).tolist()} rad/s")
    if size > 1.9e9:
        out.unlink()
        (stdlib.REC / f"{ID}.json").unlink()
        stdlib.skip(ID, "too large", f"the converted ZIP would be {size / 1e9:.2f} GB (limit 1.9 GB)")
        return rec

    # ---- verify the written ZIP: sample files exist, are named by the final codes, and hold the deposit's values
    with zipfile.ZipFile(out) as zo:
        zn = set(zo.namelist())
        main = pd.read_csv(io.BytesIO(zo.read(f"{ID}/{ID}.csv")), dtype=str, keep_default_na=False)
        ok_files = all(f"{ID}/{a}" in zn for a in main.samples_file)
        ok_names = all(a == f"samples/{p}_{c}_{'Left' if 'left' in d else 'Right'}Wrist.csv"
                       for a, p, c, d in zip(main.samples_file, main.person, main.condition, main.device))
        ok_n = len([n for n in zn if n.startswith(f"{ID}/samples/")]) == len(main) == 10318
        pick = np.random.RandomState(0).choice(len(main), 300, replace=False)
        same = 0
        for i in pick:
            got = zo.read(f"{ID}/{main.samples_file[i]}").decode("ascii").split("\r\n")
            want = read(dict(recs)[main.samples_file[i]]).decode("ascii").splitlines()
            same += got[0] == HEAD.strip() and got[1:-1] == want and got[-1] == ""
        ok_same = same == 300
    print("post-release verification:", dict(files_exist=ok_files, names_follow_codes=ok_names, n_samples_files=ok_n, values_equal_in_300=ok_same))
    assert ok_files and ok_names and ok_n and ok_same
    return rec


if __name__ == "__main__":
    main()
