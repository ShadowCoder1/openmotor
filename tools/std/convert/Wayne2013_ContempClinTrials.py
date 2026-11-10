"""Tai Chi, Physiological Complexity, and Healthy Aging - Gait (PhysioNet taichidb 1.0.2; Wayne, Gow, Hausdorff, Peng, Lipsitz, Ahn,
Novak & Manor, doi 10.13026/gq9q-rr81), the data of the randomized trial described by Wayne et al. (2013) Contemp Clin Trials 34(1):21-34:
older adults walking at their preferred speed for 10 minutes (single task) and 90 seconds while subtracting serial 3s (dual task), wearing
foot switches under both feet, at baseline, 3 and 6 months (Tai Chi-naive adults) or once (Tai Chi experts). One row per recording; the
foot-switch signals (1500 Hz) are CSV files inside the ZIP. The surface EMG channels of the same recordings (anterior tibialis and lateral
gastrocnemius of both legs) are not converted: at 1500 Hz they would take several GB as CSV.

Raw file: raw_extra/Wayne2013_ContempClinTrials/taichidb.zip (PhysioNet get-zip of version 1.0.2, ODC-By 1.0): the WFDB .hea headers (text)
and .dat files (16-bit signals, read as plain integers with numpy and scaled with the header's gain and baseline), TCPCHA_Subjects.csv.
Run from the site folder: python3 -I tools/std/convert/Wayne2013_ContempClinTrials.py"""
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wayne2013_ContempClinTrials"
MODULE = "timeseries"
GROUP = {"Control": "Control (usual care)", "Tai-Chi": "Tai Chi training", "Master": "Tai Chi experts"}
DEVICE = ("Wireless force-sensitive resistor foot switches under the toes and heels (Mega Elektronika ME6000 acquisition, 1500 Hz) and "
          "Noraxon surface EMG; walking on a hallway loop at preferred speed")


def main():
    raw = stdlib.raw_dir(ID)
    z = zipfile.ZipFile(raw / "taichidb.zip")
    names = z.namelist()
    root = names[0].split("/")[0] + "/"
    subj = pd.read_csv(io.BytesIO(z.read(root + "TCPCHA_Subjects.csv")), encoding="utf-8-sig")
    subj.columns = ["sid", "sex", "age", "visit", "group", "random", "height_in", "activity_text", "activity", "education", "bmi",
                    "hallway", "trail_a", "trail_b", "trail_ratio", "animals", "supermarket", "category", "laps", "last_lap_m",
                    "distance_m", "time_s"]
    subj["group"] = subj.group.str.strip()
    subj["visit"] = subj.visit.str.strip().str.upper()
    subj = subj.set_index(["sid", "visit"])
    assert subj.index.is_unique
    heas = sorted(n for n in names if n.endswith(".hea"))
    recs = []
    for n in heas:
        m = re.fullmatch(r"(Single|Dual)-task/(S\d+)_(ST|DT)_(V\d|v\d|master)(?:_P(\d))?\.hea", n[len(root):])
        assert m, n
        task, sid, kind, vis, part = m.groups()
        vis = "V1" if vis == "master" else vis.upper()
        recs.append((sid, vis, 0 if kind == "ST" else 1, int(part or 0), n))
    recs.sort()
    code = {sid: f"p{i + 1:03d}" for i, sid in enumerate(sorted({r[0] for r in recs}))}
    tmp = Path("tmp2_" + ID)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rows, extra, seq, dur, nometa = [], [], {}, {"ST": [], "DT": []}, set()
    fs_seen, unit_seen = set(), set()
    try:
        for sid, vis, kind, part, hn in recs:
            lines = z.read(hn).decode("ascii").splitlines()
            h0 = lines[0].split()
            ns, fs, nsamp = int(h0[1]), float(h0[2]), int(h0[3])
            assert ns == 6 and len(lines) == 7, hn
            sig = []
            for l in lines[1:]:
                f = l.split()
                assert f[1] == "16", l
                mm = re.fullmatch(r"([-\d.eE+]+)\(([-\d]+)\)/(\w+)", f[2])
                assert mm and f[3] == "0" and f[4] == "0", l
                sig.append((" ".join(f[8:]), float(mm.group(1)), int(mm.group(2)), mm.group(3), f[0]))
            assert len({s[4] for s in sig}) == 1
            dat = z.read(hn.replace(".hea", ".dat"))
            a = np.frombuffer(dat, dtype="<i2")
            assert a.size == ns * nsamp, (hn, a.size, ns * nsamp)
            a = a.reshape(nsamp, ns)
            idx = [i for i, s in enumerate(sig) if s[0].endswith("FOOT")]
            assert [sig[i][0] for i in idx] == ["RT FOOT", "LT FOOT"], sig
            assert all(sig[i][3] == "V" for i in idx)
            fs_seen.add(fs)
            cols = {"time_s": np.arange(nsamp) / fs}
            for i, label in zip(idx, ("foot_switch_right", "foot_switch_left")):
                cols[label] = (a[:, i].astype(float) - sig[i][2]) / sig[i][1]
            kindname = "ST" if kind == 0 else "DT"
            tag = f"{code[sid]}_{kindname}_{vis}" + (f"_P{part}" if part else "")
            fname = f"{tag}.csv"
            pd.DataFrame(cols).to_csv(tmp / fname, index=False, float_format="%.6f", lineterminator="\r\n")
            extra.append((f"samples/{fname}", str(tmp / fname)))
            dur[kindname].append(nsamp / fs)
            s = subj.loc[(sid, vis)] if (sid, vis) in subj.index else None
            nometa.add(sid) if s is None else None
            seq[sid] = seq.get(sid, 0) + 1
            visit_no = int(vis[1])
            task_txt = "single task: walking at preferred speed" if kind == 0 else "dual task: walking while subtracting serial 3s aloud"
            rows.append(dict(
                dataset=ID, study=ID, experiment="", group=(GROUP[s.group] if s is not None else ""), subject=sid, person=sid, subject_excluded_by_authors=False,
                age=(float(s.age) if s is not None else np.nan), sex=(s.sex if s is not None else ""), population="healthy older adults" + (" (Tai Chi experts)" if s is not None and s.group == "Master" else ""),
                setting="lab", device=DEVICE, session=visit_no, block=("baseline" if s is not None and s.group == "Master" else {1: "baseline", 2: "3 months", 3: "6 months"}[visit_no]),
                trial=seq[sid], phase="test", time_s=np.nan, task="walking", unit="recording", condition=task_txt,
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False, handedness="",
                recording=seq[sid], samples_file=f"samples/{fname}",
                signal_kind="foot switches under the right and left foot (force-sensitive resistors under toes and heels)",
                channels="foot_switch_right;foot_switch_left", channel_units="V;V", sample_rate_hz=fs, n_samples=nsamp,
                duration_s=round(nsamp / fs, 4), events="",
                processing=("none beyond the WFDB scaling: volts = (integer sample - baseline) / gain with the header's baseline (%d) and gain (%g) "
                            "for the right switch and baseline (%d), gain (%g) for the left" % (sig[idx[0]][2], sig[idx[0]][1], sig[idx[1]][2], sig[idx[1]][1]))
                + ("; recording part %d (the walk had to be recorded in two segments)" % part if part else "")))
        df = pd.DataFrame(rows)
        df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"])
        first = subj.reset_index()
        first = first[first.visit == "V1"]
        pp = pd.DataFrame(dict(person=first.sid, height_in=first.height_in, bmi=first.bmi, education_years=first.education,
                               trial_arm=first.random.str.strip()))
        pp = pp[pp.person.isin(df.subject)]
        pcols = [("height_in", "number", "inches", "Body height at the baseline visit."),
                 ("bmi", "number", "kg/m2", "Body mass index at the baseline visit."),
                 ("education_years", "number", "years", "Years of education reported at the baseline visit; empty if not reported."),
                 ("trial_arm", "category", "Random, Master",
                  "Random = Tai Chi-naive participant in the randomized trial; Master = Tai Chi expert in the observational comparison group.")]
        ck = stdlib.Checks()
        vis1 = subj.reset_index()
        vis1 = vis1[vis1.visit == "V1"]
        naive = vis1[vis1.group != "Master"]
        ck.published("Tai Chi-naive participants in the randomized trial", naive.sid.nunique(), 60, 0, "Design paper abstract (60 Tai Chi-naive older adults) and PhysioNet description")
        ck.published("Tai Chi expert comparison group", vis1[vis1.group == "Master"].sid.nunique(), 27, 0, "PhysioNet project description (27 healthy Tai Chi experts)")
        ck.published("youngest participant is at least 50 years", float((vis1.age >= 50).all()), 1, 0, "Design paper abstract (aged 50-79)")
        ck.published("oldest participant is at most 79 years", float((vis1.age <= 79).all()), 1, 0, "Design paper abstract (aged 50-79)")
        ck.published("dual-task recording length (s, median)", np.median(dur["DT"]), 90, 3, "PhysioNet project description (90 seconds)")
        ck.published("sampling rate (Hz)", list(fs_seen)[0], 1500, 0, "PhysioNet project description (1500 Hz)")
        ck.internal("one sampling rate in all recordings", len(fs_seen) == 1, str(fs_seen))
        ck.internal("participants with recordings but no subjects-table row: %d" % len(nometa), len(nometa) <= 1)
        ck.internal("all %d recordings of the deposit converted, each with a sample file" % len(recs), len(df) == len(recs) == len(extra), str(len(df)))
        ck.internal("single-task recordings last %.0f to %.0f s (median %.0f s)" % (min(dur["ST"]), max(dur["ST"]), np.median(dur["ST"])), min(dur["ST"]) > 60)
        meta = dict(
            paper=dict(authors="Wayne PM, Manor B, Novak V, Costa MD, Hausdorff JM, Goldberger AL, Ahn AC, Yeh GY, Peng CK, Lough M, Davis RB, Quilty MT, Lipsitz LA",
                       year=2013, title="A systems biology approach to studying Tai Chi, physiological complexity and healthy aging: design and rationale of a pragmatic randomized controlled trial",
                       venue="Contemp Clin Trials 34(1):21-34", doi="10.1016/j.cct.2012.09.006"),
            lab="Wayne / Manor (Harvard / Hebrew SeniorLife)",
            data_url="https://physionet.org/content/taichidb/1.0.2/ (Wayne, Gow, Hausdorff, Peng, Lipsitz, Ahn, Novak, Manor. Tai Chi, Physiological Complexity, and Healthy Aging - Gait. PhysioNet, 2021, doi 10.13026/gq9q-rr81)",
            license="ODC-By 1.0 (Open Data Commons Attribution License v1.0)", license_url="https://opendatacommons.org/licenses/by/1-0/",
            changes=("Converted each WFDB recording (.hea header + .dat 16-bit samples) into a CSV of the two foot-switch channels (right, left) in volts, "
                     "scaled with the gain and baseline of the header; time_s was added as sample / 1500. The four surface EMG channels (tibialis anterior and "
                     "lateral gastrocnemius, both legs) of every recording are not included because they are about 3 GB as CSV. Age and sex are from "
                     "TCPCHA_Subjects.csv for the visit of each recording; the cognitive and walking test columns of that file are not included."),
            how=["One row is one walking recording of one participant at one visit: 10 minutes at preferred speed (single task) or 90 seconds while "
                 "subtracting serial 3s aloud (dual task), with the right and left foot-switch signals sampled at 1500 Hz in the CSV named in samples_file. "
                 "The foot switches mark the foot contacts, from which the authors derived stride intervals; no stride table is shared, so none is made here.",
                 "Tai Chi-naive participants (groups Control and Tai Chi training, randomized) came for a baseline, a 3-month and a 6-month visit (session 1 to 3); the 27 "
                 "Tai Chi experts were tested once (session 1, baseline). Walks that had to be recorded in two segments are two rows (processing says which part).",
                 "The exact signal of each foot switch (one channel per foot) is as the authors recorded it; values are close to two or three levels, so contact "
                 "timing has to be found by thresholding. Participant codes follow the order of first appearance, not the deposit's subject numbers."],
            unit_note="walking recording (single or dual task)",
            differences=["Recordings of one subject number (6 recordings, three visits) have no row in TCPCHA_Subjects.csv, so group, age and sex are empty "
                         "for that participant, while the table has one subject number that has no recordings (a Control participant). They may be "
                         "the same person with a mistyped number, but this was not assumed and the two were not merged; 87 participants have recordings. "
                         "The 60 Tai Chi-naive and 27 expert participants counted in the checks are those of the subjects table. " +
                         "The design paper (2013) describes the planned trial: 60 Tai Chi-naive adults, 30 per arm. The deposit holds 29 control and 31 Tai Chi "
                         "participants at the first visit (60 in total); 27 experts are in the data but not in the design paper. "
                         "The PhysioNet entry (version 1.0.2, Dec 2021) is the data release, titled 'Tai Chi, Physiological Complexity, and Healthy Aging - Gait'; the design paper is the "
                         "one the curator listed as the original publication."])
        stdlib.release(ID, MODULE, df, meta, ck, participants=pp, participant_columns=pcols, extra_files=extra)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))


main()
