"""Haberfehlner et al. 2023, Front Robot AI: DeepLabCut 2D landmarks of 188 rest videos (lying, sitting) of 34 people with
dyskinetic cerebral palsy, with DIS scores. Run on the PC: python3 -I tools/std/convert/Haberfehlner2023_FrontRobotAI.py
Raw (read only): README.txt, data_clinical_scoring.xlsx, data_stickfigure_coordinates_{lying,sitting}.zip."""
import io, re, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, openpyxl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib

DS = "Haberfehlner2023_FrontRobotAI"
D = stdlib.raw_dir(DS)
SESS = {"T0": 1, "T3": 2, "T12": 3}
REGIONS = {"sitting": ["TRUNK", "RAP", "LAP"], "lying": ["RLP", "LLP"]}  # README: trunk/arms scored sitting, legs lying
FS = 25.0


def main():
    ck = stdlib.Checks()
    ws = openpyxl.load_workbook(D / "data_clinical_scoring.xlsx", read_only=True).active
    allr = list(ws.iter_rows(values_only=True))
    head = allr[0]
    clin = [dict(zip(head, r)) for r in allr[1:] if r[0] is not None]
    vid_info = {}
    for r in clin:
        vid_info.setdefault(int(r["video"]), (int(r["ID"]), r["time"]))
        assert vid_info[int(r["video"])] == (int(r["ID"]), r["time"])
    zips = {k: zipfile.ZipFile(D / f"data_stickfigure_coordinates_{k}.zip") for k in REGIONS}
    files = {k: {int(re.match(r"(\d+)_", Path(n).name).group(1)): n for n in z.namelist() if n.endswith(".csv")}
             for k, z in zips.items()}
    vids = sorted(set(files["lying"]) | set(files["sitting"]))
    ck.internal("lying and sitting each hold 94 files", len(files["lying"]) == 94 and len(files["sitting"]) == 94)
    pid_of = {v: vid_info[v][0] for v in vids if v in vid_info}
    unscored = [v for v in vids if v not in vid_info]
    order = []  # participants by first appearance
    recs = []
    for v in vids:
        for k in ("lying", "sitting"):
            if v in files[k]:
                recs.append((v, k))
    # participant ids: videos without clinical rows get no participant id and are kept under their own code
    keyp = lambda v: pid_of.get(v, f"unscored{v}")
    recs.sort(key=lambda t: (min(vv for vv in vids if keyp(vv) == keyp(t[0])), t[0], t[1] == "sitting"))
    code, rows, extra, nrec = {}, [], [], {}
    for v, k in recs:
        p = keyp(v)
        code.setdefault(p, f"{DS}_S{len(code) + 1:04d}")
        nrec[p] = nrec.get(p, 0) + 1
        n = nrec[p]
        d = pd.read_csv(io.BytesIO(zips[k].read(files[k][v])), header=[1, 2], index_col=0)
        chans, units, cols = [], [], {}
        for bp in dict.fromkeys(c[0] for c in d.columns):
            for c, u in (("x", "px"), ("y", "px"), ("likelihood", "probability")):
                chans.append(f"{bp}_{c}"); units.append(u); cols[f"{bp}_{c}"] = d[(bp, c)].to_numpy()
        out = pd.DataFrame({"time_s": d.index.to_numpy() / FS, **cols})
        fname = f"samples/{code[p]}_rec{n:03d}.csv"
        extra.append((fname, out.to_csv(index=False, lineterminator="\r\n").encode()))
        info = []
        for r in clin:
            if int(r["video"]) != v:
                continue
            for reg in REGIONS[k]:
                for pre in ("D", "CA"):
                    for m in ("tD", "tA"):
                        key = f"{pre}_{reg}_R_{m}"
                        info.append(f"scorer{int(r['scorer'])}_{key}={r[key]}")
        t = vid_info.get(v, (None, None))[1]
        rows.append(dict(
            dataset=DS, study="Haberfehlner 2023 Front Robot AI", experiment="DIS rest videos", group=None, subject=code[p],
            person=code[p], subject_excluded_by_authors=False, age=None, sex=None, handedness=None,
            population="dyskinetic cerebral palsy", setting="clinic", device="video camera, DeepLabCut", session=SESS.get(t),
            block=f"{k} in rest", trial=n, phase="other", time_s=None, task="sports and games", unit="recording",
            condition=f"{k} in rest, {t}" if t else f"{k} in rest", pert_type="none", pert_value=0, pert_unit=None,
            feedback="normal", excluded_by_authors=False, recording=n, samples_file=fname,
            signal_kind="2D body landmark positions from video (DeepLabCut)", channels=";".join(chans),
            channel_units=";".join(units), sample_rate_hz=FS, n_samples=len(d), duration_s=len(d) / FS, events="",
            processing="DeepLabCut output as shared (file names ending 'filtered'); coordinates not rescaled",
            trial_info=f"video={v};time_point={t};" + ";".join(info)))
    ck.internal("188 recordings", len(rows) == 188, str(len(rows)))
    ck.internal("all samples have the 14 named landmarks x 3 channels", all(len(r["channels"].split(";")) % 3 == 0 for r in rows))
    ck.internal("scores 0-4 or 999 for not scored",
                all(float(s.split("=")[1]) in (0, 1, 2, 3, 4, 999) for r in rows for s in r["trial_info"].split(";")[2:] if "=" in s
                    and s.split("=")[1] not in ("None", "")))
    ck.published("participants", len(code) - len(unscored), 34, 0, "Zenodo record description (188 videos of 34 children)")
    ck.published("videos", len(rows), 188, 0, "Zenodo record description (188 videos of 34 children)")
    ck.published("sample rate (Hz)", FS, 25, 0, "Zenodo record description (sampled with 25 Hz)")
    meta = dict(
        study=DS,
        paper=dict(authors="Haberfehlner H, van de Ven SS, van der Burg SA, Huber F, Georgievska S, Aleo I, Harlaar J, Bonouvrie LA, van der Krogt MM, Buizer AI",
                   year=2023, title="Towards automated video-based assessment of dystonia in dyskinetic cerebral palsy: A novel approach using markerless motion tracking and machine learning",
                   venue="Front Robot AI 10:1108114", doi="10.3389/frobt.2023.1108114"),
        lab="Buizer / Bonouvrie (Amsterdam UMC)", data_url="https://zenodo.org/records/5638470",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("One row per video; the DeepLabCut csv (x, y, likelihood per landmark) is a samples file with time_s = frame / 25. "
                 "Clinical DIS item scores (duration and amplitude of dystonia and choreoathetosis) of each scorer for the regions "
                 "scored in that video type are in trial_info; percentage and maximum columns (formulas) were left out. "
                 "Participant numbers were re-coded; the stick-figure movies were not carried."),
        how=["One row is one resting video (lying or sitting) of a child or young adult with dyskinetic cerebral palsy, with 14 body landmarks "
             "tracked by DeepLabCut: x and y in pixels of the 720x575 video (origin top left, y down) and the model likelihood, at 25 Hz.",
             "trial_info lists scorerK_<item>=score, with K the scorer number (1-3) and items such as D_TRUNK_R_tD (dystonia, trunk, rest, duration) "
             "or CA_RLP_R_tA (choreoathetosis, right leg proximal, amplitude); scores are 0-4 and 999 means not scored. Trunk and arms are scored on sitting "
             "videos, legs on lying videos (deposit README). session 1-3 is the time point T0 (baseline), T3 (3 months), T12 (12 months).",
             "The deposit README gives file-name suffixes for the two video types the opposite way round from the zip names; the zip names were followed "
             "and agree with the landmark spread (wider horizontally in the lying zip)."],
        unit_note="resting video with tracked landmarks",
        differences=["The paper's classifier results were not reproduced (no code was run); published checks are participant, video and frame-rate counts "
                     "from the dataset description. Videos without clinical rows (if any) are kept under their own codes."])
    stdlib.release(DS, "timeseries", rows, meta, ck, participants=None, extra_files=extra)


if __name__ == "__main__":
    main()
