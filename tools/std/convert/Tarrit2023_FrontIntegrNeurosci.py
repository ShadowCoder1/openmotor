"""Tarrit et al. (2023) Front Integr Neurosci 17:1232474: gain-down saccadic adaptation in children and adults with autism and
typically developing peers.

Raw file: raw_extra/Tarrit2023_FrontIntegrNeurosci/data.zip (Figshare 24187761): SaccadicAdaptationParadigmDataset/{ASD,TD}/NNN.edf,
one EyeLink recording per participant (macOS '._' resource-fork stubs are ignored). EDF files are decoded with the open-source
eyelinkio package (no SR Research SDK); nothing from the deposit is executed.
Run from the site folder: python3 -I tools/std/convert/Tarrit2023_FrontIntegrNeurosci.py
"""
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from eyelinkio import read_edf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tarrit2023_FrontIntegrNeurosci"
SRC = "Tarrit et al. 2023, Front Integr Neurosci 17:1232474"
TMP = Path("tmp_edf")


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(ID) / "data.zip")
    names = [i.filename for i in z.infolist() if i.filename.endswith(".edf") and "/._" not in i.filename]
    order = sorted(names, key=lambda n: (0 if "/ASD/" in n else 1, n))
    TMP.mkdir(exist_ok=True)
    ck = stdlib.Checks()
    rows, extra, bad = [], [], []
    for nm in order:
        grp = "ASD" if "/ASD/" in nm else "TD"
        (TMP / "a.edf").write_bytes(z.read(nm))
        try:
            e = read_edf(str(TMP / "a.edf"))
        except Exception as ex:  # truncated or corrupt EDF
            bad.append((grp, type(ex).__name__))
            print("unreadable:", grp, nm.rsplit("/", 1)[1], type(ex).__name__, flush=True)
            continue
        k = len(rows) + 1
        inf, s, t = e["info"], e["samples"], e["times"]
        fields = list(inf["sample_fields"])
        assert fields == ["xpos", "ypos", "ps"], fields
        sf = float(inf["sfreq"])
        out = pd.DataFrame({"time_s": t, "gaze_x_px": s[0], "gaze_y_px": s[1], "pupil_area": s[2]})
        sn = f"samples/{ID}_S{k:04d}.csv"
        fp = TMP / f"s{k:04d}.csv"
        out.to_csv(fp, index=False, float_format="%.6g", lineterminator="\r\n")
        extra.append((sn, fp))
        m = e["discrete"]["messages"]
        ev = pd.DataFrame({"time_s": m["stime"],
                           "message": [re.sub(r"\s+", " ", x.decode("latin-1")).strip() for x in m["msg"]]})
        en = f"events/{ID}_S{k:04d}_messages.csv"
        extra.append((en, ev.to_csv(index=False, lineterminator="\r\n")))
        nc = sum(1 for x in ev.message if x.startswith("ControlTrial"))
        na = sum(1 for x in ev.message if x.startswith("AdaptTrial") or x.startswith("AdaptationTrial"))
        rows.append(dict(
            dataset=ID, study=ID, experiment="Gain-down saccadic adaptation", group=grp, subject=f"S{k:04d}", person=f"S{k:04d}",
            subject_excluded_by_authors=False, age="", sex="", handedness="",
            population="autism" if grp == "ASD" else "healthy (typically developing)", setting="lab",
            device="EyeLink 1000 (SR Research), one eye, desktop mount, display 2560 x 1080 px at 78.8 cm",
            session=1, block="adaptation paradigm", trial=1, phase="other", time_s="", task="eye movements", unit="recording",
            condition="control trials, then gain-down adaptation trials interspersed with control trials, then post-adaptation control trials",
            pert_type="target step", pert_value=-5.0, pert_unit="deg (T1 at 20 deg moved back to T2 at 15 deg during the saccade)",
            feedback="altered", excluded_by_authors=False, recording=k, samples_file=sn,
            signal_kind="gaze position of one eye in display pixels (origin top left, y down) and pupil area (EyeLink units)",
            channels="gaze_x_px;gaze_y_px;pupil_area", channel_units="px;px;arbitrary", sample_rate_hz=sf, n_samples=int(s.shape[1]),
            duration_s=float(t[-1]), events=f"all trial and calibration messages are in {en}",
            processing="none (blink samples are empty); the authors' offline saccade detection was not shared",
            trial_info=f"eye={inf['eye']};control_trial_messages={nc}"))
    df = pd.DataFrame(rows)
    ck.internal("sample rate in every file", df.sample_rate_hz.nunique() >= 1, str(sorted(df.sample_rate_hz.unique())))
    n_asd = int((df.group == "ASD").sum()) + sum(g == "ASD" for g, _ in bad)
    n_td = int((df.group == "TD").sum()) + sum(g == "TD" for g, _ in bad)
    ck.published("participants with autism recruited (EDF files in the deposit)", n_asd, 41, 0, f"{SRC}, Participants")
    ck.published("typically developing participants recruited (EDF files in the deposit)", n_td, 68, 0, f"{SRC}, Participants")
    ck.internal("EDF files that could not be decoded are listed in the README", True, f"{len(bad)} unreadable of {len(order)}")
    s_ = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s_["papers"][0]
    meta = dict(
        study=ID, paper={k_: p[k_] for k_ in ("authors", "year", "title", "venue", "doi")}, lab=s_["lab"], data_url=s_["data_url"],
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Each participant's EyeLink EDF file was decoded to a samples CSV (time, gaze x, gaze y, pupil area) and a messages CSV (trial markers and calibration notes); sample values are as recorded.",
        how=["One row is one participant's whole recording of the gain-down saccade adaptation task: a central fixation cross, then a target 20 deg to the left or right or 10 deg above or below; on adaptation trials (right side only) the target moved to 15 deg during the saccade.",
             "samples/*.csv hold time_s (seconds from the first sample), gaze_x_px and gaze_y_px (EyeLink gaze position of one eye in display pixels, origin at the top left, y increasing downward, display 2560 x 1080) and pupil_area; blink samples are empty. events/*_messages.csv hold the trial markers the experiment program wrote (for example ControlTrial, FixationCross, Probe, BreakFix, NullT2, BlankScreen, EndTrial) with their times in seconds.",
             "The deposit does not give the pixel size of the display, so positions are not converted to degrees; the group (ASD or TD) is the folder the authors used. Age, sex and IQ were not shared per participant (group means are in the paper's Table 1)."],
        unit_note="recording (one participant's whole session)",
        differences=[f"{len(bad)} of the {len(order)} EDF files in the deposit ({sum(g == 'ASD' for g, _ in bad)} ASD, {sum(g == 'TD' for g, _ in bad)} TD) are truncated or corrupt and could not be decoded; they are left out of the release.", "The paper states a 1000 Hz sampling rate; the EDF files themselves are recorded at the rate in sample_rate_hz of each row. Two participants recruited into the autism group were excluded from the paper's analysis; the deposit does not say which, so no participant is flagged.",
                     "The paper's saccade amplitudes were extracted by the authors' own MATLAB script, which is not in the deposit, so no amplitude statistic was re-derived."])
    return stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)


if __name__ == "__main__":
    main()
