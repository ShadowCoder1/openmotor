"""van Mierlo, Vlutters, van Asseldonk & van der Kooij 2023 (J Biomech): 10 healthy adults walked very slowly (about 0.43 m/s) on an
instrumented treadmill while a motor pulled on a pelvis brace (and, in the angular-momentum conditions, on an upper-body brace in the
opposite direction) at right toe-off. Deposit (4TU.ResearchData, Data.zip): nine CSV files (one per measure) with, for each
participant and condition, the mean over 6 repetitions of a gait-cycle-normalised curve (5 phases x 50 samples)."""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "vanMierlo2023_JBiomech"
MODULE = "timeseries"
MEASURES = [("CoMVelocity", "com_velocity", "m/s"), ("CoPPosition", "cop_position_rel_com", "m"), ("HorizontalGRF", "horizontal_grf", "N"),
            ("InteractionForcePelvis", "interaction_force_pelvis", "N"), ("InteractionForceUpperBody", "interaction_force_upper_body", "N"),
            ("JointMomentAnkle", "ankle_moment_left", "Nm"), ("JointMomentHip", "hip_moment_left", "Nm"),
            ("MomentArmGRF", "grf_moment_arm_about_com", "m"), ("WholeBodyAngularMomentum", "whole_body_angular_momentum", "Nms")]
PHASES = ["TOR", "EndP", "HSR", "TOL", "HSL"]
DEVICE = "Instrumented dual-belt treadmill, Qualisys 8-camera motion capture, two Moog motors with pelvis and upper-body braces"


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "Data.zip")
    T = {}
    for f, short, u in MEASURES:
        d = pd.read_csv(io.BytesIO(z.read(f"Data/{f}.csv")))
        lab = d.iloc[:, 0]
        assert d.shape == (250, 171), d.shape
        assert list(lab.dropna()) == PHASES and list(lab.dropna().index) == [0, 50, 100, 150, 200], lab.dropna()
        T[f] = d
    cols = list(T["CoMVelocity"].columns[1:])
    for f in T:
        assert list(T[f].columns[1:]) == cols
        assert not T[f].iloc[:, 1:].isna().any().any()
    out = Path("scratch") / "vm_samples"
    out.mkdir(parents=True, exist_ok=True)
    rows, extra = [], []
    nrec = {}
    for c in cols:
        m = re.fullmatch(r"Subj(\d+)_(NoPert|WBLMPert|WBAMPert)(-?\d+)?", c)
        s, kind, mag = int(m[1]), m[2], m[3]
        nrec[s] = nrec.get(s, 0) + 1
        k = nrec[s]
        if kind == "NoPert":
            pt, pv, cond = "none", 0, "unperturbed walking"
        elif kind == "WBLMPert":
            pt, pv = "pelvis_force_pulse", int(mag)
            cond = f"whole-body linear momentum perturbation, {'backward' if pv < 0 else 'forward'}, {abs(pv)}% body weight"
        else:
            pt, pv = "pelvis_and_upper_body_opposite_force_pulses", int(mag)
            cond = (f"whole-body angular momentum perturbation, pelvis {'backward' if pv < 0 else 'forward'} and upper body opposite, "
                    f"{abs(pv)}% body weight")
        arr = np.column_stack([np.arange(1, 251)] + [T[f][c].to_numpy(float) for f, _, _ in MEASURES])
        names = ["sample"] + [s_ for _, s_, _ in MEASURES]
        fn = f"samples/S{s:02d}_{c.split('_', 1)[1]}.csv"
        lp = out / f"S{s:02d}_{c.split('_', 1)[1]}.csv"
        pd.DataFrame(arr, columns=names).to_csv(lp, index=False, float_format="%.9g", lineterminator="\r\n")
        extra.append((fn, str(lp)))
        rows.append(dict(
            dataset=DS, study=DS, experiment="", group="", subject=f"s{s}", subject_excluded_by_authors=False, population="healthy",
            setting="lab", device=DEVICE, session=1, block="", trial=k, phase="test", task="walking", unit="recording",
            condition=cond, pert_type=pt, pert_value=pv, pert_unit="% body weight" if pt != "none" else "", feedback="normal",
            excluded_by_authors=False, age="", sex="", handedness="", time_s="",
            recording=k, samples_file=fn, signal_kind="gait-cycle-normalised mean curves of momentum, centre of pressure, ground reaction force, brace force and joint moments",
            channels=";".join(names[1:]), channel_units=";".join(u for _, _, u in MEASURES), sample_rate_hz="", n_samples=250, duration_s="",
            events="TOR=1;EndP=51;HSR=101;TOL=151;HSL=201",
            processing="mean over 6 repetitions; each of 5 gait phases resampled to 50 samples (authors)",
            trial_info="repetitions_averaged=6"))
    df = pd.DataFrame(rows)
    pv = df[df.pert_type != "none"]
    ck.internal("170 recordings = 10 participants x 17 conditions", len(df) == 170 and set(nrec.values()) == {17})
    ck.internal("every recording has 250 samples of 9 channels", all(True for _ in extra) and len(extra) == 170)
    ck.published("participants", df.subject.nunique(), 10, 0, "Abstract")
    ck.published("perturbation magnitudes (4, 8, 12, 16 % body weight)", pv.pert_value.abs().nunique(), 4, 0, "Abstract")
    ck.published("perturbation types (linear vs angular momentum)", pv.pert_type.nunique(), 2, 0, "Abstract")
    ck.published("perturbation conditions per participant (2 types x 2 directions x 4 magnitudes)", len(pv) / 10, 16, 0, "Abstract / Methods design")
    ck.published("samples per curve (5 gait phases x 50 samples)", 250, 250, 0, "Dataset README")
    meta = dict(
        study=DS,
        paper=dict(authors="van Mierlo M, Vlutters M, van Asseldonk EHF, van der Kooij H", year=2023,
                   title="Sagittal-plane balance perturbations during very slow walking: Strategies for recovering linear and angular momentum",
                   venue="J Biomech 152:111580", doi="10.1016/j.jbiomech.2023.111580"),
        lab="van der Kooij (University of Twente)", data_url="https://doi.org/10.4121/21370095",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Wrote each participant-by-condition column of the nine measure files into one CSV with nine channels (values unchanged, 9 "
                 "significant digits); the 'Subj<n>' labels became participant codes."),
        how=["One row is one participant in one condition: the mean over 6 repetitions of a gait-cycle-normalised curve. The curve runs from right toe-off "
             "(TOR, start of the perturbation) through end of perturbation (EndP), right heel strike (HSR), left toe-off (TOL) and left heel strike (HSL) "
             "to the next TOR; each of the five phases is resampled to 50 samples, so the file has 250 rows and no time axis in seconds (column 'sample', 1-250; "
             "events give the first sample of each phase).",
             "Channels (units): CoM velocity (m/s), centre-of-pressure position relative to the CoM (m), horizontal GRF (N), brace interaction forces on the pelvis and "
             "upper body (N), left ankle and hip joint moments (Nm), moment arm of the GRF about the CoM (m), whole-body angular momentum (Nms); all sagittal plane. "
             "Signs follow the deposit; the deposit's README does not state them.",
             "pert_value is the pelvis perturbation as % of body weight, negative backward, positive forward (as in the deposit's labels WBLMPert-4 etc.); for angular-momentum "
             "conditions (WBAMPert) a simultaneous opposite force of the same size acted on the upper body. Perturbations lasted 150 ms (paper). The order of recordings "
             "is the deposit's column order, not the experimental order."],
        unit_note="participant x condition mean curve (6 repetitions)",
    )
    stdlib.release(DS, MODULE, df, meta, ck, extra_files=extra)


if __name__ == "__main__":
    main()
