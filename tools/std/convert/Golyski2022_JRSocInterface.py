"""Golyski & Sawicki 2022 (J R Soc Interface): seven young adults walked at 1.25 m/s on an instrumented treadmill while one belt
briefly sped up (to 2.5 m/s) in early or late stance of the left or right leg. Deposit (Zenodo 6338829): Pert_MechEnergetics.mat,
a MATLAB struct Sub -> side (L/R) -> timing (Early/Late) -> iteration, each holding about 13 s of 200 Hz signals (belt speeds, ground
reaction forces, joint angles, moments, powers) and frame numbers of foot strikes and perturbation. The mat file also has a MATLAB
table of work values (metrics_table_stats) that cannot be read outside MATLAB; it is not converted."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Golyski2022_JRSocInterface"
MODULE = "timeseries"
DEVICE = "Instrumented split-belt treadmill (CAREN, Motek) with motion capture"
EVENT_SCALARS = ["des_start", "I_SM1_FS", "I_S0_FS", "I_SP1_FS", "I_SP2_FS", "I_SP3_FS", "I_SP4_FS"]
EVENT_NAMES = {"des_start": "commanded_perturbation_onset", "I_SM1_FS": "perturbed_leg_heel_strike_of_stride_before",
               "I_S0_FS": "perturbed_leg_heel_strike_of_perturbed_stride", "I_SP1_FS": "perturbed_leg_heel_strike_recovery_stride_1",
               "I_SP2_FS": "perturbed_leg_heel_strike_recovery_stride_2", "I_SP3_FS": "perturbed_leg_heel_strike_recovery_stride_3",
               "I_SP4_FS": "perturbed_leg_final_heel_strike_recovery_stride_3"}


def unit_of(ch):
    b = ch.split("_", 1)[1] if ch[:4] in ("ips_", "con_") else ch
    if b == "spd":
        return "m/s"
    if b in ("APGRF", "vGRF") or ch in ("apGRFtot", "mlGRFtot", "vGRFtot"):
        return "N"
    if "power" in b.lower() or "Power" in b or b.endswith("PCOM") or ch == "PCOM" or b in ("ankP", "kneP", "hipP", "hipPA", "hipPR", "hipPT"):
        return "W"
    if b.endswith("AVA") or b.endswith("AVR") or b in ("ankAV", "kneAV", "hipAV"):
        return "deg/s"
    if b in ("ankA", "kneA", "hipA", "hipAA", "hipAR"):
        return "deg"
    if b in ("ankM", "kneM", "hipM", "hipMA", "hipMR"):
        return "Nm"
    raise ValueError(ch)


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "Pert_MechEnergetics.zip")
    m = sio.loadmat(io.BytesIO(z.read("Pert_MechEnergetics.mat")), struct_as_record=False, squeeze_me=True,
                    variable_names=["Pert_MechEnergetics"])
    root = m["Pert_MechEnergetics"]
    out = Path("scratch") / "gol_samples"
    out.mkdir(parents=True, exist_ok=True)
    rows, extra, masses, persons, lens, nchan = [], [], {}, [], set(), set()
    chan_ref = None
    n_rec = 0
    for sub in root._fieldnames:
        S = getattr(root, sub)
        rec = 0
        for side in S._fieldnames:
            for timing in getattr(S, side)._fieldnames:
                T = getattr(getattr(S, side), timing)
                for it in T._fieldnames:
                    R = getattr(T, it)
                    N = len(R.ips_spd)
                    chans = [f for f in R._fieldnames if isinstance(getattr(R, f), np.ndarray) and getattr(R, f).ndim == 1
                             and len(getattr(R, f)) == N and f not in ("ips_FS", "con_FS")]
                    chan_ref = chan_ref or chans
                    nchan.add(len(chans))
                    lens.add(N)
                    dt = float(R.tstep)
                    masses[sub] = float(R.mass)
                    assert R.side == side and R.DomL in ("L", "R")
                    rec += 1
                    n_rec += 1
                    arr = np.column_stack([np.arange(N) * dt] + [getattr(R, c).astype(float) for c in chans])
                    fname = f"samples/{sub}_{side}_{timing}_{it}.csv"
                    lp = out / f"{sub}_{side}_{timing}_{it}.csv"
                    pd.DataFrame(arr, columns=["time_s"] + chans).to_csv(lp, index=False, float_format="%.7g", lineterminator="\r\n")
                    extra.append((fname, str(lp)))
                    ev = [f"{EVENT_NAMES[e]}={(int(getattr(R, e)) - 1) * dt:.3f}" for e in EVENT_SCALARS]
                    ev += [f"{'perturbed' if k == 'ips_FS' else 'other'}_leg_heel_strike={(int(f) - 1) * dt:.3f}" for k in ("ips_FS", "con_FS") for f in getattr(R, k)]
                    rows.append(dict(
                        dataset=DS, study=DS, experiment="", group="", subject=sub, subject_excluded_by_authors=False,
                        population="healthy", setting="lab", device=DEVICE, session=1, block=f"{side} leg, {timing.lower()} stance",
                        trial=rec, phase="test", task="walking", unit="recording",
                        condition=f"{'left' if side == 'L' else 'right'} belt perturbed in {timing.lower()} stance",
                        pert_type="belt_speed_increase", pert_value=1.25, pert_unit="m/s", feedback="normal", excluded_by_authors=False,
                        age="", sex="", handedness="", time_s="",
                        recording=rec, samples_file=fname, signal_kind="treadmill belt speeds, ground reaction forces, joint angles, angular velocities, moments and mechanical powers",
                        channels=";".join(chans), channel_units=";".join(unit_of(c) for c in chans), sample_rate_hz=round(1 / dt, 3),
                        n_samples=N, duration_s=round((N - 1) * dt, 6), events=";".join(ev),
                        processing="authors' processed signals (inverse dynamics and power computations); see the paper. ips_ = perturbed (ipsilateral) leg, con_ = other leg."))
        persons.append(dict(person=sub, mass_kg=masses[sub], dominant_leg={"L": "left", "R": "right"}[R.DomL]))
    df = pd.DataFrame(rows)
    ck.internal("seven participants", len(masses) == 7 and df.subject.nunique() == 7)
    ck.internal("every recording is sampled at 200 Hz", len(lens) >= 1 and (df.sample_rate_hz == 200).all(),
                f"lengths {sorted(lens)[:5]}")
    ck.internal("channel units all assigned", all(len(a.split(';')) == len(b.split(';')) for a, b in zip(df.channels, df.channel_units)))
    w = np.array(list(masses.values()))
    ck.published("number of participants (7)", len(masses), 7, 0, "Methods, Participants")
    ck.published("trials analysed after removing crossover trials (222)", n_rec, 222, 0, "Results / Methods, trials analysed")
    ck.published("mean body mass, kg (72.7)", w.mean(), 72.7, 0.05, "Methods, Participants")
    ck.published("SD of body mass, kg (13.3)", w.std(ddof=1), 13.3, 0.05, "Methods, Participants")
    print(n_rec, w.mean(), w.std(ddof=1), nchan)
    meta = dict(
        study=DS,
        paper=dict(authors="Golyski PR, Sawicki GS", year=2022, title="Which lower limb joints compensate for destabilizing energy during walking in humans?",
                   venue="J R Soc Interface 19(191):20220024", doi="10.1098/rsif.2022.0024"),
        lab="Sawicki (Georgia Tech)",
        data_url="https://doi.org/10.5281/zenodo.6338829",
        license="CC BY 3.0 US", license_url="https://creativecommons.org/licenses/by/3.0/us/",
        changes=("Wrote each perturbation's signals to a CSV (time_s plus one column per channel, same values, 7 significant digits) and listed "
                 "frame numbers of the foot strikes and perturbation onset as times in seconds; participant IDs became codes; the MATLAB table of "
                 "work values, which cannot be read outside MATLAB, is not included."),
        how=["One row is one belt-speed perturbation (one recording of about 13 s at 200 Hz) from one participant; the samples are in the CSV named in "
             "samples_file. Perturbed belt = left or right, timing = early or late stance, as in the deposit's structure; the deposit keeps only perturbations with clean foot strikes (222 in all).",
             "Channels with the prefix ips_ belong to the perturbed (ipsilateral) leg and con_ to the other leg; units follow the deposit's readme "
             "(N, W, deg, deg/s, Nm, m/s). Event times are the deposit's frame numbers converted to seconds as (frame - 1) x 0.005 s.",
             "Participants walked at 1.25 m/s; the perturbation briefly raised one belt to 2.5 m/s (pert_value is the 1.25 m/s increase). The recording order inside a participant is the "
             "deposit's structure order (side, timing, iteration), not the order in which they were delivered (randomized). participants.csv holds body mass and dominant leg."],
        unit_note="perturbation (recording of about 13 s)",
        differences=["The Zenodo record is licensed CC BY 3.0 US, not CC BY 4.0 as the OpenMotor listing says. The Zenodo record title is the paper's earlier wording: 'Which lower-limb joints compensate for destabilising energy introduced by unilateral treadmill belt accelerations delivered during walking in humans?'."],
    )
    pdf = pd.DataFrame(persons)
    stdlib.release(DS, MODULE, df, meta, ck, participants=pdf,
                   participant_columns=[("mass_kg", "number", "kg", "Body mass."), ("dominant_leg", "category", "left, right", "Leg the participant kicks a ball with.")],
                   extra_files=extra)


if __name__ == "__main__":
    main()
