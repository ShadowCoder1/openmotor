"""Violin bowing module: one row per full-bow exercise on the violin, with the authors' bow-motion descriptors (Kinect tracking of
bow and violin) and tone-quality descriptors (from the audio), and the bow-motion series of the trial as a CSV inside the ZIP.

Defined with the first bowing dataset (Blanco2021_FrontPsychol_554201). No other module has per-trial bow kinematics and tone
descriptors: pitch_matching rows are notes, timeseries rows carry no descriptors."""

DESCRIPTION = ("Violin bowing: one row per full-bow exercise (alternating up and down bows over the full length of the bow), with the "
               "authors' per-trial descriptors of bow motion (skewness, tilt, bow-bridge distance, speed, inclination, bow-violin "
               "distance) and of tone quality (pitch and dynamic instability, aperiodicity), and the trial's bow-motion series as a "
               "CSV inside the ZIP.")

# (column, type, values or units, description)
COLUMNS = [
    ("bow_skewness", "number", "deg", "Bow skewness descriptor: mean absolute value of the third Euler angle (yaw) of the bow in the violin's coordinate system over the trial; 0 = bow perpendicular to the strings (authors' bowSkew)."),
    ("bow_tilt_sd", "number", "deg", "Standard deviation over the trial of the bow's second Euler angle (pitch) (authors' tilt)."),
    ("bow_inclination_sd", "number", "deg", "Standard deviation over the trial of the bow's first Euler angle (roll), the inclination towards the other strings (authors' inclination)."),
    ("bow_bridge_distance_sd", "number", "authors' units (not stated)", "Standard deviation over the trial of the distance between the contact point and the bridge (authors' bowBridge); the deposit does not state the unit."),
    ("bow_speed_sd", "number", "authors' units (not stated)", "Standard deviation over the trial of the bow speed (authors' Speed); the deposit does not state the unit."),
    ("bow_violin_distance_sd", "number", "authors' units (not stated)", "Standard deviation over the trial of the distance between the bow and the violin (authors' ViolinDist); the deposit does not state the unit."),
    ("pitch_instability", "number", "Hz", "Standard deviation of the fundamental frequency (Yin, 33 ms window, 0.7 ms hop) over the trial; lower = more stable. The authors' field pitch_stab, which their readme says should have been called pitch instability."),
    ("dynamic_instability", "number", "authors' units (dB in the paper)", "Standard deviation of the instantaneous power over the trial; lower = more stable. The authors' field dyn_stab."),
    ("aperiodicity_sd", "number", "", "Standard deviation of the Yin aperiodicity over the trial (authors' aperiodicity)."),
    ("n_motion_samples", "integer", "", "Number of bow-motion samples of the trial (86.13 per second)."),
    ("motion_sample_rate_hz", "number", "Hz", "Sampling rate of the bow-motion series (86.13 Hz, as the paper states)."),
    ("samples_file", "text", "path inside the ZIP", "CSV with the trial's bow-motion series: time_s, then bow_skewness_deg, bow_tilt_deg, bow_inclination_deg, bow_bridge_distance, bow_speed, bow_violin_distance, as the authors shared them (not filtered or resampled)."),
]
