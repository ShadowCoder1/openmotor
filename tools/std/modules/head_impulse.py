"""OpenMotor task module: head impulse test (vestibulo-ocular reflex gain and the saccades that follow a head impulse)."""

DESCRIPTION = (
    "Head impulse test (video head impulse test, vHIT) of the vestibulo-ocular reflex (VOR). A clinician turns the seated "
    "participant's head briskly in the plane of one semicircular canal while the participant fixates a target; the eye "
    "movement is recorded and compared with the head movement. One row is one corrective (refixation) saccade after one "
    "head impulse, or the impulse itself when no saccade followed; head and VOR values repeat on all rows of an impulse. "
    "Velocities and accelerations are in degrees; times are in ms from the start of the head impulse."
)

# (column, type, values or units, description)
COLUMNS = [
    ("canal", "category", "LL, RL, LA, RA, LP, RP",
     "Direction of the head impulse, named by the semicircular canal it tests: left or right (L, R) lateral, anterior or posterior."),
    ("impulse_number", "integer", "", "Impulse sequence number within the participant and canal, as the authors numbered it."),
    ("head_peak_acceleration_deg_s2", "number", "deg/s/s", "Peak acceleration of the head impulse."),
    ("head_peak_acceleration_time_ms", "number", "ms", "Time of the head peak acceleration from impulse onset."),
    ("head_peak_velocity_deg_s", "number", "deg/s", "Peak velocity of the head impulse."),
    ("head_peak_velocity_time_ms", "number", "ms", "Time of the head peak velocity from impulse onset."),
    ("head_zero_crossing_ms", "number", "ms", "Duration of the head impulse up to the first zero velocity crossing, from onset."),
    ("head_bounce_percent", "number", "percent", "Head movement after the zero crossing as a percentage of the main impulse."),
    ("head_mean_frequency", "number", "unit not stated", "Mean frequency component of the head impulse."),
    ("head_amplitude_deg", "number", "deg", "Head displacement (magnitude) of the impulse."),
    ("vor_gain_macdougall2013", "number", "ratio", "VOR gain computed with the method of MacDougall et al. 2013."),
    ("vor_gain_weber2008", "number", "ratio", "VOR gain computed with the method of Weber et al. 2008."),
    ("vor_gain_mahfuz2018", "number", "ratio", "VOR gain computed with the method of Mahfuz et al. 2018."),
    ("vor_gain_blodow2013", "number", "ratio", "VOR gain computed with the method of Blodow et al. 2013."),
    ("vor_gain_strupp2018", "number", "ratio", "VOR gain computed with the method of Strupp et al. 2018."),
    ("head_velocity_category", "category", "0, 100, 200, 300 deg/s",
     "Head peak velocity class: 0 = below 100, 100 = 100-199.9, 200 = 200-299.9, 300 = 300 or more deg/s."),
    ("saccade_number", "integer", "-1, 0, 1, 2, 3, 4",
     "Order of the saccade after this impulse, counted from 50 ms after impulse onset; 0 = no saccade followed the impulse; "
     "-1 = a saccade starting before that 50 ms window (onsets of a few ms)."),
    ("saccade_onset_ms", "number", "ms", "Saccade onset latency from the impulse onset (counted from 50 ms after onset)."),
    ("saccade_amplitude_deg", "number", "deg", "Saccade magnitude (displacement)."),
    ("saccade_peak_velocity_deg_s", "number", "deg/s", "Saccade peak velocity."),
    ("saccade_cv", "number", "unit not stated",
     "Saccade coefficient of variation, grouped by canal, for saccade numbers 1 and 2 only."),
    ("age_decade", "integer", "10, 20, ... 80", "Age decade of the participant: 10 = 11-19 years, 20 = 20-29, and so on."),
]
