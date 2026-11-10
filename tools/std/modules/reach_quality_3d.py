"""Task module: reaches to 3D targets in a virtual or augmented environment, one row per reach, with the authors' movement-quality
measures (duration, path straightness, peak velocity, number of velocity peaks, onset) instead of a hand angle. First defined
with Wenk2022_JNER (touching fruits with a tracked controller or a weight-supported arm device under three displays)."""

DESCRIPTION = ("Reaches to 3D target locations (touching virtual fruits) under different displays, one row per reach, with the "
               "authors' movement-quality measures and the answers of a concurrent counting task.")

# (column, type, values or units, description)
COLUMNS = [
    ("display", "category", "VR, AR, SCREEN", "Display used: immersive head-mounted VR, augmented-reality head-mounted display, or 2D computer screen."),
    ("block_index", "integer", "", "The authors' TrialIndex: block number within the display condition, from 0."),
    ("reach_in_block", "integer", "", "The authors' CatchIndex: reach number within the block, from 0."),
    ("loc_x_usage", "integer", "-1, 0, 1", "Target location on the horizontal axis relative to the workspace centre: 0 not used, 1 positive side, -1 negative side."),
    ("loc_y_usage", "integer", "-1, 0, 1", "Same for the vertical axis."),
    ("loc_z_usage", "integer", "-1, 0, 1", "Same for the depth axis."),
    ("loc_dimensionality", "category", "1D, 2D, 3D", "Number of axes the target location used."),
    ("depth_usage", "category", "No, Only, Combined", "Whether the movement used depth: not at all, depth alone, or depth with another axis."),
    ("fruit_type", "text", "", "Type of fruit shown (the object to touch)."),
    ("fruit_number", "integer", "", "Running count of this fruit type in the block (the value the participant was to say)."),
    ("catch_success", "true/false", "", "False if the reach was skipped by the researcher."),
    ("duration_s", "number", "seconds", "Time to reach the fruit."),
    ("ideal_distance_m", "number", "m", "Straight-line distance from the start to the fruit."),
    ("duration_normalized", "number", "s/m", "Duration divided by ideal distance."),
    ("path_length_m", "number", "m", "Length of the path followed."),
    ("straightness_ratio", "number", "", "Path length divided by ideal distance."),
    ("peak_velocity_m_s", "number", "m/s", "Highest velocity during the reach."),
    ("peak_velocity_time_s", "number", "seconds", "Time from movement start at which peak velocity was reached."),
    ("peak_velocity_time_norm", "number", "", "The authors' PeakVelocityTime.Normalized, as shared (their description says peak velocity divided by ideal distance; values equal peak velocity time divided by ideal distance, see README)."),
    ("n_velocity_peaks", "integer", "", "Number of velocity peaks (smoothness; more is less smooth)."),
    ("onset_s", "number", "seconds", "Time from the disappearance of the start sphere until speed reached 0.2 m/s. Empty if the threshold was never reached."),
    ("endpoint_velocity_m_s", "number", "m/s", "Velocity when the fruit was touched."),
    ("count_said", "integer", "", "Counting value said by the participant."),
    ("type_said", "text", "", "Fruit type said by the participant. Empty if none."),
    ("fruit_correct", "true/false", "", "The counted fruit was correct (type and count matched)."),
    ("fruit_correctness", "text", "", "The authors' detailed correctness label (Correct, WrongFruit, WrongCount, WrongCount_FollowingPrevious, Skipped, MissedRecorded, SeveralCountingValues, NoFruit, NoCount)."),
]
