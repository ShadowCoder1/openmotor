"""Task module: judging the duration of a brief visual stimulus while preparing or making a goal-directed finger reach, one row
per trial. First defined with Binetti2026_iScience (duration discrimination during reach preparation, with a cursor mapping that
could be congruent or incongruent, plus a sensory control without reaching). Columns hold the authors' own trial values."""

DESCRIPTION = ("Duration judgements made while preparing or making a goal-directed reach: one row per trial, with the reach "
               "task and cursor mapping, the duration of the test stimulus and the judgement (longer than the standard or not), "
               "and the authors' kinematic measures of the reach that went with it.")

# (column, type, values or units, description)
COLUMNS = [
    ("reach_task", "category", "Direction, Distance", "The authors' Task: reach to a target left or right of the start (Direction) or to a near or far target on the right (Distance)."),
    ("mapping", "category", "congruent, incongruent", "Mapping between hand and cursor of the row's own task (DirMap in the Direction task, DistMap in the Distance task). Incongruent: reversed direction (Direction task) or a changed extent (Distance task)."),
    ("target_direction", "category", "left, right", "The authors' Dir."),
    ("target_distance", "category", "near, far", "The authors' Dist (near about 205 px from the start, far about 410 px)."),
    ("cursor_gain", "number", "ratio", "The authors' Gain: 1 on congruent trials, 0.5 or 2 on some incongruent trials."),
    ("run", "integer", "1, 2", "The authors' Run (half of the session)."),
    ("trial_in_run", "integer", "", "The authors' Trial number, which counts through the participant's rows of the file as the authors numbered them."),
    ("probe_onset_s", "number", "seconds", "The authors' probeOnset (time of the test stimulus onset in the deposit's units; as shared)."),
    ("probe_duration_s", "number", "seconds", "Duration of the test stimulus (probeDur): 0.44, 0.64, 0.84, 1.04 or 1.24 s. The standard lasted 0.84 s."),
    ("judged_longer", "true/false", "", "The authors' timeResp: the test was judged longer than the standard. Inferred coding: 1 is 'longer', because the share of 1 rises from 3% at 0.44 s to 96% at 1.24 s; the paper does not give the numeric code."),
    ("response_time_s", "number", "seconds", "The authors' RT column. Its definition is not given in the paper or code; values are mostly 0.1 to 0.3 s, with some long outliers."),
    ("early_start", "true/false", "", "The authors' EarlyStart flag (reach started before the go signal, as flagged by the authors)."),
    ("late_start", "true/false", "", "The authors' LateStart flag."),
    ("probe_error_samples", "integer", "samples", "The authors' probeErrorSamp: timing error of the test stimulus, in display samples (-1, 0 or 1)."),
    ("score", "number", "", "The authors' score column, empty on rows without a reach."),
    ("action_start_frames", "integer", "frames of 75 Hz", "Action latency after the go signal, in frames of the 75 Hz hand tracker (the authors convert with x 1000 / 75 ms). Empty on rows without a reach (the deposit holds -1)."),
    ("action_stop_frames", "integer", "frames of 75 Hz", "The authors' actStop. Empty on rows without a reach."),
    ("action_duration_frames", "integer", "frames of 75 Hz", "The authors' actDur. Empty on rows without a reach."),
    ("peak_velocity", "number", "authors' units", "The authors' peakVel (units not stated)."),
    ("mean_velocity", "number", "authors' units", "The authors' meanVel (units not stated); the authors use actDur x meanVel as the distance traced."),
    ("sse", "number", "", "The authors' SSE (not defined in the paper)."),
    ("mse", "number", "", "The authors' MSE (not defined in the paper)."),
    ("trajectory_inefficiency", "number", "", "The authors' trajIneff (not defined in the paper)."),
    ("hand_error", "number", "", "The authors' handErr (not defined in the paper or code)."),
    ("hand_stop_x", "number", "px", "X of the final index position, as the authors gave it."),
    ("hand_stop_y", "number", "px", "Y of the final index position."),
    ("goal_x", "number", "px", "The authors' GoalX: x of the target centre (about +-205 px, or 410 px for the far target)."),
    ("goal_y", "number", "px", "The authors' GoalY (0)."),
    ("goal_distance_px", "number", "px", "The authors' GoalDist: distance between the final index position and the target centre (reaching error). Empty on rows without a reach."),
    ("outlier_action", "true/false", "", "The authors' outlierAction flag. The deposit holds it for half of the rows only (empty in the Direction task)."),
]
