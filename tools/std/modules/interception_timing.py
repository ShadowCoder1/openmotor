"""Task module: intercepting a moving target with a hand movement (or button press) at the right moment, one row per trial. First
defined with LopezMoliner2019_SciRep (stylus interception of a target that moves horizontally and reaches the interception zone
0.8 s after it starts). Columns hold the authors' trial values; where the deposit does not define a column the description says so."""

DESCRIPTION = ("Timed interception of a moving target: one row per trial, with the target's speed and starting position, the "
               "authors' temporal and spatial error, whether the trial counted as a hit, and the hand's timing and speed.")

# (column, type, values or units, description)
COLUMNS = [
    ("trial_in_block", "integer", "from 0", "The authors' trial number within the block (the deposit's 'trial'), 0 to 89."),
    ("target_speed_cm_s", "number", "cm/s", "Speed of the moving target: 20, 25 or 30 cm/s."),
    ("target_direction", "category", "leftward, rightward", "Direction in which the target moved, from the sign of the deposit's velx (negative: leftward). The paper says the target moved rightward, or leftward for left-handed participants, but the deposit's signs do not match the stated number of left-handed participants."),
    ("target_start_x_cm", "number", "cm", "The authors' xinit: horizontal distance of the target's starting point from the interception zone; equals the target speed x 0.8 s."),
    ("movement_onset_s", "number", "seconds", "The authors' tStart.f: movement onset time after the start of the trial (the paper finds onset with a speed-threshold algorithm)."),
    ("crossing_time_s", "number", "seconds", "The authors' timeEnd: time at which the stylus crossed the target's path."),
    ("movement_time_s", "number", "seconds", "The authors' mt: time from movement onset to crossing the target's path."),
    ("temporal_error_s", "number", "seconds", "The authors' tErr, as shared. The paper defines temporal error as the time the stylus crossed the target's path minus the time the target reached that position; the sign of the deposit's column was not verified against that definition. Empty on 3 rows."),
    ("spatial_error_cm", "number", "cm", "The authors' sErr (no definition given in the paper's methods; centimetres inferred)."),
    ("hit", "true/false", "", "The authors' hit flag (the paper defines a hit as crossing between the lines of the interception zone with an absolute temporal error below 20 ms; the shared flag does not reproduce that rule exactly, so it is given as shared)."),
    ("peak_velocity_cm_s", "number", "cm/s", "The authors' peakV: peak hand speed."),
    ("crossing_velocity_cm_s", "number", "cm/s", "The authors' crossV: hand speed at the crossing (empty on 3 rows)."),
    ("time_to_contact_s", "number", "seconds", "The authors' ttc (not defined in the paper's methods)."),
    ("hand_end_x_cm", "number", "cm", "The authors' xhand_end: horizontal stylus position at the end of the movement, relative to the interception zone (inferred)."),
    ("hand_end_y_cm", "number", "cm", "The authors' yhand_end."),
    ("target_end_cm", "number", "cm", "The authors' targetEnd (not defined in the paper's methods)."),
]
