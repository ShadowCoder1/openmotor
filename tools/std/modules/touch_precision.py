"""Task module: precision of discrete touchpad or screen movements to targets, one row per trial. First defined with
Oldrati2026_NeuralPlast (drag-and-drop of a paper ball into one of eight bins with the index finger on a touchpad, during
brain stimulation): the authors' precision error and reaction time per trial."""

DESCRIPTION = ("Precision of discrete finger movements on a touchpad or screen to targets (for example dragging an object into "
               "a bin): one row per trial, with the target distance and the authors' precision error and reaction time.")

# (column, type, values or units, description)
COLUMNS = [
    ("trial_in_part", "integer", "", "The authors' trial number, which restarts at 1 after the first 160 trials of a session part (as in the deposit)."),
    ("target_distance", "category", "near, far", "Whether the target bin was one of the four near the centre or one of the four far from it."),
    ("trial_status", "text", "", "The authors' trial label (valid: completed within the time limits). The deposit shares valid trials only."),
    ("precision_error", "number", "authors' units (see README)", "The authors' precision error: distance between the final drop position and the centre of the target bin, as shared in the deposit."),
    ("rt_ms", "number", "ms", "Time from the object's appearance to the click that picks it up."),
]
