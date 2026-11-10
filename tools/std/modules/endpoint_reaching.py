"""Task module: reaching to targets on a digitizing tablet where only the target and the endpoint of each reach are shared (no
hand direction, no trajectory), one row per reach. First defined with Fassold2023_PLoSCompBiol (stylus reaches to a target on a
tablet seen through a mirror, with a control task in which people also report where they think the reach ended)."""

DESCRIPTION = ("Reaches to a target on a tablet: one row per reach, with the target position, the endpoint of the reach and, "
               "where the task asked for it, the endpoint the participant reported, all in the tablet's own coordinates.")

# (column, type, values or units, description)
COLUMNS = [
    ("trial_role", "category", "control_test, feedback_trial, confidence_trial",
     "Role of the reach in the authors' design: control_test (session 1 control task, reach and then report the felt endpoint, no feedback); "
     "feedback_trial (first or second trial of a triplet in the main task, veridical endpoint feedback); confidence_trial (third trial of a "
     "triplet, no endpoint feedback, followed by a confidence-circle report; the confidence values are not included)."),
    ("triplet_position", "integer", "1 to 3", "Position of the reach within its triplet in the main task (1, 2 or 3); empty in the control task."),
    ("target_x_mm", "number", "mm", "Horizontal target position in the tablet's coordinates (tablet 476 mm wide)."),
    ("target_y_mm", "number", "mm", "Vertical target position in the tablet's coordinates (tablet 268 mm high)."),
    ("endpoint_x_mm", "number", "mm", "Horizontal position where the stylus ended the reach."),
    ("endpoint_y_mm", "number", "mm", "Vertical position where the stylus ended the reach."),
    ("reported_x_mm", "number", "mm", "Horizontal position the participant reported as the endpoint (control task only)."),
    ("reported_y_mm", "number", "mm", "Vertical position the participant reported as the endpoint (control task only)."),
]
