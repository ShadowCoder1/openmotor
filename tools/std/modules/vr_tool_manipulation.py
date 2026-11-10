"""Task module: reaching to, grasping and then moving or using novel virtual tools, one row per successful trial, with the
tool's identity, whether it carried a label, and the times of reach onset, grasp and end of the action. First defined with
Foerster2020_Cortex (Labels strengthen motor learning of new tools)."""

DESCRIPTION = ("Learning to manipulate novel tools in virtual reality: on each trial the participant reaches for a tool, grasps "
               "it and then either moves it to another location or uses it (its learned manipulation). One row is one "
               "successful trial, with the tool, whether it had a label, and the times of reach onset, grasp and end of the action.")

# (column, type, values or units, description)
COLUMNS = [
    ("tool_id", "text", "", "Tool shown (the authors' Mesh: one of six 3D models)."),
    ("tool_function", "text", "", "The manipulation associated with the tool, in the authors' naming (their Function: for example growing, vanishing)."),
    ("tool_name", "text", "", "Name given to the tool in the authors' file: a pseudo-word label for labelled tools, NoName1-3 for unlabelled ones."),
    ("labelled", "true/false", "", "The tool had been given a label (the authors' Name_learnt)."),
    ("action_task", "category", "move, use", "Whether the participant had to move the tool to another location or use it (carry out its manipulation)."),
    ("trial_number", "integer", "", "The authors' TrialNumber. Only successful trials are in the file, so numbers are missing for unsuccessful ones."),
    ("trial_block", "integer", "1, 2, 3", "The authors' TrialBlock of the trial."),
    ("reach_onset_s", "number", "seconds", "The authors' RTMovement: time at which the movement toward the tool started. The time origin is not stated in the deposit. Empty on trials where the authors have no value."),
    ("grasp_time_s", "number", "seconds", "The authors' RTGrasping: time at which the tool was grasped, from the same origin."),
    ("action_end_s", "number", "seconds", "The authors' RTAction: time at which the move or use action was completed, from the same origin."),
    ("reach_to_grasp_s", "number", "seconds", "The authors' RTMovementGrasping: grasp time minus reach onset (reproduced from the other columns to within 0.5 ms)."),
    ("grasp_to_action_s", "number", "seconds", "The authors' RTGraspingAction: action end minus grasp time."),
    ("trial_success", "true/false", "", "The authors' SuccessTask; the deposit's trial table holds successful trials only."),
]
