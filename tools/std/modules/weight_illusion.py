"""Task module: lifts of objects whose visible material or weight distribution differs from their real weight (material-weight
illusion), one row per trial, with the participant's heaviness rating and the forces and torque measured while lifting.
First defined with Paulun2019_JNeurophysiol (precision-grip lifts of bipartite objects with force-torque sensors in the handle)."""

DESCRIPTION = ("Lifts of objects that look heavy or light (stone, wood, Styrofoam) with a precision grip: one row per trial, with "
                "the object and its material at the index-finger and thumb side, the heaviness rating the participant gave (z-scored "
                "within the participant), and the peak grip force, load force, their rates of change and the torque measured while lifting.")

# (column, type, values or units, description)
COLUMNS = [
    ("rating_task", "category", "numerical rating, centre-of-mass judgement",
     "How the participant reported perceived weight in this experiment: a number for each half of the object, or the point where the object would balance. Empty when the experiment had one task."),
    ("lifting_order", "integer", "1-6", "Which of the six pseudorandom orders of objects the participant had."),
    ("lift_number", "integer", "",
     "How many times this object had been lifted when the row was recorded; 0 is a rating given by looking at the object before any lift (no force data). Empty for practice trials without a number."),
    ("object", "text", "", "Object in the authors' naming (for example Wood_Stone: wood-looking and stone-looking halves; Expected, Unexpected, Equal in the weight-difference experiment)."),
    ("material_index", "text", "stone, wood, Styrofoam",
     "Apparent material at the index-finger side of the object (the deposit's Material_Finger). Empty for uniform objects in experiments that give one material per object."),
    ("material_thumb", "text", "stone, wood, Styrofoam", "Apparent material at the thumb side of the object (the deposit's Material_Thumb)."),
    ("weight_index_kg", "number", "kg", "Real weight of the half at the index-finger side (only the experiment with real weight differences)."),
    ("weight_thumb_kg", "number", "kg", "Real weight of the half at the thumb side."),
    ("half_index", "text", "", "Combination of apparent material and real weight at the index-finger side, for example Stone_Light (only the experiment with real weight differences)."),
    ("half_thumb", "text", "", "Combination of apparent material and real weight at the thumb side."),
    ("rating_index_z", "number", "z-score within participant", "Heaviness rating (or the centre-of-mass judgement turned into a rating) for the half at the index-finger side, standardized within each participant."),
    ("rating_thumb_z", "number", "z-score within participant", "The same for the half at the thumb side."),
    ("rating_object_z", "number", "z-score within participant", "Heaviness rating of the whole object, standardized within each participant (experiments in which the whole object was rated)."),
    ("peak_torque_nmm", "number", "N.mm",
     "Peak torque around the axis through the handle during the loading phase of the lift (first local extremum). Positive is a rotation toward the heavier (or heavier-looking) side, by the authors' alignment across object orientations."),
    ("hold_torque_nmm", "number", "N.mm", "Median torque during the holding phase (only the experiment with real weight differences), same sign convention."),
    ("peak_grip_force_n", "number", "N", "Peak grip force, the first peak of the mean of both digits' grip force (only the experiment with uniform-looking objects)."),
    ("peak_grip_force_rate_n_s", "number", "N/s", "Peak rate of change of grip force."),
    ("peak_load_force_n", "number", "N", "Peak load force, the first peak of the mean of both digits' load force."),
    ("peak_load_force_rate_n_s", "number", "N/s", "Peak rate of change of load force."),
]
