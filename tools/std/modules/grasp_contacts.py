"""Task module: where the thumb and index finger touch an object in a precision grasp, one row per grasp. First defined with
Klein2020_PLoSComputBiol (grasping novel 3D objects at several orientations): the object, its orientation, the hand's start
position and the thumb and index contact points in the tracking system's coordinates."""

DESCRIPTION = ("Precision grip grasp locations: one row per grasp of an object with thumb and index finger, with the object, "
               "its orientation, the hand's start position and the 3D contact points of the thumb and index finger.")

# (column, type, values or units, description)
COLUMNS = [
    ("object_id", "integer", "", "Object number as in the deposit (Experiment 1: 4 shapes; Experiment 2: 12 objects; Experiment 3: 4 objects)."),
    ("object_angle_deg", "number", "deg", "Orientation of the object as in the deposit's Angle column (Experiments 1 and 2); empty when not given."),
    ("start_x_mm", "number", "mm", "x of the hand's start position (Experiments 1 and 2), tracking-system coordinates."),
    ("start_y_mm", "number", "mm", "y of the hand's start position."),
    ("start_z_mm", "number", "mm", "z of the hand's start position."),
    ("thumb_contact_x_mm", "number", "mm", "x of the thumb contact point (fingertip at first contact), tracking-system coordinates."),
    ("thumb_contact_y_mm", "number", "mm", "y of the thumb contact point."),
    ("thumb_contact_z_mm", "number", "mm", "z of the thumb contact point."),
    ("index_contact_x_mm", "number", "mm", "x of the index finger contact point."),
    ("index_contact_y_mm", "number", "mm", "y of the index finger contact point."),
    ("index_contact_z_mm", "number", "mm", "z of the index finger contact point."),
    ("thumb_contact_v2_x_mm", "number", "mm", "Experiment 3 only: x of the deposit's second version (V02) of the thumb contact, a point on the object's reachable surface found by the authors' code."),
    ("thumb_contact_v2_y_mm", "number", "mm", "Experiment 3 only: y of the V02 thumb contact."),
    ("thumb_contact_v2_z_mm", "number", "mm", "Experiment 3 only: z of the V02 thumb contact."),
    ("index_contact_v2_x_mm", "number", "mm", "Experiment 3 only: x of the V02 index finger contact."),
    ("index_contact_v2_y_mm", "number", "mm", "Experiment 3 only: y of the V02 index finger contact."),
    ("index_contact_v2_z_mm", "number", "mm", "Experiment 3 only: z of the V02 index finger contact."),
    ('thumb_contact_obj_x_mm', 'number', 'mm', "Paulun/Maiello deposits: x of the thumb contact point relative to the object's centre (the deposit's TouchedThumbXn), in the table plane, mm; the raw tracker coordinates are in thumb_contact_x_mm."),
    ('thumb_contact_obj_y_mm', 'number', 'mm', "y of the thumb contact point relative to the object's centre (deposit's TouchedThumbYn), mm."),
    ('thumb_contact_obj_z_mm', 'number', 'mm', "z (height) of the thumb contact point relative to the table surface plane as given in the deposit's TouchedThumbZn, mm."),
    ('index_contact_obj_x_mm', 'number', 'mm', "x of the index finger contact point relative to the object's centre (deposit's TouchedFingerXn), mm."),
    ('index_contact_obj_y_mm', 'number', 'mm', "y of the index finger contact point relative to the object's centre (deposit's TouchedFingerYn), mm."),
    ('index_contact_obj_z_mm', 'number', 'mm', "z of the index finger contact point as given in the deposit's TouchedFingerZn, mm."),
    ('reaction_time_ms', 'number', 'ms', 'Time from the start signal to movement onset (hand speed above 0.025 m/s), as the authors computed it.'),
    ('reach_duration_ms', 'number', 'ms', 'Time from movement onset to first contact with the object (the approach time), as the authors computed it.'),
    ('contact_to_lift_ms', 'number', 'ms', 'Time from first contact to the moment of lift (the handling duration), as the authors computed it.'),
    ('max_grip_aperture_mm', 'number', 'mm', 'Maximum distance between thumb and index finger between movement onset and first contact.'),
]
