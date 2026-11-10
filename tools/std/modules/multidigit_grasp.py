"""Task module: grasps of an object with any number of digits, one row per grasp, with how many digits touched the object and when each
digit reached it. First defined with Lepori2025_JNeurophysiol (precision-grip and unconstrained grasps of 3D polycube objects recorded
with motion capture); the per-trial values are the authors' own derived measures."""

DESCRIPTION = ("Grasps with a free number of digits (precision grip or unconstrained): one row per grasp of an object, with the number "
               "of digits that touched it at first contact, the time each digit reached the object relative to the object starting to "
               "move, and the largest tilt of the object during the trial.")

# (column, type, values or units, description)
COLUMNS = [
    ("object_code", "text", "VW, VB, SB, SW",
     "The authors' object code: the first letter is the object's shape (V or S), the second the material configuration (W or B) of the two brass-and-wood versions of that shape."),
    ("repetition", "integer", "1-5", "Repetition of this object within the session, in the order of the authors' data arrays (not necessarily the order of the trials in time)."),
    ("n_digits_contact", "integer", "0-5",
     "Number of digits (thumb, index, middle, ring, little finger) whose fingertip marker was within 25 mm of the object's surface at the moment of first contact."),
    ("first_contact_frame", "integer", "frame at 100 Hz",
     "Frame of the recording taken as first contact: 15 frames (150 ms) before the object started to move (vertical velocity above 0.4 mm/s), or set by hand by the authors where the automatic value was wrong. 1 marks a trial the authors flagged as mislabelled."),
    ("contact_time_thumb_s", "number", "s", "Time the thumb marker reached its minimum distance to the object, relative to the object starting to move (negative = before). Empty when the thumb was not in contact."),
    ("contact_time_index_s", "number", "s", "The same for the index finger."),
    ("contact_time_middle_s", "number", "s", "The same for the middle finger."),
    ("contact_time_ring_s", "number", "s", "The same for the ring finger."),
    ("contact_time_little_s", "number", "s", "The same for the little finger."),
    ("max_object_tilt_deg", "number", "deg", "Largest tilt of the object during the trial, the maximum over frames of sqrt(roll^2 + pitch^2) of the object's 6D pose from the motion-capture software. Empty for trials the authors flagged."),
]
